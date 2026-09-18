"""`selfwalk`: the V-JEPA half and the orchestrator. See selfwalk.py for the idea.

    flyte run pipeline.py selfwalk                        # 4 rounds x 50M steps
    flyte run pipeline.py selfwalk --rounds 2 --steps_per_round 20000000

    round 0   selfwalk_label   film the rl-mujoco G1 doing nine things, V-JEPA scores
                               every clip against clips of it walking, fit the reward net
    round r   train_student    PPO from scratch (r=0) or from round r-1, reward = the net
              selfwalk_label   film what the STUDENT does, V-JEPA scores it for real,
                               report the net-vs-V-JEPA gap, add it to the data, refit

Three reports, all live: the labeller's (what V-JEPA thinks of each behaviour), the
trainer's (the student filmed at every eval, next to the reference), and this
orchestrator's (one clip per round and the true V-JEPA score of the student by round).
"""

from __future__ import annotations

import logging
import pickle
import time

import flyte
import flyte.report
import numpy as np
from flyte.io import File

import reports
import selfwalk as sw
import viz
from config import VITL, gpu_env, orch_env
from selfwalk_train import train_student

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _paint(stage: str, detail: str, rows, extra: str = "") -> None:
    try:
        flyte.report.replace(reports.progress_html(stage, detail, rows) + extra, do_flush=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("paint failed: %s", exc)


def _montage(clips: np.ndarray, captions: list[list[tuple[str, tuple]]], cols: int = 3,
             loops: int = 3) -> np.ndarray:
    """k clips [k, 16, S, S, 3] -> one looping video, a captioned grid of clips."""
    k = len(clips)
    rows = -(-k // cols)
    S = clips.shape[2]
    out = []
    for t in range(clips.shape[1]):
        grid = np.full((rows * S, cols * S, 3), 20, np.uint8)
        for i in range(k):
            r, c = divmod(i, cols)
            grid[r * S:(r + 1) * S, c * S:(c + 1) * S] = viz.annotate(clips[i, t], captions[i], scale=1)
        out.append(grid)
    return np.stack(out * loops)


def _pixel_scores(clips: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """The control: the same max-over-bank match in raw pixels (64 px grey)."""
    import torch

    def small(x):
        t = torch.as_tensor(x).float().mean(-1)                           # [B, 16, S, S]
        return torch.nn.functional.adaptive_avg_pool2d(t, 64).flatten(1) / 255.0

    b = small(bank)
    out = []
    for i in range(0, len(clips), 64):
        c = small(clips[i:i + 64])
        out.append((-torch.cdist(c, b, p=1) / c.shape[1]).max(1).values.numpy())
    return np.concatenate(out)


def _render_score(cam, scorer, clip_q, bank_clips, on_chunk=None, chunk: int = 64):
    raw, pix = [], []
    for i in range(0, len(clip_q), chunk):
        clips = np.stack([cam.clip(c) for c in clip_q[i:i + chunk]])
        raw.append(scorer.score(clips))
        pix.append(_pixel_scores(clips, bank_clips))
        if on_chunk is not None:
            on_chunk(min(i + chunk, len(clip_q)), len(clip_q), clips, raw[-1])
    return np.concatenate(raw), np.concatenate(pix)


@gpu_env.task(report=True)
async def selfwalk_label(
    round_i: int = 0,
    prev: File | None = None,
    student: File | None = None,
    windows_per_source: int = 300,
    student_windows: int = 800,
    fit: bool = True,
    repo: str = VITL,
    reward: str = "vjepa",
) -> File:
    """Round 0: score the nine reference behaviours. Round r: score the student.

    Returns one pickle carrying everything downstream needs: the dataset, the reward
    net, the normalisation, the reference clip, and a per-round summary.
    """
    import jepa

    t0 = time.time()
    guard = jepa.guard_memory()
    rows = [("round", str(round_i)), ("reward source", reward), ("GPU", guard)]
    _paint(f"selfwalk label, round {round_i}", "loading the G1 and V-JEPA", rows)

    env = sw.load_env("jax")
    cam = sw.ClipCamera(env.mj_model)
    scorer = sw.Scorer(repo)
    live = {"clip": ""}

    def on_chunk(done, total, clips, scores):
        try:
            idx = np.argsort(scores)[[-1, len(scores) // 2, 0]]
            live["clip"] = viz.video_html(
                viz.encode_mp4(_montage(clips[idx], [[(f"V-JEPA {scores[j]:.3f}", viz._AMBER)] for j in idx]), fps=12),
                "what V-JEPA is shown: this chunk's best, median and worst clip", max_width=780)
        except Exception as exc:  # noqa: BLE001
            log.warning("live clip failed: %s", exc)
        _paint(f"round {round_i}: V-JEPA scoring {done}/{total} clips", "render 16 frames, encode, match "
               "token-wise against the reference walker", rows, live["clip"])

    if prev is None:
        # ── round 0: the reference behaviours ────────────────────────────────────
        ck = await File.from_existing_remote(sw.TEACHER_CHECKPOINT).download()
        pol = sw.load_policy(env, ck)
        clip_q, raw4, src = [], [], []
        for n, (name, (cmd, noise, actor)) in enumerate(sw.SOURCES.items()):
            _paint(f"round 0: filming '{name}' ({n + 1}/{len(sw.SOURCES)})",
                   "the rl-mujoco G1 under different commands, plus random and frozen actions", rows)
            tr = sw.rollout(env, pol, 16, 300, cmd, noise, actor, seed=n + 1)
            ci, r4 = sw.windows(tr, stride=8, limit=windows_per_source, seed=n)
            clip_q.append(ci); raw4.append(r4); src += [name] * len(ci)
            if name == sw.TARGET:
                reference_mp4 = viz.encode_mp4(cam.follow(tr["qpos"][0, :300]), fps=50)
        clip_q, raw4, src = np.concatenate(clip_q), np.concatenate(raw4), np.array(src)
        tb = sw.rollout(env, pol, 8, 300, sw.SOURCES[sw.TARGET][0], 0.0, "policy", seed=99)
        bank_q, _ = sw.windows(tb, stride=9, first=40, limit=48, seed=99)
        bank_clips = np.stack([cam.clip(c) for c in bank_q])
        scorer.set_bank(bank_clips)
        vj, pix = _render_score(cam, scorer, clip_q, bank_clips, on_chunk)
        # `reward="pixel"` is the control: the same pipeline, rewarded by raw pixel match.
        raw = pix if reward == "pixel" else vj
        lo, hi = float(np.percentile(raw, 10)), float(np.median(raw[src == sw.TARGET]))
        data = {"raw4": raw4, "raw": raw, "src": src, "weight": np.ones(len(raw), np.float32)}
        state = {"norm": (lo, hi), "bank_q": bank_q, "reference_mp4": reference_mp4, "rounds": [],
                 "by_source": {k: float(np.median(raw[src == k])) for k in sw.SOURCES},
                 "pixel_by_source": {k: float(np.median(pix[src == k])) for k in sw.SOURCES},
                 "vjepa_by_source": {k: float(np.median(vj[src == k])) for k in sw.SOURCES}}
        examples = [np.flatnonzero(src == k)[len(np.flatnonzero(src == k)) // 2] for k in sw.SOURCES]
        example_clips = np.stack([cam.clip(clip_q[i]) for i in examples])
        example_caps = [[(k, viz._AMBER), (f"V-JEPA {raw[i]:.3f}", viz._GREY)]
                        for k, i in zip(sw.SOURCES, examples)]
        summary = {"round": 0}
    else:
        # ── round r: judge the student for real ──────────────────────────────────
        with open(await prev.download(), "rb") as f:
            old = pickle.load(f)
        with open(await student.download(), "rb") as f:
            stu = pickle.load(f)
        data, state = old["data"], old["state"]
        lo, hi = state["norm"]
        bank_clips = np.stack([cam.clip(c) for c in state["bank_q"]])
        scorer.set_bank(bank_clips)
        ci, r4 = sw.windows(stu["rollouts"], stride=6, limit=student_windows, seed=round_i)
        vj, pix = _render_score(cam, scorer, ci, bank_clips, on_chunk)
        raw = pix if reward == "pixel" else vj
        y_true = sw.normalise(raw, lo, hi)
        feats = sw.window_features(r4)
        y_net = sw.rewnet_apply(old["rewnet"], feats)
        fwd = feats[:, 4 * 68 + 2]                    # forward displacement over the window (m)
        summary = {"round": round_i, "steps": int(stu["total_steps"]), "n": int(len(raw)),
                   "vjepa_median": float(np.median(raw)), "vjepa_p90": float(np.percentile(raw, 90)),
                   "reward_true": float(y_true.mean()), "reward_net": float(y_net.mean()),
                   "gap": float((y_net - y_true).mean()), "fwd_per_window_m": float(np.median(fwd)),
                   "net_r": float(np.corrcoef(y_net, y_true)[0, 1]) if len(raw) > 2 else float("nan"),
                   "vjepa_true_median": float(np.median(vj)), "reward_source": reward}
        log.info("student round %d: %s", round_i, summary)
        data = {"raw4": np.concatenate([data["raw4"], r4]), "raw": np.concatenate([data["raw"], raw]),
                "src": np.concatenate([data["src"], np.array([f"student-r{round_i}"] * len(raw))]),
                "weight": np.concatenate([data["weight"], np.full(len(raw), 2.0, np.float32)])}
        order = np.argsort(raw)
        pick = list(order[-3:][::-1]) + list(order[:3])
        example_clips = np.stack([cam.clip(ci[i]) for i in pick])
        example_caps = [[("best" if n < 3 else "worst", viz._AMBER), (f"V-JEPA {raw[i]:.3f}", viz._GREY),
                         (f"net said {y_net[i]:.2f} vs {y_true[i]:.2f}", viz._GREY)]
                        for n, i in enumerate(pick)]
    cam.close()

    # ── fit (or keep) the reward net ────────────────────────────────────────────
    if fit:
        _paint(f"round {round_i}: fitting the reward net", "a 2-layer MLP learns V-JEPA's score from the "
               "robot's state over the same 0.62 s window", rows, live["clip"])
        rewnet, fit_stats = sw.fit_rewnet(sw.window_features(data["raw4"]), sw.normalise(data["raw"], lo, hi),
                                          data["weight"], seed=round_i)
    else:
        rewnet, fit_stats = old["rewnet"], {"skipped": True}
    summary["fit"] = fit_stats
    state["rounds"] = state["rounds"] + [summary]

    # ── report ──────────────────────────────────────────────────────────────────
    names = list(sw.SOURCES)
    meds = [state["by_source"][k] for k in names]
    body = reports.heading("What V-JEPA thinks of each behaviour")
    body += viz.bar_chart("V-JEPA score vs the reference walker (median per behaviour)", names,
                          {"V-JEPA": meds}, "centred token cosine")
    body += reports.note(f"Reward = (score - {lo:.3f}) / ({hi:.3f} - {lo:.3f}), clipped to [0, 1]: 0 at the "
                         "10th percentile of everything filmed in round 0, 1 at the median walking clip.")
    pix = state["pixel_by_source"]
    body += reports.note("Control, the same match in raw pixels (higher = more similar): "
                         + ", ".join(f"{k} {pix[k]:.3f}" for k in names))
    students = [r for r in state["rounds"] if r["round"] > 0]
    if students:
        body += viz.curve_chart(f"The student, judged by the real {reward} score", "round", "median score",
                                {"student": [(r["round"], r["vjepa_median"]) for r in students]},
                                hlines={"walk (the target)": state["by_source"][sw.TARGET],
                                        "march in place": state["by_source"]["march"],
                                        "flail": state["by_source"]["flail"]})
        body += viz.curve_chart("Reward hacking check: what the net promised vs what V-JEPA gave",
                                "round", "mean reward in [0, 1]",
                                {"reward net": [(r["round"], r["reward_net"]) for r in students],
                                 "real V-JEPA": [(r["round"], r["reward_true"]) for r in students]})
    body += reports.heading("Examples, as V-JEPA sees them")
    body += viz.video_html(viz.encode_mp4(_montage(example_clips, example_caps), fps=12), "", max_width=780)
    rows += [("clips scored", str(len(data["raw"]))), ("reward net held-out r", f"{fit_stats.get('val_r', float('nan')):.3f}"),
             ("minutes", f"{(time.time() - t0) / 60:.1f}")]
    if round_i > 0:
        rows += [("student median V-JEPA", f"{summary['vjepa_median']:.3f} (walk {state['by_source'][sw.TARGET]:.3f})"),
                 ("net vs V-JEPA gap", f"{summary['gap']:+.3f}")]
    flyte.report.replace(reports.final_html(f"selfwalk label, round {round_i}", rows, body), do_flush=True)

    path = f"/tmp/selfwalk_label_r{round_i}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"data": data, "state": state, "rewnet": rewnet, "reference_mp4": state["reference_mp4"],
                     "summary": summary}, f)
    return await File.from_local(path)


@orch_env.task(report=True)
async def selfwalk(
    rounds: int = 4,
    steps_per_round: int = 50_000_000,
    num_envs: int = 4096,
    windows_per_source: int = 300,
    student_windows: int = 800,
    reward: str = "vjepa",
) -> dict:
    """Teach a G1 to walk with V-JEPA as the only reward. See selfwalk.py."""
    table: list[dict] = []
    clips: list[tuple[str, str]] = []

    def paint(stage: str, lab_state: dict | None = None) -> None:
        rows = [("stage", stage), ("reward source", reward), ("rounds", f"{len(table)}/{rounds}"),
                ("steps per round", f"{steps_per_round:,}")]
        body = ""
        if table:
            body += reports.data_table(
                ["round", "env steps", "distilled reward / ep", "m forward / ep", "episode length",
                 "V-JEPA median (student)", "net - V-JEPA gap"],
                [[str(r["round"]), f"{r['steps']:,}", f"{r['reward']:.2f}", f"{r['fwd_m']:.2f}",
                  f"{r['ep_len']:.0f}", f"{r['vjepa']:.3f}", f"{r['gap']:+.3f}"] for r in table])
        if lab_state and table:
            bs = lab_state.get("vjepa_by_source", lab_state["by_source"])
            body += viz.curve_chart("Does the student look like walking to V-JEPA?", "round", "median V-JEPA score",
                                    {"student": [(r["round"], r["vjepa"]) for r in table]},
                                    hlines={"walk (the target)": bs[sw.TARGET], "march in place": bs["march"],
                                            "flail": bs["flail"]})
        for title, html in clips[::-1]:
            body += reports.heading(title) + html
        _paint("selfwalk: a G1 learns to walk from V-JEPA alone", stage, rows, body)

    paint("round 0: filming the reference behaviours and asking V-JEPA about them")
    lab = await selfwalk_label(round_i=0, windows_per_source=windows_per_source, reward=reward)
    ck = None
    for r in range(rounds):
        paint(f"round {r + 1}/{rounds}: PPO (open train_student's report to watch it live)")
        ck = await train_student(label=lab, round_i=r, num_timesteps=steps_per_round, restore=ck,
                                 num_envs=num_envs)
        paint(f"round {r + 1}/{rounds}: V-JEPA judges what the student does")
        lab = await selfwalk_label(round_i=r + 1, prev=lab, student=ck, student_windows=student_windows,
                                   fit=r < rounds - 1, reward=reward)
        with open(await ck.download(), "rb") as f:
            stu = pickle.load(f)
        with open(await lab.download(), "rb") as f:
            labd = pickle.load(f)
        h = stu["history"][-1] if stu["history"] else {}
        s = labd["summary"]
        table.append({"round": r + 1, "steps": int(stu["total_steps"]), "reward": h.get("reward", 0.0),
                      "fwd_m": h.get("fwd_m", 0.0), "ep_len": h.get("ep_len", 0.0),
                      "vjepa": s["vjepa_true_median"], "gap": s["gap"]})
        if stu.get("final_mp4"):
            fin = stu["final"]
            clips.append((f"after round {r + 1} ({stu['total_steps'] / 1e6:.0f}M steps)",
                          viz.video_html(stu["final_mp4"], f"survived {fin['alive']}/300 steps, "
                                                           f"{fin['fwd']:.2f} m forward", max_width=480)))
        paint(f"round {r + 1}/{rounds} done", labd["state"])
    return {"rounds": table}
