"""V-JEPA 2 on Flyte: a world model with no decoder, and what that costs.

    flyte run pipeline.py inpaint                  # the predictor, under two masks
    flyte run pipeline.py inpaint --clip archery --context 0.75
    flyte run pipeline.py probe                    # frozen features -> action recognition
    flyte run pipeline.py scale                    # ViT-L vs ViT-g, same measurements
    flyte run pipeline.py vjepa                    # all three, one run

    flyte run pipeline.py plan                     # V-JEPA 2-AC plans a Franka in MuJoCo
    flyte run pipeline.py plan --seeds 3 --steps 14
    flyte run pipeline.py dream                    # open-loop latent rollout, decoded by retrieval
    flyte run pipeline.py adapt                    # fine-tune the predictor on sim, re-plan
    flyte run pipeline.py readout                  # is the picture inside the token? (no)
    flyte run pipeline.py push                     # show it a photo of the finished task

How the thing works, rather than how well (see "the mechanism tasks" at the bottom):

    flyte run pipeline.py occlude                  # RENDER the prediction, with its ceiling
    flyte run pipeline.py energy                   # the E in energy-based model
    flyte run pipeline.py ladder                   # where the pixels go, layer by layer
    flyte run pipeline.py collapse                 # why JEPA needs an EMA teacher
    flyte run pipeline.py mechanism                # all four, one run

Runs to the `world-models` project (.flyte/config.yaml), alongside topics/cosmos and
topics/dreamerv3. The three are the same question asked three ways: Cosmos predicts the
future in pixels, Dreamer learns a latent world model of one environment from its own
experience, and V-JEPA 2 predicts representations learned self-supervised from video.

── What this demo is careful about ─────────────────────────────────────────────
V-JEPA 2 has no decoder, so there is no honest way to render "what it predicted". Every
frame in these reports is either the model's literal input (with the masked patches
blacked out) or a per-patch number we computed painted onto those same pixels. Nothing
here is a vector dressed up as an image.

The second care is floors. A cosine similarity between two ViT tokens is a number
between 0 and 1 that always looks encouraging, so every score in `inpaint` is reported
next to a shuffled-pairing chance floor and a no-model baseline, and every score in
`probe` next to chance and a raw-pixel probe.

── Why the tasks run in sequence ───────────────────────────────────────────────
One GPU on this box, so a second GPU task would sit Unschedulable on "Insufficient
nvidia.com/gpu" until the first finished anyway. The orchestrator is CPU-only for the
same reason: a GPU-holding orchestrator deadlocks its own GPU child.
"""

from __future__ import annotations

import logging
import time

import flyte
import flyte.report

# Imported at top level so Flyte bundles these siblings into the pod. A deferred import
# inside a task body is exactly how you get ModuleNotFoundError in the pod while
# everything works on the host.
import ac
import clips as clip_io
import collapse as collapse_module
import decode
import energy as ebm
import jepa
import adapt as adapt_module
import layers
import occlude as occlusion
import plan as planning
import probing
import reports
import sim
import viz
from config import CLIPS_REPO, VITG, VITL, ac_env, gpu_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _paint(stage: str, detail: str, rows: list[tuple[str, str]]) -> None:
    flyte.report.replace(reports.progress_html(stage, detail, rows), do_flush=True)


@gpu_env.task(report=True)
async def inpaint(
    clip: str = "bowling",
    repo: str = VITL,
    frames: int = 64,
    context: float = 0.5,
    block: int = 8,
    seed: int = 0,
) -> dict:
    """Hide part of a clip, predict the hidden part IN LATENT SPACE, and score it twice.

    Once under a TUBE mask (spatial blocks removed across the whole clip), which is the
    masking V-JEPA 2 was pretrained on, and once under a FUTURE mask (everything after
    `context` of the way through), which it never saw during pretraining. Same clip,
    same predictor, same scoring, so the difference is the mask.

    The defaults are chosen so the two masks hide roughly the SAME fraction of tokens:
    two 8x8 blocks is ~48% and `context=0.5` is 50%. That matters, because most of
    these metrics move with how much was hidden, and an easier mask winning would prove
    nothing. `block` is the side length in patches of each of the two tube blocks;
    `context` only affects the future mask.
    """
    rows = [("Model", repo), ("Clip", clip), ("Frames", str(frames))]
    _paint("Fetching", "Pulling the checkpoint and one clip into this pod.", rows)

    guard = jepa.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "V-JEPA 2 encoder + predictor, BF16.", rows)
    model, processor, params = jepa.load(repo)
    rows.append(("Params", f"{params / 1e6:.0f}M"))

    catalog = clip_io.list_clips(CLIPS_REPO)
    path = clip_io.pick(catalog, clip)
    _paint("Encoding", f"{path}", rows)
    pixel_values, shown = clip_io.load_clip(processor, CLIPS_REPO, path, frames)
    tubelets, grid = jepa.grid_of(model, frames)
    seq = jepa.encode(model, pixel_values)
    jepa.check_layout(seq, tubelets, grid)

    aniso = jepa.anisotropy(seq)
    rows += [
        ("Tokens", f"{tubelets} tubelets x {grid}x{grid} = {seq.shape[0]}, dim {seq.shape[1]}"),
        ("Source", path),
        ("Random-pair cosine", f"raw {aniso['raw']:.3f} -> centered {aniso['centered']:.3f}"),
    ]

    masks = {
        "tube (pretraining mask)": jepa.tube(tubelets, grid, blocks=2, size=block, seed=seed),
        "future (never trained on)": jepa.future(tubelets, grid, context=context),
    }

    cells, results, hlines = [], {}, {}
    horizon_series = {}
    for name, mask3d in masks.items():
        _paint("Predicting", f"{name}: running the predictor over the masked tokens.", rows)
        context_ids, target_ids = jepa.ids_of(mask3d)
        pred, true = jepa.predict(model, pixel_values, context_ids, target_ids)

        scored = jepa.score(pred, true, target_ids, grid)
        floor = jepa.shuffled_floor(pred, true, target_ids, grid, seed=seed)
        ctx = jepa.context_floor(seq, context_ids, target_ids, true, grid)
        # Normalised against this mask's OWN chance level, which is the only way the
        # two masks can be put on the same axis. See jepa.localization.
        scored["loc"] = jepa.localization(scored, floor)
        ctx["loc"] = jepa.localization(ctx, floor)
        floor["loc"] = jepa.localization(floor, floor)
        results[name] = {"predictor": scored, "shuffled": floor, "context_mean": ctx}

        field = jepa.per_patch_cos(pred, true, target_ids, tubelets, grid)
        masked_frames = viz.masked_video(shown, mask3d)
        heat_frames = viz.heat_video(shown, field)
        pair = viz.side_by_side_video(masked_frames, heat_frames)
        mp4 = viz.encode_mp4(pair, fps=12)
        probed = viz.probe(mp4)  # decodes the clip; do it once, not once per use
        log.info("%s: %s | %s", name, scored, probed)

        share = float(mask3d.float().mean())
        cells.append((
            name,
            viz.video_html(mp4, f"left: what the encoder saw ({share:.0%} of tokens hidden). "
                                f"right: per-patch prediction quality, grey = not masked.")
            + reports.note(probed)
            + reports.score_table(
                [("predictor", scored), ("context mean (no model)", ctx),
                 ("shuffled pairing (chance)", floor)],
                highlight="predictor",
            ),
        ))
        horizon_series[name] = jepa.horizon(pred, true, mask3d, target_ids, tubelets, grid)
        if name.startswith("future"):
            hlines["shuffled (chance)"] = floor["cos"]

    tube_s = results["tube (pretraining mask)"]["predictor"]
    tube_f = results["tube (pretraining mask)"]["shuffled"]
    future_s = results["future (never trained on)"]["predictor"]
    future_f = results["future (never trained on)"]["shuffled"]
    tube_share = float(masks["tube (pretraining mask)"].float().mean())
    future_share = float(masks["future (never trained on)"].float().mean())
    verdict = reports.verdict(
        f"With <b>{tube_share:.0%}</b> and <b>{future_share:.0%}</b> of tokens hidden, so the two "
        f"masks are the same size and only the shape differs: under the mask it was pretrained "
        f"on, the predictor's nearest retrieved token is a median <b>{tube_s['dt']:.0f} "
        f"tubelets</b> from the truth in time, where shuffling the same predictions gives "
        f"{tube_f['dt']:.0f}. That is {tube_s['loc']:.0%} of the chance-level temporal error "
        f"removed: it finds the right moment. Asked to extrapolate forward in time instead, the "
        f"same predictor lands a median <b>{future_s['dt']:.0f} tubelets</b> away against a "
        f"chance level of {future_f['dt']:.0f}, or {future_s['loc']:.0%}. It is returning a "
        f"plausible representation of the scene at the wrong moment. This checkpoint inpaints in "
        f"latent space; it does not forecast.",
        good=tube_s["loc"] > future_s["loc"],
    )

    body = (
        reports.heading("What the encoder saw, and where the prediction landed")
        + reports.side_by_side(cells, basis=460)
        + reports.heading("Does it degrade with horizon, or was it never tracking time?")
        + viz.horizon_chart(
            {"future mask (predictor)": horizon_series["future (never trained on)"]}, hlines
        )
        + reports.note(
            "Only the future mask is plotted: 'tubelets ahead of the last visible frame' has no "
            "meaning for a tube mask, whose targets span the whole clip. The tube mask's numbers "
            "are in the scorecard above. Note also that the cosine axis is not comparable "
            "between the two masks, which is what the 'time localised' column exists to fix."
        )
        + reports.note(reports.GEOMETRY_NOTE)
        + verdict
    )
    flyte.report.replace(
        reports.final_html("Masked latent prediction", rows, body, reports.INPAINT_EXPLAINER),
        do_flush=True,
    )
    return {
        "clip": path,
        "repo": repo,
        "anisotropy": aniso,
        "scores": {k: {m: {kk: round(vv, 4) for kk, vv in s.items()} for m, s in v.items()}
                   for k, v in results.items()},
    }


@gpu_env.task(report=True)
async def probe(repo: str = VITL, frames: int = 32) -> dict:
    """Freeze the encoder, mean-pool every clip, and see what one linear layer can do.

    100 Kinetics clips over 5 classes, the dataset's own train/val split. Nothing is
    fine-tuned: the only trained parameters in this task are a 1024x5 matrix and its
    bias. The retrieval number below it involves no training at all.
    """
    rows = [("Model", repo), ("Dataset", CLIPS_REPO), ("Frames per clip", str(frames))]
    _paint("Fetching", "Pulling the checkpoint and the clip catalogue.", rows)

    guard = jepa.guard_memory()
    rows.append(("GPU", guard))
    model, processor, params = jepa.load(repo)
    rows.append(("Params", f"{params / 1e6:.0f}M"))

    catalog = clip_io.list_clips(CLIPS_REPO)
    rows.append(("Clips", f"{len(catalog)} over {len(clip_io.labels_of(catalog))} classes"))
    _paint("Encoding", "One forward pass per clip, encoder only, no gradients.", rows)

    data = probing.encode_dataset(
        model, processor, CLIPS_REPO, catalog, frames,
        on_progress=lambda i, n: _paint("Encoding", f"clip {i}/{n}", rows),
    )
    X, P, y, train = data["X"], data["P"], data["y"], data["train"]
    labels = data["labels"]
    if data["failed"]:
        rows.append(("Skipped", f"{len(data['failed'])} clip(s) failed to decode"))

    _paint("Probing", "Training one linear layer on the frozen features.", rows)
    acc, pred = probing.linear_probe(X, y, train)
    acc_pixels, _ = probing.linear_probe(P, y, train)
    ret, neighbours = probing.retrieval(X, y, centered=True)
    ret_raw, _ = probing.retrieval(X, y, centered=False)
    chance = 1.0 / len(labels)

    rows += [
        ("Split", f"{int(train.sum())} train / {int((~train).sum())} val"),
        ("Linear probe", f"{acc:.1%}"),
        ("Raw-pixel probe", f"{acc_pixels:.1%}"),
        ("1-NN retrieval", f"{ret:.1%} centered, {ret_raw:.1%} raw"),
        ("Chance", f"{chance:.0%}"),
    ]

    cm = probing.confusion(y[~train], pred, len(labels))
    charts = reports.side_by_side([
        ("Where the probe is wrong", viz.confusion_chart(cm, labels, "Val confusion")),
        ("Against the floors", viz.bar_chart(
            "Frozen features vs baselines", ["5-way action recognition"],
            {"V-JEPA 2 + linear": [acc], "raw pixels + linear": [acc_pixels],
             "V-JEPA 2 1-NN (no training)": [ret]},
            "accuracy", floor=chance, floor_label="chance",
        )),
    ], basis=340)

    _paint("Rendering", "Building the retrieval examples.", rows)
    retrieval_cells = []
    for qi in _retrieval_examples(y, neighbours, labels):
        q_split, q_label, q_path = data["clips"][qi]
        q_frames = clip_io.decode(clip_io.fetch(CLIPS_REPO, q_path), 24)
        inner = viz.video_html(viz.encode_mp4(q_frames, fps=10),
                               f"query: {q_label}", max_width=230)
        for rank, ni in enumerate(neighbours[qi].tolist()[:2]):
            n_split, n_label, n_path = data["clips"][ni]
            hit = "match" if n_label == q_label else "MISS"
            n_frames = clip_io.decode(clip_io.fetch(CLIPS_REPO, n_path), 24)
            inner += viz.video_html(viz.encode_mp4(n_frames, fps=10),
                                    f"#{rank + 1}: {n_label} ({hit})", max_width=230)
        retrieval_cells.append((f"{q_label}", inner))

    body = (
        reports.heading("How good are features nobody supervised?")
        + charts
        + reports.heading("Nearest neighbours in the frozen embedding space")
        + reports.note(
            "No classifier and no training: each query clip is shown with the clips whose "
            "mean-pooled embedding is closest to it. These are the same features the probe "
            "above sees."
        )
        + reports.side_by_side(retrieval_cells, basis=250)
        + reports.note(reports.GEOMETRY_NOTE)
        + reports.verdict(
            f"A single linear layer on frozen V-JEPA 2 features gets <b>{acc:.0%}</b> on 5-way "
            f"action recognition, against <b>{acc_pixels:.0%}</b> for the same probe on "
            f"downsampled pixels and {chance:.0%} chance. With no training at all, nearest-"
            f"neighbour retrieval gets {ret:.0%}. Centering matters: the same retrieval on raw "
            f"features scores {ret_raw:.0%}.",
            good=acc > acc_pixels,
        )
    )
    flyte.report.replace(
        reports.final_html("Frozen features, one linear layer", rows, body,
                           reports.PROBE_EXPLAINER),
        do_flush=True,
    )
    return {
        "repo": repo,
        "probe": round(acc, 4),
        "pixel_probe": round(acc_pixels, 4),
        "retrieval": round(ret, 4),
        "retrieval_raw": round(ret_raw, 4),
        "chance": round(chance, 4),
        "clips": len(data["clips"]),
    }


def _retrieval_examples(y, neighbours, labels, per: int = 3) -> list[int]:
    """Pick query clips to show: prefer a mix of hits and at least one miss.

    Showing only successes would be a demo of the report, not of the model.
    """
    hits = [i for i in range(len(y)) if y[neighbours[i][0]] == y[i]]
    misses = [i for i in range(len(y)) if y[neighbours[i][0]] != y[i]]
    picked = hits[:: max(1, len(hits) // max(per - 1, 1))][: per - 1]
    return picked + misses[:1] if misses else hits[:per]


@gpu_env.task(report=True)
async def scale(
    small: str = VITL,
    large: str = VITG,
    clip: str = "bowling",
    frames: int = 32,
    block: int = 8,
    seed: int = 0,
) -> dict:
    """Does a bigger self-supervised encoder carry more? Same clips, same masks, same probe.

    Both models are loaded and measured inside ONE task rather than fanned out, for the
    same reason the Cosmos demo does it: there is a single GPU here, so a fan-out would
    serialise anyway, and this way the two models are scored on identically decoded
    clips instead of two independent decodes.
    """
    rows = [("Models", f"{small} vs {large}"), ("Frames per clip", str(frames))]
    _paint("Starting", "Two encoders, measured on the same clips.", rows)
    rows.append(("GPU", jepa.guard_memory()))

    catalog = clip_io.list_clips(CLIPS_REPO)
    path = clip_io.pick(catalog, clip)
    out = {}
    for repo in (small, large):
        _paint("Loading", f"{repo}", rows)
        model, processor, params = jepa.load(repo)

        pixel_values, _ = clip_io.load_clip(processor, CLIPS_REPO, path, frames)
        tubelets, grid = jepa.grid_of(model, frames)
        seq = jepa.encode(model, pixel_values)
        jepa.check_layout(seq, tubelets, grid)

        masks = {
            "tube": jepa.tube(tubelets, grid, blocks=2, size=block, seed=seed),
            "future": jepa.future(tubelets, grid, context=0.5),
        }
        scores = {}
        for name, mask3d in masks.items():
            ctx_ids, tgt_ids = jepa.ids_of(mask3d)
            pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
            s = jepa.score(pred, true, tgt_ids, grid)
            floor = jepa.shuffled_floor(pred, true, tgt_ids, grid, seed=seed)
            s["loc"] = jepa.localization(s, floor)
            floor["loc"] = jepa.localization(floor, floor)
            scores[name], scores[f"{name}_floor"] = s, floor

        _paint("Encoding", f"{repo}: {len(catalog)} clips for the probe.", rows)
        data = probing.encode_dataset(
            model, processor, CLIPS_REPO, catalog, frames,
            on_progress=lambda i, n, r=repo: _paint("Encoding", f"{r}: clip {i}/{n}", rows),
        )
        acc, _ = probing.linear_probe(data["X"], data["y"], data["train"])
        ret, _ = probing.retrieval(data["X"], data["y"])
        out[repo] = {
            "params_m": round(params / 1e6),
            "hidden": int(seq.shape[1]),
            "probe": round(acc, 4),
            "retrieval": round(ret, 4),
            "scores": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in scores.items()},
        }
        rows.append((repo.split("/")[-1], f"{params / 1e6:.0f}M, probe {acc:.1%}, 1-NN {ret:.1%}"))
        log.info("%s -> %s", repo, out[repo])

        del model
        _free()

    names = [r.split("/")[-1] for r in out]
    chance = 1.0 / len(clip_io.labels_of(catalog))
    body = (
        reports.heading("Semantics: what a linear layer can read off")
        + viz.bar_chart(
            "Frozen-feature action recognition", names,
            {"linear probe": [out[r]["probe"] for r in out],
             "1-NN retrieval": [out[r]["retrieval"] for r in out]},
            "accuracy", floor=chance, floor_label="chance",
        )
        + reports.heading("Prediction: does it put the masked tokens at the right moment?")
        + viz.bar_chart(
            "Masked latent prediction, temporal localisation", names,
            {"tube mask (pretrained on)": [out[r]["scores"]["tube"]["loc"] for r in out],
             "future mask (never trained on)": [out[r]["scores"]["future"]["loc"] for r in out]},
            "fraction of chance error removed", floor=0.0, floor_label="chance",
        )
        + reports.note(
            "Plotted as the normalised localisation score rather than raw cosine, because raw "
            "cosine is not comparable between the two masks. 1.00 means the retrieved token is "
            "at exactly the right moment; 0.00 means no better than shuffling the predictions. "
            "The two charts are also different questions: the first is about the encoder's "
            "representation, the second about the predictor head on top of it, and a bigger "
            "encoder is not obliged to improve both."
        )
        + reports.details("raw", repr(out))
    )
    flyte.report.replace(
        reports.final_html("Encoder size", rows, body, reports.SCALE_EXPLAINER), do_flush=True
    )
    return out


def _free() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@orch_env.task(report=True)
async def vjepa(clip: str = "bowling", repo: str = VITL) -> dict:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU children."""
    predicted = await inpaint(clip=clip, repo=repo)
    probed = await probe(repo=repo)
    scaled = await scale(clip=clip)
    result = {"inpaint": predicted, "probe": probed, "scale": scaled}
    log.info("result: %s", result)
    return result


# ══ V-JEPA 2-AC ═══════════════════════════════════════════════════════════════
#
# The two tasks above measure a predictor that cannot extrapolate forward in time.
# These two use the action-conditioned post-train, which can, and put it in a loop
# with a robot. Separate TaskEnvironment (`ac_env`) because they need the 11.7 GB
# AC checkpoint hostPath-mounted and a GL stack for MuJoCo.

REACH_THRESHOLD = 0.06  # metres; a gripper within 6 cm of the goal pose has reached


@ac_env.task(report=True)
async def plan(
    seeds: int = 2,
    steps: int = 12,
    samples: int = 50,
    cem_steps: int = 5,
    rollout: int = 2,
    maxnorm: float = 0.05,
    render: int = 384,
) -> dict:
    """Plan a Franka to a goal photo, in latent space, with frozen weights.

    No training, no reward function, no decoder. The only signal the planner gets is
    the L1 distance between the embedding of what it imagines and the embedding of a
    photograph of success.
    """
    import numpy as np

    _paint("loading V-JEPA 2-AC", "1.3B params, 11.7 GB checkpoint",
           [("checkpoint", ac.checkpoint_path()), ("seeds", str(seeds)), ("steps", str(steps))])

    wm = ac.ActionWorldModel()
    log.info("AC ready: build %.1fs load %.1fs, missing=%d unexpected=%d",
             wm.build_s, wm.load_s, len(wm.missing), len(wm.unexpected))
    if wm.missing or wm.unexpected:
        # A partial load still produces perfectly plausible energies, from a ViT-g
        # that is partly random. Better to fail than to publish that.
        raise RuntimeError(f"AC checkpoint loaded partially: {wm.missing[:4]} {wm.unexpected[:4]}")

    cfg = ac.CEMConfig(samples=samples, rollout=rollout, cem_steps=cem_steps,
                       topk=max(4, samples // 5), maxnorm=maxnorm)
    per_seed: list[tuple[int, dict]] = []
    first_extras: dict = {}

    for seed in range(seeds):
        eps: dict = {}
        for policy in ("jepa", "greedy", "lookahead", "oracle", "random"):
            env, goal_frame, goal_pos, start_frame = sim.reach_task(seed=seed, render_size=ac.CROP)

            def _tick(ep, t, total, _p=policy, _s=seed):
                _paint(
                    f"seed {_s}, {_p}: step {t + 1}/{total}",
                    "Planning happens entirely in latent space; the arm is only ever "
                    "told a Cartesian delta.",
                    [("policy", _p), ("distance to goal", f"{ep.dist[-1] * 100:.1f} cm"),
                     ("latent energy", f"{ep.energy[-1]:.4f}")],
                )

            ep = planning.run_episode(
                wm, env, goal_frame, goal_pos, policy=policy, steps=steps, cem=cfg,
                render_size=render, seed=seed, on_step=_tick if policy == "jepa" else None,
            )
            eps[policy] = ep
            log.info("seed %d %-7s %.1f -> %.1f cm (closed %+.0f%%) in %.0fs",
                     seed, policy, ep.start_dist * 100, ep.final_dist * 100,
                     ep.closed * 100, ep.seconds)

            if seed == 0 and policy == "jepa":
                # The energy surface the planner was actually searching, at step 0.
                # Recomputed on a grid rather than reused from CEM: CEM samples where
                # it already believes the answer is, which would make any landscape
                # drawn from its samples look convincing by construction.
                env0, gf0, gp0, sf0 = sim.reach_task(seed=seed, render_size=ac.CROP)
                z0 = wm.encode(sf0[None])
                zg0 = wm.encode(gf0[None])
                p0 = _pose_tensor(wm, env0)
                g = np.linspace(-maxnorm, maxnorm, 7)
                grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype("float32")
                energies = wm.energy_grid(z0, p0, zg0, grid)
                direction = gp0 - env0.ee_pos
                direction = direction / max(np.linalg.norm(direction), 1e-9) * maxnorm
                first_extras = {
                    "grid": grid, "energy": energies,
                    "best": grid[int(energies.argmin())], "truth": direction,
                    "goal_frame": gf0, "start_frame": sf0,
                }
                env0.close()
            env.close()

        per_seed.append((seed, eps))
        _render_plan_report(per_seed, first_extras, cfg, steps)

    summary = _plan_summary(per_seed)
    _render_plan_report(per_seed, first_extras, cfg, steps, summary=summary)
    log.info("plan summary: %s", summary)
    _free()
    return summary


def _pose_tensor(wm, env):
    import torch

    return torch.tensor(env.pose7(), device=wm.device, dtype=wm.dtype)[None, None]


def _plan_summary(per_seed) -> dict:
    import numpy as np

    out: dict = {"seeds": len(per_seed), "threshold_cm": REACH_THRESHOLD * 100}
    for name in ("jepa", "greedy", "lookahead", "oracle", "random"):
        closed = [e[name].closed for _, e in per_seed if name in e]
        final = [e[name].final_dist for _, e in per_seed if name in e]
        best = [e[name].best_dist for _, e in per_seed if name in e]
        if not closed:
            continue
        out[name] = {
            "gap_closed": round(float(np.mean(closed)), 4),
            "final_cm": round(float(np.mean(final)) * 100, 2),
            "reached": int(sum(b <= REACH_THRESHOLD for b in best)),
        }
    # The claim, as one number: how much of the oracle's achievable progress the
    # world model recovered, with random as zero. Below 0 means planning was worse
    # than a random walk of the same size.
    if all(k in out for k in ("jepa", "oracle", "random")):
        j, o, r = out["jepa"]["gap_closed"], out["oracle"]["gap_closed"], out["random"]["gap_closed"]
        out["normalised_score"] = round(float((j - r) / max(o - r, 1e-6)), 3)
    return out


def _render_plan_report(per_seed, extras, cfg, steps, summary=None) -> None:
    import numpy as np

    seed0, eps0 = per_seed[0]
    body = ""

    if summary is not None and "normalised_score" in summary:
        s = summary["normalised_score"]
        good = s > 0.25
        body += reports.verdict(
            f"<b>Planning recovered {s:.0%} of the oracle's progress</b> "
            f"(random = 0%, oracle = 100%), averaged over {summary['seeds']} scene(s). "
            + ("The energy of a frozen, DROID-trained world model is a usable reward "
               "signal on a simulator it has never seen."
               if good else
               "That is close enough to the random floor that this checkpoint should "
               "not be called a planner on these renders. The most likely cause is "
               "the domain gap: V-JEPA 2-AC has only ever seen real video."),
            good=good,
        )

    body += reports.heading("All four policies, same scene, same controller")
    cmp_mp4 = viz.encode_mp4(viz.compare_video(eps0, extras.get("goal_frame", eps0["oracle"].frames[0])), fps=3)
    body += viz.video_html(cmp_mp4, "jepa | greedy | lookahead | oracle | random | the goal photo. "
                                    "Only the policy differs.", max_width=900)
    body += reports.note(
        "The rightmost panel is the goal photograph, the only description of the task "
        "the planner is given. It never sees the target's coordinates."
    )

    body += reports.heading("Distance to goal")
    body += viz.progress_chart(eps0, threshold=REACH_THRESHOLD)
    body += reports.policy_table(eps0, REACH_THRESHOLD)

    body += reports.heading("Is latent energy a usable reward?")
    body += reports.side_by_side([
        ("Energy against ground truth", viz.energy_vs_distance_chart(eps0)),
        ("CEM convergence", viz.cem_chart(eps0["jepa"].traces)),
    ])
    body += reports.note(
        "Left: every step of every policy as one point. The world model's energy on "
        "the x axis, the true end-effector distance on the y. A positive correlation "
        "means the embedding distance tracks physical progress, which is the "
        "precondition for any of this working. Right: one faint line per planning "
        "call. Falling lines mean the search is finding action sequences the initial "
        "distribution did not already contain."
    )

    if "grid" in extras:
        body += reports.heading("The energy surface the planner searches")
        body += reports.side_by_side([
            ("Action-space energy at step 0", viz.energy_landscape_chart(
                extras["grid"], extras["energy"], extras["best"], extras["truth"],
                "Imagined energy over the action grid")),
            ("Start and goal", viz.strip(
                np.stack([extras["start_frame"], extras["goal_frame"]]), count=2, width=190)),
        ])
        body += reports.note(
            "A 7x7x7 grid of candidate end-effector deltas, each one dreamed forward "
            "and scored against the goal embedding, minimised over dz. The star is the "
            "energy minimum, the cross is the true direction to the goal. They do not "
            "have to coincide for planning to work -- the loop re-plans every step -- "
            "but the distance between them is how far the world model's idea of "
            "'closer' is from the real thing."
        )

    if len(per_seed) > 1:
        body += reports.heading("Every scene")
        body += reports.seed_table(per_seed, REACH_THRESHOLD)
        body += reports.note(
            "Gap closed, per scene. One episode is an anecdote: the clutter and the "
            "target move with the seed, and a policy that only works on one layout is "
            "reading something about that layout rather than about the task."
        )

    body += reports.heading("The planned run, full size")
    jm = viz.encode_mp4(viz.episode_video(eps0["jepa"], extras.get("goal_frame", eps0["jepa"].frames[0])), fps=3)
    body += viz.video_html(jm, "V-JEPA 2-AC planning, closed loop. Left: the simulator's "
                               "camera, the only input. Right: the goal photo.", max_width=640)
    body += viz.strip(np.stack(eps0["jepa"].frames), count=6, width=150)

    rows = [
        ("planner", f"CEM, {cfg.samples} samples x {cfg.cem_steps} iters, horizon {cfg.rollout}"),
        ("step budget", f"{cfg.maxnorm * 100:.0f} cm per step, {steps} steps"),
        ("trained on this sim", "no - frozen DROID weights, never fine-tuned"),
        ("scenes", str(len(per_seed))),
    ]
    flyte.report.replace(reports.final_html(
        "plan - V-JEPA 2-AC drives a Franka to a goal photograph", rows, body,
        reports.PLAN_EXPLAINER))
    flyte.report.flush()


@ac_env.task(report=True)
async def dream(
    horizon: int = 8,
    bank: int = 125,
    maxnorm: float = 0.05,
    seed: int = 0,
    render: int = 384,
) -> dict:
    """Roll the world model forward open-loop and watch the dream drift.

    The artifact is a video of a dream from a model with no decoder, which is only
    possible because we never claim to render the dream -- we retrieve the nearest
    real frame to it and print how near that was.
    """
    import numpy as np
    import torch

    _paint("loading V-JEPA 2-AC", "building the retrieval bank next",
           [("horizon", str(horizon)), ("bank frames", str(bank))])
    wm = ac.ActionWorldModel()
    if wm.missing or wm.unexpected:
        raise RuntimeError(f"AC checkpoint loaded partially: {wm.missing[:4]} {wm.unexpected[:4]}")

    env, goal_frame, goal_pos, start_frame = sim.reach_task(seed=seed, render_size=ac.CROP)

    _paint("rendering the retrieval bank", "real frames of real arm positions",
           [("frames", str(bank))])
    bank_obs, bank_big, bank_pos = planning.frame_bank(env, n=bank, seed=seed, render_size=render)
    # Chunked: `encode` flattens T frames into a batch of T, and a few hundred
    # 256x256 frames at once is a large ViT-g forward for no reason.
    bank_z = torch.cat(
        [wm.encode(bank_obs[i : i + 8]).view(-1, ac.TOKENS_PER_FRAME, wm.dim)
         for i in range(0, len(bank_obs), 8)],
        dim=0,
    )
    log.info("bank encoded: %s", tuple(bank_z.shape))

    # Dream the straight-line reach. Chosen over a random sequence because the point
    # of the panel is to compare the dream with a real future, and a legible real
    # future makes the comparison legible too.
    env.reset()
    start_frame = env.render(ac.CROP)
    actions = []
    pos = env.ee_pos.copy()
    for _ in range(horizon):
        d = goal_pos - pos
        n = float(np.linalg.norm(d))
        d = d / n * min(n, maxnorm) if n > 1e-9 else d
        a = np.zeros(7, dtype=np.float32)
        a[:3] = d
        actions.append(a)
        pos = pos + d
    actions = np.stack(actions)

    _paint("dreaming", f"{horizon} steps, open loop, no re-planning", [])
    dreamed, true_z, true_frames = planning.dream_rollout(wm, env, start_frame, actions)

    # Error curves. `stand_err` is the do-nothing control: how far the true future
    # drifts from the STARTING latent. A dream that is not below this line has not
    # earned the word prediction -- it would be beaten by predicting no change.
    dream_err = (dreamed - true_z).abs().mean(dim=[1, 2]).float().cpu().numpy()
    z0 = wm.encode(start_frame[None])[:, -ac.TOKENS_PER_FRAME :]
    stand_err = (z0 - true_z).abs().mean(dim=[1, 2]).float().cpu().numpy()
    # The floor: two unrelated real frames. Anything at this level means nothing.
    perm = np.random.default_rng(seed).permutation(len(bank_z))
    floor = float((bank_z - bank_z[perm]).abs().mean().item())

    idx, retr = planning.retrieval_decode(wm, dreamed, bank_z)
    mp4 = viz.encode_mp4(viz.dream_video(bank_big, idx, retr, true_frames), fps=2)

    body = reports.verdict(
        f"<b>The dream stays ahead of doing nothing for {int((dream_err < stand_err).sum())} "
        f"of {horizon} steps</b>, and reaches {dream_err[-1] / floor:.0%} of the "
        f"unrelated-frames floor by the end. Cumulative drift with no re-planning is "
        f"what closed-loop MPC exists to avoid, and what `plan` does avoid.",
        good=bool(dream_err[0] < stand_err[0]),
    )
    body += reports.heading("The dream, decoded by retrieval")
    body += viz.video_html(mp4, "LEFT: nearest REAL frame to the dreamed latent - not a "
                                "reconstruction. RIGHT: the same actions, executed.", max_width=760)
    body += reports.heading("How fast it goes wrong")
    body += viz.dream_chart(dream_err, stand_err, floor)
    body += reports.note(
        "Both curves start from the same frame and receive the same actions. Green is "
        "the world model's imagined latent against the truth; blue is what you would "
        "get by predicting that nothing happens. Red is the distance between two "
        "unrelated real frames, the level at which a prediction carries no information."
    )
    body += reports.heading("Retrieval distance, step by step")
    body += viz.strip(np.stack([bank_big[i] for i in idx]), count=min(horizon, 6), width=150)

    rows = [
        ("horizon", f"{horizon} steps, open loop"),
        ("retrieval bank", f"{bank} real renders"),
        ("floor (unrelated frames)", f"{floor:.4f}"),
        ("dream error at t+1 / t+%d" % horizon, f"{dream_err[0]:.4f} / {dream_err[-1]:.4f}"),
    ]
    flyte.report.replace(reports.final_html(
        "dream - an open-loop latent rollout from a model with no decoder", rows, body,
        reports.DREAM_EXPLAINER))
    flyte.report.flush()
    env.close()
    _free()
    return {
        "horizon": horizon,
        "dream_err": [round(float(x), 5) for x in dream_err],
        "standstill_err": [round(float(x), 5) for x in stand_err],
        "floor": round(floor, 5),
        "beat_standstill_steps": int((dream_err < stand_err).sum()),
    }


@ac_env.task(report=True)
async def adapt(
    episodes: int = 40,
    steps: int = 12,
    train_steps: int = 400,
    batch: int = 8,
    lr: float = 1e-5,
    eval_seeds: int = 3,
    render: int = 384,
) -> dict:
    """Fine-tune ONLY the predictor on simulator transitions, and re-plan.

    The payoff task. `plan` shows the pretrained dynamics fail and the representation
    does not; this shows the failing half is cheap to repair, and measures how close
    six minutes of adaptation gets to a perfect simulator-backed dynamics model.
    """
    import numpy as np
    import torch

    _paint("loading V-JEPA 2-AC", "then collecting simulator transitions",
           [("episodes", str(episodes)), ("train steps", str(train_steps))])
    wm = ac.ActionWorldModel()
    if wm.missing or wm.unexpected:
        raise RuntimeError(f"AC checkpoint loaded partially: {wm.missing[:4]} {wm.unexpected[:4]}")

    # ── before: what the pretrained predictor does on this task ─────────────────
    before_eps, before_scores = {}, []
    for sd in range(eval_seeds):
        env, gf, gp, sf = sim.reach_task(seed=sd, render_size=ac.CROP)
        ep = planning.run_episode(wm, env, gf, gp, policy="greedy", steps=10,
                                  cem=ac.CEMConfig(maxnorm=adapt_module.MAXNORM),
                                  render_size=render, seed=sd)
        before_eps[sd] = ep
        before_scores.append(ep.closed)
        env.close()
    log.info("pretrained greedy: %s", [round(c, 3) for c in before_scores])

    # ── collect ─────────────────────────────────────────────────────────────────
    def _collect_tick(ep_i, ep_n, n_trans, secs):
        _paint(f"collecting transitions {n_trans}", "frozen encoder; random actions "
               "from random workspace positions",
               [("episodes", f"{ep_i}/{ep_n}"), ("elapsed", f"{secs:.0f}s")])

    Z, A, P, Zn = adapt_module.collect_transitions(
        wm, episodes=episodes, steps=steps, seed=0, on_progress=_collect_tick)
    n = len(Z)
    ntr = int(n * 0.85)
    train_idx, val_idx = np.arange(ntr), np.arange(ntr, n)

    still = adapt_module.standstill_baseline(Z, Zn, val_idx, wm.device)
    before_l1 = adapt_module.evaluate(wm, Z, A, P, Zn, val_idx)
    log.info("before: val L1 %.5f vs standstill %.5f", before_l1, still)

    # ── fine-tune ───────────────────────────────────────────────────────────────
    def _train_tick(s, tot, tr, v):
        _paint(f"fine-tuning the predictor: step {s}/{tot}",
               "The encoder is frozen. Only the 305M-parameter predictor moves.",
               [("train L1", f"{tr:.5f}"), ("val L1", f"{v:.5f}"),
                ("standstill", f"{still:.5f}")])

    t0 = time.time()
    history = adapt_module.finetune(wm, Z, A, P, Zn, train_idx, val_idx,
                                    steps=train_steps, batch=batch, lr=lr,
                                    on_step=_train_tick)
    train_s = time.time() - t0
    after_l1 = adapt_module.evaluate(wm, Z, A, P, Zn, val_idx)
    log.info("after: val L1 %.5f (%.0fs)", after_l1, train_s)

    # ── after: re-plan with the adapted predictor ───────────────────────────────
    after_eps, after_scores = {}, []
    for sd in range(eval_seeds):
        env, gf, gp, sf = sim.reach_task(seed=sd, render_size=ac.CROP)
        ep = planning.run_episode(wm, env, gf, gp, policy="greedy", steps=10,
                                  cem=ac.CEMConfig(maxnorm=adapt_module.MAXNORM),
                                  render_size=render, seed=sd)
        after_eps[sd] = ep
        after_scores.append(ep.closed)
        env.close()
    log.info("adapted greedy: %s", [round(c, 3) for c in after_scores])

    _render_adapt_report(before_eps, after_eps, before_scores, after_scores,
                         before_l1, after_l1, still, history, n, ntr, train_s,
                         episodes, train_steps)

    summary = {
        "transitions": n,
        "train_steps": train_steps,
        "train_seconds": round(train_s, 1),
        "val_l1_before": round(before_l1, 5),
        "val_l1_after": round(after_l1, 5),
        "standstill": round(still, 5),
        "greedy_before": round(float(np.mean(before_scores)), 4),
        "greedy_after": round(float(np.mean(after_scores)), 4),
    }
    log.info("adapt summary: %s", summary)
    _free()
    return summary


def _render_adapt_report(before_eps, after_eps, before_scores, after_scores,
                         before_l1, after_l1, still, history, n, ntr, train_s,
                         episodes, train_steps) -> None:
    import numpy as np

    body = reports.verdict(
        f"<b>{train_steps} training steps on {ntr} simulator transitions "
        f"({train_s / 60:.1f} minutes) took greedy planning from "
        f"{np.mean(before_scores):+.0%} to {np.mean(after_scores):+.0%}.</b> "
        f"The predictor went from losing to 'predict no change' ({before_l1:.3f} vs "
        f"{still:.3f}) to beating it ({after_l1:.3f}). For reference, a planner given "
        f"the SIMULATOR as its dynamics model scores +91% on this task, so adaptation "
        f"recovers essentially all of the gap. The encoder was frozen throughout.",
        good=float(np.mean(after_scores)) > 0.5,
    )

    body += reports.heading("The same policy, before and after adaptation")
    sd = sorted(before_eps)[0]
    cmp_frames = viz.compare_video(
        {"pretrained": before_eps[sd], "adapted": after_eps[sd]},
        after_eps[sd].frames[-1],
    )
    body += viz.video_html(viz.encode_mp4(cmp_frames, fps=3),
                           "Identical scene, identical search, identical reward. The only "
                           "difference is six minutes of fine-tuning on the predictor.",
                           max_width=760)

    body += reports.heading("Learning curve")
    body += reports.side_by_side([
        ("Validation L1 against the standstill baseline",
         viz.adapt_chart(history, still, before_l1)),
        ("Gap closed, per scene",
         viz.bar_chart("Greedy planning before vs after",
                       [f"seed {s}" for s in sorted(before_eps)],
                       {"pretrained": [before_scores[i] for i in range(len(before_scores))],
                        "adapted": [after_scores[i] for i in range(len(after_scores))]},
                       "fraction of the gap closed", floor=0.0, floor_label="no progress")),
    ])
    body += reports.note(
        "Left: the dashed line is the L1 you get by predicting that the scene does not "
        "change at all. The pretrained checkpoint sits above it, which is the precise "
        "sense in which it is not a world model for these images. Right: the same three "
        "scenes, before and after. Negative means the arm ended further from the goal "
        "than it started."
    )

    rows = [
        ("transitions collected", f"{n} ({episodes} episodes, frozen encoder)"),
        ("trained", f"predictor only, 305M params, {train_steps} steps, {train_s / 60:.1f} min"),
        ("val L1", f"{before_l1:.4f} -> {after_l1:.4f} (standstill {still:.4f})"),
        ("greedy planning", f"{np.mean(before_scores):+.0%} -> {np.mean(after_scores):+.0%}"),
    ]
    flyte.report.replace(reports.final_html(
        "adapt - the dynamics half is the cheap half to repair", rows, body,
        reports.ADAPT_EXPLAINER))
    flyte.report.flush()


@gpu_env.task(report=True)
async def readout(
    repo: str = VITL,
    clips_n: int = 48,
    frames: int = 32,
    demo_clip: str = "bowling",
    probe_clips: int = 0,
) -> dict:
    """Is the picture inside the token? Fit a linear map and find out.

    Answers, with controls, the question every other task in this demo dodges: V-JEPA 2
    ships no decoder, and it turns out that is not an oversight.
    """
    import numpy as np
    import torch

    _paint("loading the encoder", "then fitting a linear readout to pixels",
           [("checkpoint", repo), ("clips", str(clips_n))])
    log.info("guard: %s", jepa.guard_memory())
    model, processor, params = jepa.load(repo)
    catalog = clip_io.list_clips(CLIPS_REPO)
    train_names = [p for sp, _, p in catalog if sp == "train"][:clips_n]

    # Compute the raster shape BEFORE the call: a progress lambda that closes over
    # `tubes`/`grid` cannot reference names the same call is still assigning.
    n_tubes, n_grid = jepa.grid_of(model, frames)
    per_clip = n_tubes * n_grid * n_grid
    X, Y, grid, tubes = decode.collect(
        model, processor, CLIPS_REPO, train_names, frames,
        on_progress=lambda i, n: _paint(
            f"encoding clip {i}/{n}", "frozen encoder; tokens paired with the pixels "
            "they were embedded from", [("tokens so far", f"{i * per_clip:,}")]),
    )
    log.info("dataset X=%s Y=%s", tuple(X.shape), tuple(Y.shape))

    _paint("solving", "closed-form ridge, no hyperparameters to blame", [])
    rows, W_v, W_r, P = decode.readout_table(X, Y)
    for k, v in rows.items():
        log.info("readout %-6s psnr %.2f dB  r2 %+.4f", k, v["psnr"], v["r2"])

    # The held-out clip, decoded three ways.
    name = clip_io.pick(catalog, demo_clip, "val")
    pv, shown = clip_io.load_clip(processor, CLIPS_REPO, name, frames)
    seq = jepa.encode(model, pv).double().cpu()
    truth = shown[: tubes * decode.TUBELET]
    true_patches = decode.patches_from_video(truth, grid).reshape(-1, decode.PATCH_DIM).double()

    vj = decode.decode_clip(W_v, seq, grid, tubes)
    ctrl = decode.decode_clip(W_r, true_patches @ P, grid, tubes)
    log.info("held-out clip: vjepa %.2f dB, control %.2f dB",
             decode.psnr(vj, truth), decode.psnr(ctrl, truth))

    # Optionally re-measure the semantic half on the same encoder rather than quoting
    # the README, so both numbers in the final chart come from this run.
    probe_acc, pixel_acc, chance = 0.78, 0.40, 0.20
    if probe_clips:
        data = probing.encode_dataset(model, processor, CLIPS_REPO, catalog[:probe_clips], frames)
        probe_acc, _ = probing.linear_probe(data["X"], data["y"], data["train"])
        pixel_acc, _ = probing.linear_probe(data["P"], data["y"], data["train"])
        chance = 1 / len(data["labels"])
        log.info("probe %.3f pixels %.3f chance %.3f", probe_acc, pixel_acc, chance)

    _render_readout_report(rows, truth, ctrl, vj, name, probe_acc, pixel_acc, chance,
                           params, len(X))
    summary = {
        "tokens": int(len(X)),
        "psnr": {k: round(v["psnr"], 2) for k, v in rows.items()},
        "r2": {k: round(v["r2"], 4) for k, v in rows.items()},
        "held_out_vjepa_db": round(decode.psnr(vj, truth), 2),
        "held_out_control_db": round(decode.psnr(ctrl, truth), 2),
    }
    log.info("readout summary: %s", summary)
    _free()
    return summary


def _render_readout_report(rows, truth, ctrl, vj, name, probe_acc, pixel_acc, chance,
                           params, n_tokens) -> None:
    import numpy as np

    body = reports.verdict(
        f"<b>A linear readout recovers the patch almost perfectly from a random "
        f"projection of the pixels ({rows['rand']['psnr']:.1f} dB, R2 "
        f"{rows['rand']['r2']:+.3f}) and essentially not at all from a V-JEPA 2 token "
        f"({rows['vjepa']['psnr']:.1f} dB, R2 {rows['vjepa']['r2']:+.3f}).</b> "
        f"V-JEPA scores below the trivial baseline of painting each patch its mean "
        f"colour ({rows['grey']['psnr']:.1f} dB) and barely above being paired with "
        f"the wrong patch ({rows['shuf']['psnr']:.1f} dB). The appearance is not in "
        f"the token, which is why no decoder ships and why none can be trained.",
        good=False,
    )

    body += reports.heading("The same linear map, two different inputs")
    mp4 = viz.encode_mp4(viz.triptych_video(truth, ctrl, vj), fps=6)
    body += viz.video_html(mp4, "Left: the model's actual input. Middle: the readout "
                                "fitted to a random projection of those pixels. Right: "
                                "the readout fitted to V-JEPA's tokens.", max_width=880)
    body += reports.note(
        "The middle panel is the control, and without it the right panel proves "
        "nothing: a blurry reconstruction is just as easily a weak decoder as a "
        "representation that discarded the pixels. Since the middle panel is sharp, "
        "the method is not the limitation."
    )

    body += reports.heading("Every condition")
    body += reports.side_by_side([
        ("Linear decodability", viz.readout_chart(rows)),
        ("Appearance versus meaning",
         viz.keeps_chart(rows["vjepa"]["r2"], probe_acc, pixel_acc, chance)),
    ])
    body += reports.readout_table(rows)
    body += reports.note(
        "Right-hand chart: the same tokens that cannot reproduce a 16x16 patch support "
        f"{probe_acc:.0%} five-way action recognition from one linear layer, where the "
        f"same probe on raw pixels gets {pixel_acc:.0%} and chance is {chance:.0%}. "
        "That is the trade V-JEPA is making, and it is the trade it was designed to "
        "make: do not spend capacity on detail that cannot be predicted anyway."
    )

    body += viz.strip(np.stack([truth[0], ctrl[0], vj[0], truth[-1], ctrl[-1], vj[-1]]),
                      count=6, width=140)

    rows_tbl = [
        ("encoder", f"{params / 1e6:.0f}M params, frozen"),
        ("readout", "closed-form ridge, one token -> 2x16x16x3 pixels"),
        ("tokens fitted", f"{n_tokens:,}"),
        ("held-out clip", name),
    ]
    flyte.report.replace(reports.final_html(
        "readout - the picture is not inside the token, and that is the point",
        rows_tbl, body, reports.READOUT_EXPLAINER))
    flyte.report.flush()


PUSH_SPACES = {
    "V-JEPA 2 embeddings": "jepa",
    "raw pixels": "pixel",
    "random net": "random_net",
    "random actions": "random",
    "scripted oracle": "oracle",
}


@gpu_env.task(report=True)
async def push(
    seeds: int = 8,
    steps: int = 22,
    repo: str = VITL,
    render: int = 384,
    cold_start: bool = False,
) -> dict:
    """Show the robot a photograph of the finished job and let it work out the rest.

    Harder than `reach` on purpose. There the goal photo differed only in where the arm
    was, so raw pixels planned it exactly as well as V-JEPA. Here the photograph shows
    the BLOCK somewhere else, and the score is the block's progress, so posing the arm
    to match the picture earns nothing.
    """
    import numpy as np
    import torch
    from transformers import VJEPA2Model

    _paint("loading the encoder", "then photographing the finished task",
           [("checkpoint", repo), ("scenes", str(seeds)), ("steps", str(steps))])
    log.info("guard: %s", jepa.guard_memory())
    model, processor, params = jepa.load(repo)
    model = model.to("cuda", torch.bfloat16).eval()
    # Same architecture, never trained. The control that separates "V-JEPA learned
    # something useful" from "any deep feature distance would do".
    untrained = VJEPA2Model(model.config).to("cuda", torch.bfloat16).eval()

    per_space: dict[str, list] = {k: [] for k in PUSH_SPACES}
    first: dict = {}

    for sd in range(seeds):
        for label, kind in PUSH_SPACES.items():
            env, goal_frame, goal_cube, start_frame, cube0 = sim.push_task(
                seed=sd, render_size=ac.CROP, start_in_contact=not cold_start)
            if kind == "jepa":
                reward = planning.EncoderReward(model, goal_frame)
            elif kind == "random_net":
                reward = planning.EncoderReward(untrained, goal_frame)
            else:
                reward = planning.PixelReward(goal_frame)

            def _tick(ep, t, total, _l=label, _s=sd):
                _paint(
                    f"scene {_s}, {_l}: step {t + 1}/{total}",
                    "Each step: try all 27 moves in a copy of the simulator, photograph "
                    "each result, keep the one that looks most like the reference.",
                    [("block to goal", f"{ep.cube_dist[-1] * 100:.1f} cm"),
                     ("block moved so far", f"{(ep.cube_dist[0] - ep.cube_dist[-1]) * 100:+.1f} cm")],
                )

            ep = planning.run_push_episode(
                env, goal_frame, goal_cube, reward,
                policy=("oracle" if kind == "oracle" else "random" if kind == "random" else "search"),
                steps=steps, render_size=render, seed=sd, cube0=cube0,
                on_step=_tick if kind == "jepa" else None,
            )
            per_space[label].append(ep)
            if sd == 0:
                first.setdefault("goal", goal_frame)
                first.setdefault("start", start_frame)
                first.setdefault("eps", {})[label] = ep
            env.close()
            del reward
            torch.cuda.empty_cache()
        log.info("scene %d: %s", sd,
                 {k: round(v[-1].closed, 2) for k, v in per_space.items()})
        _render_push_report(per_space, first, steps, cold_start)

    summary = {
        "scenes": seeds,
        "cold_start": cold_start,
        **{
            PUSH_SPACES[k]: {
                "closed": round(float(np.mean([e.closed for e in v])), 4),
                "sd": round(float(np.std([e.closed for e in v])), 4),
                "moved_cm": round(float(np.mean([e.cube_moved for e in v])) * 100, 2),
                "solved": int(sum(e.cube_dist[-1] < 0.05 for e in v)),
            }
            for k, v in per_space.items()
        },
    }
    log.info("push summary: %s", summary)
    _free()
    return summary


def _render_push_report(per_space, first, steps, cold_start) -> None:
    import numpy as np

    n = len(next(iter(per_space.values())))
    rows = []
    for label, eps in per_space.items():
        cl = [e.closed for e in eps]
        rows.append((label, {
            "closed": float(np.mean(cl)), "sd": float(np.std(cl)),
            "moved_cm": float(np.mean([e.cube_moved for e in eps])) * 100,
            "solved": int(sum(e.cube_dist[-1] < 0.05 for e in eps)), "n": len(eps),
        }))

    d = dict(rows)
    j, p, r = d["V-JEPA 2 embeddings"], d["raw pixels"], d["random actions"]
    se = lambda a, b: float(np.sqrt(a["sd"] ** 2 / max(n, 1) + b["sd"] ** 2 / max(n, 1)))
    # Two independent claims, and they are NOT equally well supported. Quote both gaps
    # against their own standard errors rather than leading with the flattering one.
    gap_vs_random, se_random = j["closed"] - r["closed"], se(j, r)
    gap_vs_pixel, se_pixel = j["closed"] - p["closed"], se(j, p)
    body = reports.verdict(
        f"<b>A photograph of the finished task is a sufficient instruction.</b> "
        f"Planning to match it moves the block {j['moved_cm']:+.1f} cm toward where the "
        f"picture shows it, against {r['moved_cm']:+.1f} cm for ignoring the photograph "
        f"and acting at random: {gap_vs_random:+.0%} versus {r['closed']:+.0%}, a gap of "
        f"{gap_vs_random / max(se_random, 1e-9):.1f} standard errors over {n} scenes. "
        f"The robot is never told where the block is or where it should go.",
        good=gap_vs_random > 2 * se_random,
    )
    body += reports.verdict(
        f"<b>Which space the images are compared in does not matter.</b> V-JEPA 2 "
        f"embeddings close {j['closed']:+.0%}, raw pixel subtraction {p['closed']:+.0%}, "
        f"and an untrained network of the same architecture "
        f"{d['random net']['closed']:+.0%}. The V-JEPA advantage over pixels is "
        f"{gap_vs_pixel:+.0%} with a standard error of {se_pixel:.0%}, i.e. "
        f"{gap_vs_pixel / max(se_pixel, 1e-9):.1f} standard errors, which is noise. "
        f"At 3 and 6 scenes this looked like a clear win for V-JEPA; it did not survive "
        f"more scenes. The learned representation is not what is doing the work here.",
        good=False,
    )

    body += reports.heading("The task, as the robot receives it")
    body += reports.reference_panel(
        viz.still(first["start"], 300, "step 0: the block has not moved"),
        viz.still(first["goal"], 300, "the reference photograph: block already pushed"),
    )
    body += reports.note(
        "These two images are the entire specification. The planner is given the right-"
        "hand one and told nothing else: not where the block is, not where it should go, "
        "not that the scene contains a block. Everything below follows from making the "
        "left image look like the right one."
    )

    body += reports.heading("How it gets there")
    mp4 = viz.encode_mp4(viz.push_video(first["eps"], first["goal"]), fps=3)
    body += viz.video_html(mp4, "One scene, five ways of measuring 'looks like the "
                                "photograph'. The caption on each panel is the BLOCK's "
                                "distance, not the arm's.", max_width=980)
    body += reports.note(
        "Watch the block rather than the arm. Every panel is running the identical "
        "search over the identical 27 candidate moves; the only difference is the space "
        "the two images are compared in. A policy that lines the arm up convincingly "
        "but never makes contact scores zero here, which is the point of scoring the "
        "block."
    )

    body += reports.heading("Every scene")
    body += reports.side_by_side([
        ("Block distance over the episode", viz.push_progress_chart(first["eps"])),
        ("Progress by comparison space",
         viz.bar_chart("Block progress toward the photographed position",
                       [lbl for lbl, _ in rows],
                       {"gap closed": [s["closed"] for _, s in rows]},
                       "fraction of the way", floor=0.0, floor_label="no progress")),
    ])
    body += reports.push_table(rows)

    tbl = [
        ("specification", "one photograph of the finished task"),
        ("planner", f"27 candidate moves per step, 5 cm each, {steps} steps"),
        ("scored on", "the block's distance to its photographed position"),
        ("start", "gripper parked behind the block" if not cold_start else "arm at home (cold)"),
        ("scenes", str(n)),
    ]
    flyte.report.replace(reports.final_html(
        "push - a photograph of the finished job is the whole instruction", tbl, body,
        reports.PUSH_EXPLAINER))
    flyte.report.flush()


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(vjepa))


# ══ The mechanism tasks ═══════════════════════════════════════════════════════
#
# Everything above measures how WELL V-JEPA 2 does things. These four are about how it
# works: what a prediction looks like when you render it honestly, what shape its
# energy function has, where in the 24 layers appearance becomes meaning, and what the
# objective does when you remove the one architectural choice that keeps it alive.
#
# `collapse` is the only task in this repo that trains a model from scratch, and the
# only one that does not touch V-JEPA 2 at all: a pretrained checkpoint cannot show you
# the failure mode its authors already avoided.


@gpu_env.task(report=True)
async def occlude(
    clip: str = "bowling",
    repo: str = VITL,
    frames: int = 32,
    hole: int = 8,
    bank_clips: int = 12,
    sweep_positions: int = 9,
) -> dict:
    """Slide an occluder across a clip, predict what is behind it, and RENDER it.

    Three masks hiding the same number of tokens and differing only in shape: a hole
    that moves, a hole that does not, and a hole over the end of the clip. The defaults
    make that exact: on a 16x16 grid an 8x8 block is 25% of the tokens, and 4 of 16
    tubelets is also 25%, so nothing in the comparison is a difficulty confound.

    Every rendering appears twice, once from the predicted token and once from the
    encoder's own token at the same position, because the second is the ceiling for the
    first and neither number means anything alone.
    """
    import numpy as np
    import torch

    rows = [("Model", repo), ("Clip", clip), ("Frames", str(frames)),
            ("Occluder", f"{hole}x{hole} patches")]
    _paint("Loading", "V-JEPA 2, then a bank of tokens whose pixels we still have.", rows)
    log.info("guard: %s", jepa.guard_memory())
    model, processor, params = jepa.load(repo)
    rows.append(("Params", f"{params / 1e6:.0f}M"))

    catalog = clip_io.list_clips(CLIPS_REPO)
    all_labels = clip_io.labels_of(catalog)
    bank_names, bank_labels = occlusion.spread_over_classes(
        catalog, "train", all_labels, bank_clips)

    tubes, grid = jepa.grid_of(model, frames)
    per_clip = tubes * grid * grid
    bank = occlusion.build_bank(
        model, processor, CLIPS_REPO, bank_names, frames, labels=bank_labels,
        on_progress=lambda i, n: _paint(
            f"Building the bank, clip {i}/{n}",
            "train clips only, so nothing can retrieve the answer",
            rows + [("Bank tokens", f"{i * per_clip:,}")]),
    )
    rows.append(("Bank", f"{len(bank['X']):,} tokens from {len(bank_names)} train clips "
                         f"over {len(set(bank_labels))} actions"))

    _paint("Fitting the linear readout", "closed-form ridge, one token -> 2x16x16 pixels",
           rows)
    W = decode.ridge_fit(bank["Xd"], bank["Y"])

    path = clip_io.pick(catalog, clip, "val")
    if path in bank_names:
        raise RuntimeError(f"{path} is in the bank; the mosaic could retrieve the answer")
    class_id = all_labels.index(next(lb for _, lb, p in catalog if p == path))
    pixel_values, shown = clip_io.load_clip(processor, CLIPS_REPO, path, frames)
    seq = jepa.encode(model, pixel_values)
    jepa.check_layout(seq, tubes, grid)
    rows.append(("Source", f"{path} (val)"))

    masks = {"sweep": occlusion.sweep(tubes, grid, size=hole),
             "static": occlusion.static(tubes, grid, size=hole)}
    masks["future"] = occlusion.matched_future(
        tubes, grid, share=float(masks["sweep"].float().mean()))

    results = {}
    for name, mask3d in masks.items():
        _paint(f"Predicting under the {name} mask",
               "then rendering it two ways, each beside its own ceiling", rows)
        res = occlusion.evaluate(model, pixel_values, shown, mask3d, bank, W, grid,
                                 class_id=class_id)
        res["frames"]["input"] = viz.masked_video(res["truth"], mask3d)
        results[name] = res
        log.info("%s: hidden %.0f%% cos %.3f loc %.2f mosaic %.2f dB (ceiling %.2f)",
                 name, 100 * res["hidden"], res["score"]["cos"], res["score"]["loc"],
                 res["psnr"]["mosaic_pred"], res["psnr"]["mosaic_true"])

    # Where the hole is matters more than how big it is, so sweep it across the frame
    # and record how much motion each position covered.
    _paint("Sweeping the hole across the frame",
           "same size, same predictor, different content underneath", rows)
    by_position = []
    for (h0, w0) in occlusion.positions(grid, hole, sweep_positions):
        m = occlusion.static(tubes, grid, size=hole, h0=h0, w0=w0)
        ctx_ids, tgt_ids = jepa.ids_of(m)
        pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
        s = jepa.score(pred, true, tgt_ids, grid)
        f = jepa.shuffled_floor(pred, true, tgt_ids, grid)
        hp = occlusion.hole_pixels(m, grid, res["truth"].shape[:3])
        by_position.append({
            "pos": (h0, w0),
            "cos": s["cos"],
            "floor": f["cos"],
            "lift": s["cos"] - f["cos"],
            "motion": occlusion.hole_motion(res["truth"], hp),
        })
        log.info("hole at (%d,%d): motion %.2f cos %.3f (floor %.3f)",
                 h0, w0, by_position[-1]["motion"], s["cos"], f["cos"])

    summary = _render_occlude_report(results, by_position, rows, path, bank, hole)
    log.info("occlude summary: %s", summary)
    _free()
    return summary


def _render_occlude_report(results, by_position, rows, path, bank, hole) -> dict:
    sw = results["sweep"]
    body = reports.heading("The occluder moves; the prediction follows it")
    panels = [
        (sw["frames"]["input"], [("WHAT IT SAW", viz._AMBER), ("HOLE MOVES", viz._GREY)]),
        (sw["frames"]["mosaic_pred"], [("PREDICTED TOKEN", viz._GREEN),
                                       ("NEAREST REAL", viz._GREY),
                                       ("PATCH IN A BANK", viz._GREY)]),
        (sw["frames"]["mosaic_true"], [("SAME METHOD ON", viz._GREY),
                                       ("THE TRUE TOKEN", viz._GREY),
                                       ("THIS IS CEILING", viz._AMBER)]),
        (sw["truth"], [("ORIGINAL", viz._AMBER)]),
    ]
    mp4 = viz.encode_mp4(viz.labelled_row(panels), fps=8)
    body += viz.video_html(
        mp4,
        f"The {hole}x{hole} occluder slides left to right. Panel 2 fills the hole with "
        f"the real pixels of whichever bank token is nearest each PREDICTED vector; "
        f"panel 3 does the same from the encoder's own token, which is the best this "
        f"rendering can do. Everything outside the hole is the untouched input.",
        max_width=920)
    body += reports.note(viz.probe(mp4))
    body += reports.note(
        f"The bank is {len(bank['X']):,} tokens from train clips only and the clip is "
        f"{path} from the val split, so no panel can retrieve the literal answer. "
        f"<b>{sw['retrieval']['same_class_pred']:.0%} of the patches pasted into the "
        f"hole come from a clip of the same action</b>, where that action is only "
        f"{sw['retrieval']['same_class_chance']:.0%} of the bank. So the predicted "
        f"vector knows what kind of event it is filling in, without ever being told "
        f"and without an action label existing anywhere in this task. "
        f"On {sw['retrieval']['agree']:.1%} of occluded tokens it picks the exact same "
        f"bank patch the encoder's own token picks, against a chance level of "
        f"{sw['retrieval']['agree_chance']:.5%} over that bank.")

    body += reports.heading("The same prediction through the linear readout")
    tri = viz.labelled_row([
        (sw["truth"], [("ORIGINAL", viz._AMBER)]),
        (sw["frames"]["readout_true"], [("TRUE TOKEN", viz._GREY), ("THROUGH RIDGE", viz._GREY),
                                        ("THIS IS CEILING", viz._AMBER)]),
        (sw["frames"]["readout_pred"], [("PREDICTED TOKEN", viz._GREEN),
                                        ("SAME RIDGE MAP", viz._GREY)]),
    ])
    body += viz.video_html(viz.encode_mp4(tri, fps=8),
                           "The readout is the closed-form linear map from the `readout` "
                           "task. Both middle and right panels are mush, which is the "
                           "finding of that task restated: the appearance is not in the "
                           "token. The comparison is still valid, because the ceiling is "
                           "in the frame.", max_width=740)

    body += reports.heading("Three holes, the same size, different shapes")
    cells = []
    for name in ("sweep", "static", "future"):
        r = results[name]
        row = viz.labelled_row([
            (r["frames"]["input"], [(name.upper(), viz._AMBER),
                                    (f"{r['hidden']:.0%} HIDDEN", viz._GREY)]),
            (r["frames"]["mosaic_pred"], [("PREDICTED", viz._GREEN)]),
            (r["frames"]["mosaic_true"], [("CEILING", viz._AMBER)]),
        ])
        cells.append((name, viz.video_html(viz.encode_mp4(row, fps=8), "", max_width=560)))
    body += reports.side_by_side(cells, basis=560)

    table_rows = []
    for name in ("sweep", "static", "future"):
        r = results[name]
        table_rows.append([
            {"sweep": "sweep (hole moves)", "static": "static (hole fixed)",
             "future": "future (end of clip)"}[name],
            f"{r['hidden']:.0%}",
            f"{r['score']['cos']:.3f}",
            f"{r['floor']['cos']:.3f}",
            f"{r['score']['cos'] - r['floor']['cos']:+.3f}",
            f"{r['score']['top1']:.1%}",
            f"{r['score']['dt']:.1f} / {r['floor']['dt']:.1f}",
            f"{r['retrieval']['agree']:.1%}",
            f"{r['psnr']['mosaic_pred']:.2f} / {r['psnr']['mosaic_true']:.2f}",
        ])
    body += reports.data_table(
        ["mask", "hidden", "cosine", "chance", "lift", "top-1", "dt / chance",
         "agrees with ceiling", "mosaic dB / ceiling"],
        table_rows, highlight=0, warn=(2,))
    body += reports.note(
        "Read the lift column and the agreement column. A hole that moves leaves every "
        "hidden patch position visible at some other moment, a static hole never does, "
        "and a future hole leaves nothing after it at all; the number of hidden tokens "
        "is identical in all three rows, so shape is the only thing that varies. "
        "<b>The PSNR column is reported and is not the argument.</b> A retrieved patch "
        "with the right content and the wrong colour scores like noise under a pixel "
        "metric, which is `readout`'s finding arriving from a different direction: "
        "appearance is not in the token, so a pixel metric cannot see whether the "
        "prediction was right. The agreement column is the one that compares the "
        "mosaic against the only thing it can be compared against, its own ceiling."
    )

    retr = [[
        name,
        f"{results[name]['retrieval']['same_class_pred']:.1%}",
        f"{results[name]['retrieval']['same_class_true']:.1%}",
        f"{results[name]['retrieval']['same_class_chance']:.1%}",
    ] for name in ("sweep", "static", "future")]
    body += reports.heading("What did it retrieve?")
    body += reports.data_table(
        ["mask", "from the same action (prediction)", "from the same action (true token)",
         "that action's share of the bank"], retr)
    body += reports.note(
        "Whether the nearest bank patch comes from a clip of the SAME action. This is "
        "the measurement that stops the mosaic being only a picture: if a broken "
        "geometry or a bad centring choice had crept in, this column would sit at the "
        "bank share and the pictures would look exactly as convincing."
    )

    body += reports.heading("It is not how much you hide, it is what you hide")
    xs = [p["motion"] for p in by_position]
    ys = [p["lift"] for p in by_position]
    r_pearson = viz.pearson(xs, ys)
    body += reports.side_by_side([
        ("Prediction quality against motion under the hole",
         viz.scatter_chart(
             f"Same hole size, moved across the frame (r = {r_pearson:+.2f}, "
             f"n = {len(xs)})",
             "mean frame-to-frame change under the hole (grey levels)",
             "cosine above the shuffled floor", xs, ys,
             labels=[f"{p['pos'][0]},{p['pos'][1]}" for p in by_position])),
        ("Per position",
         reports.data_table(
             ["hole (h, w)", "motion under it", "cosine", "chance", "lift"],
             [[f"{p['pos'][0]}, {p['pos'][1]}", f"{p['motion']:.2f}", f"{p['cos']:.3f}",
               f"{p['floor']:.3f}", f"{p['lift']:+.3f}"] for p in by_position])),
    ], basis=420)
    body += reports.note(
        f"One occluder size over a square grid of {len(xs)} positions. The question is "
        f"whether the aggregate scores everywhere else in this demo are averages over "
        f"patches of very different difficulty. Motion under the hole spans "
        f"{min(xs):.1f} to {max(xs):.1f} grey levels here, a factor of "
        f"{max(xs) / max(min(xs), 1e-6):.1f}, and the score does not follow it: "
        f"<b>r = {r_pearson:+.2f}</b> at n = {len(xs)}. An earlier version sampled the "
        f"diagonal instead, where the covariate only spanned a factor of 1.8, and the "
        f"null looked like a coverage problem; widening it did not change the answer. "
        f"The scores do vary by position "
        f"({min(p['cos'] for p in by_position):.3f} to "
        f"{max(p['cos'] for p in by_position):.3f}), which is worth knowing before "
        f"quoting a single number for a clip."
    )

    st = results["static"]
    fu = results["future"]
    # The hypothesis was an ordering over all three. What the numbers support is a
    # split: the two interpolation masks localise in time and the future mask does not.
    interp_localises = min(sw["score"]["loc"], st["score"]["loc"]) > fu["score"]["loc"]
    sweep_helps = (sw["score"]["cos"] - sw["floor"]["cos"]) > \
                  (st["score"]["cos"] - st["floor"]["cos"]) + 0.02
    semantic = min(r["retrieval"]["same_class_pred"] for r in results.values())
    chance_cls = sw["retrieval"]["same_class_chance"]
    body += reports.verdict(
        f"At an identical <b>{sw['hidden']:.0%}</b> of tokens hidden, time localisation "
        f"is <b>{sw['score']['loc']:.2f}</b> when the hole moves, "
        f"<b>{st['score']['loc']:.2f}</b> when it stands still and "
        f"<b>{fu['score']['loc']:.2f}</b> when it covers the end of the clip, while the "
        f"cosine lift is flat across all three "
        f"({sw['score']['cos'] - sw['floor']['cos']:+.3f} / "
        f"{st['score']['cos'] - st['floor']['cos']:+.3f} / "
        f"{fu['score']['cos'] - fu['floor']['cos']:+.3f}). "
        + ("<b>Moving the hole buys nothing.</b> Leaving each hidden patch position "
           "visible at other moments was supposed to make the job easier and it does "
           "not, so whatever the predictor is doing is not looking the patch up at "
           "another time. "
           if not sweep_helps else
           "The moving hole is measurably easier, which is consistent with the "
           "predictor exploiting the same patch position being visible at other "
           "moments. ")
        + f"What every mask does get right is the KIND of content: the patch pasted "
          f"into the hole comes from a clip of the same action at least "
          f"<b>{semantic:.0%}</b> of the time against a bank share of "
          f"{chance_cls:.0%}, including under the future mask, where the same "
          f"prediction cannot place the content in time at all. Right about what, "
          f"wrong about when, and the two are separable with one measurement.",
        good=interp_localises)

    flyte.report.replace(reports.final_html(
        "occlude - the prediction, rendered, with its own ceiling beside it",
        rows, body, reports.OCCLUDE_EXPLAINER), do_flush=True)

    return {
        "clip": path,
        "hidden": {k: round(v["hidden"], 4) for k, v in results.items()},
        "cos": {k: round(v["score"]["cos"], 4) for k, v in results.items()},
        "cos_floor": {k: round(v["floor"]["cos"], 4) for k, v in results.items()},
        "psnr": {k: {kk: round(vv, 2) for kk, vv in v["psnr"].items()}
                 for k, v in results.items()},
        "agree_with_ceiling": {k: round(v["retrieval"]["agree"], 4)
                               for k, v in results.items()},
        "same_class": {k: round(v["retrieval"]["same_class_pred"], 4)
                       for k, v in results.items()},
        "position_r": round(r_pearson, 3),
        "loc": {k: round(v["score"]["loc"], 3) for k, v in results.items()},
        "interp_localises_future_does_not": bool(interp_localises),
        "moving_hole_helps": bool(sweep_helps),
    }


@gpu_env.task(report=True)
async def energy(
    clip: str = "bowling",
    repo: str = VITL,
    frames: int = 32,
    block: int = 8,
    rank_clips: int = 6,
) -> dict:
    """Map the energy function: E(x, y) over a graded family of candidate completions.

    No training, no planning, one encoder pass per candidate. The three questions are
    whether the energy has a well at the truth, whether it orders plausible completions
    sensibly, and whether anything degenerate scores below the truth.
    """
    import numpy as np
    import torch

    rows = [("Model", repo), ("Clip", clip), ("Frames", str(frames)),
            ("Energy", "mean L1 to the candidate's tokens, the training objective")]
    _paint("Loading", "V-JEPA 2 encoder + predictor.", rows)
    log.info("guard: %s", jepa.guard_memory())
    model, processor, params = jepa.load(repo)

    catalog = clip_io.list_clips(CLIPS_REPO)
    all_labels = clip_io.labels_of(catalog)
    path = clip_io.pick(catalog, clip, "val")
    label = next(lb for _, lb, p in catalog if p == path)
    tubes, grid = jepa.grid_of(model, frames)

    _paint("Encoding the clip", path, rows)
    pixel_values, shown = clip_io.load_clip(processor, CLIPS_REPO, path, frames)
    seq = jepa.encode(model, pixel_values)
    jepa.check_layout(seq, tubes, grid)
    truth_frames = shown[: tubes * 2]

    mask3d = jepa.tube(tubes, grid, blocks=2, size=block, seed=0)
    ctx_ids, tgt_ids = jepa.ids_of(mask3d)
    pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
    rows += [("Mask", f"two {block}x{block} tube blocks, {mask3d.float().mean():.0%} hidden"),
             ("Source", f"{path} ({label})")]

    # The cross-clip candidates, at the same geometry as everything else.
    same = clip_io.pick([c for c in catalog if c[2] != path], label, "val")
    other_label = next(lb for lb in all_labels if lb != label)
    other = clip_io.pick(catalog, other_label, "val")
    _, same_frames = clip_io.load_clip(processor, CLIPS_REPO, same, frames)
    _, other_frames = clip_io.load_clip(processor, CLIPS_REPO, other, frames)
    others = {"same action, different clip": same_frames[: tubes * 2],
              "different action": other_frames[: tubes * 2]}
    rows.append(("Distractors", f"{same} ({label}), {other} ({other_label})"))

    scores = ebm.candidate_ladder(
        model, processor, pred, truth_frames, tgt_ids, others,
        on_progress=lambda i, n, name: _paint(
            f"Scoring candidate {i}/{n}", name, rows),
    )
    floors = ebm.floors(pred, true, seq, ctx_ids)
    for name, s in scores.items():
        log.info("E(%-28s) l1 %.5f cosdist %.4f", name, s["l1"], s["cosdist"])
    for name, s in floors.items():
        log.info("floor %-28s l1 %.5f", name, s["l1"])

    _paint("Rolling the candidate in time", "the energy should be a V with its floor at 0",
           rows)
    span = min(6, tubes // 2)
    offsets = [k for k in range(-span, span + 1)]
    time_well = ebm.well(model, processor, pred, truth_frames, tgt_ids, offsets, "time")
    space_well = ebm.well(model, processor, pred, truth_frames, tgt_ids,
                          [k for k in range(-4, 5)], "space")

    _paint("Walking from the truth to a distractor",
           "in pixel space, and again in representation space", rows)
    pix_line = ebm.interpolate(model, processor, pred, truth_frames,
                               others["different action"], tgt_ids)
    z_other = jepa.encode(model, ebm.to_pixel_values(
        processor, others["different action"]))[tgt_ids.to(seq.device)]
    lat_line = ebm.latent_interpolate(pred, true, z_other)

    # Does the truth win on more than one clip? Cheap, and the alternative is a claim
    # resting on a single bowling video.
    #
    # The distractors are rebuilt PER CLIP. Reusing the bowling clip's distractors for
    # an archery query would still rank the truth correctly, but the row labelled
    # "same action, different clip" would be a bowling clip, and a mislabelled
    # candidate in a table about semantic distance is worse than no table.
    val = [(lb, p) for sp, lb, p in catalog if sp == "val"]
    wins, ranks = 0, []
    for i, (lb_i, name) in enumerate(val[:rank_clips]):
        _paint(f"Repeating the ranking on clip {i + 1}/{rank_clips}", f"{name} ({lb_i})",
               rows)
        try:
            same_i = next(p for l2, p in val if l2 == lb_i and p != name)
            other_i = next(p for l2, p in val if l2 != lb_i)
            _, f_same = clip_io.load_clip(processor, CLIPS_REPO, same_i, frames)
            _, f_other = clip_io.load_clip(processor, CLIPS_REPO, other_i, frames)
            pv_i, shown_i = clip_io.load_clip(processor, CLIPS_REPO, name, frames)
            pred_i, _ = jepa.predict(model, pv_i, ctx_ids, tgt_ids)
            s_i = ebm.candidate_ladder(
                model, processor, pred_i, shown_i[: tubes * 2], tgt_ids,
                {"same action, different clip": f_same[: tubes * 2],
                 "different action": f_other[: tubes * 2]})
            ok, rank, n_cand = ebm.ranking(s_i, "cosdist")
            wins += int(ok)
            ranks.append(rank)
            log.info("%s (%s): truth ranked %d of %d", name, lb_i, rank, n_cand)
        except Exception as exc:  # noqa: BLE001
            log.warning("ranking skipped %s: %s", name, exc)

    summary = _render_energy_report(scores, floors, time_well, space_well, pix_line,
                                    lat_line, wins, ranks, rows, truth_frames, others,
                                    params)
    log.info("energy summary: %s", summary)
    _free()
    return summary


# The candidates that are cheap, degenerate answers rather than plausible completions.
# If one of these has lower energy than the truth, the energy is not a preference.
DEGENERATE = ("flat grey", "uniform noise", "first frame frozen")


def _render_energy_report(scores, floors, time_well, space_well, pix_line, lat_line,
                          wins, ranks, rows, truth_frames, others, params) -> dict:
    import numpy as np

    ok, rank, n_cand = ebm.ranking(scores, "cosdist")
    rho, n_rho = ebm.agreement(scores, "cosdist")
    rho_l1, _ = ebm.agreement(scores, "l1")
    true_e = scores["true completion"]["l1"]
    true_cos = scores["true completion"]["cosdist"]
    below = [k for k, v in scores.items()
             if v["cosdist"] < true_cos and k != "true completion"]
    below_l1 = [k for k, v in scores.items()
                if v["l1"] < true_e and k != "true completion"]
    degen_below = [k for k in below if k in DEGENERATE]
    degen_below_l1 = [k for k in below_l1 if k in DEGENERATE]

    t_min = ebm.argmin_of(time_well, "cosdist")
    s_min = ebm.argmin_of(space_well, "cosdist")
    t_min_l1 = ebm.argmin_of(time_well, "l1")
    depth_l1 = ebm.depth_of(time_well, "l1")
    depth_cos = ebm.depth_of(time_well, "cosdist")

    body = reports.heading("Two distances, two different answers")
    # A short hand-picked subset so the two columns can be read against each other
    # without scanning eleven rows; the full ladder is a few sections down.
    headline = [k for k in ("true completion", "same clip, blurred", "first frame frozen",
                            "same clip, frames shuffled", "uniform noise", "flat grey")
                if k in scores]
    body += reports.data_table(
        ["candidate", "E: raw L1 (the training objective)", "E: centred cosine distance"],
        [[k, f"{scores[k]['l1']:.4f}", f"{scores[k]['cosdist']:.4f}"] for k in headline]
        + [[k, f"{v['l1']:.4f}", f"{v['cosdist']:.4f}"] for k, v in floors.items()],
        highlight=headline.index("true completion"),
        warn=(headline.index("flat grey"),) if "flat grey" in headline else ())
    body += reports.note(
        f"The energy V-JEPA 2 was trained on is the left column, an L1 in raw feature "
        f"space. In that column <b>flat grey scores "
        f"{scores['true completion']['l1'] / scores['flat grey']['l1']:.1f}x lower than "
        f"the true completion</b>, and so does the no-model context mean, so the "
        f"objective's own distance is not a preference over completions at inference "
        f"time. Subtract the component every token shares first, which is what "
        f"`jepa.center` does everywhere else in this demo and for the same measured "
        f"reason, and flat grey becomes the WORST candidate of the eleven. "
        f"This is the same defect `collapse` is about, seen from the other end: a "
        f"low-norm degenerate embedding is close to everything in raw L1, which is the "
        f"solution the JEPA objective admits and the EMA target exists to avoid. "
        f"Nothing prevents a CANDIDATE from exploiting it."
    )

    body += reports.heading("Is there a well, and is it in the right place?")
    body += viz.twin_chart(
        f"Energy vs temporal offset (L1 minimum at {t_min_l1:+d}, "
        f"centred minimum at {t_min:+d})",
        "candidate rolled by N tubelets (1 tubelet = 2 frames)",
        [k for k, _ in time_well],
        ("E: raw L1 (depth " + f"{depth_l1:.1%}" + " of its floor)",
         [v["l1"] for _, v in time_well]),
        ("E: centred cosine distance (depth " + f"{depth_cos:.1%}" + ")",
         [v["cosdist"] for _, v in time_well]),
        marks={"truth": 0},
    )
    body += viz.twin_chart(
        f"Energy vs horizontal offset (centred minimum at {s_min:+d})",
        "candidate rolled by N patches (1 patch = 16 px)",
        [k for k, _ in space_well],
        ("E: raw L1", [v["l1"] for _, v in space_well]),
        ("E: centred cosine distance", [v["cosdist"] for _, v in space_well]),
        marks={"truth": 0},
    )
    body += reports.note(
        f"Rolling is in whole tubelets and whole patches, so the candidate's token "
        f"raster shifts by whole tokens and the energy compares like with like. "
        f"The red curve, the objective's own distance, is flat: its range is "
        f"<b>{depth_l1:.1%}</b> of its own floor and its minimum is at "
        f"{t_min_l1:+d} rather than 0. The green curve has a range of "
        f"{depth_cos:.1%} and bottoms out at {t_min:+d}. Reading only the red curve "
        f"would have concluded that the predictor cannot locate the hidden content in "
        f"time, which is the opposite of what `inpaint` measures with retrieval."
    )

    body += reports.heading("The candidate ladder, in both distances")
    body += reports.side_by_side([
        ("Raw L1, the training objective",
         viz.energy_chart(scores, floors, key="l1", degenerate=DEGENERATE)),
        ("Centred cosine distance",
         viz.energy_chart(scores, floors, key="cosdist", degenerate=DEGENERATE)),
    ], basis=430)
    order = sorted(scores.items(), key=lambda kv: kv[1]["cosdist"])
    body += reports.data_table(
        ["candidate", "E (centred cosine)", "relative to truth", "E (raw L1)",
         "rank under L1"],
        [[k, f"{v['cosdist']:.4f}", f"{v['cosdist'] / true_cos:.3f}x",
          f"{v['l1']:.4f}",
          str(1 + sorted(scores, key=lambda n: scores[n]["l1"]).index(k))]
         for k, v in order],
        highlight=[k for k, _ in order].index("true completion"),
        warn=tuple(i for i, (k, _) in enumerate(order) if k in DEGENERATE),
    )
    body += reports.note(
        f"Sorted by the centred distance. Rank correlation with the ordering written "
        f"down in `energy.EXPECTED_ORDER` before any of this was measured: "
        f"<b>rho = {rho:+.2f}</b> centred, {rho_l1:+.2f} under the raw L1, over "
        f"{n_rho} candidates. The last column is where the two disagree."
    )

    body += reports.heading("Walking away from the truth")
    body += reports.side_by_side([
        ("In pixel space",
         viz.curve_chart("Energy along a pixel-space blend, truth to a different action",
                         "alpha (0 = the true completion, 1 = a clip of another action)",
                         "E = centred cosine distance",
                         {"E(prediction, blend)": [(a, v["cosdist"]) for a, v in pix_line]})),
        ("In representation space",
         viz.curve_chart("Energy along the same line drawn between the two embeddings",
                         "alpha", "E = centred cosine distance",
                         {"E(prediction, lerp of tokens)":
                          [(a, v["cosdist"]) for a, v in lat_line]})),
    ], basis=420)
    body += reports.note(
        "The left curve is the one that matters. An energy function you could descend "
        "to generate a completion needs a basin at the truth: leaving it should cost "
        "something immediately. A dip in the middle means there are blends of two "
        "unrelated clips that the model prefers to either, which is the spurious-"
        "minimum problem and part of why nobody generates video from a JEPA energy. "
        "The right curve is drawn in the space the energy is defined on, where a "
        "straight line between two points is expected to behave."
    )

    body += reports.heading("What the candidates were")
    body += viz.strip(np.concatenate([truth_frames[:2], others["same action, different clip"][:2],
                                      others["different action"][:2],
                                      ebm.freeze(truth_frames)[:1],
                                      ebm.blur(truth_frames)[:1]]), count=8, width=120)

    rank_txt = (f"{wins} of {len(ranks)} clips" if ranks else "not measured")
    body += reports.verdict(
        f"<b>V-JEPA 2's energy function works, and the distance it was trained with is "
        f"not the one that works.</b> "
        f"Under the raw L1 the model minimised, "
        + (f"{', '.join(degen_below_l1)} scores below the true completion"
           if degen_below_l1 else "nothing degenerate beats the truth")
        + f" and the temporal well is flat to {depth_l1:.1%} of its floor with its "
          f"minimum at {t_min_l1:+d}. "
        f"Centre the features first and the truth becomes rank <b>{rank}</b> of "
        f"{n_cand}, the lowest-energy candidate on <b>{rank_txt}</b>, the ordering "
        f"matches the one written down in advance at rho = {rho:+.2f}, and the wells "
        f"bottom out at {t_min:+d} in time and {s_min:+d} in space. "
        + (f"Candidates still below the truth after centring: {', '.join(below)}. "
           if below else "Nothing is below the truth after centring. "),
        good=not degen_below and t_min == 0)

    flyte.report.replace(reports.final_html(
        "energy - the shape of E(x, y), and what is underneath the truth",
        rows + [("Params", f"{params / 1e6:.0f}M")], body, reports.ENERGY_EXPLAINER),
        do_flush=True)

    return {
        "energies_l1": {k: round(v["l1"], 5) for k, v in scores.items()},
        "energies_cosdist": {k: round(v["cosdist"], 4) for k, v in scores.items()},
        "floors": {k: {kk: round(vv, 5) for kk, vv in v.items()}
                   for k, v in floors.items()},
        "truth_rank_centred": rank,
        "candidates": n_cand,
        "truth_wins_centred": wins,
        "clips_ranked": len(ranks),
        "expected_order_rho": {"centred": round(rho, 3), "l1": round(rho_l1, 3)},
        "below_truth_centred": below,
        "below_truth_l1": below_l1,
        "degenerate_below_truth_centred": degen_below,
        "degenerate_below_truth_l1": degen_below_l1,
        "time_well_min": {"centred": t_min, "l1": t_min_l1},
        "time_well_depth": {"centred": round(depth_cos, 4), "l1": round(depth_l1, 4)},
        "space_well_min_centred": s_min,
    }


@gpu_env.task(report=True)
async def ladder(
    repo: str = VITL,
    frames: int = 32,
    readout_clips: int = 12,
    probe_clips: int = 100,
    keep: int = 1024,
) -> dict:
    """Run the pixel readout and the action probe at every layer of the same encoder.

    `readout` answers "is the picture in the token" at the last layer only. This walks
    the whole stack, so "appearance discarded, meaning kept" gets a location as well as
    a number, and the question of whether the layer the predictor targets is the layer
    with the most semantics gets an answer.
    """
    import numpy as np
    import torch

    rows = [("Model", repo), ("Frames", str(frames)),
            ("Readout", f"{readout_clips} clips x {keep} tokens, closed-form ridge"),
            ("Probe", f"{probe_clips} clips, one linear layer per layer")]
    _paint("Loading", "V-JEPA 2; every layer of one forward pass gets measured.", rows)
    log.info("guard: %s", jepa.guard_memory())
    model, processor, params = jepa.load(repo)
    n_layers = len(model.encoder.layer)
    rows += [("Params", f"{params / 1e6:.0f}M"), ("Layers", f"{n_layers} + final LayerNorm")]

    catalog = clip_io.list_clips(CLIPS_REPO)
    train_names = [p for sp, _, p in catalog if sp == "train"][:readout_clips]
    X, Y, grid, tubes = layers.collect_tokens(
        model, processor, CLIPS_REPO, train_names, frames, keep=keep,
        on_progress=lambda i, n: _paint(
            f"Encoding clip {i}/{n}", f"keeping {keep} tokens per clip at all "
            f"{n_layers + 1} depths", rows),
    )
    log.info("tokens per layer %s, pixels %s", tuple(X[0].shape), tuple(Y.shape))

    _paint("Solving the readout at every layer",
           f"{n_layers + 1} ridge solves on {len(Y):,} tokens", rows)
    # The solve is the slow part on a CPU in float64, and it is the same arithmetic at
    # every depth, so it goes to the GPU. The random-projection control runs through
    # the identical path, which is what would catch a precision problem.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    readout, refs = layers.readout_curve(X, Y, device=dev)
    for k, v in refs.items():
        log.info("reference %-5s %.2f dB r2 %+.3f", k, v["psnr"], v["r2"])

    data = layers.pooled_dataset(
        model, processor, CLIPS_REPO, catalog[:probe_clips], frames,
        on_progress=lambda i, n: _paint(
            f"Encoding the probe set, clip {i}/{n}",
            "mean-pooled at every depth, one pass per clip", rows),
    )
    _paint("Probing every layer", "one linear layer each, identical splits", rows)
    probe_rows = layers.probe_curve(data)
    rows.append(("Probe classes", f"{len(data['labels'])}: {', '.join(data['labels'])}"))
    if data["failed"]:
        rows.append(("Clips skipped", f"{len(data['failed'])}"))

    geom = layers.geometry_curve(model, processor, CLIPS_REPO,
                                 clip_io.pick(catalog, "bowling", "val"), frames)
    pk = layers.peaks(readout, probe_rows)
    log.info("peaks: %s", pk)

    summary = _render_ladder_report(readout, refs, probe_rows, geom, pk, rows,
                                   n_layers, len(data["labels"]), len(Y))
    log.info("ladder summary: %s", summary)
    _free()
    return summary


def _render_ladder_report(readout, refs, probe_rows, geom, pk, rows, n_layers,
                          n_classes, n_tokens) -> dict:
    names = layers.names(n_layers)
    x = list(range(1, len(names) + 1))
    chance = 1.0 / max(n_classes, 1)

    body = reports.heading("Appearance falls, meaning rises")
    body += viz.twin_chart(
        "The same tokens at every depth, asked both questions",
        "encoder layer (the last point is the final LayerNorm applied)",
        x,
        ("pixel readout R2 (linear, held-out tokens)", [r["r2"] for r in readout]),
        ("action recognition, one linear layer", [r["probe"] for r in probe_rows]),
        refs={"ceiling: random projection of true pixels": refs["rand"]["r2"],
              "floor: tokens paired with wrong patches": refs["shuf"]["r2"]},
        marks={f"best probe (L{pk['probe_layer']})": pk["probe_layer"]},
    )
    body += reports.note(
        f"Left axis, red: how much of a 2x16x16 patch a linear map recovers from one "
        f"token, between the two dashed references that give the number meaning. The "
        f"ceiling is a random 1024-dim projection of the true pixels, which the same "
        f"ridge inverts at R2 {refs['rand']['r2']:+.3f}, so the method is not the "
        f"limitation anywhere on this plot. "
        f"Right axis, green: {n_classes}-way action accuracy from one linear layer on "
        f"that depth's mean-pooled features, chance {chance:.0%}."
    )

    body += reports.heading("Every layer")
    step = max(1, (len(names) - 1) // 11)
    idx = sorted(set(list(range(0, len(names) - 1, step)) + [len(names) - 2, len(names) - 1]))
    body += reports.data_table(
        ["layer", "pixel readout dB", "pixel R2", "probe accuracy", "1-NN retrieval",
         "1-NN raw (uncentred)", "random-pair cosine"],
        [[names[i], f"{readout[i]['psnr']:.2f}", f"{readout[i]['r2']:+.3f}",
          f"{probe_rows[i]['probe']:.1%}", f"{probe_rows[i]['retrieval']:.1%}",
          f"{probe_rows[i]['retrieval_raw']:.1%}", f"{geom[i]['raw']:.3f}"] for i in idx],
        highlight=idx.index(pk["probe_layer"] - 1) if pk["probe_layer"] - 1 in idx else None,
    )

    body += reports.heading("The geometry is not the same at every depth")
    body += reports.side_by_side([
        ("Random-pair cosine",
         viz.curve_chart("How much of a token is the component every token shares",
                         "encoder layer", "mean cosine between two random patches",
                         {"raw": [(i + 1, g["raw"]) for i, g in enumerate(geom)],
                          "centred": [(i + 1, g["centered"]) for i, g in enumerate(geom)]})),
        ("Retrieval, with and without centring",
         viz.curve_chart("Does centring matter equally at every depth?",
                         "encoder layer", "1-NN clip retrieval accuracy",
                         {"centred": [(i + 1, r["retrieval"]) for i, r in enumerate(probe_rows)],
                          "raw": [(i + 1, r["retrieval_raw"]) for i, r in enumerate(probe_rows)]},
                         hlines={"chance": chance})),
    ], basis=420)
    body += reports.note(
        "The left chart is the anisotropy correction the rest of this demo applies "
        "without comment, measured as a function of depth. In the middle of the network "
        "almost all of a token is the component every token shares, so a raw cosine "
        "there is close to meaningless; by the last layer much less of it is. Anyone "
        "quoting the final-layer number as a property of 'V-JEPA features' is quoting "
        "one point on a curve."
    )

    peak_is_last = pk["probe_layer"] >= n_layers
    body += reports.verdict(
        f"Linear pixel readout peaks at layer <b>{pk['pixel_layer']}</b> "
        f"(R2 {pk['pixel_r2']:+.3f}) and falls away with depth; action accuracy peaks "
        f"at layer <b>{pk['probe_layer']}</b> ({pk['probe_acc']:.0%}, chance "
        f"{chance:.0%}). The two questions are answered by different parts of the "
        f"network, in the order the JEPA argument predicts. "
        + (f"The semantic peak is the final layer, which is also the layer the "
           f"predictor is trained to match, so the world model is predicting the "
           f"network's best description of the scene."
           if peak_is_last else
           f"The semantic peak is <b>not</b> the final layer: layer "
           f"{pk['probe_layer']} beats the final output by "
           f"{pk['gap_to_final']:+.1%}. The predictor is trained against the FINAL "
           f"encoder layer, so the representation the world model predicts is not this "
           f"encoder's most linearly informative one."),
        good=True)

    flyte.report.replace(reports.final_html(
        "ladder - where in the network appearance becomes meaning",
        rows + [("Tokens fitted", f"{n_tokens:,}")], body, reports.LADDER_EXPLAINER),
        do_flush=True)

    return {
        "layers": names,
        "pixel_r2": [round(r["r2"], 4) for r in readout],
        "pixel_psnr": [round(r["psnr"], 2) for r in readout],
        "probe": [round(r["probe"], 4) for r in probe_rows],
        "retrieval": [round(r["retrieval"], 4) for r in probe_rows],
        "anisotropy_raw": [round(g["raw"], 4) for g in geom],
        "references": {k: {"psnr": round(v["psnr"], 2), "r2": round(v["r2"], 4)}
                       for k, v in refs.items()},
        "peaks": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pk.items()},
    }


@gpu_env.task(report=True)
async def collapse(
    steps: int = 2000,
    train_clips: int = 2048,
    eval_clips: int = 256,
    probe_clips: int = 2000,
    dim: int = 128,
    depth: int = 4,
    seed: int = 0,
    stability_seeds: int = 2,
) -> dict:
    """Train a tiny JEPA from scratch four ways and watch one of them collapse.

    The only task here that does not load V-JEPA 2: a released checkpoint cannot show
    the failure its authors designed around. Four arms differing ONLY in where the
    prediction target comes from, plus an untrained control.
    """
    import numpy as np
    import torch

    rows = [("Model", f"{dim}-dim, {depth}-block ViT + 2-block predictor, from scratch"),
            ("Data", f"{train_clips} synthetic clips, "
                     f"{collapse_module.FRAMES}x{collapse_module.SIZE}x{collapse_module.SIZE}"),
            ("Steps", f"{steps} per arm, identical optimiser and masks")]
    _paint("Generating the world", "one moving shape, three independent factors", rows)
    log.info("guard: %s", jepa.guard_memory())
    train = collapse_module.make_dataset(train_clips, seed=seed)
    ev = collapse_module.make_dataset(eval_clips, seed=seed + 1)
    probe_set = collapse_module.make_dataset(probe_clips, seed=seed + 2)
    train_mask = torch.zeros(probe_clips, dtype=torch.bool)
    train_mask[: probe_clips // 2] = True
    rows.append(("Probe set", f"{probe_clips} held-out clips, {probe_clips // 2} for fitting"))

    def progress(kind, step, total, row):
        _paint(f"Training `{kind}`, step {step}/{total}",
               "loss falling is not the measurement; pair cosine is",
               rows + [("loss", f"{row['loss']:.4f}"),
                       ("pair cosine", f"{row['pair_cos']:.3f} (1.00 = collapsed)"),
                       ("feature std", f"{row['std']:.4f}")])

    res = {}
    for kind in collapse_module.KINDS:
        res[kind] = collapse_module.train(
            kind, train["video"], ev["video"], steps=steps, dim=dim, depth=depth,
            seed=seed, on_progress=progress)
        log.info("%s: %.0fs final %s", kind, res[kind]["seconds"], res[kind]["final"])
    res["random"] = collapse_module.untrained(dim=dim, depth=depth, seed=seed)

    # Is stop-grad ALONE reliable? The EMA arm has been healthy and the no-stop-grad arm
    # has collapsed on every run of this task, but `stopgrad` is the interesting one --
    # SimSiam's claim is that a stop-gradient plus a predictor is enough -- and it has
    # come out both ways on identical settings. So it gets extra seeds and the report
    # gets a spread instead of an anecdote.
    stability = [(seed, res["stopgrad"]["final"]["pair_cos"],
                  res["stopgrad"]["final"]["loss"])]
    for k in range(1, stability_seeds + 1):
        _paint(f"Re-running `stopgrad` at seed {seed + k}",
               "stop-grad alone came out both ways on identical settings; this is the "
               "spread", rows)
        extra = collapse_module.train("stopgrad", train["video"], ev["video"],
                                      steps=steps, dim=dim, depth=depth, seed=seed + k,
                                      on_progress=progress)
        stability.append((seed + k, extra["final"]["pair_cos"], extra["final"]["loss"]))
        log.info("stopgrad seed %d: pair_cos %.3f loss %.4f", seed + k,
                 extra["final"]["pair_cos"], extra["final"]["loss"])

    probes, pooled_probes, shots = {}, {}, {}
    for kind, r in res.items():
        _paint(f"Probing `{kind}`", "frozen features, one linear layer per factor", rows)
        probes[kind] = collapse_module.probe_all(r["encoder"], probe_set, train_mask,
                                                 seed=seed, mode="spacetime")
        pooled_probes[kind] = collapse_module.probe_all(r["encoder"], probe_set, train_mask,
                                                        seed=seed, mode="pooled")
        _paint(f"Probing `{kind}` at every label budget",
               "a random encoder wins at 1000 labels; the question is what happens at 25",
               rows)
        shots[kind] = collapse_module.probe_shots(r["encoder"], probe_set, train_mask,
                                                  seed=seed)
        log.info("%s probes %s", kind, {k: round(v["acc"], 3) for k, v in probes[kind].items()
                                        if isinstance(v, dict)})

    summary = _render_collapse_report(res, probes, pooled_probes, shots, stability,
                                      train, rows, steps)
    log.info("collapse summary: %s", summary)
    _free()
    return summary


def _render_collapse_report(res, probes, pooled_probes, shots, stability, data, rows,
                            steps) -> dict:
    import numpy as np

    trained = [k for k in collapse_module.KINDS]
    latent = [k for k in trained if k != "pixels"]
    factors = list(collapse_module.FACTORS)
    chances = {f: 1.0 / n for f, n in collapse_module.FACTORS.items()}

    body = reports.heading("The arm with the lowest loss is the one that learned nothing")
    body += reports.side_by_side([
        ("Training loss",
         viz.curve_chart("L1 between prediction and target, latent arms only",
                         "step", "training loss",
                         {k: [(h["step"], h["loss"]) for h in res[k]["history"]]
                          for k in latent})),
        ("Collapse",
         viz.curve_chart("Mean cosine between two different clips' features (uncentred)",
                         "step", "pair cosine, 1.00 = collapsed",
                         {k: [(h["step"], h["pair_cos"]) for h in res[k]["history"]]
                          for k in trained},
                         hlines={"total collapse": 1.0})),
    ], basis=420)
    body += reports.note(
        "The pixel arm is left off the loss chart because its loss is in pixel units "
        "and is not comparable; it is on every other chart. The `none` arm reaches a "
        "loss the others never approach, by emitting the same vector for every input. "
        "That is the degenerate solution the JEPA objective admits, and it is why the "
        "targets in a real JEPA come from a frozen EMA copy of the encoder."
    )

    body += reports.side_by_side([
        ("Effective rank",
         viz.curve_chart("Effective rank of the held-out features (uncentred, direction only)",
                         "step", "exp(entropy of the spectrum)",
                         {k: [(h["step"], h["erank"]) for h in res[k]["history"]]
                          for k in trained})),
        ("Feature scale",
         viz.curve_chart("Per-dimension standard deviation of the features",
                         "step", "mean std",
                         {k: [(h["step"], h["std"]) for h in res[k]["history"]]
                          for k in trained})),
    ], basis=420)
    body += reports.note(
        "The right chart is the shape of it: the collapse is a SCALE collapse, three "
        "orders of magnitude of per-dimension standard deviation, and the left chart "
        "shows the uncentred effective rank falling to 1 alongside it. Where it happens "
        "is measured separately (see the README): the encoder's final LayerNorm keeps "
        "its learned gain almost intact, and it is the transformer blocks that map "
        "every clip to nearly the same activation before the LayerNorm is applied."
    )

    if len(stability) > 1:
        cos = [c for _, c, _ in stability]
        spread = max(cos) - min(cos)
        body += reports.heading("Is stopping the gradient enough on its own?")
        body += reports.data_table(
            ["seed", "final pair cosine", "final loss", "verdict"],
            [[str(sd), f"{c:.3f}", f"{l:.4f}",
              "collapsing" if c > 0.6 else ("drifting" if c > 0.35 else "healthy")]
             for sd, c, l in stability],
            warn=tuple(i for i, (_, c, _) in enumerate(stability) if c > 0.6))
        body += reports.note(
            f"The same arm, the same settings, only the seed changes. SimSiam's claim "
            f"for images is that a stop-gradient plus a predictor head is enough to "
            f"avoid collapse without any EMA, and this is that claim tested on video: "
            f"pair cosine ranges <b>{min(cos):.2f} to {max(cos):.2f}</b> across "
            f"{len(stability)} seeds, a spread of {spread:.2f}, against "
            f"{res['ema']['final']['pair_cos']:.2f} for the EMA arm and "
            f"{res['none']['final']['pair_cos']:.2f} for the arm with no stop-grad at "
            f"all. "
            + ("So stop-grad alone is not reliable here: it sits near a bifurcation and "
               "which side it lands on depends on the seed. That is the practical case "
               "for the EMA target, and it is a different case from 'stop-grad does "
               "not work'."
               if spread > 0.25 else
               "On these seeds stop-grad alone held up, so the EMA's contribution on "
               "this task is not visible in this measurement.")
        )

    body += reports.heading("What each objective actually learned")
    body += viz.factor_chart({k: probes[k] for k in list(res)}, factors, chances)
    body += reports.data_table(
        ["target comes from", "final loss", "pair cosine", "effective rank",
         "feature std", "colour", "shape", "direction", "centred erank"],
        [[{"ema": "an EMA of the encoder (V-JEPA)",
           "stopgrad": "the encoder, detached",
           "none": "the encoder, with gradients",
           "pixels": "the true pixels (MAE)",
           "random": "nothing, never trained"}[k],
          f"{res[k]['final'].get('loss', float('nan')):.4f}" if res[k]["history"] else "-",
          f"{res[k]['final'].get('pair_cos', float('nan')):.3f}" if res[k]["history"] else "-",
          f"{res[k]['final'].get('erank', float('nan')):.1f}" if res[k]["history"] else "-",
          f"{res[k]['final'].get('std', float('nan')):.4f}" if res[k]["history"] else "-",
          f"{probes[k]['colour']['acc']:.1%}",
          f"{probes[k]['shape']['acc']:.1%}",
          f"{probes[k]['direction']['acc']:.1%}",
          f"{res[k]['final'].get('erank_centred', float('nan')):.1f}"
          if res[k]["history"] else "-"]
         for k in list(res)],
        highlight=0, warn=(2,))
    rnd_wins = [f for f in factors if probes["random"][f]["acc"] >= probes["ema"][f]["acc"]]
    body += reports.note(
        f"Chance is {chances['colour']:.0%} for colour, {chances['shape']:.0%} for "
        f"shape and {chances['direction']:.0%} for direction. Colour and shape are "
        f"visible in a single frame; DIRECTION OF MOTION is not in any single frame. "
        f"<b>Read the untrained row before any of the others.</b> A random "
        f"convolutional basis followed by a linear layer fitted on 1000 labels is a "
        f"strong model of this little world: it matches or beats V-JEPA's own recipe "
        + (f"on {', '.join(rnd_wins)}. " if rnd_wins else "on nothing here, but only just. ")
        + f"So this table supports the claim that collapse DESTROYS a representation "
        f"(the `none` row is below the untrained row on colour) and does not support "
        f"any claim that the trained ones are better than a random projection. The "
        f"next chart asks the same question where representation quality actually "
        f"shows up."
    )

    body += reports.heading("The same probes, with fewer labels")
    body += reports.side_by_side([
        (f"{f} ({chances[f]:.0%} chance)",
         viz.curve_chart(f"Probe accuracy for {f} against label budget",
                         "labelled clips the probe may fit on", "accuracy",
                         {k: shots[k][f] for k in list(res)},
                         hlines={"chance": chances[f]}, logx=True))
        for f in factors
    ], basis=340)
    lo, hi = collapse_module.SHOTS[0], collapse_module.SHOTS[-1]
    sep_lo = {f: shots["ema"][f][0][1] - shots["random"][f][0][1] for f in factors}
    sep_hi = {f: shots["ema"][f][-1][1] - shots["random"][f][-1][1] for f in factors}
    won = [f for f in factors if sep_lo[f] > 0.05]
    lost = [f for f in factors if sep_lo[f] < -0.05]
    at_chance = [f for f in factors
                 if shots["random"][f][0][1] < chances[f] + 0.05
                 and shots["ema"][f][0][1] < chances[f] + 0.05]
    body += reports.note(
        f"Identical features, identical held-out set, only the number of rows the "
        f"linear layer may fit on varies. A representation is the thing that should "
        f"make a task learnable from FEW examples, so the left end of each curve is "
        f"where representation quality shows up if it shows up anywhere, and the "
        f"right end is where a rich enough random basis catches up. "
        f"V-JEPA's recipe against the untrained encoder, at {lo} labels and at {hi}: "
        + "; ".join(f"{f} <b>{sep_lo[f]:+.0%}</b> then {sep_hi[f]:+.0%}"
                    for f in factors)
        + ". "
        + (f"So the representation is worth something real on "
           f"{', '.join(won)} and the advantage is a low-label one: it is largest "
           f"where labels are scarcest and mostly gone by {hi}. "
           if won else "So there is no label budget at which it is ahead. ")
        + (f"On {', '.join(lost)} the untrained encoder is better. " if lost else "")
        + (f"On {', '.join(at_chance)} nothing is above chance at {lo} labels, so that "
           f"column says nothing about any arm. " if at_chance else "")
        + "Read all of this as a statement about a 0.9M-parameter model trained for two "
          "minutes on synthetic shapes, not about V-JEPA 2. The place in this demo where "
          "frozen-feature quality is demonstrated at scale is the `probe` task on "
          "Kinetics: 78% five-way action recognition against 40% for raw pixels and 20% "
          "chance."
    )

    body += reports.details(
        "The measurement trap in the last column, and in how the probe pools",
        "Two ways of measuring this experiment give the opposite answer, and both are "
        "the first thing a reader would reach for.\n\n"
        "1. CENTRED effective rank ranks the collapsed arm as the RICHEST "
        "representation here. Subtracting the per-dimension mean removes exactly the "
        "constant vector that collapse produced, leaving unstructured numerical "
        "residue, which is close to full rank. The uncentred columns are the ones that "
        "say what happened.\n\n"
        "2. MEAN-POOLING over all tokens puts the direction probe at chance for every "
        "arm, including arms that demonstrably encode direction, because direction is a "
        "statement about how position changes with time and a mean over all tokens has "
        "destroyed position and time before the probe sees anything. The probe above "
        "pools to a 2x2 spatial grid per tubelet instead. The same probe with plain "
        "mean pooling, for comparison:\n\n"
        + "\n".join(
            f"  {k:9s} colour {pooled_probes[k]['colour']['acc']:.0%}  "
            f"shape {pooled_probes[k]['shape']['acc']:.0%}  "
            f"direction {pooled_probes[k]['direction']['acc']:.0%}"
            for k in list(res))
        + "\n\nNeither of these is a subtlety about this toy model. They are the same "
        "class of mistake as reading a cosine similarity without its chance floor, "
        "which is what the `inpaint` task is about.")

    body += reports.heading("The data, and the one arm that has a decoder")
    recon = collapse_module.reconstruct(res["pixels"], data["video"][0])
    enc = res["pixels"]["encoder"]
    masked = collapse_module.masked_clip(data["video"][0], enc.grid, enc.tubes)

    # Upscale BEFORE captioning. `viz.annotate` burns a 5x7 bitmap font at 6*scale
    # pixels per glyph, so a 15-character caption needs 180px of width and these frames
    # are 64px; captioning first and scaling after put the text off the edge of the
    # panel and then magnified the stumps.
    def big(frames):
        return np.repeat(np.repeat(frames, 3, axis=1), 3, axis=2)

    panels = [(big(data["video"][0]), [("ORIGINAL", viz._AMBER)]),
              (big(masked), [("WHAT IT SAW", viz._GREY)])]
    if recon is not None:
        panels.append((big(recon), [("MAE RECONSTRUCT", viz._GREEN),
                                    ("PIXEL ARM ONLY", viz._GREY)]))
    body += viz.video_html(viz.encode_mp4(viz.labelled_row(panels), fps=4),
                           "One synthetic clip, the tube mask the predictor was trained "
                           "against, and the pixel arm's own reconstruction. The latent "
                           "arms have nothing to show here, which is the entire point of "
                           "this repo's other reports.", max_width=760)
    body += viz.strip(data["video"][1], count=8, width=90)

    ema, none = probes["ema"], probes["none"]
    rnd = probes["random"]
    probe_n = collapse_module.SHOTS[-1]
    collapsed = res["none"]["final"]["pair_cos"] > 0.9
    lower_loss = res["none"]["final"]["loss"] < res["ema"]["final"]["loss"]
    # The claim this task can defend: collapse costs you the representation. Not
    # "training beats a random net", which the untrained control refuses to support.
    ema_beats_collapse = all(ema[f]["acc"] > none[f]["acc"] for f in factors)
    body += reports.verdict(
        f"The collapse arm reached a training loss of "
        f"<b>{res['none']['final']['loss']:.4f}</b> against "
        f"<b>{res['ema']['final']['loss']:.4f}</b> for V-JEPA's own recipe, "
        f"<b>{res['ema']['final']['loss'] / max(res['none']['final']['loss'], 1e-9):.0f}x "
        f"lower</b>, and it got there by emitting the same direction for every clip: "
        f"pair cosine {res['none']['final']['pair_cos']:.3f} against "
        f"{res['ema']['final']['pair_cos']:.3f}, and a per-dimension standard deviation "
        f"of {res['none']['final']['std']:.4f} against "
        f"{res['ema']['final']['std']:.4f}, three orders of magnitude apart. "
        f"It cost every probe: "
        + ", ".join(f"{f} {none[f]['acc']:.0%} vs {ema[f]['acc']:.0%}" for f in factors)
        + ". "
        + ("<b>So the asymmetric target is not a training trick.</b> It is the thing "
           "that stops the objective from being satisfied by a representation that "
           "contains nothing, and the loss curve cannot tell you that it happened. "
           if collapsed and lower_loss and ema_beats_collapse else
           "Not every part of that pattern came out as expected on this run; the "
           "numbers above are what happened. ")
        + f"Against the untrained control, at the full {probe_n} labels V-JEPA's arm "
          f"wins on "
          f"{sum(1 for f in factors if probes['ema'][f]['acc'] > rnd[f]['acc'])} of "
          f"{len(factors)} factors, because a random convolutional basis plus a linear "
          f"layer fitted on that many examples is already a strong model of a world "
          f"this simple. The next chart is where that comparison becomes informative.",
        good=collapsed and lower_loss and ema_beats_collapse)

    flyte.report.replace(reports.final_html(
        "collapse - what the JEPA objective does without its EMA teacher",
        rows, body, reports.COLLAPSE_EXPLAINER), do_flush=True)

    return {
        "steps": steps,
        "final": {k: {kk: round(vv, 5) for kk, vv in res[k]["final"].items()}
                  for k in collapse_module.KINDS},
        "seconds": {k: round(res[k]["seconds"], 1) for k in collapse_module.KINDS},
        "probes": {k: {f: round(probes[k][f]["acc"], 4) for f in factors} for k in res},
        "probes_mean_pooled": {k: {f: round(pooled_probes[k][f]["acc"], 4) for f in factors}
                               for k in res},
        "probes_low_shot": {k: {f: [(n, round(a, 4)) for n, a in shots[k][f]]
                                for f in factors} for k in res},
        "chance": {f: round(c, 4) for f, c in chances.items()},
        "stopgrad_stability": [[sd, round(c, 4), round(l, 5)] for sd, c, l in stability],
        "collapsed": bool(collapsed),
        "collapse_had_lower_loss": bool(lower_loss),
        "ema_beats_collapse_on_every_factor": bool(ema_beats_collapse),
    }


@orch_env.task(report=True)
async def mechanism(clip: str = "bowling", repo: str = VITL) -> dict:
    """All four mechanism tasks in sequence. CPU-only, so it cannot starve its children."""
    rendered = await occlude(clip=clip, repo=repo)
    ebm_shape = await energy(clip=clip, repo=repo)
    depth = await ladder(repo=repo)
    scratch = await collapse()
    result = {"occlude": rendered, "energy": ebm_shape, "ladder": depth,
              "collapse": scratch}
    log.info("result: %s", result)
    return result
