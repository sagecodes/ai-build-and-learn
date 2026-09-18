"""Run the real code paths on the host before spending a pod on them.

    .venv/bin/python smoke_test.py            # ViT-L, one clip, ~2 min
    .venv/bin/python smoke_test.py --full     # also encodes all 100 clips for the probe
    .venv/bin/python smoke_test.py --ac       # also the V-JEPA 2-AC world model + MuJoCo
    .venv/bin/python smoke_test.py --mech     # also occlude/energy/ladder/collapse, ~4 min

Deliberately exercises the parts that are easy to get silently wrong rather than the
parts that are easy to test: token layout, overlay alignment, and whether the scores
actually beat their own chance floor. A green run here means `flyte run` is worth it.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

import clips as clip_io
import collapse as collapse_module
import decode
import energy as ebm
import jepa
import layers
import occlude as occlusion
import probing
import viz
from config import CLIPS_REPO, VITL

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
FRAMES = 32


def main(full: bool, mech: bool = False) -> int:
    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print("guard:", jepa.guard_memory())
    model, processor, params = jepa.load(VITL)
    print(f"loaded {VITL}: {params / 1e6:.0f}M params")

    catalog = clip_io.list_clips(CLIPS_REPO)
    labels = clip_io.labels_of(catalog)
    print(f"catalog: {len(catalog)} clips, classes {labels}")
    check("catalog non-empty", len(catalog) > 0)

    path = clip_io.pick(catalog, "bowling")
    pixel_values, shown = clip_io.load_clip(processor, CLIPS_REPO, path, FRAMES)
    tubelets, grid = jepa.grid_of(model, FRAMES)
    print(f"clip {path}: pv {tuple(pixel_values.shape)} shown {shown.shape}")

    # The overlay contract: one patch is exactly one (256/grid)^2 pixel block of `shown`.
    check("shown frames match the model input", shown.shape[0] == pixel_values.shape[1]
          and shown.shape[1] == model.config.crop_size,
          f"{shown.shape} vs crop {model.config.crop_size}")
    check("patch grid divides the frame", shown.shape[1] % grid == 0)

    seq = jepa.encode(model, pixel_values)
    jepa.check_layout(seq, tubelets, grid)          # raises if the raster ever changes
    check("token layout", True, f"{tubelets} tubelets x {grid}x{grid} = {seq.shape[0]}")

    aniso = jepa.anisotropy(seq)
    print(f"anisotropy: raw {aniso['raw']:.3f} -> centered {aniso['centered']:.3f}")
    check("centering removes the shared component", abs(aniso["centered"]) < 0.05
          and aniso["raw"] > aniso["centered"])

    loc = {}
    for name, mask3d in (
        ("tube", jepa.tube(tubelets, grid, blocks=2, size=8, seed=0)),
        ("future", jepa.future(tubelets, grid, context=0.5)),
    ):
        ctx_ids, tgt_ids = jepa.ids_of(mask3d)
        check(f"{name} mask ids partition the sequence",
              len(ctx_ids) + len(tgt_ids) == seq.shape[0] and len(tgt_ids) > 0,
              f"{len(ctx_ids)} context / {len(tgt_ids)} target")

        pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
        check(f"{name} predictor shape", pred.shape == true.shape,
              f"{tuple(pred.shape)}")

        s = jepa.score(pred, true, tgt_ids, grid)
        floor = jepa.shuffled_floor(pred, true, tgt_ids, grid)
        loc[name] = jepa.localization(s, floor)
        print(f"  {name}: masked {len(tgt_ids) / seq.shape[0]:.0%}  "
              f"cos {s['cos']:.3f} (chance {floor['cos']:.3f})  "
              f"top1 {s['top1']:.1%} (chance {floor['top1']:.1%})  "
              f"median dt {s['dt']:.1f} (chance {floor['dt']:.1f}) dh {s['dh']:.1f} "
              f"dw {s['dw']:.1f}  time-localised {loc[name]:.2f}")
        check(f"{name} beats its shuffled floor", s["cos"] > floor["cos"])

        field = jepa.per_patch_cos(pred, true, tgt_ids, tubelets, grid)
        check(f"{name} score map masks correctly",
              int(np.isfinite(field).sum()) == len(tgt_ids),
              f"{int(np.isfinite(field).sum())} finite of {field.size}")

        masked = viz.masked_video(shown, mask3d)
        check(f"{name} masked video differs from the input", not np.array_equal(masked, shown))
        pair = viz.side_by_side_video(masked, viz.heat_video(shown, field))
        mp4 = viz.encode_mp4(pair, fps=12)
        report = viz.probe(mp4)
        print(f"  {name} mp4: {len(mp4) / 1024:.0f} KB, {report}")
        check(f"{name} mp4 decodes and is not black", "black" in report
              and report.split("black")[0].strip().endswith("0"), report[:80])

    # The claim the whole `inpaint` task rests on. If this ever flips, the report's
    # verdict is wrong and the README needs rewriting, so it is a check and not a note.
    check("tube mask localises in time better than the future mask",
          loc["tube"] > loc["future"],
          f"tube {loc['tube']:.2f} vs future {loc['future']:.2f}")

    # Charts must render headless (Agg) or the report silently loses them.
    check("horizon chart renders",
          viz.horizon_chart({"x": [(1, 0.5), (2, 0.4)]}, {"chance": 0.3}).startswith("<img"))
    check("bar chart renders",
          viz.bar_chart("t", ["a"], {"s": [0.5]}, "y", floor=0.2).startswith("<img"))
    check("confusion chart renders",
          viz.confusion_chart(np.eye(3, dtype=int), list("abc"), "t").startswith("<img"))

    if mech:
        print("\nmechanism tasks (occlude / energy / ladder / collapse)")
        check_mechanism(check, model, processor, catalog, labels, path, pixel_values,
                        shown, seq, tubelets, grid)

    if full:
        print("encoding the full dataset for the probe...")
        data = probing.encode_dataset(
            model, processor, CLIPS_REPO, catalog, FRAMES,
            on_progress=lambda i, n: print(f"  {i}/{n}", end="\r"),
        )
        acc, pred = probing.linear_probe(data["X"], data["y"], data["train"])
        acc_px, _ = probing.linear_probe(data["P"], data["y"], data["train"])
        ret, _ = probing.retrieval(data["X"], data["y"])
        chance = 1 / len(data["labels"])
        print(f"\nprobe {acc:.1%}  pixels {acc_px:.1%}  1-NN {ret:.1%}  chance {chance:.0%}")
        check("probe beats the pixel baseline", acc > acc_px)
        check("retrieval beats chance", ret > 2 * chance)
        check("no clips lost", not data["failed"], f"{len(data['failed'])} failed")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0


def check_mechanism(check, model, processor, catalog, labels, path, pixel_values,
                    shown, seq, tubelets, grid) -> None:
    """The four mechanism tasks, on the real code paths at reduced size.

    Deliberately checks the things that would produce a plausible-looking report while
    being wrong: whether the three masks really hide the same number of tokens, whether
    the mosaic bank spans more than one class, whether the candidate re-quantisation
    changes the energy, whether the layer loop returns the same tokens the model's own
    forward does, and whether the collapse arm actually collapses.
    """
    import torch

    # ── occlude ────────────────────────────────────────────────────────────────
    sw = occlusion.sweep(tubelets, grid, size=8)
    st = occlusion.static(tubelets, grid, size=8)
    fu = occlusion.matched_future(tubelets, grid, share=float(sw.float().mean()))
    shares = {n: float(m.float().mean()) for n, m in
              (("sweep", sw), ("static", st), ("future", fu))}
    check("the three masks hide the same fraction",
          max(shares.values()) - min(shares.values()) < 0.005,
          " ".join(f"{k} {v:.0%}" for k, v in shares.items()))
    check("the sweeping hole actually moves",
          not torch.equal(sw[0], sw[-1]) and sw.sum() == st.sum())

    bank_names, bank_labels = occlusion.spread_over_classes(catalog, "train", labels, 5)
    check("the bank spans more than one class", len(set(bank_labels)) > 1,
          f"{len(set(bank_labels))} of {len(labels)} actions in 5 clips")
    check("the query clip is not in the bank", path not in bank_names)

    bank = occlusion.build_bank(model, processor, CLIPS_REPO, bank_names, FRAMES,
                                labels=bank_labels)
    W = decode.ridge_fit(bank["Xd"], bank["Y"])
    class_id = labels.index(next(lb for _, lb, p in catalog if p == path))
    res = occlusion.evaluate(model, pixel_values, shown, sw, bank, W, grid,
                             class_id=class_id)
    r = res["retrieval"]
    print(f"  sweep: cos {res['score']['cos']:.3f} (chance {res['floor']['cos']:.3f}) "
          f"agree-with-ceiling {r['agree']:.1%} (chance {r['agree_chance']:.5%}) "
          f"same-class {r['same_class_pred']:.1%} (bank share {r['same_class_chance']:.1%})")
    check("prediction beats its shuffled floor under the sweep mask",
          res["score"]["cos"] > res["floor"]["cos"])
    check("the mosaic agrees with its ceiling far above chance",
          r["agree"] > 100 * r["agree_chance"], f"{r['agree']:.1%}")
    check("same-class retrieval is a real number", not np.isnan(r["same_class_pred"]))

    # The fill must leave the visible context untouched, or the ceiling panel is
    # quietly showing a reconstruction of pixels the model could already see.
    truth = res["truth"]
    hole = occlusion.hole_pixels(sw, grid, truth.shape[:3])
    filled = res["frames"]["mosaic_pred"]
    check("fill leaves the visible context byte-identical",
          np.array_equal(filled[~hole], truth[~hole]))
    check("fill actually changed the hole", not np.array_equal(filled[hole], truth[hole]))

    row = viz.labelled_row([(viz.masked_video(truth, sw), [("SAW", viz._AMBER)]),
                            (filled, [("PREDICTED", viz._GREEN)]),
                            (res["frames"]["mosaic_true"], [("CEILING", viz._AMBER)])])
    mp4 = viz.encode_mp4(row, fps=8)
    report = viz.probe(mp4)
    check("mosaic video decodes and is not black", "black" in report
          and report.split("black")[0].strip().endswith("0"), report[:70])

    # ── energy ─────────────────────────────────────────────────────────────────
    ctx_ids, tgt_ids = jepa.ids_of(jepa.tube(tubelets, grid, blocks=2, size=8, seed=0))
    pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)
    truth_frames = shown[: tubelets * 2]

    back = clip_io.shown_pixels(processor, ebm.to_pixel_values(processor, truth_frames))
    check("to_pixel_values round trips to within one grey level",
          int(np.abs(back.astype(int) - truth_frames.astype(int)).max()) <= 1)

    e_true = ebm.candidate_energy(model, processor, pred, truth_frames, tgt_ids)["l1"]
    e_target = ebm.energy(pred, true)["l1"]  # the encoder's own tokens, no re-quantise
    print(f"  E(re-quantised truth) {e_true:.5f} vs E(encoder's own targets) "
          f"{e_target:.5f}, {abs(e_true - e_target) / e_target:.2%} apart")
    check("re-encoding the truth barely moves its energy",
          abs(e_true - e_target) / e_target < 0.02)

    truth_e = ebm.candidate_energy(model, processor, pred, truth_frames, tgt_ids)
    e_noise = ebm.candidate_energy(model, processor, pred, ebm.noise(truth_frames), tgt_ids)
    e_grey = ebm.candidate_energy(model, processor, pred, ebm.grey(truth_frames), tgt_ids)
    fl = ebm.floors(pred, true, seq, ctx_ids)
    for nm, e in (("truth", truth_e), ("flat grey", e_grey), ("noise", e_noise),
                  ("shuffled-pairing floor", fl["shuffled prediction (chance)"]),
                  ("context mean", fl["context mean (no model)"])):
        print(f"  E({nm:22s}) l1 {e['l1']:.4f}  centred cosdist {e['cosdist']:.4f}")

    # Under the CENTRED distance the energy behaves like an energy.
    check("noise has higher energy than the truth (centred)",
          e_noise["cosdist"] > truth_e["cosdist"])
    check("flat grey has higher energy than the truth (centred)",
          e_grey["cosdist"] > truth_e["cosdist"])
    check("the truth beats both no-model floors (centred)",
          truth_e["cosdist"] < min(fl["shuffled prediction (chance)"]["cosdist"],
                                   fl["context mean (no model)"]["cosdist"]))

    # Under the RAW L1 the model was trained with, it does not, and that disagreement
    # is the task's finding. A tripwire rather than a note: if a future checkpoint or
    # transformers release fixes it, the report's headline is wrong.
    check("raw L1 and the centred distance disagree about flat grey",
          (e_grey["l1"] < truth_e["l1"]) != (e_grey["cosdist"] < truth_e["cosdist"]),
          f"grey/truth is {e_grey['l1'] / truth_e['l1']:.2f}x in L1 and "
          f"{e_grey['cosdist'] / truth_e['cosdist']:.2f}x centred")

    span = min(3, tubelets // 2)
    w = ebm.well(model, processor, pred, truth_frames, tgt_ids,
                 list(range(-span, span + 1)), "time")
    print("  temporal well (l1 / centred): "
          + "  ".join(f"{k:+d}:{v['l1']:.4f}/{v['cosdist']:.4f}" for k, v in w))
    check("the centred temporal well bottoms out at zero offset",
          ebm.argmin_of(w, "cosdist") == 0,
          f"minimum at {ebm.argmin_of(w, 'cosdist'):+d}, depth "
          f"{ebm.depth_of(w, 'cosdist'):.1%}")
    check("the raw-L1 temporal well is flat, which is why both are plotted",
          ebm.depth_of(w, "l1") < ebm.depth_of(w, "cosdist") / 3,
          f"L1 depth {ebm.depth_of(w, 'l1'):.1%} vs centred "
          f"{ebm.depth_of(w, 'cosdist'):.1%}")
    # The rank correlation in the report is only a prediction if the ordering was
    # written down for every candidate the task actually scores.
    named = ebm.build_candidates(truth_frames, {"same action, different clip": truth_frames,
                                                "different action": truth_frames})
    missing = set(named) - set(ebm.EXPECTED_ORDER)
    check("EXPECTED_ORDER covers every candidate the task scores", not missing,
          f"missing {sorted(missing)}" if missing else f"{len(named)} candidates")
    check("every candidate is a distinct clip from the truth",
          sum(1 for k, v in named.items() if k != "true completion"
              and np.array_equal(v, truth_frames)) == 2,
          "only the two stand-ins passed in above may match")
    check("energy chart renders",
          viz.energy_chart({"true completion": {"l1": 1.0}, "uniform noise": {"l1": 2.0}},
                           {"chance": {"l1": 1.8}}).startswith("<img"))

    # ── ladder ─────────────────────────────────────────────────────────────────
    stack = layers.per_layer(model, pixel_values)
    check("per_layer returns one entry per layer plus the final LayerNorm",
          len(stack) == len(model.encoder.layer) + 1, f"{len(stack)} entries")
    check("per_layer tokens match the model's own forward",
          torch.allclose(stack[-1], seq, atol=2e-2),
          f"max abs diff {float((stack[-1] - seq).abs().max()):.4f}")
    check("the final LayerNorm is not a no-op",
          not torch.allclose(stack[-1], stack[-2], atol=1e-3))
    geom = [jepa.anisotropy(t) for t in stack]
    mid = geom[len(geom) // 3]["raw"]
    print(f"  anisotropy: layer {len(geom) // 3 + 1} raw {mid:.3f} -> "
          f"final raw {geom[-1]['raw']:.3f}")
    check("the token cone is tighter in the middle of the network than at the end",
          mid > geom[-1]["raw"])

    X, Y, g2, t2 = layers.collect_tokens(model, processor, CLIPS_REPO, bank_names[:2],
                                         FRAMES, keep=256)
    check("collect_tokens pairs every layer with the same pixels",
          all(len(x) == len(Y) for x in X) and len(X) == len(stack),
          f"{len(X)} layers x {len(Y)} tokens")
    # On the device the task uses: the ridge solve is the one place in this repo that
    # runs on CUDA tensors, and a CPU-only check would not have caught the bias column
    # being built on the wrong device.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ro, refs = layers.readout_curve(X, Y, val_frac=0.25, device=dev)
    print(f"  readout solve ran on {dev}")
    print(f"  readout: layer 1 {ro[0]['psnr']:.2f} dB, final {ro[-1]['psnr']:.2f} dB, "
          f"random-projection ceiling {refs['rand']['psnr']:.2f} dB, "
          f"shuffled floor {refs['shuf']['psnr']:.2f} dB")
    check("the readout ceiling inverts the random projection",
          refs["rand"]["r2"] > 0.9, f"R2 {refs['rand']['r2']:+.3f}")
    check("early layers keep more pixel information than the last one",
          ro[0]["r2"] > ro[-1]["r2"])
    check("twin chart renders",
          viz.twin_chart("t", "x", [1, 2], ("a", [0.1, 0.2]), ("b", [0.3, 0.4]),
                         refs={"r": 0.5}, marks={"m": 1}).startswith("<img"))

    # ── collapse ───────────────────────────────────────────────────────────────
    data = collapse_module.make_dataset(96, seed=0)
    v = collapse_module.to_tensor(data["video"][:2])
    patches = collapse_module.patchify(v)
    enc0 = collapse_module.Encoder()
    block = v[0, :, 0:2, 0:8, 8:16].permute(1, 2, 3, 0).reshape(-1)
    check("patchify uses the same t*G*G + h*G + w raster as the tokens",
          torch.allclose(block, patches[0, 1]))
    check("the tiny encoder's token count matches its own geometry",
          enc0(v).shape[1] == enc0.n_tokens == enc0.tubes * enc0.grid * enc0.grid)
    ctx, tgt = collapse_module.tube_mask(enc0.tubes, enc0.grid, 4)
    check("the tiny tube mask hides a quarter of the tokens",
          abs(len(tgt) / enc0.n_tokens - 0.25) < 0.01,
          f"{len(tgt)}/{enc0.n_tokens}")
    check("dropping masked tokens changes what the encoder sees",
          enc0(v, keep=ctx).shape[1] == len(ctx) < enc0.n_tokens)

    rep = collapse_module.to_tensor(np.repeat(data["video"][:1], 8, axis=0))
    with torch.no_grad():
        Zsame = enc0.pooled(rep.to(next(enc0.parameters()).device))
    sp = collapse_module.spread(Zsame)
    check("the collapse detectors fire on identical inputs",
          sp["pair_cos"] > 0.999 and sp["erank"] < 1.01, f"{sp}")

    # The generative factors must be independent, or a probe for one reads another.
    for a, b in (("shape", "colour"), ("shape", "direction"), ("colour", "direction")):
        r_ab = abs(viz.pearson(data[a].tolist(), data[b].tolist()))
        check(f"{a} and {b} are uncorrelated in the generated data", r_ab < 0.25,
              f"|r| = {r_ab:.2f}")

    tr = collapse_module.make_dataset(256, seed=0)
    ev = collapse_module.make_dataset(64, seed=1)
    finals = {}
    for kind in collapse_module.KINDS:
        out = collapse_module.train(kind, tr["video"], ev["video"], steps=150,
                                    log_every=149)
        finals[kind] = out["final"]
        print(f"  {kind:9s} loss {out['final']['loss']:.4f} "
              f"pair_cos {out['final']['pair_cos']:.3f} "
              f"std {out['final']['std']:.4f} in {out['seconds']:.0f}s")
        if kind == "pixels":
            recon = collapse_module.reconstruct(out, tr["video"][0])
            check("the pixel arm reconstructs something",
                  recon is not None and not np.array_equal(recon, tr["video"][0]))
            check("the latent arms have nothing to reconstruct",
                  collapse_module.reconstruct({"kind": "ema"}, tr["video"][0]) is None)

    # 150 steps is not enough to finish collapsing, so this is directional: the arm
    # with differentiable targets must already be heading for a lower loss and a
    # tighter feature cone than V-JEPA's recipe.
    check("the collapse arm has the lowest loss of the latent arms",
          finals["none"]["loss"] < min(finals["ema"]["loss"], finals["stopgrad"]["loss"]),
          f"none {finals['none']['loss']:.4f} vs ema {finals['ema']['loss']:.4f}")
    check("the collapse arm's features are the least spread out",
          finals["none"]["std"] < finals["ema"]["std"],
          f"none std {finals['none']['std']:.4f} vs ema {finals['ema']['std']:.4f}")
    check("factor chart renders",
          viz.factor_chart({"a": {"shape": {"acc": 0.5}}}, ["shape"],
                           {"shape": 0.33}).startswith("<img"))
    check("curve chart renders",
          viz.curve_chart("t", "x", "y", {"s": [(0, 1.0), (1, 0.5)]},
                          hlines={"h": 0.2}, vlines={"v": 0}).startswith("<img"))
    check("scatter chart renders",
          viz.scatter_chart("t", "x", "y", [1.0, 2.0], [0.5, 0.8]).startswith("<img"))


def check_ac(check) -> None:
    """The V-JEPA 2-AC half: sim, then model, then the two together.

    Ordered cheapest-first on purpose. The MuJoCo checks cost a second and catch the
    failures that would otherwise be blamed on the world model, which is the trap
    this file exists to avoid: an arm that under-reaches because of a controller bug
    reads exactly like a planner that cannot plan.
    """
    import ac
    import plan as planning
    import sim

    # ── the simulator ───────────────────────────────────────────────────────────
    env, goal_frame, goal_pos, start_frame = sim.reach_task(seed=0)
    check("scene builds and renders", start_frame.shape == (256, 256, 3), str(start_frame.shape))
    # A camera inside a wall renders flat brown with no error at all.
    check("render is not a flat colour", float(start_frame.std()) > 12.0,
          f"std {float(start_frame.std()):.1f}")
    check("goal frame differs from start",
          float(np.abs(start_frame.astype(int) - goal_frame.astype(int)).mean()) > 1.0)

    errs = []
    for d in ([0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05], [-0.03, 0.04, 0.02]):
        before = env.ee_pos.copy()
        env.step(np.array(d + [0, 0, 0, 0], dtype=np.float32))
        errs.append(float(np.linalg.norm((env.ee_pos - before) - np.array(d))))
    # The controller must deliver what the planner asks for. Solving IK once and
    # commanding it lands at 16 mm on a 50 mm request; the integral loop gets under 1 mm.
    check("cartesian control tracks", max(errs) < 0.005,
          f"worst {max(errs) * 1000:.1f} mm on a 50 mm command")
    env.close()

    # ── the model ───────────────────────────────────────────────────────────────
    wm = ac.ActionWorldModel()
    check("AC checkpoint loads completely",
          not wm.missing and not wm.unexpected,
          f"{len(wm.missing)} missing / {len(wm.unexpected)} unexpected")

    # Our six-line transform replaces an upstream one that imports cv2 for a resize.
    # Bit-exact or it is not a shortcut, it is a bug.
    try:
        import os

        sys.path.insert(0, os.environ.get("VJEPA2_SRC", "/opt/vjepa2"))
        from app.vjepa_droid.transforms import make_transforms

        ref = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                              random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                              motion_shift=False, crop_size=ac.CROP)
        frames = np.stack([start_frame, goal_frame])
        diff = float((ac.transform(frames) - ref(frames)).abs().max())
        check("transform is bit-exact vs upstream", diff == 0.0, f"max diff {diff:.2e}")
    except ImportError as e:
        print(f"  skip transform parity ({e}); needs opencv-python-headless")

    # ── the two together ────────────────────────────────────────────────────────
    env, goal_frame, goal_pos, start_frame = sim.reach_task(seed=0)
    z = wm.encode(start_frame[None])
    z_goal = wm.encode(goal_frame[None])
    check("encode shape", z.shape[1] == ac.TOKENS_PER_FRAME, str(tuple(z.shape)))

    # Does moving toward the goal lower the energy? If not, nothing downstream can
    # work, and this is far cheaper to discover here than inside a Flyte pod.
    e_start = float(wm.energy(z[:, -ac.TOKENS_PER_FRAME:], z_goal[:, -ac.TOKENS_PER_FRAME:]))
    env.move_to(env.ee_pos + (goal_pos - env.ee_pos) * 0.7)
    z_near = wm.encode(env.render(ac.CROP)[None])
    e_near = float(wm.energy(z_near[:, -ac.TOKENS_PER_FRAME:], z_goal[:, -ac.TOKENS_PER_FRAME:]))
    check("energy falls when the arm approaches the goal", e_near < e_start,
          f"{e_start:.4f} -> {e_near:.4f}")

    env.reset()
    import torch

    pose = torch.tensor(env.pose7(), device=wm.device, dtype=wm.dtype)[None, None]
    actions, trace = wm.plan(z, pose, z_goal, cfg=ac.CEMConfig(samples=16, cem_steps=3))
    check("planner returns a bounded action",
          actions.shape == (2, 7) and float(np.abs(actions[:, :3]).max()) <= 0.0501,
          f"|a| max {float(np.abs(actions[:, :3]).max()):.4f}")
    check("CEM lowers its own energy", trace.energy_best[-1] <= trace.energy_best[0],
          f"{trace.energy_best[0]:.4f} -> {trace.energy_best[-1]:.4f}")

    # The load-bearing claim of the whole `plan` task, in one cheap check: with the
    # dynamics supplied by the simulator rather than imagined, searching on the V-JEPA
    # energy alone moves the arm TOWARD a goal it is never given the coordinates of.
    # If this regresses, the reward has stopped working and nothing downstream means
    # anything.
    env.reset()
    grid = np.stack(np.meshgrid(*[np.linspace(-0.05, 0.05, 3)] * 3, indexing="ij"), -1)
    grid = grid.reshape(-1, 3).astype(np.float32)
    d_before = float(np.linalg.norm(env.ee_pos - goal_pos))
    for _ in range(4):
        e = planning._true_energies(wm, env, z_goal, grid)
        a = np.zeros(7, dtype=np.float32)
        a[:3] = grid[int(e.argmin())]
        env.step(a)
    d_after = float(np.linalg.norm(env.ee_pos - goal_pos))
    check("V-JEPA energy alone moves the arm toward the goal", d_after < d_before,
          f"{d_before * 100:.1f} -> {d_after * 100:.1f} cm in 4 steps")

    # Rendering paths used by the report.
    ep = planning.run_episode(wm, env, goal_frame, goal_pos, policy="oracle", steps=3,
                              render_size=256)
    mp4 = viz.encode_mp4(viz.episode_video(ep, goal_frame), fps=3)
    check("episode video encodes", len(mp4) > 1000, f"{len(mp4) / 1024:.0f} KiB")
    check("episode video is not a black rectangle", "static" not in viz.probe(mp4), viz.probe(mp4))
    check("progress chart renders", viz.progress_chart({"oracle": ep}).startswith("<img"))
    check("oracle actually reaches", ep.final_dist < ep.start_dist,
          f"{ep.start_dist * 100:.1f} -> {ep.final_dist * 100:.1f} cm")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the 100-clip probe")
    ap.add_argument("--ac", action="store_true", help="also run the V-JEPA 2-AC + MuJoCo checks")
    ap.add_argument("--mech", action="store_true",
                    help="also run the mechanism tasks: occlude, energy, ladder, collapse")
    args = ap.parse_args()
    if args.ac:
        fails: list[str] = []

        def _check(name: str, ok: bool, detail: str = "") -> None:
            print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
            if not ok:
                fails.append(name)

        print("V-JEPA 2-AC checks")
        check_ac(_check)
        print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'AC checks passed'}")
        sys.exit(1 if fails else 0)
    sys.exit(main(args.full, args.mech))
