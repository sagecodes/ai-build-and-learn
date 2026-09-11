"""Prove Cosmos 3 works on this host, without Flyte in the way.

    ./.venv/bin/python smoke_test.py              # one short 480p clip
    ./.venv/bin/python smoke_test.py --image      # 1-frame text-to-image, ~30s
    ./.venv/bin/python smoke_test.py --action     # one action-conditioned chunk
    ./.venv/bin/python smoke_test.py --reason     # the UNDERSTANDING surface + handover
    ./.venv/bin/python smoke_test.py --all        # every surface, one model load

This is the control number that the containerised run from pipeline.py gets compared
against, the same role smoke_test.py plays in topics/isaac-sim. If this passes and
the Flyte run does not, the problem is the pod, not the model.

Writes into out/ and prints a luminance probe of what it produced, because a black
clip is a valid mp4 and "it ran without an error" is not the same as "it rendered".
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import media
import prompts
import world

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/Cosmos3-Nano")
    ap.add_argument("--scene", default="box-topple", choices=sorted(prompts.SCENES))
    ap.add_argument("--image", action="store_true", help="1 frame (text-to-image)")
    ap.add_argument("--action", action="store_true", help="action-conditioned rollout")
    ap.add_argument("--reason", action="store_true",
                    help="the understanding surface, and the handover to generation")
    ap.add_argument("--all", action="store_true",
                    help="exercise every surface once, from a single model load")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    # Before the generation load, not after: this mode's whole point is that the two
    # experts are loaded one at a time, and loading the 30 GB one first would make the
    # handover it is checking impossible to reach.
    if args.reason:
        _exercise_reasoning(args, outdir)
        return

    print(f"guard: {world.guard_memory()}")
    t0 = time.monotonic()
    pipe = world.load(args.repo)
    print(f"loaded in {(time.monotonic() - t0) / 60:.1f} min")

    if args.all:
        _exercise_everything(pipe, args, outdir)
        return

    if args.action:
        meta = world.load_action_example(world.snapshot(args.repo))
        frames, per_chunk = world.rollout(pipe, meta, num_chunks=1, steps=args.steps)
        mp4 = media.encode(frames, fps=int(meta.get("fps", 10)))
        (outdir / "action.mp4").write_bytes(mp4)
        print(f"action rollout: {sum(per_chunk):.1f}s -> out/action.mp4")
        print(f"probe: {media.probe(mp4)}")
        return

    prompt = prompts.get(args.scene)
    result, secs = world.generate(
        pipe,
        prompt,
        negative_prompt=prompts.NEGATIVE,
        num_frames=1 if args.image else args.frames,
        height=480,
        width=832,
        steps=args.steps,
    )

    if args.image:
        result.video[0].save(outdir / "sample.jpg", format="JPEG", quality=90)
        print(f"text-to-image: {secs:.1f}s -> out/sample.jpg")
        return

    mp4 = media.encode(result.video, fps=24)
    (outdir / "sample.mp4").write_bytes(mp4)
    print(f"text-to-video: {secs:.1f}s ({secs / args.steps:.1f}s/step) -> out/sample.mp4")
    print(f"probe: {media.probe(mp4)}")


def _exercise_everything(pipe, args, outdir) -> None:
    """Call into every generation surface once, at the smallest useful settings.

    The point is coverage, not quality: catching a wrong kwarg or a wrong shape here
    costs a couple of minutes, and catching it in a pod costs a model load plus the
    round trip. Each check is isolated so one broken surface does not hide the others,
    which matters because these fail independently and for unrelated reasons.
    """
    import traceback

    meta = world.load_action_example(world.snapshot(args.repo))
    results: dict[str, str] = {}

    def check(name, fn):
        t0 = time.monotonic()
        try:
            results[name] = f"OK  ({time.monotonic() - t0:5.0f}s) {fn()}"
        except Exception as exc:  # noqa: BLE001
            results[name] = f"FAIL {type(exc).__name__}: {exc}"
            traceback.print_exc()
        print(f"[{name}] {results[name]}", flush=True)

    def t2v():
        result, secs = world.generate(
            pipe, prompts.get(args.scene), negative_prompt=prompts.NEGATIVE,
            num_frames=args.frames, height=480, width=832, steps=args.steps,
        )
        mp4 = media.encode(result.video, fps=24)
        (outdir / "all_t2v.mp4").write_bytes(mp4)
        return media.probe(mp4)

    def forward():
        frames, _ = world.rollout_chunk(pipe, meta, meta["chunks"][0], steps=args.steps)
        (outdir / "all_forward.mp4").write_bytes(media.encode(frames, fps=10))
        return f"{len(frames)} frames"

    def counterfact():
        variants = world.counterfactuals(meta["chunks"][0])
        base, _ = world.rollout_chunk(pipe, meta, variants[0][1], steps=args.steps, seed=0)
        held, _ = world.rollout_chunk(pipe, meta, variants[1][1], steps=args.steps, seed=0)
        return f"held vs recorded divergence {world.frame_divergence(base, held):.2f}"

    def policy():
        frames, pred, _ = world.policy(pipe, meta, steps=args.steps)
        score = world.action_error(meta["chunks"][0], pred)
        (outdir / "all_policy.mp4").write_bytes(media.encode(frames, fps=10))
        return f"actions {tuple(pred.shape)}, mae {score['mae']:.4f}"

    def trajectory():
        stages, final, _ = world.generate_trajectory(
            pipe, prompts.get(args.scene), snapshots=3, negative_prompt=prompts.NEGATIVE,
            num_frames=args.frames, height=256, width=448, steps=args.steps,
        )
        sharp = [round(world.clip_stats(f)["sharpness"]) for _, f in stages]
        return f"{len(stages)} snapshots, sharpness {sharp}"

    def extend():
        result, _ = world.generate(
            pipe, prompts.get("dashcam"), negative_prompt=prompts.NEGATIVE,
            num_frames=args.frames, height=256, width=448, steps=args.steps,
        )
        base = list(result.video)
        seg, _ = world.extend(
            pipe, base, prompts.get("dashcam"), negative_prompt=prompts.NEGATIVE,
            num_frames=args.frames, height=256, width=448, steps=args.steps, seed=1,
        )
        full = world.stitch([base, seg])
        (outdir / "all_extend.mp4").write_bytes(media.encode(full, fps=24))
        return f"{len(base)} + {len(seg)} -> {len(full)} stitched"

    for name, fn in (
        ("text-to-video", t2v),
        ("forward-dynamics", forward),
        ("counterfactual", counterfact),
        ("policy", policy),
        ("trajectory", trajectory),
        ("extend", extend),
    ):
        check(name, fn)

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k:18s} {v}")
    failed = [k for k, v in results.items() if v.startswith("FAIL")]
    print("FAILURES:", failed or "none")
    raise SystemExit(1 if failed else 0)


def _exercise_reasoning(args, outdir) -> None:
    """The understanding surface, then the handover to the generation surface.

    Two claims to prove here, and the second is the one that bites. The first is that
    `Cosmos3OmniForConditionalGeneration` loads at all from a checkpoint that was
    staged for diffusers -- it does, because model.safetensors.index.json and
    model_index.json point at the same shards, so this costs no extra download.

    The second is that the two experts can be used in the same process one after the
    other. They cannot be used at the same time: 30 GB plus 17.5 GB plus a VAE decode
    does not fit in the GB10's single 119.7 GiB pool, and an oversized load on this box
    hangs the machine rather than raising. So this loads the understanding expert,
    releases it, and then loads the generation expert, printing what was still held
    across the handover. That number failing to come back down is the failure mode, and
    it is much cheaper to find here than eight minutes into a pod.
    """
    path = world.snapshot(args.repo)
    example = world.load_reasoning_example(path)

    t0 = time.monotonic()
    model, processor = world.load_reasoner(args.repo)
    print(f"understanding surface loaded in {(time.monotonic() - t0) / 60:.1f} min")

    answer, secs = world.ask(model, processor, example["prompt"], image=example["image"])
    print(f"\nplan ({secs:.1f}s): {answer}")
    subtasks = world.split_subtasks(answer)
    print(f"parsed {len(subtasks)} subtask(s): {subtasks}")

    # Ask it about a video too, since that is what `judge` and `blind` depend on and it
    # exercises a different tower (the vision encoder over N frames, not one image).
    clip = pathlib.Path(path) / world.INVERSE_CLIP.format(i=0)
    if clip.exists():
        frames = media.decode(str(clip))
        described, secs = world.ask(model, processor,
                                    "Describe what is happening in this video in one sentence.",
                                    video=frames, max_new_tokens=128)
        print(f"\nvideo ({len(frames)} frames -> {world.REASON_FRAMES}, {secs:.1f}s): {described}")

    model, processor = None, None
    still = world.release()
    print(f"\nhandover: {still:.1f} GiB still allocated after releasing the reasoner")

    print(f"guard: {world.guard_memory()}")
    t0 = time.monotonic()
    pipe = world.load(args.repo)
    print(f"generation surface loaded in {(time.monotonic() - t0) / 60:.1f} min")

    prompt = subtasks[0] if subtasks else prompts.get(args.scene)
    result, secs = world.generate(
        pipe, prompt, image=example["image"], negative_prompt=prompts.NEGATIVE,
        num_frames=args.frames, steps=args.steps,
    )
    mp4 = media.encode(result.video, fps=24)
    (outdir / "reason.mp4").write_bytes(mp4)
    print(f"\nimagined {prompt!r} in {secs:.1f}s -> out/reason.mp4")
    print(media.probe(mp4))


if __name__ == "__main__":
    main()
