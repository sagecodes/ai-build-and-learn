"""NVIDIA Cosmos 3 on Flyte: a world model in a pod, with the video in the report.

    flyte run pipeline.py imagine                      # text -> predicted world
    flyte run pipeline.py imagine --scene forklift
    flyte run pipeline.py imagine --sound           # video AND ambient sound
    flyte run pipeline.py rollout                      # actions -> predicted future
    flyte run pipeline.py compare                      # short vs structured prompt
    flyte run pipeline.py invert                       # video -> the actions behind it
    flyte run pipeline.py world_models                 # all of them, one run

Runs to the `world-models` project (.flyte/config.yaml), alongside topics/dreamerv3.
The pairing is the point of the event: Dreamer LEARNS a world model of one small
environment from its own experience, Cosmos 3 IS a pretrained world model of the
physical world that you condition and roll forward. The robotics demos that supply
the physics ground truth (topics/rl-mujoco, topics/isaac-sim) live in `physical-ai`.

── Why every task loads the pipeline itself, and only once ─────────────────────
There is no shared model cache across pods on this cluster, so each task pod pulls
the 35 GB snapshot into its own /tmp/hf. That download dominates the wall clock:
denoising 45 frames is a couple of minutes, fetching the weights is longer. So each
task loads ONE pipeline and generates everything it needs from it, and `compare`
exists as a single task rather than a fan-out for exactly that reason. Splitting it
into two tasks would double the download to parallelise the cheap part.

── Why the report gets painted before there is anything to show ────────────────
Same lesson as the DreamerV3 task next door. A pod that spends its first ten minutes
downloading and its next few loading a 16B transformer, while its report stays
blank, is indistinguishable from a hung pod. Each task repaints at every stage so
the report always says what it is doing.
"""

from __future__ import annotations

import logging

import flyte
import flyte.report

# Imported at top level so Flyte bundles these siblings into the pod. A deferred
# import inside a task body is exactly how you get ModuleNotFoundError in the pod
# while everything works on the host.
import media
import prompts
import reports
import world
from config import NANO, gpu_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _paint(stage: str, detail: str, rows: list[tuple[str, str]]) -> None:
    flyte.report.replace(reports.progress_html(stage, detail, rows), do_flush=True)


@gpu_env.task(report=True)
async def imagine(
    scene: str = "box-topple",
    repo: str = NANO,
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
    sound: bool = False,
) -> dict:
    """Generate a predicted world from text, and show it in the report.

    Defaults are 480p / 45 frames rather than NVIDIA's 720p / 189. The clip is going
    to be base64'd into an HTML report, so this is sized to embed rather than to
    impress; raise both for the quality shot.
    """
    rows = [("Model", repo), ("Scene", scene), ("Frames", f"{frames} at {width}x{height}")]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo, sound=sound)

    prompt = prompts.get(scene)
    _paint("Generating", f"{steps} denoising steps.", rows)
    result, secs = world.generate(
        pipe, prompt,
        negative_prompt=prompts.NEGATIVE,
        num_frames=frames, height=height, width=width,
        steps=steps, guidance=guidance, seed=seed, sound=sound,
    )

    # result.sound is None unless enable_sound was on. Passing it through is what
    # makes the --sound flag mean anything: without it the sound tokenizer is loaded,
    # the waveform is generated, and then discarded into a silent mp4.
    wav = getattr(result, "sound", None) if sound else None
    mp4 = media.encode(result.video, fps=24, sound=wav)
    probe = media.probe(mp4)
    log.info("clip: %s%s", probe, " (with audio)" if wav is not None else "")

    rows += [
        ("Denoise time", f"{secs / 60:.1f} min ({secs / steps:.1f}s/step)"),
        ("Clip", f"{len(mp4) / 1024:.0f} KB"),
        ("Audio", "48 kHz stereo, muxed into the clip" if wav is not None
                  else "off (--sound to predict ambient sound)"),
    ]
    caption = f"{frames} frames at {width}x{height}, seed {seed}"
    if wav is not None:
        caption += " - press play, the sound is generated too"
    body = reports.clip_block(
        f"Predicted world: {scene}",
        media.video_html(mp4, caption, sound=wav is not None),
        media.strip(result.video),
        probe,
        prompt,
    )
    flyte.report.replace(
        reports.final_html("Text to world", rows, body, reports.IMAGINE_EXPLAINER),
        do_flush=True,
    )
    return {
        "scene": scene, "seconds": round(secs, 1), "probe": probe,
        "kb": len(mp4) // 1024, "audio": wav is not None,
    }


@gpu_env.task(report=True)
async def rollout(
    repo: str = NANO,
    chunks: int = 2,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Action-conditioned rollout: one observed frame plus robot actions -> the future.

    Uses the action example that ships INSIDE the checkpoint rather than one we
    invent: an AgiBotWorld humanoid picking items in a supermarket, with four chunks
    of sixteen 29-dimensional actions and the matching first frame. Using NVIDIA's own
    asset means the action encoding is unambiguously correct, so if the rollout looks
    wrong it is the model or the box, not our action tensor.

    `chunks` is the interesting knob. Chunk 0 is conditioned on the real observed
    frame; every chunk after it is conditioned on the last frame the model itself
    predicted, so raising this is how you watch prediction error compound.
    """
    rows = [("Model", repo), ("Task", "forward dynamics (action-conditioned)")]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({meta['chunks'].shape[-1]}-D actions)"),
        ("Prompt", meta["prompt"]),
        ("Chunks", f"{chunks} x {meta['action_chunk_size']} actions"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    _paint("Rolling forward", "Each chunk is conditioned on the previous chunk's last frame.", rows)
    frames, per_chunk = world.rollout(
        pipe, meta, num_chunks=chunks, steps=steps, guidance=guidance, seed=seed
    )

    fps = int(meta.get("fps", 10))
    mp4 = media.encode(frames, fps=fps)
    probe = media.probe(mp4)
    total = sum(per_chunk)
    log.info("rollout: %s frames, %s", len(frames), probe)

    rows += [
        ("Rollout time", f"{total / 60:.1f} min ({', '.join(f'{s:.0f}s' for s in per_chunk)})"),
        ("Frames", f"{len(frames)} at {fps} fps"),
    ]
    body = reports.clip_block(
        "Predicted future, driven by actions",
        media.video_html(mp4, f"{len(frames)} frames, {chunks} chunk(s) autoregressive"),
        media.strip(frames, count=8),
        probe,
        meta["prompt"],
    )
    body = reports.side_by_side([
        ("Observed: the one frame the model was given", media.image_html(meta["first_frame"])),
        ("Predicted: everything after it", body),
    ])
    flyte.report.replace(
        reports.final_html("Actions to future", rows, body, reports.ROLLOUT_EXPLAINER),
        do_flush=True,
    )
    return {
        "embodiment": meta["domain_name"],
        "chunks": chunks,
        "frames": len(frames),
        "seconds": round(total, 1),
        "probe": probe,
    }


@gpu_env.task(report=True)
async def compare(
    scene: str = "box-topple",
    repo: str = NANO,
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Same scene and seed, prompted twice: one sentence vs the structured caption.

    Cosmos 3 was trained on long structured JSON captions, and NVIDIA's guidance is
    to upsample a short prompt into that form with an LLM before generating. This
    task is the measurement of what that step is worth, and it costs one extra
    generation rather than one extra 35 GB download because both clips come from the
    same loaded pipeline.
    """
    rows = [("Model", repo), ("Scene", scene), ("Seed", str(seed))]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    cells, timings = [], {}
    for style, label in (("short", "One sentence"), ("structured", "Structured caption")):
        prompt = prompts.get(scene, style)
        _paint("Generating", f"{label.lower()}, {steps} denoising steps.", rows)
        result, secs = world.generate(
            pipe, prompt,
            negative_prompt=prompts.NEGATIVE,
            num_frames=frames, height=height, width=width,
            steps=steps, guidance=guidance, seed=seed,
        )
        mp4 = media.encode(result.video, fps=24)
        timings[style] = round(secs, 1)
        cells.append((
            label,
            media.video_html(mp4, f"{len(mp4) / 1024:.0f} KB", max_width=440)
            + media.strip(result.video, count=4, width=104)
            + reports.note(media.probe(mp4))
            + reports.details("prompt", prompt),
        ))
        log.info("%s: %.1fs", style, secs)

    rows.append(("Denoise time", ", ".join(f"{k} {v}s" for k, v in timings.items())))
    flyte.report.replace(
        reports.final_html(
            "Prompt shape", rows, reports.side_by_side(cells), reports.COMPARE_EXPLAINER
        ),
        do_flush=True,
    )
    return {"scene": scene, "seconds": timings}


@gpu_env.task(report=True)
async def invert(
    repo: str = NANO,
    example: int = 0,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Inverse dynamics: give it the video, ask what actions produced it.

    The one task here whose output is a measurement rather than a clip. NVIDIA ships
    the answer key inside the checkpoint: two 61-frame driving clips, each paired with
    the 60 nine-dimensional actions that connect their frames. So this run recovers
    actions from pixels and scores them against the truth, and the report shows both
    lines on the same axes.

    Worth saying on the stream: a forward-only world model cannot do this at all.
    DreamerV3's RSSM is trained as p(next state | state, action) and has no inverse,
    so it can dream a future but can never watch footage and say what was done. Cosmos
    denoises the action channel alongside the pixels, so the inverse is a mode flag.

    `example` picks which of the two bundled clips to run (0 or 1).
    """
    rows = [("Model", repo), ("Task", "inverse dynamics (video -> actions)")]
    _paint("Fetching weights", f"{repo} is ~35 GB and lands in this pod's /tmp/hf.", rows)

    path = world.snapshot(repo)
    meta = world.load_inverse_example(path, index=example)
    truth = meta["truth"]
    rows += [
        ("Clip", f"bundled AV example {example}, {len(meta['frames'])} frames at {meta['fps']:.0f} fps"),
        ("Embodiment", f"{meta['domain_name']} ({truth.shape[-1]}-D actions)"),
        ("To recover", f"{truth.shape[0]} action steps"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    _paint("Running it backwards", "Denoising the action channel from the video.", rows)
    pred, secs = world.invert(pipe, meta, steps=steps, guidance=guidance, seed=seed)
    score = world.action_error(truth, pred)

    # The clip the model watched, played back in the report. Re-encoded from the
    # decoded frames rather than shipped through as the original file, so what plays
    # is provably the same list of frames that was handed to the pipeline.
    mp4 = media.encode(meta["frames"], fps=int(meta["fps"]))
    probe = media.probe(mp4)

    # Channels ranked by how much they actually move. A rotation basis pinned near 1.0
    # is trivially easy to predict and would pad the chart with flat lines.
    spread = [hi - lo for lo, hi in score["truth_range"]]
    busiest = sorted(range(len(spread)), key=lambda d: spread[d], reverse=True)[:4]
    lead = busiest[0]

    # Deliberately NOT reporting the worst channel across all nine. In these clips one
    # channel carries almost all the motion (~0.41 of range) and the other eight barely
    # leave their start value (~0.02), so a range-normalised error on those eight is
    # noise divided by nothing: it reads as catastrophic however good the model is.
    # The honest headline is the channel that actually moves.
    rows += [
        ("Inverse time", f"{secs / 60:.1f} min ({secs / steps:.1f}s/step)"),
        ("Mean abs error", f"{score['mae_moving']:.4f} over {score['steps']} steps, "
                           f"{len(score['moving_dims'])} moving channel(s)"),
        ("Leading channel", f"ch {lead}: {score['nmae_per_dim'][lead] * 100:.1f}% "
                            f"of its {spread[lead]:.3f} range"),
        ("Static channels", f"{sum(1 for s in spread if s < spread[lead] / 10)} of "
                            f"{len(spread)} barely move in this clip"),
    ]

    body = reports.side_by_side([
        ("What the model watched", reports.clip_block(
            "Input clip, no actions supplied",
            media.video_html(mp4, f"{len(meta['frames'])} frames"),
            media.strip(meta["frames"], count=8),
            probe,
            meta["description"],
        )),
        ("What it read off the pixels", reports.action_traces(
            truth.tolist(), pred.tolist(),
            labels=[f"channel {i}" for i in range(truth.shape[-1])],
            dims=busiest,
            caption=(
                "The four channels that move most, since a channel that barely "
                "changes is trivially easy to guess. Solid is truth, dashed is "
                "recovered."
            ),
        )),
    ])
    flyte.report.replace(
        reports.final_html("Video to actions", rows, body, reports.INVERT_EXPLAINER),
        do_flush=True,
    )
    log.info("inverse: mae=%.4f over %s steps", score["mae"], score["steps"])
    return {
        "example": example,
        "embodiment": meta["domain_name"],
        "steps_recovered": score["steps"],
        "mae": round(score["mae"], 5),
        "mae_moving": round(score["mae_moving"], 5),
        "moving_dims": score["moving_dims"],
        "lead_channel": lead,
        "lead_nmae": round(score["nmae_per_dim"][lead], 5),
        "nmae_per_dim": [round(x, 5) for x in score["nmae_per_dim"]],
        "seconds": round(secs, 1),
    }


@orch_env.task(report=True)
async def world_models(scene: str = "box-topple", repo: str = NANO) -> dict:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU children.

    The children run in SEQUENCE, not in parallel, and that is not a style choice:
    there is one GPU on this box, so a second GPU task would sit Unschedulable on
    "Insufficient nvidia.com/gpu" until the first finished anyway.
    """
    generated = await imagine(scene=scene, repo=repo)
    predicted = await rollout(repo=repo)
    prompted = await compare(scene=scene, repo=repo)
    recovered = await invert(repo=repo)
    result = {
        "imagine": generated,
        "rollout": predicted,
        "compare": prompted,
        "invert": recovered,
    }
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(world_models))
