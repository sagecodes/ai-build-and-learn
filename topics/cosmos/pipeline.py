"""NVIDIA Cosmos 3 on Flyte: a world model in a pod, with the video in the report.

    flyte run pipeline.py imagine        # text -> predicted world
    flyte run pipeline.py imagine --scene forklift
    flyte run pipeline.py imagine --sound     # video AND ambient sound
    flyte run pipeline.py rollout        # actions -> predicted future
    flyte run pipeline.py compare        # short vs structured prompt
    flyte run pipeline.py invert         # video -> the actions behind it
    flyte run pipeline.py counterfact    # same frame, four action sequences
    flyte run pipeline.py policy         # frame + goal -> actions AND video
    flyte run pipeline.py emerge         # the clip decoded partway through
    flyte run pipeline.py extend         # a clip, then continuations of it
    flyte run pipeline.py horizon        # the long one; hours, renders as it goes
    flyte run pipeline.py plan           # it decomposes a goal, then imagines the steps
    flyte run pipeline.py judge          # it watches its own rollout and scores it
    flyte run pipeline.py blind          # it rules on the counterfactuals, unlabelled
    flyte run pipeline.py cycle          # actions -> video -> actions, scored on truth
    flyte run pipeline.py dream          # generate novel behaviours, critique, label
    flyte run pipeline.py choose         # imagine four futures, pick one, twice
    flyte run pipeline.py world_models   # the short generation ones, one run
    flyte run pipeline.py reasoning      # the three understanding ones, one run
    flyte run pipeline.py data_engine    # cycle, choose, dream, one run
    flyte run pipeline.py overnight      # short ones then horizon, unattended

The generation tasks are four questions about the same checkpoint. `imagine`, `compare`
and `emerge` ask what it can generate. `rollout`, `policy` and `invert` ask what it
knows about actions, one per direction: actions to video, goal to actions, video to
actions. `counterfact` is the control that checks the action channel is doing anything
at all. `extend` and `horizon` ask the only question that separates a world model from
a good video model, which is what happens when you refuse to stop.

`plan`, `judge` and `blind` use the checkpoint's OTHER surface. Cosmos 3 is omnimodal:
the same shards that diffusers loads as `Cosmos3OmniPipeline` also load in transformers
as `Cosmos3OmniForConditionalGeneration`, a vision-language model that answers questions
about images and video. On a box where the shared cache is already staged that surface
costs no extra download, and it turns the model into an instrument for measuring itself
-- which is what all three of those tasks do.

`cycle`, `dream` and `choose` put the two surfaces to work together, on the question the
"world model as synthetic data" pitch actually turns on: can you train on what it dreams?
They are stages 3 to 5 of NVIDIA's own GR00T-Dreams pipeline (generate, critique with a
video critic, label with inverse dynamics), plus the measurement that says whether the
output is worth anything, plus the planning loop that is the closest analogue in here to
what topics/dreamerv3 does with its world model. See the README for the mapping.

Runs to the `world-models` project (.flyte/config.yaml), alongside topics/dreamerv3.
The pairing is the point of the event: Dreamer LEARNS a world model of one small
environment from its own experience, Cosmos 3 IS a pretrained world model of the
physical world that you condition and roll forward. The robotics demos that supply
the physics ground truth (topics/rl-mujoco, topics/isaac-sim) live in `physical-ai`.

── Why every task loads the pipeline itself, and only once ─────────────────────
The 33 GB checkpoint is staged into the devbox once and mounted into every pod (see
the shared-cache section of config.py), so a task now resolves the snapshot in
milliseconds instead of downloading it. What is left is a 2.6 minute load of a 16B
transformer onto the device, and that is still the fixed cost of starting any task
here. So each task loads ONE pipeline and generates everything it needs from it, and
`compare` and `counterfact` are single tasks rather than fan-outs for that reason:
splitting them would pay the load again to parallelise work that only takes minutes,
on a box with exactly one GPU to parallelise onto.

── Why the report gets painted before there is anything to show ────────────────
Same lesson as the DreamerV3 task next door. A pod that spends its first ten minutes
downloading and its next few loading a 16B transformer, while its report stays
blank, is indistinguishable from a hung pod. Each task repaints at every stage so
the report always says what it is doing.
"""

from __future__ import annotations

import logging
import time

import flyte
import flyte.report

# Imported at top level so Flyte bundles these siblings into the pod. A deferred
# import inside a task body is exactly how you get ModuleNotFoundError in the pod
# while everything works on the host.
import bc
import media
import mjc
import prompts
import reports
import world
from config import EDGE, NANO, gpu_env, orch_env

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
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

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
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

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
    generation rather than one extra model load because both clips come from the
    same loaded pipeline.
    """
    rows = [("Model", repo), ("Scene", scene), ("Seed", str(seed))]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

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
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

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


@gpu_env.task(report=True)
async def counterfact(
    repo: str = NANO,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """The control experiment: same frame, same seed, four different action sequences.

    Everything else here takes it on faith that Cosmos is conditioning on the actions
    it is handed. This is the task that checks. A video model that has learned to
    ignore its action channel and simply continue the scene plausibly would produce
    four near-identical clips; a world model produces four different futures, and the
    differences should be the ones the actions describe.

    The variants are built in `world.counterfactuals`, all anchored at the recorded
    first action so none of them leave the data manifold. `held` is the one to watch:
    it commands the starting pose for every step of the chunk, so a model that is
    really reading the actions has to predict a robot that stops.

    Four generations from one loaded pipeline, which is the only reason this is
    affordable at all.
    """
    rows = [("Model", repo), ("Task", "action counterfactuals")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    chunk = meta["chunks"][0]
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({chunk.shape[-1]}-D actions)"),
        ("Prompt", meta["prompt"]),
        ("Variants", f"4 x {chunk.shape[0]} actions, identical seed and conditioning frame"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    variants = world.counterfactuals(chunk)
    clips: dict[str, list] = {}
    cells: list[tuple[str, str]] = []
    total = 0.0

    for i, (name, actions, why) in enumerate(variants):
        _paint("Generating", f"variant {i + 1} of {len(variants)}: {name}", rows)
        frames, secs = world.rollout_chunk(
            pipe, meta, actions, steps=steps, guidance=guidance, seed=seed
        )
        clips[name] = frames
        total += secs
        mp4 = media.encode(frames, fps=int(meta.get("fps", 10)))
        stats = world.clip_stats(frames)
        cells.append((
            name,
            media.video_html(mp4, f"{len(frames)} frames, {secs:.0f}s", max_width=380)
            + media.strip(frames, count=4, width=92)
            + reports.note(why)
            + reports.note(
                f"motion {stats['motion']:.2f}, sharpness {stats['sharpness']:.0f}"
            ),
        ))
        log.info("counterfactual %s: %.1fs, %s", name, secs, media.probe(mp4))

    # Every variant against the recorded rollout. The recorded one scores 0 against
    # itself by construction and is kept in the chart as the zero line, because a
    # reader needs to see that the baseline is there rather than assume it.
    baseline = clips["recorded"]
    divergence = [(name, world.frame_divergence(baseline, frames)) for name, frames in clips.items()]
    spread = max(v for _, v in divergence)

    rows += [
        ("Generation time", f"{total / 60:.1f} min for {len(variants)} rollouts"),
        ("Largest divergence", f"{spread:.2f} grey levels from the recorded rollout"),
    ]

    body = reports.side_by_side(cells) + reports._heading("How far each variant moved")
    body += reports.bars(
        divergence,
        caption=(
            "Mean absolute pixel difference from the recorded rollout, in grey levels "
            "out of 255. The number that matters is not any single bar but the spread: "
            "if changing the actions moves the pixels, the model is conditioning on "
            "them. Bars near zero for every variant would mean the action channel is "
            "decoration and the model is just continuing the scene."
        ),
    )
    flyte.report.replace(
        reports.final_html(
            "Actions are the input, not decoration", rows, body, reports.COUNTERFACT_EXPLAINER
        ),
        do_flush=True,
    )
    return {
        "variants": [n for n, _, _ in variants],
        "divergence": {n: round(v, 3) for n, v in divergence},
        "spread": round(spread, 3),
        "seconds": round(total, 1),
    }


@gpu_env.task(report=True)
async def policy(
    repo: str = NANO,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Policy mode: give it a frame and a goal, it decides the actions AND renders them.

    The third action mode, and the one that makes the "world model or policy?" question
    stop making sense. Forward dynamics is given actions and predicts pixels. Inverse
    dynamics is given pixels and predicts actions. This is given NEITHER: one frame,
    the task description, and both channels left noisy, so the model denoises a plan
    and its consequences at the same time.

    The scoring is against the recorded human demonstration, and it deserves a caveat
    said out loud on the stream: there is no single correct action sequence for
    "pick up items in the supermarket". A low error means the model chose roughly what
    the demonstrator chose; a high one means it chose something else, which may still
    be a perfectly good way to do the task. Read the video next to the number.
    """
    rows = [("Model", repo), ("Task", "policy (frame + goal -> actions and video)")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    truth = meta["chunks"][0]
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({truth.shape[-1]}-D actions)"),
        ("Goal", meta["prompt"]),
        ("Given", "one frame, and nothing about what to do next"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    _paint("Deciding", "Denoising the action channel and the pixels together.", rows)
    frames, pred, secs = world.policy(pipe, meta, steps=steps, guidance=guidance, seed=seed)
    score = world.action_error(truth, pred)

    fps = int(meta.get("fps", 10))
    mp4 = media.encode(frames, fps=fps)
    probe = media.probe(mp4)

    spread = [hi - lo for lo, hi in score["truth_range"]]
    busiest = sorted(range(len(spread)), key=lambda d: spread[d], reverse=True)[:4]
    rows += [
        ("Decision time", f"{secs / 60:.1f} min ({secs / steps:.1f}s/step)"),
        ("Actions chosen", f"{tuple(pred.shape)}"),
        ("Agreement with the demo", f"MAE {score['mae_moving']:.4f} over "
                                    f"{len(score['moving_dims'])} moving channel(s)"),
    ]

    body = reports.side_by_side([
        ("The frame it was given", media.image_html(meta["first_frame"])),
        ("What it decided to do", reports.clip_block(
            "Predicted plan, rendered",
            media.video_html(mp4, f"{len(frames)} frames at {fps} fps"),
            media.strip(frames, count=8),
            probe,
            meta["prompt"],
        )),
    ])
    body += reports._heading("Its actions against the human demonstration")
    body += reports.action_traces(
        truth.tolist(), pred.tolist(),
        labels=[f"channel {i}" for i in range(truth.shape[-1])],
        dims=busiest,
        caption=(
            "Solid is what the human demonstrator did, dashed is what the model chose "
            "to do from the same starting frame. These are not expected to coincide: "
            "the task admits many valid executions, and the model was never told which "
            "one was recorded. Divergence here is a different claim from divergence in "
            "the inverse-dynamics task, where there genuinely is one right answer."
        ),
    )
    flyte.report.replace(
        reports.final_html("Goal to plan", rows, body, reports.POLICY_EXPLAINER),
        do_flush=True,
    )
    return {
        "embodiment": meta["domain_name"],
        "frames": len(frames),
        "action_shape": list(pred.shape),
        "mae": round(score["mae"], 5),
        "mae_moving": round(score["mae_moving"], 5),
        "seconds": round(secs, 1),
        "probe": probe,
    }

@gpu_env.task(report=True)
async def emerge(
    scene: str = "box-topple",
    repo: str = NANO,
    snapshots: int = 8,
    show: str = "x0",
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """The same generation, decoded partway through, so you can watch it resolve.

    Every other task here shows the finished clip. This one opens the denoiser up:
    `callback_on_step_end` hands back the latents after each scheduler step, and
    decoding a handful of them turns 35 invisible steps into something you can watch.

    WHAT gets decoded is the entire methodology, and getting it wrong produces a
    confident demo of nothing. `show="x0"` (the default) decodes the model's current
    prediction of the finished video. `show="latent"` decodes the noisy sample the
    solver is actually holding, and on this checkpoint that is visually static for
    thirty of thirty five steps with a flat sharpness around 130 until it jumps to 259
    at the last step, because a flow-matching schedule keeps x_t noise-dominated until
    the end. Both are honest pictures of a real quantity; only the first is the thing
    people mean when they say they want to watch the video emerge. `world.
    generate_trajectory` explains how the x0 prediction is captured.

    Each snapshot costs a full VAE decode, which is the largest allocation in the run
    (10 to 14s for 45 frames at 480p), so this samples `snapshots` steps rather than
    all of them.
    """
    rows = [
        ("Model", repo),
        ("Scene", scene),
        ("Schedule", f"{steps} steps, decoded at {snapshots} of them"),
        ("Decoding", "the model's prediction of the finished clip (x0)" if show == "x0"
                     else "the raw noisy sample (x_t)"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    prompt = prompts.get(scene)
    _paint("Generating", f"{steps} steps, decoding {snapshots} of them on the way.", rows)
    stages, final, secs = world.generate_trajectory(
        pipe, prompt,
        snapshots=snapshots, show=show,
        negative_prompt=prompts.NEGATIVE,
        num_frames=frames, height=height, width=width,
        steps=steps, guidance=guidance, seed=seed,
    )

    # One video, the stages played back to back with the step number burned in. The
    # label is not decoration: without it the clip is a sequence of similar-looking
    # scenes and there is no way to tell which end is which.
    reel: list = []
    for step, stage_frames in stages:
        reel.extend(media.label_frames(stage_frames, f"step {step}/{steps}"))
    reel_mp4 = media.encode(reel, fps=24)
    final_mp4 = media.encode(final, fps=24)

    # The same frame index at each step, side by side. This is the view that makes the
    # "settles early, refines late" point legible in one glance.
    mid = len(final) // 2
    progression = "".join(
        media.image_html(stage_frames[mid], f"step {step}", width=200)
        for step, stage_frames in stages
    )

    stage_stats = [world.clip_stats(f) for _, f in stages]
    step_numbers = [n for n, _ in stages]

    def settles_at(values: list[float], frac: float) -> int:
        """First sampled step at which a metric reaches `frac` of its final value.

        The measured version of "settles early, refines late". Quoting the first and
        last value alone cannot distinguish a metric that climbs steadily to the end
        from one that is done a third of the way in, and those are completely
        different claims about where the compute goes.
        """
        final = abs(values[-1]) or 1.0
        for step, value in zip(step_numbers, values):
            if abs(value) >= frac * final:
                return step
        return step_numbers[-1]

    motion_at = settles_at([s["motion"] for s in stage_stats], 0.95)
    sharp_at = settles_at([s["sharpness"] for s in stage_stats], 0.90)
    rows += [
        ("Generation time", f"{secs / 60:.1f} min including {len(stages)} decodes"),
        ("Snapshots", ", ".join(f"step {s}" for s, _ in stages)),
        ("Sharpness", f"{stage_stats[0]['sharpness']:.0f} at the first snapshot to "
                      f"{stage_stats[-1]['sharpness']:.0f} at the last"),
        ("Motion settled by", f"step {motion_at} of {steps} "
                              f"({motion_at / steps * 100:.0f}% of the schedule) at 95% "
                              f"of its final value"),
        ("Detail settled by", f"step {sharp_at} of {steps} "
                              f"({sharp_at / steps * 100:.0f}% of the schedule) at 90% "
                              f"of its final value"),
    ]

    body = reports.clip_block(
        "The whole schedule, played end to end",
        media.video_html(reel_mp4, f"{len(stages)} snapshots, {len(reel)} frames total"),
        "",
        media.probe(reel_mp4),
    )
    body += reports._heading("The same frame, at each step of the schedule")
    body += (
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{progression}</div>'
        + reports.note(
            f"Frame {mid} of the clip, decoded at each sampled step. This is the "
            f"model's prediction of the FINISHED video at that point in the schedule, "
            f"not the noisy sample it is currently holding. Watch which properties "
            f"are settled in the leftmost picture and never really change afterwards, "
            f"and which ones keep moving to the end."
        )
    )
    body += reports.metric_lines(
        {
            "sharpness (variance of Laplacian)": [s["sharpness"] for s in stage_stats],
            "inter-frame motion": [s["motion"] for s in stage_stats],
        },
        caption=(
            "Measured on each snapshot, and each line is labelled with the direction "
            "it actually moved rather than the direction the story wants. Both climb "
            "from a low start, because the model's first prediction of the finished "
            "clip is smooth and nearly still: it has committed to a scene before it "
            "has committed to any texture or any movement within it. The number worth "
            "reading is not the direction but where each line flattens, which is in "
            "the run facts above: whatever is already at its final value halfway "
            "through the schedule is not what the second half of the compute is buying."
        ),
    )
    body += reports.clip_block(
        "The finished clip",
        media.video_html(final_mp4, f"{frames} frames at {width}x{height}, seed {seed}"),
        media.strip(final),
        media.probe(final_mp4),
        prompt,
    )
    flyte.report.replace(
        reports.final_html("Watching it resolve", rows, body, reports.EMERGE_EXPLAINER),
        do_flush=True,
    )
    return {
        "scene": scene,
        "snapshots": [s for s, _ in stages],
        "sharpness": [round(s["sharpness"], 1) for s in stage_stats],
        "seconds": round(secs, 1),
        "probe": media.probe(final_mp4),
    }


@gpu_env.task(report=True)
async def extend(
    scene: str = "dashcam",
    repo: str = NANO,
    segments: int = 3,
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Generate a clip, then keep going: each segment conditions on the last one.

    The fourth generation mode, selected by passing `video=` with no `action=`. It
    matters because it is the only way past a fixed clip length: the model's context
    is one clip, so a longer video has to be built by re-entering its own output as
    conditioning.

    That also makes it the honest demonstration of the central weakness. Segment 1
    conditions on a real generated clip. Segment 2 conditions on segment 1, which was
    itself a prediction. Nothing anchors the sequence to anything real, so error does
    not stay put: it accumulates. Watch the sharpness number across segments.
    """
    rows = [
        ("Model", repo),
        ("Scene", scene),
        ("Chain", f"1 generated clip + {segments} continuations"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    prompt = prompts.get(scene)
    _paint("Generating", "The opening clip, from text alone.", rows)
    result, secs = world.generate(
        pipe, prompt,
        negative_prompt=prompts.NEGATIVE,
        num_frames=frames, height=height, width=width,
        steps=steps, guidance=guidance, seed=seed,
    )

    pieces = [list(result.video)]
    stats = [world.clip_stats(pieces[0])]
    cells = [(
        "segment 0 (from text)",
        media.video_html(media.encode(pieces[0], fps=24), f"{secs:.0f}s", max_width=380),
    )]
    total = secs

    for i in range(1, segments + 1):
        _paint("Extending", f"segment {i} of {segments}, conditioned on the previous one.", rows)
        # Seeded per segment, so the chain is reproducible without every segment
        # denoising from identical noise.
        seg, seg_secs = world.extend(
            pipe, pieces[-1], prompt,
            negative_prompt=prompts.NEGATIVE,
            num_frames=frames, height=height, width=width,
            steps=steps, guidance=guidance, seed=seed + i,
        )
        pieces.append(seg)
        stats.append(world.clip_stats(seg))
        total += seg_secs
        cells.append((
            f"segment {i} (continues {i - 1})",
            media.video_html(media.encode(seg, fps=24), f"{seg_secs:.0f}s", max_width=380)
            + reports.note(f"sharpness {stats[-1]['sharpness']:.0f}, "
                           f"motion {stats[-1]['motion']:.2f}"),
        ))
        log.info("segment %s: %.1fs, %s", i, seg_secs, world.clip_stats(seg))

    full = world.stitch(pieces)
    full_mp4 = media.encode(full, fps=24)
    probe = media.probe(full_mp4)

    decay = (stats[-1]["sharpness"] - stats[0]["sharpness"]) / (stats[0]["sharpness"] or 1.0)
    rows += [
        ("Total time", f"{total / 60:.1f} min for {len(pieces)} segments"),
        ("Length", f"{len(full)} frames, {len(full) / 24:.1f}s at 24 fps"),
        ("Sharpness change", f"{decay * 100:+.0f}% from first segment to last"),
    ]

    body = reports.clip_block(
        "The whole chain, stitched",
        media.video_html(full_mp4, f"{len(full)} frames across {len(pieces)} segments"),
        media.strip(full, count=8),
        probe,
        prompt,
    )
    body += reports.metric_lines(
        {
            "sharpness (variance of Laplacian)": [s["sharpness"] for s in stats],
            "inter-frame motion": [s["motion"] for s in stats],
            "mean luminance": [s["luminance"] for s in stats],
        },
        caption=(
            "One point per segment. Sharpness falling is the model smoothing its own "
            "output and then re-reading the smoothed version as if it were real. "
            "Motion falling toward zero is the other failure: the chain freezing."
        ),
    )
    body += reports._heading("Segment by segment")
    body += reports.side_by_side(cells)
    flyte.report.replace(
        reports.final_html("Past the end of one clip", rows, body, reports.EXTEND_EXPLAINER),
        do_flush=True,
    )
    return {
        "scene": scene,
        "segments": len(pieces),
        "frames": len(full),
        "sharpness": [round(s["sharpness"], 1) for s in stats],
        "sharpness_change_pct": round(decay * 100, 1),
        "seconds": round(total, 1),
        "probe": probe,
    }

@gpu_env.task(report=True)
async def horizon(
    repo: str = NANO,
    chunks: int = 4,
    segments: int = 20,
    frames: int = 45,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """The long one: roll the world model forward until it stops being a world.

    This is the overnight run, and it exists to answer the question every short demo
    dodges. A world model that looks convincing for two seconds is a video model with
    good manners. The claim that it has learned physics is a claim about what happens
    when you keep going, so this keeps going: the four action chunks that ship in the
    checkpoint first, then video-to-video continuations for as long as `segments` says,
    every one of them conditioned on the model's own previous output.

    Nothing in the chain is ever re-anchored to anything real after the first frame.
    That is the point and it is also the whole difficulty: at segment 20 the model is
    predicting the future of a scene that it invented, from a frame that it drew, in a
    style that has drifted from the one it started in. If generative world models are
    going to stand in for simulators, this is the failure that has to be fixed, and it
    is much easier to argue about with the video in front of you.

    Renders into the report after EVERY segment, the same way topics/dreamerv3 paints
    its training run: a run that takes hours and shows nothing until it finishes is
    indistinguishable from a hung one, and the interesting part is the trajectory
    rather than the final frame anyway.

    Segment failures are caught rather than fatal. An hour into an unattended run, a
    report holding fifteen good segments and a note about the sixteenth is worth far
    more than a stack trace and nothing else.
    """
    rows = [
        ("Model", repo),
        ("Task", "long-horizon autoregressive rollout"),
        ("Plan", f"{chunks} action chunk(s), then {segments} video continuation(s)"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    fps = int(meta.get("fps", 10))
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({meta['chunks'].shape[-1]}-D actions)"),
        ("Prompt", meta["prompt"]),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    # How often the whole-rollout video is re-encoded. Every segment is too often
    # (see `repaint`); never is useless.
    FULL_EVERY = 5

    pieces: list[list] = []
    stats: list[dict] = []
    labels: list[str] = []
    timings: list[float] = []
    note: str = ""
    frame = meta["first_frame"]

    # Cache for the stitched clip, so the expensive artifact is not rebuilt every time.
    full_clip: dict = {"mp4": b"", "at": 0}

    def repaint(stage: str, detail: str, full: bool = False) -> None:
        """Rebuild the whole report from what exists so far.

        `replace` rather than `log`, deliberately: log() appends, so a run that
        repaints sixty times stacks sixty copies of a growing video into one report
        and the browser dies somewhere around segment eight. topics/rl-mujoco is the
        reference for this.

        The whole-rollout video is rebuilt only every `FULL_EVERY` segments, and that
        is a cost decision rather than a cosmetic one. Every repaint UPLOADS the report
        to the object store, so re-encoding a clip that grows to a couple of thousand
        frames and shipping it sixty times is most of a gigabyte of writes into a
        component this box already knows leaks heap (see the rustfs note in
        preflight.sh), plus around twenty minutes of encoding stolen from the run.
        What repaints every single segment is the cheap and genuinely live part: the
        newest segment, the metrics, and a strip across the whole rollout.
        """
        stitched = world.stitch(pieces) if pieces else []
        if not stitched:
            _paint(stage, detail, rows)
            return

        if full or not full_clip["mp4"]:
            # crf 30 rather than the usual 26, and a width that steps down as the
            # rollout grows. Checking the encoded size rather than guessing from the
            # frame count means an unattended overnight run cannot quietly cross the
            # embed limit and replace its own video with an apologetic paragraph.
            for target_width in (480, 384, 288, 224):
                candidate = media.encode(media.downscale(stitched, target_width), fps=fps, crf=30)
                if len(candidate) < 14 * 2**20:
                    break
            full_clip["mp4"], full_clip["at"] = candidate, len(pieces)

        mp4 = full_clip["mp4"]
        live = list(rows) + [
            ("Progress", f"{len(pieces)} of {chunks + segments} segments"),
            ("Length so far", f"{len(stitched)} frames, {len(stitched) / fps:.1f}s at {fps} fps"),
            ("Elapsed", f"{sum(timings) / 60:.1f} min"),
        ]
        body = reports.clip_block(
            f"{stage}: everything predicted so far",
            media.video_html(
                mp4,
                f"the first {full_clip['at']} segments"
                + ("" if full_clip["at"] == len(pieces) else
                   f"; the strip and the charts below are already current to "
                   f"segment {len(pieces)}, this clip is rebuilt every {FULL_EVERY}"),
            ),
            media.strip(stitched, count=10, width=110),
            media.probe(mp4),
        )
        if pieces:
            body += reports._heading("The newest segment on its own")
            body += media.video_html(
                media.encode(pieces[-1], fps=fps, crf=28),
                f"segment {len(pieces) - 1}: {labels[-1] if labels else ''}",
                max_width=380,
            )
        if len(stats) > 1:
            body += reports.metric_lines(
                {
                    "sharpness (variance of Laplacian)": [s["sharpness"] for s in stats],
                    "inter-frame motion": [s["motion"] for s in stats],
                    "mean luminance": [s["luminance"] for s in stats],
                },
                caption=(
                    "One point per segment, in order. Sharpness sliding is the model "
                    "smoothing its own output and then treating the smoothed version "
                    "as ground truth. Motion heading for zero is the rollout freezing. "
                    "Luminance walking off is the exposure anchor going."
                ),
            )
        if note:
            body += reports.note(note)
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.HORIZON_EXPLAINER), do_flush=True
        )

    # ── Phase 1: driven by the recorded actions ─────────────────────────────────
    #
    # Only these first chunks have real actions behind them. They are the part of the
    # rollout with an external reference, so they double as the control: whatever the
    # clip looks like here is what this model does when it is on its best behaviour.
    available = int(meta["chunks"].shape[0])
    for i in range(min(chunks, available)):
        repaint("Rolling forward on recorded actions", f"chunk {i + 1} of {chunks}", full=True)
        try:
            seg, secs = world.rollout_chunk(
                pipe, meta, meta["chunks"][i], frame=frame,
                steps=steps, guidance=guidance, seed=seed + i,
            )
        except Exception as exc:  # noqa: BLE001
            note = f"Action chunk {i} failed and the run moved on: {exc}"
            log.warning(note)
            break
        # Drop the reproduced conditioning frame on every chunk after the first, or
        # the seam is a duplicated frame in the middle of the rollout.
        pieces.append(seg if i == 0 else seg[1:])
        stats.append(world.clip_stats(seg))
        labels.append(f"action chunk {i}")
        timings.append(secs)
        frame = seg[-1]
        log.info("chunk %s: %s frames, %.1fs, %s", i, len(seg), secs, stats[-1])

    # ── Phase 2: past the end of the actions ────────────────────────────────────
    #
    # The asset runs out of actions after four chunks. Video-to-video conditioning is
    # what carries the rollout past that, and from here the model is extending a scene
    # it drew itself with nothing but its own last frames to go on.
    if pieces:
        tail = pieces[-1]
        w, h = tail[-1].size
        # The pipeline requires height and width to be multiples of the VAE's spatial
        # scale factor. Flooring to 16 satisfies that with room to spare and matches
        # the size the action rollout already produced, so nothing gets rescaled.
        w, h = w - (w % 16), h - (h % 16)
        for i in range(segments):
            repaint(
                "Continuing past the recorded actions",
                f"continuation {i + 1} of {segments}",
                full=(i % FULL_EVERY == 0),
            )
            try:
                seg, secs = world.extend(
                    pipe, tail, meta["prompt"],
                    num_frames=frames, height=h, width=w, fps=fps,
                    steps=steps, guidance=guidance, seed=seed + 100 + i,
                )
            except Exception as exc:  # noqa: BLE001
                note = (
                    f"Continuation {i} failed after {len(pieces)} good segments, and "
                    f"the run stopped there rather than losing them: {exc}"
                )
                log.warning(note)
                break
            pieces.append(seg[world.V2V_OVERLAP:])
            stats.append(world.clip_stats(seg))
            labels.append(f"continuation {i}")
            timings.append(secs)
            tail = seg
            log.info("continuation %s: %.1fs, %s", i, secs, stats[-1])

    repaint("Finished", "", full=True)

    stitched = world.stitch(pieces)
    sharp = [s["sharpness"] for s in stats]
    decay = (sharp[-1] - sharp[0]) / (sharp[0] or 1.0) if len(sharp) > 1 else 0.0
    log.info("horizon: %s segments, %s frames, sharpness %+.0f%%",
             len(pieces), len(stitched), decay * 100)
    return {
        "segments": len(pieces),
        "labels": labels,
        "frames": len(stitched),
        "seconds_total": round(sum(timings), 1),
        "sharpness": [round(s, 1) for s in sharp],
        "motion": [round(s["motion"], 3) for s in stats],
        "luminance": [round(s["luminance"], 1) for s in stats],
        "sharpness_change_pct": round(decay * 100, 1),
        "note": note,
    }


@gpu_env.task(report=True)
async def gallery(
    repo: str = NANO,
    scenes: str = "",
    frames: int = 45,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Every scene in prompts.py, generated from one model load, into one report.

    `imagine --scene X` four times costs four 2.6 minute loads to do four 8 minute
    generations. This pays the load once, which is the same trade `compare` makes and
    the reason neither of them is a fan-out: there is one GPU here, so parallelising
    would not help even if the load were free.

    The four scenes are all physical-AI scenes on purpose rather than scenery. A world
    model earns its name on contact, occlusion and momentum, and a drone shot over a
    mountain range tests none of those. Each one names a specific physical event you
    can check in the output, which is what the per-clip captions are for: does the
    sponge deform against the plate, does the box pivot about its bottom edge before
    it topples, does the pallet stay level when the forks lift, does the car pitch
    forward under braking.

    `scenes` is a comma-separated subset; empty means all of them.
    """
    wanted = [s.strip() for s in scenes.split(",") if s.strip()] or sorted(prompts.SCENES)
    rows = [
        ("Model", repo),
        ("Scenes", f"{len(wanted)}: {', '.join(wanted)}"),
        ("Each", f"{frames} frames at {width}x{height}, {steps} steps, seed {seed}"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    cells: list[tuple[str, str]] = []
    summary: dict[str, dict] = {}
    total = 0.0

    for i, scene in enumerate(wanted):
        _paint("Generating", f"scene {i + 1} of {len(wanted)}: {scene}", rows)
        prompt = prompts.get(scene)
        result, secs = world.generate(
            pipe, prompt,
            negative_prompt=prompts.NEGATIVE,
            num_frames=frames, height=height, width=width,
            steps=steps, guidance=guidance, seed=seed,
        )
        clip = list(result.video)
        total += secs
        mp4 = media.encode(clip, fps=24)
        probe = media.probe(mp4)
        stats = world.clip_stats(clip)
        summary[scene] = {
            "seconds": round(secs, 1),
            "kb": len(mp4) // 1024,
            "sharpness": round(stats["sharpness"], 1),
            "motion": round(stats["motion"], 3),
            "probe": probe,
        }
        cells.append((
            scene,
            media.video_html(mp4, f"{secs / 60:.1f} min, {len(mp4) / 1024:.0f} KB", max_width=440)
            + media.strip(clip, count=4, width=104)
            + reports.note(f"{probe}, sharpness {stats['sharpness']:.0f}")
            + reports.details("structured prompt", prompt),
        ))
        log.info("%s: %.1fs, %s", scene, secs, probe)

        # Repaint with everything finished so far rather than waiting for the last
        # scene. Four generations is over half an hour, and a report that shows the
        # first clip after eight minutes is a very different experience from one that
        # shows nothing for thirty five.
        flyte.report.replace(
            reports.final_html(
                f"Scene gallery ({i + 1} of {len(wanted)})",
                rows + [("Elapsed", f"{total / 60:.1f} min")],
                reports.side_by_side(cells),
                reports.GALLERY_EXPLAINER,
            ),
            do_flush=True,
        )

    rows.append(("Total", f"{total / 60:.1f} min for {len(wanted)} scenes"))
    flyte.report.replace(
        reports.final_html(
            "Scene gallery", rows, reports.side_by_side(cells), reports.GALLERY_EXPLAINER
        ),
        do_flush=True,
    )
    return {"scenes": summary, "seconds": round(total, 1)}


# ── The reasoning surface ───────────────────────────────────────────────────────
#
# Everything above this line uses `Cosmos3OmniPipeline`, the generation surface. The
# three tasks below use `Cosmos3OmniForConditionalGeneration`, the understanding
# surface, and they load it FROM THE SAME FILES: the checkpoint's two index manifests
# both point at transformer/*.safetensors and vision_encoder/model.safetensors, so on
# a box where the shared cache is already staged this costs no extra download at all.
#
# They are not three more generation demos. Each one uses the understanding surface as
# an INSTRUMENT on the generation surface, which is the thing that only an omnimodal
# model can do without a second vendor in the loop:
#
#   plan    the model decomposes a goal, then imagines each step it wrote
#   judge   the model watches its own long rollout and scores it
#   blind   the model rules on the counterfactuals without seeing the actions
#
# The questions are constants rather than inline strings because they are measurement
# instruments: `judge` compares scores ACROSS segments and `blind` compares them
# across variants, and either comparison is void if the wording drifted between calls.

Q_DESCRIBE = "Describe what is happening in this video in one sentence."

# Asked only when the first, unprompted answer does not decompose into steps. The
# checkpoint's own planning prompt gets a four-step plan out of transformers 5.14 and a
# single sentence out of 5.16 (see the pinning note in config.py), so a task that only
# ever asks once is at the mercy of which of those it is talking to. Both answers go in
# the report.
Q_REPLAN_SUFFIX = " Answer as a numbered list, with one short step on each line."

Q_PLAUSIBLE = (
    "Does the motion in this video obey real-world physics? "
    "Answer with a score from 1 to 10, then one sentence of justification."
)

# "still" rather than "stationary": the model answers the free-text half in its own
# words either way, but the one-word half is what gets parsed, and a short common
# word is the one it reliably leads with.
Q_MOVING = (
    "Is the robot in this video moving, or is it holding still? "
    "Answer with one word, moving or still, then one sentence of justification."
)

# A three-way category, not the 1-10 scale this started as. The scale was measured and
# it does not work: on the four counterfactual clips it returned held=10, recorded=1,
# reversed=1, amplified=2, which is close to the reverse of the truth, and it flatly
# contradicted the model's own moving/still answer about the SAME clip in two of four
# cases ("still ... stationary" paired with "10. actively moving"). The binary verdict
# was right in the same run. So the instrument was changed to ask for something the
# model can actually answer: a coarse bucket rather than a number it has no calibration
# for on 1.7 seconds of video.
MOTION_CHOICES = ("stationary", "small movement", "large movement")

Q_MOTION = (
    "How much does the robot move in this video? Choose exactly one of: "
    "stationary, small movement, large movement. "
    "Answer with your choice first, then one sentence of justification."
)


@gpu_env.task(report=True)
async def plan(
    repo: str = NANO,
    goal: str = "",
    clips: int = 4,
    frames: int = 45,
    height: int = 480,
    # 720x480 rather than the 832x480 the rest of this file uses. The reasoning asset
    # is 512x341, which is 3:2 to within a rounding error, and 720x480 is exactly 3:2
    # with both sides a multiple of 16 (the VAE's spatial scale factor). Generating at
    # 832x480 would letterbox or stretch the one frame the whole clip is anchored on.
    width: int = 720,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Both surfaces of one checkpoint: the model plans a task, then imagines its plan.

    The README used to claim Cosmos 3 "exposes two surfaces, a Reasoner for
    understanding and planning and a Generator for world simulation" while every task
    in this file demonstrated only the second one. This is the first one.

    Phase 1 is `Cosmos3OmniForConditionalGeneration` given a photograph and a goal --
    both ship inside the checkpoint as `assets/example_reasoning_*` -- and asked to
    decompose it into subtasks. No video is involved and no diffusion runs; it is a
    17.5 GB vision-language model answering a question, in about four seconds.

    Phase 2 drops those weights and loads the diffusion expert from the same shards,
    then generates a predicted clip for every subtask the model just wrote, each one
    anchored on the real photograph. So the prompts driving the generator are not ours:
    they are the model's own decomposition of the goal handed back to itself.

    The interesting failure is a subtask that reads perfectly and renders as something
    else, because that is the seam between the two surfaces rather than a flaw in
    either. `--clips 0` runs phase 1 alone in about three minutes.
    """
    rows = [("Model", repo), ("Task", "plan, then imagine the plan")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    example = world.load_reasoning_example(path)
    question = goal or example["prompt"]
    rows += [
        ("Goal", question),
        ("Conditioning frame", f"{example['image'].size[0]}x{example['image'].size[1]}, "
                               "from the checkpoint's own assets"),
    ]

    _paint("Loading the understanding surface",
           "Cosmos3OmniForConditionalGeneration, ~17.5 GB in BF16.", rows)
    model, processor = world.load_reasoner(repo)

    _paint("Planning", "One image, one goal, no video and no diffusion.", rows)
    budget = int(example.get("max_tokens", 1024))
    answer, plan_secs = world.ask(
        model, processor, question, image=example["image"], max_new_tokens=budget
    )
    subtasks = world.split_subtasks(answer, limit=max(clips, 1))

    # The open-ended question first, and only then a format instruction, in that order
    # on purpose: asking for a numbered list up front would be us supplying the
    # decomposition's shape, and the interesting claim is that the model decomposes at
    # all. The retry is recorded and shown rather than quietly replacing the first
    # answer, so a reader can see which of the two the clips below actually came from.
    retry = ""
    if len(subtasks) < 2:
        retry, retry_secs = world.ask(
            model, processor, question + Q_REPLAN_SUFFIX,
            image=example["image"], max_new_tokens=budget,
        )
        plan_secs += retry_secs
        retried = world.split_subtasks(retry, limit=max(clips, 1))
        if len(retried) > len(subtasks):
            subtasks = retried

    rows += [
        ("Planning time", f"{plan_secs:.1f}s" + (" (two questions)" if retry else "")),
        ("Subtasks", f"{len(subtasks)} parsed from the answer"),
    ]

    # Show the plan the moment it exists. Phase 2 is another ten minutes and the plan
    # is already the more interesting half of this report.
    def paint_plan(stage: str, cells: list[tuple[str, str]]) -> None:
        body = reports._heading("The goal, and the frame it was asked about")
        body += media.image_html(example["image"], question, width=420)
        body += reports._heading("The plan the model wrote")
        body += reports.quote(answer, "asked: the checkpoint's own planning prompt")
        if retry:
            body += reports._note(
                "That answer did not decompose into steps, so it was asked once more "
                "with a format instruction appended. Both are shown; the clips below "
                "came from whichever produced more subtasks."
            )
            body += reports.quote(retry, "asked again: " + Q_REPLAN_SUFFIX.strip())
        body += reports._note(
            "Parsed into " + str(len(subtasks)) + " subtask(s): "
            + " / ".join(subtasks) + ". The raw answer is above it so a bad split is "
            "visible rather than silently reshaping what the model said."
        )
        if cells:
            body += reports._heading("Each subtask, imagined")
            body += reports.side_by_side(cells)
        flyte.report.replace(
            reports.final_html(stage, rows, body, reports.PLAN_EXPLAINER), do_flush=True
        )

    paint_plan("Planned", [])
    if clips <= 0 or not subtasks:
        log.info("plan: %s subtasks, generation skipped", len(subtasks))
        return {"goal": question, "answer": answer, "retry": retry,
                "subtasks": subtasks, "plan_seconds": round(plan_secs, 1), "clips": 0}

    # Hand the pool back before asking for 30 GB of diffusion expert. Both experts at
    # once does not fit, and on the GB10 that does not raise, it wedges the box.
    model, processor = None, None
    still = world.release()
    rows.append(("Handover", f"understanding surface released, {still:.1f} GiB still held"))

    _paint("Loading the generation surface",
           "Same checkpoint, same shards, the other expert.", rows)
    pipe = world.load(repo)

    cells: list[tuple[str, str]] = []
    total = 0.0
    for i, subtask in enumerate(subtasks[:clips]):
        _paint("Imagining", f"subtask {i + 1} of {min(clips, len(subtasks))}: {subtask}", rows)
        try:
            result, secs = world.generate(
                pipe, subtask,
                image=example["image"],
                negative_prompt=prompts.NEGATIVE,
                num_frames=frames, height=height, width=width,
                steps=steps, guidance=guidance, seed=seed + i,
            )
        except Exception as exc:  # noqa: BLE001
            cells.append((f"{i + 1}. {subtask}", reports.note(f"generation failed: {exc}")))
            log.warning("subtask %s failed: %s", i, exc)
            continue
        total += secs
        mp4 = media.encode(result.video, fps=24)
        cells.append((
            f"{i + 1}. {subtask}",
            media.video_html(mp4, f"{frames} frames, {secs:.0f}s", max_width=380,
                             autoplay=False)
            + media.strip(result.video, count=4, width=92),
        ))
        paint_plan("Imagining the plan", cells)
        log.info("subtask %s: %.1fs, %s", i, secs, media.probe(mp4))

    rows.append(("Generation time", f"{total / 60:.1f} min for {len(cells)} clip(s)"))
    paint_plan("Plan, and the plan imagined", cells)
    return {
        "goal": question,
        "answer": answer,
        "retry": retry,
        "subtasks": subtasks,
        "plan_seconds": round(plan_secs, 1),
        "clips": len(cells),
        "generate_seconds": round(total, 1),
    }


@gpu_env.task(report=True)
async def judge(
    repo: str = NANO,
    chunks: int = 1,
    # 20, which makes this a ~50 minute task, because 5 was measured and shows nothing.
    # Over six segments the plausibility score sat at 8,8,8,8,8,7 and the description
    # overlap at 0.86 to 1.0: the rollout has not drifted yet, so the instrument is
    # untested rather than validated. The two drifts do not even start together --
    # description overlap breaks at segment 7 and plausibility holds until 17 -- so
    # anything short enough to be comfortable is short enough to be uninformative.
    # `--segments 5` is still the right smoke test for the plumbing.
    segments: int = 20,
    frames: int = 45,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Roll the world model forward, then make it watch what it produced.

    `horizon` measures a long rollout with sharpness, inter-frame motion and
    luminance. Those are good at what they do and they share one blind spot: they can
    only say whether the rollout is still a well-formed VIDEO. They cannot say whether
    it is still a video of the same thing, and the 94-segment run in the README hit
    exactly that wall -- sharpness collapsed and then stopped falling, so the honest
    summary had to be the hedged sentence "stays a plausible video while no longer
    being the video it started as". That sentence is a semantic claim being made by a
    human looking at a frame strip.

    This measures it instead. Same rollout, shorter, then the diffusion expert is
    dropped and the understanding expert is loaded from the same shards and shown each
    segment on its own, with no segment index, no ordering and none of the numbers. It
    is asked what is happening and whether the motion is physically plausible.

    Two series come out that pixel statistics cannot produce:

      plausibility    the model grading its own generation, 1 to 10
      description     how many content words each segment's description still shares
       overlap        with segment 0, which is drift measured semantically

    Both are plotted against sharpness, so the report shows where they agree and where
    they part company. A rollout that stays sharp while its description quietly stops
    mentioning the original scene is the failure worth naming.
    """
    rows = [
        ("Model", repo),
        ("Task", "generate a rollout, then judge it with the same checkpoint"),
        ("Plan", f"{chunks} action chunk(s), then {segments} continuation(s), then a verdict each"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    fps = int(meta.get("fps", 10))
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({meta['chunks'].shape[-1]}-D actions)"),
        ("Prompt", meta["prompt"]),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)

    pieces: list[list] = []
    clips: list[bytes] = []
    stats: list[dict] = []
    labels: list[str] = []
    verdicts: list[dict] = []
    note = ""

    def paint(stage: str, detail: str) -> None:
        """Repaint from whatever exists. Phase 1 fills in clips, phase 2 verdicts.

        `replace` and never `log`: log() appends, so repainting a dozen times would
        stack a dozen copies of every embedded video into one report and kill the
        browser. topics/rl-mujoco is the reference for this and topics/dreamerv3
        repeats it.
        """
        if not pieces:
            _paint(stage, detail, rows)
            return
        live = list(rows) + [("Progress", detail)] if detail else list(rows)
        cells = []
        for i, mp4 in enumerate(clips):
            # autoplay off: this report embeds one clip per segment and the default is
            # 20 of them. Twenty autoplaying loops is a blank tab, not a report.
            block = media.video_html(mp4, f"{labels[i]}, {len(pieces[i])} frames",
                                     max_width=380, autoplay=False)
            block += media.strip(pieces[i], count=4, width=92)
            block += reports.note(
                f"sharpness {stats[i]['sharpness']:.0f}, motion {stats[i]['motion']:.2f}"
            )
            if i < len(verdicts):
                v = verdicts[i]
                block += reports.score_chip(v["score"], label="plausibility")
                block += reports.quote(v["description"], "asked: what is happening here")
                block += reports.quote(v["plausibility"], "asked: does this obey physics")
            else:
                block += reports.note("not yet judged")
            cells.append((f"segment {i}", block))

        body = reports._heading("The rollout, one segment at a time")
        body += reports.side_by_side(cells)

        scored = [v["score"] for v in verdicts if v["score"] is not None]
        if not verdicts and len(stats) > 1:
            # Phase 1 has no verdicts yet, but it does have pixel statistics, and a
            # generation phase that shows no chart for ten minutes is the blank-report
            # problem this file exists to avoid.
            body += reports._heading("What the pixels say so far")
            body += reports.metric_lines(
                {
                    "sharpness (variance of Laplacian)": [s["sharpness"] for s in stats],
                    "inter-frame motion": [s["motion"] for s in stats],
                },
                caption="One point per segment. The judged series join these once "
                        "generation finishes and the other expert is loaded.",
            )
        if len(verdicts) > 1:
            # Sliced to the verdicts that exist, so every series on this chart has one
            # point per segment and they share an x axis while phase 2 fills in.
            series = {"sharpness (variance of Laplacian)": [s["sharpness"] for s in stats[:len(verdicts)]]}
            # Only chart the plausibility line when EVERY segment produced a score.
            # Dropping the misses and plotting what is left would silently slide the
            # remaining points onto the wrong x positions, so a chart captioned "one
            # point per segment" would stop being one.
            if len(scored) == len(verdicts) and len(scored) > 1:
                series["physical plausibility, judged 1-10"] = scored
            elif len(scored) < len(verdicts):
                note_missing = len(verdicts) - len(scored)
                body += reports.note(
                    f"{note_missing} of {len(verdicts)} answers did not contain a 1-10 "
                    "score, so the plausibility line is left off rather than plotted "
                    "with the gaps closed up. The answers themselves are under each clip."
                )
            overlaps = [v["overlap"] for v in verdicts]
            if len(overlaps) > 1:
                series["description overlap with segment 0"] = overlaps
            body += reports._heading("Pixels versus meaning")
            body += reports.metric_lines(series, caption=(
                "One point per segment, in order. Sharpness is what `horizon` already "
                "measures and it only reports whether the clip is still well formed. "
                "The other two come from the model's own reading of its own output: a "
                "plausibility score that holds up while the description overlap falls "
                "is a rollout that still looks like competent video of something else."
            ))
        if note:
            body += reports.note(note)
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.JUDGE_EXPLAINER), do_flush=True
        )

    # ── Phase 1: generate the rollout ───────────────────────────────────────────
    frame = meta["first_frame"]
    available = int(meta["chunks"].shape[0])
    gen_secs = 0.0
    for i in range(min(chunks, available)):
        paint("Rolling forward on recorded actions", f"chunk {i + 1} of {chunks}")
        try:
            seg, secs = world.rollout_chunk(
                pipe, meta, meta["chunks"][i], frame=frame,
                steps=steps, guidance=guidance, seed=seed + i,
            )
        except Exception as exc:  # noqa: BLE001
            note = f"Action chunk {i} failed and the run moved on: {exc}"
            log.warning(note)
            break
        pieces.append(seg)
        clips.append(media.encode(seg, fps=fps, crf=28))
        stats.append(world.clip_stats(seg))
        labels.append(f"action chunk {i}, driven by recorded actions")
        gen_secs += secs
        frame = seg[-1]

    if pieces:
        tail = pieces[-1]
        w, h = tail[-1].size
        w, h = w - (w % 16), h - (h % 16)
        for i in range(segments):
            paint("Continuing past the recorded actions", f"continuation {i + 1} of {segments}")
            try:
                seg, secs = world.extend(
                    pipe, tail, meta["prompt"],
                    num_frames=frames, height=h, width=w, fps=fps,
                    steps=steps, guidance=guidance, seed=seed + 100 + i,
                )
            except Exception as exc:  # noqa: BLE001
                note = (f"Continuation {i} failed after {len(pieces)} good segments, and "
                        f"the run kept them rather than raising: {exc}")
                log.warning(note)
                break
            # Drop the reproduced conditioning frames so the judge sees only new
            # prediction. Handing it five frames it has already seen would flatter the
            # motion rating on every segment but the first.
            body_frames = seg[world.V2V_OVERLAP:]
            pieces.append(body_frames)
            clips.append(media.encode(body_frames, fps=fps, crf=28))
            stats.append(world.clip_stats(body_frames))
            labels.append(f"continuation {i}, conditioned on the model's own output")
            gen_secs += secs
            tail = seg

    # ── Phase 2: hand the pool over and judge ───────────────────────────────────
    pipe = None
    still = world.release()
    rows.append(("Generation time", f"{gen_secs / 60:.1f} min for {len(pieces)} segment(s)"))
    rows.append(("Handover", f"generation surface released, {still:.1f} GiB still held"))
    paint("Handing over to the understanding surface",
          f"{len(pieces)} segments generated; loading the other expert")

    model, processor = world.load_reasoner(repo)
    judge_secs = 0.0
    for i, seg in enumerate(pieces):
        paint("Judging", f"segment {i + 1} of {len(pieces)}")
        try:
            described, s1 = world.ask(model, processor, Q_DESCRIBE, video=seg, max_new_tokens=128)
            scored_text, s2 = world.ask(model, processor, Q_PLAUSIBLE, video=seg, max_new_tokens=192)
        except Exception as exc:  # noqa: BLE001
            note = f"Judging stopped at segment {i}: {exc}"
            log.warning(note)
            break
        judge_secs += s1 + s2
        verdicts.append({
            "description": described,
            "plausibility": scored_text,
            "score": world.parse_score(scored_text),
            # Against segment 0's description, so the series starts at 1.0 by
            # construction and every later point is drift from where it began.
            "overlap": round(world.description_overlap(verdicts[0]["description"], described), 3)
                       if verdicts else 1.0,
        })

    rows.append(("Judging time", f"{judge_secs:.0f}s for {len(verdicts)} verdict(s), "
                                 f"{world.REASON_FRAMES} frames each"))
    paint("Finished", "")

    scores = [v["score"] for v in verdicts if v["score"] is not None]
    overlaps = [v["overlap"] for v in verdicts]
    log.info("judge: %s segments, scores %s, overlap %s", len(pieces), scores, overlaps)
    return {
        "segments": len(pieces),
        "labels": labels,
        "plausibility": scores,
        "overlap": overlaps,
        "descriptions": [v["description"] for v in verdicts],
        "sharpness": [round(s["sharpness"], 1) for s in stats],
        "generate_seconds": round(gen_secs, 1),
        "judge_seconds": round(judge_secs, 1),
        "note": note,
    }


@gpu_env.task(report=True)
async def blind(
    repo: str = NANO,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """The counterfactual test, ruled on by a judge that cannot see the actions.

    `counterfact` is the strongest result in this repo: hold the frame, the prompt,
    the seed and the schedule fixed, change only the actions, and the predicted clips
    differ in the ordering the actions describe. Its measurement is a mean absolute
    pixel difference, which proves the clips are not identical and cannot prove the
    difference is the one the actions asked for. A model with a decorative action
    channel that merely reseeded on it would also produce four different clips.

    So this generates the same four variants and then asks the understanding expert,
    which never sees an action tensor, a variant name or an ordering: is this robot
    moving, or holding still? `held` commands the starting pose at every step. A model
    that reads its actions has to predict a robot that stops, and a judge with no
    access to the actions has to be able to see that it stopped.

    The clips are shuffled before judging. That is not decoration: the four questions
    are separate `generate` calls with no shared history, so there is no context to
    contaminate, but the presentation order is recorded in the report so the claim
    "the judge was blind" is checkable rather than asserted.
    """
    import random

    rows = [("Model", repo), ("Task", "counterfactual actions, judged blind")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    chunk = meta["chunks"][0]
    fps = int(meta.get("fps", 10))
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({chunk.shape[-1]}-D actions)"),
        ("Prompt", meta["prompt"]),
        ("Variants", f"4 x {chunk.shape[0]} actions, identical seed and conditioning frame"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)

    variants = world.counterfactuals(chunk)
    order: list[str] = []
    clips: dict[str, list] = {}
    mp4s: dict[str, bytes] = {}
    why: dict[str, str] = {}
    verdicts: dict[str, dict] = {}
    gen_secs = 0.0

    def paint(stage: str, detail: str) -> None:
        cells = []
        for name in [n for n, _, _ in variants if n in mp4s]:
            block = media.video_html(mp4s[name], f"{len(clips[name])} frames",
                                     max_width=380, autoplay=False)
            block += media.strip(clips[name], count=4, width=92)
            block += reports.note(why[name])
            if name in verdicts:
                v = verdicts[name]
                block += reports.score_chip(
                    v["bucket"], scale=f" of 2 ({v['bucket_label']})", label="judged motion")
                block += reports.quote(v["moving"], "asked: moving, or holding still?")
                block += reports.quote(v["bucket_text"], "asked: how much does it move?")
                if v["self_consistent"] is False:
                    block += reports.note(
                        "The two questions disagree about this clip. Both are shown "
                        "rather than reconciled: an unreliable instrument is a result."
                    )
            cells.append((name, block))
        body = reports._heading("The four futures") + reports.side_by_side(cells)

        if verdicts:
            baseline = clips.get("recorded")
            div = ([(n, world.frame_divergence(baseline, clips[n])) for n, _, _ in variants
                    if n in clips] if baseline else [])
            # Canonical variant order, not the shuffled presentation order, so this
            # chart's bars line up row for row with the divergence chart below it.
            # Comparing the two is the whole point and it does not survive one of them
            # being in a different order.
            rated = [(n, float(verdicts[n]["bucket"])) for n, _, _ in variants
                     if n in verdicts and verdicts[n]["bucket"] is not None]
            if rated:
                agreement = world.rank_agreement(
                    {n: verdicts[n]["bucket"] for n, _ in rated},
                    {n: world.clip_stats(clips[n])["motion"] for n, _ in rated},
                )
                body += reports._heading("What the judge saw, with no access to the actions")
                body += reports.note(
                    f"Ranked against measured inter-frame motion: "
                    f"<b>{agreement['concordant']} concordant</b>, "
                    f"<b>{agreement['discordant']} discordant</b>, "
                    f"{agreement['tied']} tied, of {agreement['pairs']} pairs. "
                    + ("A discordant pair is the judge putting two clips in the "
                       "opposite order from the pixels, and there is at least one here, "
                       "so do not read this as a clean confirmation."
                       if agreement["inverted"] else
                       "No inversions: wherever the judge does commit to an order, it "
                       "is the order the pixels are in. Ties are the bucket declining "
                       "to separate two clips, not a mistake.")
                )
                body += reports.bars(rated, unit=" of 2", caption=(
                    "How much motion the understanding expert reports in each clip, "
                    "shown to it unlabelled and out of order. `held` is the one to "
                    "read: its actions command the starting pose at every step, so a "
                    "low bar here is a judge independently confirming that the action "
                    "channel changed the physics and not just the pixels."
                ))
            if div:
                body += reports._heading("What the pixels say, for comparison")
                body += reports.bars(div, caption=(
                    "Mean absolute difference from the recorded rollout, in grey "
                    "levels out of 255 -- the measurement `counterfact` reports. The "
                    "claim is the agreement between these two charts, not either one "
                    "of them: one is arithmetic on pixels and the other is a model "
                    "that was shown the clips and never told what produced them."
                ))
            if order:
                body += reports.note(
                    "Presentation order to the judge: " + " then ".join(order)
                    + ". Each question is an independent generate() call with no "
                    "shared history, so the shuffle is a check on this task rather "
                    "than a guard against context leaking between clips."
                )
        flyte.report.replace(
            reports.final_html(stage, rows + [("Progress", detail)] if detail else rows,
                               body, reports.BLIND_EXPLAINER),
            do_flush=True,
        )

    # ── Phase 1: four futures from one conditioning frame ───────────────────────
    for i, (name, actions, description) in enumerate(variants):
        _paint("Generating", f"variant {i + 1} of {len(variants)}: {name}", rows)
        frames, secs = world.rollout_chunk(
            pipe, meta, actions, steps=steps, guidance=guidance, seed=seed
        )
        clips[name], why[name] = frames, description
        mp4s[name] = media.encode(frames, fps=fps)
        gen_secs += secs
        paint("Generating", f"variant {i + 1} of {len(variants)}: {name}")
        log.info("variant %s: %.1fs, %s", name, secs, media.probe(mp4s[name]))

    # ── Phase 2: judge them ─────────────────────────────────────────────────────
    pipe = None
    still = world.release()
    rows.append(("Generation time", f"{gen_secs / 60:.1f} min for {len(clips)} rollouts"))
    rows.append(("Handover", f"generation surface released, {still:.1f} GiB still held"))
    paint("Handing over to the understanding surface", "loading the other expert")

    model, processor = world.load_reasoner(repo)
    order = [n for n, _, _ in variants]
    random.Random(seed).shuffle(order)
    judge_secs = 0.0
    for i, name in enumerate(order):
        paint("Judging blind", f"clip {i + 1} of {len(order)}")
        # Every frame, not a sample. These clips are 17 frames of 1.7 seconds, so the
        # frame budget that makes sense for a 45-frame continuation is just throwing
        # away more than half the evidence about the one thing being asked.
        shown = len(clips[name])
        moving, s1 = world.ask(model, processor, Q_MOVING, video=clips[name],
                               frames=shown, max_new_tokens=128)
        bucket_text, s2 = world.ask(model, processor, Q_MOTION, video=clips[name],
                                    frames=shown, max_new_tokens=128)
        judge_secs += s1 + s2
        bucket = world.parse_choice(bucket_text, MOTION_CHOICES)
        says_still = world.parse_still(moving)
        verdicts[name] = {
            "moving": moving,
            "bucket_text": bucket_text,
            "bucket": bucket,
            "bucket_label": MOTION_CHOICES[bucket] if bucket is not None else "no choice given",
            "says_still": says_still,
            # The two questions are independent calls about the same clip, so they can
            # disagree, and when they do that is the result rather than something to
            # average away. Recorded per clip and surfaced in the report.
            "self_consistent": None if says_still is None or bucket is None
                               else (says_still == (bucket == 0)),
        }

    rows.append(("Judging time", f"{judge_secs:.0f}s for {len(verdicts)} clips"))
    paint("Finished", "")

    buckets = {n: v["bucket"] for n, v in verdicts.items()}
    still = {n: v["says_still"] for n, v in verdicts.items()}
    motion = {n: world.clip_stats(f)["motion"] for n, f in clips.items()}
    # The headline is the RANKING, not any single clip. "Did it call `held` still" turns
    # on one wording; whether its ordering of the four inverts the pixel ordering is the
    # claim a three-way bucket can actually carry.
    agreement = world.rank_agreement(buckets, motion)
    log.info("blind: buckets %s, still %s, agreement %s", buckets, still, agreement)
    return {
        "order_shown": order,
        "buckets": buckets,
        "bucket_labels": {n: v["bucket_label"] for n, v in verdicts.items()},
        "says_still": still,
        "self_consistent": {n: v["self_consistent"] for n, v in verdicts.items()},
        "verdicts": {n: v["moving"] for n, v in verdicts.items()},
        "divergence": {n: round(world.frame_divergence(clips["recorded"], f), 3)
                       for n, f in clips.items()},
        "measured_motion": {n: round(v, 3) for n, v in motion.items()},
        "rank_agreement": agreement,
        "generate_seconds": round(gen_secs, 1),
        "judge_seconds": round(judge_secs, 1),
    }


@gpu_env.task(report=True)
async def cycle(
    repo: str = NANO,
    examples: int = 2,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Actions in, video out, actions back. How much does dreaming cost the labels?

    This is the precondition for using Cosmos as a data engine, and nothing else in
    this file measures it. NVIDIA's GR00T-Dreams pipeline generates synthetic robot
    video, filters it with a video critic (`judge` here), labels it with an inverse
    dynamics model (`invert` here) and trains a policy on the labels. Every stage of
    that rests on an assumption nobody states out loud: **that actions recovered from
    generated video are accurate enough to train on.** If they are not, the whole
    pipeline is an expensive way to manufacture label noise.

    So this closes the loop on the only assets in the checkpoint where it can be scored
    honestly. The inverse-dynamics examples are the one place a real video is paired
    with the real actions that produced it, which means the round trip has an answer
    key rather than being graded against itself:

      1. inverse dynamics on the REAL clip           -> the floor
      2. forward dynamics from the SAME real actions -> a dreamed clip
      3. inverse dynamics on the DREAMED clip        -> the round trip

    Steps 1 and 3 are the identical operation on two videos of the same event, one
    filmed and one imagined, so the difference between their errors isolates what the
    generation step cost and nothing else. That is the number to quote. Quoting step 3
    alone would blame the dream engine for the inverse model's own inaccuracy.

    Three generations per example and one model load, which is the only reason two
    examples fit in a task rather than four.
    """
    rows = [("Model", repo), ("Task", "actions -> video -> actions, scored against truth")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)
    path = world.snapshot(repo)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    results: list[dict] = []
    blocks: list[str] = []
    note = ""

    def paint(stage: str, detail: str) -> None:
        body = ""
        for blk in blocks:
            body += blk
        if results:
            body += reports._heading("The tax, in one chart")
            bars: list[tuple[str, float]] = []
            for r in results:
                bars.append((f"example {r['index']}: real clip", r["real"]["mae_moving"]))
                bars.append((f"example {r['index']}: dreamed clip", r["dream"]["mae_moving"]))
            body += reports.bars(bars, caption=(
                "Mean absolute error of the recovered actions on the channels that "
                "actually move, against the actions the real robot executed. The lower "
                "bar of each pair is inverse dynamics reading a real video and is the "
                "floor for this model; the upper bar is the same operation on a video "
                "Cosmos generated from those same actions. The distance between them is "
                "what one round trip through the generator costs a label."
            ))
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.CYCLE_EXPLAINER), do_flush=True
        )

    for i in range(examples):
        _paint("Reading the answer key", f"inverse example {i}", rows)
        try:
            example = world.load_inverse_example(path, i)
        except Exception as exc:  # noqa: BLE001
            note = f"Example {i} could not be read and the run moved on: {exc}"
            log.warning(note)
            break
        truth = example["truth"]
        real_frames = example["frames"]
        fps = int(example["fps"])

        # 1. The floor: how well does inverse dynamics read the REAL video?
        paint("Inverse dynamics on the real clip", f"example {i}, step 1 of 3")
        a_real, t_real = world.invert(pipe, example, steps=steps, guidance=guidance, seed=seed)
        e_real = world.action_error(truth, a_real)

        # 2. Dream it: the same actions, pushed forward from the same first frame.
        paint("Forward dynamics from the same actions", f"example {i}, step 2 of 3")
        fwd = world.forward_meta_from_inverse(example)
        dreamed, t_fwd = world.rollout_chunk(
            pipe, fwd, truth, steps=steps, guidance=guidance, seed=seed
        )

        # 3. Read the dream back.
        paint("Inverse dynamics on the dreamed clip", f"example {i}, step 3 of 3")
        dream_example = dict(example)
        dream_example["frames"] = dreamed
        a_dream, t_dream = world.invert(
            pipe, dream_example, steps=steps, guidance=guidance, seed=seed
        )
        e_dream = world.action_error(truth, a_dream)

        real_mp4 = media.encode(real_frames, fps=fps)
        dream_mp4 = media.encode(dreamed, fps=fps)
        # The channels worth charting are the ones that move; `action_error` already
        # identified them, and plotting the eight that sit flat would bury the one that
        # does not under a row of straight lines.
        dims = e_real["moving_dims"] or [0]
        tax = e_dream["mae_moving"] / (e_real["mae_moving"] or 1e-9)

        blocks.append(
            reports._heading(f"Example {i}: the same event, filmed and imagined")
            + reports.side_by_side([
                ("the real clip", media.video_html(
                    real_mp4, f"{len(real_frames)} frames as recorded", max_width=380,
                    autoplay=False)),
                ("Cosmos dreaming the same actions", media.video_html(
                    dream_mp4, f"{len(dreamed)} frames, {t_fwd:.0f}s", max_width=380,
                    autoplay=False)),
            ])
            + reports.note(
                f"Recovered-action error on the moving channel(s) {dims}: "
                f"<b>{e_real['mae_moving']:.4f}</b> from the real clip, "
                f"<b>{e_dream['mae_moving']:.4f}</b> from the dreamed one, "
                f"a <b>{tax:.1f}x</b> increase. Normalised to the channel's own range "
                f"that is {min(e_real['nmae_per_dim'][d] for d in dims) * 100:.1f}% "
                f"against {min(e_dream['nmae_per_dim'][d] for d in dims) * 100:.1f}%."
            )
            + reports._heading(f"Example {i}: what the model read back")
            + reports.action_traces(
                truth.tolist(), a_real.tolist(), dims=dims,
                labels=["executed", "read off the real clip"],
                caption="Inverse dynamics on the real video. This is the floor.",
            )
            + reports.action_traces(
                truth.tolist(), a_dream.tolist(), dims=dims,
                labels=["executed", "read off the dreamed clip"],
                caption="The same operation on the generated video. The extra wander "
                        "here is what the round trip added, and it is the label noise a "
                        "policy trained on dreams would be learning from.",
            )
        )
        results.append({
            "index": i,
            "real": e_real,
            "dream": e_dream,
            "tax": round(tax, 2),
            "seconds": round(t_real + t_fwd + t_dream, 1),
        })
        log.info("cycle %s: real %.4f, dream %.4f, %.1fx",
                 i, e_real["mae_moving"], e_dream["mae_moving"], tax)
        paint("Round trip complete", f"example {i} done")

    if results:
        mean_tax = sum(r["tax"] for r in results) / len(results)
        rows.append(("Round-trip tax", f"{mean_tax:.1f}x the inverse model's own error, "
                                       f"averaged over {len(results)} example(s)"))
    paint("Finished", "")

    log.info("cycle: %s", [(r["index"], r["tax"]) for r in results])
    return {
        "examples": len(results),
        "real_mae_moving": [round(r["real"]["mae_moving"], 5) for r in results],
        "dream_mae_moving": [round(r["dream"]["mae_moving"], 5) for r in results],
        "tax": [r["tax"] for r in results],
        "seconds": [r["seconds"] for r in results],
        "note": note,
    }


# ── The data engine ─────────────────────────────────────────────────────────────
#
# DreamGen's central claim is that a world model can produce behaviours you never
# demonstrated: one real frame plus a language instruction, and out comes a plausible
# execution of something that is not in your dataset. These are those instructions.
#
# The `av` domain rather than `agibotworld`, and that is a deliberate constraint. The
# round trip has to work in BOTH directions for a dream to become training data, and
# `cycle` has only proven that on `av`. Generating beautiful supermarket dreams that
# inverse dynamics cannot label would be a demo of the wrong half of the pipeline.
#
# The first entry is the behaviour the real clip actually shows, kept as a control: if
# the critic rejects the scenario the model was conditioned on, the critic is the
# problem rather than the dreams.
DREAM_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("The ego vehicle drives forward along the road.",
     "Is the vehicle driving forward along the road?"),
    ("The ego vehicle slows down as the truck ahead brakes.",
     "Does the vehicle slow down or approach a braking vehicle ahead?"),
    ("The ego vehicle changes lane to the left, overtaking the truck ahead.",
     "Does the vehicle move sideways into a different lane?"),
    ("The ego vehicle comes to a stop at a red traffic light.",
     "Does the vehicle come to a stop?"),
    ("The ego vehicle turns right at the intersection.",
     "Does the vehicle turn right?"),
    ("The ego vehicle drives forward as rain begins to fall on the windscreen.",
     "Is it raining in this video?"),
)

Q_PLAUSIBLE_SHORT = (
    "Does the motion in this video obey real-world physics? "
    "Answer with a score from 1 to 10, then one short sentence."
)

# The bar a dream has to clear to become training data. 5 of 10 rather than something
# stricter because the point is to filter out the broken ones, not to curate: over-
# filtering a synthetic dataset throws away exactly the unusual behaviours it was
# generated to supply.
KEEP_THRESHOLD = 5.0


@gpu_env.task(report=True)
async def dream(
    repo: str = NANO,
    scenarios: int = 6,
    frames: int = 61,
    height: int = 480,
    width: int = 832,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """The GR00T-Dreams pipeline end to end: generate, critique, label.

    Stages 3, 4 and 5 of NVIDIA's own diagram, all three running off one checkpoint:

      3. generate novel behaviours from one real frame and a language instruction
      4. filter them with a video critic (Cosmos Reason's job; the understanding
         surface here)
      5. label the survivors with inverse dynamics (a mode flag, not a second model)

    The behaviours are NOT in the training clip. The real video shows a car following a
    truck down a road; the instructions ask for lane changes, stops, right turns and
    rain. That is the whole DreamGen proposition, which is that a world model can supply
    demonstrations of things nobody demonstrated.

    What this task does NOT do is stage 6, training a policy on the output, and the
    honest reason is data: the bundled assets support a working pipeline at toy scale
    and not a claim about whether dreams beat real data. What it does give you is the
    two numbers stage 6 would need before it was worth attempting, which are the
    rejection rate and, from `cycle`, the label error.

    Three model loads, in NVIDIA's order rather than the convenient one. Generating and
    labelling in a single pass would save a 2.6 minute reload and would mean paying ~170s
    of inverse dynamics on dreams the critic was about to throw away. At six scenarios
    that is close to a wash; at sixty it is not, and the order that scales is the one
    worth writing down.
    """
    plan = list(DREAM_SCENARIOS[:max(1, scenarios)])
    rows = [
        ("Model", repo),
        ("Task", "generate novel behaviours, critique them, label the survivors"),
        ("Scenarios", f"{len(plan)}, none of which the conditioning clip demonstrates"),
        ("Keep threshold", f"plausibility >= {KEEP_THRESHOLD:g}/10 AND the critic "
                           f"confirms the behaviour"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    example = world.load_inverse_example(path, 0)
    first_frame = example["frames"][0]
    fps = int(example["fps"])
    rows.append(("Conditioning frame", f"{first_frame.size[0]}x{first_frame.size[1]}, "
                                       f"frame 0 of a real driving clip"))

    guard = world.guard_memory()
    rows.append(("GPU", guard))

    dreams: list[dict] = []
    note = ""

    def paint(stage: str, detail: str) -> None:
        cells = []
        for d in dreams:
            block = media.video_html(d["mp4"], f"{d['frames']} frames, {d['secs']:.0f}s",
                                     max_width=380, autoplay=False)
            block += media.strip(d["video"], count=4, width=92)
            if d.get("judged"):
                verdict = "KEPT" if d["keep"] else "REJECTED"
                colour = "#00b894" if d["keep"] else "#d63031"
                block += (f'<p style="font-family:monospace;font-size:13px;'
                          f'color:{colour};font-weight:bold;margin:8px 0 4px;">'
                          f'{verdict}</p>')
                block += reports.score_chip(d["score"], label="plausibility")
                block += reports.quote(d["behaviour_answer"], "asked: " + d["question"])
                block += reports.quote(d["plausible_answer"], "asked: does it obey physics?")
            if d.get("labelled"):
                block += reports.note(
                    f"Labelled: {d['action_shape']} actions recovered in "
                    f"{d['label_secs']:.0f}s."
                )
            cells.append((d["prompt"], block))

        body = reports._heading("The dreams") + reports.side_by_side(cells)

        judged = [d for d in dreams if d.get("judged")]
        if judged:
            kept = [d for d in judged if d["keep"]]
            body += reports._heading("What the critic kept")
            body += reports.note(
                f"<b>{len(kept)} of {len(judged)}</b> dreams survived. Rejection is not "
                "failure: a generator that never produced anything the critic threw out "
                "would either be generating only the behaviour it was trained on, or the "
                "critic would not be doing anything. The control scenario (the behaviour "
                "the real clip actually shows) is the one that must survive."
            )
            body += reports.bars(
                [(d["prompt"][:38], d["score"] or 0.0) for d in judged],
                unit="/10",
                caption="Self-judged physical plausibility per dream. The threshold is "
                        f"{KEEP_THRESHOLD:g}; a dream also has to have the critic confirm "
                        "the behaviour it was asked for, so a high bar alone is not a pass.",
            )
        labelled = [d for d in dreams if d.get("labelled")]
        if labelled:
            body += reports._heading("The dataset this produced")
            body += reports.note(
                f"<b>{len(labelled)}</b> labelled trajectories, "
                f"{sum(d['frames'] for d in labelled)} frames total, each with a "
                f"{labelled[0]['action_shape']} action tensor recovered by inverse "
                "dynamics. Per <code>cycle</code>, expect roughly 3x the inverse model's "
                "own error on those labels, because they were read off generated video "
                "rather than filmed video. That is the budget stage 6 would be spending."
            )
            body += reports.action_traces(
                labelled[0]["actions"], labelled[0]["actions"],
                dims=[2], labels=["recovered", ""],
                caption=f"The recovered action trace for '{labelled[0]['prompt']}'. "
                        "There is no ground truth to overlay here and that is the point: "
                        "this behaviour was never demonstrated, so the label is all there "
                        "is. `cycle` is what tells you how much to trust it.",
            )
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.DREAM_EXPLAINER), do_flush=True
        )

    # ── Stage 3: generate ───────────────────────────────────────────────────────
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)
    gen_secs = 0.0
    for i, (prompt, question) in enumerate(plan):
        paint("Dreaming", f"scenario {i + 1} of {len(plan)}")
        try:
            result, secs = world.generate(
                pipe, prompt, image=first_frame, negative_prompt=prompts.NEGATIVE,
                num_frames=frames, height=height, width=width,
                steps=steps, guidance=guidance, seed=seed + i,
            )
        except Exception as exc:  # noqa: BLE001
            note = f"Scenario {i} failed to generate and the run moved on: {exc}"
            log.warning(note)
            continue
        video = list(result.video)
        gen_secs += secs
        dreams.append({
            "index": i, "prompt": prompt, "question": question,
            "video": video, "frames": len(video), "secs": secs,
            "mp4": media.encode(video, fps=fps),
        })
        log.info("dream %s: %s frames, %.0fs", i, len(video), secs)
    paint("Dreamt", f"{len(dreams)} generated")

    # ── Stage 4: critique ───────────────────────────────────────────────────────
    pipe = None
    still = world.release()
    rows.append(("Generation time", f"{gen_secs / 60:.1f} min for {len(dreams)} dreams"))
    rows.append(("Handover", f"generation surface released, {still:.1f} GiB still held"))
    paint("Handing over to the critic", "loading the understanding surface")

    model, processor = world.load_reasoner(repo)
    judge_secs = 0.0
    for i, d in enumerate(dreams):
        paint("Critiquing", f"dream {i + 1} of {len(dreams)}")
        behaviour, s1 = world.ask(model, processor, d["question"] + " Answer yes or no, "
                                  "then one short sentence.", video=d["video"],
                                  max_new_tokens=128)
        plausible, s2 = world.ask(model, processor, Q_PLAUSIBLE_SHORT,
                                  video=d["video"], max_new_tokens=160)
        judge_secs += s1 + s2
        score = world.parse_score(plausible)
        # Both conditions, deliberately. A dream can be beautifully physical and show
        # the wrong behaviour, which as training data is worse than an obviously broken
        # clip: it is a correctly-labelled demonstration of something you did not ask for.
        confirms = world.parse_choice(behaviour, ("yes", "no")) == 0
        d.update({
            "judged": True,
            "behaviour_answer": behaviour,
            "plausible_answer": plausible,
            "score": score,
            "confirms": confirms,
            "keep": bool(confirms and score is not None and score >= KEEP_THRESHOLD),
        })
        log.info("critic %s: confirms=%s score=%s keep=%s", i, confirms, score, d["keep"])
    paint("Critiqued", f"{sum(1 for d in dreams if d['keep'])} kept")

    # ── Stage 5: label the survivors ────────────────────────────────────────────
    model, processor = None, None
    still = world.release()
    rows.append(("Critique time", f"{judge_secs:.0f}s for {len(dreams)} dreams"))
    survivors = [d for d in dreams if d["keep"]]
    rows.append(("Survivors", f"{len(survivors)} of {len(dreams)} going to labelling"))
    paint("Handing back to label the survivors", f"{len(survivors)} dreams to label")

    label_secs = 0.0
    if survivors:
        pipe = world.load(repo)
        for i, d in enumerate(survivors):
            paint("Labelling", f"survivor {i + 1} of {len(survivors)}")
            meta = dict(example)
            meta["frames"] = d["video"]
            meta["description"] = d["prompt"]
            try:
                actions, secs = world.invert(
                    pipe, meta, steps=steps, guidance=guidance, seed=seed
                )
            except Exception as exc:  # noqa: BLE001
                note = f"Labelling dream {d['index']} failed: {exc}"
                log.warning(note)
                continue
            label_secs += secs
            d.update({
                "labelled": True,
                "actions": actions.tolist(),
                "action_shape": str(tuple(actions.shape)),
                "label_secs": secs,
            })
            log.info("labelled %s: %s in %.0fs", d["index"], tuple(actions.shape), secs)

    rows.append(("Labelling time", f"{label_secs / 60:.1f} min"))
    paint("Finished", "")

    labelled = [d for d in dreams if d.get("labelled")]
    log.info("dream: %s generated, %s kept, %s labelled",
             len(dreams), len(survivors), len(labelled))
    return {
        "generated": len(dreams),
        "kept": len(survivors),
        "labelled": len(labelled),
        "rejection_rate": round(1 - len(survivors) / max(len(dreams), 1), 3),
        "verdicts": {d["prompt"]: {"score": d.get("score"),
                                   "confirms": d.get("confirms"),
                                   "keep": d.get("keep")} for d in dreams},
        "generate_seconds": round(gen_secs, 1),
        "critique_seconds": round(judge_secs, 1),
        "label_seconds": round(label_secs, 1),
        "note": note,
    }


# ── Planning in imagination ─────────────────────────────────────────────────────
#
# Two goals over the SAME four imagined futures, and they want opposite things. That
# pairing is the entire experiment. Scoring four clips against one goal and announcing
# a winner proves nothing: the highest score might just be the clip the model likes
# best, and you could not tell from a single ranking. Ask for the opposite goal over the
# identical clips and the winner has to MOVE. If it does not, the "planner" is reading
# the video and ignoring the goal.
#
# Each goal names the bucket a satisfying clip should land in, so the ranking uses the
# three-way categorical measurement `blind` established works (4 concordant pairs, 0
# discordant against measured motion) rather than the 1-10 scale that saturates.
CHOOSE_GOALS: tuple[tuple[str, str, int], ...] = (
    (
        "get the item moved",
        "Does the robot reach for and move an item in this video? "
        "Answer yes or no, then one short sentence.",
        2,  # wants large movement
    ),
    (
        "hold the arm still and disturb nothing",
        "Does the robot hold still without disturbing anything in this video? "
        "Answer yes or no, then one short sentence.",
        0,  # wants stationary
    ),
)


@gpu_env.task(report=True)
async def choose(
    repo: str = NANO,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Imagine four futures, then pick the one that serves the goal.

    The closest thing here to what `topics/dreamerv3` does with its world model, and the
    oldest argument for having one at all: you do not have to try an action in the world
    if you can predict what it would do. Cosmos rolls four different action sequences
    forward from one identical conditioning frame, and the understanding surface of the
    same checkpoint scores each imagined future against a goal stated in words. The
    highest scorer is the action a planner would execute.

    Dreamer learns a value function to do this scoring and needs millions of environment
    steps to train it. Here the scorer is a pretrained VLM that has never seen this
    robot, and the whole loop is one checkpoint and no training at all. That is the
    trade being illustrated: Dreamer's critic is sharp but expensive and specific to one
    environment; this one is general, free, and much blunter.

    **The control is the second goal.** Announcing a winner for a single goal would be
    unfalsifiable, because the top-scoring clip might simply be the one the model finds
    most appealing and nothing would distinguish that from planning. So the same four
    futures are scored twice, against goals that want opposite things, and the claim is
    only earned if the winner MOVES. `held` (the arm commanded to its starting pose at
    every step) should lose the first goal and win the second.

    What this does not do is execute anything: there is no robot, so the output is the
    choice rather than its consequences. The honest framing is that this shows the
    selection step of a planner, not a closed control loop.
    """
    rows = [("Model", repo), ("Task", "score four imagined futures against two goals")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    chunk = meta["chunks"][0]
    fps = int(meta.get("fps", 10))
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({chunk.shape[-1]}-D actions)"),
        ("Candidates", f"4 action sequences, identical seed and conditioning frame"),
        ("Goals", " / ".join(f"'{g}'" for g, _, _ in CHOOSE_GOALS)),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)

    variants = world.counterfactuals(chunk)
    clips: dict[str, list] = {}
    mp4s: dict[str, bytes] = {}
    # goal -> {candidate -> {"answer","achieves"}}
    scored: dict[str, dict] = {}
    winners: dict[str, str | None] = {}
    buckets: dict[str, int | None] = {}
    gen_secs = 0.0

    def paint(stage: str, detail: str) -> None:
        cells = []
        for name, _, why in variants:
            if name not in mp4s:
                continue
            block = media.video_html(mp4s[name], f"{len(clips[name])} frames",
                                     max_width=380, autoplay=False)
            block += media.strip(clips[name], count=4, width=92)
            block += reports.note(why)
            if name in buckets and buckets[name] is not None:
                block += reports.score_chip(
                    buckets[name], scale=f" of 2 ({MOTION_CHOICES[buckets[name]]})",
                    label="judged motion")
            for goal, _, _ in CHOOSE_GOALS:
                if goal in scored and name in scored[goal]:
                    v = scored[goal][name]
                    mark = "yes" if v["achieves"] else "no"
                    won = " <b>(chosen)</b>" if winners.get(goal) == name else ""
                    block += reports.note(
                        f"goal '<b>{goal}</b>': {mark}{won}"
                    )
                    block += reports.quote(v["answer"], f"asked about: {goal}")
            cells.append((name, block))
        body = reports._heading("Four imagined futures, one conditioning frame")
        body += reports.side_by_side(cells)

        if winners:
            body += reports._heading("What a planner would have executed")
            picked = {g: w for g, w in winners.items() if w}
            for goal, _, _ in CHOOSE_GOALS:
                if goal in winners:
                    w = winners[goal] or "nothing (no candidate satisfied it)"
                    body += reports.note(f"To <b>{goal}</b>, it picks <b>{w}</b>.")
            distinct = len(set(picked.values()))
            if len(picked) > 1:
                body += reports.note(
                    ("<b>The choice moved with the goal.</b> Same four clips, same "
                     "scorer, opposite goals, different winners: the ranking is being "
                     "driven by what was asked for and not by which video the model "
                     "likes. That is the difference between planning and a beauty "
                     "contest, and it is the only part of this that a single ranking "
                     "could not have shown."
                     if distinct > 1 else
                     "<b>The choice did NOT move.</b> The same candidate won both goals, "
                     "including the one it should have lost. Read that as the scorer "
                     "ignoring the goal and ranking on something else; the selection "
                     "step is not doing what this task claims until this line changes.")
                )
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.CHOOSE_EXPLAINER), do_flush=True
        )

    note = ""

    # ── Imagine: four futures from one frame ────────────────────────────────────
    for i, (name, actions, _) in enumerate(variants):
        _paint("Imagining", f"candidate {i + 1} of {len(variants)}: {name}", rows)
        frames, secs = world.rollout_chunk(
            pipe, meta, actions, steps=steps, guidance=guidance, seed=seed
        )
        clips[name] = frames
        mp4s[name] = media.encode(frames, fps=fps)
        gen_secs += secs
        paint("Imagining", f"candidate {i + 1} of {len(variants)}: {name}")
        log.info("candidate %s: %.0fs", name, secs)

    # ── Score: the same futures against opposite goals ──────────────────────────
    pipe = None
    still = world.release()
    rows.append(("Imagining time", f"{gen_secs / 60:.1f} min for {len(clips)} futures"))
    rows.append(("Handover", f"generation surface released, {still:.1f} GiB still held"))
    paint("Handing over to the scorer", "loading the understanding surface")

    model, processor = world.load_reasoner(repo)
    score_secs = 0.0

    # The motion bucket is measured ONCE per clip and reused as the tiebreak for both
    # goals. Re-asking per goal would let the same clip get two different motion
    # readings, and then a winner could change because the measurement wandered rather
    # than because the goal did, which would destroy the control.
    for name, frames in clips.items():
        text, s = world.ask(model, processor, Q_MOTION, video=frames,
                            frames=len(frames), max_new_tokens=128)
        score_secs += s
        buckets[name] = world.parse_choice(text, MOTION_CHOICES)

    for goal, question, target in CHOOSE_GOALS:
        scored[goal] = {}
        for i, (name, _, _) in enumerate(variants):
            paint("Scoring against the goal", f"'{goal}': {i + 1} of {len(variants)}")
            answer, s = world.ask(model, processor, question, video=clips[name],
                                  frames=len(clips[name]), max_new_tokens=128)
            score_secs += s
            scored[goal][name] = {
                "answer": answer,
                "achieves": world.parse_choice(answer, ("yes", "no")) == 0,
            }
        # Rank: candidates that satisfy the goal first, then whichever sits closest to
        # the motion bucket the goal wants. The bucket is a coarse tiebreak and nothing
        # more, which is all `blind` established it can carry.
        ranked = sorted(
            (n for n, _, _ in variants),
            key=lambda n: (
                not scored[goal][n]["achieves"],
                abs((buckets[n] if buckets[n] is not None else 1) - target),
            ),
        )
        winners[goal] = ranked[0] if scored[goal][ranked[0]]["achieves"] else None
        log.info("goal %r -> %s", goal, winners[goal])
        paint("Scored", f"'{goal}' resolved")

    rows.append(("Scoring time", f"{score_secs:.0f}s"))
    paint("Finished", "")

    picked = [w for w in winners.values() if w]
    moved = len(set(picked)) > 1
    log.info("choose: winners %s, choice moved with the goal: %s", winners, moved)
    return {
        "candidates": [n for n, _, _ in variants],
        "buckets": buckets,
        "winners": winners,
        "achieves": {g: {n: v["achieves"] for n, v in d.items()} for g, d in scored.items()},
        "choice_moved_with_goal": moved,
        "imagine_seconds": round(gen_secs, 1),
        "score_seconds": round(score_secs, 1),
        "note": note,
    }


@gpu_env.task(report=True)
async def robust(
    repo: str = NANO,
    seeds: int = 3,
    steps: int = 35,
    guidance: float = 6.0,
) -> dict:
    """Does the counterfactual ordering survive a change of seed?

    `counterfact` is the result this repo leads with, and the argument it makes is an
    ORDERING: commanding the arm to hold its pose predicts a robot that nearly stops
    (0.41), doubling the motion overshoots the real demonstration (2.68), and the
    recorded and reversed sequences land in between. That ordering, not any single
    clip, is the evidence the action channel does work.

    It has been measured at exactly one seed. It reproduces to two decimal places across
    separate runs on separate days, which is a fact about determinism and says nothing
    at all about whether the ordering is a property of the actions or a property of seed
    0. Those look identical from inside a single run, and only one of them is worth
    saying out loud.

    So this re-runs all four variants at several seeds and asks whether the ranking
    holds. The measurement is inter-frame motion, the same one `counterfact` reports.
    A variant's absolute number is allowed to move; what must not move is which variant
    sits above which, because that is the entire claim.

    Cheap for what it protects: `seeds` x 4 rollouts, about 50 seconds each, from one
    model load.
    """
    rows = [("Model", repo), ("Task", "is the counterfactual ordering seed-independent?")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    chunk = meta["chunks"][0]
    fps = int(meta.get("fps", 10))
    variants = world.counterfactuals(chunk)
    names = [n for n, _, _ in variants]
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({chunk.shape[-1]}-D actions)"),
        ("Grid", f"{len(names)} variants x {seeds} seeds = {len(names) * seeds} rollouts"),
        ("Claim under test", "held < reversed < recorded < amplified, by inter-frame motion"),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    # motion[seed][variant]
    motion: dict[int, dict[str, float]] = {}
    strips: dict[int, dict[str, str]] = {}
    total = 0.0
    note = ""

    # The ordering seed 0 produced, and the one the README states. Stated here as data
    # rather than recomputed from the run, so a run in which seed 0 itself came out
    # differently is visible as a disagreement instead of silently redefining the claim.
    CLAIMED = ["held", "reversed", "recorded", "amplified"]

    def paint(stage: str, detail: str) -> None:
        body = ""
        if motion:
            body += reports._heading("Inter-frame motion, every variant at every seed")
            header = "".join(
                f'<th style="padding:6px 10px;text-align:right;color:#888;'
                f'font-weight:normal;">seed {s}</th>' for s in sorted(motion)
            )
            table = (f'<tr><th style="padding:6px 10px;text-align:left;color:#888;'
                     f'font-weight:normal;">variant</th>{header}'
                     f'<th style="padding:6px 10px;text-align:right;color:#888;'
                     f'font-weight:normal;">spread</th></tr>')
            for name in names:
                vals = [motion[s][name] for s in sorted(motion) if name in motion[s]]
                if not vals:
                    continue
                cells = "".join(
                    f'<td style="padding:6px 10px;text-align:right;color:#00b894;">'
                    f'{motion[s][name]:.2f}</td>' for s in sorted(motion) if name in motion[s]
                )
                table += (f'<tr><td style="padding:6px 10px;">{name}</td>{cells}'
                          f'<td style="padding:6px 10px;text-align:right;color:#fdcb6e;">'
                          f'{max(vals) - min(vals):.2f}</td></tr>')
            body += (f'<table style="border-collapse:collapse;font-family:monospace;'
                     f'font-size:13px;color:#ccc;background:#0f0f23;border-radius:8px;'
                     f'padding:8px;">{table}</table>')

            body += reports._heading("The ordering each seed produced")
            agree = 0
            complete = [s for s in sorted(motion) if len(motion[s]) == len(names)]
            for s in complete:
                order = world.rank_string(motion[s])
                ok = order.split(" < ") == CLAIMED
                agree += ok
                body += reports.note(
                    f"seed {s}: <b>{order}</b> "
                    + ("&#10003; matches the claim" if ok else
                       "&#10007; DIFFERENT from the claim")
                )
            if complete:
                body += reports.note(
                    (f"<b>{agree} of {len(complete)} seeds reproduce the ordering.</b> "
                     + ("The claim is a property of the actions, not of a lucky seed, "
                        "which is what makes it safe to state as a result."
                        if agree == len(complete) else
                        "The ordering is NOT seed-independent. The absolute numbers in "
                        "`counterfact` are still real, but the ranking should be stated "
                        "as what one seed produced rather than as a property of the "
                        "model, until a larger sweep says otherwise."))
                )
            for s in sorted(strips):
                body += reports._heading(f"seed {s}")
                body += reports.side_by_side(
                    [(n, strips[s][n]) for n in names if n in strips[s]]
                )
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.ROBUST_EXPLAINER), do_flush=True
        )

    for s in range(seeds):
        motion[s], strips[s] = {}, {}
        for i, (name, actions, _) in enumerate(variants):
            paint("Sweeping seeds", f"seed {s}, variant {i + 1} of {len(names)}: {name}")
            try:
                frames, secs = world.rollout_chunk(
                    pipe, meta, actions, steps=steps, guidance=guidance, seed=s
                )
            except Exception as exc:  # noqa: BLE001
                note = f"seed {s} variant {name} failed and the sweep moved on: {exc}"
                log.warning(note)
                continue
            total += secs
            motion[s][name] = world.clip_stats(frames)["motion"]
            # Strips rather than videos, and that is the point of the task rather than a
            # saving: twelve embedded clips is the report that renders blank, and what
            # this task is actually about is twelve numbers.
            strips[s][name] = media.strip(frames, count=4, width=92)
            log.info("seed %s %s: motion %.2f (%.0fs)", s, name, motion[s][name], secs)
        paint("Sweeping seeds", f"seed {s} complete")

    rows.append(("Sweep time", f"{total / 60:.1f} min"))
    paint("Finished", "")

    complete = [s for s in sorted(motion) if len(motion[s]) == len(names)]
    orders = {s: world.rank_string(motion[s]).split(" < ") for s in complete}
    agree = sum(1 for s in complete if orders[s] == CLAIMED)
    log.info("robust: %s/%s seeds reproduce %s", agree, len(complete), CLAIMED)
    return {
        "seeds": len(complete),
        "motion": {s: {n: round(v, 3) for n, v in motion[s].items()} for s in complete},
        "orderings": {s: " < ".join(o) for s, o in orders.items()},
        "claimed": " < ".join(CLAIMED),
        "seeds_reproducing": agree,
        "ordering_is_seed_independent": bool(complete) and agree == len(complete),
        "seconds": round(total, 1),
        "note": note,
    }


@gpu_env.task(report=True)
async def detail(
    repo: str = NANO,
    examples: int = 2,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Is the round-trip tax lost detail, or is generated video simply out of distribution?

    `cycle` measures that inverse dynamics reads a dreamed clip about three times worse
    than the real clip of the same event. It does not say why, and the README's first
    reading of it -- that the generated video is softer so there is less to read -- was a
    hypothesis dressed as a conclusion. The alternative has the opposite consequence:

      lost detail        the generated video carries less information. You wait for a
                         better generator, or you generate at higher resolution.
      distribution gap   the generated video is perfectly legible and simply is not
                         CAMERA video, and the inverse model was trained on footage. You
                         fine-tune the labeller, which is cheap and possible today.

    The control is to degrade the REAL clip to the dreamed clip's sharpness and hand that
    to the same inverse model. Three readings of the same event:

      real          the floor
      blurred real  the same footage carrying the dreamed clip's amount of detail
      dreamed       generated from the same actions

    If blurred-real lands on dreamed, detail explains the tax. If blurred-real stays down
    near the floor while dreamed sits three times higher, detail does NOT explain it.

    One honest limit, and it goes in the report rather than in a footnote: blur matches
    one axis. Generated video also differs in temporal coherence and in its artifacts, so
    a blurred real clip controls for spatial detail alone. Landing on the dreamed error
    is therefore good evidence that detail is sufficient; missing it is evidence for a
    distribution gap rather than proof of one.

    Four inverse passes and one generation per example, from a single model load.
    """
    rows = [("Model", repo), ("Task", "is the round-trip tax detail, or distribution?")]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)
    path = world.snapshot(repo)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    results: list[dict] = []
    blocks: list[str] = []
    note = ""

    def paint(stage: str, detail_line: str) -> None:
        body = "".join(blocks)
        if results:
            body += reports._heading("Three readings of the same event")
            bars: list[tuple[str, float]] = []
            for r in results:
                bars.append((f"ex {r['index']}: real", r["real"]))
                bars.append((f"ex {r['index']}: real, blurred to match", r["blurred"]))
                bars.append((f"ex {r['index']}: dreamed", r["dream"]))
            body += reports.bars(bars, caption=(
                "Recovered-action error on the moving channel, against the actions the "
                "real robot executed. The middle bar is the control: real footage "
                "carrying exactly as much spatial detail as the generated clip. Where it "
                "lands is the whole experiment."
            ))
            verdicts = [r["verdict"] for r in results]
            agreed = len(set(verdicts)) == 1
            body += reports.note(
                (f"<b>{verdicts[0]}</b>" if agreed else
                 "<b>The examples disagree: " + ", ".join(verdicts) + ".</b> With two "
                 "samples that is a reason to run more rather than to pick one.")
            )
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail_line)] if detail_line else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.DETAIL_EXPLAINER), do_flush=True
        )

    for i in range(examples):
        _paint("Reading the answer key", f"inverse example {i}", rows)
        try:
            example = world.load_inverse_example(path, i)
        except Exception as exc:  # noqa: BLE001
            note = f"Example {i} could not be read and the run moved on: {exc}"
            log.warning(note)
            break
        real_frames, truth = example["frames"], example["truth"]
        fps = int(example["fps"])

        paint("Inverse dynamics on the real clip", f"example {i}: 1 of 3")
        a_real, _ = world.invert(pipe, example, steps=steps, guidance=guidance, seed=seed)
        e_real = world.action_error(truth, a_real)

        paint("Dreaming the same actions", f"example {i}: generating")
        fwd = world.forward_meta_from_inverse(example)
        dreamed, t_fwd = world.rollout_chunk(
            pipe, fwd, truth, steps=steps, guidance=guidance, seed=seed
        )

        paint("Inverse dynamics on the dreamed clip", f"example {i}: 2 of 3")
        dream_example = dict(example); dream_example["frames"] = dreamed
        a_dream, _ = world.invert(pipe, dream_example, steps=steps, guidance=guidance, seed=seed)
        e_dream = world.action_error(truth, a_dream)

        # The control. Match the dreamed clip's sharpness using the real footage.
        s_real, s_dream = world.sharpness(real_frames), world.sharpness(dreamed)
        blurred, radius = world.blur_to_match(real_frames, s_dream)
        s_blur = world.sharpness(blurred)

        paint("Inverse dynamics on the blurred real clip", f"example {i}: 3 of 3")
        blur_example = dict(example); blur_example["frames"] = blurred
        a_blur, _ = world.invert(pipe, blur_example, steps=steps, guidance=guidance, seed=seed)
        e_blur = world.action_error(truth, a_blur)

        real, dream, blur = (e_real["mae_moving"], e_dream["mae_moving"], e_blur["mae_moving"])
        # How much of the real-to-dreamed gap does blur alone reproduce? Above ~70% and
        # detail is a sufficient explanation; below ~30% and it is clearly not the story.
        span = dream - real
        explained = (blur - real) / span if abs(span) > 1e-9 else 0.0
        verdict = (
            "Lost detail explains the tax: blurring the real clip to the same sharpness "
            "reproduces most of the gap." if explained >= 0.7 else
            "Lost detail does NOT explain the tax. Real footage carrying the same amount "
            "of detail is still read far better than generated video, which points at a "
            "distribution gap rather than an information one, and at fine-tuning the "
            "labeller rather than waiting for a better generator."
            if explained <= 0.3 else
            "Mixed: blur reproduces part of the gap but not most of it, so detail is one "
            "contributor among others."
        )

        dims = e_real["moving_dims"] or [0]
        blocks.append(
            reports._heading(f"Example {i}: the same event, three ways")
            + reports.side_by_side([
                ("real", media.video_html(media.encode(real_frames, fps=fps),
                                          f"sharpness {s_real:.0f}", max_width=300,
                                          autoplay=False)),
                (f"real, blurred (radius {radius:.2f})",
                 media.video_html(media.encode(blurred, fps=fps),
                                  f"sharpness {s_blur:.0f}", max_width=300, autoplay=False)),
                ("dreamed", media.video_html(media.encode(dreamed, fps=fps),
                                             f"sharpness {s_dream:.0f}, {t_fwd:.0f}s",
                                             max_width=300, autoplay=False)),
            ])
            + reports.note(
                f"Recovered-action error on channel(s) {dims}: real <b>{real:.4f}</b>, "
                f"blurred real <b>{blur:.4f}</b>, dreamed <b>{dream:.4f}</b>. "
                f"Blur reproduces <b>{explained * 100:.0f}%</b> of the real-to-dreamed gap."
            )
            + reports.action_traces(
                truth.tolist(), a_blur.tolist(), dims=dims,
                labels=["executed", "read off the blurred real clip"],
                caption="The control. Real footage, dreamed-clip sharpness.",
            )
        )
        results.append({
            "index": i, "real": real, "blurred": blur, "dream": dream,
            "explained": round(explained, 3), "verdict": verdict,
            "sharpness": {"real": round(s_real, 1), "blurred": round(s_blur, 1),
                          "dreamed": round(s_dream, 1)},
            "blur_radius": round(radius, 3),
        })
        log.info("detail %s: real %.4f blur %.4f dream %.4f -> blur explains %.0f%%",
                 i, real, blur, dream, explained * 100)
        paint("Control complete", f"example {i} done")

    paint("Finished", "")
    log.info("detail: %s", [(r["index"], r["explained"]) for r in results])
    return {
        "examples": len(results),
        "real": [round(r["real"], 5) for r in results],
        "blurred": [round(r["blurred"], 5) for r in results],
        "dreamed": [round(r["dream"], 5) for r in results],
        "gap_explained_by_blur": [r["explained"] for r in results],
        "sharpness": [r["sharpness"] for r in results],
        "note": note,
    }


# Burned-in captions are drawn with PIL's bitmap font at ~6px per character, on frames
# 560px wide. Past this the plate runs off the edge and the sentence is cut mid-word by
# the frame rather than by us. The full text always goes in the report.
CAPTION_CHARS = 54


@gpu_env.task(report=True)
async def odyssey(
    repo: str = NANO,
    steps_count: int = 40,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
    film_every: int = 5,
) -> dict:
    """An agent acting inside a dreamed world, for hours, narrating itself as it goes.

    This is the closest thing in this repo to what `topics/dreamerv3` does, and the
    thing a video generator cannot do at all. At every step the model is handed one
    frame and a task, and `policy` mode denoises the action channel and the pixel
    channel TOGETHER: it decides what the robot should do and renders the consequence
    of having done it, in one pass. Take the last frame of that and hand it back, and
    you have a closed loop in which nobody supplies actions, nobody supplies a physics
    engine, and nobody supplies an environment. The agent and the universe are the same
    16B network.

    Dreamer does this too, and the contrast is the point of running them side by side.
    Dreamer's world model is small, learned from scratch from one agent's own
    experience, and works only in the environment it was trained on; its actor is
    trained by backpropagating through imagined rollouts. Cosmos's world model is
    pretrained on the physical world, works on embodiments it was never shown by you,
    and its "actor" is not trained at all -- it is the same denoiser, run with the
    action tokens left noisy.

    Both experts stay resident, which this file said for a day was impossible. Measured:
    29.6 GiB for the generation expert, 46.0 GiB for both, 50.5 GiB peak with a VAE
    decode running while the language expert sits there. That is what makes narration
    LIVE rather than a second pass: captioning each step as it happens would otherwise
    cost four minutes of model loading per step.

    The artifact is one continuous film with the model's own description of each step
    burned into it. Watching the caption drift away from the picture is the finding, and
    it is the same content-drift result `judge` measures, except you feel it instead of
    reading it off an axis.

    Long by design. 40 steps is about 75 minutes; 120 is an overnight run and about
    three minutes of continuous film.
    """
    rows = [
        ("Model", repo),
        ("Task", "closed-loop agent inside its own dream, narrated live"),
        ("Plan", f"{steps_count} policy steps, both experts resident"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    fps = int(meta.get("fps", 10))
    rows += [
        ("Embodiment", f"{meta['domain_name']} ({meta['chunks'].shape[-1]}-D actions)"),
        ("Task given to the agent", meta["prompt"]),
    ]

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)
    _paint("Loading the understanding surface", "Both stay resident; ~46 GiB total.", rows)
    model, processor = world.load_reasoner(repo)

    import torch

    rows.append(("Both resident", f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated"))

    pieces: list[list] = []       # captioned frames, for the film
    captions: list[str] = []
    stats: list[dict] = []
    effort: list[float] = []      # mean |action| per step: is the agent still trying?
    overlap: list[float] = []
    timings: list[float] = []
    note = ""
    film: dict = {"mp4": b"", "at": 0}

    def repaint(stage: str, detail: str, rebuild: bool = False) -> None:
        """Repaint from what exists. The film is expensive, so rebuild it on a schedule.

        `replace`, never `log`: log() appends, and a run that repaints forty times would
        stack forty copies of a growing video into one report.
        """
        if not pieces:
            _paint(stage, detail, rows)
            return
        stitched = [f for seg in pieces for f in seg]
        if rebuild or not film["mp4"]:
            for target_width in (480, 384, 288, 224):
                candidate = media.encode(media.downscale(stitched, target_width),
                                         fps=fps, crf=30)
                if len(candidate) < 14 * 2**20:
                    break
            film["mp4"], film["at"] = candidate, len(pieces)

        live = list(rows) + [
            ("Progress", f"step {len(pieces)} of {steps_count}"),
            ("Film so far", f"{len(stitched)} frames, {len(stitched) / fps:.1f}s at {fps} fps"),
            ("Elapsed", f"{sum(timings) / 60:.1f} min"),
        ]
        body = reports._heading("The dream so far, with the dreamer's own subtitles")
        body += media.video_html(
            film["mp4"],
            f"steps 0 to {film['at'] - 1}"
            + ("" if film["at"] == len(pieces) else
               f"; the caption list and charts below are current to step {len(pieces)}, "
               f"the film is rebuilt every {film_every}"),
        )
        body += media.strip(stitched, count=10, width=110)

        body += reports._heading("The newest step")
        body += media.video_html(media.encode(pieces[-1], fps=fps, crf=28),
                                 f"step {len(pieces) - 1}", max_width=380, autoplay=False)
        body += reports.quote(captions[-1], "the model, watching what it just did")

        body += reports._heading("The story it told itself")
        story = ""
        for i, cap in enumerate(captions):
            drift = f" <span style='color:#888;'>(overlap {overlap[i]:.2f})</span>" if i else ""
            story += (f"<li style='margin:0 0 4px;'><span style='color:#888;'>step "
                      f"{i}</span> {cap}{drift}</li>")
        body += (f"<ol style='font-family:monospace;font-size:12px;color:#ccc;"
                 f"line-height:1.6;background:#0f0f23;padding:14px 14px 14px 34px;"
                 f"border-radius:8px;'>{story}</ol>")

        if len(stats) > 1:
            body += reports.metric_lines(
                {
                    "action magnitude the agent chose": effort,
                    "description overlap with step 0": overlap,
                    "sharpness (variance of Laplacian)": [s["sharpness"] for s in stats],
                    "inter-frame motion": [s["motion"] for s in stats],
                },
                caption=(
                    "One point per step. Action magnitude is the one with no counterpart "
                    "in `horizon`: it is what the agent DECIDED to do, not what the video "
                    "did, so it falling toward zero means the policy has given up rather "
                    "than the renderer having frozen. Those two look identical in the "
                    "video and are completely different failures."
                ),
            )
        if note:
            body += reports.note(note)
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.ODYSSEY_EXPLAINER), do_flush=True
        )

    frame = meta["first_frame"]
    for i in range(steps_count):
        repaint("Acting inside the dream", f"step {i + 1} of {steps_count}",
                rebuild=(i % film_every == 0))
        step_meta = dict(meta)
        step_meta["first_frame"] = frame
        try:
            seg, actions, secs = world.policy(
                pipe, step_meta, steps=steps, guidance=guidance, seed=seed + i
            )
        except Exception as exc:  # noqa: BLE001
            note = (f"Step {i} failed after {len(pieces)} good steps, and the run kept "
                    f"them rather than raising: {exc}")
            log.warning(note)
            break
        timings.append(secs)

        # Narrate BEFORE captioning, and from the raw frames. Showing the model a clip
        # with a subtitle already burned into it would let its own previous description
        # leak into the next one, which would fake exactly the continuity this task is
        # trying to measure.
        caption, _ = world.ask(
            model, processor, Q_DESCRIBE, video=seg, frames=len(seg), max_new_tokens=96
        )
        captions.append(caption)
        overlap.append(1.0 if len(captions) == 1
                       else round(world.description_overlap(captions[0], caption), 3))
        stats.append(world.clip_stats(seg))
        effort.append(float(actions.abs().mean()))

        burned = media.label_frames(seg, f"{i}: {caption[:CAPTION_CHARS]}")
        # Drop the reproduced conditioning frame on every step after the first, or the
        # seam is a duplicated frame in the middle of the film.
        pieces.append(burned if i == 0 else burned[1:])
        # The RAW last frame, never the captioned one. Feeding back a frame with a
        # subtitle painted on it would ask the model to treat its own text overlay as
        # part of the world, and it would faithfully start rendering it.
        frame = seg[-1]
        log.info("odyssey %s: %.0fs effort %.4f overlap %.2f | %s",
                 i, secs, effort[-1], overlap[-1], caption[:70])

    repaint("The dream ends", "", rebuild=True)

    total = sum(len(p) for p in pieces)
    log.info("odyssey: %s steps, %s frames, effort %.4f -> %.4f, overlap %.2f -> %.2f",
             len(pieces), total,
             effort[0] if effort else 0, effort[-1] if effort else 0,
             overlap[0] if overlap else 0, overlap[-1] if overlap else 0)
    return {
        "steps": len(pieces),
        "frames": total,
        "seconds_of_film": round(total / fps, 1),
        "captions": captions,
        "effort": [round(e, 5) for e in effort],
        "overlap": overlap,
        "sharpness": [round(s["sharpness"], 1) for s in stats],
        "motion": [round(s["motion"], 3) for s in stats],
        "wall_clock_minutes": round(sum(timings) / 60, 1),
        "note": note,
    }


# Each entry is a real LeRobot dataset paired with the embodiment name Cosmos knows it
# by, so the conditioning frame, the actions and the ground-truth continuation all come
# from the same moment of the same real episode. `pusht` is in here as the NEGATIVE
# control: it is known to fail, and a survey whose method cannot reproduce a known
# failure is not measuring anything.
EMBODIMENTS: tuple[dict, ...] = (
    {
        "name": "droid_lerobot",
        "dataset": "lerobot/droid_100",
        "video": "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4",
        "size": (320, 192),      # tier 256, the 1.667 aspect bin
        "tier": 256,
        "fps": 15,
        "convert": "axis_angle_to_6d",   # 7-D dataset -> 10-D Cosmos
        "prompt": "A Franka robot arm manipulates objects on a tabletop.",
        "note": "A real Franka. The embodiment a MuJoCo bridge would target.",
    },
    {
        "name": "pusht",
        "dataset": "lerobot/pusht",
        "video": "videos/observation.image/chunk-000/file-000.mp4",
        "size": (256, 256),      # tier 256, the 1.0 aspect bin
        "tier": 256,
        "fps": 10,
        "convert": None,         # already 2-D
        "prompt": "A robot pusher moves a T-shaped block across the table.",
        "note": "NEGATIVE CONTROL. Known to fail; if this passes, the method is broken.",
    },
)


@gpu_env.task(report=True)
async def embodiments(
    repo: str = NANO,
    frames: int = 17,
    start: int = 30,
    steps: int = 20,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Which embodiments does this checkpoint actually know, as opposed to accept?

    `_EMBODIMENT_TO_RAW_ACTION_DIM` lists fifteen domains. That is what the ARCHITECTURE
    accepts. It is not what these weights were trained on, and the gap is not academic:
    `pusht` runs without error, generates faster than anything else here, and produces a
    photorealistic robot arm on a wooden desk instead of PushT. Nothing in NVIDIA's
    documentation says which of the fifteen Nano actually saw.

    The test is scene retention against REAL in-domain data. Every entry pulls a
    conditioning frame, the actions beside it, and the ground-truth continuation from the
    same moment of the same LeRobot episode, so "what should have happened next" is not
    a matter of opinion. The model gets the frame and the real actions; the report shows
    its continuation next to the real one.

    An earlier version of this scored ACTION SENSITIVITY instead -- flat action versus
    large action, per embodiment, on a borrowed conditioning frame -- and it failed
    completely. All fifteen scored a divergence of 16 to 30, including `pusht`, because
    arbitrary actions on a foreign frame make every domain wander. That is why `pusht` is
    kept here as a negative control rather than dropped: a survey that cannot reproduce a
    known failure is not measuring anything, and without it the first method looked fine.
    """
    rows = [
        ("Model", repo),
        ("Task", "which embodiments does the checkpoint know, not merely accept?"),
        ("Method", "scene retention against real in-domain LeRobot data"),
        ("Cases", ", ".join(e["name"] for e in EMBODIMENTS)),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading", "Streaming a 16B transformer to the device in BF16.", rows)
    pipe = world.load(repo)

    import torch
    from diffusers import CosmosActionCondition

    results: list[dict] = []
    blocks: list[str] = []
    note = ""

    def paint(stage: str, detail: str) -> None:
        body = "".join(blocks)
        if results:
            body += reports._heading("Verdicts")
            table = [(r["name"],
                      ("scene KEPT" if r["kept"] else "scene ABANDONED")
                      + f" (luminance {r['luminance_generated']} vs {r['luminance_real']} real, "
                        f"ratio {r['ratio']})")
                     for r in results]
            for name, line in table:
                body += reports.note(f"<b>{name}</b>: {line}")
            body += reports.note(
                "Mean luminance against the real continuation, which is crude and is the "
                "number that separated the two cases by a mile. It catches a model that "
                "swaps a bright synthetic scene for a dark photoreal one; it would NOT "
                "catch one that wandered somewhere of similar brightness. The frame "
                "strips are the evidence, this is what makes it sortable."
            )
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.EMBODIMENTS_EXPLAINER),
            do_flush=True,
        )

    for i, spec in enumerate(EMBODIMENTS):
        paint("Fetching real data", f"{spec['name']} from {spec['dataset']}")
        try:
            sample = world.load_lerobot_sample(
                spec["dataset"], spec["video"],
                start=start, count=frames, size=tuple(spec["size"]),
            )
        except Exception as exc:  # noqa: BLE001
            note = f"{spec['name']}: could not fetch {spec['dataset']}: {exc}"
            log.warning(note)
            continue

        acts = sample["actions"]
        if spec["convert"] == "axis_angle_to_6d":
            acts = world.axis_angle_to_6d(acts)
        actions = torch.tensor(acts, dtype=torch.float32)

        paint("Generating", f"{spec['name']} ({i + 1} of {len(EMBODIMENTS)})")
        try:
            out = pipe(
                prompt=spec["prompt"],
                action=CosmosActionCondition(
                    mode="forward_dynamics",
                    chunk_size=int(actions.shape[0]),
                    domain_name=spec["name"],
                    resolution_tier=int(spec["tier"]),
                    raw_actions=actions,
                    image=sample["first_frame"],
                    view_point="ego_view",
                ),
                fps=float(spec["fps"]),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator().manual_seed(seed),
                use_system_prompt=False,
                enable_safety_check=False,
            )
        except Exception as exc:  # noqa: BLE001
            note = f"{spec['name']} failed to generate and the survey moved on: {exc}"
            log.warning(note)
            continue

        gen = list(out.video)
        real = sample["frames"]
        verdict = world.scene_retained(gen, real)
        verdict.update({"name": spec["name"], "dataset": spec["dataset"],
                        "action_dim": int(actions.shape[-1])})
        results.append(verdict)

        mark = "scene KEPT" if verdict["kept"] else "scene ABANDONED"
        blocks.append(
            reports._heading(f"{spec['name']}: {mark}")
            + reports.note(spec["note"])
            + reports.side_by_side([
                ("Cosmos, given the real frame and the real actions",
                 media.video_html(media.encode(gen, fps=int(spec["fps"])),
                                  f"{len(gen)} frames", max_width=340, autoplay=False)
                 + media.strip(gen, count=4, width=82)),
                ("what actually happened next",
                 media.video_html(media.encode(real, fps=int(spec["fps"])),
                                  f"{len(real)} real frames", max_width=340, autoplay=False)
                 + media.strip(real, count=4, width=82)),
            ])
            + reports.note(
                f"action dim {actions.shape[-1]}"
                + (" (converted from 7-D axis-angle)" if spec["convert"] else "")
                + f" | luminance {verdict['luminance_generated']} generated vs "
                  f"{verdict['luminance_real']} real | motion "
                  f"{verdict['motion_generated']} vs {verdict['motion_real']} | "
                  f"divergence {verdict['divergence']}"
            )
        )
        log.info("embodiment %s: kept=%s %s", spec["name"], verdict["kept"], verdict)
        paint("Generating", f"{spec['name']} done")

    paint("Finished", "")
    known = [r["name"] for r in results if r["kept"]]
    broken = [r["name"] for r in results if not r["kept"]]
    # The negative control has to fail, or the method is not measuring anything.
    control_ok = "pusht" in broken
    log.info("embodiments: known=%s broken=%s control_ok=%s", known, broken, control_ok)
    return {
        "known": known,
        "broken": broken,
        "negative_control_failed_as_expected": control_ok,
        "detail": results,
        "aliases": {"droid_lerobot": "robomind-franka",
                    "agibotworld": "agibot_gear_gripper / agibot_gear_gripper_ext"},
        "note": note,
    }


@gpu_env.task(report=True)
async def train(
    repo: str = NANO,
    chunks: int = 24,
    held_out: int = 8,
    epochs: int = 60,
    steps: int = 20,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Stage 6 at last: train a policy on dreams, and test it on reality.

    Every other task here sets this up and stops short of it. `dream` builds the data
    engine, `cycle` prices a dreamed label at about 3x, `detail` shows that price is a
    distribution gap rather than lost detail. None of them trained anything, so none of
    them could say whether the output is USABLE, which is the only question that matters
    to anyone deciding whether to run a synthetic data programme.

    The design is a paired comparison with one variable:

        real set    (real DROID frame_t,     action_t)
        dream set   (Cosmos dreamed frame_t, action_t)   <- the SAME actions
        test set    held-out REAL frames and actions

    Both training sets carry identical labels, because each dream was generated FROM the
    actions it is labelled with. So the label noise `cycle` measured is deliberately
    pinned at zero, the two networks are identical, the seed is shared, and the only
    thing that differs is the pixels. Whatever gap appears on the real held-out test set
    is the pixel domain gap by itself.

    `droid_lerobot` because `embodiments` verified it: real Franka data, scene retained,
    and it is the one embodiment here a simulator could later be bridged to. Running this
    on `pusht` would train a policy on a photorealistic robot arm that has nothing to do
    with the labels.

    The report paints a live training curve, both dream and real, plus the actual frames
    each policy learned from, because "we trained something" with no picture of the data
    is the claim easiest to get wrong and hardest to check.
    """
    rows = [
        ("Model", repo),
        ("Task", "train on dreams, test on reality"),
        ("Embodiment", "droid_lerobot (verified by `embodiments`)"),
        ("Design", "identical labels, identical net, identical seed; only the pixels differ"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    import torch

    SPEC = EMBODIMENTS[0]          # droid_lerobot
    CHUNK = 16
    size = tuple(SPEC["size"])

    # ── Real data: one contiguous slice per chunk, so each dream has a real twin ──
    _paint("Fetching real DROID data", f"{chunks + held_out} windows", rows)
    real_frames, real_actions10 = [], []
    windows = []
    for c in range(chunks + held_out):
        start = 30 + c * (CHUNK + 1)
        try:
            sample = world.load_lerobot_sample(
                SPEC["dataset"], SPEC["video"], start=start, count=CHUNK + 1, size=size
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("window %s unavailable: %s", c, exc)
            break
        a10 = world.axis_angle_to_6d(sample["actions"])
        windows.append({"start": start, "sample": sample, "a10": a10})
        real_frames.extend(sample["frames"][:CHUNK])
        real_actions10.append(a10)

    n_train = min(chunks, max(len(windows) - held_out, 1))
    rows.append(("Windows", f"{len(windows)} of {CHUNK} frames; {n_train} train, "
                            f"{len(windows) - n_train} held out"))

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)

    from diffusers import CosmosActionCondition

    history = {"real": {"train": [], "val": []}, "dream": {"train": [], "val": []}}
    finals: dict = {}
    # Bound before the painter is ever called; the real value arrives once the
    # training sets exist, and paint() guards on it being None until then.
    baseline: dict | None = None

    def paint(stage: str, detail: str) -> None:
        body = reports._heading("What each policy is learning from")
        cells = []
        if dreamed_previews:
            cells.append(("Cosmos dreamed these (the dream set)",
                          media.video_html(dreamed_previews[0], "one training window",
                                           max_width=320, autoplay=False)))
        cells.append(("the real footage of the same window (the real set)",
                      media.video_html(
                          media.encode(windows[0]["sample"]["frames"][:CHUNK],
                                       fps=int(SPEC["fps"])),
                          "identical actions, identical labels", max_width=320,
                          autoplay=False)))
        body += reports.side_by_side(cells)
        body += reports.note(
            "Same window, same actions, same labels. One policy sees the left column and "
            "one sees the right, and both are tested on real footage they have never seen."
        )

        series = {}
        if history["real"]["val"]:
            series["held-out real MAE, policy trained on REAL"] = history["real"]["val"]
        if history["dream"]["val"]:
            series["held-out real MAE, policy trained on DREAMS"] = history["dream"]["val"]
        if series:
            body += reports._heading("Training, live")
            body += reports.metric_lines(series, caption=(
                "Both curves are error on the SAME held-out real frames, so they are "
                "directly comparable. The dream curve sitting above the real one is the "
                "pixel domain gap, and where it flattens is how much of that gap more "
                "training will not close."
            ))
        if finals and baseline is not None:
            body += reports._heading("The answer")
            body += reports.bars(
                [("predict the mean (learns nothing)", baseline["mae_moving"])]
                + [(k, v["mae_moving"]) for k, v in finals.items()],
                caption=("Mean absolute error on held-out REAL frame pairs, on the action "
                         "channels that actually move. Lower is better. The first bar is "
                         "a constant predictor that ignores the images entirely: any "
                         "policy above it has learned nothing, and the ratio between the "
                         "other two is then meaningless. An earlier run of this task "
                         "produced a tidy 1.10x between two policies that were BOTH worse "
                         "than this bar.")
            )
            beat = [k for k, v in finals.items() if v["mae_moving"] < baseline["mae_moving"]]
            body += reports.note(
                f"Policies that beat the trivial baseline: <b>{', '.join(beat) or 'NONE'}</b>."
                + ("" if beat else " Nothing below can be read as a result about synthetic "
                                   "data; it is a statement about an undertrained network.")
            )
            if len(beat) == 2 and "real" in finals and "dream" in finals:
                ratio = finals["dream"]["mae_moving"] / (finals["real"]["mae_moving"] or 1e-9)
                body += reports.note(
                    f"Training on dreams instead of real footage costs <b>{ratio:.2f}x</b> "
                    f"on held-out real data ({finals['real']['mae_moving']:.4f} to "
                    f"{finals['dream']['mae_moving']:.4f}). Labels were identical in both "
                    f"runs by construction, so this is the pixel gap alone, with the "
                    f"label noise `cycle` measured deliberately held at zero."
                )
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.TRAIN_EXPLAINER), do_flush=True
        )



    # ── Dream each TRAINING window from its own real actions ────────────────────
    dream_frames: list = []
    dreamed_previews: list[bytes] = []
    gen_secs = 0.0
    for c in range(n_train):
        _paint("Dreaming the training set", f"window {c + 1} of {n_train}", rows)
        w = windows[c]
        try:
            t0 = time.monotonic()
            out = pipe(
                prompt=SPEC["prompt"],
                action=CosmosActionCondition(
                    mode="forward_dynamics", chunk_size=CHUNK,
                    domain_name=SPEC["name"], resolution_tier=int(SPEC["tier"]),
                    raw_actions=torch.tensor(w["a10"], dtype=torch.float32),
                    image=w["sample"]["first_frame"], view_point="ego_view",
                ),
                fps=float(SPEC["fps"]), num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator().manual_seed(seed + c),
                use_system_prompt=False, enable_safety_check=False,
            )
            gen_secs += time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
            log.warning("dream %s failed: %s", c, exc)
            break
        frames = list(out.video)[:CHUNK]
        dream_frames.extend(frames)
        if len(dreamed_previews) < 3:
            dreamed_previews.append(media.encode(frames, fps=int(SPEC["fps"])))
        log.info("dreamed window %s (%s frames)", c, len(frames))
        # As soon as ONE dream exists the report can show it. The first version only
        # called the rich paint after the whole dreaming loop, so the report sat as
        # plain text for twenty minutes while the interesting part -- what the policy
        # is about to learn from -- already existed and was being withheld.
        paint("Dreaming the training set", f"window {c + 1} of {n_train}")

    n_dreamed = len(dream_frames) // CHUNK
    rows.append(("Dreaming time", f"{gen_secs / 60:.1f} min for {n_dreamed} windows"))

    # Hand the pool back. Training needs very little, and holding 30 GB of diffusion
    # expert through it is 30 GB nothing else on this box can have.
    pipe = None
    still = world.release()
    rows.append(("Handover", f"generation surface released, {still:.1f} GiB still held"))

    # ── Assemble the three sets ─────────────────────────────────────────────────
    import numpy as np

    def acts_for(lo, hi):
        return torch.tensor(np.concatenate([windows[i]["a10"] for i in range(lo, hi)]),
                            dtype=torch.float32)

    # FRAME PAIRS, not single frames. DROID's action is a velocity command and velocity
    # is not observable in one static image; the first version of this task used single
    # frames and neither policy beat a constant predictor, which is the correct outcome
    # for an ill-posed input rather than a fact about synthetic data.
    all_actions = acts_for(0, n_dreamed)
    x_real, keep = bc.pairs_to_tensor(real_frames[:n_dreamed * CHUNK], CHUNK)
    x_dream, _ = bc.pairs_to_tensor(dream_frames[:n_dreamed * CHUNK], CHUNK)
    y_train = all_actions[keep]

    test_lo, test_hi = n_train, len(windows)
    test_frames = [f for i in range(test_lo, test_hi)
                   for f in windows[i]["sample"]["frames"][:CHUNK]]
    x_test, keep_t = bc.pairs_to_tensor(test_frames, CHUNK)
    y_test = acts_for(test_lo, test_hi)[keep_t]

    baseline = bc.mean_baseline(y_train, y_test)
    rows += [
        ("Sets", f"train {len(x_real)} frame pairs each, test {len(x_test)} real pairs"),
        ("Trivial baseline", f"predicting the training mean scores "
                             f"{baseline['mae_moving']:.4f} on the test set"),
    ]

    paint("Training", "starting")

    for label, x in (("real", x_real), ("dream", x_dream)):
        def on_epoch(i, tr, va, _label=label):
            history[_label]["train"].append(tr)
            history[_label]["val"].append(va)
            if i % 5 == 0 or i == epochs - 1:
                paint("Training", f"{_label} policy, epoch {i + 1} of {epochs}")

        model, _ = bc.train_policy(
            x, y_train, epochs=epochs, seed=seed,
            val=(x_test, y_test), on_epoch=on_epoch,
        )
        finals[label] = {k: v for k, v in bc.evaluate(model, x_test, y_test).items()
                         if k not in ("pred", "truth")}
        log.info("policy trained on %s: %s", label, finals[label])
        paint("Training", f"{label} policy done")

    paint("Finished", "")
    ratio = (finals["dream"]["mae_moving"] / finals["real"]["mae_moving"]
             if finals.get("real", {}).get("mae_moving") else None)
    log.info("train: real %s dream %s ratio %s",
             finals.get("real", {}).get("mae_moving"),
             finals.get("dream", {}).get("mae_moving"), ratio)
    return {
        "windows_trained": n_dreamed,
        "trivial_baseline": {k: round(v, 5) for k, v in baseline.items()},
        "beats_baseline": [k for k, v in finals.items()
                           if v.get("mae_moving", 9e9) < baseline["mae_moving"]],
        "train_frames": int(len(x_real)),
        "test_frames": int(len(x_test)),
        "real": {k: v for k, v in finals.get("real", {}).items() if k != "per_dim"},
        "dream": {k: v for k, v in finals.get("dream", {}).items() if k != "per_dim"},
        "dream_cost_ratio": round(ratio, 3) if ratio else None,
        "dreaming_seconds": round(gen_secs, 1),
    }


@orch_env.task(report=True)
async def access(repo: str = "") -> dict:
    """Can this cluster's HF_TOKEN actually reach the gated Cosmos repos?

    CPU-only and seconds long, because the question is about credentials rather than
    compute. It exists because the answer is genuinely hard to get from a laptop: the
    token lives as a Flyte secret on the devbox and not in anyone's shell, so an
    anonymous check from the host returns GatedRepoError whether or not the licence has
    been accepted. Those two states look identical and mean opposite things.

    Run it with the secret mounted, which is opt-in for the reason config.py explains
    (declaring a secret the cluster does not hold fails the pod at ADMISSION):

        COSMOS_HF_SECRET=1 flyte run pipeline.py access

    Probes a real weight file rather than the README, because HF serves repo metadata
    and even some files on a gated repo to anyone; only fetching something real tells
    you the gate is open.
    """
    import os

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    token = os.environ.get("HF_TOKEN")
    # The whole roadmap, so one run answers "what else should I accept?" definitively.
    # A real file per repo, never the README: the Hub serves metadata and some files on
    # gated repos to anyone, so only fetching something substantial proves the gate.
    targets = [
        # --- in use, or the next thing to build ---
        ("nvidia/Cosmos3-Nano", "config.json", "IN USE, ungated: the control row"),
        ("nvidia/Cosmos-Transfer2.5-2B",
         "general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt",
         "IN USE by `restyle`: sim2real, the Isaac bridge"),
        ("nvidia/Cosmos-1.0-Guardrail", "config.json",
         "REQUIRED by Transfer; its pipeline raises without a guardrail"),
        ("google/siglip-so400m-patch14-384", "config.json",
         "the guardrail's vision encoder, open, 3.5 GB"),
        # --- reasoning ---
        ("nvidia/Cosmos-Reason2-2B", "config.json", "dedicated reasoner, 4.9 GB"),
        ("nvidia/Cosmos-Reason2-8B", "config.json", "dedicated reasoner, 17.5 GB"),
        ("nvidia/Cosmos-Reason2-32B", "config.json", "dedicated reasoner, 64 GB"),
        # --- embodiment / policy, the closed-loop roadmap ---
        ("nvidia/Cosmos3-Edge", "config.json", "4B: the size-vs-quality axis, never run"),
        ("nvidia/Cosmos3-Edge-Policy-DROID", "config.json", "a real VLA policy, 9.2 GB"),
        ("nvidia/Cosmos3-Nano-Policy-DROID", "config.json", "a real VLA policy, 32.9 GB"),
        # --- the Predict line Transfer is designed to pair with ---
        ("nvidia/Cosmos-Predict2.5-2B", "README.md", "Predict 2.5, pairs with Transfer"),
        ("nvidia/Cosmos-Predict2.5-14B", "README.md", "Predict 2.5, larger"),
        # --- measurement and extras ---
        ("nvidia/Cosmos-Embed1-448p", "config.json",
         "video embeddings: a better drift metric than pixel diff"),
        ("nvidia/GEN3C-Cosmos-7B", "README.md", "camera-controlled 3D-consistent generation"),
        ("nvidia/Cosmos-H-Dreams", "README.md", "real-time streaming world model (surgical)"),
    ]
    if repo:
        targets = [(repo, "config.json", "requested")]

    results, rows = {}, [("Token present in pod", "yes" if token else "NO")]
    for name, path, why in targets:
        try:
            meta = get_hf_file_metadata(hf_hub_url(name, path), token=token)
            results[name] = {"open": True, "bytes": meta.size}
            rows.append((name, f"OPEN ({(meta.size or 0) / 1e6:.1f} MB) - {why}"))
        except Exception as exc:  # noqa: BLE001
            results[name] = {"open": False, "error": type(exc).__name__}
            rows.append((name, f"{type(exc).__name__} - {why}"))
        log.info("access %s: %s", name, results[name])

    gated = [n for n, v in results.items() if not v["open"]]
    control = results.get("nvidia/Cosmos3-Nano", {}).get("open")
    body = reports.note(
        "The last row is the control: `Cosmos3-Nano` is ungated, so if it fails the "
        "problem is the token or the network rather than any licence."
        if control is not False else
        "<b>The ungated control FAILED.</b> This is not a licence problem; the pod "
        "cannot reach the Hub at all."
    )
    flyte.report.replace(
        reports.final_html("Gated repo access", rows, body, reports.ACCESS_EXPLAINER),
        do_flush=True,
    )
    if gated:
        body += reports._heading("Still to accept")
        body += reports.note(
            "Open each page and accept the licence, then re-run this task:<br>"
            + "<br>".join(f'&nbsp;&nbsp;https://huggingface.co/{n}' for n in gated)
        )
        flyte.report.replace(
            reports.final_html("Gated repo access", rows, body, reports.ACCESS_EXPLAINER),
            do_flush=True,
        )
    return {"token_in_pod": bool(token), "open": [n for n, v in results.items() if v["open"]],
            "still_gated": gated, "results": results, "control_ungated_ok": control}


# Cosmos Transfer 2.5. A different model from everything above, and a different job:
# Predict GENERATES a world, Transfer RESTYLES one you already have, preserving the
# geometry and motion while pushing the rendering toward realism.
#
# Loaded off revision branches of one repo rather than separate repos, which is easy to
# get wrong: the weights at the repo root are raw .pt checkpoints, and only the
# `diffusers/...` branches carry a loadable layout.
TRANSFER_REPO = "nvidia/Cosmos-Transfer2.5-2B"
TRANSFER_BRANCH = "diffusers/general"
CONTROL_BRANCHES = {
    "edge": "diffusers/controlnet/general/edge",
    "depth": "diffusers/controlnet/general/depth",
    "seg": "diffusers/controlnet/general/seg",
    "blur": "diffusers/controlnet/general/blur",
}

# Two sources, and which one is the DEFAULT matters more than it looks.
#
# `mujoco` is the real use case: a three-dimensional, physically correct scene that looks
# synthetic, which is exactly what Isaac Sim and Omniverse produce and exactly what the
# sim2real pipeline has to convert. The simulator also commands the actions, which is why
# this path has no label tax at all.
#
# `pusht` was the original default and it was the wrong choice, kept only as a contrast.
# It is the clip `embodiments` proved Cosmos PREDICT cannot handle, which made a tidy
# story, but it is a flat 2D diagram: a white field with coloured polygons and no
# photorealistic counterpart for Transfer to move it toward. Transfer restyles a render;
# it is not a renderer for a schematic.
RESTYLE_SOURCES = {
    "mujoco": {
        "kind": "sim",
        "size": (640, 384),
        "fps": 10,
        "prompt": (
            "A red robotic end effector pushes a blue plastic block across a white "
            "laboratory bench toward a marked target. Photorealistic, shot on a "
            "high-end camera, soft overhead studio lighting, realistic material "
            "textures, subtle shadows and shallow depth of field."
        ),
        "note": "A 3D simulator render: physically correct, and it looks synthetic. "
                "This is the shape of input Transfer exists for.",
    },
    # The fidelity control. Everything above asks what Transfer does to a SYNTHETIC scene;
    # this asks what it does to one that is already real. If a real lab comes back as the
    # same real lab, then the PushT result was about the input being two-dimensional and
    # not about Transfer being loose with scenes in general. Without this row that stays
    # an assumption.
    "droid": {
        "kind": "lerobot",
        "dataset": "lerobot/droid_100",
        "video": "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4",
        "size": (320, 192),
        "fps": 15,
        "prompt": (
            "A Franka robot arm on a laboratory bench, cluttered workspace, natural "
            "indoor lighting, photorealistic, shot on a high-end camera."
        ),
        "note": "Already-real footage. The CONTROL: does Transfer preserve a scene that "
                "needs no restyling, or does it wander even here?",
    },
    "pusht": {
        "kind": "lerobot",
        "dataset": "lerobot/pusht",
        "video": "videos/observation.image/chunk-000/file-000.mp4",
        "size": (256, 256),
        "fps": 10,
        "prompt": (
            "A robotic manipulator pushes a T-shaped wooden block across a white "
            "laboratory bench. Overhead studio lighting, photorealistic, crisp shadows "
            "and realistic material textures."
        ),
        "note": "A flat 2D diagram. Kept as a CONTRAST: it is the clip Predict could not "
                "handle, and it is also not what Transfer is for, because there is no "
                "realistic counterpart for a schematic to be moved toward.",
    },
}


@gpu_env.task(report=True)
async def restyle(
    source: str = "mujoco",
    control: str = "edge",
    sweep: str = "",
    offload: bool = False,
    frames: int = 29,
    steps: int = 25,
    guidance: float = 4.0,
    scale: float = 1.0,
    seed: int = 0,
) -> dict:
    """Cosmos Transfer: keep the geometry, change the rendering. The sim2real half.

    Every other task in this file uses Cosmos PREDICT, which generates a world. This uses
    Cosmos TRANSFER, which restyles one you already have. That distinction is the whole
    reason the sim2real pipeline works:

        Isaac Sim / Omniverse  ->  structured output (depth, segmentation, edges)
                               ->  Cosmos Transfer  ->  photorealistic video
                               ->  train a policy on it

    and it sidesteps every problem the Predict path ran into. The simulator supplies the
    actions, so there is **no label tax** at all (`cycle` measured 3x for IDM-recovered
    labels). You never leave the simulator's robot, so there is **no embodiment
    mismatch**. And Transfer is handed the geometry as a control signal rather than
    asked to infer it from an out-of-distribution frame, so the failure that killed
    `pusht` under Predict cannot happen in the same way.

    Which is why the input here is PushT. `embodiments` proved Predict cannot handle it:
    given a white background and flat coloured shapes it threw the scene away and
    rendered a photorealistic arm on a wooden desk. Same clip, different model, and the
    question is whether the T-block is still a T-block in the same place.

    Note this is a genuinely separate checkpoint, not another door into Cosmos3-Nano, and
    a larger one than "2B" suggests: the transformer is 2B but the text encoder is
    Qwen2.5-VL-7B. Gated, so it needs COSMOS_HF_SECRET=1 and an accepted licence; run
    `access` first if unsure.

    **The guardrail is not optional here, and that is a licence term rather than a
    dependency accident.** Every Predict task in this file passes
    `enable_safety_checker=False`, which is sanctioned: the flag exists and NVIDIA's own
    runner exposes `--disable-safety-checker`. `Cosmos2_5_TransferPipeline` has no such
    flag, constructs a checker unconditionally, and RAISES if one is absent with a
    message citing the NVIDIA Open Model License. So `cosmos_guardrail` is installed in
    the image rather than stubbed out, and these runs DO have a content guardrail even
    though the rest of this file does not. It pulls `nvidia/Cosmos-1.0-Guardrail` (gated,
    so it needs the same licence acceptance) and `google/siglip-so400m-patch14-384`
    (open, 3.5 GB) the first time it runs.
    """
    import numpy as np
    import torch

    spec = RESTYLE_SOURCES[source]
    rows = [
        ("Model", f"{TRANSFER_REPO} ({TRANSFER_BRANCH})"),
        ("Control", f"{control} ({CONTROL_BRANCHES.get(control, '?')})"),
        ("Source", f"{source}: {spec['note']}"),
        ("Task", "keep the geometry, change the rendering"),
    ]
    _paint("Building the source clip", source, rows)

    sim_actions, sim_depth = None, None
    if spec["kind"] == "sim":
        src_frames, sim_actions, sim_depth = mjc.rollout(
            frames=frames, width=spec["size"][0], height=spec["size"][1]
        )
        rows.append(("Ground-truth actions", f"{sim_actions.shape} commanded by the "
                                             f"simulator, so no label tax at all"))
    else:
        sample = world.load_lerobot_sample(
            spec["dataset"], spec["video"], start=30, count=frames,
            size=tuple(spec["size"]),
        )
        src_frames = sample["frames"]
    source_clip = src_frames
    fps = int(spec["fps"])

    # Canny on each frame. This is the "structured simulation output" stand-in: an Isaac
    # pipeline would hand over real depth or segmentation buffers instead of deriving
    # edges from pixels, and would be strictly better for it, but the control signal
    # enters the model at exactly the same place.
    from PIL import Image

    if control == "depth" and sim_depth is not None:
        # The simulator's OWN depth buffer, which is the entire reason to source from a
        # simulator rather than a video. Everything below is the fallback, and measuring
        # it across three sources is what made the case for this branch: Canny edges were
        # too sparse on a 2D scene (the model invented geometry), dominated by furniture
        # on a badly framed one, and so dense on a cluttered real lab that Transfer simply
        # reproduced the edge map as grey lines. Depth is what Isaac and Omniverse hand
        # over, and a simulator can give it exactly instead of inferring it from pixels.
        controls = sim_depth
    else:
        import cv2

        edges = [
            cv2.Canny(cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR), 100, 200)
            for f in source_clip
        ]
        stacked = torch.from_numpy(np.stack(edges)[None]).expand(3, -1, -1, -1)
        controls = [Image.fromarray(x.numpy()) for x in stacked.permute(1, 2, 3, 0)]
    rows.append(("Control frames", f"{len(controls)} {control} maps at {controls[0].size}"
                                   + (" from the simulator's own buffer"
                                      if control == "depth" and sim_depth is not None
                                      else " derived from pixels")))
    # The prompt belongs in the report, not just in the source. It is the single biggest
    # lever on what Transfer produces: the control signal fixes the geometry, and the
    # prompt decides what material, lighting and setting that geometry is rendered as.
    # Reading an output without seeing the prompt that shaped it is guesswork.
    rows.append(("Prompt", spec["prompt"]))
    rows.append(("Negative prompt", prompts.NEGATIVE[:160] + "..."))

    variants: list[dict] = []

    def paint(stage: str, detail: str, out=None) -> None:
        cells = [
            ("the synthetic source", media.video_html(
                media.encode(source_clip, fps=fps), "as the simulator drew it",
                max_width=300, autoplay=False) + media.strip(source_clip, count=3, width=90)),
            (f"the {control} control signal", media.video_html(
                media.encode(controls, fps=fps), "the geometry, handed to the model",
                max_width=300, autoplay=False) + media.strip(controls, count=3, width=90)),
        ]
        for v in (variants if variants else []):
            cells.append((
                f"Transfer, {v['steps']} steps",
                media.video_html(media.encode(v["frames"], fps=fps),
                                 f"{v['seconds']:.0f}s ({v['per_step']}s/step), "
                                 f"motion {v['motion']}, sharpness {v['sharpness']}",
                                 max_width=300, autoplay=False)
                + media.strip(v["frames"], count=3, width=90)))
        body = reports._heading("Same geometry, different rendering")
        body += reports.side_by_side(cells)
        if variants:
            body += reports.note(
                "The question is not whether the output looks good. It is whether the "
                "T-block is still a T-block, still in the same place, still moving the "
                "same way. Transfer preserving that is what makes the simulator's "
                "actions valid labels for the restyled video, and it is the entire "
                "reason this path has no label tax."
            )
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.RESTYLE_EXPLAINER), do_flush=True
        )

    paint("Loading Cosmos Transfer", "a separate checkpoint; 2B transformer + a 7B text encoder")

    from diffusers import AutoModel, Cosmos2_5_TransferPipeline

    world.guard_memory(40.0)
    controlnet = AutoModel.from_pretrained(
        TRANSFER_REPO, revision=CONTROL_BRANCHES[control], torch_dtype=torch.bfloat16
    )
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(
        TRANSFER_REPO, revision=TRANSFER_BRANCH, controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )
    # Offload is OFF by default, which is a correction. The pipeline declares an offload
    # sequence (text_encoder -> transformer -> controlnet -> vae) and turning it on looked
    # like free prudence, but it cost 1336s for 29 frames at 256x256, about 46 s/frame,
    # almost all of it swapping the 7B Qwen2.5-VL text encoder across the PCIe bus on
    # every step. Transfer is far smaller than Cosmos3-Nano and this pod has 96 GiB, so
    # there is nothing to be prudent about. Pass --offload to put it back if a bigger
    # resolution ever needs it.
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    rows.append(("Resident", f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB after load"))

    # A sweep generates one clip per step count FROM ONE MODEL LOAD, which is the only
    # reason a throughput question is affordable to ask: the load is minutes and separate
    # runs would pay it every time. `--sweep 6,10,25` is the shape.
    #
    # The question it exists to answer is whether the sim2real pipeline is usable at all.
    # Transfer at 25 steps takes about 30 minutes for 29 frames, and a behaviour-cloning
    # dataset wants 50+ clips. That is 25 hours, which is not a plan. If 8 steps looks as
    # good as 25 it becomes an overnight job instead.
    schedule = [int(x) for x in sweep.split(",") if x.strip()] or [steps]
    for n in schedule:
        paint("Restyling", f"{n} steps, control scale {scale}")
        t0 = time.monotonic()
        result = pipe(
            controls=controls,
            controls_conditioning_scale=scale,
            prompt=spec["prompt"],
            negative_prompt=prompts.NEGATIVE,
            num_frames=len(controls),
            num_inference_steps=n,
            guidance_scale=guidance,
            generator=torch.Generator().manual_seed(seed),
        )
        secs = time.monotonic() - t0
        frames_out = list(result.frames[0])
        st = world.clip_stats(frames_out)
        variants.append({"steps": n, "seconds": round(secs, 1),
                         "per_step": round(secs / n, 1), "frames": frames_out,
                         "motion": round(st["motion"], 2),
                         "sharpness": round(st["sharpness"], 1)})
        log.info("restyle %s steps: %.0fs (%.1fs/step) motion %.2f sharp %.0f",
                 n, secs, secs / n, st["motion"], st["sharpness"])
        paint("Restyling", f"{n} steps done")

    out = variants[-1]["frames"]
    secs = sum(v["seconds"] for v in variants)
    rows.append(("Restyle time", f"{secs / 60:.1f} min total for "
                                 f"{len(schedule)} setting(s) at {len(out)} frames"))

    stats_in, stats_out = world.clip_stats(source_clip), world.clip_stats(out)
    rows.append(("Motion preserved", f"{stats_in['motion']:.2f} source -> "
                                     f"{stats_out['motion']:.2f} restyled"))
    paint("Finished", "", out=out)

    log.info("restyle: %s frames in %.0fs, motion %.2f -> %.2f",
             len(out), secs, stats_in["motion"], stats_out["motion"])
    return {
        "source": source,
        "control": control,
        "sweep": [{k: v for k, v in d.items() if k != "frames"} for d in variants],
        "ground_truth_actions": None if sim_actions is None else list(sim_actions.shape),
        "frames": len(out),
        "seconds": round(secs, 1),
        "motion_source": round(stats_in["motion"], 3),
        "motion_restyled": round(stats_out["motion"], 3),
        "sharpness_source": round(stats_in["sharpness"], 1),
        "sharpness_restyled": round(stats_out["sharpness"], 1),
    }


# Standing questions, asked of every window of a long video. Written as yes/no on
# purpose: `blind`, `judge` and `dream` all found the same thing independently, which is
# that this model saturates on numeric scales and is reliable on categorical answers.
# An alerting system wants a decision anyway, not a score.
WATCH_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("arm_holding",
     "Is the robot arm gripping or holding an object? Answer yes or no, then one short sentence."),
    ("human",
     "Is a human hand or person visible? Answer yes or no, then one short sentence."),
    ("moving",
     "Is anything in the scene moving? Answer yes or no, then one short sentence."),
)

WATCH_SOURCE = {
    "dataset": "lerobot/droid_100",
    "video": "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4",
    "size": (320, 192),
    "fps": 15,
}


@gpu_env.task(report=True)
async def watch(
    repo: str = NANO,
    windows: int = 20,
    window: int = 16,
    start: int = 30,
    stride: int = 0,
    summarise: bool = True,
) -> dict:
    """A video agent: standing questions asked of a long recording, with timestamps.

    The one use case on NVIDIA's list that nothing else here touches, and the one that
    has nothing to do with robots being controlled. Factories, warehouses, traffic
    cameras, smart spaces: "alert me if a forklift enters the pedestrian area",
    "summarise what happened on camera 8", "find every occurrence of someone entering
    this zone". It needed a task rather than a capability, because the understanding
    surface already answers questions about arbitrary video in a second or two.

    No generation at all. Only the 16 GB understanding expert is loaded, which is why
    this is the cheapest task in the file: a window is one forward pass per question, so
    twenty windows and three questions is about ninety seconds of GPU after the load.

    The questions are yes/no by design. `blind`, `judge` and `dream` each independently
    found that this model saturates on 1-10 scales and holds up on categorical answers,
    and an alerting system wants a decision rather than a score anyway.

    Two honest limits, both visible in the report. The model sees `window` frames at a
    time and has no memory across windows, so it cannot answer "is this the same person
    as before". And a yes/no with no confidence attached means a false positive looks
    exactly like a true one; the frames behind every alert are shown so a human can
    check, which is the only reason to trust the timeline at all.
    """
    stride = stride or window
    rows = [
        ("Model", f"{repo}, understanding surface only"),
        ("Task", "standing questions over a long recording"),
        ("Source", WATCH_SOURCE["dataset"]),
        ("Coverage", f"{windows} windows of {window} frames, stride {stride}"),
        ("Questions", ", ".join(k for k, _ in WATCH_QUESTIONS)),
    ]
    _paint("Fetching the recording", WATCH_SOURCE["dataset"], rows)

    fps = int(WATCH_SOURCE["fps"])
    clips: list[dict] = []
    for i in range(windows):
        begin = start + i * stride
        try:
            sample = world.load_lerobot_sample(
                WATCH_SOURCE["dataset"], WATCH_SOURCE["video"],
                start=begin, count=window, size=tuple(WATCH_SOURCE["size"]),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("window %s unavailable: %s", i, exc)
            break
        clips.append({"index": i, "start": begin, "frames": sample["frames"],
                      "t": begin / fps})
    rows.append(("Fetched", f"{len(clips)} windows spanning "
                            f"{(clips[-1]['t'] - clips[0]['t']):.1f}s of footage"
                            if clips else "nothing"))

    world.guard_memory(world.REASONER_NEEDS_GIB)
    _paint("Loading the understanding surface", "~16 GB; no generation expert at all", rows)
    model, processor = world.load_reasoner(repo)

    alerts: dict[str, list] = {k: [] for k, _ in WATCH_QUESTIONS}
    answers: dict[int, dict] = {}
    summary = ""
    ask_secs = 0.0

    def timeline(key: str) -> str:
        """One cell per window, filled when that question answered yes."""
        cells = ""
        for c in clips:
            hit = c["index"] in [a["index"] for a in alerts[key]]
            known = c["index"] in answers
            colour = "#00b894" if hit else ("#2d3436" if known else "#1a1a2e")
            cells += (f'<span title="t={c["t"]:.1f}s" style="display:inline-block;'
                      f'width:16px;height:16px;margin:1px;border-radius:3px;'
                      f'background:{colour};"></span>')
        return (f'<div style="font-family:monospace;font-size:12px;color:#ccc;'
                f'margin:0 0 8px;"><span style="display:inline-block;width:120px;'
                f'color:#888;">{key}</span>{cells}'
                f'<span style="color:#fdcb6e;padding-left:10px;">'
                f'{len(alerts[key])} hit(s)</span></div>')

    def paint(stage: str, detail: str) -> None:
        body = reports._heading("The recording")
        if clips:
            whole = [f for c in clips for f in c["frames"]]
            body += media.video_html(media.encode(whole, fps=fps, crf=30),
                                     f"{len(whole)} frames, {len(whole) / fps:.1f}s",
                                     max_width=480)
            body += media.strip(whole, count=10, width=100)
        if answers:
            body += reports._heading("Alert timeline")
            for key, _ in WATCH_QUESTIONS:
                body += timeline(key)
            body += reports.note(
                "One cell per window, left to right in time. Green is a yes, dark grey a "
                "no, empty not yet looked at. Hover for the timestamp."
            )
            # The frames behind the alerts, because a yes/no with no confidence attached
            # is only trustworthy if a human can check it in one glance.
            for key, _ in WATCH_QUESTIONS:
                if not alerts[key]:
                    continue
                body += reports._heading(f"What fired '{key}'")
                cells = []
                for a in alerts[key][:3]:
                    c = clips[a["index"]]
                    cells.append((f"t = {c['t']:.1f}s",
                                  media.strip(c["frames"], count=3, width=96)
                                  + reports.quote(a["answer"], "the model")))
                body += reports.side_by_side(cells)
        if summary:
            body += reports._heading("What happened, in one paragraph")
            body += reports.quote(summary, "asked to summarise the whole recording")
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.WATCH_EXPLAINER), do_flush=True
        )

    paint("Watching", "starting")

    for c in clips:
        answers[c["index"]] = {}
        for key, question in WATCH_QUESTIONS:
            text, secs = world.ask(model, processor, question, video=c["frames"],
                                   frames=len(c["frames"]), max_new_tokens=96)
            ask_secs += secs
            yes = world.parse_choice(text, ("yes", "no")) == 0
            answers[c["index"]][key] = {"yes": yes, "answer": text}
            if yes:
                alerts[key].append({"index": c["index"], "t": round(c["t"], 2),
                                    "answer": text})
        log.info("window %s (t=%.1fs): %s", c["index"], c["t"],
                 {k: v["yes"] for k, v in answers[c["index"]].items()})
        paint("Watching", f"window {c['index'] + 1} of {len(clips)}")

    if summarise and clips:
        paint("Summarising", "one pass over frames sampled across the whole recording")
        spread = [f for c in clips for f in world.sample_frames(c["frames"], 2)]
        summary, secs = world.ask(
            model, processor,
            "Summarise everything that happens in this recording in one short paragraph.",
            video=spread, frames=min(len(spread), 16), max_new_tokens=256,
        )
        ask_secs += secs

    rows.append(("Watch time", f"{ask_secs:.0f}s of GPU for "
                               f"{len(clips) * len(WATCH_QUESTIONS)} questions"))
    paint("Finished", "")

    log.info("watch: %s", {k: len(v) for k, v in alerts.items()})
    return {
        "windows": len(clips),
        "seconds_of_footage": round(len(clips) * window / fps, 1),
        "alerts": {k: [a["t"] for a in v] for k, v in alerts.items()},
        "hit_counts": {k: len(v) for k, v in alerts.items()},
        "summary": summary,
        "ask_seconds": round(ask_secs, 1),
    }


@gpu_env.task(report=True)
async def sizes(
    small: str = EDGE,
    large: str = NANO,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Does the 4B model still know that actions matter, or only the 16B one?

    The size axis this repo has never run, and the question is sharper than "is the small
    one worse". `counterfact` is the control that decides whether a checkpoint is usable
    as a simulator at all: same conditioning frame, same seed, four action sequences, and
    the predicted motion has to come out in the order the actions describe. A model that
    quietly ignores its action channel produces beautiful video and is worthless for
    generating training data.

    So this runs that exact control on BOTH checkpoints. `Cosmos3-Edge` is 4B and 9.2 GB
    against Nano's 16B and 33 GB, and if the ordering survives at 4B then bulk data
    generation can run on a quarter of the weights. That is not a cosmetic saving: it is
    the difference between the throughput wall this pipeline keeps hitting and not.

    Judge it on the ORDERING, not on which clips look nicer. `held` must sit at the
    bottom and `amplified` at the top on both, and `robust` already established that the
    ordering is a property of the actions rather than of one seed.
    """
    rows = [
        ("Small", f"{small} (4B)"),
        ("Large", f"{large} (16B)"),
        ("Test", "the `counterfact` control, run on both"),
        ("Claim", "held < reversed < recorded < amplified, by inter-frame motion"),
    ]
    _paint("Fetching weights", "the small checkpoint is a fresh ~9 GB download.", rows)

    path = world.snapshot(large)
    meta = world.load_action_example(path)
    chunk = meta["chunks"][0]
    fps = int(meta.get("fps", 10))
    variants = world.counterfactuals(chunk)
    names = [n for n, _, _ in variants]

    CLAIMED = ["held", "reversed", "recorded", "amplified"]
    results: dict[str, dict] = {}
    strips: dict[str, dict] = {}
    note = ""

    def paint(stage: str, detail: str) -> None:
        body = ""
        for repo in [r for r in (small, large) if r in results]:
            motion = results[repo]["motion"]
            order = world.rank_string(motion)
            ok = order.split(" < ") == CLAIMED
            body += reports._heading(f"{repo} ({results[repo]['params']})")
            body += reports.note(
                f"ordering: <b>{order}</b> "
                + ("&#10003; matches the claim" if ok else "&#10007; DIFFERENT")
                + f" &nbsp;|&nbsp; load {results[repo]['load_min']:.1f} min, "
                  f"{results[repo]['gen_min']:.1f} min for {len(motion)} rollouts"
            )
            body += reports.side_by_side([
                (f"{n}<br/><span style='color:#888;'>motion {motion[n]:.2f}</span>",
                 strips[repo][n]) for n in names if n in strips[repo]
            ])
        if len(results) == 2:
            body += reports._heading("Same comparison, both checkpoints")
            bars = []
            for repo in (small, large):
                tag = "4B" if repo == small else "16B"
                for n in names:
                    if n in results[repo]["motion"]:
                        bars.append((f"{tag} {n}", results[repo]["motion"][n]))
            body += reports.bars(bars, caption=(
                "Inter-frame motion per variant, both checkpoints. Read the ORDER within "
                "each block, not the absolute heights: the two models need not agree on "
                "how much motion a scene has, only on which action sequence produces more "
                "of it. If the small model preserves the ordering, bulk generation can run "
                "on a quarter of the weights."
            ))
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.SIZES_EXPLAINER), do_flush=True
        )

    for repo in (small, large):
        # Sequentially, releasing between: 9 GB and 33 GB co-resident is affordable but
        # pointless, and one at a time keeps the peak where the rest of this file keeps it.
        _paint("Loading", f"{repo}", rows)
        try:
            t0 = time.monotonic()
            pipe = world.load(repo)
            load_min = (time.monotonic() - t0) / 60
        except Exception as exc:  # noqa: BLE001
            note = f"{repo} failed to load and the comparison moved on: {exc}"
            log.warning(note)
            continue

        import torch

        params = f"{sum(p.numel() for p in pipe.transformer.parameters()) / 1e9:.1f}B"
        motion, gen = {}, 0.0
        for i, (name, actions, _) in enumerate(variants):
            paint("Generating", f"{repo}: variant {i + 1} of {len(names)}")
            try:
                frames, secs = world.rollout_chunk(
                    pipe, meta, actions, steps=steps, guidance=guidance, seed=seed
                )
            except Exception as exc:  # noqa: BLE001
                note = f"{repo}/{name} failed: {exc}"
                log.warning(note)
                continue
            gen += secs
            motion[name] = world.clip_stats(frames)["motion"]
            # Video, not just a strip. A comparison of two checkpoints that shows only
            # still frames is asking the reader to take the motion numbers on faith, and
            # motion is the entire quantity being compared.
            mp4 = media.encode(frames, fps=fps, crf=26)
            strips.setdefault(repo, {})[name] = (
                media.video_html(mp4, f"{len(frames)} frames, {secs:.0f}s",
                                 max_width=300, autoplay=False)
                + media.strip(frames, count=4, width=92)
            )
            log.info("%s %s: motion %.2f (%.0fs)", repo, name, motion[name], secs)
        results[repo] = {"motion": motion, "params": params,
                         "load_min": load_min, "gen_min": gen / 60}
        pipe = None
        world.release()
        paint("Generating", f"{repo} done")

    paint("Finished", "")

    verdict = {}
    for repo, r in results.items():
        if len(r["motion"]) == len(names):
            verdict[repo] = world.rank_string(r["motion"]).split(" < ") == CLAIMED
    log.info("sizes: %s", verdict)
    return {
        "checkpoints": {r: {"params": v["params"],
                            "motion": {k: round(m, 3) for k, m in v["motion"].items()},
                            "ordering": world.rank_string(v["motion"]),
                            "load_minutes": round(v["load_min"], 2),
                            "generate_minutes": round(v["gen_min"], 2)}
                        for r, v in results.items()},
        "preserves_ordering": verdict,
        "note": note,
    }


EMBED_REPO = "nvidia/Cosmos-Embed1-448p"


@gpu_env.task(report=True)
async def embed(
    repo: str = NANO,
    chunks: int = 1,
    segments: int = 14,
    frames: int = 45,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int = 0,
) -> dict:
    """Measure long-horizon drift with a real video embedder instead of word overlap.

    `judge` found the most interesting result in this repo: over 21 segments a rollout's
    CONTENT drifts at segment 7 while its PHYSICS holds until 17. But it measured content
    drift with `world.description_overlap`, a Jaccard set overlap between two sentences
    the model wrote. That is transparent and checkable, and it is also crude in a way
    worth being honest about: a segment scoring 0.857 rather than 1.0 can just be the
    word "arm" appearing or not.

    `Cosmos-Embed1` is a joint video-text embedder built for exactly this, 2.4 GB, and it
    gives a cosine distance between two clips directly from pixels, with no sentence in
    between. So this rolls a world forward and measures the drift BOTH ways on the same
    segments: embedding distance from segment 0, and the word overlap `judge` uses.

    Two outcomes and both are worth having. If the curves agree, the cheap metric is
    validated and `judge`'s headline result stands on something firmer than a word count.
    If they disagree, the embedder is the better instrument and says so.

    Three models in sequence: the generation expert to produce the rollout, the
    understanding expert to describe each segment, then the embedder. Released between,
    which is why this is minutes of loading on top of the generation.
    """
    rows = [
        ("Model", repo),
        ("Embedder", f"{EMBED_REPO} (2.4 GB)"),
        ("Task", "is `judge`'s word-overlap drift metric measuring the right thing?"),
        ("Plan", f"{chunks} action chunk(s) + {segments} continuation(s), scored twice"),
    ]
    _paint("Fetching weights", f"{repo} from the shared model cache.", rows)

    path = world.snapshot(repo)
    meta = world.load_action_example(path)
    fps = int(meta.get("fps", 10))

    guard = world.guard_memory()
    rows.append(("GPU", guard))
    _paint("Loading the generation surface", "Streaming a 16B transformer in BF16.", rows)
    pipe = world.load(repo)

    pieces: list[list] = []
    clips: list[bytes] = []
    labels: list[str] = []
    stats: list[dict] = []
    captions: list[str] = []
    overlap: list[float] = []
    cosine: list[float] = []
    note = ""

    def paint(stage: str, detail: str) -> None:
        if not pieces:
            _paint(stage, detail, rows)
            return
        body = reports._heading("The rollout")
        cells = []
        for i, mp4 in enumerate(clips[-4:]):
            j = len(clips) - min(4, len(clips)) + i
            blk = media.video_html(mp4, labels[j], max_width=300, autoplay=False)
            blk += media.strip(pieces[j], count=3, width=90)
            if j < len(captions):
                blk += reports.quote(captions[j], "the model, on this segment")
            cells.append((f"segment {j}", blk))
        body += reports.side_by_side(cells)
        body += reports.note("The four most recent segments; the charts below cover all of them.")

        series = {}
        if len(stats) > 1:
            series["sharpness (variance of Laplacian)"] = [s["sharpness"] for s in stats]
        if len(overlap) > 1:
            series["word overlap with segment 0 (what `judge` uses)"] = overlap
        if len(cosine) > 1:
            series["Cosmos-Embed1 similarity to segment 0"] = cosine
        if series:
            body += reports._heading("Two ways of measuring the same drift")
            body += reports.metric_lines(series, caption=(
                "One point per segment. The word overlap compares two sentences the model "
                "wrote about the clips; the embedding similarity compares the clips "
                "themselves, with no sentence in between. If they fall together, the cheap "
                "metric is measuring something real. If the embedding holds while the words "
                "move, the words were tracking phrasing rather than content."
            ))
        if len(cosine) > 2 and len(overlap) == len(cosine):
            import statistics

            try:
                agree = statistics.correlation(cosine, overlap)
                body += reports.note(
                    f"Correlation between the two curves: <b>{agree:+.2f}</b>. "
                    + ("They are measuring the same thing, so `judge`'s word overlap is "
                       "doing real work despite being a word count."
                       if agree > 0.5 else
                       "They are NOT tracking together, which means at least one of them "
                       "is not measuring content drift and the embedder is the one with a "
                       "claim to be.")
                )
            except statistics.StatisticsError:
                pass
        if note:
            body += reports.note(note)
        live = list(rows) + ([("Progress", detail)] if detail else [])
        flyte.report.replace(
            reports.final_html(stage, live, body, reports.EMBED_EXPLAINER), do_flush=True
        )

    # ── Generate ────────────────────────────────────────────────────────────────
    frame = meta["first_frame"]
    gen_secs = 0.0
    for i in range(min(chunks, int(meta["chunks"].shape[0]))):
        paint("Rolling forward on recorded actions", f"chunk {i + 1}")
        seg, secs = world.rollout_chunk(pipe, meta, meta["chunks"][i], frame=frame,
                                        steps=steps, guidance=guidance, seed=seed + i)
        pieces.append(seg); clips.append(media.encode(seg, fps=fps, crf=28))
        stats.append(world.clip_stats(seg)); labels.append(f"action chunk {i}")
        gen_secs += secs; frame = seg[-1]

    if pieces:
        tail = pieces[-1]
        w, h = tail[-1].size
        w, h = w - (w % 16), h - (h % 16)
        for i in range(segments):
            paint("Continuing past the recorded actions", f"continuation {i + 1} of {segments}")
            try:
                seg, secs = world.extend(pipe, tail, meta["prompt"], num_frames=frames,
                                         height=h, width=w, fps=fps, steps=steps,
                                         guidance=guidance, seed=seed + 100 + i)
            except Exception as exc:  # noqa: BLE001
                note = f"Continuation {i} failed after {len(pieces)} segments: {exc}"
                log.warning(note); break
            body_frames = seg[world.V2V_OVERLAP:]
            pieces.append(body_frames); clips.append(media.encode(body_frames, fps=fps, crf=28))
            stats.append(world.clip_stats(body_frames))
            labels.append(f"continuation {i}")
            gen_secs += secs; tail = seg

    pipe = None
    world.release()
    rows.append(("Generation", f"{gen_secs / 60:.1f} min for {len(pieces)} segments"))

    # ── Describe (the metric judge uses) ─────────────────────────────────────────
    paint("Describing each segment", "loading the understanding surface")
    model, processor = world.load_reasoner(repo)
    for i, seg in enumerate(pieces):
        paint("Describing", f"segment {i + 1} of {len(pieces)}")
        text, _ = world.ask(model, processor, Q_DESCRIBE, video=seg, max_new_tokens=96)
        captions.append(text)
        overlap.append(1.0 if i == 0
                       else round(world.description_overlap(captions[0], text), 3))
    model, processor = None, None
    world.release()

    # ── Embed (the metric that skips the sentence) ──────────────────────────────
    paint("Embedding each segment", f"loading {EMBED_REPO}")
    try:
        vecs = world.embed_clips(EMBED_REPO, pieces)
        import torch

        ref = vecs[0]
        cosine.extend([round(float(torch.nn.functional.cosine_similarity(
            ref.unsqueeze(0), v.unsqueeze(0)).item()), 4) for v in vecs])
    except Exception as exc:  # noqa: BLE001
        note = f"Embedding failed, so only the word overlap is charted: {exc}"
        log.warning(note)

    paint("Finished", "")
    log.info("embed: overlap %s cosine %s", overlap, cosine)
    return {
        "segments": len(pieces),
        "word_overlap": overlap,
        "embed_similarity": cosine,
        "sharpness": [round(s["sharpness"], 1) for s in stats],
        "captions": captions,
        "generate_seconds": round(gen_secs, 1),
        "note": note,
    }


@orch_env.task(report=True)
async def world_models(scene: str = "box-topple", repo: str = NANO) -> dict:
    """Entry point for the short tasks. CPU-only orchestrator, so it cannot deadlock
    its own GPU children.

    The children run in SEQUENCE, not in parallel, and that is not a style choice:
    there is one GPU on this box, so a second GPU task would sit Unschedulable on
    "Insufficient nvidia.com/gpu" until the first finished anyway.

    `horizon` is deliberately NOT in here. It runs for hours by design, and burying it
    behind seven other tasks means its report does not start painting until the rest
    have finished. Run it on its own:  flyte run pipeline.py horizon

    Neither are the three understanding-surface tasks; `reasoning` below is their
    entry point. Splitting them is not tidiness. Each of those tasks loads BOTH experts
    in sequence, so a failure in the handover between them is a different failure from
    anything in here, and mixing the two sets means a report you open to check one
    thing has to be read past eight others to find it.
    """
    result = {
        "imagine": await imagine(scene=scene, repo=repo),
        "rollout": await rollout(repo=repo),
        "compare": await compare(scene=scene, repo=repo),
        "invert": await invert(repo=repo),
        "counterfact": await counterfact(repo=repo),
        "policy": await policy(repo=repo),
        "emerge": await emerge(scene=scene, repo=repo),
        "extend": await extend(repo=repo),
    }
    log.info("result: %s", result)
    return result


@orch_env.task(report=True)
async def reasoning(repo: str = NANO) -> dict:
    """Entry point for the three understanding-surface tasks, in sequence.

    Sequential for the same reason as `world_models`: one GPU, so a second GPU task
    would sit Unschedulable on "Insufficient nvidia.com/gpu" until the first finished.

    Each child loads the generation expert and the understanding expert one after the
    other inside its own pod, which is the part worth watching the first time this is
    run on a new box. 30 GB then 17.5 GB in one 119.7 GiB pool shared with the OS is
    fine; the two at once is not, and on the GB10 that is a wedged machine rather than
    an exception. The handover shows up in each child's report as a "Handover" row
    saying how much was still held after the release.

    `plan` runs first because it is the cheapest and it exercises the handover in the
    easy direction (understanding first, then generation), so a broken handover is
    found in three minutes rather than twenty.
    """
    result = {
        "plan": await plan(repo=repo),
        "blind": await blind(repo=repo),
        "judge": await judge(repo=repo),
    }
    log.info("result: %s", result)
    return result


@orch_env.task(report=True)
async def data_engine(repo: str = NANO) -> dict:
    """Entry point for the three data-engine tasks, in sequence. About an hour.

    Ordered cheapest-first and, more usefully, in the order their results depend on each
    other. `cycle` establishes what a label recovered from generated video is worth, so
    running it first means `dream`'s labelled output arrives with an error bar already
    attached rather than looking more trustworthy than it is. `dream` runs last because
    it is by far the longest and loads the model three times.

    Every child loads both experts in sequence inside its own pod. Watch the Handover
    rows if one dies on memory: that is where a leak between the two would show up.
    """
    result = {
        "cycle": await cycle(repo=repo),
        "choose": await choose(repo=repo),
        "dream": await dream(repo=repo),
    }
    log.info("result: %s", result)
    return result


@orch_env.task(report=True)
async def overnight(
    repo: str = NANO,
    segments: int = 90,
    scene: str = "forklift",
) -> dict:
    """The unattended run: the remaining short tasks, then the long one, in sequence.

    Chaining these here rather than from a shell loop on the host is not a style
    preference, it is the thing that makes an overnight run survivable. A GPU task on
    this box holds most of a 119 GiB unified pool for as long as it runs, and anything
    else waiting around on the host is a candidate to be killed for memory long before
    the run finishes; that is exactly how the first attempt at this sequence died,
    with the driving shell loop killed while the pod it was waiting on carried on
    perfectly happily. A CPU-only orchestrator pod has no such problem: it holds 4 GiB,
    the cluster does the sequencing, and nothing on the host has to stay alive.

    Every child is wrapped, and a failure is recorded rather than raised. Eight hours
    in, a result holding three good reports and a note about the fourth is worth much
    more than one traceback and nothing else, and the children each write their own
    report as they go, so a failure here costs only the tasks after it.

    `horizon` runs last on purpose: it is by far the longest, and putting it first
    would mean a small mistake in a five minute task is not discovered until morning.
    """
    plan = [
        ("emerge", lambda: emerge(repo=repo)),
        ("gallery", lambda: gallery(repo=repo)),
        ("imagine_sound", lambda: imagine(scene=scene, repo=repo, sound=True)),
        ("horizon", lambda: horizon(repo=repo, segments=segments)),
    ]
    rows = [("Plan", " -> ".join(name for name, _ in plan)), ("Long run", f"horizon x {segments}")]
    results: dict = {}

    for i, (name, run) in enumerate(plan):
        flyte.report.replace(
            reports.progress_html(
                f"Running {name} ({i + 1} of {len(plan)})",
                "Each child writes its own report; open the child action to watch it.",
                rows + [(k, "done" if not str(v).startswith("FAILED") else str(v)[:60])
                        for k, v in results.items()],
            ),
            do_flush=True,
        )
        try:
            results[name] = await run()
        except Exception as exc:  # noqa: BLE001
            results[name] = f"FAILED {type(exc).__name__}: {exc}"
            log.warning("%s failed, continuing: %s", name, exc)

    flyte.report.replace(
        reports.final_html(
            "Overnight sequence",
            rows + [(k, "failed" if str(v).startswith("FAILED") else "done")
                    for k, v in results.items()],
            reports.details("results", "\n\n".join(f"{k}: {v}" for k, v in results.items())),
        ),
        do_flush=True,
    )
    log.info("overnight: %s", {k: type(v).__name__ for k, v in results.items()})
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(world_models))
