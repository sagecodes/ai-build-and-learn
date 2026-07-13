"""Prompt suites for the video-model comparison.

The image-generation demo has a "prompts to try, and what each stresses" list. This
is the video equivalent, and the axes are genuinely different. A text-to-image model
is judged on a single frame; a video model can render eight beautiful frames and
still fail, because the thing being tested is what happens *between* them.

So every prompt here targets a failure mode that only exists in time:

  - **Temporal consistency.** Does a face stay the same face? Do stripes stay put?
    This is where most open models break first, and it is invisible in any single
    frame. The report's frame strip is the surface that exposes it: identity drift
    is obvious across six frames side by side and easy to miss while a 3-second clip
    loops past you.
  - **Physics and causality.** Diffusion models learn what things look like, not what
    they do. Spills, collisions and falling objects are where that gap shows.
  - **Camera vs subject motion.** "The camera moves" and "the thing moves" are
    different capabilities, and a model can have one without the other. A model with
    no camera control renders a moving subject on a locked-off tripod forever.
  - **Text stability.** Text-in-image is hard; text-in-video is brutal, because the
    letters must be legible *and* not reshuffle every frame. Expect failure. That's
    the point: it's the clearest quality separator in the grid.
  - **Motion magnitude.** Wan in particular drifts toward near-static clips (which is
    exactly what its long default negative prompt is fighting). A prompt with fast
    motion tells you whether you're getting video or an animated photograph.
  - **Audio.** Only LTX-2 generates a soundtrack, so one prompt is chosen to have an
    obvious, rhythmic, visually-synced sound source. If the hammer hits and the clang
    lands on the wrong frame, you'll hear it immediately.

Ordered cheap-and-legible first, so a truncated run still produces a usable grid.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    text: str
    axis: str      # the capability under test
    watch_for: str  # the specific failure mode to look for in the report


# ── The overnight suite ─────────────────────────────────────────────────────────

SUITE: list[Prompt] = [
    Prompt(
        text="waves crashing against black volcanic rock, sea spray in the air, "
             "overcast sky, slow motion",
        axis="fluid motion + the control prompt",
        watch_for="Our baseline: every model has produced something plausible here. "
                  "Water is forgiving because it has no fixed shape to be "
                  "inconsistent about. If a model fails THIS, it is broken.",
    ),
    Prompt(
        text="a blacksmith hammering a glowing orange horseshoe on an anvil, "
             "sparks flying with each strike, dark workshop",
        axis="rhythmic motion + AUDIO SYNC",
        watch_for="The audio prompt. LTX-2 should produce a clang that lands ON the "
                  "hammer strike. Watch for sparks that appear before the impact, "
                  "which is the giveaway that a model has no causal model of the "
                  "event, just a texture of 'blacksmith'.",
    ),
    Prompt(
        text="a woman with curly red hair and round green glasses turns her head "
             "to face the camera and smiles, soft window light",
        axis="identity consistency over time",
        watch_for="THE hardest common failure. Compare frame 1 and frame 6 in the "
                  "strip: is it the same person? Do the glasses stay round and green? "
                  "Faces morphing mid-clip is the single most common open-model tell.",
    ),
    Prompt(
        text="a slow dolly push forward through a misty pine forest at dawn, "
             "shafts of light between the trunks",
        axis="camera motion + 3D parallax",
        watch_for="Does the CAMERA move, or does the scene just shimmer? Real parallax "
                  "means near trunks sweep past faster than far ones. A model without "
                  "camera control renders a pretty, static forest.",
    ),
    Prompt(
        text="a glass of red wine tips over on a marble counter and the wine spills "
             "across the surface",
        axis="physics + causality",
        watch_for="Expect failure, and expect it to be interesting. Does the glass "
                  "actually tip, or vibrate in place? Does wine leave the glass before "
                  "it falls? Volume should be conserved; it usually isn't.",
    ),
    Prompt(
        text="a red sports car speeds past the camera on a wet city street at night, "
             "neon reflections on the asphalt",
        axis="fast motion + motion blur",
        watch_for="The motion-magnitude test. Wan drifts toward static clips (its "
                  "default negative prompt exists to fight exactly this). Does the car "
                  "actually traverse the frame, and does it deform while doing it?",
    ),
    Prompt(
        text="a neon sign reading OPEN 24 HOURS flickering above a diner door at "
             "night in the rain",
        axis="text stability in motion",
        watch_for="Brutal, and the clearest separator in the grid. Legible text is a "
                  "win; text that stays the SAME text across all frames is a bigger "
                  "one. Most models will reshuffle the letters every frame.",
    ),
    Prompt(
        text="three paper boats float down a rain gutter, and the middle one tips "
             "over and sinks",
        axis="object permanence + counting + a scripted event",
        watch_for="Are there three boats in frame 1 AND frame 6? Counting fails often. "
                  "And does the specific scripted event (the middle one sinks) actually "
                  "happen, or does the model just render generic boats? This tests "
                  "prompt adherence over time, not just at t=0.",
    ),
]


# A 3-prompt subset for a quick pass: the control, the audio one, and the hardest.
QUICK: list[Prompt] = [SUITE[0], SUITE[1], SUITE[2]]


SUITES: dict[str, list[Prompt]] = {
    "overnight": SUITE,
    "quick": QUICK,
}


def get_suite(name: str) -> list[str]:
    """Suite name -> plain prompt strings (what the pipeline actually takes)."""
    try:
        return [p.text for p in SUITES[name]]
    except KeyError:
        raise ValueError(
            f"Unknown suite {name!r}. Known: {', '.join(SUITES)}"
        ) from None


def describe(name: str = "overnight") -> str:
    """The suite as a readable table, for the README and the run's report header."""
    rows = []
    for i, p in enumerate(SUITES[name], 1):
        rows.append(f"{i}. [{p.axis}]\n   {p.text}\n   -> {p.watch_for}")
    return "\n\n".join(rows)


if __name__ == "__main__":
    print(describe("overnight"))
