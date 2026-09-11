"""HTML for the Flyte reports: generated clips, frame strips, and run facts.

Same palette and helpers as topics/rl-mujoco/reports.py, topics/isaac-sim/reports.py
and topics/dreamerv3/reports.py, so a viewer moving between the world-model demos is
not re-learning the colours. Everything is inline HTML and inline data: URIs; no
JavaScript, no external assets, no object-store round trip.
"""

from __future__ import annotations

import html

_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"

_TITLE = "NVIDIA Cosmos 3 - a world model you can roll forward"


def _table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (k, v) in enumerate(rows):
        border = "border-bottom:1px solid #333;" if i < len(rows) - 1 else ""
        body += (
            f'<tr><td style="padding:6px;{border}white-space:nowrap;">{k}</td>'
            f'<td style="padding:6px;{border}color:{_ACCENT};">{v}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;width:100%;">{body}</table>'


def _panel(title: str, inner: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};'
        f'padding:20px;border-radius:8px;">'
        f'<h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def _heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;">{text}</h3>'


def _note(text: str) -> str:
    return (
        f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;'
        f'line-height:1.5;">{text}</p>'
    )


def _details(summary: str, body: str) -> str:
    return (
        f'<details><summary style="cursor:pointer;color:{_MUTED};font-family:monospace;">'
        f"{summary}</summary>"
        f'<pre style="font-size:11px;color:{_TEXT};background:{_PANEL};padding:12px;'
        f'border-radius:4px;overflow-x:auto;white-space:pre-wrap;">'
        f"{html.escape(body)}</pre></details>"
    )


# Public aliases: pipeline.py assembles the comparison cells itself, so it needs
# these two. Exported by name rather than reached for as `reports._note`, which is
# the kind of thing that quietly breaks when this file is tidied.
note = _note
details = _details


def progress_html(stage: str, detail: str, rows: list[tuple[str, str]]) -> str:
    """Painted while the pod is still working, so the report is never blank.

    Worth having here specifically: this task spends its first several minutes
    downloading 35 GB and its next few loading a 16B transformer, and a report that
    stays empty through all of that is indistinguishable from a hung pod.
    """
    return (
        f"<h2>{_TITLE}</h2>"
        + _panel(stage, _table(rows) + _note(detail))
    )


def clip_block(
    title: str,
    video_html: str,
    strip_html: str = "",
    probe: str = "",
    prompt: str = "",
) -> str:
    """One generated clip: the video, its frame strip, and what produced it."""
    out = _heading(title) + video_html
    if strip_html:
        out += (
            _note("The same clip laid out in time. A world model that has collapsed to "
                  "a single repeated frame looks fine as a loop and obvious as a strip.")
            + strip_html
        )
    if probe:
        out += _note(probe)
    if prompt:
        out += _details("prompt", prompt)
    return out + "<br/>"


def side_by_side(cells: list[tuple[str, str]]) -> str:
    """Lay several clip blocks out in a responsive row.

    flex-wrap rather than a grid: Flyte reports render at a width you do not control,
    and two 560px clips side by side have to be allowed to become two stacked clips.
    """
    inner = ""
    for label, body in cells:
        inner += (
            f'<div style="flex:1 1 380px;min-width:320px;">'
            f'<h4 style="color:{_HILITE};font-family:monospace;margin:0 0 8px;">'
            f"{label}</h4>{body}</div>"
        )
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{inner}</div>'


def action_traces(
    truth: list[list[float]],
    pred: list[list[float]],
    labels: list[str] | None = None,
    dims: list[int] | None = None,
    caption: str = "",
    width: int = 520,
    height: int = 84,
) -> str:
    """Overlay recovered actions on ground truth, one small chart per channel.

    Inline SVG rather than a plotting library. matplotlib is not in the image spec and
    is not worth adding for two polylines, and an <img> of a chart cannot be zoomed in
    a Flyte report the way vector text can.

    Each channel gets its own y-scale taken from both series together. Sharing one
    scale across channels would be the honest thing for commensurable numbers and is
    the wrong thing here: the `av` action packs translation next to a rotation basis
    resting near 1.0, so a shared axis flattens every channel that matters.
    """
    n = min(len(truth), len(pred))
    if n < 2:
        return _note("Not enough steps to plot.")
    ncols = len(truth[0])
    dims = list(range(ncols)) if dims is None else [d for d in dims if d < ncols]

    out = ""
    for d in dims:
        a = [row[d] for row in truth[:n]]
        b = [row[d] for row in pred[:n]]
        lo, hi = min(min(a), min(b)), max(max(a), max(b))
        span = (hi - lo) or 1.0
        pad = span * 0.08
        lo, hi = lo - pad, hi + pad
        span = hi - lo

        def path(series: list[float]) -> str:
            step = width / (n - 1)
            pts = [
                f"{i * step:.1f},{height - (v - lo) / span * height:.1f}"
                for i, v in enumerate(series)
            ]
            return " ".join(pts)

        name = labels[d] if labels and d < len(labels) else f"channel {d}"
        out += (
            f'<div style="margin:0 0 10px;">'
            f'<div style="color:{_MUTED};font-family:monospace;font-size:11px;'
            f'margin-bottom:2px;">{name} '
            f'<span style="color:{_ACCENT};">&#9473; truth</span> '
            f'<span style="color:{_HILITE};">&#9473; recovered</span></div>'
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" style="background:{_BG};border-radius:4px;">'
            f'<polyline fill="none" stroke="{_ACCENT}" stroke-width="1.6" '
            f'points="{path(a)}"/>'
            f'<polyline fill="none" stroke="{_HILITE}" stroke-width="1.6" '
            f'stroke-dasharray="4 3" points="{path(b)}"/>'
            f"</svg></div>"
        )
    return out + (_note(caption) if caption else "")


def final_html(
    subtitle: str,
    rows: list[tuple[str, str]],
    body: str,
    explainer: str = "",
    logs: list[str] | None = None,
) -> str:
    """The finished report: run facts, the generated media, and what to look at."""
    out = f"<h2>{_TITLE}</h2>" + _panel(subtitle, _table(rows))
    if explainer:
        out += _panel("What to look for", _note(explainer))
    out += "<br/>" + body
    if logs:
        out += _panel("Logs", _details("run tail", "\n".join(logs[-30:])))
    return out


# ── Explainers ──────────────────────────────────────────────────────────────────
#
# Kept as text in the report rather than only in the README. The report is the
# artifact people scroll past on the stream, and a clip with no claim attached to it
# is just a pretty video.

IMAGINE_EXPLAINER = (
    "This is the generator surface: text (and optionally one image) in, video out. "
    "Judge it as physics, not as cinematography. Do surfaces in contact stay in "
    "contact? Does anything accelerate without a cause, or come to rest without "
    "touching something? Do objects keep their shape and volume across frames, and "
    "does what passes behind something reappear on the other side? Those are the "
    "questions a world model is supposed to answer, and they are the ones a video "
    "model that has only learned appearance gets wrong."
)

ROLLOUT_EXPLAINER = (
    "This is the part that makes it a world model rather than a video generator. "
    "The model is given ONE observed frame and a sequence of raw robot actions, and "
    "asked what happens next: p(future video | first frame, actions). Nothing about "
    "the future is described in words. If the predicted motion tracks the commanded "
    "actions, the model has learned the dynamics of that embodiment, which is what "
    "lets it stand in for a simulator when you are generating training data or "
    "evaluating a policy without a robot. Compare that with topics/dreamerv3, where "
    "the agent learns exactly this same p(next state | state, action) from scratch "
    "for one environment, and with topics/isaac-sim, where the dynamics are a hand-"
    "written physics engine instead of a learned model."
)

COMPARE_EXPLAINER = (
    "Same scene, same seed, two prompts. The left one is the sentence you would "
    "hand any other video model. The right one is the structured JSON caption Cosmos "
    "3 was actually trained on, naming subjects, lighting, camera and a beat-by-beat "
    "temporal description of the physical event. NVIDIA's guidance is to upsample the "
    "short form into the long one with an LLM before generation; this is what that "
    "step buys, without a second model in the loop."
)


INVERT_EXPLAINER = (
    "Forward dynamics asks what happens next. This asks the opposite: here is a "
    "video, what actions produced it? Cosmos denoises the action channel with the "
    "same machinery it uses for pixels, so running the model backwards is a mode "
    "flag rather than a second model. The clip and the answer key both ship inside "
    "the checkpoint, so the dashed line is the model's guess and the solid line is "
    "what actually happened, and the gap between them is a number rather than an "
    "impression. This is also the sharpest contrast with topics/dreamerv3: that "
    "world model maps (state, action) to the next state and has no path in the "
    "other direction at all, so a Dreamer agent can dream a future but can never "
    "watch a video and say what was done in it."
)


def bars(rows: list[tuple[str, float]], caption: str = "", unit: str = "") -> str:
    """A horizontal bar per row, scaled to the largest value.

    Used for the counterfactual divergence numbers, where the shape of the comparison
    is the whole point and the absolute values are close to meaningless on their own:
    "held differs from recorded by 6.8 grey levels" says nothing until it sits next to
    the same measurement for the other variants.
    """
    if not rows:
        return ""
    top = max(abs(v) for _, v in rows) or 1.0
    out = ""
    for name, value in rows:
        pct = abs(value) / top * 100.0
        out += (
            f'<div style="display:flex;align-items:center;gap:10px;margin:0 0 6px;'
            f'font-family:monospace;font-size:12px;color:{_TEXT};">'
            f'<span style="width:120px;flex:0 0 120px;text-align:right;color:{_MUTED};">'
            f"{name}</span>"
            f'<span style="flex:1 1 auto;background:{_PANEL};border-radius:3px;height:16px;'
            f'position:relative;overflow:hidden;">'
            f'<span style="display:block;height:100%;width:{pct:.1f}%;background:{_ACCENT};'
            f'border-radius:3px;"></span></span>'
            f'<span style="width:90px;flex:0 0 90px;color:{_HILITE};">'
            f"{value:.2f}{unit}</span></div>"
        )
    return out + (_note(caption) if caption else "")


def metric_lines(
    series: dict[str, list[float]],
    caption: str = "",
    width: int = 560,
    height: int = 70,
) -> str:
    """One small line chart per named series, each on its own y-scale.

    Per-series scaling rather than a shared axis, for the same reason `action_traces`
    does it: these metrics are not commensurable. Sharpness is in the hundreds,
    inter-frame motion in single digits, and putting them on one axis renders motion
    as a flat line along the bottom exactly when its collapse is the thing worth
    seeing. The min and max are printed on each chart so the scale is never implied.
    """
    out = ""
    for name, values in series.items():
        if len(values) < 2:
            continue
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        step = width / (len(values) - 1)
        pts = " ".join(
            f"{i * step:.1f},{height - (v - lo) / span * (height - 8) - 4:.1f}"
            for i, v in enumerate(values)
        )
        # Direction of travel is the signal in every one of these, so state it rather
        # than making the reader eyeball the slope of a 70px chart.
        drift = values[-1] - values[0]
        arrow = "steady" if abs(drift) < span * 0.05 else ("rising" if drift > 0 else "falling")
        out += (
            f'<div style="margin:0 0 10px;">'
            f'<div style="color:{_MUTED};font-family:monospace;font-size:11px;'
            f'margin-bottom:2px;">{name} '
            f'<span style="color:{_HILITE};">{values[0]:.1f} to {values[-1]:.1f} '
            f"({arrow})</span></div>"
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" style="background:{_BG};border-radius:4px;">'
            f'<polyline fill="none" stroke="{_ACCENT}" stroke-width="1.8" points="{pts}"/>'
            f"</svg></div>"
        )
    return out + (_note(caption) if caption else "")


COUNTERFACT_EXPLAINER = (
    "The control experiment for everything else on this page. Every other task takes "
    "it on trust that Cosmos is conditioning on the actions it is handed, and a video "
    "model that quietly ignored its action channel and simply continued the scene "
    "would look exactly as convincing. So this holds the conditioning frame, the "
    "prompt, the seed and the schedule fixed and changes ONLY the actions. Four "
    "sequences, all anchored at the recorded starting pose so none of them ask the "
    "model for something it has never seen. The one to watch is `held`, which "
    "commands the starting pose at every step: a model that is genuinely reading the "
    "actions has to predict a robot that stops moving. If all four clips looked the "
    "same, the phrase 'world model' would not be earned."
)

POLICY_EXPLAINER = (
    "The third action mode, and the one that dissolves the distinction between a "
    "world model and a policy. Forward dynamics is given actions and predicts pixels. "
    "Inverse dynamics is given pixels and predicts actions. This is given neither: "
    "one frame and a goal, with both channels left noisy, so the model denoises a "
    "plan and the consequences of that plan in the same pass. To the transformer the "
    "action tokens and the pixel tokens are one sequence, and which of them counts as "
    "'the input' is decided entirely by which ones you leave noisy, which is why the "
    "same weights can be called a simulator and a policy in the same breath. Read the "
    "error against the human demonstration with care: there is no single right way to "
    "pick up items in a supermarket, so disagreement is not automatically failure."
)

EMERGE_EXPLAINER = (
    "The same generation as the first task, with the denoiser opened up: instead of "
    "decoding once at the end, it decodes at eight points along the schedule. What "
    "gets decoded matters more than it sounds. The sample the solver is holding "
    "part-way through, x_t, stays noise-dominated under a flow-matching schedule and "
    "decoding it shows essentially nothing until the final step, which is a true "
    "picture of a real quantity and not the one anybody means. What is shown here is "
    "x0: the model's PREDICTION of the finished video, which exists at every step and "
    "is what the solver steers with. So each frame below is an answer to 'if you had "
    "to stop right now, what would you say the clip is?' "
    "For a world model the interesting question is which properties are settled in "
    "the first answer and never revised, and which keep changing to the end: that is "
    "the difference between deciding what physically happens and deciding what it "
    "looks like, and it says where the compute is actually going."
)

EXTEND_EXPLAINER = (
    "The fourth generation mode, selected by handing the pipeline a video instead of "
    "an image: the leading frames are kept clean and the rest are denoised, so the "
    "model continues a clip rather than starting one. It is the only way past a fixed "
    "clip length, because the model's context is a single clip and a longer video has "
    "to be built by feeding its own output back in. That also makes it the honest "
    "demonstration of the weak point. Segment 1 continues a real generated clip; "
    "segment 2 continues segment 1, which was itself a prediction. Nothing re-anchors "
    "the chain to anything real, so error is not merely retained, it accumulates. "
    "Read the sharpness line with the sample size in mind: three continuations is "
    "enough to see the mechanism and not enough to establish a trend, and the middle "
    "segment routinely scores worse than the one after it. `horizon` is the same "
    "mechanism run until the trend is unambiguous."
)

HORIZON_EXPLAINER = (
    "The long run, and the question the short demos dodge. A world model that holds "
    "together for two seconds is a video model with good manners; the claim that it "
    "has learned physics is a claim about what happens when you keep going. So this "
    "keeps going: first the action chunks that ship in the checkpoint, which are the "
    "only part of the rollout with real actions behind it, and then video-to-video "
    "continuations, each one conditioned on the model's own previous output and never "
    "re-anchored to anything real. "
    "What a 94-segment run of this actually did, so you can compare: sharpness fell "
    "from 734 on the action-driven chunks to a mean of 82 within about fifteen "
    "self-conditioned segments, and then STOPPED falling, wandering between roughly "
    "50 and 250 for the remaining seventy five without ever recovering the "
    "action-driven level. Inter-frame motion went the other way, rising from 2.5 to a "
    "mean near 6 and peaking at 10.8. Luminance drifted up about 7 percent. "
    "So the failure is not the one the phrase 'error accumulation' suggests. It does "
    "not freeze, and it does not fade to grey: it loses most of its detail quickly, "
    "then settles into a soft, more agitated steady state that stays a plausible "
    "video forever while no longer being the video it started as. Watch the strip "
    "rather than the charts for the part that matters, which is the scene quietly "
    "ceasing to be a supermarket."
)


GALLERY_EXPLAINER = (
    "Four scenes from one model load, all of them physical-AI scenes rather than "
    "scenery. A world model earns its name on contact, occlusion and momentum, and a "
    "sweeping drone shot over a mountain range tests none of those, which is why "
    "nothing here is pretty for its own sake. Each prompt names a specific physical "
    "event, so each clip can be marked right or wrong rather than admired: does the "
    "sponge compress against the plate and spring back, does the box pivot about its "
    "bottom edge before it goes over, does the load stay level as the forks lift, "
    "does the car pitch forward when it brakes. Judge these as physics. A video model "
    "that has only learned appearance produces things that accelerate without a "
    "cause, come to rest without contact, or change volume between frames."
)


# ── The reasoning surface ───────────────────────────────────────────────────────

def quote(text: str, who: str = "Cosmos 3, reasoning surface") -> str:
    """Text the MODEL wrote, styled so it cannot be mistaken for text we wrote.

    Worth the extra element. Every other sentence in these reports is a claim this
    repo is making and stands behind; these are the model's own words being shown as
    evidence, including when they are wrong, and a reader skimming on a stream has no
    other way to tell the two apart.
    """
    return (
        f'<blockquote style="margin:8px 0;padding:10px 14px;background:{_PANEL};'
        f'border-left:3px solid {_HILITE};border-radius:0 4px 4px 0;'
        f'font-family:monospace;font-size:12px;color:{_TEXT};line-height:1.55;">'
        f"{html.escape(text)}"
        f'<footer style="color:{_MUTED};font-size:10px;padding-top:6px;">{who}</footer>'
        f"</blockquote>"
    )


def score_chip(score: float | None, scale: str = "/10", label: str = "") -> str:
    """A single number the model gave, or an honest blank when it gave none."""
    if score is None:
        return (
            f'<span style="display:inline-block;padding:4px 10px;border-radius:12px;'
            f'background:{_PANEL};color:{_MUTED};font-family:monospace;font-size:12px;">'
            f"{label + ' ' if label else ''}no score in the answer</span>"
        )
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:12px;'
        f'background:{_ACCENT};color:{_BG};font-family:monospace;font-size:13px;'
        f'font-weight:bold;">{label + " " if label else ""}{score:g}{scale}</span>'
    )


PLAN_EXPLAINER = (
    "Both surfaces of one checkpoint, in one task. First the understanding expert is "
    "shown a photograph and a goal and asked to decompose it, with no video involved "
    "at all. Then those weights are dropped, the diffusion expert is loaded from the "
    "SAME files on disk, and each subtask it wrote becomes a prompt for a predicted "
    "clip. So the plan is not something a human wrote for the generator: it is the "
    "model's own decomposition of the goal, handed back to itself to imagine. Read "
    "the subtasks first and the clips second, and check whether the clip actually "
    "shows the step it was given -- a plan the generator cannot render is the "
    "interesting failure here, not a plan that reads well."
)

JUDGE_EXPLAINER = (
    "The long-horizon rollout measured by the model's own understanding rather than "
    "by pixel statistics. `horizon` charts sharpness, inter-frame motion and "
    "luminance, and those can only say whether the rollout is still a well-formed "
    "video; they cannot say whether it is still a video OF THE SAME THING. So every "
    "segment here is shown back to the understanding expert, which never sees a "
    "segment index or any of the numbers, and is asked what is happening and whether "
    "the motion is physically plausible. Two things to watch: the plausibility score, "
    "which is the model grading its own generation, and the description overlap, "
    "which is how many content words each segment's description still shares with "
    "segment 0. Measured over 21 segments the two do not break together: the "
    "description overlap goes at segment 7 while the plausibility score holds until "
    "17, so the rollout stops being ABOUT the same thing long before it stops being "
    "believable. Watch for the point where the justifications stop being generic and "
    "start naming a specific physical violation, which is the thing no pixel metric "
    "can report."
)

BLIND_EXPLAINER = (
    "The counterfactual test re-run with an independent judge. `counterfact` "
    "establishes that changing the actions changes the pixels, which it measures as a "
    "mean absolute difference -- a real number, but one that only says the clips are "
    "not identical. It cannot say the difference is THE DIFFERENCE THE ACTIONS "
    "DESCRIBE. So each clip is shown to the understanding expert with no label, no "
    "variant name, no ordering and no access to the actions, and asked one question: "
    "is this robot moving or holding still? The `held` variant commands the starting "
    "pose at every step, so a model that reads its action channel has to predict a "
    "robot that stops, and a judge that cannot see the actions has to be able to see "
    "that it stopped. Read the moving/still verdicts first; the coarse "
    "stationary/small/large bucket is a second, independent question about the same "
    "clip, and where the two disagree that is shown rather than reconciled. An "
    "instrument that is unreliable at a particular question is a result about the "
    "instrument, and hiding it would make every other judged number here worth less."
)


CYCLE_EXPLAINER = (
    "The measurement that decides whether Cosmos can be used as a data engine. "
    "NVIDIA's own GR00T-Dreams pipeline generates synthetic robot video, filters it "
    "with a video critic, labels it with an inverse dynamics model, and trains a policy "
    "on the result. Every stage of that rests on one unstated assumption: that actions "
    "recovered from GENERATED video are accurate enough to train on. This measures it. "
    "The same real actions are pushed through forward dynamics to make a dreamed clip, "
    "the dreamed clip is pushed back through inverse dynamics, and the recovered actions "
    "are scored against the actions we started with. The baseline is the same inverse "
    "step run on the REAL clip, which is the floor: whatever error the inverse model has "
    "on real video, it has before any dreaming happens. The gap between the two bars is "
    "the tax the dream engine charges, and it is the number to quote, not either bar. "
    "Read the moving-channel error rather than the overall mean: most channels of these "
    "action vectors barely leave their start value in a given clip, so an average across "
    "all nine is dominated by channels where there was nothing to get right."
)


DREAM_EXPLAINER = (
    "NVIDIA's GR00T-Dreams pipeline, stages 3 to 5, running off one checkpoint. The "
    "model is given ONE real frame and a series of language instructions describing "
    "behaviours that frame's clip never shows, and asked to generate each one. Then the "
    "understanding surface of the same checkpoint acts as the video critic, deciding "
    "which dreams are physically plausible AND actually show the behaviour that was "
    "asked for. Then inverse dynamics labels the survivors with actions, turning video "
    "into trajectories a policy could train on. Two things to read: the rejection rate, "
    "because a generator nothing gets thrown out of is either not being adventurous or "
    "not being checked, and the control scenario, which is the behaviour the real clip "
    "does show and which therefore has to survive. What is NOT here is training a policy "
    "on the output: the bundled assets support a working pipeline at toy scale, not a "
    "claim about whether dreams beat real data. Pair it with `cycle`, which measures how "
    "much label accuracy the generation step costs in the first place."
)


CHOOSE_EXPLAINER = (
    "The oldest argument for having a world model at all: you do not need to try an "
    "action in the world if you can predict what it would do. Four action sequences are "
    "rolled forward from one identical conditioning frame, and the understanding surface "
    "of the same checkpoint scores each imagined future against a goal stated in plain "
    "words. The winner is the action a planner would execute. topics/dreamerv3 does this "
    "with a learned value function that costs millions of environment steps to train; "
    "the scorer here is a pretrained model that has never seen this robot and cost "
    "nothing, and is correspondingly blunter. The load-bearing part is that there are "
    "TWO goals wanting opposite things over the SAME four clips. A single ranking cannot "
    "distinguish planning from the model simply preferring one video, so the claim is "
    "only earned if the winner moves when the goal does. Nothing is executed here: there "
    "is no robot, so this is the selection step of a planner rather than a closed loop."
)


ROBUST_EXPLAINER = (
    "The counterfactual result is an ORDERING, and this checks whether it is a property "
    "of the actions or a property of one seed. Those are indistinguishable from inside a "
    "single run, and only one of them is worth saying out loud. Every variant is "
    "regenerated at several seeds with everything else held fixed, and what matters is "
    "not whether a variant's number moves -- it is allowed to -- but whether the ranking "
    "survives. If it does, `counterfact` is stating a fact about the model. If it does "
    "not, `counterfact` is stating a fact about seed 0, and should say so."
)


DETAIL_EXPLAINER = (
    "A control for the round-trip result. `cycle` shows inverse dynamics reading a "
    "generated clip about three times worse than the real clip of the same event, and "
    "there are two very different reasons that could happen. Either the generated video "
    "carries less information, in which case you wait for a better generator; or the "
    "generated video is perfectly legible and simply is not camera footage, in which "
    "case the inverse model is out of distribution and you fine-tune it, which is cheap "
    "and possible now. So the real clip is blurred until it has exactly the generated "
    "clip's sharpness and read by the same model. If that middle bar lands on the "
    "dreamed bar, lost detail is a sufficient explanation. If it stays down by the real "
    "one, it is not. Note the limit: blur matches spatial detail only, and generated "
    "video also differs in temporal coherence and artifacts, so this is strong evidence "
    "in one direction and suggestive rather than conclusive in the other."
)


ODYSSEY_EXPLAINER = (
    "An agent acting inside a world that is being dreamed around it, and both are the "
    "same network. At every step the model is given one frame and a task, and policy "
    "mode denoises the action channel and the pixel channel together: it decides what "
    "the robot should do and renders the consequence of having done it, in one pass. "
    "The last frame goes back in and it does it again. Nobody supplies actions, nobody "
    "supplies a physics engine, nobody supplies an environment. The subtitles are the "
    "model's own description of what it just did, generated live by the understanding "
    "surface of the same checkpoint while the generation surface stays loaded. Two "
    "failures to watch for, which look identical in the video and are completely "
    "different: the action magnitude falling to zero is the AGENT giving up, and the "
    "inter-frame motion falling to zero is the RENDERER freezing. And watch the point "
    "where the subtitle stops matching the picture, which is the same content drift "
    "`judge` measures, except here you can see the model narrating a scene it has "
    "already left. Compare with topics/dreamerv3, where the world model is small, "
    "learned from one agent's own experience, and the actor is trained by "
    "backpropagating through imagined rollouts; here nothing is trained at all."
)


EMBODIMENTS_EXPLAINER = (
    "diffusers lists fifteen action embodiments for Cosmos 3. That table is what the "
    "ARCHITECTURE accepts, not what these weights were trained on, and the gap is not "
    "academic: `pusht` runs without error, generates faster than anything else, and "
    "replaces PushT with a photorealistic robot arm on a wooden desk. Nothing in the "
    "documentation says which of the fifteen this checkpoint actually saw, so this "
    "measures it. Each case pulls a conditioning frame, the actions beside it, and the "
    "ground-truth continuation from the same moment of the same real robot episode, so "
    "'what should have happened next' is not a matter of opinion: the model's "
    "continuation sits next to the real one. `pusht` is included as a NEGATIVE CONTROL "
    "and is expected to fail. An earlier version of this survey scored action "
    "sensitivity instead and gave all fifteen a passing number including pusht, which "
    "is exactly the mistake a negative control catches and nothing else does."
)


TRAIN_EXPLAINER = (
    "Stage 6 of the GR00T-Dreams pipeline, the one every other task here sets up and "
    "stops short of: a policy trained on generated video and tested on real video. The "
    "design holds everything fixed except the pixels. Both training sets carry the SAME "
    "action labels, because each dreamed window was generated from the actions it is "
    "labelled with, so the label noise `cycle` priced at about 3x is deliberately pinned "
    "at zero here. Identical network, identical seed, identical batch order. Both "
    "policies are then scored on the same held-out REAL frames neither has seen. The gap "
    "between the two curves is therefore the pixel domain gap on its own, which is the "
    "number a synthetic data programme is actually buying or paying. Read the moving "
    "channels rather than the overall mean: a DROID action is a mixed vector, most of "
    "whose channels barely move in a given window, and averaging over all ten makes a "
    "useless policy look respectable."
)
