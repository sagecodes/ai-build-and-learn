"""HTML for the Flyte reports.

Same palette and helpers as topics/cosmos/reports.py, topics/rl-mujoco/reports.py and
topics/isaac-sim/reports.py, so a viewer moving between the world-model demos is not
re-learning the colours. Everything is inline HTML and inline data: URIs; no
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
_WARN = "#e17055"

_TITLE = "V-JEPA 2 - a world model that predicts representations, not pixels"


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
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};padding:20px;'
        f'border-radius:8px;"><h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;">{text}</h3>'


def note(text: str) -> str:
    return (
        f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;line-height:1.55;">'
        f"{text}</p>"
    )


def details(summary: str, body: str) -> str:
    return (
        f'<details><summary style="cursor:pointer;color:{_MUTED};font-family:monospace;">'
        f"{summary}</summary>"
        f'<pre style="font-size:11px;color:{_TEXT};background:{_PANEL};padding:12px;'
        f'border-radius:4px;overflow-x:auto;white-space:pre-wrap;">{html.escape(body)}</pre>'
        f"</details>"
    )


def progress_html(stage: str, detail: str, rows: list[tuple[str, str]]) -> str:
    """Painted while the pod is still working, so the report is never blank.

    A pod that spends its first minutes pulling weights and clips while its report
    stays empty is indistinguishable from a hung pod. Same lesson as the Cosmos and
    DreamerV3 tasks next door.
    """
    return f"<h2>{_TITLE}</h2>" + _panel(stage, _table(rows) + note(detail))


def side_by_side(cells: list[tuple[str, str]], basis: int = 380) -> str:
    """Lay blocks out in a responsive row.

    flex-wrap rather than a grid: Flyte reports render at a width you do not control,
    and two side-by-side clips have to be allowed to become two stacked clips.
    """
    inner = ""
    for label, body in cells:
        inner += (
            f'<div style="flex:1 1 {basis}px;min-width:300px;">'
            f'<h4 style="color:{_HILITE};font-family:monospace;margin:0 0 8px;">{label}</h4>'
            f"{body}</div>"
        )
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{inner}</div>'


def score_table(rows: list[tuple[str, dict]], highlight: str | None = None) -> str:
    """The prediction scorecard: one row per method, chance floors included.

    Columns are in the order you should read them, worst metric first. `cos` is the
    intuitive one and the least trustworthy: it is not comparable between masks, since
    it depends on how many tokens were hidden and which. `top-1` and the median
    displacements are measured against candidates drawn from the masked tokens only.
    The last column, `time localised`, is the one that settles the argument: the
    fraction of the chance-level temporal error the prediction removed, so 1.00 means
    it found the right moment and 0.00 means it did no better than shuffling.
    """
    head = (
        f'<tr style="color:{_HILITE};">'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">{c}</th>'
            for c in ("method", "cosine", "top-1", "median dt", "median dh", "median dw",
                      "time localised")
        )
        + "</tr>"
    )
    body = ""
    for name, s in rows:
        strong = highlight is not None and name == highlight
        colour = _ACCENT if strong else _TEXT
        weight = "bold" if strong else "normal"
        cells = "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};'
            f'font-weight:{weight};">{v}</td>'
            for v in (
                name,
                f"{s['cos']:.3f}",
                f"{s['top1']:.1%}",
                f"{s['dt']:.1f}",
                f"{s['dh']:.1f}",
                f"{s['dw']:.1f}",
                f"{s['loc']:.2f}" if "loc" in s else "-",
            )
        )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}</table>'
    )


def verdict(text: str, good: bool = True) -> str:
    colour = _ACCENT if good else _WARN
    return (
        f'<div style="font-family:monospace;border-left:3px solid {colour};background:{_PANEL};'
        f'padding:12px 16px;margin:12px 0;color:{_TEXT};font-size:13px;line-height:1.6;">'
        f"{text}</div>"
    )


def final_html(subtitle: str, rows: list[tuple[str, str]], body: str, explainer: str = "") -> str:
    out = f"<h2>{_TITLE}</h2>" + _panel(subtitle, _table(rows))
    if explainer:
        out += _panel("What to look for", note(explainer))
    return out + "<br/>" + body


# ── Explainers ──────────────────────────────────────────────────────────────────
#
# Kept in the report rather than only in the README, because the report is the artifact
# people scroll past on the stream and a heatmap with no claim attached to it is just a
# picture.

INPAINT_EXPLAINER = (
    "V-JEPA 2 has no decoder. The predictor emits 1024-dimensional vectors, and there "
    "is no head in any released checkpoint that turns one back into a pixel, so "
    "'show me what it predicted' is not a screenshot anyone can take. What the video "
    "shows instead is the model's ACTUAL input with the masked patches blacked out, "
    "and next to it a per-patch measurement of how close the predicted vector landed "
    "to the true one. "
    "The argument is the two masks. A TUBE mask removes a spatial block across the "
    "whole clip, which is the shape V-JEPA 2 was pretrained on. A FUTURE mask removes "
    "everything after a moment in time, which it never saw: pretraining masks always "
    "span all of time, so no token was ever predicted from a strictly earlier one. "
    "Read the median dt column. Under the trained mask the predictor's nearest "
    "retrieved token is at the RIGHT MOMENT; asked to extrapolate forward it lands "
    "several tubelets away, which is a representation of roughly the right scene at "
    "the wrong time. That gap is the honest boundary of what this checkpoint is, and "
    "it is what the action-conditioned V-JEPA 2-AC post-train exists to close."
)

PROBE_EXPLAINER = (
    "Nothing here is fine-tuned. The encoder is frozen, every clip becomes one mean-"
    "pooled vector, and the only trained parameters are a single linear layer. So the "
    "accuracy is a statement about the representation, not about the classifier. "
    "Read it against the two floors rather than on its own: chance is 20% for five "
    "balanced classes, and the pixel baseline is the same probe trained on a "
    "downsampled space-time thumbnail of the same clips, which is how much of the "
    "score you get for free because the classes look different. The retrieval row is "
    "the version with no training at all: are clips of the same action each other's "
    "nearest neighbours in a space nobody supervised?"
)

SCALE_EXPLAINER = (
    "Same clips, same masks, same probe, three encoder sizes. Worth separating the two "
    "questions this answers. Probe accuracy asks whether a bigger self-supervised "
    "encoder carries more linearly accessible semantics. The inpainting scores ask "
    "whether its predictor is better at the job it was trained on. These do not have "
    "to move together, and the interesting outcome is if they don't."
)

GEOMETRY_NOTE = (
    "Every similarity in this report is computed on CENTERED features (the per-clip "
    "mean token subtracted before normalising). This is not a stylistic choice. "
    "V-JEPA 2's token cloud sits in a narrow cone: the mean cosine between two random "
    "patches of the same clip is measured below, and raw it is dominated by a "
    "component every token shares. Centered, random pairs sit at ~0.00 while adjacent "
    "patches sit around 0.33, which is a geometry you can actually measure against."
)


# ── V-JEPA 2-AC: planning ───────────────────────────────────────────────────────

PLAN_EXPLAINER = (
    "This is the demo the rest of this repo could not do. The pretrained V-JEPA 2 "
    "predictor inpaints and cannot extrapolate forward in time, because its "
    "pretraining masks always spanned all of time. V-JEPA 2-AC is the action-"
    "conditioned post-train, and it IS a forward model: give it one frame's tokens, "
    "a 7-DoF end-effector delta and the arm's pose, and it returns the tokens of the "
    "next frame. "
    "So planning needs no reward function, no training and no decoder. Photograph the "
    "goal, encode it once, and search for the action sequence whose IMAGINED latent "
    "lands closest to it. Every number the planner sees is an L1 distance between two "
    "embeddings; it never sees the target's coordinates, and no gradient is taken "
    "anywhere. "
    "The controls are the whole argument. An arm drifting toward a block is easy to "
    "produce by accident, so `random` walks with the same step budget and `oracle` is "
    "told the answer outright. Planning is only interesting in the gap between them. "
    "`greedy` drops the search to a single-step grid minimum, which separates 'the "
    "energy landscape is informative' from 'the planner is doing something with it'. "
    "One caveat stated up front: V-JEPA 2-AC was trained on DROID, real video of real "
    "Franka arms. A MuJoCo render is out of distribution for its encoder, and this "
    "report measures that rather than hoping about it."
)

DREAM_EXPLAINER = (
    "A world model that can only be scored is hard to believe in, so this panel makes "
    "the dream watchable without inventing a decoder. "
    "The left video is NOT a reconstruction. Each dreamed latent is matched against a "
    "bank of a few hundred real renders of the arm at real positions, and what you see "
    "is the nearest one. The model imagines an embedding; we show the closest real "
    "photograph to it and print how close that was. The right video is the same action "
    "sequence executed in the simulator. "
    "Watch the retrieval distance under the left panel. While it stays low the dream "
    "corresponds to a real reachable scene and the left video tracks the right one. "
    "When it climbs, the dream has drifted somewhere no real frame lives, and the left "
    "panel should be read as the model failing rather than as the model predicting. "
    "The divergence chart puts the same thing on a number line against the only floor "
    "worth quoting: the distance between two unrelated real frames, which is what a "
    "dream that has stopped meaning anything converges to."
)


def policy_table(eps: dict, threshold: float) -> str:
    """One row per policy. Ordered worst-to-best reading: the floor is the context."""
    head = (
        f'<tr style="color:{_HILITE};">'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">{c}</th>'
            for c in ("policy", "start", "final", "closest", "gap closed", "reached", "sec")
        )
        + "</tr>"
    )
    body = ""
    for name in ("jepa", "greedy", "lookahead", "oracle", "random"):
        ep = eps.get(name)
        if ep is None:
            continue
        strong = name == "jepa"
        colour = _ACCENT if strong else (_MUTED if name == "random" else _TEXT)
        reached = ep.best_dist <= threshold
        cells = "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};'
            f'font-weight:{"bold" if strong else "normal"};">{v}</td>'
            for v in (
                name,
                f"{ep.start_dist * 100:.1f} cm",
                f"{ep.final_dist * 100:.1f} cm",
                f"{ep.best_dist * 100:.1f} cm",
                f"{ep.closed:+.0%}",
                "yes" if reached else "no",
                f"{ep.seconds:.0f}",
            )
        )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}</table>'
    )


def seed_table(rows: list[tuple[int, dict]], threshold: float) -> str:
    """Per-seed gap closed, because one episode is an anecdote."""
    names = ("jepa", "greedy", "lookahead", "oracle", "random")
    head = (
        f'<tr style="color:{_HILITE};">'
        + '<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">seed</th>'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">{n}</th>'
            for n in names
        )
        + "</tr>"
    )
    body = ""
    for seed, eps in rows:
        cells = f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{_TEXT};">{seed}</td>'
        for n in names:
            ep = eps.get(n)
            v = f"{ep.closed:+.0%}" if ep is not None else "-"
            colour = _ACCENT if n == "jepa" else (_MUTED if n == "random" else _TEXT)
            cells += (
                f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};">{v}</td>'
            )
        body += f"<tr>{cells}</tr>"
    import numpy as _np

    means = f'<td style="padding:6px 10px;color:{_HILITE};font-weight:bold;">mean</td>'
    for n in names:
        vals = [e[n].closed for _, e in rows if n in e]
        means += (
            f'<td style="padding:6px 10px;color:{_HILITE};font-weight:bold;">'
            f'{_np.mean(vals):+.0%}</td>' if vals else '<td>-</td>'
        )
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}<tr>{means}</tr></table>'
    )


ADAPT_EXPLAINER = (
    "The `plan` task ends on a split verdict: V-JEPA 2's encoder transfers to a "
    "simulator it has never seen and its action-conditioned predictor does not. This "
    "task asks whether the broken half is expensive to fix. It is not. "
    "The encoder is frozen -- it is the part that already works, and training it would "
    "risk the one thing the demo has established. The arm is driven around with random "
    "actions from random workspace positions, every transition is encoded once, and "
    "only the 305M-parameter predictor is fine-tuned, on exactly the L1 objective it "
    "was originally trained with. No new losses, no tricks. "
    "Read the learning curve against the dashed line. That line is the error you get "
    "by predicting that nothing happens, and the pretrained checkpoint sits ABOVE it: "
    "on these images, standing still is a better forecast than the world model. "
    "Crossing below it is the moment the predictor becomes worth consulting, and the "
    "before/after video is what that looks like in the arm. "
    "Worth noting what was tried first and rejected: blur, JPEG artifacts and sensor "
    "noise all failed. Noise in particular moved the action-ranking correlation from "
    "+0.09 to +0.61 and then made closed-loop planning WORSE, which is a good reminder "
    "that a correlation measured over a handful of candidate actions at one state is "
    "not the quantity anyone cares about."
)


READOUT_EXPLAINER = (
    "Everywhere else this demo says V-JEPA 2 has no decoder, so you cannot be shown "
    "what it predicted. This task asks whether that is a packaging problem or a fact "
    "about the representation, by fitting the best possible LINEAR map from one patch "
    "token back to the 2x16x16 patch of pixels it came from. Closed-form ridge "
    "regression, so there is no learning rate, no step count and nothing to blame. "
    "The middle panel of the video is the one that makes the result readable. It is "
    "the SAME map fitted to a random 1024-dimensional projection of the true pixels, "
    "which is very nearly lossless and therefore shows what 'linearly decodable' looks "
    "like when the information really is present. It comes back sharp. The right-hand "
    "panel, fitted the same way to V-JEPA's tokens, does not. "
    "V-JEPA scores below the trivial baseline of painting each patch its own average "
    "colour, and barely above being paired with the wrong patch altogether. The "
    "appearance is not in the token. "
    "That is the JEPA argument working as designed rather than failing: the reason to "
    "predict in representation space is that pixels are mostly unpredictable detail, "
    "and a model that refuses to spend capacity on them has it free for structure. "
    "The second chart is the other half of the sentence. The same tokens that cannot "
    "reproduce a 16x16 square carry enough for 78% five-way action recognition from a "
    "single linear layer, where the same probe on raw pixels manages 40% and chance is "
    "20%. Appearance discarded, meaning kept."
)


def readout_table(rows: dict) -> str:
    """Linear decodability per condition, ordered best to worst."""
    names = {
        "rand": ("random projection of the true pixels", "the ceiling: the method works"),
        "grey": ("each patch's own mean colour", "knowing only the average brightness"),
        "vjepa": ("V-JEPA 2 patch tokens", "what the model actually keeps"),
        "shuf": ("V-JEPA tokens, wrong patches", "the floor"),
    }
    head = (
        f'<tr style="color:{_HILITE};">'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">{c}</th>'
            for c in ("readout from", "PSNR", "R2", "what it is")
        )
        + "</tr>"
    )
    body = ""
    for key in ("rand", "grey", "vjepa", "shuf"):
        if key not in rows:
            continue
        label, note_ = names[key]
        strong = key == "vjepa"
        colour = _ACCENT if strong else (_MUTED if key == "shuf" else _TEXT)
        r2 = f"{rows[key]['r2']:+.3f}" if "r2" in rows[key] else "-"
        cells = "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};'
            f'font-weight:{"bold" if strong else "normal"};">{v}</td>'
            for v in (label, f"{rows[key]['psnr']:.2f} dB", r2, note_)
        )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}</table>'
    )


PUSH_EXPLAINER = (
    "<b>How this task is posed.</b> The robot is shown ONE photograph: the same scene, "
    "from the same camera, with the job already finished. In it the red block sits "
    "somewhere it is not now. That photograph is the entire specification. Nobody tells "
    "the arm where the block is, where it should end up, or that a block is involved at "
    "all. There is no reward function, no waypoints and no demonstration to imitate. "
    "<b>How it gets there.</b> At every step the planner asks a single question: of the "
    "27 small movements I could make next, which one makes the camera image most like "
    "the reference photograph? To answer it, each candidate is actually carried out in "
    "a copy of the simulator, the result is photographed, and the two pictures are "
    "compared in the chosen space. The winner is executed for real, the copy is thrown "
    "away, and the question is asked again from wherever the arm now is. Twenty-two "
    "times. That is the whole algorithm: photograph, compare, step, repeat. "
    "<b>Why the comparison space is the experiment.</b> Exactly one thing changes "
    "between the runs below: how 'most like the photograph' is measured. V-JEPA 2 "
    "compares learned embeddings; `pixels` subtracts the two images; `random net` uses "
    "an untrained network of the identical architecture; `random` ignores the "
    "photograph entirely. Same arm, same scenes, same search, same 22 steps. "
    "<b>Why this task and not reaching.</b> In the reach task the goal photo differed "
    "from the start photo only in where the ARM was, so any image distance was monotone "
    "in arm position and raw pixels scored identically to V-JEPA (+90.4% each). Moving "
    "an object asks a different question, and the score below is the BLOCK's progress "
    "toward where the photograph shows it, never the arm's. A policy that poses the arm "
    "to match the picture without pushing anything scores zero."
)


def push_table(rows: list[tuple[str, dict]]) -> str:
    """One row per comparison space, scored on the block rather than the arm."""
    head = (
        f'<tr style="color:{_HILITE};">'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">{c}</th>'
            for c in ("compares images using", "block progress", "block moved", "scenes solved")
        )
        + "</tr>"
    )
    body = ""
    for name, s in rows:
        strong = name.startswith("V-JEPA")
        colour = _ACCENT if strong else (_MUTED if "random actions" in name else _TEXT)
        cells = "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};'
            f'font-weight:{"bold" if strong else "normal"};">{v}</td>'
            for v in (
                name,
                f"{s['closed']:+.0%} +- {s['sd']:.0%}",
                f"{s['moved_cm']:+.1f} cm",
                f"{s['solved']}/{s['n']}",
            )
        )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}</table>'
    )


def reference_panel(start_b64: str, goal_b64: str) -> str:
    """The two pictures that define the task, side by side and captioned as such."""
    return side_by_side([
        ("What the robot sees at step 0", start_b64),
        ("The reference photograph it is given", goal_b64),
    ], basis=300)


# ── The mechanism tasks: occlude, energy, ladder, collapse ─────────────────────


def data_table(headers: list[str], rows: list[list[str]], highlight: int | None = None,
               warn: tuple[int, ...] = ()) -> str:
    """A generic table. `highlight` and `warn` colour whole rows by index."""
    head = (
        f'<tr style="color:{_HILITE};">'
        + "".join(
            f'<th style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;">'
            f"{c}</th>" for c in headers
        )
        + "</tr>"
    )
    body = ""
    for i, row in enumerate(rows):
        colour = _ACCENT if i == highlight else (_WARN if i in warn else _TEXT)
        weight = "bold" if i == highlight else "normal"
        body += "<tr>" + "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #2a2a3e;color:{colour};'
            f'font-weight:{weight};">{c}</td>' for c in row
        ) + "</tr>"
    return (
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:12px;'
        f'width:100%;color:{_TEXT};">{head}{body}</table>'
    )


OCCLUDE_EXPLAINER = (
    "This is the one report in the demo that renders the prediction, and the rule that "
    "makes it honest is the CEILING PANEL. V-JEPA 2 has no decoder, so a predicted "
    "token can only be turned into pixels by (a) retrieving the nearest token in a bank "
    "built from other clips and pasting the real pixels that token came from, or (b) "
    "the closed-form linear map from the `readout` task. Both of those are lossy before "
    "the predictor is involved at all, so each is shown twice: once on the PREDICTED "
    "token and once on the ENCODER'S OWN token at the same position. The second panel "
    "is what the method can express; the first is how much of that the predictor got. "
    "A shuffled-prediction panel gives the floor. "
    "The bank and the linear map are fitted on TRAIN clips and the clip shown is a VAL "
    "clip, so nothing can retrieve the literal answer. "
    "The argument is then the three masks, all hiding the same number of tokens and "
    "differing only in shape: a hole that MOVES (so each hidden position is visible at "
    "other moments), a hole that does not (so those positions are hidden for the whole "
    "clip), and a hole covering the end of the clip (which the pretraining masks never "
    "did). If prediction quality falls in that order, then 'V-JEPA 2 interpolates and "
    "does not extrapolate' is a measured progression rather than a slogan."
)

ENERGY_EXPLAINER = (
    "JEPA is an energy-based model: E(x, y) = distance between the predictor's output "
    "given the visible context x and the encoder's embedding of a candidate completion "
    "y, in the L1 the model was actually trained on. Low energy means 'this is a "
    "plausible completion'. Nothing in JEPA's training pushes any energy UP, so "
    "whether the result is a well-shaped energy function is an empirical question. "
    "The well is the first answer: roll the candidate forward and backward in time and "
    "the energy should be a V with its minimum at zero offset. A flat line would mean "
    "the model located nothing. "
    "The candidate ladder is the second. Read it from the bottom up, and read it for "
    "the failure that matters: not whether the true completion is lowest, but whether "
    "anything DEGENERATE is lower. Flat grey and a frozen first frame are the cheapest "
    "possible answers to 'what is behind the occluder', and if either beats the truth "
    "then this energy could not be used to generate a completion no matter how good "
    "its retrieval scores look. The two dashed lines are the no-model floors: the same "
    "prediction with its pairing shuffled, and the mean of the visible tokens."
)

LADDER_EXPLAINER = (
    "The same two measurements at every layer of one forward pass. On the left axis, "
    "how much of a 16x16 patch a linear map can still recover from a token at that "
    "depth, between the two references that give it meaning: a random projection of the "
    "true pixels (which inverts almost perfectly, so it says what linearly decodable "
    "means) and tokens paired with the wrong patches (the floor). On the right axis, "
    "how well one linear layer names the action from that depth's mean-pooled features. "
    "If the first curve falls while the second rises, 'appearance discarded, meaning "
    "kept' has a location in the network as well as a number. "
    "Two details are load-bearing. Intermediate layers are taken before the encoder's "
    "final LayerNorm, whose learned affine is calibrated for the last layer, so the "
    "final layer appears twice, raw and normed. And the predictor is trained to match "
    "the FINAL layer, so if the semantic peak is earlier than that, the representation "
    "the world model predicts is not the network's best description of the scene."
)

COLLAPSE_EXPLAINER = (
    "Nothing here is V-JEPA 2. This is a ~0.9M parameter JEPA trained from scratch on "
    "synthetic video, four ways, to show the failure mode the pretrained checkpoint "
    "cannot show you because Meta already avoided it. "
    "'Predict the representation of the hidden part' is solved perfectly by a constant "
    "representation: emit the same vector for everything and the loss is zero. Nothing "
    "in the loss forbids it, which is why the targets in a real JEPA come from an "
    "exponential moving average of the encoder with no gradient flowing into them. "
    "The four arms differ ONLY in where the target comes from: an EMA copy, the "
    "encoder itself detached, the encoder itself with gradients (the collapse arm), or "
    "the true pixels (a masked autoencoder, the thing JEPA argues against). Same data, "
    "same masks, same steps, same shapes. "
    "Read the loss curve first and notice that the collapsed arm wins it. Then read the "
    "pair-cosine curve, where 1.00 means the encoder returns the same direction "
    "whatever it is shown, and the probe table, where COLOUR and SHAPE are visible in "
    "one frame and DIRECTION OF MOTION is not in any single frame and can only come "
    "from the video. The untrained row is there so that none of the others can take "
    "credit for what the architecture gives away for free."
)
