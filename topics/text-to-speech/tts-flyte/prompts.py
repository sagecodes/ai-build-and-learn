"""Scripts for the text-to-speech comparison.

The video demo's prompts each targeted a failure mode that only exists in *time*.
The TTS equivalent targets failure modes you can only hear: the things that separate
a model that "says the words" from one that sounds like a person read them.

Every script here stresses one axis:

  - **Naturalness / prosody.** Plain narration with commas and a full stop. The
    baseline: does the rhythm sound human, or like a word-by-word reader? If a model
    fails this, nothing else matters.
  - **Text normalization.** Numbers, a date, a time, a currency amount, an acronym,
    a URL. TTS models don't read digits; they have to *expand* them, and this is
    where cheap models say "one thousand two hundred three dollars" as "one two zero
    three dollars" or spell out "NASA" letter by letter. The clearest quality tell.
  - **Questions and emphasis.** Rising intonation on a question, stress on an
    italicized word. Flat models read every sentence with the same contour.
  - **Hard phonemes.** A tongue-twister plus a couple of commonly-mangled words.
    Listen for slurred consonant clusters and wrong lexical stress.
  - **Emotion.** An excited line with an exclamation. Expressive models (Chatterbox,
    Dia) should audibly lift; Kokoro will stay level. That gap is the point.
  - **Dialogue.** A two-speaker exchange with a nonverbal (a laugh). This is Dia's
    home turf and most single-voice models can't do it at all, which is itself the
    comparison: it shows what "dialogue TTS" buys you.

Ordered short-and-easy first, so a truncated run still yields a usable grid, and the
first cell is the fast one that always works.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Script:
    text: str
    axis: str        # the capability under test
    listen_for: str  # the specific thing to judge in the report


SUITE: list[Script] = [
    Script(
        text="The morning light came slow and gold across the harbor, and for a long "
             "moment, nobody said anything at all.",
        axis="baseline naturalness + prosody",
        listen_for="Our control. Every model should sound plausibly human here. Listen "
                   "for the pause at the comma and a falling pitch into the full stop. "
                   "A flat, word-by-word read means the model has no prosody model, and "
                   "everything below will be worse.",
    ),
    Script(
        text="Your order of 3 items ships on March 14th and arrives by 9:30 AM; "
             "the total was $1,203.50, charged to card ending 0042. Track it at "
             "acme.co/orders.",
        axis="text normalization (numbers, date, time, currency, URL)",
        listen_for="The single clearest quality separator. Does it say 'March "
                   "fourteenth', 'nine thirty A M', 'one thousand two hundred three "
                   "dollars and fifty cents', 'zero zero four two', and read the URL "
                   "as 'acme dot co slash orders'? Cheap models mangle at least one.",
    ),
    Script(
        text="Wait, you actually finished the whole thing? That's incredible, I "
             "genuinely did not think it was possible.",
        axis="questions + emphasis + excitement",
        listen_for="Rising intonation on the question, a real lift on 'incredible', "
                   "stress on 'genuinely'. Expressive models (Chatterbox, Dia) should "
                   "audibly react; Kokoro and the smaller Qwen will read it more level. "
                   "That gap is exactly what you're comparing.",
    ),
    Script(
        text="She sells seashells by the seashore; the specific statistics she cited "
             "were both thorough and thoroughly unusual.",
        axis="hard phonemes + consonant clusters + lexical stress",
        listen_for="Slurred 's'/'sh' clusters, and whether 'thorough' vs 'thoroughly' "
                   "keep the right stress. This is where small models smear the "
                   "consonants together.",
    ),
    Script(
        text="[S1] Okay but did it actually work? [S2] It worked on the first try. "
             "(laughs) I could not believe it either. [S1] No way. You have to show me.",
        axis="two-speaker dialogue + a nonverbal",
        listen_for="Dia's home turf: two distinct voices and an actual laugh on "
                   "'(laughs)'. Single-voice models will read the '[S1]/[S2]' and "
                   "'(laughs)' tags aloud or flatten it to one narrator, which is the "
                   "comparison, it shows what dialogue TTS actually buys you.",
    ),
]


# A 3-script quick pass: the control, the normalization torture test, and the emotion one.
QUICK: list[Script] = [SUITE[0], SUITE[1], SUITE[2]]


# ── The cloning suite ────────────────────────────────────────────────────────────
#
# Different job, so different scripts. The compare suite asks "is this good speech";
# the clone suite asks "is this still THEM, and did it say the words". Each script
# targets a way a clone specifically comes apart, which the SIM/WER pair can measure:
#
#   - identity DRIFT over a long utterance (SIM computed on the whole clip hides it,
#     so this one is deliberately long enough to hear the voice wander),
#   - identity loss under EXPRESSION, the most common real failure: the clone holds up
#     on flat narration and snaps back to the model's own default voice the moment the
#     line needs feeling,
#   - generalization to phonemes the reference recording never contained,
#   - normalization, which is now objectively scored instead of judged by ear.
#
# No dialogue script here: a two-speaker line is incoherent when the whole point is
# that every word comes out in one specific person's voice.
CLONE_SUITE: list[Script] = [
    Script(
        text="I was going to call you back, but the whole afternoon got away from me.",
        axis="control: short, neutral, conversational",
        listen_for="The baseline every other row is read against. Short and prosodically "
                   "easy, so whatever similarity score a model posts here is roughly its "
                   "ceiling. If a model is already unrecognizable on this line, its "
                   "scores below are noise.",
    ),
    Script(
        text="Honestly? That is the single best news I have heard all month, and I am "
             "not even exaggerating!",
        axis="identity under expression",
        listen_for="The failure mode that matters most in practice. Models that clone "
                   "timbre but not identity hold the voice on flat narration and lose it "
                   "the moment the line gets excited: the pitch lifts and it becomes the "
                   "model's own default speaker again. Watch for a similarity score that "
                   "drops sharply against the control row.",
    ),
    Script(
        text="The point I keep coming back to is that none of this was obvious at the "
             "start, and if you had asked me a year ago I would have told you the "
             "opposite, with total confidence, and I would have been completely wrong.",
        axis="identity drift over a long utterance",
        listen_for="Long enough for the voice to wander. Clones often start accurate and "
                   "decay toward a generic voice as the reference falls out of the "
                   "attention window. Listen to the first and last clauses back to back; "
                   "a whole-clip similarity score averages this away.",
    ),
    Script(
        text="Rural jurors squarely judged the thorough authenticity of the sixth "
             "witness statement.",
        axis="phonemes the reference never contained",
        listen_for="Zero-shot cloning has to generalize the speaker beyond the sounds in "
                   "the reference clip. Clustered r/j/th/x sounds are where a thin clone "
                   "audibly falls back on the base model's articulation.",
    ),
    Script(
        text="Call me back at 555 0147 before 4:30, the invoice came to $2,840.75.",
        axis="text normalization, now objectively scored",
        listen_for="The compare demo judged this by ear; here Whisper transcribes it and "
                   "the word error rate scores it. This is also the row where SIM and WER "
                   "most often disagree: a model can say this in a perfect clone of the "
                   "voice while getting the digits wrong.",
    ),
]

# Two rows: the control and the expression one, the pair whose SIM gap is the headline.
CLONE_QUICK: list[Script] = [CLONE_SUITE[0], CLONE_SUITE[1]]


SUITES: dict[str, list[Script]] = {
    "full": SUITE,
    "quick": QUICK,
    "clone": CLONE_SUITE,
    "clone-quick": CLONE_QUICK,
}


def get_suite(name: str) -> list[str]:
    """Suite name -> plain text strings (what the pipeline actually takes)."""
    try:
        return [s.text for s in SUITES[name]]
    except KeyError:
        raise ValueError(f"Unknown suite {name!r}. Known: {', '.join(SUITES)}") from None


def describe(name: str = "full") -> str:
    """The suite as a readable table, for the README and the run's report header."""
    rows = []
    for i, s in enumerate(SUITES[name], 1):
        rows.append(f"{i}. [{s.axis}]\n   {s.text}\n   -> {s.listen_for}")
    return "\n\n".join(rows)


if __name__ == "__main__":
    print(describe("full"))
