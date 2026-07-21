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


SUITES: dict[str, list[Script]] = {
    "full": SUITE,
    "quick": QUICK,
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
