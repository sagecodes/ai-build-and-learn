# Reference voices

One `.wav` plus one `.txt` per voice, same stem: `sage.wav` + `sage.txt`. The `.txt` is
the **exact** transcript of the wav, and it is load-bearing, not documentation: Qwen,
Dia and CSM all condition on it, so three of the five cloners degrade if it drifts from
what was actually said. That is why the reference is a fixed script read aloud rather
than a found clip that gets Whisper-transcribed: an ASR error in the reference silently
becomes a worse clone in every model that reads it.

## Recording one

Read `sage.txt` (or write your own and save both files) and save as:

- **mono WAV**, 24kHz or higher
- **8-15 seconds.** Every model here accepts 3s, all of them clone better at 8-15s,
  and most only read the first ~15s, so past ~30s you are storing audio nobody uses.
- No music, no room echo, no background noise. The models clone the *recording* as much
  as the voice: reverb in the reference comes back as reverb in every clip.
- Do not normalize to full scale. A clipped reference distorts the cloned timbre, and
  `RefVoice.warnings()` will call it out in the report.

The pipeline surfaces these as warnings on the report's reference card rather than
failing, so a marginal clip still produces a run you can look at.

## The wavs are gitignored on purpose

`.gitignore` here excludes `*.wav`. A reference clip is a **voiceprint**: it is the
exact input someone else would need to clone that voice with this same pipeline, and
committing one to a public repo hands it over permanently, in a repo whose whole point
is demonstrating how easy the cloning is. The transcripts are committed (they are just
sentences), the audio is not.

If you deliberately want a shareable reference so the demo runs for other people, use a
public-domain clip with a clear license (LibriSpeech, or a Common Voice sample) and
force-add it with a note about where it came from. Do not commit anyone's voice,
including your own, without deciding you meant to.
