"""Objective scores for a cloned voice: does it sound like them, and is it intelligible?

Flyte-free, like tts_core, so run_local and the studio can score a clip too.

The compare demo could only ever say "listen and judge". Cloning can do better, because
it has a ground truth on both axes:

  SIM  speaker similarity. Cosine distance between x-vector speaker embeddings of the
       reference and the clone. This is the standard SIM-o metric: it asks "is this the
       same person", not "is this good audio".
  WER  word error rate. Transcribe the clone with Whisper and diff against the text the
       model was asked to say. This is the axis SIM cannot see: a model can nail someone's
       timbre while slurring every third word, and a clone that scores 0.9 SIM with 40%
       WER is a worse product than one at 0.8 SIM and 2%.

Report them TOGETHER or they mislead. High SIM alone is the classic way to make a cloning
demo look better than it is.

Both models are transformers-native on purpose: they load in the image the Dia/CSM tasks
already use, so scoring adds no new image to the build.

── On the numbers ───────────────────────────────────────────────────────────────
SIM is not a percentage and does not have an absolute meaning. wavlm-base-plus-sv's own
card gives 0.86 as a same-speaker threshold, but it is dataset-dependent, and TTS output
sits in a different distribution than the human speech the verifier was tuned on. The
number that MEANS something here is the control: we also score the reference against
itself (SIM ~1.0, the ceiling) and, when there is more than one model, each clone
against the others. Read SIM as a ranking, not a grade.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("tts.metrics")

# The speaker verifier. Small (~380MB), transformers-native, and its x-vector head is
# trained for exactly this comparison. 16kHz input is not a preference, it is the rate
# the feature extractor was trained at; feed it 24k and the embeddings are garbage.
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
SPEAKER_SR = 16000

# Whisper for the intelligibility side. large-v3-turbo is the accuracy/speed sweet spot
# (~1.6GB, 4 decoder layers instead of 32); the clips here are a few seconds each, so
# transcription is a rounding error next to synthesis.
ASR_MODEL = "openai/whisper-large-v3-turbo"
ASR_SR = 16000

# The same-speaker threshold from the wavlm-base-plus-sv card. Shown as a reference line
# in the report, NOT used to pass/fail anything: see the note above.
SAME_SPEAKER_THRESHOLD = 0.86


# ── Text normalization for WER ───────────────────────────────────────────────────
#
# WER is only meaningful if both sides are normalized the same way, and the choice of
# normalizer moves the number by several points. We are deliberately LENIENT about
# things a listener would not count as an error (case, punctuation, hyphens) and strict
# about everything else. Numbers are the interesting case: the reference text says "24"
# and Whisper writes "24", but a TTS model that correctly says "twenty-four" will have
# Whisper write "24" too, so digits are left alone rather than spelled out.

_PUNCT = re.compile(r"[^\w\s']")
_TAGS = re.compile(r"\[S\d+\]|\((?:laughs?|sighs?|coughs?|clears throat|gasps?)\)", re.I)


def normalize_for_wer(text: str) -> list[str]:
    text = _TAGS.sub(" ", text)
    text = text.lower().replace("-", " ").replace("'", "'")
    text = _PUNCT.sub(" ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """WER as (rate, n_errors, n_reference_words). Standard Levenshtein over words.

    Hand-rolled rather than pulling in jiwer: it is fifteen lines of DP, and the part
    that actually moves the number is normalize_for_wer above, which we want to own.
    """
    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    if not ref:
        return (0.0 if not hyp else 1.0), len(hyp), 0

    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = prev[j - 1] if r == h else 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    errors = prev[-1]
    return errors / len(ref), errors, len(ref)


# ── Audio prep ───────────────────────────────────────────────────────────────────

def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    import torch
    import torchaudio
    out = torchaudio.functional.resample(torch.from_numpy(wav.astype(np.float32)), sr, target_sr)
    return out.numpy().astype(np.float32)


# ── The scorer ───────────────────────────────────────────────────────────────────

@dataclass
class ClipScore:
    similarity: float = 0.0      # cosine vs the reference speaker embedding, -1..1
    wer: float = 0.0             # 0..1+ (insertions can push it past 1)
    n_errors: int = 0
    n_words: int = 0
    transcript: str = ""         # what Whisper actually heard
    watermarked: bool | None = None   # None = detector unavailable in this image
    error: str = ""

    @property
    def same_speaker(self) -> bool:
        return self.similarity >= SAME_SPEAKER_THRESHOLD


@dataclass
class Scorer:
    """Loads the two scoring models once, then scores many clips.

    Built as a class because the pipeline scores every clip from every model in a single
    task: loading Whisper once and reusing it is the difference between a fast scoring
    pass and paying a model load per clip.
    """
    device: str = "cuda"
    _sv: object = field(default=None, repr=False)
    _sv_fe: object = field(default=None, repr=False)
    _asr: object = field(default=None, repr=False)
    _ref_emb: object = field(default=None, repr=False)

    def load(self) -> "Scorer":
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

        log.info(f"loading speaker verifier {SPEAKER_MODEL}")
        self._sv_fe = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
        self._sv = WavLMForXVector.from_pretrained(SPEAKER_MODEL).to(self.device).eval()

        log.info(f"loading ASR {ASR_MODEL}")
        # fp16 on the GB10; Whisper is stable in half precision and this halves the load.
        self._asr = pipeline(
            "automatic-speech-recognition", model=ASR_MODEL,
            torch_dtype=torch.float16, device=self.device,
        )
        return self

    def embed(self, wav: np.ndarray, sr: int):
        """L2-normalized x-vector for one clip."""
        import torch
        audio = _resample(wav, sr, SPEAKER_SR)
        inputs = self._sv_fe(audio, sampling_rate=SPEAKER_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._sv(**inputs).embeddings
        return torch.nn.functional.normalize(emb, dim=-1).cpu()

    def set_reference(self, wav: np.ndarray, sr: int) -> None:
        """Embed the reference voice once; every clip is scored against this."""
        self._ref_emb = self.embed(wav, sr)

    def transcribe(self, wav: np.ndarray, sr: int) -> str:
        audio = _resample(wav, sr, ASR_SR)
        # Force English: without the hint Whisper will occasionally decide a badly
        # cloned clip is another language and "transcribe" it into gibberish, which
        # scores as ~100% WER for a reason that has nothing to do with the TTS model.
        out = self._asr(audio, generate_kwargs={"language": "english", "task": "transcribe"})
        return (out.get("text") or "").strip()

    def score(self, wav: np.ndarray, sr: int, target_text: str) -> ClipScore:
        import torch
        s = ClipScore()
        try:
            if self._ref_emb is None:
                raise RuntimeError("set_reference() must be called before score()")
            if wav.size < sr * 0.1:      # <100ms: a failed generation, not a clip
                raise RuntimeError(f"clip is only {wav.size / max(sr, 1):.3f}s; nothing to score")

            emb = self.embed(wav, sr)
            s.similarity = float(torch.nn.CosineSimilarity(dim=-1)(self._ref_emb[0], emb[0]))

            s.transcript = self.transcribe(wav, sr)
            s.wer, s.n_errors, s.n_words = word_error_rate(target_text, s.transcript)
            # NB: watermark detection is NOT done here. The detector only exists in the
            # Chatterbox image, so it runs in the generation task (see detect_watermark).
        except Exception as e:                       # one bad clip must not sink the pass
            log.warning(f"scoring failed: {e}")
            s.error = f"{type(e).__name__}: {e}"
        return s


# ── Watermark detection ──────────────────────────────────────────────────────────

def detect_watermark(wav: np.ndarray, sr: int) -> bool | None:
    """Is there a Perth watermark in this clip? None if the detector isn't installed.

    Chatterbox embeds a Resemble Perth watermark in everything it generates, so its
    clones are detectable as synthetic after the fact. That is the honest thing to
    surface in a voice-cloning demo, and it is a genuine capability difference between
    the models here, not a disclaimer: the other four ship no watermark at all, so
    nothing downstream can tell their output from a recording.

    Called from the GENERATION task, not the scoring one, and that is deliberate: the
    only image on the box with `perth` in it is Chatterbox's (it ships as one of that
    package's runtime deps), and adding it to the scoring image would mean building a
    new image just for this. Since metrics.py is Flyte-free it rides into every adapter
    image with the code bundle, so the same call returns a real verdict where the
    detector exists and None everywhere else, at zero build cost.

    Returns None rather than False when `perth` is missing, because "we couldn't check"
    and "we checked and there's no watermark" must not render as the same cell.
    """
    try:
        import perth
    except ImportError:
        return None
    try:
        watermarker = perth.PerthImplicitWatermarker()
        return bool(watermarker.get_watermark(_resample(wav, sr, 44100), sample_rate=44100))
    except Exception as e:
        log.warning(f"watermark detection failed: {e}")
        return None
