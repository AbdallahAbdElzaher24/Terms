"""Speech-to-text — Whisper, run locally through faster-whisper
(CTranslate2 backend: much lower RAM/VRAM and faster than the stock
openai-whisper package for the same accuracy).

Defaults to the "small" size (~250MB) rather than "large-v3" (~3GB) —
audio isn't this app's primary input path (contracts are), so this keeps
the download light while still handling Arabic/English reasonably well.
Set WHISPER_MODEL=large-v3 in .env if you have the disk/RAM/latency budget
and want the higher accuracy ceiling, or WHISPER_MODEL=base/tiny for an
even smaller footprint.

First call downloads the weights for whichever size you pick — needs
internet once, then runs fully offline.
"""
from dataclasses import dataclass
from functools import lru_cache
import os

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")  # "cuda" if you have a GPU
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # int8 is fine on CPU, float16 on GPU


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel  # imported lazily — downloads weights on first use

    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe(path: str, language: str | None = None) -> tuple[str, list[TranscriptSegment]]:
    """language: None lets Whisper auto-detect (handles Arabic/English switching
    within one clip reasonably well); pass 'ar' or 'en' to force one."""
    model = _get_model()
    segments, info = model.transcribe(path, language=language, vad_filter=True)
    out_segments = [TranscriptSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
    full_text = " ".join(s.text for s in out_segments)
    return full_text, out_segments
