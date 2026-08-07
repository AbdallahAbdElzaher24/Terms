"""
Image / scanned-PDF OCR — fully ONNX Runtime based (via rapidocr-onnxruntime),
with a hosted vision-model fallback.

WHY THIS REPLACED PaddleOCR
----------------------------
The previous version of this file drove the `paddleocr` Python package
directly, which pulls in the full PaddlePaddle deep-learning framework
(a multi-hundred-MB install with its own CUDA/MKL wiring) just to run
three small CNN forward passes (detect -> classify orientation -> recognize).
`rapidocr-onnxruntime` ships the exact same PaddleOCR-trained models
(PP-OCRv4 det/cls/rec), pre-converted to ONNX, and runs them with plain
`onnxruntime` — same accuracy, a fraction of the install size, no PaddlePaddle
dependency, and it's the same inference engine every other model in this
project already uses (see legalbert_classifier.py).

    pip install rapidocr-onnxruntime

LANGUAGE SUPPORT
-----------------
- lang="en" (or anything other than "ar"): uses the models bundled straight
  inside the rapidocr-onnxruntime wheel (downloaded from PyPI, nothing else
  needed) — detection (ch_PP-OCRv4_det_mobile), angle classification, and a
  recognizer whose dictionary covers Latin letters/digits/punctuation. Good
  for English contracts out of the box.
- lang="ar": needs one extra local file this repo doesn't ship the weights
  for (see MODELS_ROOT below) — the Arabic recognition model is ~4-8MB and
  isn't bundled in the pip package. Run:

      backend/scripts/download_arabic_ocr_model.py

  once (needs regular internet access — downloads from the same place
  rapidocr's own installer pulls its models from) and it drops
  `arabic_rec.onnx` next to the `arabic_dict.txt` character map already
  checked into this repo at backend/models/ocr/. Until that file exists,
  ocr_image(lang="ar") raises a clear FileNotFoundError (same pattern as
  every other services/legal_ai/* loader in this project) and
  ocr_image_full_text() transparently falls back to the hosted Groq vision
  model below instead of failing the upload.

Detection + orientation-classification are effectively script-agnostic (they
just find "where is text" via stroke-density patterns, not what it says), so
only the recognition model needs to be swapped per language — this is the
same architecture PaddleOCR/RapidOCR use internally.

If rapidocr-onnxruntime isn't installed at all, every OCR call automatically
falls back to Groq's hosted vision model (qwen/qwen3.6-27b by default — set
GROQ_VISION_MODEL in .env). That needs GROQ_API_KEY and internet, but zero
local install.

TRULY MIXED-SCRIPT LINES (lang="mixed" only)
---------------------------------------------
RapidOCR's "ar" and "en" recognizers are each single-script — neither can
correctly read a line where Arabic and English characters are interleaved
with no visual gap the detector can split on (e.g. "المادة 3 - Force
Majeure" run together with no clear break). For those specific lines, this
module re-reads the crop with Tesseract's combined ``ara+eng`` model, which
decodes both scripts in one pass. This only kicks in for boxes where the
Arabic and English detectors landed on (almost) the same region — a strong
signal of a genuinely mixed line, not just two adjacent single-script boxes
that happen to overlap a bit — see `_tesseract_region` / `_dedup_by_position`.

This is optional and fully degrades gracefully: if `pytesseract` isn't
installed, or the `tesseract` binary isn't on PATH, or the Arabic language
data isn't installed, this fallback is silently skipped and behaviour is
exactly what it was before (pick whichever single-script engine was more
confident for that region). To enable it:

    pip install pytesseract
    apt-get install tesseract-ocr tesseract-ocr-ara   # Debian/Ubuntu
"""
import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import re
import time

GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
GROQ_VISION_URL = "https://api.groq.com/openai/v1/chat/completions"

MODELS_ROOT = Path(__file__).parent.parent.parent / "models" / "ocr"
ARABIC_REC_MODEL = MODELS_ROOT / "arabic_rec.onnx"
ARABIC_REC_DICT = MODELS_ROOT / "arabic_dict.txt"  # checked into the repo — just a ~160-line character list

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: list  # 4 corner points, best-first reading order


@lru_cache(maxsize=2)
def _get_engine(lang: str):
    """Returns a cached RapidOCR instance for the given language. Raising
    FileNotFoundError for a missing Arabic model (rather than e.g. silently
    falling back to the Latin recognizer and returning garbage) matches the
    fail-loud-and-clear pattern every model loader in services/legal_ai/*
    already uses — the caller (ocr_image_full_text) is what turns this into
    a graceful degrade to the hosted vision fallback."""
    from rapidocr_onnxruntime import RapidOCR  # imported lazily — heavy-ish import (onnxruntime + opencv)

    if lang == "ar":
        if not ARABIC_REC_MODEL.exists():
            raise FileNotFoundError(
                f"No Arabic OCR recognition model at {ARABIC_REC_MODEL}. Run "
                "`python backend/scripts/download_arabic_ocr_model.py` once (needs internet) "
                "to fetch it — see services/parsing/ocr.py for details."
            )
        if not ARABIC_REC_DICT.exists():
            raise FileNotFoundError(f"Arabic character dictionary missing at {ARABIC_REC_DICT}.")
        # Detection + orientation-classification stay the bundled defaults
        # (script-agnostic); only the recognizer + its character map swap.
        return RapidOCR(rec_model_path=str(ARABIC_REC_MODEL), rec_keys_path=str(ARABIC_REC_DICT))

    # Any non-Arabic language: bundled ch/en models cover Latin script fine.
    return RapidOCR()


def _bbox_top(bbox: list) -> float:
    """Returns the average Y coordinate of the top edge of a bounding box.
    RapidOCR returns 4 corner points as [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    in clockwise order starting from top-left, so top edge = points 0 and 1."""
    return (bbox[0][1] + bbox[1][1]) / 2.0


def _bbox_center_y(bbox: list) -> float:
    """Vertical center of a bounding box — used for line-grouping."""
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2.0


def _bbox_center_x(bbox: list) -> float:
    """Horizontal center of a bounding box."""
    xs = [pt[0] for pt in bbox]
    return (min(xs) + max(xs)) / 2.0


def _boxes_overlap_vertically(bbox_a: list, bbox_b: list, tolerance: float = 0.5) -> bool:
    """True when two boxes sit on roughly the same text line.
    tolerance=0.5 means their vertical centres must be within 50% of the
    smaller box's height of each other — tuned for typical document fonts."""
    ay_min = min(pt[1] for pt in bbox_a)
    ay_max = max(pt[1] for pt in bbox_a)
    by_min = min(pt[1] for pt in bbox_b)
    by_max = max(pt[1] for pt in bbox_b)
    height_a = ay_max - ay_min
    height_b = by_max - by_min
    centre_a = (ay_min + ay_max) / 2.0
    centre_b = (by_min + by_max) / 2.0
    return abs(centre_a - centre_b) <= tolerance * min(height_a, height_b)


def _tesseract_region(image_path: str, bbox: list, lang: str = "ara+eng", padding: int = 4) -> str | None:
    """Crop *bbox* out of the source image and re-recognize it with Tesseract's
    combined Arabic+English model in a single pass.

    WHY THIS EXISTS: RapidOCR's "ar" and "en" recognizers are each trained on
    one script's character set, so a line that genuinely mixes Arabic and
    English within the same physical run of text (no visual gap the detector
    can split on) can't be read correctly by either one alone — you get a
    real reading of half the line and garbage for the other half, whichever
    engine "wins" the confidence comparison. Tesseract decodes both scripts
    from one crop when given lang="ara+eng" (order matters — "ara+eng" reads
    correctly; "eng+ara" has been reported to emit garbage — so always pass
    Arabic first), so it's used as a targeted re-read only for the regions
    where the Arabic and English detectors landed on (almost) the same box —
    a strong signal that the box itself is mixed-script, not two adjacent
    single-script boxes RapidOCR just happened to detect with high overlap.

    Returns None (never raises) if pytesseract or the tesseract binary isn't
    installed, or the crop yields nothing — callers fall back to the
    higher-confidence single-script reading, i.e. the pre-existing behaviour.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None

    try:
        img = Image.open(image_path)
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        left = max(0, int(min(xs)) - padding)
        top = max(0, int(min(ys)) - padding)
        right = min(img.width, int(max(xs)) + padding)
        bottom = min(img.height, int(max(ys)) + padding)
        if right <= left or bottom <= top:
            return None
        crop = img.crop((left, top, right, bottom))
        # PSM 7 = "treat the image as a single text line", which this crop is.
        text = pytesseract.image_to_string(crop, lang=lang, config="--psm 7").strip()
        return text or None
    except Exception:
        # Covers pytesseract.TesseractNotFoundError (the `tesseract` binary
        # missing from PATH even though the python package is installed) and
        # any decode/crop error — treat as "not available for this region"
        # rather than failing the whole OCR pipeline over one line.
        return None


def _dedup_by_position(lines_ar: list[OCRLine], lines_en: list[OCRLine],
                       image_path: str | None = None,
                       iou_threshold: float = 0.4,
                       full_overlap_threshold: float = 0.75) -> list[OCRLine]:
    """When the same region is detected by both the Arabic and English engines,
    decide what to keep for that region.

    Two distinct situations get merged into "overlap" by IoU alone, so they're
    now handled differently:

    - Partial/moderate overlap (iou_threshold <= iou < full_overlap_threshold):
      two adjacent-but-distinct boxes RapidOCR happened to detect with some
      overlap. Keep whichever single-script reading has higher confidence —
      the region most likely genuinely belongs to one script.
    - Near-total overlap (iou >= full_overlap_threshold): both engines are
      looking at essentially the same box, which is a strong signal the line
      itself mixes Arabic and English — the kind of line a single-script
      recognizer can't read correctly no matter which one "wins". When
      image_path is given, re-read that crop with Tesseract's combined
      ar+eng model (see _tesseract_region) instead of picking a partially-
      wrong single-script reading.

    Overlap is measured as the Intersection-over-Union (IoU) of the axis-aligned
    bounding rectangles — fast and good enough for near-duplicate boxes."""

    def to_rect(bbox):
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = to_rect(a)
        bx1, by1, bx2, by2 = to_rect(b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    kept_en: list[OCRLine] = []
    kept_ar_winners: list[OCRLine] = []
    kept_mixed: list[OCRLine] = []
    used_ar = set()

    for en_line in lines_en:
        best_iou, best_idx = 0.0, -1
        for i, ar_line in enumerate(lines_ar):
            v = iou(en_line.bbox, ar_line.bbox)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= iou_threshold:
            ar_line = lines_ar[best_idx]
            used_ar.add(best_idx)  # this region is claimed by one side or the other either way

            if best_iou >= full_overlap_threshold and image_path:
                # Both engines detected essentially the same box — likely a
                # genuinely mixed-script line. Re-read the crop with Tesseract
                # instead of trusting either single-script engine.
                tess_text = _tesseract_region(image_path, en_line.bbox)
                if tess_text:
                    kept_mixed.append(OCRLine(
                        text=tess_text,
                        confidence=max(en_line.confidence, ar_line.confidence),
                        bbox=en_line.bbox,
                    ))
                    continue
                # Tesseract unavailable or returned nothing — fall through to
                # the ordinary higher-confidence-wins comparison below.

            if en_line.confidence >= ar_line.confidence:
                kept_en.append(en_line)
            else:
                # Arabic version wins — keep it explicitly. (Previously this branch
                # relied on `kept_ar` picking it up below, but marking the index
                # "used" excluded it from that list too, so the winning Arabic
                # line was silently dropped instead of kept.)
                kept_ar_winners.append(ar_line)
        else:
            kept_en.append(en_line)

    # Arabic lines that had no English counterpart at all (genuine Arabic-only text)
    kept_ar_unmatched = [l for i, l in enumerate(lines_ar) if i not in used_ar]
    return kept_ar_unmatched + kept_ar_winners + kept_mixed + kept_en


def _merge_bilingual_lines(all_lines: list[OCRLine]) -> list[OCRLine]:
    """Sort mixed-script boxes into natural reading order.

    Strategy:
    1. Group boxes that share the same visual text line (overlap vertically).
    2. Within each line group, sort left-to-right (RTL Arabic text is stored
       LTR by RapidOCR's detection, so a simple x-sort works across scripts).
    3. Sort the line groups top-to-bottom.

    The result is a flat list in the order a human would read the page,
    regardless of whether individual words are Arabic or English.
    """
    if not all_lines:
        return []

    # Sort initially by vertical centre so grouping is stable
    all_lines = sorted(all_lines, key=lambda l: _bbox_center_y(l.bbox))

    groups: list[list[OCRLine]] = []
    for line in all_lines:
        placed = False
        for group in groups:
            # Use the first element of the group as the reference line
            if _boxes_overlap_vertically(group[0].bbox, line.bbox):
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])

    # Sort each group left-to-right, then sort groups top-to-bottom
    ordered: list[OCRLine] = []
    groups.sort(key=lambda g: _bbox_center_y(g[0].bbox))
    for group in groups:
        group.sort(key=lambda l: _bbox_center_x(l.bbox))
        ordered.extend(group)

    return ordered


def ocr_image(path: str, lang: str = "ar") -> list[OCRLine]:
    """Run OCR on *path* and return lines in reading order.

    lang values
    -----------
    "ar"      Arabic-only document — uses the Arabic recognition model.
    "en"      English/Latin-only document — uses the bundled Latin model.
    "mixed"   Bilingual Arabic + English document (most common for legal
              contracts in the MENA region).  Both engines run on the same
              image; results are de-duplicated by bounding-box overlap and
              re-sorted into natural reading order so that a line like
              ``مادة 3 — Force Majeure`` comes out as a single coherent unit
              rather than two unrelated fragments.

    For truly mixed documents, prefer ``lang="mixed"`` over calling this
    function twice manually — the merge logic here handles overlapping
    detections and reading-order reconstruction correctly.
    """
    if lang == "mixed":
        # Run both engines; Arabic model may be unavailable — handle gracefully
        try:
            ar_engine = _get_engine("ar")
            ar_result, _ = ar_engine(path)
            ar_lines = [
                OCRLine(text=text, confidence=float(score), bbox=box)
                for box, text, score in (ar_result or [])
                if text
            ]
        except (ImportError, FileNotFoundError):
            ar_lines = []

        en_engine = _get_engine("en")
        en_result, _ = en_engine(path)
        en_lines = [
            OCRLine(text=text, confidence=float(score), bbox=box)
            for box, text, score in (en_result or [])
            if text
        ]

        merged = _dedup_by_position(ar_lines, en_lines, image_path=path)
        return _merge_bilingual_lines(merged)

    # Single-language path (unchanged behaviour for "ar" and "en")
    engine = _get_engine(lang)
    result, _elapse = engine(path)  # -> [[box, text, score], ...] or None if nothing detected

    lines: list[OCRLine] = []
    for box, text, score in (result or []):
        if not text:
            continue
        lines.append(OCRLine(text=text, confidence=float(score), bbox=box))
    return lines


def _image_data_uri(path: str) -> str:
    mime = _IMAGE_MIME.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def ocr_image_hosted(path: str, lang: str = "ar") -> str:
    """Hosted fallback when local OCR isn't available: asks Groq's vision
    model (GROQ_VISION_MODEL, default qwen/qwen3.6-27b) to read all text
    out of the image. Needs GROQ_API_KEY + internet. Raises ImportError
    (not an API error) when there's no key — callers treat that the same
    as a missing local model."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ImportError(
            "No local OCR model available and no GROQ_API_KEY — install rapidocr-onnxruntime "
            "(and the Arabic model if needed) or set GROQ_API_KEY for the hosted vision fallback"
        )
    import httpx

    prompt = (
        "This is a legal document image that may contain Arabic text, English text, "
        "or both scripts mixed within the same line (e.g. an Arabic clause heading "
        "followed by an English defined term, or a line number in English next to "
        "Arabic body text). Extract ALL text exactly as written, preserving the "
        "original reading order line by line. For mixed lines, output the full line "
        "content in the order it appears visually (left to right across the page). "
        "Keep every word in its original script — do NOT transliterate. "
        "Reply with ONLY the extracted text, no commentary."
    )
    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_uri(path)}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }
    with httpx.Client(timeout=120) as client:
        for attempt in range(3):
            resp = client.post(
                GROQ_VISION_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            if resp.status_code != 429 or attempt == 2:
                break
            time.sleep(2 * (attempt + 1))  # 429 = rate limit — back off and retry
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        # qwen reasoning models wrap their answer in <think>…</think> — drop it
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return text


def ocr_image_full_text(path: str, lang: str = "mixed") -> str:
    """Return all text from *path* as a single newline-joined string.

    Default lang changed to ``"mixed"`` so that bilingual Arabic/English
    documents (the dominant format for MENA legal contracts) are handled
    correctly out of the box.  Pass ``lang="ar"`` or ``lang="en"`` only
    when you are certain the document is monolingual.

    Fallback chain
    --------------
    1. rapidocr-onnxruntime local inference (fast, offline).
       - ``lang="mixed"``: Arabic engine + English engine → merged.
       - ``lang="ar"``: Arabic engine only (requires arabic_rec.onnx).
       - ``lang="en"``: English engine only (bundled, always available).
    2. If the required local model is missing or the package isn't installed,
       fall back to ``ocr_image_hosted`` (Groq vision model) — which always
       handles mixed scripts correctly because it's a large multimodal LLM.
    """
    try:
        lines = ocr_image(path, lang=lang)
        if lines:
            return "\n".join(line.text for line in lines)
        # Empty result from local OCR on a mixed doc (e.g. Arabic model missing
        # but English-only pass returned nothing useful) — try hosted fallback.
        raise FileNotFoundError("Local OCR returned no text — trying hosted fallback")
    except (ImportError, FileNotFoundError):
        # rapidocr-onnxruntime isn't installed, or (for lang="ar") the
        # Arabic recognition model hasn't been downloaded yet — degrade to
        # the hosted vision model instead of failing the whole upload.
        # The hosted model handles Arabic, English, and mixed scripts natively.
        return ocr_image_hosted(path, lang=lang)


def ocr_pdf_page(pixmap_path: str, lang: str = "mixed") -> str:
    """For scanned PDF pages: render the page to an image first (e.g. with
    PyMuPDF's page.get_pixmap().save(path)), then pass that image path here.

    Default lang is ``"mixed"`` — correct for most MENA legal documents that
    blend Arabic article headings with English defined terms or clause numbers.
    """
    return ocr_image_full_text(pixmap_path, lang=lang)
