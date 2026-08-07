"""Downloads the Arabic OCR recognition model (ONNX) that
services/parsing/ocr.py needs for lang="ar" and that isn't bundled inside
the rapidocr-onnxruntime pip package (only the Chinese/English recognizer
ships in the wheel — every other language's recognizer is a separate
download, same as upstream PaddleOCR/RapidOCR work).

WHY THIS IS A SEPARATE SCRIPT
------------------------------
This has to run somewhere with normal internet access reaching
modelscope.cn (RapidAI's model host — the same place `rapidocr-onnxruntime`
itself downloads extra-language models from when you pass lang="ar" to it
directly). If that host is unreachable in your network, see the FALLBACK
section at the bottom of this docstring.

USAGE
-----
    cd backend
    python scripts/download_arabic_ocr_model.py

Downloads ~4-8MB to backend/models/ocr/arabic_rec.onnx, verifies it against
the published SHA256, and you're done — services/parsing/ocr.py picks it up
automatically the next time OCR runs with lang="ar". No server restart
strictly required, but the model is cached in-process after first use
(see ocr.py's @lru_cache), so restart the backend if it's already running.

The Arabic character dictionary (arabic_dict.txt) is NOT downloaded by this
script — it's a small (~160-line, plain-text) file already checked into
this repo at backend/models/ocr/arabic_dict.txt, sourced from PaddleOCR's
own repo (ppocr/utils/dict/arabic_dict.txt).

WHERE THIS FILE COMES FROM
----------------------------
RapidOCR (https://github.com/RapidAI/RapidOCR) converts PaddleOCR's
officially trained models to ONNX and republishes them; this script points
at their PP-OCRv4 Arabic mobile recognizer, which is what RapidOCR itself
downloads on demand when you ask it for lang="ar" without a custom
rec_model_path. URL and checksum taken from RapidOCR's own
python/rapidocr/default_models.yaml (v3.9.2).

FALLBACK — if modelscope.cn is unreachable from your machine
---------------------------------------------------------------
Convert PaddleOCR's original model yourself instead:

    pip install paddlepaddle paddleocr paddle2onnx
    python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ar')"  # triggers the download
    # Find the downloaded inference model under ~/.paddleocr/ or ~/.paddleocr/whl/rec/ar/ —
    # the exact subfolder name changes between PaddleOCR versions, so `find ~/.paddleocr -iname "*.pdmodel"` to locate it.
    paddle2onnx \\
        --model_dir <that folder> \\
        --model_filename inference.pdmodel \\
        --params_filename inference.pdiparams \\
        --save_file backend/models/ocr/arabic_rec.onnx \\
        --opset_version 14
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
    "onnx/PP-OCRv4/rec/arabic_PP-OCRv4_rec_mobile.onnx"
)
EXPECTED_SHA256 = "4a9011bef71687bb84288dc86ad2471bd5d37b717ddf672dd156f9e7a5601bac"

OUT_DIR = Path(__file__).parent.parent / "models" / "ocr"
OUT_PATH = OUT_DIR / "arabic_rec.onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    def _report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100 // total_size
        print(f"\r  {done // 1024:,} KB / {total_size // 1024:,} KB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()


def main() -> int:
    dict_path = OUT_DIR / "arabic_dict.txt"
    if not dict_path.exists():
        print(
            f"[warning] {dict_path} not found — it should already be checked into the repo. "
            "OCR for Arabic won't work without it even after this script finishes."
        )

    if OUT_PATH.exists() and _sha256(OUT_PATH) == EXPECTED_SHA256:
        print(f"[skip] {OUT_PATH} already present and checksum-verified.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = OUT_PATH.with_suffix(".onnx.part")
    print(f"[download] {MODEL_URL}\n       -> {OUT_PATH}")
    try:
        _download(MODEL_URL, tmp_path)
    except Exception as e:  # noqa: BLE001 — this is a user-facing CLI script, not a library call
        print(f"[error] download failed: {e}")
        print("See the FALLBACK section in this script's module docstring for an alternate path.")
        return 1

    actual = _sha256(tmp_path)
    if actual != EXPECTED_SHA256:
        tmp_path.unlink(missing_ok=True)
        print(f"[error] checksum mismatch — expected {EXPECTED_SHA256}, got {actual}. Not saving; try again.")
        return 1

    tmp_path.rename(OUT_PATH)
    print(f"[done] wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes, checksum verified).")
    print("Arabic OCR is ready — services/parsing/ocr.py will pick this up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
