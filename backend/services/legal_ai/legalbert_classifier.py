"""Loader for the fine-tuned Legal-BERT+DeBERTa ONNX contract *risk*
classifier. Produced by notebooks/legalbert_ensemble_export.ipynb and dropped
into:

    backend/models/legalbert/clause/
        label_map.json          {"id2label": {"0": "...", ...}}  (shared)
        bert/    model.onnx + tokenizer files (WordPiece)
        deberta/ model.onnx + tokenizer files (SentencePiece)

This is a 2-model ensemble (Legal-BERT-base + DeBERTa-v3-base, simple
probability averaging) — see the model card for
Nikhil-AI-Labs/legal-contract-classifier-best. Both sub-models must be
present: loading only one silently downgrades you from the card's advertised
~97.7% ensemble accuracy to that one model's solo ~92% F1.

IMPORTANT — this classifies contract *risk categories* (Safe/Standard,
Unilateral Termination, Unlimited Liability, Non-Compete: 4 mutually
exclusive classes), NOT CUAD-style clause *types* (Governing Law,
Termination, IP Assignment, ...). If you need clause-type tagging, this is
the wrong model regardless of how correctly it's loaded here.

Because the 4 classes are mutually exclusive (see the model card's
confusion matrix — one predicted class per input), inference below is
softmax-per-model -> average the two probability vectors -> argmax, i.e.
single-label, matching how the ensemble was actually trained and how the
card's own SimpleLegalEnsemble.predict() behaves. This intentionally
replaced an earlier sigmoid/multi-label-threshold implementation, which was
built for a different (CUAD, multi-label) task and produced miscalibrated
confidence scores when pointed at this single-label risk model.

Until you've run the notebook and copied its output in, calls here raise a
clear FileNotFoundError rather than silently returning nonsense.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(__file__).parent.parent.parent / "models" / "legalbert"
MAX_LENGTH = 256
SUB_MODELS = ("bert", "deberta")  # must match the export notebook's SUB_MODELS keys


def _session_options():
    """ORT_ENABLE_ALL turns on the constant-folding/op-fusion graph
    optimizations onnxruntime otherwise leaves off by default for
    InferenceSession; intra_op_num_threads=0 means "let onnxruntime pick
    based on available cores" instead of its conservative single-thread
    default. Cheap to set, meaningful latency win on CPU, zero accuracy
    impact — this only changes how the same math gets executed."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0
    return opts


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    all_scores: dict[str, float]  # every class's averaged probability, not just the top one


def _softmax(logits: np.ndarray) -> np.ndarray:
    # Subtract max for numerical stability before exponentiating.
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


@lru_cache(maxsize=1)
def _load(subdir: str):
    """Loads both ensemble members (bert + deberta) plus the shared label map.
    Cached as a single unit since they're always used together — there's no
    valid partial-ensemble state."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    clause_dir = MODELS_ROOT / subdir
    label_map_path = clause_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"No label_map.json at {label_map_path}. Run "
            "notebooks/legalbert_ensemble_export.ipynb and copy its output here first."
        )
    id2label = json.loads(label_map_path.read_text())["id2label"]

    sessions = {}
    tokenizers = {}
    for key in SUB_MODELS:
        sub_dir = clause_dir / key
        onnx_path = sub_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No fine-tuned model at {onnx_path}. The ensemble needs BOTH "
                f"'bert/' and 'deberta/' sub-models present — loading only one "
                f"would silently downgrade accuracy below the card's advertised "
                f"figure. Run notebooks/legalbert_ensemble_export.ipynb and copy "
                f"its full output here."
            )
        sessions[key] = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"], sess_options=_session_options()
        )
        tokenizers[key] = AutoTokenizer.from_pretrained(str(sub_dir))

    return sessions, tokenizers, id2label


def classify_clause_types(text: str, threshold: float = 0.5) -> list[ClassificationResult]:
    """Single-label contract-risk classification (Safe/Standard, Unilateral
    Termination, Unlimited Liability, Non-Compete — mutually exclusive).

    Returns a list for interface compatibility with callers, but it contains
    at most one entry: the ensemble's top prediction, only if its averaged
    confidence clears `threshold` — never multiple simultaneous labels, since
    this task isn't multi-label (see module docstring)."""
    sessions, tokenizers, id2label = _load("clause")

    per_model_probs = []
    for key in SUB_MODELS:
        inputs = tokenizers[key](
            text, return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LENGTH
        )
        logits = sessions[key].run(
            ["logits"], {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        )[0][0]
        per_model_probs.append(_softmax(logits))

    # Simple probability averaging across the ensemble — same method as the
    # model card's SimpleLegalEnsemble.predict().
    avg_probs = np.mean(per_model_probs, axis=0)

    top_id = int(np.argmax(avg_probs))
    top_confidence = float(avg_probs[top_id])
    all_scores = {id2label[str(i)]: float(p) for i, p in enumerate(avg_probs)}

    if top_confidence < threshold:
        return []
    return [ClassificationResult(label=id2label[str(top_id)], confidence=top_confidence, all_scores=all_scores)]
