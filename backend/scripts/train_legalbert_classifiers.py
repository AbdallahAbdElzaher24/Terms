"""Fine-tunes and exports the clause-type classifier that
services/legal_ai/legalbert_classifier.py loads at inference time.

(This script also has a `--task risk` mode that used to feed
services/legal_ai/risk_classifier.py — that module was removed since
nothing in the pipeline wired it up yet; the risk-training path below is
kept only as a reference if that feature comes back.)

THE GAP THIS FILLS
-------------------
services/legal_ai/legalbert_classifier.py raises a clear FileNotFoundError
pointing at "notebooks/legalbert_cuad_finetune.ipynb" — but that notebook
doesn't exist anywhere in the repo. There is currently no way to produce
backend/models/legalbert/clause/model.onnx, which means classify_clause_types
can never actually run; every upload and every chat turn falls back to the
"model not set up yet" warning path. This script is that missing piece, as
a plain Python script (easier to run in CI / a Makefile target than a
notebook, and this file IS the whole record of how the shipped model was
produced instead of that living only in someone's
notebook cell-execution order).

USAGE
-----
Multi-label (clause type — a clause can match more than one CUAD category):
    python scripts/train_legalbert_classifiers.py \
        --task clause --data data/clause_type.jsonl --multi-label \
        --out ../models/legalbert/clause

Risk level (uses DeBERTa-v3 instead of Legal-BERT — pass --base-model):
    python scripts/train_legalbert_classifiers.py \
        --task risk --data data/risk_level.jsonl \
        --base-model microsoft/deberta-v3-base --out ../models/deberta_risk

No labeled data yet? Run with --demo to fine-tune on a small synthetic
dataset built into this script — this won't produce a model you should
ship, but it proves the whole pipeline (train -> ONNX export -> label_map ->
loadable by legalbert_classifier.py) end-to-end before you invest in real
data collection:
    python scripts/train_legalbert_classifiers.py --task clause --demo --out /tmp/clause_test

DATA FORMAT
-----------
JSONL, one example per line.
  Single-label:  {"text": "...", "label": "NDA"}
  Multi-label:   {"text": "...", "labels": ["Auto-Renewal", "Governing Law"]}

Getting real data: the CUAD dataset (https://www.atticusprojectai.org/cuad,
also on the Hugging Face Hub) is the standard source for contract/clause
labels — it ships as extractive QA (context + per-category answer spans)
rather than this JSONL format, so convert it once with something like:

    from datasets import load_dataset
    ds = load_dataset("cuad")
    # group by contract, take each category whose question has a non-empty
    # answer span as a positive label for that contract -> write JSONL here.

This is left as a conversion step outside this script (rather than a hidden
assumption baked into it) because CUAD's exact HF schema does change between
dataset revisions, and getting the context/question/category mapping wrong
silently would be worse than making it explicit.

WHAT THIS PRODUCES
-------------------
<out>/model.onnx        — matches the input names ("input_ids",
                           "attention_mask") and output name ("logits")
                           that legalbert_classifier.py / risk_classifier.py
                           call with session.run(...).
<out>/label_map.json     — {"id2label": {"0": "...", ...}}
<out>/tokenizer files    — AutoTokenizer.save_pretrained() output, so
                           AutoTokenizer.from_pretrained(<out>) works.

Also writes <out>/run_info.json (config + eval metrics) next to the
exported model, so a training run stays comparable over time instead of
"check the terminal scrollback."
"""
import argparse
import json
import sys
from pathlib import Path

MAX_LENGTH = 256

DEMO_CLAUSE_EXAMPLES = [
    ("This Agreement shall automatically renew for successive one-year terms unless either party provides written "
     "notice of non-renewal at least sixty (60) days prior to the end of the then-current term.",
     ["Auto-Renewal"]),
    ("The term of this Agreement shall automatically extend for additional one (1) year periods unless terminated "
     "in accordance with Section 8.", ["Auto-Renewal"]),
    ("This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, "
     "without regard to its conflict of laws principles.", ["Governing Law"]),
    ("Any dispute arising under this Agreement shall be governed by the laws of England and Wales.",
     ["Governing Law"]),
    ("Any dispute, controversy, or claim arising out of or relating to this Agreement shall be resolved by binding "
     "arbitration administered by the American Arbitration Association.", ["Arbitration"]),
    ("The parties agree to submit any dispute to binding arbitration in lieu of litigation in court.",
     ["Arbitration"]),
    ("This Agreement shall be governed by California law, and any dispute shall be resolved through binding "
     "arbitration in San Francisco, California.", ["Governing Law", "Arbitration"]),
    ("Neither party may assign this Agreement, or any of its rights or obligations hereunder, without the prior "
     "written consent of the other party.", ["Anti-Assignment"]),
]

DEMO_RISK_EXAMPLES = [
    ("This Agreement shall automatically renew for successive one-year terms unless notice of non-renewal is "
     "given at least sixty (60) days in advance, and the Client shall have no right to terminate for convenience "
     "during any renewal term.", "high"),
    ("Client's sole and exclusive remedy for any breach shall be limited to the fees paid in the preceding thirty "
     "(30) days, and in no event shall Vendor be liable for any indirect, consequential, or punitive damages.",
     "high"),
    ("Either party may terminate this Agreement for convenience upon thirty (30) days' written notice to the "
     "other party.", "medium"),
    ("Payment shall be due within thirty (30) days of invoice; late payments accrue interest at 1% per month.",
     "medium"),
    ("This Agreement may be executed in counterparts, each of which shall be deemed an original.", "low"),
    ("Notices under this Agreement shall be sent by email to the addresses set forth on the signature page.",
     "low"),
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dataset(args) -> tuple[list[str], list, list[str]]:
    """Returns (texts, raw_labels, sorted_unique_label_names)."""
    if args.demo:
        demo = {"clause": DEMO_CLAUSE_EXAMPLES, "risk": DEMO_RISK_EXAMPLES}[args.task]
        texts = [t for t, _ in demo]
        raw_labels = [lab for _, lab in demo]
    else:
        if not args.data:
            sys.exit("--data <path.jsonl> is required unless --demo is set")
        rows = _read_jsonl(Path(args.data))
        if not rows:
            sys.exit(f"No examples found in {args.data}")
        texts = [r["text"] for r in rows]
        key = "labels" if args.multi_label else "label"
        if key not in rows[0]:
            sys.exit(f"--multi-label={args.multi_label} but examples use '{'label' if args.multi_label else 'labels'}' — "
                      f"expected the '{key}' field. See the DATA FORMAT section in this file's docstring.")
        raw_labels = [r[key] for r in rows]

    if args.multi_label:
        all_labels = sorted({lab for labs in raw_labels for lab in labs})
    else:
        all_labels = sorted(set(raw_labels))

    if len(all_labels) < 2:
        sys.exit(f"Only found {len(all_labels)} distinct label(s) ({all_labels}) — need at least 2 to train a classifier.")

    return texts, raw_labels, all_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["clause", "risk"])
    ap.add_argument("--data", help="Path to a JSONL file (see DATA FORMAT in the module docstring)")
    ap.add_argument("--demo", action="store_true", help="Use the small built-in synthetic dataset instead of --data")
    ap.add_argument("--multi-label", action="store_true", help="Set for clause classification (a text can have several labels)")
    ap.add_argument("--out", required=True, help="Output directory for model.onnx / label_map.json / tokenizer files")
    ap.add_argument("--base-model", default="nlpaueb/legal-bert-base-uncased",
                     help="HF hub id to fine-tune from. Use microsoft/deberta-v3-base for --task risk.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task == "clause":
        args.multi_label = True

    try:
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        sys.exit(
            f"Missing training dependency: {e}. Uncomment the 'Legal AI' fine-tuning block in "
            "requirements.txt (transformers, torch, scikit-learn, onnx, onnxruntime) and pip install."
        )

    texts, raw_labels, label_names = _load_dataset(args)
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}
    num_labels = len(label_names)
    print(f"[data] {len(texts)} examples, {num_labels} labels: {label_names}")

    if args.multi_label:
        y = np.zeros((len(texts), num_labels), dtype=np.float32)
        for i, labs in enumerate(raw_labels):
            for lab in labs:
                y[i, label2id[lab]] = 1.0
    else:
        y = np.array([label2id[lab] for lab in raw_labels])

    n_val = max(1, int(len(texts) * args.val_fraction)) if len(texts) >= 8 else 0
    if n_val:
        train_idx, val_idx = train_test_split(
            range(len(texts)), test_size=n_val, random_state=args.seed,
            stratify=(y if not args.multi_label else None),
        )
    else:
        train_idx, val_idx = list(range(len(texts))), []
        print("[data] too few examples for a held-out val split — training on everything (--demo only; do not ship this)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    class ClauseDataset(torch.utils.data.Dataset):
        def __init__(self, idxs):
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            idx = self.idxs[i]
            enc = tokenizer(texts[idx], truncation=True, padding="max_length", max_length=MAX_LENGTH)
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"] = torch.tensor(y[idx], dtype=torch.float32 if args.multi_label else torch.long)
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        problem_type="multi_label_classification" if args.multi_label else "single_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if args.multi_label:
            preds = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
            return {
                "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            }
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch" if val_idx else "no",
        save_strategy="no",
        logging_steps=10,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ClauseDataset(train_idx),
        eval_dataset=ClauseDataset(val_idx) if val_idx else None,
        compute_metrics=compute_metrics if val_idx else None,
    )

    trainer.train()
    metrics = trainer.evaluate() if val_idx else {}
    if metrics:
        print(f"[eval] {metrics}")

    # ---- Export: ONNX model + label map + tokenizer, exactly what
    # legalbert_classifier.py / risk_classifier.py expect to load. ----
    model.eval().to("cpu")
    dummy = tokenizer("dummy input for tracing", return_tensors="pt", padding="max_length", max_length=MAX_LENGTH)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(out_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "label_map.json").write_text(json.dumps({"id2label": {str(k): v for k, v in id2label.items()}}, indent=2))
    print(f"[export] wrote model.onnx, tokenizer files, label_map.json -> {out_dir}")

    # Also drop the run config + eval metrics next to the exported model as
    # plain JSON — enough to know what produced this model.onnx and how it
    # scored, without needing a tracking server or extra dependency.
    run_record = {
        "task": args.task,
        "base_model": args.base_model,
        "multi_label": args.multi_label,
        "num_labels": num_labels,
        "num_examples": len(texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_record, indent=2))
    print(f"[export] wrote run_info.json (config + eval metrics) -> {out_dir}")


if __name__ == "__main__":
    main()
