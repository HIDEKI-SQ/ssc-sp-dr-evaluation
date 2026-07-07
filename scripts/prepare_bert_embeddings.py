#!/usr/bin/env python3
"""Regenerate the BERT embeddings used by the value-coupling sweep.

Optional: base reproduction ships the precomputed embeddings at
data/intermediate/bert_embeddings.npz. This script reproduces them from the 50
concept labels using bert-base-uncased [CLS]-token embeddings (seed 601),
matching the recorded run. Requires requirements-bert.txt (transformers, torch).

It is NOT Sentence-BERT: single-word [CLS] pooling from bert-base-uncased.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 601
MODEL_NAME = "bert-base-uncased"

WORDS = [
    "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger",
    "car", "bus", "train", "plane", "boat", "bicycle", "truck", "motorcycle",
    "tree", "flower", "grass", "mountain", "river", "ocean", "sun", "moon",
    "house", "building", "castle", "bridge", "tower", "church", "school", "hospital",
    "book", "chair", "table", "lamp", "clock", "phone", "computer", "camera",
    "apple", "bread", "rice", "meat", "milk", "coffee", "tea", "water",
    "love", "time", "life", "death",
][:50]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "intermediate"
OUT.mkdir(parents=True, exist_ok=True)


def set_torch_seed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def main() -> int:
    try:
        import torch
        from transformers import BertTokenizer, BertModel
    except ImportError:
        print("transformers/torch not installed. Install requirements-bert.txt to")
        print("regenerate, or use the shipped data/intermediate/bert_embeddings.npz.")
        return 1

    set_torch_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(device).eval()

    embs = []
    with torch.no_grad():
        for w in WORDS:
            inputs = tokenizer(w, return_tensors="pt").to(device)
            out = model(**inputs)
            embs.append(out.last_hidden_state[0, 0, :].cpu().numpy())
    embeddings = np.asarray(embs, dtype=np.float64)

    meta = {
        "model_name": MODEL_NAME, "pooling": "cls_token", "seed": SEED,
        "n_items": len(WORDS), "dim": int(embeddings.shape[1]),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "sha256_embeddings": hashlib.sha256(embeddings.tobytes()).hexdigest(),
    }
    np.savez(OUT / "bert_embeddings.npz", embeddings=embeddings,
             words=np.array(WORDS), meta=json.dumps(meta))
    (OUT / "bert_embeddings_provenance.json").write_text(json.dumps(meta, indent=2))
    print("wrote", OUT / "bert_embeddings.npz")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
