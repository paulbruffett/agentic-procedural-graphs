"""SQuAD-style answer normalization, exact match and token F1 (as in the official HotpotQA evaluator)."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    # yes/no answers must match exactly
    if (pred in ("yes", "no", "noanswer") or ref in ("yes", "no", "noanswer")) and pred != ref:
        return 0.0
    pred_toks, ref_toks = pred.split(), ref.split()
    common = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred_toks), common / len(ref_toks)
    return 2 * precision * recall / (precision + recall)
