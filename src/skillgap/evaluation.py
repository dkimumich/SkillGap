"""Measure how well SkillGap's scores agree with human fit labels.

Each example is a (score, label) pair, where label is "poor", "okay" or "good"
as judged by a person *before* seeing SkillGap's score.

Metrics:
  * pairwise agreement: of all pairs with different labels, the share where
    the better-labelled posting got the higher score (ties count half).
    0.5 = no better than a coin flip, 1.0 = perfect ordering.
  * Spearman correlation between score and label (-1 to 1).
  * label accuracy: scores turned into labels with fixed thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from scipy.stats import spearmanr

LABELS = {"poor": 0, "okay": 1, "good": 2}
OKAY_AT = 40   # scores >= this are predicted "okay"
GOOD_AT = 60   # scores >= this are predicted "good"


def predict_label(score: float, okay_at: float = OKAY_AT, good_at: float = GOOD_AT) -> str:
    if score >= good_at:
        return "good"
    if score >= okay_at:
        return "okay"
    return "poor"


def _check(labels: list[str]) -> list[int]:
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown labels {unknown}; use poor, okay or good")
    return [LABELS[label] for label in labels]


def pairwise_agreement(scores: list[float], labels: list[str]) -> float | None:
    """Share of differently-labelled pairs that the scores put in the right order."""
    ranks = _check(labels)
    correct = 0.0
    total = 0
    for i, j in combinations(range(len(scores)), 2):
        if ranks[i] == ranks[j]:
            continue
        total += 1
        better, worse = (i, j) if ranks[i] > ranks[j] else (j, i)
        if scores[better] > scores[worse]:
            correct += 1
        elif scores[better] == scores[worse]:
            correct += 0.5
    return correct / total if total else None


def label_accuracy(scores: list[float], labels: list[str]) -> float | None:
    _check(labels)
    if not scores:
        return None
    hits = sum(predict_label(s) == label for s, label in zip(scores, labels, strict=True))
    return hits / len(scores)


def confusion(scores: list[float], labels: list[str]) -> dict[str, dict[str, int]]:
    """confusion[actual][predicted] = count."""
    _check(labels)
    table = {actual: dict.fromkeys(LABELS, 0) for actual in LABELS}
    for s, label in zip(scores, labels, strict=True):
        table[label][predict_label(s)] += 1
    return table


def spearman(scores: list[float], labels: list[str]) -> float | None:
    ranks = _check(labels)
    if len(set(ranks)) < 2 or len(set(scores)) < 2:
        return None
    rho = spearmanr(scores, ranks).statistic
    return None if math.isnan(rho) else float(rho)


@dataclass(frozen=True)
class Report:
    n: int
    pairwise: float | None
    spearman: float | None
    accuracy: float | None


def evaluate(scores: list[float], labels: list[str]) -> Report:
    return Report(
        n=len(scores),
        pairwise=pairwise_agreement(scores, labels),
        spearman=spearman(scores, labels),
        accuracy=label_accuracy(scores, labels),
    )
