import pytest

from skillgap.evaluation import (
    confusion,
    evaluate,
    label_accuracy,
    pairwise_agreement,
    predict_label,
    spearman,
)


@pytest.mark.parametrize("score, label", [(0, "poor"), (39, "poor"), (40, "okay"), (59, "okay"),
                                          (60, "good"), (100, "good")])
def test_predict_label_thresholds(score, label):
    assert predict_label(score) == label


def test_perfect_ordering():
    assert pairwise_agreement([90, 50, 10], ["good", "okay", "poor"]) == 1.0


def test_reversed_ordering():
    assert pairwise_agreement([10, 50, 90], ["good", "okay", "poor"]) == 0.0


def test_ties_count_half():
    assert pairwise_agreement([50, 50], ["good", "poor"]) == 0.5


def test_same_labels_give_no_pairs():
    assert pairwise_agreement([80, 20], ["good", "good"]) is None


def test_label_accuracy_and_confusion():
    scores, labels = [75, 45, 70, 10], ["good", "okay", "poor", "poor"]
    assert label_accuracy(scores, labels) == 0.75
    table = confusion(scores, labels)
    assert table["poor"] == {"poor": 1, "okay": 0, "good": 1}
    assert table["good"]["good"] == 1


def test_spearman():
    assert spearman([90, 50, 10], ["good", "okay", "poor"]) == pytest.approx(1.0)
    assert spearman([80], ["good"]) is None


def test_evaluate_report():
    report = evaluate([90, 10], ["good", "poor"])
    assert report.n == 2
    assert report.pairwise == 1.0
    assert report.accuracy == 1.0


def test_empty_input():
    report = evaluate([], [])
    assert report.n == 0 and report.pairwise is None and report.accuracy is None


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="unknown labels"):
        pairwise_agreement([50], ["great"])
