
import pytest

from ml.promotion_policy import evaluate_promotion


def test_promotion_accepted():
    """Our v2 candidate passes the predefined gates."""
    result = evaluate_promotion(
        champion_ap=0.6186,
        candidate_ap=0.6482,
        candidate_recall=0.6765,
    )

    assert result["approved"] is True
    assert result["failed_gates"] == []


def test_rejection_insufficient_improvement():
    """A small improvement is not enough."""
    result = evaluate_promotion(
        champion_ap=0.6482,
        candidate_ap=0.6550,
        candidate_recall=0.70,
    )

    assert result["approved"] is False
    assert "minimum_improvement" in result["failed_gates"]


def test_rejection_low_recall():
    """A candidate cannot ignore too many failures."""
    result = evaluate_promotion(
        champion_ap=0.6482,
        candidate_ap=0.70,
        candidate_recall=0.40,
    )

    assert result["approved"] is False
    assert "minimum_recall" in result["failed_gates"]


def test_rejection_low_average_precision():
    """The minimum quality floor must be respected."""
    result = evaluate_promotion(
        champion_ap=0.30,
        candidate_ap=0.38,
        candidate_recall=0.70,
    )

    assert result["approved"] is False
    assert "minimum_average_precision" in result["failed_gates"]


@pytest.mark.parametrize("invalid_ap", [-0.1, 1.1, float("nan")])
def test_rejection_invalid_metrics(invalid_ap):
    """Invalid metric values must never allow promotion."""
    with pytest.raises(ValueError):
        evaluate_promotion(
            champion_ap=0.50,
            candidate_ap=invalid_ap,
            candidate_recall=0.70,
        )
