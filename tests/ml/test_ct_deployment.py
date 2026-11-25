
import pytest

from ml.promote_candidate import report_promotion
from ml.verify_deployment import verify


def test_promotion_outputs(monkeypatch, tmp_path):
    output_file = tmp_path / "github_output"
    monkeypatch.setenv(
        "GITHUB_OUTPUT", str(output_file)
    )

    report_promotion(
        decision="ELIGIBLE",
        executed=True,
        previous_version="2",
        new_version="3",
    )

    content = output_file.read_text()

    assert "promotion_decision=ELIGIBLE" in content
    assert "promotion_executed=true" in content
    assert "previous_version=2" in content
    assert "new_version=3" in content


def test_live_verification_success(monkeypatch):
    responses = {
        "/ready": {
            "status": "ready",
            "loaded_version": "2",
        },
        "/model-info": {
            "status": "loaded",
            "model_name": "predictmaint-ai",
            "model_version": "2",
            "alias": "champion",
        },
        "/predict": {
            "failure_prediction": 0,
            "risk_score": 0.017,
            "model_name": "predictmaint-ai",
            "model_version": "2",
        },
    }

    monkeypatch.setattr(
        "ml.verify_deployment.fetch_json",
        lambda base, path, payload=None: responses[path],
    )

    result = verify("https://example.com", "2")

    assert result["failure_prediction"] == 0


def test_live_verification_rejects_wrong_version(
    monkeypatch,
):
    responses = {
        "/ready": {
            "status": "ready",
            "loaded_version": "2",
        },
        "/model-info": {
            "status": "loaded",
            "model_name": "predictmaint-ai",
            "model_version": "2",
            "alias": "champion",
        },
    }

    monkeypatch.setattr(
        "ml.verify_deployment.fetch_json",
        lambda base, path, payload=None: responses[path],
    )

    with pytest.raises(AssertionError):
        verify("https://example.com", "3")
