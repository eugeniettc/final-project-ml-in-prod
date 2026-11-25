import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Activation explicite du mode mock pour isoler les tests du réseau
os.environ["ENABLE_MOCK_MODEL"] = "true"

from app.main import app
from app.model_loader import model_service
from app.schemas import MachineDataInput


SMOKE_PAYLOAD = {
    "type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 20.0,
}


@pytest.fixture(autouse=True)
def setup_mock_model(monkeypatch):
    """
    Fixture exécutée avant chaque test pour s'assurer
    que le service est réinitialisé avec le modèle de test.
    """
    monkeypatch.setenv("ENABLE_MOCK_MODEL", "true")
    model_service.load_model()
    yield


@pytest.fixture
def client():
    """Client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def mlflow_model(monkeypatch):
    """Exercise the real loading path with all MLflow calls mocked locally."""
    monkeypatch.setenv("ENABLE_MOCK_MODEL", "false")
    registry = MagicMock()
    registry.get_model_version_by_alias.return_value.version = "1"
    model = MagicMock(spec=["predict", "predict_proba"])
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.876543, 0.123457]])

    with patch("mlflow.MlflowClient", return_value=registry), \
         patch("mlflow.sklearn.load_model", return_value=model) as load, \
         patch("mlflow.set_tracking_uri"):
        yield model, registry, load


# =====================================================================
# TEST 1 : Prédictions réussies et conformité du format de sortie
# =====================================================================

def test_prediction_success_and_format(client):
    """
    Vérifie qu'une prédiction nominale fonctionne et respecte
    le format exact demandé (failure_prediction, risk_score, model_name, model_version).
    """
    payload_healthy = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 20.0
    }

    response = client.post("/predict", json=payload_healthy)
    assert response.status_code == 200

    data = response.json()
    assert "failure_prediction" in data
    assert data["failure_prediction"] in [0, 1]
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["model_name"] == "predictmaint-ai"
    assert data["model_version"] is not None

    # Test avec un cas critique d'usure d'outil (> 200 min)
    payload_failure = {
        "type": "H",
        "air_temperature": 304.0,
        "process_temperature": 313.0,
        "rotational_speed": 1200.0,
        "torque": 75.0,
        "tool_wear": 240.0
    }

    res_failure = client.post("/predict", json=payload_failure)
    assert res_failure.status_code == 200
    assert res_failure.json()["failure_prediction"] == 1
    assert res_failure.json()["risk_score"] > 0.5


# =====================================================================
# TEST 2 : Entrées malformées et cas limites (Edge cases)
# =====================================================================

@pytest.mark.parametrize("invalid_payload, expected_field", [
    # 1. Type inconnu (doit être L, M ou H)
    ({
        "type": "Z",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 50.0
    }, "type"),

    # 2. Température négative (physiquement impossible en Kelvin)
    ({
        "type": "M",
        "air_temperature": -10.0,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 50.0
    }, "air_temperature"),

    # 3. Vitesse de rotation négative ou nulle
    ({
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 0.0,
        "torque": 40.0,
        "tool_wear": 50.0
    }, "rotational_speed"),

    # 4. Couple négatif
    ({
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": -5.0,
        "tool_wear": 50.0
    }, "torque"),

    # 5. Usure d'outil négative
    ({
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": -1.0
    }, "tool_wear"),

    # 6. Champ manquant (oubli de 'torque')
    ({
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "tool_wear": 50.0
    }, "torque"),
])
def test_invalid_inputs_and_edge_cases(client, invalid_payload, expected_field):
    """
    Vérifie que l'API rejette strictement (erreur 422) les données aberrantes.
    """
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    errors = str(response.json())
    assert expected_field in errors


# =====================================================================
# TEST 3 : Disponibilité (/ready) et respect du modèle Champion
# =====================================================================

def test_readiness_failure_when_model_not_loaded(client):
    """
    RÈGLE D'OR : Si aucun modèle n'est prêt, /ready doit renvoyer 503
    et refuser les prédictions (pas de mock silencieux en prod).
    """
    # On simule un modèle non chargé
    model_service.is_ready = False
    model_service.model = None

    # 1. /ready doit renvoyer 503 Service Unavailable
    res_ready = client.get("/ready")
    assert res_ready.status_code == 503
    assert res_ready.json()["detail"]["status"] == "not_ready"

    # 2. /predict doit également refuser avec 503
    res_predict = client.post("/predict", json={
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 20.0
    })
    assert res_predict.status_code == 503


def test_champion_loaded_when_newer_candidate_exists(mlflow_model):
    """
    Vérifie que le service charge bien la version de l'alias 'champion',
    même si une version candidate plus récente (ex: v2) existe dans le registre.
    """
    model, registry, load = mlflow_model

    assert model_service.load_model() is True
    assert model_service.loaded_version == "1"
    assert model_service.is_ready is True
    assert model_service.model is model
    registry.get_model_version_by_alias.assert_called_once_with(
        name="predictmaint-ai", alias="champion"
    )
    load.assert_called_once_with("models:/predictmaint-ai/1")


def test_readiness_waits_for_valid_inference(mlflow_model):
    model, _, _ = mlflow_model

    def check_before_ready(frame):
        assert model_service.is_ready is False
        assert model_service.model is None
        assert model_service.loaded_version is None
        assert frame.columns.tolist() == [
            "Type", "Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear",
        ]
        assert frame.iloc[0].tolist() == ["L", 298.1, 308.6, 1500.0, 40.0, 20.0]

    def predict(frame):
        check_before_ready(frame)
        return np.array([0])

    def predict_proba(frame):
        check_before_ready(frame)
        return np.array([[0.876543, 0.123457]])

    model.predict.side_effect = predict
    model.predict_proba.side_effect = predict_proba

    with TestClient(app) as ready_client:
        assert model_service.is_ready is True
        assert model_service.error_message is None
        response = ready_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["loaded_version"] == "1"
        model.predict.assert_called_once()
        model.predict_proba.assert_called_once()

        # Serving uses the same calculation and existing rounding convention.
        model.predict.side_effect = None
        model.predict_proba.side_effect = None
        response = ready_client.post("/predict", json=SMOKE_PAYLOAD)
        assert response.status_code == 200
        assert response.json() == {
            "failure_prediction": 0,
            "risk_score": 0.1235,
            "model_name": "predictmaint-ai",
            "model_version": "1",
        }


@pytest.mark.parametrize("prediction, score", [(0, 0.0), (1, 1.0)])
def test_binary_predictions_and_score_boundaries_are_ready(mlflow_model, prediction, score):
    model, _, _ = mlflow_model
    model.predict.return_value = np.array([prediction])
    model.predict_proba.return_value = np.array([[1 - score, score]])

    assert model_service.load_model() is True
    assert model_service.is_ready is True
    assert model_service.predict(MachineDataInput(**SMOKE_PAYLOAD).to_model_dataframe()) == (
        prediction, score
    )


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_inference_failure_keeps_service_unavailable(mlflow_model, method):
    model, _, load = mlflow_model
    getattr(model, method).side_effect = RuntimeError("Inference failed")

    with TestClient(app) as failed_client:
        load.assert_called_once_with("models:/predictmaint-ai/1")
        assert model_service.is_ready is False
        assert model_service.model is None
        assert model_service.loaded_version is None
        assert "Inference failed" in model_service.error_message
        response = failed_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "not_ready"
        assert failed_client.post("/predict", json=SMOKE_PAYLOAD).status_code == 503

    with pytest.raises(RuntimeError, match="pas prêt"):
        model_service.predict(MachineDataInput(**SMOKE_PAYLOAD).to_model_dataframe())


@pytest.mark.parametrize("prediction", [-1, 2, 0.5, float("nan"), float("inf"), "1", None])
def test_invalid_prediction_fails_readiness(mlflow_model, prediction):
    model, _, _ = mlflow_model
    model.predict.return_value = [prediction]

    assert model_service.load_model() is False
    assert model_service.is_ready is False
    assert model_service.model is None
    assert model_service.loaded_version is None
    assert "binaire" in model_service.error_message


@pytest.mark.parametrize("score", [
    float("nan"), float("inf"), float("-inf"), -0.1, 1.1,
    -0.00001, 1.00001, "0.5", None, 0.5 + 0.1j,
])
def test_invalid_risk_score_fails_readiness(mlflow_model, score):
    model, _, _ = mlflow_model
    model.predict_proba.return_value = [[0.5, score]]

    assert model_service.load_model() is False
    assert model_service.is_ready is False
    assert model_service.model is None
    assert model_service.loaded_version is None
    assert "score de risque" in model_service.error_message


@pytest.mark.parametrize("failure_stage", ["alias", "load"])
def test_mlflow_failure_keeps_service_unavailable(mlflow_model, failure_stage):
    model, registry, load = mlflow_model
    failing_call = registry.get_model_version_by_alias if failure_stage == "alias" else load
    failing_call.side_effect = RuntimeError("MLflow unavailable")

    assert model_service.load_model() is False
    assert model_service.is_ready is False
    assert model_service.model is None
    assert model_service.loaded_version is None
    assert "MLflow unavailable" in model_service.error_message
    model.predict.assert_not_called()
    model.predict_proba.assert_not_called()


def test_model_without_predict_proba_preserves_binary_risk(mlflow_model):
    _, _, load = mlflow_model
    model = MagicMock(spec=["predict"])
    model.predict.return_value = np.array([1])
    load.return_value = model

    assert model_service.load_model() is True
    assert model_service.predict(MachineDataInput(**SMOKE_PAYLOAD).to_model_dataframe()) == (1, 1.0)
