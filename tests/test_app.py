import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Activation explicite du mode mock pour isoler les tests du réseau
os.environ["ENABLE_MOCK_MODEL"] = "true"

from app.main import app
from app.model_loader import model_service


@pytest.fixture(autouse=True)
def setup_mock_model():
    """
    Fixture exécutée avant chaque test pour s'assurer
    que le service est réinitialisé avec le modèle de test.
    """
    os.environ["ENABLE_MOCK_MODEL"] = "true"
    model_service.load_model()
    yield


@pytest.fixture
def client():
    """Client de test FastAPI."""
    return TestClient(app)


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


def test_champion_loaded_when_newer_candidate_exists():
    """
    Vérifie que le service charge bien la version de l'alias 'champion',
    même si une version candidate plus récente (ex: v2) existe dans le registre.
    """
    os.environ["ENABLE_MOCK_MODEL"] = "false"

    mock_client = MagicMock()
    # On simule que 'champion' pointe toujours sur la version 1
    mock_champion_info = MagicMock()
    mock_champion_info.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_champion_info

    mock_sklearn_model = MagicMock()

    with patch("mlflow.MlflowClient", return_value=mock_client), \
         patch("mlflow.sklearn.load_model", return_value=mock_sklearn_model) as mock_load, \
         patch("mlflow.set_tracking_uri"):

        # Chargement
        success = model_service.load_model()

        assert success is True
        assert model_service.loaded_version == "1"
        assert model_service.is_ready is True
        # Vérifie que c'est bien la version 1 qui a été demandée
        mock_load.assert_called_once_with("models:/predictmaint-ai/1")