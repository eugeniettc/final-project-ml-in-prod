import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ml import rollback


@pytest.mark.parametrize(
    "extra_args, expected_model",
    [
        ([], "predictmaint-ai"),
        (
            ["--model-name", "predictmaint-ai-ct-demo-henri-fixed"],
            "predictmaint-ai-ct-demo-henri-fixed",
        ),
    ],
)
def test_rollback_cli_dry_run(
    monkeypatch, capsys, extra_args, expected_model
):
    champion = SimpleNamespace(
        version="4",
        status="READY",
        tags={
            "approved_by": "continuous_training_policy",
            "previous_champion_version": "1",
        },
    )

    target = SimpleNamespace(
        version="1",
        status="READY",
        tags={},
    )

    client = MagicMock()
    client.get_model_version_by_alias.return_value = champion
    client.get_model_version.return_value = target

    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.8, 0.2]])

    monkeypatch.setattr(rollback.dagshub, "init", lambda **kwargs: None)
    monkeypatch.setattr(rollback, "MlflowClient", lambda: client)

    mock_load = MagicMock(return_value=model)
    monkeypatch.setattr(rollback.mlflow.sklearn, "load_model", mock_load)

    test_data = pd.DataFrame([
        {feature: 0 for feature in rollback.FEATURES}
    ])
    monkeypatch.setattr(
        rollback.pd, "read_csv", lambda *args, **kwargs: test_data
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback",
            "--expected-current", "4",
            "--to-version", "1",
            *extra_args,
        ],
    )

    rollback.main()

    client.get_model_version_by_alias.assert_called_once_with(
        expected_model, "champion"
    )
    client.get_model_version.assert_called_once_with(
        expected_model, "1"
    )
    mock_load.assert_called_once_with(
        f"models:/{expected_model}/1"
    )
    client.set_registered_model_alias.assert_not_called()

    output = capsys.readouterr().out
    assert "DRY-RUN PASSED" in output
    assert f"Model name: {expected_model}" in output


def test_rollback_cli_execute(monkeypatch, capsys):
    model_name = "predictmaint-ai-ct-demo-henri-fixed"

    champion = SimpleNamespace(
        version="4",
        status="READY",
        tags={
            "approved_by": "continuous_training_policy",
            "previous_champion_version": "1",
        },
    )
    target = SimpleNamespace(version="1", status="READY", tags={})

    client = MagicMock()
    client.get_model_version_by_alias.side_effect = [
        champion,
        champion,
        target,
    ]
    client.get_model_version.return_value = target

    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.8, 0.2]])

    monkeypatch.setattr(rollback.dagshub, "init", lambda **kwargs: None)
    monkeypatch.setattr(rollback, "MlflowClient", lambda: client)
    monkeypatch.setattr(
        rollback.mlflow.sklearn,
        "load_model",
        MagicMock(return_value=model),
    )

    test_data = pd.DataFrame([
        {feature: 0 for feature in rollback.FEATURES}
    ])
    monkeypatch.setattr(
        rollback.pd, "read_csv", lambda *args, **kwargs: test_data
    )

    mock_log_params = MagicMock()
    mock_set_tags = MagicMock()

    monkeypatch.setattr(rollback.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(rollback.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(rollback.mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(rollback.mlflow, "set_tags", mock_set_tags)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback",
            "--model-name", model_name,
            "--expected-current", "4",
            "--to-version", "1",
            "--execute",
        ],
    )

    rollback.main()

    client.set_registered_model_alias.assert_called_once_with(
        name=model_name,
        alias="champion",
        version="1",
    )

    assert client.get_model_version_by_alias.call_count == 3
    assert all(
        call.args == (model_name, "champion")
        for call in client.get_model_version_by_alias.call_args_list
    )

    assert mock_log_params.call_args.args[0]["model_name"] == model_name
    assert mock_set_tags.call_args.args[0]["decision"] == "RESTORED"

    output = capsys.readouterr().out
    assert "ROLLBACK SUCCESSFUL" in output
    assert "New champion: 1" in output
