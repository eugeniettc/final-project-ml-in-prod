
from types import SimpleNamespace

import pytest

from ml.rollback import validate_rollback_policy


def model(version, status="READY", tags=None):
    return SimpleNamespace(
        version=str(version),
        status=status,
        tags=tags or {},
    )


def approved_champion(version, previous_version):
    return model(
        version,
        tags={
            "approved_by": "continuous_training_policy",
            "previous_champion_version": str(previous_version),
        },
    )


def test_rollback_to_previous_champion_allowed():
    champion = approved_champion(3, 2)
    previous = model(2)

    validate_rollback_policy(champion, previous)


def test_rollback_to_arbitrary_version_rejected():
    champion = approved_champion(4, 3)
    target = model(2)

    with pytest.raises(RuntimeError, match="not the recorded"):
        validate_rollback_policy(champion, target)


def test_rollback_unapproved_champion_rejected():
    champion = model(3)
    target = model(2)

    with pytest.raises(RuntimeError, match="no approved CT"):
        validate_rollback_policy(champion, target)


def test_rollback_unready_model_rejected():
    champion = approved_champion(3, 2)
    target = model(2, status="FAILED")

    with pytest.raises(RuntimeError, match="not READY"):
        validate_rollback_policy(champion, target)


def test_rollback_to_same_version_rejected():
    champion = approved_champion(3, 2)
    target = model(3)

    with pytest.raises(RuntimeError, match="already"):
        validate_rollback_policy(champion, target)
