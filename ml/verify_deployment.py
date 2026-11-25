
"""Verify the exact model version served by a deployed API."""

import argparse
import json
import time
from urllib.request import Request, urlopen


PAYLOAD = {
    "type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1500,
    "torque": 40,
    "tool_wear": 20,
}


def fetch_json(base_url, path, payload=None):
    body = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )

    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=(
            {"Content-Type": "application/json"}
            if body is not None
            else {}
        ),
        method="POST" if body is not None else "GET",
    )

    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}")
        return json.load(response)


def verify(base_url, expected_version):
    ready = fetch_json(base_url, "/ready")
    info = fetch_json(base_url, "/model-info")

    assert ready["status"] == "ready"
    assert info["status"] == "loaded"

    assert info["model_name"] == "predictmaint-ai"
    assert info["alias"] == "champion"

    assert str(info["model_version"]) == expected_version
    assert str(ready["loaded_version"]) == expected_version

    prediction = fetch_json(
        base_url,
        "/predict",
        PAYLOAD,
    )

    assert prediction["failure_prediction"] in (0, 1)
    assert 0 <= prediction["risk_score"] <= 1
    assert prediction["model_name"] == "predictmaint-ai"
    assert str(prediction["model_version"]) == expected_version

    return prediction


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--attempts", type=int, default=12)

    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("--attempts must be >= 1")

    for attempt in range(args.attempts):
        try:
            result = verify(args.url, args.expected_version)

            print("\n=== LIVE DEPLOYMENT VERIFIED ===")
            print("Model version:", args.expected_version)
            print("Prediction:", result["failure_prediction"])
            print("Risk score:", result["risk_score"])
            print("Status: PASSED")
            return

        except (Exception) as exc:
            print(
                f"Attempt {attempt + 1}/{args.attempts}: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt == args.attempts - 1:
                raise RuntimeError(
                    "Live deployment verification FAILED"
                ) from exc

            time.sleep(15)


if __name__ == "__main__":
    main()
