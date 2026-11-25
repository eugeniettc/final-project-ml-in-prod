
"""Redeploy Render and verify the exact MLflow model version."""

import argparse
import json
import os
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ml.verify_deployment import verify


def deploy_and_verify(url, version, sha):
    hook = os.getenv("RENDER_PRODUCTION_DEPLOY_HOOK_URL")
    api_key = os.getenv("RENDER_API_KEY")
    service_id = os.getenv("RENDER_PRODUCTION_SERVICE_ID")

    if not all([hook, api_key, service_id]):
        raise RuntimeError("Missing Render deployment credentials")

    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Expected a full Git commit SHA")

    parsed = urlsplit(hook)

    if parsed.scheme != "https" or parsed.hostname != "api.render.com":
        raise ValueError("Invalid Render deploy hook URL")

    query = dict(parse_qsl(parsed.query))
    query["ref"] = sha

    deploy_url = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(query),
        parsed.fragment,
    ))

    request = Request(deploy_url, data=b"", method="POST")

    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Render deploy hook returned HTTP {response.status}"
            )
        deployment = json.load(response)

    deploy_id = deployment["deploy"]["id"]
    print("Render deployment started:", deploy_id)

    api_url = (
        f"https://api.render.com/v1/services/"
        f"{service_id}/deploys/{deploy_id}"
    )

    for attempt in range(80):
        request = Request(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )

        with urlopen(request, timeout=30) as response:
            deployment = json.load(response)

        status = deployment["status"]
        print(f"Render check {attempt + 1}: {status}")

        if status == "live":
            deployed_sha = (deployment.get("commit") or {}).get("id")

            if deployed_sha != sha:
                raise RuntimeError("Wrong Git commit deployed")

            print("Expected Git commit is live.")
            break

        if status in {
            "build_failed",
            "update_failed",
            "pre_deploy_failed",
            "canceled",
            "deactivated",
        }:
            raise RuntimeError(f"Render deployment failed: {status}")

        time.sleep(15)
    else:
        raise TimeoutError("Render deployment did not become live")

    # The new container must serve the expected MLflow version.
    for attempt in range(20):
        try:
            result = verify(url, version)
            print("\n=== RENDER DEPLOYMENT VERIFIED ===")
            print("Git commit:", sha)
            print("MLflow model version:", version)
            print("Prediction:", result["failure_prediction"])
            print("Status: PASSED")
            return
        except Exception:
            if attempt == 19:
                raise
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--commit-sha")
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    if not args.execute:
        print("READ-ONLY verification: no Render deployment.")
        verify(args.url, args.expected_version)
        print("Live model verification PASSED.")
        return

    if not args.commit_sha:
        parser.error("--commit-sha is required with --execute")

    deploy_and_verify(
        args.url,
        args.expected_version,
        args.commit_sha,
    )


if __name__ == "__main__":
    main()
