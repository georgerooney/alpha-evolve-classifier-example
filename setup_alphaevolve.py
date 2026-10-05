#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provision the Gemini Enterprise engine + assistant that AlphaEvolve runs under.

Automates "Create Discovery Engine resources" from the AlphaEvolve setup guide
(docs.cloud.google.com/gemini/enterprise/docs/alphaevolve/...). Settings come from .env, the same file the
AlphaEvolve client reads, so the engine that gets created is the one the experiment uses. Uses stdlib + local
`gcloud` user credentials so it runs before any `pip install`.

    python3 setup_alphaevolve.py            # idempotent create (engine, wait, assistant)
    python3 setup_alphaevolve.py status     # list engines in the project
    python3 setup_alphaevolve.py delete     # delete GE_APP_ID
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

Api = Callable[..., tuple[int, dict[str, Any]]]


class ApiError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Config:
    project_id: str
    engine_id: str
    assistant_id: str = "default_assistant"
    location: str = "us"
    collection: str = "default_collection"
    base_url: str = "us-discoveryengine.googleapis.com"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Config:
        missing = [k for k in ("PROJECT_ID", "GE_APP_ID") if not env.get(k)]
        if missing:
            raise SystemExit(f"Missing required setting(s) in .env: {', '.join(missing)}")
        location = (env.get("LOCATION") or cls.location).lower()
        raw_base = (env.get("BASE_URL") or "discoveryengine.googleapis.com").removeprefix("https://")
        for prefix in ("us-", "eu-"):
            raw_base = raw_base.removeprefix(prefix)
        base_url = f"{location}-{raw_base}" if location in ("us", "eu") else raw_base
        return cls(
            project_id=env["PROJECT_ID"],
            engine_id=env["GE_APP_ID"],
            assistant_id=env.get("ASSISTANT") or cls.assistant_id,
            location=location,
            collection=env.get("COLLECTION") or cls.collection,
            base_url=base_url,
        )

    @property
    def engines(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/collections/{self.collection}/engines"

    @property
    def engine(self) -> str:
        return f"{self.engines}/{self.engine_id}"


def load_dotenv(path: pathlib.Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _check(status: int, body: dict, what: str) -> dict:
    if status >= 400:
        msg = body.get("error", {}).get("message", body)
        raise ApiError(f"{what} failed (HTTP {status}): {msg}")
    return body


def _exists(api: Api, path: str) -> bool:
    status, body = api("GET", path)
    if status == 404:
        return False
    _check(status, body, f"GET {path}")
    return True


def ensure(api: Api, cfg: Config, poll_seconds: float = 10, max_polls: int = 60) -> None:
    """Create the engine (async on the server) and its assistant, skipping whatever already exists."""
    if _exists(api, cfg.engine):
        print(f"engine {cfg.engine_id}: exists")
    else:
        _check(
            *api(
                "POST",
                f"{cfg.engines}?engineId={cfg.engine_id}",
                {
                    "display_name": cfg.engine_id,
                    "data_store_ids": [],
                    "solution_type": "SOLUTION_TYPE_GENERATIVE_CHAT",
                },
            ),
            f"create engine {cfg.engine_id}",
        )
        print(f"engine {cfg.engine_id}: creating (takes a few minutes)", end="", flush=True)
        for _ in range(max_polls):
            if _exists(api, cfg.engine):
                print(" ready")
                break
            print(".", end="", flush=True)
            time.sleep(poll_seconds)
        else:
            raise TimeoutError(f"engine {cfg.engine_id} not ready after {max_polls} polls; re-run to resume")

    assistant = f"{cfg.engine}/assistants/{cfg.assistant_id}"
    if _exists(api, assistant):
        print(f"assistant {cfg.assistant_id}: exists")
        return
    _check(
        *api(
            "POST",
            f"{cfg.engine}/assistants?assistantId={cfg.assistant_id}",
            {
                "display_name": cfg.assistant_id,
                "web_grounding_type": "WEB_GROUNDING_TYPE_UNSPECIFIED",
            },
        ),
        f"create assistant {cfg.assistant_id}",
    )
    print(f"assistant {cfg.assistant_id}: created")


def list_engines(api: Api, cfg: Config) -> list[str]:
    body = _check(*api("GET", cfg.engines), "list engines")
    return sorted(e["name"].rsplit("/", 1)[-1] for e in body.get("engines", []))


def delete_engine(api: Api, cfg: Config) -> None:
    _check(*api("DELETE", cfg.engine), f"delete engine {cfg.engine_id}")


def http_api(cfg: Config) -> Api:
    cmd = ["gcloud", "auth", "print-access-token"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode:
        lines = proc.stderr.strip().splitlines() or [""]
        raise SystemExit(
            f"{' '.join(cmd)} failed:\n{next((l for l in lines if l.startswith('ERROR')), lines[-1])}"
        )
    token = proc.stdout.strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": cfg.project_id,
    }

    def api(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"https://{cfg.base_url}/v1alpha/{path}", data, headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return e.code, {"error": {"message": raw.decode(errors="replace")[:500]}}

    return api


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", nargs="?", default="setup", choices=["setup", "status", "delete"])
    p.add_argument("--env-file", default=pathlib.Path(__file__).with_name(".env"), type=pathlib.Path)
    args = p.parse_args()

    env = {**load_dotenv(args.env_file), **os.environ} if args.env_file.exists() else dict(os.environ)
    cfg = Config.from_env(env)
    api = http_api(cfg)
    try:
        if args.command == "setup":
            ensure(api, cfg)
        elif args.command == "status":
            names = list_engines(api, cfg)
            print("\n".join(f"{n}{'  <- GE_APP_ID' if n == cfg.engine_id else ''}" for n in names) or "(no engines)")
        else:
            delete_engine(api, cfg)
            print(f"engine {cfg.engine_id}: delete requested")
    except (ApiError, TimeoutError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
