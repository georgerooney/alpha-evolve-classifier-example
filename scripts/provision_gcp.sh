#!/usr/bin/env bash
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

# Local setup script for AlphaEvolve: enables GCP APIs, generates local evaluation seeds in .env,
# provisions the Gemini Enterprise engine/assistant, and fetches the UCI Diabetes 130 dataset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .env ]]; then
  cp example.env .env
  echo "Created .env from example.env"
fi

PROJECT_ID="${1:-$(grep -E '^PROJECT_ID=' .env | cut -d= -f2- | tr -d '[:space:]')}"
if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: Set PROJECT_ID in .env or pass it as the first argument."
  exit 1
fi
sed -i "s/^PROJECT_ID=.*/PROJECT_ID=${PROJECT_ID}/" .env

echo "==> [1/5] Verifying gcloud authentication for project: ${PROJECT_ID}"
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  echo "ERROR: gcloud token expired or not logged in. Run: gcloud auth login"
  exit 1
fi
gcloud config set project "${PROJECT_ID}"

echo "==> [2/5] Enabling Discovery Engine & Vertex AI APIs..."
gcloud services enable discoveryengine.googleapis.com aiplatform.googleapis.com --project="${PROJECT_ID}"

echo "==> [3/5] Generating local private evaluation seeds in .env (if not already set)..."
if [[ -z "$(grep -E '^AE_PERTURB_SEED=[0-9]+' .env || true)" ]]; then
  PERTURB_SEED="$(python3 -c 'import secrets; print(secrets.randbelow(2_000_000_000) + 100_000)')"
  sed -i "s/^AE_PERTURB_SEED=.*/AE_PERTURB_SEED=${PERTURB_SEED}/" .env
fi
if [[ -z "$(grep -E '^AE_HELDOUT_SEED=[0-9]+' .env || true)" ]]; then
  HELDOUT_SEED="$(python3 -c 'import secrets; print(secrets.randbelow(2_000_000_000) + 100_000)')"
  sed -i "s/^AE_HELDOUT_SEED=.*/AE_HELDOUT_SEED=${HELDOUT_SEED}/" .env
fi

echo "==> [4/5] Provisioning Gemini Enterprise AlphaEvolve engine & assistant..."
python3 setup_alphaevolve.py

echo "==> [5/5] Fetching UCI Diabetes 130-US Hospitals dataset..."
bash scripts/fetch_diabetes130.sh

echo "========================================================================"
echo "READY! Ensure Application Default Credentials are set:"
echo "  gcloud auth application-default login"
echo "Then launch a local AlphaEvolve evolution run with:"
echo "  .venv/bin/python run_experiment.py --problem diabetes130_readmit30"
echo "========================================================================"
