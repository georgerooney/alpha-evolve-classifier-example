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

# Downloads the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset (CC BY 4.0) into data/diabetes130/.
# The CSV checksum is pinned so every run (and every reported AUC) uses byte-identical data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/diabetes130"
URL="https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
CSV_SHA256="0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"

mkdir -p "${DEST}"
if [[ ! -f "${DEST}/diabetic_data.csv" ]]; then
  tmp_zip="$(mktemp --suffix=.zip)"
  trap 'rm -f "${tmp_zip}"' EXIT
  curl -fsSL -o "${tmp_zip}" "${URL}"
  unzip -o -q "${tmp_zip}" -d "${DEST}"
fi

echo "${CSV_SHA256}  ${DEST}/diabetic_data.csv" | sha256sum --check --quiet
echo "OK: ${DEST}/diabetic_data.csv ($(wc -l < "${DEST}/diabetic_data.csv") lines incl. header)"
