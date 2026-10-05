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

"""Label-shuffle leakage check for real-data finalists: refit with permuted training labels; AUC must collapse to ~0.5.

Why: a candidate that still scores well when trained on shuffled labels is getting signal from somewhere other than
`y_train` (eval-time leakage, row-order artefacts, ID columns). Uses the fitness rounds (train/selection partitions)
only, so it never touches the locked test set and can be re-run freely.

    .venv/bin/python scripts/leakage_check.py --problem diabetes130_readmit30 artifacts/runs/<pid>/<exp>/rank1.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import types
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
# Same as the sandbox (evaluator._sandbox_worker): candidates with n_jobs=-1 otherwise grab every core. Must run
# before numpy/lightgbm load their thread pools.
os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

import numpy as np  # noqa: E402

import evaluator  # noqa: E402

# Under H0 (no signal) AUC has SD sqrt((n_pos + n_neg + 1) / (12 n_pos n_neg)) per round (≈0.011 on Diabetes130's
# 6.6k-row shards). PASS = shuffled mean within this many standard errors of 0.5.
PASS_Z = 4.0


def _mean_auc(res: Dict[str, Any]) -> float:
    return float(np.mean([r["roc_auc"] for r in res["rounds"]]))


def _null_auc_var(res: Dict[str, Any]) -> float:
    """Mean per-round variance of AUC under the null, from round size and prevalence."""
    out = []
    for r in res["rounds"]:
        n_pos = max(1.0, r["prevalence"] * r["n_eval"])
        n_neg = max(1.0, r["n_eval"] - n_pos)
        out.append((n_pos + n_neg + 1) / (12 * n_pos * n_neg))
    return float(np.mean(out))


def label_shuffle_check(code: str, *, problem_id: str, n_shuffles: int = 3, seed: int = 0) -> Dict[str, Any]:
    """Scores `code` on the fitness rounds with true labels and with `n_shuffles` label permutations.

    Trusted in-process only (finalists we've already reviewed), same as `tabular_report.py`.
    """
    mod = types.ModuleType("leakage_candidate")
    exec(compile(code, "leakage_candidate.py", "exec"), mod.__dict__)  # noqa: S102
    fit = mod.fit_and_score_risk
    true_res = evaluator.evaluate_tabular_in_process(fit, problem_id=problem_id)

    shuffled = []
    for k in range(n_shuffles):
        rng = np.random.default_rng(seed + k)
        shuffled_fit = lambda task, train, y, ev, _rng=rng: fit(task, train, _rng.permutation(y), ev)  # noqa: E731
        shuffled.append(_mean_auc(evaluator.evaluate_tabular_in_process(shuffled_fit, problem_id=problem_id)))
    mean = float(np.mean(shuffled))
    se = float(np.sqrt(_null_auc_var(true_res) / (len(true_res["rounds"]) * n_shuffles)))
    return {
        "true_auc": round(_mean_auc(true_res), 4),
        "shuffled_aucs": [round(a, 4) for a in shuffled],
        "shuffled_auc_mean": round(mean, 4),
        "null_se": round(se, 4),
        "z": round((mean - 0.5) / se, 2),
        "verdict": "PASS" if abs(mean - 0.5) < PASS_Z * se else "FAIL",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problem", default="diabetes130_readmit30")
    ap.add_argument("--n-shuffles", type=int, default=3)
    ap.add_argument("programs", nargs="+", type=pathlib.Path)
    args = ap.parse_args()
    results = {}
    for path in args.programs:
        results[str(path)] = res = label_shuffle_check(
            path.read_text(encoding="utf-8"), problem_id=args.problem, n_shuffles=args.n_shuffles
        )
        print(f"{path}: {json.dumps(res)}", flush=True)
    sys.exit(0 if all(r["verdict"] == "PASS" for r in results.values()) else 1)


if __name__ == "__main__":
    main()
