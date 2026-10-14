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

"""Seed program for the Diabetes 130-US Hospitals 30-day readmission problem (real data).

Strategy only: a single LightGBM on the raw columns with hyper-parameters already tuned by random search on the
fitness rounds (`scripts/tabular_report.py`), so re-tuning alone cannot beat it. Encoding mechanics live in
`clinical_toolkit`.
"""

from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np

from clinical_models import TabularTask
from clinical_toolkit import TabularEncoder, icd9_comorbidities, icd9_group, oof_target_encode  # noqa: F401


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Pre-imported in the sandbox: numpy, scipy, sklearn, lightgbm, xgboost. No file/network access.


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    tr, ev = dict(train), dict(eval_cols)

    # 1. Clinical comorbidity indicators from Charlson mapping
    cms_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cms_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    tr.update(cms_tr)
    ev.update(cms_ev)
    extra_num = list(cms_tr.keys())

    # 2. Prior healthcare utilisation intensity
    tr["total_prior_visits"] = train["number_inpatient"] + train["number_emergency"] + train["number_outpatient"]
    ev["total_prior_visits"] = eval_cols["number_inpatient"] + eval_cols["number_emergency"] + eval_cols["number_outpatient"]
    tr["has_prior_inpatient"] = (train["number_inpatient"] > 0).astype(np.float64)
    ev["has_prior_inpatient"] = (eval_cols["number_inpatient"] > 0).astype(np.float64)
    extra_num.extend(["total_prior_visits", "has_prior_inpatient"])

    # 3. Target encodings for high-cardinality features
    for col in ("discharge_disposition_id", "admission_source_id", "medical_specialty", "diag_1"):
        te_name = f"{col}_te"
        tr[te_name], ev[te_name] = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0)
        extra_num.append(te_name)

    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, tr, extra_numeric=tuple(extra_num))
    x_eval = encoder.transform(ev)

    # 4. Multi-seed LightGBM ensemble for variance reduction
    preds = np.zeros(len(x_eval), dtype=np.float64)
    seeds = (42, 123, 456)
    for seed in seeds:
        model = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.0,
            cat_smooth=20.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    return preds / float(len(seeds))


# EVOLVE-BLOCK-END
# =====================================================================
