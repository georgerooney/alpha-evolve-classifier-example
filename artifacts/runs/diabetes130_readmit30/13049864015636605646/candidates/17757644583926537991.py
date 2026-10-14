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
    # 1. Comorbidities across all diagnoses
    cm_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

    tr_cols = dict(train)
    ev_cols = dict(eval_cols)
    tr_cols.update(cm_tr)
    ev_cols.update(cm_ev)
    extra_num = list(cm_tr.keys())

    # 2. Target encode high-cardinality categorical features
    te_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "payer_code",
    ]
    for col in te_cols:
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0)
        te_name = f"{col}_te"
        tr_cols[te_name] = tr_te
        ev_cols[te_name] = ev_te
        extra_num.append(te_name)

    # 3. Utilization intensity
    inpatient_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inpatient_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emergency_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emergency_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    outpatient_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    outpatient_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    total_prior_tr = inpatient_tr + emergency_tr + outpatient_tr
    total_prior_ev = inpatient_ev + emergency_ev + outpatient_ev

    tr_cols["total_prior_visits"] = total_prior_tr
    ev_cols["total_prior_visits"] = total_prior_ev
    tr_cols["inpatient_ratio"] = inpatient_tr / (total_prior_tr + 1.0)
    ev_cols["inpatient_ratio"] = inpatient_ev / (total_prior_ev + 1.0)
    extra_num.extend(["total_prior_visits", "inpatient_ratio"])

    # 4. Tabular encoding with min_count=20 to preserve granular clinical codes
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, tr_cols, extra_numeric=extra_num)
    x_train = encoder.transform(tr_cols)
    x_eval = encoder.transform(ev_cols)

    # 5. Blend two complementary LightGBM models (deep vs shallow regularized)
    model1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.0,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred1 = model1.predict_proba(x_eval)[:, 1]

    model2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=20,
        min_child_samples=200,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=2.0,
        cat_smooth=50.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred2 = model2.predict_proba(x_eval)[:, 1]

    return 0.5 * (pred1 + pred2)


# EVOLVE-BLOCK-END
# =====================================================================
