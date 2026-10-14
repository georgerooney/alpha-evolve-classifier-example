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
    try:
        from clinical_toolkit import icd9_comorbidities, oof_target_encode
    except ImportError:
        pass
    from sklearn.decomposition import TruncatedSVD

    tr_data = dict(train)
    ev_data = dict(eval_cols)
    extra_num = []

    # 1. Low-rank clinical comorbidity factors and SVD latent axes
    cm_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in sorted(cm_tr.keys()):
        tr_data[k] = cm_tr[k]
        ev_data[k] = cm_ev[k]
        extra_num.append(k)

    cm_matrix_tr = np.column_stack([cm_tr[k] for k in sorted(cm_tr.keys()) if k != "cm_count"])
    cm_matrix_ev = np.column_stack([cm_ev[k] for k in sorted(cm_ev.keys()) if k != "cm_count"])
    svd = TruncatedSVD(n_components=3, random_state=42)
    svd_tr = svd.fit_transform(cm_matrix_tr)
    svd_ev = svd.transform(cm_matrix_ev)
    for i in range(3):
        col_name = f"cm_svd_{i}"
        tr_data[col_name] = svd_tr[:, i]
        ev_data[col_name] = svd_ev[:, i]
        extra_num.append(col_name)

    # 2. Optimal rank-1 projections (out-of-fold target encodings)
    te_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
    ]
    for col in te_cols:
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
        te_name = f"{col}_te"
        tr_data[te_name] = tr_te
        ev_data[te_name] = ev_te
        extra_num.append(te_name)

    # 3. Prior utilization intensity
    inp_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    tot_tr = inp_tr + np.nan_to_num(train["number_outpatient"], nan=0.0) + np.nan_to_num(train["number_emergency"], nan=0.0)
    tot_ev = inp_ev + np.nan_to_num(eval_cols["number_outpatient"], nan=0.0) + np.nan_to_num(eval_cols["number_emergency"], nan=0.0)

    tr_data["total_prior"] = tot_tr
    ev_data["total_prior"] = tot_ev
    tr_data["has_prior_inpatient"] = (inp_tr > 0).astype(np.float64)
    ev_data["has_prior_inpatient"] = (inp_ev > 0).astype(np.float64)
    tr_data["inpatient_ratio"] = inp_tr / (tot_tr + 1.0)
    ev_data["inpatient_ratio"] = inp_ev / (tot_ev + 1.0)
    extra_num.extend(["total_prior", "has_prior_inpatient", "inpatient_ratio"])

    # Encode tabular features
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, tr_data, extra_numeric=extra_num)
    x_train = encoder.transform(tr_data)
    x_eval = encoder.transform(ev_data)

    # 4. Multi-seed gradient boosting ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.022,
            num_leaves=31,
            min_child_samples=180,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=1.0,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.026,
            num_leaves=24,
            min_child_samples=220,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=2.0,
            cat_smooth=50.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.020,
            num_leaves=38,
            min_child_samples=150,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.5,
            cat_smooth=50.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
