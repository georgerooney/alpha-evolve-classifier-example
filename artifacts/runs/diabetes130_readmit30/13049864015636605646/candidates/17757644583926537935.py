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
    tr = dict(train)
    ev = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Low-rank Charlson comorbidity decomposition of diagnosis codes
    cm_tr = icd9_comorbidities(tr["diag_1"], tr["diag_2"], tr["diag_3"])
    cm_ev = icd9_comorbidities(ev["diag_1"], ev["diag_2"], ev["diag_3"])
    tr.update(cm_tr)
    ev.update(cm_ev)
    extra_numeric.extend(cm_tr.keys())

    # 2. ICD-9 chapter grouping decomposition
    for col in ("diag_1", "diag_2", "diag_3"):
        grp_col = f"{col}_grp"
        tr[grp_col] = np.array([icd9_group(c) for c in tr[col]], dtype=object)
        ev[grp_col] = np.array([icd9_group(c) for c in ev[col]], dtype=object)
        extra_categorical.append(grp_col)

    # 3. Rank-1 Empirical Bayes target encoding for high-cardinality features
    for col in ("diag_1", "discharge_disposition_id", "medical_specialty", "admission_source_id"):
        tr_te, ev_te = oof_target_encode(tr[col], y_train, ev[col], smoothing=25.0, seed=42)
        te_name = f"{col}_te"
        tr[te_name] = tr_te
        ev[te_name] = ev_te
        extra_numeric.append(te_name)

    # 4. Clinical utilization intensity signals
    for d in (tr, ev):
        inp = np.nan_to_num(d["number_inpatient"], nan=0.0)
        emg = np.nan_to_num(d["number_emergency"], nan=0.0)
        out = np.nan_to_num(d["number_outpatient"], nan=0.0)
        time_hosp = np.maximum(np.nan_to_num(d["time_in_hospital"], nan=1.0), 1.0)
        meds = np.nan_to_num(d["num_medications"], nan=0.0)

        pv = inp + emg + out
        d["total_prior_visits"] = pv
        d["inpatient_ratio"] = inp / np.maximum(pv, 1.0)
        d["med_density"] = meds / time_hosp
    extra_numeric.extend(["total_prior_visits", "inpatient_ratio", "med_density"])

    # 5. Fit and transform tabular features
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, tr, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_idx = encoder.categorical_indices

    # 6. Ensemble optimization: blend deep and shallow architectures for variance reduction
    base_params = dict(
        learning_rate=0.025,
        n_estimators=220,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        cat_smooth=50.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=1,
        verbose=-1,
    )

    m1 = lgb.LGBMClassifier(**base_params, num_leaves=31, colsample_bytree=0.65, random_state=42)
    m1.fit(x_train, y_train, categorical_feature=cat_idx)
    p1 = m1.predict_proba(x_eval)[:, 1]

    m2 = lgb.LGBMClassifier(**base_params, num_leaves=18, colsample_bytree=0.55, random_state=2024)
    m2.fit(x_train, y_train, categorical_feature=cat_idx)
    p2 = m2.predict_proba(x_eval)[:, 1]

    return 0.5 * p1 + 0.5 * p2


# EVOLVE-BLOCK-END
# =====================================================================
