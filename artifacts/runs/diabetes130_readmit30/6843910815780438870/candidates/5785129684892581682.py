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


import numpy as np
import lightgbm as lgb


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return readmission probabilities for `eval_cols`."""
    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    n_tr = len(y_train)
    n_ev = len(eval_data["time_in_hospital"])

    # 1. Standard comorbidity profile directly from clinical mapping
    cm_train = icd9_comorbidities(train_data["diag_1"], train_data["diag_2"], train_data["diag_3"])
    cm_eval = icd9_comorbidities(eval_data["diag_1"], eval_data["diag_2"], eval_data["diag_3"])
    for k, arr in cm_train.items():
        train_data[k] = arr
        eval_data[k] = cm_eval[k]
        extra_numeric.append(k)

    # 2. 3-digit ICD-9 disease roots (collapsing sub-variants into core disease codes)
    for col in ["diag_1", "diag_2", "diag_3"]:
        root_name = f"{col}_root"
        for d in (train_data, eval_data):
            d[root_name] = np.array(
                [str(c)[:3] if c is not None and str(c) != "" else None for c in d[col]],
                dtype=object,
            )
        extra_categorical.append(root_name)

    # 3. Target encodings of high-cardinality clinical categories
    for col in ["diag_1", "diag_2", "diag_3", "diag_1_root", "discharge_disposition_id", "medical_specialty"]:
        te_tr, te_ev = oof_target_encode(
            train_data[col], y_train, eval_data[col], n_folds=5, smoothing=20.0, seed=42
        )
        fn = f"{col}_te"
        train_data[fn] = te_tr
        eval_data[fn] = te_ev
        extra_numeric.append(fn)

    # Composite maximum-diagnosis risk: patient risk is bounded by their worst diagnosis
    train_data["max_diag_te"] = np.maximum(
        train_data["diag_1_te"], np.maximum(train_data["diag_2_te"], train_data["diag_3_te"])
    )
    eval_data["max_diag_te"] = np.maximum(
        eval_data["diag_1_te"], np.maximum(eval_data["diag_2_te"], eval_data["diag_3_te"])
    )
    extra_numeric.append("max_diag_te")

    # 4. Essential prior utilization intensity & numeric age
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    for d in (train_data, eval_data):
        inp = d["number_inpatient"].astype(np.float64)
        emg = d["number_emergency"].astype(np.float64)
        out = d["number_outpatient"].astype(np.float64)
        hosp = np.maximum(d["time_in_hospital"].astype(np.float64), 1.0)

        tot_prior = inp + emg + out
        d["total_prior_visits"] = tot_prior
        d["inpatient_ratio"] = inp / (tot_prior + 1.0)
        d["high_inpatient"] = (inp >= 1.0).astype(np.float64)
        d["labs_per_day"] = d["num_lab_procedures"].astype(np.float64) / hosp
        d["meds_per_day"] = d["num_medications"].astype(np.float64) / hosp
        d["age_num"] = np.array([age_map.get(str(a), np.nan) for a in d["age"]], dtype=np.float64)

    extra_numeric.extend([
        "total_prior_visits", "inpatient_ratio", "high_inpatient",
        "labs_per_day", "meds_per_day", "age_num",
    ])

    # 5. Transform: preserve clinical categories with min_count=10
    encoder = TabularEncoder(min_count=10)
    encoder.fit(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_idx = encoder.categorical_indices

    # 6. Streamlined multi-seed LightGBM ensemble with optimal convergence
    preds = np.zeros(n_ev, dtype=np.float64)
    seeds = [42, 123, 999]
    for s in seeds:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            cat_smooth=35.0,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=cat_idx)
        preds += model.predict_proba(x_eval)[:, 1]

    return preds / len(seeds)


# EVOLVE-BLOCK-END
# =====================================================================
