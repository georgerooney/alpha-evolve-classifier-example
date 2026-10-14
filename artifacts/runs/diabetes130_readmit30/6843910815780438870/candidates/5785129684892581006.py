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
    import lightgbm as lgb
    import numpy as np
    from clinical_toolkit import TabularEncoder, icd9_comorbidities, oof_target_encode, icd9_group

    train_copy = dict(train)
    eval_copy = dict(eval_cols)
    extra_num = []
    extra_cat = []

    # High-cardinality target encoding with tuned smoothing
    for col in ["diag_1", "diag_2", "diag_3"]:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=80.0, seed=42)
        train_copy[f"{col}_te"] = tr_enc
        eval_copy[f"{col}_te"] = ev_enc
        extra_num.append(f"{col}_te")

    for col in ["medical_specialty", "discharge_disposition_id", "admission_source_id"]:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=30.0, seed=42)
        train_copy[f"{col}_te"] = tr_enc
        eval_copy[f"{col}_te"] = ev_enc
        extra_num.append(f"{col}_te")

    # Map diag_1, diag_2, diag_3 to clinical groups
    for col in ["diag_1", "diag_2", "diag_3"]:
        train_copy[f"{col}_group"] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_copy[f"{col}_group"] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_cat.append(f"{col}_group")

    # Count clinical group occurrences
    def count_group_occurrences(group_list):
        circ_cnt = np.zeros(len(group_list[0]))
        resp_cnt = np.zeros(len(group_list[0]))
        diab_cnt = np.zeros(len(group_list[0]))
        for g_arr in group_list:
            circ_cnt += (g_arr == "circulatory")
            resp_cnt += (g_arr == "respiratory")
            diab_cnt += (g_arr == "diabetes")
        return circ_cnt, resp_cnt, diab_cnt

    tr_circ, tr_resp, tr_diab = count_group_occurrences([
        train_copy["diag_1_group"], train_copy["diag_2_group"], train_copy["diag_3_group"]
    ])
    ev_circ, ev_resp, ev_diab = count_group_occurrences([
        eval_copy["diag_1_group"], eval_copy["diag_2_group"], eval_copy["diag_3_group"]
    ])

    train_copy["circ_diag_cnt"] = tr_circ
    train_copy["resp_diag_cnt"] = tr_resp
    train_copy["diab_diag_cnt"] = tr_diab
    eval_copy["circ_diag_cnt"] = ev_circ
    eval_copy["resp_diag_cnt"] = ev_resp
    eval_copy["diab_diag_cnt"] = ev_diab

    extra_num.extend(["circ_diag_cnt", "resp_diag_cnt", "diab_diag_cnt"])

    # Clinical comorbidities
    train_comor = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    eval_comor = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in train_comor:
        train_copy[k] = train_comor[k]
        eval_copy[k] = eval_comor[k]
        extra_num.append(k)

    # Simplified utilization intensity representation
    train_copy["util_idx"] = train["number_inpatient"] + train["number_emergency"] + train["number_outpatient"]
    eval_copy["util_idx"] = eval_cols["number_inpatient"] + eval_cols["number_emergency"] + eval_cols["number_outpatient"]
    extra_num.append("util_idx")

    train_copy["has_prior_visits"] = (train_copy["util_idx"] > 0).astype(np.float64)
    eval_copy["has_prior_visits"] = (eval_copy["util_idx"] > 0).astype(np.float64)
    extra_num.append("has_prior_visits")

    train_copy["inpatient_ratio"] = train["number_inpatient"] / (train_copy["util_idx"] + 1.0)
    eval_copy["inpatient_ratio"] = eval_cols["number_inpatient"] / (eval_copy["util_idx"] + 1.0)
    extra_num.append("inpatient_ratio")

    train_copy["emergency_ratio"] = train["number_emergency"] / (train_copy["util_idx"] + 1.0)
    eval_copy["emergency_ratio"] = eval_cols["number_emergency"] / (eval_copy["util_idx"] + 1.0)
    extra_num.append("emergency_ratio")

    # Hospitalization ratios / intensity
    for col in ["num_lab_procedures", "num_medications", "num_procedures"]:
        train_copy[f"{col}_per_day"] = train[col] / (train["time_in_hospital"] + 1e-5)
        eval_copy[f"{col}_per_day"] = eval_cols[col] / (eval_cols["time_in_hospital"] + 1e-5)
        extra_num.append(f"{col}_per_day")

    # Medication features: auto-detection of 23 meds
    med_cols = []
    for col in task.categorical_columns:
        vals = set(train[col])
        if any(v in vals for v in ["Steady", "Up", "Down"]):
            med_cols.append(col)

    def get_med_features(data):
        n = len(data[next(iter(data))])
        active_count = np.zeros(n)
        change_count = np.zeros(n)
        for col in med_cols:
            arr = data[col]
            active_count += np.array([x not in ("No", None) for x in arr], dtype=bool)
            change_count += np.array([x in ("Up", "Down") for x in arr], dtype=bool)
        return active_count, change_count

    tr_act, tr_chg = get_med_features(train)
    ev_act, ev_chg = get_med_features(eval_cols)

    train_copy["num_active_meds"] = tr_act
    train_copy["num_med_changes"] = tr_chg
    eval_copy["num_active_meds"] = ev_act
    eval_copy["num_med_changes"] = ev_chg
    extra_num.extend(["num_active_meds", "num_med_changes"])

    # Age midpoint extraction
    def parse_age(age_col):
        mids = []
        for val in age_col:
            if val is None or not isinstance(val, str):
                mids.append(np.nan)
                continue
            val_clean = val.replace("[", "").replace(")", "").replace("]", "")
            try:
                low, high = val_clean.split("-")
                mids.append((float(low) + float(high)) / 2.0)
            except Exception:
                mids.append(np.nan)
        return np.array(mids, dtype=np.float64)

    train_copy["age_mid"] = parse_age(train["age"])
    eval_copy["age_mid"] = parse_age(eval_cols["age"])
    extra_num.append("age_mid")

    # A1C and max_glu_serum numeric mappings
    def map_a1c(val):
        if val == "None" or val is None:
            return np.nan
        if val == "Norm":
            return 0.0
        if val == ">7":
            return 1.0
        if val == ">8":
            return 2.0
        return np.nan

    def map_glu(val):
        if val == "None" or val is None:
            return np.nan
        if val == "Norm":
            return 0.0
        if val == ">200":
            return 1.0
        if val == ">300":
            return 2.0
        return np.nan

    train_copy["A1Cresult_num"] = np.array([map_a1c(x) for x in train["A1Cresult"]], dtype=np.float64)
    eval_copy["A1Cresult_num"] = np.array([map_a1c(x) for x in eval_cols["A1Cresult"]], dtype=np.float64)
    extra_num.append("A1Cresult_num")

    train_copy["max_glu_serum_num"] = np.array([map_glu(x) for x in train["max_glu_serum"]], dtype=np.float64)
    eval_copy["max_glu_serum_num"] = np.array([map_glu(x) for x in eval_cols["max_glu_serum"]], dtype=np.float64)
    extra_num.append("max_glu_serum_num")

    # Fit encoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_copy, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_copy)

    # Diverse LightGBM Ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.69,
            subsample_freq=1,
            colsample_bytree=0.63,
            reg_lambda=0.022,
            cat_smooth=69.23,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=150,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=0.1,
            cat_smooth=50.0,
            random_state=100,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.018,
            num_leaves=45,
            min_child_samples=250,
            subsample=0.6,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=0.01,
            cat_smooth=100.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.69,
            subsample_freq=1,
            colsample_bytree=0.63,
            reg_lambda=5.0,
            cat_smooth=69.23,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.69,
            subsample_freq=1,
            colsample_bytree=0.63,
            reg_lambda=0.022,
            cat_smooth=69.23,
            random_state=12345,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = []
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds.append(model.predict_proba(x_eval)[:, 1])

    return np.mean(preds, axis=0)


# EVOLVE-BLOCK-END
# =====================================================================
