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
    import xgboost as xgb
    import numpy as np
    from clinical_toolkit import TabularEncoder, icd9_comorbidities, oof_target_encode, icd9_group

    train_copy = dict(train)
    eval_copy = dict(eval_cols)
    extra_num = []
    extra_cat = []

    # 1. High-cardinality diagnosis target encodings
    for col in ["diag_1", "diag_2", "diag_3"]:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=80.0, seed=42)
        train_copy[f"{col}_te"] = tr_enc
        eval_copy[f"{col}_te"] = ev_enc
        extra_num.append(f"{col}_te")

    # Aggregated diagnosis risk
    train_copy["max_diag_te"] = np.maximum(np.maximum(train_copy["diag_1_te"], train_copy["diag_2_te"]), train_copy["diag_3_te"])
    eval_copy["max_diag_te"] = np.maximum(np.maximum(eval_copy["diag_1_te"], eval_copy["diag_2_te"]), eval_copy["diag_3_te"])
    train_copy["mean_diag_te"] = (train_copy["diag_1_te"] + train_copy["diag_2_te"] + train_copy["diag_3_te"]) / 3.0
    eval_copy["mean_diag_te"] = (eval_copy["diag_1_te"] + eval_copy["diag_2_te"] + eval_copy["diag_3_te"]) / 3.0
    extra_num.extend(["max_diag_te", "mean_diag_te"])

    # 2. Encounter attributes target encodings & admission pathway
    for col in ["medical_specialty", "discharge_disposition_id", "admission_source_id", "admission_type_id"]:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=30.0, seed=42)
        train_copy[f"{col}_te"] = tr_enc
        eval_copy[f"{col}_te"] = ev_enc
        extra_num.append(f"{col}_te")

    adm_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["admission_type_id"], train["discharge_disposition_id"])], dtype=object)
    adm_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["admission_type_id"], eval_cols["discharge_disposition_id"])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(adm_disch_tr, y_train, adm_disch_ev, smoothing=30.0, seed=42)
    train_copy["adm_disch_te"] = tr_enc
    eval_copy["adm_disch_te"] = ev_enc
    extra_num.append("adm_disch_te")

    # 3. ICD-9 Chapter groups (categorical)
    for col in ["diag_1", "diag_2", "diag_3"]:
        train_copy[f"{col}_group"] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_copy[f"{col}_group"] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_cat.append(f"{col}_group")

    # 4. Charlson Comorbidity indicators & count
    train_comor = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    eval_comor = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in train_comor:
        train_copy[k] = train_comor[k]
        eval_copy[k] = eval_comor[k]
        extra_num.append(k)

    # 5. Non-collinear utilization and intensity ratios
    train_copy["util_idx"] = train["number_inpatient"] + train["number_emergency"] + train["number_outpatient"]
    eval_copy["util_idx"] = eval_cols["number_inpatient"] + eval_cols["number_emergency"] + eval_cols["number_outpatient"]
    train_copy["inpatient_ratio"] = train["number_inpatient"] / (train_copy["util_idx"] + 1.0)
    eval_copy["inpatient_ratio"] = eval_cols["number_inpatient"] / (eval_copy["util_idx"] + 1.0)
    extra_num.extend(["util_idx", "inpatient_ratio"])

    for col in ["num_lab_procedures", "num_medications", "num_procedures"]:
        train_copy[f"{col}_per_day"] = train[col] / (train["time_in_hospital"] + 1e-5)
        eval_copy[f"{col}_per_day"] = eval_cols[col] / (eval_cols["time_in_hospital"] + 1e-5)
        extra_num.append(f"{col}_per_day")

    train_copy["service_intensity"] = (train["num_lab_procedures"] + 2.0 * train["num_procedures"] + train["num_medications"]) / (train["time_in_hospital"] + 1.0)
    eval_copy["service_intensity"] = (eval_cols["num_lab_procedures"] + 2.0 * eval_cols["num_procedures"] + eval_cols["num_medications"]) / (eval_cols["time_in_hospital"] + 1.0)
    extra_num.append("service_intensity")

    # 6. High-impact encounter indicators
    train_copy["discharged_to_home"] = (train["discharge_disposition_id"] == "1").astype(np.float64)
    eval_copy["discharged_to_home"] = (eval_cols["discharge_disposition_id"] == "1").astype(np.float64)
    train_copy["emergency_admission"] = (train["admission_source_id"] == "7").astype(np.float64)
    eval_copy["emergency_admission"] = (eval_cols["admission_source_id"] == "7").astype(np.float64)
    extra_num.extend(["discharged_to_home", "emergency_admission"])

    # 7. Medication regimen dynamics & insulin titration
    med_cols = [c for c in task.categorical_columns if set(train[c]) & {"Steady", "Up", "Down"}]

    def compute_med_counts(d):
        n = len(d["time_in_hospital"])
        active = np.zeros(n, dtype=np.float64)
        changes = np.zeros(n, dtype=np.float64)
        for c in med_cols:
            arr = d[c]
            active += (arr != "No") & (arr != None)
            changes += (arr == "Up") | (arr == "Down")
        return active, changes

    train_copy["num_active_meds"], train_copy["num_med_changes"] = compute_med_counts(train)
    eval_copy["num_active_meds"], eval_copy["num_med_changes"] = compute_med_counts(eval_cols)
    train_copy["insulin_changed"] = np.isin(train["insulin"], ["Up", "Down"]).astype(np.float64)
    eval_copy["insulin_changed"] = np.isin(eval_cols["insulin"], ["Up", "Down"]).astype(np.float64)
    extra_num.extend(["num_active_meds", "num_med_changes", "insulin_changed"])

    # 8. Numeric continuous age and ordinal A1C
    def parse_age(arr):
        res = np.empty(len(arr), dtype=np.float64)
        for i, val in enumerate(arr):
            if val and val.startswith("["):
                parts = val[1:-1].split("-")
                res[i] = (float(parts[0]) + float(parts[1])) / 2.0
            else:
                res[i] = np.nan
        return res

    train_copy["age_num"] = parse_age(train["age"])
    eval_copy["age_num"] = parse_age(eval_cols["age"])
    extra_num.append("age_num")

    a1c_map = {"Norm": 0.0, ">7": 1.0, ">8": 2.0}
    train_copy["a1c_score"] = np.array([a1c_map.get(x, np.nan) for x in train["A1Cresult"]], dtype=np.float64)
    eval_copy["a1c_score"] = np.array([a1c_map.get(x, np.nan) for x in eval_cols["A1Cresult"]], dtype=np.float64)
    extra_num.append("a1c_score")

    # Fit encoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_copy, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_copy)

    # Heterogeneous 3-Model Duo: Deep LGBM + Regularized LGBM + Hist XGBoost
    lgb_deep = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.015,
        num_leaves=38,
        min_child_samples=130,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.5,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    lgb_reg = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.013,
        num_leaves=22,
        min_child_samples=220,
        subsample=0.78,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=4.0,
        cat_smooth=70.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=220,
        learning_rate=0.020,
        max_depth=5,
        subsample=0.75,
        colsample_bytree=0.65,
        reg_lambda=3.0,
        min_child_weight=25,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    lgb_deep.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_deep = lgb_deep.predict_proba(x_eval)[:, 1]

    lgb_reg.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_reg = lgb_reg.predict_proba(x_eval)[:, 1]

    xgb_model.fit(x_train, y_train)
    pred_xgb = xgb_model.predict_proba(x_eval)[:, 1]

    return 0.45 * pred_deep + 0.35 * pred_reg + 0.20 * pred_xgb


# EVOLVE-BLOCK-END
# =====================================================================
