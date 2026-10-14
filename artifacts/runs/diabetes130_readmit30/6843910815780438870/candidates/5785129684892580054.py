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

    # High-cardinality target encodings for diagnoses
    for col in ["diag_1", "diag_2", "diag_3"]:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=80.0, seed=42)
        train_copy[f"{col}_te"] = tr_enc
        eval_copy[f"{col}_te"] = ev_enc
        extra_num.append(f"{col}_te")

    # Diagnosis aggregated risk (max and mean across all 3 diagnoses)
    train_copy["max_diag_te"] = np.maximum(np.maximum(train_copy["diag_1_te"], train_copy["diag_2_te"]), train_copy["diag_3_te"])
    eval_copy["max_diag_te"] = np.maximum(np.maximum(eval_copy["diag_1_te"], eval_copy["diag_2_te"]), eval_copy["diag_3_te"])
    train_copy["mean_diag_te"] = (train_copy["diag_1_te"] + train_copy["diag_2_te"] + train_copy["diag_3_te"]) / 3.0
    eval_copy["mean_diag_te"] = (eval_copy["diag_1_te"] + eval_copy["diag_2_te"] + eval_copy["diag_3_te"]) / 3.0
    extra_num.extend(["max_diag_te", "mean_diag_te"])

    # High-cardinality target encodings for encounter attributes
    for col in ["medical_specialty", "discharge_disposition_id", "admission_source_id", "admission_type_id"]:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=30.0, seed=42)
        train_copy[f"{col}_te"] = tr_enc
        eval_copy[f"{col}_te"] = ev_enc
        extra_num.append(f"{col}_te")

    # Composite admission-to-discharge pathway target encoding
    adm_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["admission_type_id"], train["discharge_disposition_id"])], dtype=object)
    adm_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["admission_type_id"], eval_cols["discharge_disposition_id"])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(adm_disch_tr, y_train, adm_disch_ev, smoothing=30.0, seed=42)
    train_copy["adm_disch_te"] = tr_enc
    eval_copy["adm_disch_te"] = ev_enc
    extra_num.append("adm_disch_te")

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

    # Charlson comorbidities across all diagnoses
    train_comor = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    eval_comor = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in train_comor:
        train_copy[k] = train_comor[k]
        eval_copy[k] = eval_comor[k]
        extra_num.append(k)

    # Weighted Charlson Comorbidity Index
    def compute_charlson_weighted(comor):
        return (
            comor["cm_mi"] * 1.0 +
            comor["cm_chf"] * 1.0 +
            comor["cm_pvd"] * 1.0 +
            comor["cm_cerebrovascular"] * 1.0 +
            comor["cm_dementia"] * 1.0 +
            comor["cm_pulmonary"] * 1.0 +
            comor["cm_rheumatic"] * 1.0 +
            comor["cm_peptic_ulcer"] * 1.0 +
            comor["cm_liver"] * 1.0 +
            comor["cm_hemiplegia"] * 2.0 +
            comor["cm_renal"] * 2.0 +
            comor["cm_cancer"] * 2.0 +
            comor["cm_diabetes_complicated"] * 2.0 +
            comor["cm_metastatic"] * 6.0 +
            comor["cm_hiv"] * 6.0
        )

    train_copy["charlson_weighted"] = compute_charlson_weighted(train_comor)
    eval_copy["charlson_weighted"] = compute_charlson_weighted(eval_comor)
    extra_num.append("charlson_weighted")

    # Primary diagnosis comorbidities (reason for admission)
    train_p_comor = icd9_comorbidities(train["diag_1"])
    eval_p_comor = icd9_comorbidities(eval_cols["diag_1"])
    train_copy["p_cm_chf"] = train_p_comor["cm_chf"]
    eval_copy["p_cm_chf"] = eval_p_comor["cm_chf"]
    train_copy["p_cm_diabetes_comp"] = train_p_comor["cm_diabetes_complicated"]
    eval_copy["p_cm_diabetes_comp"] = eval_p_comor["cm_diabetes_complicated"]
    train_copy["p_cm_pulmonary"] = train_p_comor["cm_pulmonary"]
    eval_copy["p_cm_pulmonary"] = eval_p_comor["cm_pulmonary"]
    train_copy["p_cm_renal"] = train_p_comor["cm_renal"]
    eval_copy["p_cm_renal"] = eval_p_comor["cm_renal"]
    extra_num.extend(["p_cm_chf", "p_cm_diabetes_comp", "p_cm_pulmonary", "p_cm_renal"])

    # Utilization intensity representations
    train_copy["util_idx"] = train["number_inpatient"] + train["number_emergency"] + train["number_outpatient"]
    eval_copy["util_idx"] = eval_cols["number_inpatient"] + eval_cols["number_emergency"] + eval_cols["number_outpatient"]
    extra_num.append("util_idx")

    train_copy["has_prior_visits"] = (train_copy["util_idx"] > 0).astype(np.float64)
    eval_copy["has_prior_visits"] = (eval_copy["util_idx"] > 0).astype(np.float64)

    train_copy["has_inpatient"] = (train["number_inpatient"] > 0).astype(np.float64)
    eval_copy["has_inpatient"] = (eval_cols["number_inpatient"] > 0).astype(np.float64)

    train_copy["frequent_inpatient"] = (train["number_inpatient"] >= 2).astype(np.float64)
    eval_copy["frequent_inpatient"] = (eval_cols["number_inpatient"] >= 2).astype(np.float64)

    train_copy["log_inpatient"] = np.log1p(train["number_inpatient"])
    eval_copy["log_inpatient"] = np.log1p(eval_cols["number_inpatient"])

    train_copy["log_emergency"] = np.log1p(train["number_emergency"])
    eval_copy["log_emergency"] = np.log1p(eval_cols["number_emergency"])

    train_copy["log_util_idx"] = np.log1p(train_copy["util_idx"])
    eval_copy["log_util_idx"] = np.log1p(eval_copy["util_idx"])

    train_copy["inpatient_ratio"] = train["number_inpatient"] / (train_copy["util_idx"] + 1.0)
    eval_copy["inpatient_ratio"] = eval_cols["number_inpatient"] / (eval_copy["util_idx"] + 1.0)

    train_copy["emergency_ratio"] = train["number_emergency"] / (train_copy["util_idx"] + 1.0)
    eval_copy["emergency_ratio"] = eval_cols["number_emergency"] / (eval_copy["util_idx"] + 1.0)

    train_copy["inpatient_time_prod"] = train["number_inpatient"] * train["time_in_hospital"]
    eval_copy["inpatient_time_prod"] = eval_cols["number_inpatient"] * eval_cols["time_in_hospital"]

    extra_num.extend([
        "has_prior_visits", "has_inpatient", "frequent_inpatient",
        "log_inpatient", "log_emergency", "log_util_idx",
        "inpatient_ratio", "emergency_ratio", "inpatient_time_prod"
    ])

    # Hospitalization ratios / intensity
    for col in ["num_lab_procedures", "num_medications", "num_procedures"]:
        train_copy[f"{col}_per_day"] = train[col] / (train["time_in_hospital"] + 1e-5)
        eval_copy[f"{col}_per_day"] = eval_cols[col] / (eval_cols["time_in_hospital"] + 1e-5)
        extra_num.append(f"{col}_per_day")

    train_copy["total_procedures"] = train["num_lab_procedures"] + train["num_procedures"]
    eval_copy["total_procedures"] = eval_cols["num_lab_procedures"] + eval_cols["num_procedures"]

    train_copy["service_intensity"] = (train["num_lab_procedures"] + 2.0 * train["num_procedures"] + train["num_medications"]) / (train["time_in_hospital"] + 1.0)
    eval_copy["service_intensity"] = (eval_cols["num_lab_procedures"] + 2.0 * eval_cols["num_procedures"] + eval_cols["num_medications"]) / (eval_cols["time_in_hospital"] + 1.0)
    extra_num.extend(["total_procedures", "service_intensity"])

    # High-impact binary disposition & source markers
    train_copy["discharged_to_home"] = (train["discharge_disposition_id"] == "1").astype(np.float64)
    eval_copy["discharged_to_home"] = (eval_cols["discharge_disposition_id"] == "1").astype(np.float64)

    train_copy["emergency_admission"] = (train["admission_source_id"] == "7").astype(np.float64)
    eval_copy["emergency_admission"] = (eval_cols["admission_source_id"] == "7").astype(np.float64)
    extra_num.extend(["discharged_to_home", "emergency_admission"])

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

    # Granular insulin and medication regimen dynamics
    train_copy["insulin_down"] = (train["insulin"] == "Down").astype(np.float64)
    eval_copy["insulin_down"] = (eval_cols["insulin"] == "Down").astype(np.float64)

    train_copy["insulin_up"] = (train["insulin"] == "Up").astype(np.float64)
    eval_copy["insulin_up"] = (eval_cols["insulin"] == "Up").astype(np.float64)

    train_copy["insulin_steady"] = (train["insulin"] == "Steady").astype(np.float64)
    eval_copy["insulin_steady"] = (eval_cols["insulin"] == "Steady").astype(np.float64)

    train_copy["insulin_changed"] = np.isin(train["insulin"], ["Up", "Down"]).astype(np.float64)
    eval_copy["insulin_changed"] = np.isin(eval_cols["insulin"], ["Up", "Down"]).astype(np.float64)

    train_copy["on_metformin"] = (train["metformin"] != "No").astype(np.float64)
    eval_copy["on_metformin"] = (eval_cols["metformin"] != "No").astype(np.float64)

    train_copy["med_change_flag"] = (train["change"] == "Ch").astype(np.float64)
    eval_copy["med_change_flag"] = (eval_cols["change"] == "Ch").astype(np.float64)

    train_copy["diabetesMed_flag"] = (train["diabetesMed"] == "Yes").astype(np.float64)
    eval_copy["diabetesMed_flag"] = (eval_cols["diabetesMed"] == "Yes").astype(np.float64)

    extra_num.extend([
        "num_active_meds", "num_med_changes",
        "insulin_down", "insulin_up", "insulin_steady", "insulin_changed",
        "on_metformin", "med_change_flag", "diabetesMed_flag"
    ])

    # Age midpoint extraction and interaction
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
    train_copy["age_x_inpatient"] = train_copy["age_mid"] * train["number_inpatient"]
    eval_copy["age_x_inpatient"] = eval_copy["age_mid"] * eval_cols["number_inpatient"]
    extra_num.extend(["age_mid", "age_x_inpatient"])

    # A1C and max_glu_serum numeric mappings and testing flags
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

    train_copy["max_glu_serum_num"] = np.array([map_glu(x) for x in train["max_glu_serum"]], dtype=np.float64)
    eval_copy["max_glu_serum_num"] = np.array([map_glu(x) for x in eval_cols["max_glu_serum"]], dtype=np.float64)

    train_copy["a1c_tested"] = (train["A1Cresult"] != "None").astype(np.float64)
    eval_copy["a1c_tested"] = (eval_cols["A1Cresult"] != "None").astype(np.float64)

    train_copy["glu_tested"] = (train["max_glu_serum"] != "None").astype(np.float64)
    eval_copy["glu_tested"] = (eval_cols["max_glu_serum"] != "None").astype(np.float64)

    extra_num.extend(["A1Cresult_num", "max_glu_serum_num", "a1c_tested", "glu_tested"])

    # Fit encoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_copy, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_copy)

    # Heterogeneous LightGBM + XGBoost Ensemble
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=190,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=180,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.1,
            cat_smooth=60.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=140,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=1.0,
            cat_smooth=50.0,
            random_state=100,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=170,
            learning_rate=0.018,
            num_leaves=40,
            min_child_samples=220,
            subsample=0.62,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=0.02,
            cat_smooth=80.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.015,
            num_leaves=28,
            min_child_samples=200,
            subsample=0.68,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=5.0,
            cat_smooth=65.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.60,
            reg_lambda=2.0,
            min_child_weight=20,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        ),
        xgb.XGBClassifier(
            n_estimators=210,
            learning_rate=0.020,
            max_depth=4,
            subsample=0.80,
            colsample_bytree=0.68,
            reg_lambda=5.0,
            min_child_weight=30,
            tree_method="hist",
            eval_metric="logloss",
            random_state=777,
            n_jobs=1,
            verbosity=0,
        ),
    ]

    lgb_preds = []
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds.append(model.predict_proba(x_eval)[:, 1])
    lgb_pred = np.mean(lgb_preds, axis=0)

    xgb_preds = []
    for model in xgb_models:
        model.fit(x_train, y_train)
        xgb_preds.append(model.predict_proba(x_eval)[:, 1])
    xgb_pred = np.mean(xgb_preds, axis=0)

    return 0.70 * lgb_pred + 0.30 * xgb_pred


# EVOLVE-BLOCK-END
# =====================================================================
