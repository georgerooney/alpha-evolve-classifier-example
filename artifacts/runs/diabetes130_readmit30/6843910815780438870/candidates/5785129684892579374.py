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

    # Multi-resolution 3-digit ICD-9 root category target encodings
    def get_diag_prefix(arr):
        res = np.empty(len(arr), dtype=object)
        for i, val in enumerate(arr):
            if val is None or val == "" or val == "None" or val == "?":
                res[i] = "missing"
            else:
                parts = str(val).strip().split(".")
                res[i] = parts[0] if parts[0] else "missing"
        return res

    for col in ["diag_1", "diag_2", "diag_3"]:
        tr_pref = get_diag_prefix(train[col])
        ev_pref = get_diag_prefix(eval_cols[col])
        tr_enc, ev_enc = oof_target_encode(tr_pref, y_train, ev_pref, smoothing=40.0, seed=42)
        train_copy[f"{col}_prefix_te"] = tr_enc
        eval_copy[f"{col}_prefix_te"] = ev_enc
        extra_num.append(f"{col}_prefix_te")

    train_copy["max_diag_prefix_te"] = np.maximum(np.maximum(train_copy["diag_1_prefix_te"], train_copy["diag_2_prefix_te"]), train_copy["diag_3_prefix_te"])
    eval_copy["max_diag_prefix_te"] = np.maximum(np.maximum(eval_copy["diag_1_prefix_te"], eval_copy["diag_2_prefix_te"]), eval_copy["diag_3_prefix_te"])
    train_copy["mean_diag_prefix_te"] = (train_copy["diag_1_prefix_te"] + train_copy["diag_2_prefix_te"] + train_copy["diag_3_prefix_te"]) / 3.0
    eval_copy["mean_diag_prefix_te"] = (eval_copy["diag_1_prefix_te"] + eval_copy["diag_2_prefix_te"] + eval_copy["diag_3_prefix_te"]) / 3.0
    extra_num.extend(["max_diag_prefix_te", "mean_diag_prefix_te"])

    # 2. Encounter attributes target encodings & admission pathway
    for col in ["medical_specialty", "discharge_disposition_id", "admission_source_id", "admission_type_id", "payer_code", "race"]:
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

    # 3. ICD-9 Chapter groups (categorical) & Chapter Target Encodings
    for col in ["diag_1", "diag_2", "diag_3"]:
        train_copy[f"{col}_group"] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_copy[f"{col}_group"] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_cat.append(f"{col}_group")
        tr_enc, ev_enc = oof_target_encode(train_copy[f"{col}_group"], y_train, eval_copy[f"{col}_group"], smoothing=40.0, seed=42)
        train_copy[f"{col}_group_te"] = tr_enc
        eval_copy[f"{col}_group_te"] = ev_enc
        extra_num.append(f"{col}_group_te")

    # Diagnostic Chapter + Discharge Disposition Interaction Target Encoding
    d1_disch_tr = np.array([f"{g}_{d}" for g, d in zip(train_copy["diag_1_group"], train["discharge_disposition_id"])], dtype=object)
    d1_disch_ev = np.array([f"{g}_{d}" for g, d in zip(eval_copy["diag_1_group"], eval_cols["discharge_disposition_id"])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(d1_disch_tr, y_train, d1_disch_ev, smoothing=30.0, seed=42)
    train_copy["diag1_group_disch_te"] = tr_enc
    eval_copy["diag1_group_disch_te"] = ev_enc
    extra_num.append("diag1_group_disch_te")

    # Frequency encodings for diagnosis prefix and encounter categorical levels
    def add_freq_encoding(tr_arr, ev_arr, name):
        counts = {}
        for x in tr_arr:
            counts[x] = counts.get(x, 0) + 1
        n = float(len(tr_arr))
        train_copy[f"{name}_freq"] = np.array([counts.get(x, 0) / n for x in tr_arr], dtype=np.float64)
        eval_copy[f"{name}_freq"] = np.array([counts.get(x, 0) / n for x in ev_arr], dtype=np.float64)
        extra_num.append(f"{name}_freq")

    add_freq_encoding(get_diag_prefix(train["diag_1"]), get_diag_prefix(eval_cols["diag_1"]), "diag_1_prefix")
    add_freq_encoding(train["medical_specialty"], eval_cols["medical_specialty"], "specialty")
    add_freq_encoding(train["discharge_disposition_id"], eval_cols["discharge_disposition_id"], "disch")

    # 4. Charlson Comorbidity indicators, counts & syndromic clinical phenotypes
    train_comor = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    eval_comor = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in train_comor:
        train_copy[k] = train_comor[k]
        eval_copy[k] = eval_comor[k]
        extra_num.append(k)

    # Cardiorenal syndrome (CHF + Renal: highest readmission risk in diabetes)
    train_copy["cardiorenal"] = train_comor["cm_chf"] * train_comor["cm_renal"]
    eval_copy["cardiorenal"] = eval_comor["cm_chf"] * eval_comor["cm_renal"]
    # Cardiometabolic risk (CHF + Complicated Diabetes)
    train_copy["cardiometabolic"] = train_comor["cm_chf"] * train_comor["cm_diabetes_complicated"]
    eval_copy["cardiometabolic"] = eval_comor["cm_chf"] * eval_comor["cm_diabetes_complicated"]
    # Cardiopulmonary overlap (COPD/Pulmonary + CHF)
    train_copy["pulmonary_cardiac"] = train_comor["cm_pulmonary"] * train_comor["cm_chf"]
    eval_copy["pulmonary_cardiac"] = eval_comor["cm_pulmonary"] * eval_comor["cm_chf"]
    # Vascular burden count
    train_copy["vascular_burden"] = train_comor["cm_mi"] + train_comor["cm_chf"] + train_comor["cm_pvd"] + train_comor["cm_cerebrovascular"]
    eval_copy["vascular_burden"] = eval_comor["cm_mi"] + eval_comor["cm_chf"] + eval_comor["cm_pvd"] + eval_comor["cm_cerebrovascular"]
    extra_num.extend(["cardiorenal", "cardiometabolic", "pulmonary_cardiac", "vascular_burden"])

    # 5. Non-collinear utilization and intensity ratios
    train_copy["util_idx"] = train["number_inpatient"] + train["number_emergency"] + train["number_outpatient"]
    eval_copy["util_idx"] = eval_cols["number_inpatient"] + eval_cols["number_emergency"] + eval_cols["number_outpatient"]
    train_copy["inpatient_ratio"] = train["number_inpatient"] / (train_copy["util_idx"] + 1.0)
    eval_copy["inpatient_ratio"] = eval_cols["number_inpatient"] / (eval_copy["util_idx"] + 1.0)
    train_copy["has_prior_inpatient"] = (train["number_inpatient"] > 0).astype(np.float64)
    eval_copy["has_prior_inpatient"] = (eval_cols["number_inpatient"] > 0).astype(np.float64)
    train_copy["has_prior_emergency"] = (train["number_emergency"] > 0).astype(np.float64)
    eval_copy["has_prior_emergency"] = (eval_cols["number_emergency"] > 0).astype(np.float64)
    train_copy["log_inpatient"] = np.log1p(train["number_inpatient"])
    eval_copy["log_inpatient"] = np.log1p(eval_cols["number_inpatient"])
    train_copy["log_util"] = np.log1p(train_copy["util_idx"])
    eval_copy["log_util"] = np.log1p(eval_copy["util_idx"])
    extra_num.extend(["util_idx", "inpatient_ratio", "has_prior_inpatient", "has_prior_emergency", "log_inpatient", "log_util"])

    for col in ["num_lab_procedures", "num_medications", "num_procedures"]:
        train_copy[f"{col}_per_day"] = train[col] / (train["time_in_hospital"] + 1e-5)
        eval_copy[f"{col}_per_day"] = eval_cols[col] / (eval_cols["time_in_hospital"] + 1e-5)
        extra_num.append(f"{col}_per_day")

    train_copy["service_intensity"] = (train["num_lab_procedures"] + 2.0 * train["num_procedures"] + train["num_medications"]) / (train["time_in_hospital"] + 1.0)
    eval_copy["service_intensity"] = (eval_cols["num_lab_procedures"] + 2.0 * eval_cols["num_procedures"] + eval_cols["num_medications"]) / (eval_cols["time_in_hospital"] + 1.0)
    train_copy["diag_per_day"] = train["number_diagnoses"] / (train["time_in_hospital"] + 1.0)
    eval_copy["diag_per_day"] = eval_cols["number_diagnoses"] / (eval_cols["time_in_hospital"] + 1.0)
    extra_num.extend(["service_intensity", "diag_per_day"])

    # 6. Clinically validated risk scores: LACE Index & HOSPITAL Score proxies
    def compute_lace(d, comor_dict):
        los = d["time_in_hospital"]
        l_pts = np.zeros(len(los), dtype=np.float64)
        l_pts[los == 1] = 1.0
        l_pts[los == 2] = 2.0
        l_pts[los == 3] = 3.0
        l_pts[(los >= 4) & (los <= 6)] = 4.0
        l_pts[(los >= 7) & (los <= 13)] = 5.0
        l_pts[los >= 14] = 7.0
        a_pts = np.where(np.isin(d["admission_type_id"], ["1", "2"]) | (d["admission_source_id"] == "7"), 3.0, 0.0)
        c_pts = np.minimum(comor_dict["cm_count"], 5.0)
        e_pts = np.minimum(d["number_emergency"], 4.0)
        return l_pts + a_pts + c_pts + e_pts

    train_copy["lace_score"] = compute_lace(train, train_comor)
    eval_copy["lace_score"] = compute_lace(eval_cols, eval_comor)

    def compute_hospital_score(d):
        los = d["time_in_hospital"]
        inp = d["number_inpatient"]
        proc = d["num_procedures"]
        s = np.zeros(len(los), dtype=np.float64)
        s += (los >= 5).astype(np.float64) * 2.0
        s += np.where(inp >= 3, 5.0, np.where(inp >= 1, 2.0, 0.0))
        s += (proc > 0).astype(np.float64)
        s += (d["admission_source_id"] == "7").astype(np.float64)
        return s

    train_copy["hospital_acuity"] = compute_hospital_score(train)
    eval_copy["hospital_acuity"] = compute_hospital_score(eval_cols)

    # Missingness and documentation burden indicators
    train_copy["weight_recorded"] = ((train["weight"] != None) & (train["weight"] != "?") & (train["weight"] != "")).astype(np.float64)
    eval_copy["weight_recorded"] = ((eval_cols["weight"] != None) & (eval_cols["weight"] != "?") & (eval_cols["weight"] != "")).astype(np.float64)
    train_copy["specialty_missing"] = ((train["medical_specialty"] == None) | (train["medical_specialty"] == "?")).astype(np.float64)
    eval_copy["specialty_missing"] = ((eval_cols["medical_specialty"] == None) | (eval_cols["medical_specialty"] == "?")).astype(np.float64)
    train_copy["payer_missing"] = ((train["payer_code"] == None) | (train["payer_code"] == "?")).astype(np.float64)
    eval_copy["payer_missing"] = ((eval_cols["payer_code"] == None) | (eval_cols["payer_code"] == "?")).astype(np.float64)

    # High-impact encounter indicators
    train_copy["discharged_to_home"] = (train["discharge_disposition_id"] == "1").astype(np.float64)
    eval_copy["discharged_to_home"] = (eval_cols["discharge_disposition_id"] == "1").astype(np.float64)
    train_copy["emergency_admission"] = (train["admission_source_id"] == "7").astype(np.float64)
    eval_copy["emergency_admission"] = (eval_cols["admission_source_id"] == "7").astype(np.float64)
    train_copy["heavy_utilizer"] = (train_copy["util_idx"] >= 4.0).astype(np.float64)
    eval_copy["heavy_utilizer"] = (eval_copy["util_idx"] >= 4.0).astype(np.float64)
    extra_num.extend([
        "lace_score", "hospital_acuity", "weight_recorded", "specialty_missing",
        "payer_missing", "discharged_to_home", "emergency_admission", "heavy_utilizer"
    ])

    # 7. Medication regimen dynamics, oral vs insulin status, and directional titration
    known_meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    med_cols = [c for c in known_meds if c in train]

    def compute_med_dynamics(d):
        n = len(d["time_in_hospital"])
        active = np.zeros(n, dtype=np.float64)
        ups = np.zeros(n, dtype=np.float64)
        downs = np.zeros(n, dtype=np.float64)
        oral_cnt = np.zeros(n, dtype=np.float64)
        for c in med_cols:
            arr = d[c]
            is_act = (arr != "No") & (arr != None)
            active += is_act
            ups += (arr == "Up")
            downs += (arr == "Down")
            if c != "insulin":
                oral_cnt += is_act
        return active, ups, downs, oral_cnt

    tr_act, tr_up, tr_down, tr_oral = compute_med_dynamics(train)
    ev_act, ev_up, ev_down, ev_oral = compute_med_dynamics(eval_cols)
    train_copy["num_active_meds"] = tr_act
    eval_copy["num_active_meds"] = ev_act
    train_copy["num_med_changes"] = tr_up + tr_down
    eval_copy["num_med_changes"] = ev_up + ev_down
    train_copy["num_med_up"] = tr_up
    eval_copy["num_med_up"] = ev_up
    train_copy["num_med_down"] = tr_down
    eval_copy["num_med_down"] = ev_down
    train_copy["num_oral_meds"] = tr_oral
    eval_copy["num_oral_meds"] = ev_oral
    train_copy["med_change_ratio"] = train_copy["num_med_changes"] / (tr_act + 1e-5)
    eval_copy["med_change_ratio"] = eval_copy["num_med_changes"] / (ev_act + 1e-5)
    train_copy["on_insulin"] = ((train["insulin"] != "No") & (train["insulin"] != None)).astype(np.float64)
    eval_copy["on_insulin"] = ((eval_cols["insulin"] != "No") & (eval_cols["insulin"] != None)).astype(np.float64)
    train_copy["insulin_changed"] = np.isin(train["insulin"], ["Up", "Down"]).astype(np.float64)
    eval_copy["insulin_changed"] = np.isin(eval_cols["insulin"], ["Up", "Down"]).astype(np.float64)
    train_copy["on_metformin"] = ((train["metformin"] != "No") & (train["metformin"] != None)).astype(np.float64)
    eval_copy["on_metformin"] = ((eval_cols["metformin"] != "No") & (eval_cols["metformin"] != None)).astype(np.float64)
    extra_num.extend([
        "num_active_meds", "num_med_changes", "num_med_up", "num_med_down",
        "num_oral_meds", "med_change_ratio", "on_insulin", "insulin_changed", "on_metformin"
    ])

    # 8. Latent Clinical Multimodal Factorization (Tensor Mode Unfolding & TruncatedSVD)
    from sklearn.decomposition import TruncatedSVD

    comor_keys = [k for k in train_comor.keys() if k != "cm_count"]
    comor_mat_tr = np.column_stack([train_comor[k] for k in comor_keys])
    comor_mat_ev = np.column_stack([eval_comor[k] for k in comor_keys])

    med_map = {"No": 0.0, "Steady": 1.0, "Up": 2.0, "Down": -1.0}
    med_mat_tr = np.column_stack([np.array([med_map.get(x, 0.0) for x in train[c]], dtype=np.float64) for c in med_cols])
    med_mat_ev = np.column_stack([np.array([med_map.get(x, 0.0) for x in eval_cols[c]], dtype=np.float64) for c in med_cols])

    unique_groups = ["circulatory", "respiratory", "digestive", "diabetes", "injury", "musculoskeletal", "genitourinary", "neoplasms", "other", "missing"]
    n_tr = len(train["time_in_hospital"])
    n_ev = len(eval_cols["time_in_hospital"])
    grp_mat_tr = np.zeros((n_tr, len(unique_groups)), dtype=np.float64)
    grp_mat_ev = np.zeros((n_ev, len(unique_groups)), dtype=np.float64)
    for j, grp in enumerate(unique_groups):
        grp_mat_tr[:, j] = (
            (train_copy["diag_1_group"] == grp).astype(np.float64)
            + (train_copy["diag_2_group"] == grp).astype(np.float64)
            + (train_copy["diag_3_group"] == grp).astype(np.float64)
        )
        grp_mat_ev[:, j] = (
            (eval_copy["diag_1_group"] == grp).astype(np.float64)
            + (eval_copy["diag_2_group"] == grp).astype(np.float64)
            + (eval_copy["diag_3_group"] == grp).astype(np.float64)
        )

    clin_mat_tr = np.hstack([comor_mat_tr, med_mat_tr, grp_mat_tr])
    clin_mat_ev = np.hstack([comor_mat_ev, med_mat_ev, grp_mat_ev])

    svd = TruncatedSVD(n_components=6, random_state=42)
    latent_tr = svd.fit_transform(clin_mat_tr)
    latent_ev = svd.transform(clin_mat_ev)

    for idx in range(6):
        fn = f"latent_factor_{idx}"
        train_copy[fn] = latent_tr[:, idx]
        eval_copy[fn] = latent_ev[:, idx]
        extra_num.append(fn)

    # Rank-1 outer-product clinical cross-domain interactions
    train_copy["latent_0_x_util"] = train_copy["latent_factor_0"] * np.log1p(train_copy["util_idx"])
    eval_copy["latent_0_x_util"] = eval_copy["latent_factor_0"] * np.log1p(eval_copy["util_idx"])
    train_copy["cm_count_x_inpatient"] = train_copy["cm_count"] * train_copy["inpatient_ratio"]
    eval_copy["cm_count_x_inpatient"] = eval_copy["cm_count"] * eval_copy["inpatient_ratio"]
    train_copy["intensity_x_inpatient"] = train_copy["service_intensity"] * np.log1p(train["number_inpatient"])
    eval_copy["intensity_x_inpatient"] = eval_copy["service_intensity"] * np.log1p(eval_cols["number_inpatient"])
    train_copy["los_x_cm"] = train["time_in_hospital"] * train_copy["cm_count"]
    eval_copy["los_x_cm"] = eval_cols["time_in_hospital"] * eval_copy["cm_count"]
    extra_num.extend(["latent_0_x_util", "cm_count_x_inpatient", "intensity_x_inpatient", "los_x_cm"])

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

    # Stratified 5-Fold Cross-Validation, Out-of-Fold Tracking & Bagged Ensembling
    from sklearn.model_selection import StratifiedKFold
    from scipy.optimize import minimize

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_deep = np.zeros(len(y_train), dtype=np.float64)
    oof_reg = np.zeros(len(y_train), dtype=np.float64)
    oof_xgb = np.zeros(len(y_train), dtype=np.float64)

    eval_deep = np.zeros(len(eval_cols["time_in_hospital"]), dtype=np.float64)
    eval_reg = np.zeros(len(eval_cols["time_in_hospital"]), dtype=np.float64)
    eval_xgb = np.zeros(len(eval_cols["time_in_hospital"]), dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr_f, y_tr_f = x_train[trn_idx], y_train[trn_idx]
        x_val_f = x_train[val_idx]

        m_deep = lgb.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.018,
            num_leaves=36,
            min_child_samples=120,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.0,
            cat_smooth=50.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_deep.fit(x_tr_f, y_tr_f, categorical_feature=encoder.categorical_indices)
        oof_deep[val_idx] = m_deep.predict_proba(x_val_f)[:, 1]
        eval_deep += m_deep.predict_proba(x_eval)[:, 1] / float(n_splits)

        m_reg = lgb.LGBMClassifier(
            n_estimators=230,
            learning_rate=0.015,
            num_leaves=22,
            min_child_samples=200,
            subsample=0.78,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=0.5,
            reg_lambda=5.0,
            cat_smooth=70.0,
            random_state=2024 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_reg.fit(x_tr_f, y_tr_f, categorical_feature=encoder.categorical_indices)
        oof_reg[val_idx] = m_reg.predict_proba(x_val_f)[:, 1]
        eval_reg += m_reg.predict_proba(x_eval)[:, 1] / float(n_splits)

        m_xgb = xgb.XGBClassifier(
            n_estimators=190,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=4.0,
            min_child_weight=25,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=1,
            verbosity=0,
        )
        m_xgb.fit(x_tr_f, y_tr_f)
        oof_xgb[val_idx] = m_xgb.predict_proba(x_val_f)[:, 1]
        eval_xgb += m_xgb.predict_proba(x_eval)[:, 1] / float(n_splits)

    # Direct ROC-AUC Metric Optimization on Simplex with Monotonic Rank Blending
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    n_tr_total = float(len(y_train))
    oof_r_deep = (rankdata(oof_deep) - 0.5) / n_tr_total
    oof_r_reg = (rankdata(oof_reg) - 0.5) / n_tr_total
    oof_r_xgb = (rankdata(oof_xgb) - 0.5) / n_tr_total

    # Coarse grid search for optimal weights on raw probabilities vs rank percentiles
    best_auc_raw, best_w_raw = -1.0, (0.45, 0.35, 0.20)
    best_auc_rank, best_w_rank = -1.0, (0.45, 0.35, 0.20)

    for i in range(21):
        w1 = i * 0.05
        for j in range(21 - i):
            w2 = j * 0.05
            w3 = 1.0 - w1 - w2
            p_raw = w1 * oof_deep + w2 * oof_reg + w3 * oof_xgb
            auc_raw = roc_auc_score(y_train, p_raw)
            if auc_raw > best_auc_raw:
                best_auc_raw = auc_raw
                best_w_raw = (w1, w2, w3)

            p_rank = w1 * oof_r_deep + w2 * oof_r_reg + w3 * oof_r_xgb
            auc_rank = roc_auc_score(y_train, p_rank)
            if auc_rank > best_auc_rank:
                best_auc_rank = auc_rank
                best_w_rank = (w1, w2, w3)

    # Select winning representation and polish weights with fine grid (+/- 0.04 in steps of 0.01)
    use_rank = best_auc_rank >= best_auc_raw
    init_center = best_w_rank if use_rank else best_w_raw
    target_deep = oof_r_deep if use_rank else oof_deep
    target_reg = oof_r_reg if use_rank else oof_reg
    target_xgb = oof_r_xgb if use_rank else oof_xgb

    opt_auc = best_auc_rank if use_rank else best_auc_raw
    opt_w = init_center

    w1_c, w2_c, _ = init_center
    delta_range = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    for dw1 in delta_range:
        w1 = np.clip(w1_c + dw1, 0.0, 1.0)
        for dw2 in delta_range:
            w2 = np.clip(w2_c + dw2, 0.0, 1.0 - w1)
            w3 = max(0.0, 1.0 - w1 - w2)
            tot = w1 + w2 + w3
            if tot < 1e-7:
                continue
            w1_n, w2_n, w3_n = w1 / tot, w2 / tot, w3 / tot
            pred = w1_n * target_deep + w2_n * target_reg + w3_n * target_xgb
            auc = roc_auc_score(y_train, pred)
            if auc > opt_auc:
                opt_auc = auc
                opt_w = (w1_n, w2_n, w3_n)

    if use_rank:
        n_ev = float(len(eval_deep))
        ev_r_deep = (rankdata(eval_deep) - 0.5) / n_ev
        ev_r_reg = (rankdata(eval_reg) - 0.5) / n_ev
        ev_r_xgb = (rankdata(eval_xgb) - 0.5) / n_ev
        return opt_w[0] * ev_r_deep + opt_w[1] * ev_r_reg + opt_w[2] * ev_r_xgb
    else:
        return opt_w[0] * eval_deep + opt_w[1] * eval_reg + opt_w[2] * eval_xgb


# EVOLVE-BLOCK-END
# =====================================================================
