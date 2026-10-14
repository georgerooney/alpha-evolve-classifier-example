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
    for col in ["medical_specialty", "discharge_disposition_id", "admission_source_id", "admission_type_id", "payer_code"]:
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

    # 6. High-impact encounter indicators
    train_copy["discharged_to_home"] = (train["discharge_disposition_id"] == "1").astype(np.float64)
    eval_copy["discharged_to_home"] = (eval_cols["discharge_disposition_id"] == "1").astype(np.float64)
    train_copy["emergency_admission"] = (train["admission_source_id"] == "7").astype(np.float64)
    eval_copy["emergency_admission"] = (eval_cols["admission_source_id"] == "7").astype(np.float64)
    extra_num.extend(["discharged_to_home", "emergency_admission"])

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
            n_estimators=200,
            learning_rate=0.018,
            num_leaves=36,
            min_child_samples=120,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.5,
            cat_smooth=50.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_deep.fit(x_tr_f, y_tr_f, categorical_feature=encoder.categorical_indices)
        oof_deep[val_idx] = m_deep.predict_proba(x_val_f)[:, 1]
        eval_deep += m_deep.predict_proba(x_eval)[:, 1] / float(n_splits)

        m_reg = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=22,
            min_child_samples=200,
            subsample=0.78,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=4.0,
            cat_smooth=70.0,
            random_state=2024 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_reg.fit(x_tr_f, y_tr_f, categorical_feature=encoder.categorical_indices)
        oof_reg[val_idx] = m_reg.predict_proba(x_val_f)[:, 1]
        eval_reg += m_reg.predict_proba(x_eval)[:, 1] / float(n_splits)

        m_xgb = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_lambda=3.0,
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

    # Simplex Constrained Optimization for Ensemble Blending Weights
    def loss_func(weights):
        w = np.maximum(0.0, weights)
        w_sum = np.sum(w)
        if w_sum < 1e-9:
            return 1e6
        w = w / w_sum
        blend = w[0] * oof_deep + w[1] * oof_reg + w[2] * oof_xgb
        blend = np.clip(blend, 1e-7, 1.0 - 1e-7)
        return -np.mean(y_train * np.log(blend) + (1.0 - y_train) * np.log(1.0 - blend))

    init_w = np.array([0.45, 0.35, 0.20], dtype=np.float64)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success and np.sum(res.x) > 0.01:
        opt_w = np.maximum(0.0, res.x)
        opt_w = opt_w / np.sum(opt_w)
    else:
        opt_w = init_w

    return opt_w[0] * eval_deep + opt_w[1] * eval_reg + opt_w[2] * eval_xgb


# EVOLVE-BLOCK-END
# =====================================================================
