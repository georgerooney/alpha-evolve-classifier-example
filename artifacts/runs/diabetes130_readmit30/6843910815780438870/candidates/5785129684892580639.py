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

    # Truncated 3-digit ICD-9 prefix extraction & prefix target encodings
    def get_pfx(val):
        if val is None or not isinstance(val, str):
            return "missing"
        return val.split(".")[0].strip()

    for col in ["diag_1", "diag_2", "diag_3"]:
        tr_pfx = np.array([get_pfx(x) for x in train[col]], dtype=object)
        ev_pfx = np.array([get_pfx(x) for x in eval_cols[col]], dtype=object)
        train_copy[f"{col}_pfx"] = tr_pfx
        eval_copy[f"{col}_pfx"] = ev_pfx
        extra_cat.append(f"{col}_pfx")

        tr_pfx_te, ev_pfx_te = oof_target_encode(tr_pfx, y_train, ev_pfx, smoothing=40.0, seed=42)
        train_copy[f"{col}_pfx_te"] = tr_pfx_te
        eval_copy[f"{col}_pfx_te"] = ev_pfx_te
        extra_num.append(f"{col}_pfx_te")

    train_copy["max_diag_pfx_te"] = np.maximum(
        np.maximum(train_copy["diag_1_pfx_te"], train_copy["diag_2_pfx_te"]),
        train_copy["diag_3_pfx_te"],
    )
    eval_copy["max_diag_pfx_te"] = np.maximum(
        np.maximum(eval_copy["diag_1_pfx_te"], eval_copy["diag_2_pfx_te"]),
        eval_copy["diag_3_pfx_te"],
    )
    train_copy["mean_diag_pfx_te"] = (
        train_copy["diag_1_pfx_te"] + train_copy["diag_2_pfx_te"] + train_copy["diag_3_pfx_te"]
    ) / 3.0
    eval_copy["mean_diag_pfx_te"] = (
        eval_copy["diag_1_pfx_te"] + eval_copy["diag_2_pfx_te"] + eval_copy["diag_3_pfx_te"]
    ) / 3.0
    extra_num.extend(["max_diag_pfx_te", "mean_diag_pfx_te"])

    # Additional non-linear utilization & acute ratios
    train_copy["has_prior_inpatient"] = (train["number_inpatient"] > 0).astype(np.float64)
    eval_copy["has_prior_inpatient"] = (eval_cols["number_inpatient"] > 0).astype(np.float64)
    train_copy["frequent_inpatient"] = (train["number_inpatient"] >= 2).astype(np.float64)
    eval_copy["frequent_inpatient"] = (eval_cols["number_inpatient"] >= 2).astype(np.float64)
    train_copy["has_prior_emergency"] = (train["number_emergency"] > 0).astype(np.float64)
    eval_copy["has_prior_emergency"] = (eval_cols["number_emergency"] > 0).astype(np.float64)

    train_copy["log_inpatient"] = np.log1p(train["number_inpatient"])
    eval_copy["log_inpatient"] = np.log1p(eval_cols["number_inpatient"])
    train_copy["log_emergency"] = np.log1p(train["number_emergency"])
    eval_copy["log_emergency"] = np.log1p(eval_cols["number_emergency"])
    train_copy["log_outpatient"] = np.log1p(train["number_outpatient"])
    eval_copy["log_outpatient"] = np.log1p(eval_cols["number_outpatient"])

    train_copy["acute_util_ratio"] = (train["number_inpatient"] + train["number_emergency"]) / (train_copy["util_idx"] + 1.0)
    eval_copy["acute_util_ratio"] = (eval_cols["number_inpatient"] + eval_cols["number_emergency"]) / (eval_copy["util_idx"] + 1.0)

    train_copy["inpatient_los_interaction"] = np.log1p(train["number_inpatient"]) * np.log1p(train["time_in_hospital"])
    eval_copy["inpatient_los_interaction"] = np.log1p(eval_cols["number_inpatient"]) * np.log1p(eval_cols["time_in_hospital"])
    train_copy["inpatient_diag_interaction"] = np.log1p(train["number_inpatient"]) * (train["number_diagnoses"] / 10.0)
    eval_copy["inpatient_diag_interaction"] = np.log1p(eval_cols["number_inpatient"]) * (eval_cols["number_diagnoses"] / 10.0)

    extra_num.extend([
        "has_prior_inpatient",
        "frequent_inpatient",
        "has_prior_emergency",
        "log_inpatient",
        "log_emergency",
        "log_outpatient",
        "acute_util_ratio",
        "inpatient_los_interaction",
        "inpatient_diag_interaction",
    ])

    # Granular high-risk discharge & admission pathways
    train_copy["discharged_to_snf"] = (train["discharge_disposition_id"] == "3").astype(np.float64)
    eval_copy["discharged_to_snf"] = (eval_cols["discharge_disposition_id"] == "3").astype(np.float64)
    train_copy["discharged_home_health"] = (train["discharge_disposition_id"] == "6").astype(np.float64)
    eval_copy["discharged_home_health"] = (eval_cols["discharge_disposition_id"] == "6").astype(np.float64)
    train_copy["discharged_ama"] = (train["discharge_disposition_id"] == "7").astype(np.float64)
    eval_copy["discharged_ama"] = (eval_cols["discharge_disposition_id"] == "7").astype(np.float64)
    train_copy["elective_admission"] = (train["admission_type_id"] == "3").astype(np.float64)
    eval_copy["elective_admission"] = (eval_cols["admission_type_id"] == "3").astype(np.float64)
    extra_num.extend(["discharged_to_snf", "discharged_home_health", "discharged_ama", "elective_admission"])

    # Detailed medication and glycemic control indicators
    train_copy["insulin_up"] = (train["insulin"] == "Up").astype(np.float64)
    eval_copy["insulin_up"] = (eval_cols["insulin"] == "Up").astype(np.float64)
    train_copy["insulin_down"] = (train["insulin"] == "Down").astype(np.float64)
    eval_copy["insulin_down"] = (eval_cols["insulin"] == "Down").astype(np.float64)
    train_copy["insulin_steady"] = (train["insulin"] == "Steady").astype(np.float64)
    eval_copy["insulin_steady"] = (eval_cols["insulin"] == "Steady").astype(np.float64)
    train_copy["on_insulin"] = (train["insulin"] != "No").astype(np.float64)
    eval_copy["on_insulin"] = (eval_cols["insulin"] != "No").astype(np.float64)
    extra_num.extend(["insulin_up", "insulin_down", "insulin_steady", "on_insulin"])

    def count_oral_meds(d):
        n = len(d["time_in_hospital"])
        oral_cnt = np.zeros(n, dtype=np.float64)
        for c in med_cols:
            if c != "insulin":
                arr = d[c]
                oral_cnt += (arr != "No") & (arr != None)
        return oral_cnt

    train_oral = count_oral_meds(train)
    eval_oral = count_oral_meds(eval_cols)
    train_copy["num_oral_meds"] = train_oral
    eval_copy["num_oral_meds"] = eval_oral
    train_copy["insulin_oral_combo"] = ((train_copy["on_insulin"] == 1.0) & (train_oral > 0)).astype(np.float64)
    eval_copy["insulin_oral_combo"] = ((eval_copy["on_insulin"] == 1.0) & (eval_oral > 0)).astype(np.float64)
    extra_num.extend(["num_oral_meds", "insulin_oral_combo"])

    # Clinical diagnostics saturation & lab test flags
    train_copy["diagnoses_saturated"] = (train["number_diagnoses"] >= 9).astype(np.float64)
    eval_copy["diagnoses_saturated"] = (eval_cols["number_diagnoses"] >= 9).astype(np.float64)
    train_copy["has_procedures"] = (train["num_procedures"] > 0).astype(np.float64)
    eval_copy["has_procedures"] = (eval_cols["num_procedures"] > 0).astype(np.float64)

    train_copy["a1c_tested"] = (train["A1Cresult"] != "None").astype(np.float64)
    eval_copy["a1c_tested"] = (eval_cols["A1Cresult"] != "None").astype(np.float64)
    train_copy["a1c_high"] = np.isin(train["A1Cresult"], [">7", ">8"]).astype(np.float64)
    eval_copy["a1c_high"] = np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(np.float64)

    train_copy["glu_tested"] = (train["max_glu_serum"] != "None").astype(np.float64)
    eval_copy["glu_tested"] = (eval_cols["max_glu_serum"] != "None").astype(np.float64)
    train_copy["glu_high"] = np.isin(train["max_glu_serum"], [">200", ">300"]).astype(np.float64)
    eval_copy["glu_high"] = np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(np.float64)

    train_copy["payer_missing"] = np.isin(train["payer_code"], [None, "?", "None"]).astype(np.float64)
    eval_copy["payer_missing"] = np.isin(eval_cols["payer_code"], [None, "?", "None"]).astype(np.float64)
    train_copy["specialty_missing"] = np.isin(train["medical_specialty"], [None, "?", "None"]).astype(np.float64)
    eval_copy["specialty_missing"] = np.isin(eval_cols["medical_specialty"], [None, "?", "None"]).astype(np.float64)

    age_tr_clean = np.where(np.isnan(train_copy["age_num"]), 65.0, train_copy["age_num"])
    age_ev_clean = np.where(np.isnan(eval_copy["age_num"]), 65.0, eval_copy["age_num"])
    train_copy["age_elderly"] = (age_tr_clean >= 70.0).astype(np.float64)
    eval_copy["age_elderly"] = (age_ev_clean >= 70.0).astype(np.float64)
    train_copy["age_inpatient"] = (age_tr_clean / 10.0) * np.log1p(train["number_inpatient"])
    eval_copy["age_inpatient"] = (age_ev_clean / 10.0) * np.log1p(eval_cols["number_inpatient"])

    extra_num.extend([
        "diagnoses_saturated",
        "has_procedures",
        "a1c_tested",
        "a1c_high",
        "glu_tested",
        "glu_high",
        "payer_missing",
        "specialty_missing",
        "age_elderly",
        "age_inpatient",
    ])

    # Fit encoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_copy, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_copy)

    # Heterogeneous 4-Model Ensemble: Deep LGBM + Reg LGBM + Hist XGBoost + Shallow XGBoost
    lgb_deep = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.015,
        num_leaves=38,
        min_child_samples=120,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.6,
        reg_alpha=0.1,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    lgb_reg = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.013,
        num_leaves=20,
        min_child_samples=220,
        subsample=0.78,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=5.0,
        reg_alpha=1.0,
        cat_smooth=70.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=240,
        learning_rate=0.018,
        max_depth=5,
        subsample=0.75,
        colsample_bytree=0.65,
        reg_lambda=4.0,
        min_child_weight=25,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    xgb_shallow = xgb.XGBClassifier(
        n_estimators=260,
        learning_rate=0.015,
        max_depth=4,
        subsample=0.80,
        colsample_bytree=0.55,
        reg_lambda=6.0,
        min_child_weight=35,
        tree_method="hist",
        eval_metric="logloss",
        random_state=999,
        n_jobs=1,
        verbosity=0,
    )

    lgb_deep.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_deep = lgb_deep.predict_proba(x_eval)[:, 1]

    lgb_reg.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_reg = lgb_reg.predict_proba(x_eval)[:, 1]

    xgb_model.fit(x_train, y_train)
    pred_xgb = xgb_model.predict_proba(x_eval)[:, 1]

    xgb_shallow.fit(x_train, y_train)
    pred_xgb_shallow = xgb_shallow.predict_proba(x_eval)[:, 1]

    # Rank-averaging to eliminate probability calibration differences between model families
    from scipy.stats import rankdata

    def to_rank(probs: np.ndarray) -> np.ndarray:
        return (rankdata(probs) - 0.5) / len(probs)

    r_deep = to_rank(pred_deep)
    r_reg = to_rank(pred_reg)
    r_xgb = to_rank(pred_xgb)
    r_xgb_s = to_rank(pred_xgb_shallow)

    return 0.38 * r_deep + 0.32 * r_reg + 0.18 * r_xgb + 0.12 * r_xgb_s


# EVOLVE-BLOCK-END
# =====================================================================
