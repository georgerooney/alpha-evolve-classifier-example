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
import xgboost as xgb


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

    # 1. Standard comorbidity profile & weighted Charlson score
    cm_train = icd9_comorbidities(train_data["diag_1"], train_data["diag_2"], train_data["diag_3"])
    cm_eval = icd9_comorbidities(eval_data["diag_1"], eval_data["diag_2"], eval_data["diag_3"])
    for k, arr in cm_train.items():
        train_data[k] = arr
        eval_data[k] = cm_eval[k]
        extra_numeric.append(k)

    charlson_weights = {
        "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
        "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
        "cm_liver": 1.0, "cm_diabetes_complicated": 2.0, "cm_hemiplegia": 2.0, "cm_renal": 2.0,
        "cm_cancer": 2.0, "cm_metastatic": 6.0, "cm_hiv": 6.0,
    }
    for d in (train_data, eval_data):
        cci = np.zeros(len(d["time_in_hospital"]), dtype=np.float64)
        for k, w in charlson_weights.items():
            if k in d:
                cci += w * d[k].astype(np.float64)
        d["charlson_score"] = cci
    extra_numeric.append("charlson_score")

    # 2. 3-digit ICD-9 disease roots & Strack chapter groups
    for col in ["diag_1", "diag_2", "diag_3"]:
        root_name = f"{col}_root"
        grp_name = f"{col}_group"
        for d in (train_data, eval_data):
            d[root_name] = np.array(
                [str(c)[:3] if c is not None and str(c) not in ("", "None", "nan") else None for c in d[col]],
                dtype=object,
            )
            d[grp_name] = np.array([icd9_group(c) for c in d[col]], dtype=object)
        extra_categorical.extend([root_name, grp_name])

    for d in (train_data, eval_data):
        n_s = len(d["time_in_hospital"])
        circ = np.zeros(n_s, dtype=np.float64)
        diab = np.zeros(n_s, dtype=np.float64)
        resp = np.zeros(n_s, dtype=np.float64)
        for col in ["diag_1_group", "diag_2_group", "diag_3_group"]:
            circ += (d[col] == "circulatory").astype(np.float64)
            diab += (d[col] == "diabetes").astype(np.float64)
            resp += (d[col] == "respiratory").astype(np.float64)
        d["num_circulatory_diags"] = circ
        d["num_diabetes_diags"] = diab
        d["num_respiratory_diags"] = resp
    extra_numeric.extend(["num_circulatory_diags", "num_diabetes_diags", "num_respiratory_diags"])

    # 3. Target encodings of high-cardinality clinical and administrative categories
    te_columns = [
        "diag_1", "diag_2", "diag_3",
        "diag_1_root", "diag_2_root", "diag_3_root",
        "diag_1_group", "diag_2_group", "diag_3_group",
        "discharge_disposition_id", "admission_source_id", "admission_type_id",
        "medical_specialty", "payer_code",
    ]
    for col in te_columns:
        if col in train_data:
            te_tr, te_ev = oof_target_encode(
                train_data[col], y_train, eval_data[col], n_folds=5, smoothing=20.0, seed=42
            )
            fn = f"{col}_te"
            train_data[fn] = te_tr
            eval_data[fn] = te_ev
            extra_numeric.append(fn)

    # Composite diagnosis risk indicators
    for d in (train_data, eval_data):
        d["max_diag_te"] = np.maximum(d["diag_1_te"], np.maximum(d["diag_2_te"], d["diag_3_te"]))
        d["mean_diag_te"] = (d["diag_1_te"] + d["diag_2_te"] + d["diag_3_te"]) / 3.0
        d["max_diag_root_te"] = np.maximum(d["diag_1_root_te"], np.maximum(d["diag_2_root_te"], d["diag_3_root_te"]))
        d["mean_diag_root_te"] = (d["diag_1_root_te"] + d["diag_2_root_te"] + d["diag_3_root_te"]) / 3.0
    extra_numeric.extend(["max_diag_te", "mean_diag_te", "max_diag_root_te", "mean_diag_root_te"])

    # 4. Medication complexity, change patterns & glycemic control
    med_candidates = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    active_meds = [m for m in med_candidates if m in train_data]

    for d in (train_data, eval_data):
        n_s = len(d["time_in_hospital"])
        num_act = np.zeros(n_s, dtype=np.float64)
        num_up = np.zeros(n_s, dtype=np.float64)
        num_down = np.zeros(n_s, dtype=np.float64)
        for m in active_meds:
            arr = d[m]
            num_act += ((arr == "Steady") | (arr == "Up") | (arr == "Down")).astype(np.float64)
            num_up += (arr == "Up").astype(np.float64)
            num_down += (arr == "Down").astype(np.float64)

        d["num_active_meds"] = num_act
        d["num_meds_up"] = num_up
        d["num_meds_down"] = num_down
        d["num_meds_changed"] = num_up + num_down
        d["has_polypharmacy"] = (num_act >= 2.0).astype(np.float64)

        if "insulin" in d:
            ins = d["insulin"]
            ins_act = ((ins == "Steady") | (ins == "Up") | (ins == "Down")).astype(np.float64)
            d["insulin_treated"] = ins_act
            d["insulin_up"] = (ins == "Up").astype(np.float64)
            d["insulin_down"] = (ins == "Down").astype(np.float64)
            d["insulin_changed"] = ((ins == "Up") | (ins == "Down")).astype(np.float64)
            d["insulin_plus_oral"] = ((ins_act > 0) & (num_act > 1)).astype(np.float64)
        else:
            d["insulin_treated"] = np.zeros(n_s, dtype=np.float64)
            d["insulin_up"] = np.zeros(n_s, dtype=np.float64)
            d["insulin_down"] = np.zeros(n_s, dtype=np.float64)
            d["insulin_changed"] = np.zeros(n_s, dtype=np.float64)
            d["insulin_plus_oral"] = np.zeros(n_s, dtype=np.float64)

        d["med_change_flag"] = (d["change"] == "Ch").astype(np.float64) if "change" in d else np.zeros(n_s, dtype=np.float64)
        d["has_diabetes_med"] = (d["diabetesMed"] == "Yes").astype(np.float64) if "diabetesMed" in d else np.zeros(n_s, dtype=np.float64)

        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            d["a1c_tested"] = (~np.isin(a1c, ["None", None])).astype(np.float64)
            d["a1c_high"] = np.isin(a1c, [">8", ">7"]).astype(np.float64)
            d["a1c_normal"] = (a1c == "Norm").astype(np.float64)
            d["a1c_high_no_change"] = ((d["a1c_high"] > 0) & (d["med_change_flag"] == 0)).astype(np.float64)
        else:
            d["a1c_tested"] = np.zeros(n_s, dtype=np.float64)
            d["a1c_high"] = np.zeros(n_s, dtype=np.float64)
            d["a1c_normal"] = np.zeros(n_s, dtype=np.float64)
            d["a1c_high_no_change"] = np.zeros(n_s, dtype=np.float64)

        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            d["glu_tested"] = (~np.isin(glu, ["None", None])).astype(np.float64)
            d["glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)
        else:
            d["glu_tested"] = np.zeros(n_s, dtype=np.float64)
            d["glu_high"] = np.zeros(n_s, dtype=np.float64)

    extra_numeric.extend([
        "num_active_meds", "num_meds_up", "num_meds_down", "num_meds_changed",
        "has_polypharmacy", "insulin_treated", "insulin_up", "insulin_down",
        "insulin_changed", "insulin_plus_oral", "med_change_flag", "has_diabetes_med",
        "a1c_tested", "a1c_high", "a1c_normal", "a1c_high_no_change",
        "glu_tested", "glu_high",
    ])

    # 5. Utilization intensity, acuity ratios & demographics
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
        labs = d["num_lab_procedures"].astype(np.float64)
        meds = d["num_medications"].astype(np.float64)
        procs = d["num_procedures"].astype(np.float64)
        diags = d["number_diagnoses"].astype(np.float64)

        tot_prior = inp + emg + out
        d["total_prior_visits"] = tot_prior
        d["log_prior_visits"] = np.log1p(tot_prior)
        d["log_inpatient"] = np.log1p(inp)
        d["log_emergency"] = np.log1p(emg)

        d["has_prior_inpatient"] = (inp >= 1.0).astype(np.float64)
        d["high_inpatient"] = (inp >= 1.0).astype(np.float64)
        d["frequent_inpatient"] = (inp >= 2.0).astype(np.float64)
        d["has_prior_emergency"] = (emg >= 1.0).astype(np.float64)

        d["inpatient_ratio"] = inp / (tot_prior + 1.0)
        d["emergency_ratio"] = emg / (tot_prior + 1.0)
        d["acute_utilization_score"] = inp * 3.0 + emg * 2.0 + out * 0.5

        d["labs_per_day"] = labs / hosp
        d["meds_per_day"] = meds / hosp
        d["procs_per_day"] = procs / hosp
        d["diag_per_day"] = diags / hosp
        d["meds_per_proc"] = meds / (procs + 1.0)
        d["labs_per_med"] = labs / (meds + 1.0)

        cm_cnt = d["cm_count"].astype(np.float64)
        d["comorbid_prior_inpatient"] = cm_cnt * inp

        d["age_num"] = np.array([age_map.get(str(a), np.nan) for a in d["age"]], dtype=np.float64)
        d["is_elderly"] = (d["age_num"] >= 70.0).astype(np.float64)
        d["elderly_comorbid"] = d["is_elderly"] * cm_cnt

        d["weight_known"] = (~np.isin(d["weight"], ["?", "None", None])).astype(np.float64)
        d["specialty_known"] = (~np.isin(d["medical_specialty"], ["?", "None", None])).astype(np.float64)
        d["payer_known"] = (~np.isin(d["payer_code"], ["?", "None", None])).astype(np.float64)

    extra_numeric.extend([
        "total_prior_visits", "log_prior_visits", "log_inpatient", "log_emergency",
        "has_prior_inpatient", "high_inpatient", "frequent_inpatient", "has_prior_emergency",
        "inpatient_ratio", "emergency_ratio", "acute_utilization_score",
        "labs_per_day", "meds_per_day", "procs_per_day", "diag_per_day",
        "meds_per_proc", "labs_per_med", "comorbid_prior_inpatient",
        "age_num", "is_elderly", "elderly_comorbid",
        "weight_known", "specialty_known", "payer_known",
    ])

    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 6. Transform tabular features
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

    # 7. Diverse LightGBM ensemble (varying depths, leaf capacities, and regularization)
    lgb_configs = [
        {
            "n_estimators": 320,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 100,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.6,
            "reg_lambda": 2.0,
            "cat_smooth": 35.0,
            "random_state": 42,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.035,
            "num_leaves": 20,
            "max_depth": 5,
            "min_child_samples": 150,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
            "cat_smooth": 25.0,
            "random_state": 123,
        },
        {
            "n_estimators": 350,
            "learning_rate": 0.025,
            "num_leaves": 45,
            "max_depth": 7,
            "min_child_samples": 80,
            "subsample": 0.70,
            "subsample_freq": 1,
            "colsample_bytree": 0.5,
            "reg_lambda": 3.0,
            "cat_smooth": 45.0,
            "random_state": 999,
        },
    ]

    lgb_preds = np.zeros(n_ev, dtype=np.float64)
    for cfg in lgb_configs:
        model = lgb.LGBMClassifier(**cfg, n_jobs=1, verbose=-1)
        model.fit(x_train, y_train, categorical_feature=cat_idx)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_configs)

    # 8. Fast histogram-based XGBoost ensemble for cross-model diversity
    xgb_preds = None
    try:
        xgb_configs = [
            {
                "n_estimators": 260,
                "learning_rate": 0.035,
                "max_depth": 5,
                "min_child_weight": 25.0,
                "subsample": 0.75,
                "colsample_bytree": 0.6,
                "reg_lambda": 3.0,
                "reg_alpha": 0.5,
                "tree_method": "hist",
                "random_state": 42,
            },
            {
                "n_estimators": 240,
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 35.0,
                "subsample": 0.80,
                "colsample_bytree": 0.7,
                "reg_lambda": 5.0,
                "reg_alpha": 1.0,
                "tree_method": "hist",
                "random_state": 123,
            },
        ]
        xgb_accum = np.zeros(n_ev, dtype=np.float64)
        for cfg in xgb_configs:
            m = xgb.XGBClassifier(**cfg, n_jobs=1, eval_metric="logloss")
            m.fit(x_train, y_train)
            xgb_accum += m.predict_proba(x_eval)[:, 1]
        xgb_preds = xgb_accum / len(xgb_configs)
    except Exception:
        xgb_preds = None

    if xgb_preds is not None:
        final_preds = 0.65 * lgb_preds + 0.35 * xgb_preds
    else:
        final_preds = lgb_preds

    return np.clip(np.nan_to_num(final_preds, nan=0.114), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
