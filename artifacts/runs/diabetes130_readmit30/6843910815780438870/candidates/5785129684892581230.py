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
    import numpy as np
    import lightgbm as lgb
    from scipy.stats import rankdata

    extra_numeric = []
    extra_categorical = []

    n_train = len(y_train)
    n_eval = len(next(iter(eval_cols.values())))

    # 1. Comorbidities & Charlson Index
    diag_cols_present = [col for col in ["diag_1", "diag_2", "diag_3"] if col in train]
    if diag_cols_present:
        comorbs_train = icd9_comorbidities(*(train[col] for col in diag_cols_present))
        comorbs_eval = icd9_comorbidities(*(eval_cols[col] for col in diag_cols_present))
        for k in comorbs_train.keys():
            train[k] = comorbs_train[k]
            eval_cols[k] = comorbs_eval[k]
            extra_numeric.append(k)

        cci_weights = {
            "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
            "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
            "cm_liver": 1.0, "cm_hemiplegia": 2.0, "cm_renal": 2.0, "cm_diabetes_complicated": 2.0,
            "cm_cancer": 2.0, "cm_metastatic": 6.0, "cm_hiv": 6.0,
        }
        cci_tr = np.zeros(n_train, dtype=np.float64)
        cci_ev = np.zeros(n_eval, dtype=np.float64)
        for k, w in cci_weights.items():
            if k in train:
                cci_tr += train[k] * w
                cci_ev += eval_cols[k] * w
        train["charlson_index"] = cci_tr
        eval_cols["charlson_index"] = cci_ev
        extra_numeric.append("charlson_index")

    # 2. 3-Digit ICD-9 Truncation & Chapter Groups
    def to_3digit_icd9(x):
        if x is None or x == "" or x == "?" or x == "None" or (isinstance(x, float) and np.isnan(x)):
            return "missing"
        return str(x).strip().split(".")[0]

    def safe_icd9_group(x):
        if x is None or x == "" or x == "?" or x == "None" or (isinstance(x, float) and np.isnan(x)):
            return "missing"
        return icd9_group(str(x))

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in train:
            c3_name = f"{col}_3d"
            train[c3_name] = np.array([to_3digit_icd9(x) for x in train[col]], dtype=object)
            eval_cols[c3_name] = np.array([to_3digit_icd9(x) for x in eval_cols[col]], dtype=object)
            extra_categorical.append(c3_name)

            grp_name = f"{col}_group"
            train[grp_name] = np.array([safe_icd9_group(x) for x in train[col]], dtype=object)
            eval_cols[grp_name] = np.array([safe_icd9_group(x) for x in eval_cols[col]], dtype=object)
            extra_categorical.append(grp_name)

    # 3. Clean Medication Features
    med_cols = [
        col for col in task.categorical_columns
        if col in train and any(val in ["Steady", "Up", "Down"] for val in np.unique(train[col].astype(str)))
    ]
    if med_cols:
        act_tr = np.zeros(n_train, dtype=np.float64)
        up_tr = np.zeros(n_train, dtype=np.float64)
        down_tr = np.zeros(n_train, dtype=np.float64)
        act_ev = np.zeros(n_eval, dtype=np.float64)
        up_ev = np.zeros(n_eval, dtype=np.float64)
        down_ev = np.zeros(n_eval, dtype=np.float64)

        for col in med_cols:
            vt = train[col]
            ve = eval_cols[col]
            act_tr += (vt != "No") & (vt != None)
            up_tr += (vt == "Up")
            down_tr += (vt == "Down")
            act_ev += (ve != "No") & (ve != None)
            up_ev += (ve == "Up")
            down_ev += (ve == "Down")

        train["num_active_meds"] = act_tr
        eval_cols["num_active_meds"] = act_ev
        train["num_up_meds"] = up_tr
        eval_cols["num_up_meds"] = up_ev
        train["num_down_meds"] = down_tr
        eval_cols["num_down_meds"] = down_ev
        train["med_volatility"] = (up_tr + down_tr) / (act_tr + 1e-5)
        eval_cols["med_volatility"] = (up_ev + down_ev) / (act_ev + 1e-5)
        train["med_torque"] = up_tr - down_tr
        eval_cols["med_torque"] = up_ev - down_ev
        extra_numeric.extend(["num_active_meds", "num_up_meds", "num_down_meds", "med_volatility", "med_torque"])

        if "insulin" in train:
            for d in [train, eval_cols]:
                ins = d["insulin"]
                d["on_insulin"] = ((ins != "No") & (ins != None)).astype(float)
                d["insulin_up"] = (ins == "Up").astype(float)
                d["insulin_down"] = (ins == "Down").astype(float)
            extra_numeric.extend(["on_insulin", "insulin_up", "insulin_down"])

    # 4. Glycemic Action Paradox & Renal Contraindication
    if "A1Cresult" in train and "change" in train:
        for d in [train, eval_cols]:
            a1c = d["A1Cresult"]
            chg = d["change"]
            high_a1c = (a1c == ">8") | (a1c == ">7")
            d["a1c_tested"] = (a1c != "None").astype(float)
            d["a1c_uncontrolled_neglected"] = (high_a1c & (chg == "No")).astype(float)
            d["a1c_uncontrolled_addressed"] = (high_a1c & (chg == "Ch")).astype(float)
        extra_numeric.extend(["a1c_tested", "a1c_uncontrolled_neglected", "a1c_uncontrolled_addressed"])

    if "metformin" in train and "cm_renal" in train:
        train["metformin_renal_contra"] = ((train["metformin"] != "No") & (train["metformin"] != None) & (train["cm_renal"] > 0)).astype(float)
        eval_cols["metformin_renal_contra"] = ((eval_cols["metformin"] != "No") & (eval_cols["metformin"] != None) & (eval_cols["cm_renal"] > 0)).astype(float)
        extra_numeric.append("metformin_renal_contra")

    # 5. Demographics & Continuous Age
    def parse_age(age_str):
        if age_str is None or not isinstance(age_str, str):
            return np.nan
        try:
            parts = age_str.replace("[", "").replace(")", "").replace("]", "").split("-")
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            pass
        return np.nan

    if "age" in train:
        train["age_numeric"] = np.array([parse_age(x) for x in train["age"]])
        eval_cols["age_numeric"] = np.array([parse_age(x) for x in eval_cols["age"]])
        train["is_elderly"] = (train["age_numeric"] >= 75.0).astype(float)
        eval_cols["is_elderly"] = (eval_cols["age_numeric"] >= 75.0).astype(float)
        extra_numeric.extend(["age_numeric", "is_elderly"])

    for col in ["weight", "medical_specialty", "payer_code"]:
        if col in train:
            col_name = f"has_{col}"
            train[col_name] = np.array([float(x is not None and x != "?") for x in train[col]])
            eval_cols[col_name] = np.array([float(x is not None and x != "?") for x in eval_cols[col]])
            extra_numeric.append(col_name)

    # 6. Utilization Velocity & Care Intensity
    num_cols = ["number_outpatient", "number_emergency", "number_inpatient", "num_lab_procedures", "num_procedures", "num_medications", "time_in_hospital"]
    if all(col in train for col in num_cols):
        for d in [train, eval_cols]:
            d["total_visits"] = d["number_outpatient"] + d["number_emergency"] + d["number_inpatient"]
            d["has_inpatient"] = (d["number_inpatient"] > 0).astype(float)
            d["has_emergency"] = (d["number_emergency"] > 0).astype(float)
            d["recidivism_velocity"] = 3.0 * d["number_inpatient"] + 1.5 * d["number_emergency"] + 0.5 * d["number_outpatient"]
            d["inpatient_ratio"] = d["number_inpatient"] / (d["total_visits"] + 1e-5)
            d["emergency_ratio"] = d["number_emergency"] / (d["total_visits"] + 1e-5)
            d["severity_idx"] = d["num_lab_procedures"] + d["num_procedures"] + d["num_medications"]
            time_h = d["time_in_hospital"] + 1e-5
            d["procedures_per_day"] = d["num_procedures"] / time_h
            d["meds_per_day"] = d["num_medications"] / time_h
            d["labs_per_day"] = d["num_lab_procedures"] / time_h
            d["inpatient_per_stay"] = d["number_inpatient"] / (d["time_in_hospital"] + 1.0)

        extra_numeric.extend([
            "total_visits", "has_inpatient", "has_emergency", "recidivism_velocity",
            "inpatient_ratio", "emergency_ratio", "severity_idx",
            "procedures_per_day", "meds_per_day", "labs_per_day", "inpatient_per_stay",
        ])

    # 7. Focused Target Encodings of Key High-Cardinality Categoricals
    te_cols = ["diag_1_3d", "discharge_disposition_id", "admission_source_id", "medical_specialty"]
    for col in te_cols:
        if col in train:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42)
            te_name = f"{col}_te"
            train[te_name] = tr_enc
            eval_cols[te_name] = ev_enc
            extra_numeric.append(te_name)

    # 8. Encode with TabularEncoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_cols)
    cat_feats = encoder.categorical_indices

    # 9. Clean Diverse LightGBM Ensemble (GBDT + DART + Deep)
    model1 = lgb.LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=240,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        reg_alpha=0.3,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    model2 = lgb.LGBMClassifier(
        boosting_type="dart",
        n_estimators=200,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        drop_rate=0.1,
        max_drop=30,
        skip_drop=0.5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        cat_smooth=50.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )

    model3 = lgb.LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.01,
        num_leaves=50,
        min_child_samples=60,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=3.0,
        reg_alpha=1.0,
        cat_smooth=60.0,
        random_state=456,
        n_jobs=1,
        verbose=-1,
    )

    model1.fit(x_train, y_train, categorical_feature=cat_feats)
    model2.fit(x_train, y_train, categorical_feature=cat_feats)
    model3.fit(x_train, y_train, categorical_feature=cat_feats)

    p1 = model1.predict_proba(x_eval)[:, 1]
    p2 = model2.predict_proba(x_eval)[:, 1]
    p3 = model3.predict_proba(x_eval)[:, 1]

    weights = [0.40, 0.35, 0.25]
    p_raw = weights[0] * p1 + weights[1] * p2 + weights[2] * p3

    r1 = rankdata(p1)
    r2 = rankdata(p2)
    r3 = rankdata(p3)
    r_blend = weights[0] * r1 + weights[1] * r2 + weights[2] * r3

    order = np.argsort(np.argsort(r_blend))
    p_final = np.sort(p_raw)[order]

    return p_final


# EVOLVE-BLOCK-END
# =====================================================================
