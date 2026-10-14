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
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])

    # 1. Comorbidities via Charlson mapping
    cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k, v in cm_train.items():
        train_dict[k] = v.astype(np.float64)
        eval_dict[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    if "cm_chf" in cm_train and "cm_renal" in cm_train:
        train_dict["cm_cardiorenal"] = (cm_train["cm_chf"] * cm_train["cm_renal"]).astype(np.float64)
        eval_dict["cm_cardiorenal"] = (cm_eval["cm_chf"] * cm_eval["cm_renal"]).astype(np.float64)
        extra_numeric.append("cm_cardiorenal")

    # 2. ICD-9 Chapter groupings
    d1_tr = np.array([icd9_group(x) for x in train["diag_1"]], dtype=object)
    d1_ev = np.array([icd9_group(x) for x in eval_cols["diag_1"]], dtype=object)
    d2_tr = np.array([icd9_group(x) for x in train["diag_2"]], dtype=object)
    d2_ev = np.array([icd9_group(x) for x in eval_cols["diag_2"]], dtype=object)
    d3_tr = np.array([icd9_group(x) for x in train["diag_3"]], dtype=object)
    d3_ev = np.array([icd9_group(x) for x in eval_cols["diag_3"]], dtype=object)

    train_dict["diag_1_group"] = d1_tr
    eval_dict["diag_1_group"] = d1_ev
    train_dict["diag_2_group"] = d2_tr
    eval_dict["diag_2_group"] = d2_ev
    train_dict["diag_3_group"] = d3_tr
    eval_dict["diag_3_group"] = d3_ev
    extra_categorical.extend(["diag_1_group", "diag_2_group", "diag_3_group"])

    d12_tr = np.array([f"{g1}_{g2}" for g1, g2 in zip(d1_tr, d2_tr)], dtype=object)
    d12_ev = np.array([f"{g1}_{g2}" for g1, g2 in zip(d1_ev, d2_ev)], dtype=object)
    train_dict["diag_1_2_group"] = d12_tr
    eval_dict["diag_1_2_group"] = d12_ev
    extra_categorical.append("diag_1_2_group")

    # 3. Smoothed Out-Of-Fold Target Encoding
    te_target_cols = [
        "diag_1", "diag_2", "diag_3",
        "discharge_disposition_id", "admission_source_id", "admission_type_id",
        "medical_specialty", "payer_code", "diag_1_group", "diag_1_2_group",
    ]
    for col in te_target_cols:
        if col in train_dict and col in eval_dict:
            tr_enc, ev_enc = oof_target_encode(
                train_dict[col], y_train, eval_dict[col], n_folds=5, smoothing=20.0, seed=42
            )
            te_name = f"te_{col}"
            train_dict[te_name] = tr_enc.astype(np.float64)
            eval_dict[te_name] = ev_enc.astype(np.float64)
            extra_numeric.append(te_name)

    # 4. Utilization & care intensity metrics
    def compute_utilization(d):
        los = np.maximum(d["time_in_hospital"].astype(np.float64), 1.0)
        inpatient = d["number_inpatient"].astype(np.float64)
        emergency = d["number_emergency"].astype(np.float64)
        outpatient = d["number_outpatient"].astype(np.float64)
        total_visits = inpatient + emergency + outpatient

        labs = d["num_lab_procedures"].astype(np.float64)
        procs = d["num_procedures"].astype(np.float64)
        meds = d["num_medications"].astype(np.float64)
        diags = d["number_diagnoses"].astype(np.float64)

        return {
            "total_prior_visits": total_visits,
            "has_inpatient": (inpatient > 0.0).astype(np.float64),
            "has_emergency": (emergency > 0.0).astype(np.float64),
            "inpatient_ratio": inpatient / (total_visits + 1.0),
            "emergency_ratio": emergency / (total_visits + 1.0),
            "log_inpatient": np.log1p(inpatient),
            "log_emergency": np.log1p(emergency),
            "log_total_visits": np.log1p(total_visits),
            "labs_per_day": labs / los,
            "meds_per_day": meds / los,
            "procs_per_day": procs / los,
            "diags_per_day": diags / los,
            "clinical_intensity": (labs + procs + meds) / los,
            "med_proc_interaction": np.log1p(meds) * np.log1p(procs),
            "diag_los_interaction": diags * los,
            "diag_to_med_ratio": diags / (meds + 1.0),
        }

    util_tr = compute_utilization(train)
    util_ev = compute_utilization(eval_cols)
    for k, v in util_tr.items():
        train_dict[k] = v
        eval_dict[k] = util_ev[k]
        extra_numeric.append(k)

    # 5. Admission and discharge disposition clinical flags
    def compute_adm_disch(d):
        disch = d.get("discharge_disposition_id")
        adm_s = d.get("admission_source_id")
        adm_t = d.get("admission_type_id")
        res = {}
        if disch is not None:
            res["is_disch_home"] = (disch == "1").astype(np.float64)
            res["is_disch_facility"] = np.isin(disch, ["3", "6", "22", "28"]).astype(np.float64)
        if adm_s is not None:
            res["is_adm_er"] = (adm_s == "7").astype(np.float64)
        if adm_t is not None:
            res["is_adm_elective"] = (adm_t == "3").astype(np.float64)
        return res

    adm_tr = compute_adm_disch(train)
    adm_ev = compute_adm_disch(eval_cols)
    for k, v in adm_tr.items():
        train_dict[k] = v
        eval_dict[k] = adm_ev[k]
        extra_numeric.append(k)

    # 6. Age midpoint numeric feature
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    if "age" in train:
        train_dict["age_numeric"] = np.array([age_map.get(str(x), np.nan) for x in train["age"]], dtype=np.float64)
        eval_dict["age_numeric"] = np.array([age_map.get(str(x), np.nan) for x in eval_cols["age"]], dtype=np.float64)
        extra_numeric.append("age_numeric")

    # 7. Medication regimen complexity & changes
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    avail_meds = [m for m in med_names if m in train]

    def compute_meds(d, n_rows):
        act = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        dn_cnt = np.zeros(n_rows, dtype=np.float64)
        std_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in avail_meds:
            col = d[m]
            u = (col == "Up").astype(np.float64)
            dn = (col == "Down").astype(np.float64)
            s = (col == "Steady").astype(np.float64)
            up_cnt += u
            dn_cnt += dn
            std_cnt += s
            act += (u + dn + s)

        res = {
            "num_meds_active": act,
            "num_meds_up": up_cnt,
            "num_meds_down": dn_cnt,
            "num_meds_steady": std_cnt,
            "num_meds_changed": up_cnt + dn_cnt,
        }

        if "insulin" in d:
            ins = d["insulin"]
            res["insulin_active"] = ((ins == "Up") | (ins == "Down") | (ins == "Steady")).astype(np.float64)
            res["insulin_up"] = (ins == "Up").astype(np.float64)
            res["insulin_down"] = (ins == "Down").astype(np.float64)
            res["insulin_steady"] = (ins == "Steady").astype(np.float64)
            res["insulin_changed"] = ((ins == "Up") | (ins == "Down")).astype(np.float64)
        return res

    meds_tr = compute_meds(train, n_tr)
    meds_ev = compute_meds(eval_cols, n_ev)
    for k, v in meds_tr.items():
        train_dict[k] = v
        eval_dict[k] = meds_ev[k]
        extra_numeric.append(k)

    # 8. Laboratory testing flags
    def compute_labs(d):
        res = {}
        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            res["a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
            res["a1c_high"] = ((a1c == ">8") | (a1c == ">7")).astype(np.float64)
        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            res["glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
            res["glu_high"] = ((glu == ">300") | (glu == ">200")).astype(np.float64)
        return res

    labs_tr = compute_labs(train)
    labs_ev = compute_labs(eval_cols)
    for k, v in labs_tr.items():
        train_dict[k] = v
        eval_dict[k] = labs_ev[k]
        extra_numeric.append(k)

    # Encode tabular data
    encoder = TabularEncoder(min_count=25)
    encoder.fit(
        task,
        train_dict,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # 9. Diverse LightGBM ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.1,
            cat_smooth=65.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.018,
            num_leaves=48,
            max_depth=7,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=1.5,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.025,
            num_leaves=16,
            max_depth=5,
            min_child_samples=220,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=0.05,
            reg_alpha=0.05,
            cat_smooth=70.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=230,
            learning_rate=0.02,
            num_leaves=35,
            max_depth=6,
            min_child_samples=180,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.45,
            reg_lambda=0.5,
            cat_smooth=60.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.022,
            num_leaves=25,
            min_child_samples=120,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=0.2,
            cat_smooth=55.0,
            random_state=101112,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(n_ev, dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
