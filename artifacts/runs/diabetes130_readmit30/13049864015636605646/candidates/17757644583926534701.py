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
    # 1. Comorbidities across all diagnoses
    cm_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

    tr_cols = dict(train)
    ev_cols = dict(eval_cols)
    tr_cols.update(cm_tr)
    ev_cols.update(cm_ev)
    extra_num = list(cm_tr.keys())

    # Weighted Charlson Comorbidity score and cardiorenal interaction
    tr_cols["charlson_score"] = (
        cm_tr["cm_mi"] + cm_tr["cm_chf"] + cm_tr["cm_pvd"] + cm_tr["cm_cerebrovascular"] +
        cm_tr["cm_dementia"] + cm_tr["cm_pulmonary"] + cm_tr["cm_rheumatic"] +
        cm_tr["cm_peptic_ulcer"] + cm_tr["cm_liver"] + 2.0 * cm_tr["cm_hemiplegia"] +
        2.0 * cm_tr["cm_renal"] + 2.0 * cm_tr["cm_cancer"] +
        2.0 * cm_tr["cm_diabetes_complicated"] + 4.0 * cm_tr["cm_metastatic"] + 6.0 * cm_tr["cm_hiv"]
    )
    ev_cols["charlson_score"] = (
        cm_ev["cm_mi"] + cm_ev["cm_chf"] + cm_ev["cm_pvd"] + cm_ev["cm_cerebrovascular"] +
        cm_ev["cm_dementia"] + cm_ev["cm_pulmonary"] + cm_ev["cm_rheumatic"] +
        cm_ev["cm_peptic_ulcer"] + cm_ev["cm_liver"] + 2.0 * cm_ev["cm_hemiplegia"] +
        2.0 * cm_ev["cm_renal"] + 2.0 * cm_ev["cm_cancer"] +
        2.0 * cm_ev["cm_diabetes_complicated"] + 4.0 * cm_ev["cm_metastatic"] + 6.0 * cm_ev["cm_hiv"]
    )
    tr_cols["cardiorenal"] = cm_tr["cm_chf"] * cm_tr["cm_renal"]
    ev_cols["cardiorenal"] = cm_ev["cm_chf"] * cm_ev["cm_renal"]
    extra_num.extend(["charlson_score", "cardiorenal"])

    # ICD-9 chapter groups
    tr_grp1 = np.array([icd9_group(c) for c in train["diag_1"]], dtype=object)
    ev_grp1 = np.array([icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
    tr_grp2 = np.array([icd9_group(c) for c in train["diag_2"]], dtype=object)
    ev_grp2 = np.array([icd9_group(c) for c in eval_cols["diag_2"]], dtype=object)
    tr_grp3 = np.array([icd9_group(c) for c in train["diag_3"]], dtype=object)
    ev_grp3 = np.array([icd9_group(c) for c in eval_cols["diag_3"]], dtype=object)

    tr_cols["diag_1_group"] = tr_grp1
    ev_cols["diag_1_group"] = ev_grp1
    tr_cols["diag_2_group"] = tr_grp2
    ev_cols["diag_2_group"] = ev_grp2
    tr_cols["diag_3_group"] = tr_grp3
    ev_cols["diag_3_group"] = ev_grp3
    extra_cat = ["diag_1_group", "diag_2_group", "diag_3_group"]

    for grp in ["circulatory", "diabetes", "respiratory", "genitourinary"]:
        tr_cols[f"num_{grp}_diag"] = (
            (tr_grp1 == grp).astype(float) + (tr_grp2 == grp).astype(float) + (tr_grp3 == grp).astype(float)
        )
        ev_cols[f"num_{grp}_diag"] = (
            (ev_grp1 == grp).astype(float) + (ev_grp2 == grp).astype(float) + (ev_grp3 == grp).astype(float)
        )
        extra_num.append(f"num_{grp}_diag")

    # 2. Target encode high-cardinality categorical features
    te_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "payer_code",
        "admission_type_id",
    ]
    for col in te_cols:
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0)
        te_name = f"{col}_te"
        tr_cols[te_name] = tr_te
        ev_cols[te_name] = ev_te
        extra_num.append(te_name)

    # Additional high-order target encodings: ICD-9 chapter and admission-discharge transition
    tr_grp_te, ev_grp_te = oof_target_encode(tr_grp1, y_train, ev_grp1, smoothing=30.0, seed=42)
    tr_cols["diag_1_group_te"] = tr_grp_te
    ev_cols["diag_1_group_te"] = ev_grp_te
    extra_num.append("diag_1_group_te")

    adm_disch_tr = np.array([f"{s}_{d}" for s, d in zip(train["admission_source_id"], train["discharge_disposition_id"])], dtype=object)
    adm_disch_ev = np.array([f"{s}_{d}" for s, d in zip(eval_cols["admission_source_id"], eval_cols["discharge_disposition_id"])], dtype=object)
    tr_ad_te, ev_ad_te = oof_target_encode(adm_disch_tr, y_train, adm_disch_ev, smoothing=25.0, seed=42)
    tr_cols["adm_disch_combo_te"] = tr_ad_te
    ev_cols["adm_disch_combo_te"] = ev_ad_te
    extra_num.append("adm_disch_combo_te")

    # 3. Utilization intensity
    inpatient_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inpatient_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emergency_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emergency_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    outpatient_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    outpatient_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    total_prior_tr = inpatient_tr + emergency_tr + outpatient_tr
    total_prior_ev = inpatient_ev + emergency_ev + outpatient_ev

    tr_cols["total_prior_visits"] = total_prior_tr
    ev_cols["total_prior_visits"] = total_prior_ev
    tr_cols["inpatient_ratio"] = inpatient_tr / (total_prior_tr + 1.0)
    ev_cols["inpatient_ratio"] = inpatient_ev / (total_prior_ev + 1.0)
    tr_cols["emergency_ratio"] = emergency_tr / (total_prior_tr + 1.0)
    ev_cols["emergency_ratio"] = emergency_ev / (total_prior_ev + 1.0)
    tr_cols["has_prior_inpatient"] = (inpatient_tr > 0).astype(np.float64)
    ev_cols["has_prior_inpatient"] = (inpatient_ev > 0).astype(np.float64)
    tr_cols["has_mult_inpatient"] = (inpatient_tr >= 2).astype(np.float64)
    ev_cols["has_mult_inpatient"] = (inpatient_ev >= 2).astype(np.float64)
    tr_cols["util_severity"] = 3.0 * inpatient_tr + 1.5 * emergency_tr + 0.5 * outpatient_tr
    ev_cols["util_severity"] = 3.0 * inpatient_ev + 1.5 * emergency_ev + 0.5 * outpatient_ev
    extra_num.extend([
        "total_prior_visits", "inpatient_ratio", "emergency_ratio",
        "has_prior_inpatient", "has_mult_inpatient", "util_severity"
    ])

    # Demographics & Clinical Indicators
    age_map = {"[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0, "[40-50)": 45.0,
               "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0, "[80-90)": 85.0, "[90-100)": 95.0}
    tr_cols["age_midpoint"] = np.array([age_map.get(str(x), 65.0) for x in train["age"]], dtype=np.float64)
    ev_cols["age_midpoint"] = np.array([age_map.get(str(x), 65.0) for x in eval_cols["age"]], dtype=np.float64)
    tr_cols["is_elderly"] = (tr_cols["age_midpoint"] >= 70.0).astype(np.float64)
    ev_cols["is_elderly"] = (ev_cols["age_midpoint"] >= 70.0).astype(np.float64)

    disch_tr = train["discharge_disposition_id"]
    disch_ev = eval_cols["discharge_disposition_id"]
    tr_cols["disch_home"] = (disch_tr == "1").astype(np.float64)
    ev_cols["disch_home"] = (disch_ev == "1").astype(np.float64)
    tr_cols["disch_snf"] = (disch_tr == "3").astype(np.float64)
    ev_cols["disch_snf"] = (disch_ev == "3").astype(np.float64)
    tr_cols["disch_homehealth"] = (disch_tr == "6").astype(np.float64)
    ev_cols["disch_homehealth"] = (disch_ev == "6").astype(np.float64)
    tr_cols["disch_not_home"] = (disch_tr != "1").astype(np.float64)
    ev_cols["disch_not_home"] = (disch_ev != "1").astype(np.float64)

    adm_src_tr = train["admission_source_id"]
    adm_src_ev = eval_cols["admission_source_id"]
    tr_cols["adm_er"] = (adm_src_tr == "7").astype(np.float64)
    ev_cols["adm_er"] = (adm_src_ev == "7").astype(np.float64)
    tr_cols["er_to_facility"] = tr_cols["adm_er"] * tr_cols["disch_not_home"]
    ev_cols["er_to_facility"] = ev_cols["adm_er"] * ev_cols["disch_not_home"]

    los_tr = np.maximum(np.nan_to_num(train["time_in_hospital"], nan=1.0), 1.0)
    los_ev = np.maximum(np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0), 1.0)
    tr_cols["labs_per_day"] = np.nan_to_num(train["num_lab_procedures"], nan=0.0) / los_tr
    ev_cols["labs_per_day"] = np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0) / los_ev
    tr_cols["meds_per_day"] = np.nan_to_num(train["num_medications"], nan=0.0) / los_tr
    ev_cols["meds_per_day"] = np.nan_to_num(eval_cols["num_medications"], nan=0.0) / los_ev
    tr_cols["procs_per_day"] = np.nan_to_num(train["num_procedures"], nan=0.0) / los_tr
    ev_cols["procs_per_day"] = np.nan_to_num(eval_cols["num_procedures"], nan=0.0) / los_ev
    tr_cols["services_per_day"] = tr_cols["labs_per_day"] + tr_cols["meds_per_day"] + tr_cols["procs_per_day"]
    ev_cols["services_per_day"] = ev_cols["labs_per_day"] + ev_cols["meds_per_day"] + ev_cols["procs_per_day"]

    tr_cols["a1c_tested"] = (train["A1Cresult"] != "None").astype(np.float64)
    ev_cols["a1c_tested"] = (eval_cols["A1Cresult"] != "None").astype(np.float64)
    tr_cols["a1c_abnormal"] = np.isin(train["A1Cresult"], [">7", ">8"]).astype(np.float64)
    ev_cols["a1c_abnormal"] = np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(np.float64)

    med_list = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    valid_meds = [m for m in med_list if m in train]
    n_tr, n_ev = len(y_train), len(los_ev)
    act_tr, act_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    up_tr, up_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    dn_tr, dn_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    for m in valid_meds:
        act_tr += (train[m] != "No").astype(np.float64)
        act_ev += (eval_cols[m] != "No").astype(np.float64)
        up_tr += (train[m] == "Up").astype(np.float64)
        up_ev += (eval_cols[m] == "Up").astype(np.float64)
        dn_tr += (train[m] == "Down").astype(np.float64)
        dn_ev += (eval_cols[m] == "Down").astype(np.float64)

    tr_cols["num_active_meds"] = act_tr
    ev_cols["num_active_meds"] = act_ev
    tr_cols["num_meds_up"] = up_tr
    ev_cols["num_meds_up"] = up_ev
    tr_cols["num_meds_down"] = dn_tr
    ev_cols["num_meds_down"] = dn_ev
    tr_cols["num_meds_changed"] = up_tr + dn_tr
    ev_cols["num_meds_changed"] = up_ev + dn_ev

    ins_tr = train["insulin"]
    ins_ev = eval_cols["insulin"]
    tr_cols["insulin_active"] = (ins_tr != "No").astype(np.float64)
    ev_cols["insulin_active"] = (ins_ev != "No").astype(np.float64)
    tr_cols["insulin_changed"] = np.isin(ins_tr, ["Up", "Down"]).astype(np.float64)
    ev_cols["insulin_changed"] = np.isin(ins_ev, ["Up", "Down"]).astype(np.float64)
    tr_cols["has_med_change"] = (train["change"] == "Ch").astype(np.float64)
    ev_cols["has_med_change"] = (eval_cols["change"] == "Ch").astype(np.float64)

    extra_num.extend([
        "age_midpoint", "is_elderly", "disch_home", "disch_snf", "disch_homehealth",
        "disch_not_home", "adm_er", "er_to_facility", "labs_per_day", "meds_per_day",
        "procs_per_day", "services_per_day", "a1c_tested", "a1c_abnormal",
        "num_active_meds", "num_meds_up", "num_meds_down", "num_meds_changed",
        "insulin_active", "insulin_changed", "has_med_change"
    ])

    # 4. Tabular encoding with min_count=20 to preserve granular clinical codes
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, tr_cols, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_train = encoder.transform(tr_cols)
    x_eval = encoder.transform(ev_cols)

    # 5. Blend two complementary LightGBM models (deep vs shallow regularized)
    model1 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.0,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred1 = model1.predict_proba(x_eval)[:, 1]

    model2 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.020,
        num_leaves=16,
        min_child_samples=200,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=2.0,
        cat_smooth=50.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred2 = model2.predict_proba(x_eval)[:, 1]

    model3 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.028,
        num_leaves=24,
        max_depth=5,
        min_child_samples=180,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.40,
        reg_lambda=3.0,
        reg_alpha=0.5,
        cat_smooth=40.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred3 = model3.predict_proba(x_eval)[:, 1]

    return 0.40 * pred1 + 0.35 * pred2 + 0.25 * pred3


# EVOLVE-BLOCK-END
# =====================================================================
