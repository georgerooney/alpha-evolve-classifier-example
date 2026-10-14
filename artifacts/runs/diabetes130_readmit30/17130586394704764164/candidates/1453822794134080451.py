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
    try:
        from clinical_toolkit import (
            TabularEncoder,
            icd9_comorbidities,
            icd9_group,
            oof_target_encode,
        )
    except ImportError:
        pass
    from sklearn.decomposition import TruncatedSVD
    import lightgbm as lgb
    import numpy as np

    tr_data = dict(train)
    ev_data = dict(eval_cols)
    extra_num = []
    extra_cat = []

    any_col = next(iter(train.keys()))
    n_train_rows = len(train[any_col])
    any_col_ev = next(iter(eval_cols.keys()))
    n_eval_rows = len(eval_cols[any_col_ev])

    # 1. Low-rank clinical comorbidity factors and SVD latent axes
    cm_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in sorted(cm_tr.keys()):
        tr_data[k] = cm_tr[k]
        ev_data[k] = cm_ev[k]
        extra_num.append(k)

    cm_matrix_tr = np.column_stack([cm_tr[k] for k in sorted(cm_tr.keys()) if k != "cm_count"])
    cm_matrix_ev = np.column_stack([cm_ev[k] for k in sorted(cm_ev.keys()) if k != "cm_count"])
    svd = TruncatedSVD(n_components=3, random_state=42)
    svd_tr = svd.fit_transform(cm_matrix_tr)
    svd_ev = svd.transform(cm_matrix_ev)
    for i in range(3):
        col_name = f"cm_svd_{i}"
        tr_data[col_name] = svd_tr[:, i]
        ev_data[col_name] = svd_ev[:, i]
        extra_num.append(col_name)

    # Standard Weighted Charlson Comorbidity Index
    charlson_weights = {
        "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
        "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
        "cm_liver": 1.0, "cm_diabetes_complicated": 2.0, "cm_hemiplegia": 2.0,
        "cm_renal": 2.0, "cm_cancer": 2.0, "cm_metastatic": 6.0, "cm_hiv": 6.0
    }
    charlson_tr = np.zeros(n_train_rows, dtype=np.float64)
    charlson_ev = np.zeros(n_eval_rows, dtype=np.float64)
    for k, w in charlson_weights.items():
        if k in cm_tr:
            charlson_tr += w * cm_tr[k]
            charlson_ev += w * cm_ev[k]
    tr_data["charlson_score"] = charlson_tr
    ev_data["charlson_score"] = charlson_ev
    extra_num.append("charlson_score")

    # High-risk clinical syndrome interactions
    # Cardiorenal syndrome (heart failure + renal disease)
    tr_data["cardiorenal"] = ((cm_tr["cm_chf"] > 0) & (cm_tr["cm_renal"] > 0)).astype(np.float64)
    ev_data["cardiorenal"] = ((cm_ev["cm_chf"] > 0) & (cm_ev["cm_renal"] > 0)).astype(np.float64)
    extra_num.append("cardiorenal")

    # 2. ICD-9 3-Digit Category Hierarchy & Diagnosis Groups
    def extract_icd9_cat(arr):
        return np.array(
            ["missing" if x is None or x == "None" or x == "?" else str(x).split(".")[0].strip() for x in arr],
            dtype=object
        )

    tr_diag1_cat = extract_icd9_cat(train["diag_1"])
    ev_diag1_cat = extract_icd9_cat(eval_cols["diag_1"])
    tr_diag2_cat = extract_icd9_cat(train["diag_2"])
    ev_diag2_cat = extract_icd9_cat(eval_cols["diag_2"])
    tr_diag3_cat = extract_icd9_cat(train["diag_3"])
    ev_diag3_cat = extract_icd9_cat(eval_cols["diag_3"])

    tr_data["diag_1_cat"] = tr_diag1_cat
    ev_data["diag_1_cat"] = ev_diag1_cat
    tr_data["diag_2_cat"] = tr_diag2_cat
    ev_data["diag_2_cat"] = ev_diag2_cat
    tr_data["diag_3_cat"] = tr_diag3_cat
    ev_data["diag_3_cat"] = ev_diag3_cat
    extra_cat.extend(["diag_1_cat", "diag_2_cat", "diag_3_cat"])

    for i in (1, 2, 3):
        col = f"diag_{i}"
        tr_data[f"diag_group_{i}"] = np.array([icd9_group(x) for x in train[col]])
        ev_data[f"diag_group_{i}"] = np.array([icd9_group(x) for x in eval_cols[col]])
        extra_cat.append(f"diag_group_{i}")

    # Specific conditions: Fluid/Electrolyte disorders (276.x)
    def check_prefix_in_diags(d1, d2, d3, prefixes):
        n = len(d1)
        res = np.zeros(n, dtype=np.float64)
        pref_tuple = tuple(prefixes)
        for arr in (d1, d2, d3):
            has_p = np.array(
                [1.0 if (x is not None and x != "None" and x != "?") and str(x).startswith(pref_tuple) else 0.0 for x in arr],
                dtype=np.float64
            )
            res = np.maximum(res, has_p)
        return res

    tr_data["has_electrolyte_disorder"] = check_prefix_in_diags(train["diag_1"], train["diag_2"], train["diag_3"], ["276"])
    ev_data["has_electrolyte_disorder"] = check_prefix_in_diags(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"], ["276"])
    extra_num.append("has_electrolyte_disorder")

    # 3. Target Encodings (individual and joint projections)
    te_cols = [
        "diag_1", "diag_2", "diag_3",
        "discharge_disposition_id", "admission_source_id", "admission_type_id",
        "medical_specialty", "payer_code"
    ]
    for col in te_cols:
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
        te_name = f"{col}_te"
        tr_data[te_name] = tr_te
        ev_data[te_name] = ev_te
        extra_num.append(te_name)

    for cat_name, tr_arr, ev_arr in [
        ("diag_1_cat", tr_diag1_cat, ev_diag1_cat),
        ("diag_2_cat", tr_diag2_cat, ev_diag2_cat),
        ("diag_3_cat", tr_diag3_cat, ev_diag3_cat)
    ]:
        tr_te, ev_te = oof_target_encode(tr_arr, y_train, ev_arr, smoothing=20.0, seed=42)
        te_col = f"{cat_name}_te"
        tr_data[te_col] = tr_te
        ev_data[te_col] = ev_te
        extra_num.append(te_col)

    # Joint interactions
    tr_joint = np.array([f"{a}_{b}" for a, b in zip(train["admission_source_id"], train["discharge_disposition_id"])])
    ev_joint = np.array([f"{a}_{b}" for a, b in zip(eval_cols["admission_source_id"], eval_cols["discharge_disposition_id"])])
    tr_te_joint, ev_te_joint = oof_target_encode(tr_joint, y_train, ev_joint, smoothing=20.0, seed=42)
    tr_data["adm_dis_joint_te"] = tr_te_joint
    ev_data["adm_dis_joint_te"] = ev_te_joint
    extra_num.append("adm_dis_joint_te")

    tr_spec_dis = np.array([f"{a}_{b}" for a, b in zip(train["medical_specialty"], train["discharge_disposition_id"])])
    ev_spec_dis = np.array([f"{a}_{b}" for a, b in zip(eval_cols["medical_specialty"], eval_cols["discharge_disposition_id"])])
    tr_te_spec_dis, ev_te_spec_dis = oof_target_encode(tr_spec_dis, y_train, ev_spec_dis, smoothing=20.0, seed=42)
    tr_data["spec_dis_joint_te"] = tr_te_spec_dis
    ev_data["spec_dis_joint_te"] = ev_te_spec_dis
    extra_num.append("spec_dis_joint_te")

    # Prior utilization & joint with discharge
    inp_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    em_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    em_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    tot_tr = inp_tr + np.nan_to_num(train["number_outpatient"], nan=0.0) + em_tr
    tot_ev = inp_ev + np.nan_to_num(eval_cols["number_outpatient"], nan=0.0) + em_ev

    tr_data["total_prior"] = tot_tr
    ev_data["total_prior"] = tot_ev
    tr_data["has_prior_inpatient"] = (inp_tr > 0).astype(np.float64)
    ev_data["has_prior_inpatient"] = (inp_ev > 0).astype(np.float64)
    tr_data["frequent_inpatient"] = (inp_tr >= 2).astype(np.float64)
    ev_data["frequent_inpatient"] = (inp_ev >= 2).astype(np.float64)
    tr_data["inpatient_ratio"] = inp_tr / (tot_tr + 1.0)
    ev_data["inpatient_ratio"] = inp_ev / (tot_ev + 1.0)
    tr_data["has_prior_emergency"] = (em_tr > 0).astype(np.float64)
    ev_data["has_prior_emergency"] = (em_ev > 0).astype(np.float64)
    tr_data["emergency_ratio"] = em_tr / (tot_tr + 1.0)
    ev_data["emergency_ratio"] = em_ev / (tot_ev + 1.0)
    tr_data["log_inpatient"] = np.log1p(inp_tr)
    ev_data["log_inpatient"] = np.log1p(inp_ev)
    tr_data["log_emergency"] = np.log1p(em_tr)
    ev_data["log_emergency"] = np.log1p(em_ev)
    tr_data["log_total_prior"] = np.log1p(tot_tr)
    ev_data["log_total_prior"] = np.log1p(tot_ev)
    extra_num.extend([
        "total_prior", "has_prior_inpatient", "frequent_inpatient", "inpatient_ratio",
        "has_prior_emergency", "emergency_ratio", "log_inpatient", "log_emergency", "log_total_prior"
    ])

    # Discharge disposition conditional on prior inpatient history
    dis_tr = train["discharge_disposition_id"]
    dis_ev = eval_cols["discharge_disposition_id"]
    tr_dis_inp = np.array([f"{d}_{'inp' if i > 0 else 'no'}" for d, i in zip(dis_tr, inp_tr)])
    ev_dis_inp = np.array([f"{d}_{'inp' if i > 0 else 'no'}" for d, i in zip(dis_ev, inp_ev)])
    tr_te_dis_inp, ev_te_dis_inp = oof_target_encode(tr_dis_inp, y_train, ev_dis_inp, smoothing=20.0, seed=42)
    tr_data["dis_inp_joint_te"] = tr_te_dis_inp
    ev_data["dis_inp_joint_te"] = ev_te_dis_inp
    extra_num.append("dis_inp_joint_te")

    # 4. Care Intensity & Compounding Interactions
    time_tr = np.nan_to_num(train["time_in_hospital"], nan=1.0)
    time_ev = np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0)
    time_tr = np.where(time_tr <= 0, 1.0, time_tr)
    time_ev = np.where(time_ev <= 0, 1.0, time_ev)

    tr_data["lab_per_day"] = np.nan_to_num(train["num_lab_procedures"], nan=0.0) / time_tr
    ev_data["lab_per_day"] = np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0) / time_ev
    tr_data["proc_per_day"] = np.nan_to_num(train["num_procedures"], nan=0.0) / time_tr
    ev_data["proc_per_day"] = np.nan_to_num(eval_cols["num_procedures"], nan=0.0) / time_ev
    tr_data["med_per_day"] = np.nan_to_num(train["num_medications"], nan=0.0) / time_tr
    ev_data["med_per_day"] = np.nan_to_num(eval_cols["num_medications"], nan=0.0) / time_ev
    extra_num.extend(["lab_per_day", "proc_per_day", "med_per_day"])

    # High-leverage risk compounding interactions
    tr_data["inp_x_charlson"] = inp_tr * charlson_tr
    ev_data["inp_x_charlson"] = inp_ev * charlson_ev
    tr_data["inp_x_time"] = inp_tr * time_tr
    ev_data["inp_x_time"] = inp_ev * time_ev
    extra_num.extend(["inp_x_charlson", "inp_x_time"])

    # 5. Demographics and Medical Context
    def parse_age(age_arr):
        res = np.full(age_arr.shape, np.nan, dtype=np.float64)
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        for k, v in age_map.items():
            res[age_arr == k] = v
        return res

    tr_age = parse_age(train["age"])
    ev_age = parse_age(eval_cols["age"])
    tr_data["age_midpoint"] = tr_age
    ev_data["age_midpoint"] = ev_age
    tr_data["is_elderly"] = (tr_age >= 70.0).astype(np.float64)
    ev_data["is_elderly"] = (ev_age >= 70.0).astype(np.float64)
    extra_num.extend(["age_midpoint", "is_elderly"])

    tr_data["weight_present"] = (~np.isin(train["weight"], [None, "None"])).astype(np.float64)
    ev_data["weight_present"] = (~np.isin(eval_cols["weight"], [None, "None"])).astype(np.float64)
    extra_num.append("weight_present")

    # Discharge destinations
    tr_data["discharged_home"] = (dis_tr == "1").astype(np.float64)
    ev_data["discharged_home"] = (dis_ev == "1").astype(np.float64)
    tr_data["discharged_snf_icf"] = np.isin(dis_tr, ["3", "4", "5"]).astype(np.float64)
    ev_data["discharged_snf_icf"] = np.isin(dis_ev, ["3", "4", "5"]).astype(np.float64)
    tr_data["discharged_ama"] = (dis_tr == "7").astype(np.float64)  # Against medical advice
    ev_data["discharged_ama"] = (dis_ev == "7").astype(np.float64)
    extra_num.extend(["discharged_home", "discharged_snf_icf", "discharged_ama"])

    # Payer flags
    tr_payer = train["payer_code"]
    ev_payer = eval_cols["payer_code"]
    tr_data["is_medicare"] = (tr_payer == "MC").astype(np.float64)
    ev_data["is_medicare"] = (ev_payer == "MC").astype(np.float64)
    tr_data["is_medicaid"] = (tr_payer == "MD").astype(np.float64)
    ev_data["is_medicaid"] = (ev_payer == "MD").astype(np.float64)
    tr_data["is_self_pay"] = (tr_payer == "SP").astype(np.float64)
    ev_data["is_self_pay"] = (ev_payer == "SP").astype(np.float64)
    extra_num.extend(["is_medicare", "is_medicaid", "is_self_pay"])

    # 6. Lab Results & Clinical Inertia
    a1c_tr = train["A1Cresult"]
    a1c_ev = eval_cols["A1Cresult"]
    glu_tr = train["max_glu_serum"]
    glu_ev = eval_cols["max_glu_serum"]

    tr_data["a1c_performed"] = (~np.isin(a1c_tr, [None, "None"])).astype(np.float64)
    ev_data["a1c_performed"] = (~np.isin(a1c_ev, [None, "None"])).astype(np.float64)
    tr_data["glu_performed"] = (~np.isin(glu_tr, [None, "None"])).astype(np.float64)
    ev_data["glu_performed"] = (~np.isin(glu_ev, [None, "None"])).astype(np.float64)
    tr_data["a1c_gt8"] = (a1c_tr == ">8").astype(np.float64)
    ev_data["a1c_gt8"] = (a1c_ev == ">8").astype(np.float64)
    tr_data["glu_high"] = np.isin(glu_tr, [">200", ">300"]).astype(np.float64)
    ev_data["glu_high"] = np.isin(glu_ev, [">200", ">300"]).astype(np.float64)

    ch_tr = (train["change"] == "Ch").astype(np.float64)
    ch_ev = (eval_cols["change"] == "Ch").astype(np.float64)
    tr_data["clinical_inertia"] = ((tr_data["a1c_gt8"] == 1.0) & (ch_tr == 0.0)).astype(np.float64)
    ev_data["clinical_inertia"] = ((ev_data["a1c_gt8"] == 1.0) & (ch_ev == 0.0)).astype(np.float64)
    extra_num.extend(["a1c_performed", "glu_performed", "a1c_gt8", "glu_high", "clinical_inertia"])

    # 7. Medication Regimens & Insulin Dynamics
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    num_active_tr = np.zeros(n_train_rows, dtype=np.float64)
    num_active_ev = np.zeros(n_eval_rows, dtype=np.float64)
    num_change_tr = np.zeros(n_train_rows, dtype=np.float64)
    num_change_ev = np.zeros(n_eval_rows, dtype=np.float64)

    for col in med_cols:
        if col in train:
            num_active_tr += (~np.isin(train[col], [None, "None", "No"])).astype(np.float64)
            num_change_tr += np.isin(train[col], ["Up", "Down"]).astype(np.float64)
        if col in eval_cols:
            num_active_ev += (~np.isin(eval_cols[col], [None, "None", "No"])).astype(np.float64)
            num_change_ev += np.isin(eval_cols[col], ["Up", "Down"]).astype(np.float64)

    tr_data["num_active_meds"] = num_active_tr
    ev_data["num_active_meds"] = num_active_ev
    tr_data["num_change_meds"] = num_change_tr
    ev_data["num_change_meds"] = num_change_ev
    extra_num.extend(["num_active_meds", "num_change_meds"])

    ins_tr = train["insulin"]
    ins_ev = eval_cols["insulin"]
    tr_data["on_insulin"] = (~np.isin(ins_tr, [None, "None", "No"])).astype(np.float64)
    ev_data["on_insulin"] = (~np.isin(ins_ev, [None, "None", "No"])).astype(np.float64)
    tr_data["insulin_up"] = (ins_tr == "Up").astype(np.float64)
    ev_data["insulin_up"] = (ins_ev == "Up").astype(np.float64)
    tr_data["insulin_down"] = (ins_tr == "Down").astype(np.float64)
    ev_data["insulin_down"] = (ins_ev == "Down").astype(np.float64)
    extra_num.extend(["on_insulin", "insulin_up", "insulin_down"])

    # Encode tabular features
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, tr_data, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_train = encoder.transform(tr_data)
    x_eval = encoder.transform(ev_data)

    # 8. Architecturally diverse gradient boosting ensemble
    models = [
        # Model 1: Deep leaf-wise exploration
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.022,
            num_leaves=35,
            min_child_samples=160,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=1.5,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Shallow, robust depth-constrained trees
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.025,
            max_depth=5,
            num_leaves=22,
            min_child_samples=220,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=3.0,
            cat_smooth=50.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Extremely Randomized Trees (randomized threshold splits)
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.024,
            num_leaves=42,
            min_child_samples=140,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            extra_trees=True,
            reg_lambda=1.0,
            cat_smooth=40.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: Narrow feature subspace & high L2 regularization
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.019,
            num_leaves=26,
            min_child_samples=180,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.45,
            reg_lambda=4.0,
            reg_alpha=1.0,
            cat_smooth=35.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 5: High categorical smoothing & broad coverage
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.024,
            num_leaves=48,
            min_child_samples=240,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            cat_smooth=80.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
