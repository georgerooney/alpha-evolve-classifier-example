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

    # 2. Optimal rank-1 projections (out-of-fold target encodings)
    te_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "medical_specialty",
    ]
    for col in te_cols:
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
        te_name = f"{col}_te"
        tr_data[te_name] = tr_te
        ev_data[te_name] = ev_te
        extra_num.append(te_name)

    # Joint Admission/Discharge & Specialty/Discharge target encodings
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

    # 3. Prior utilization intensity & rates
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
    tr_data["inpatient_ratio"] = inp_tr / (tot_tr + 1.0)
    ev_data["inpatient_ratio"] = inp_ev / (tot_ev + 1.0)
    tr_data["has_prior_emergency"] = (em_tr > 0).astype(np.float64)
    ev_data["has_prior_emergency"] = (em_ev > 0).astype(np.float64)
    tr_data["emergency_ratio"] = em_tr / (tot_tr + 1.0)
    ev_data["emergency_ratio"] = em_ev / (tot_ev + 1.0)
    extra_num.extend(["total_prior", "has_prior_inpatient", "inpatient_ratio", "has_prior_emergency", "emergency_ratio"])

    # 4. Care Intensity & Ratios
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

    # 5. Demographics and Medical context
    def parse_age(age_arr):
        res = np.full(age_arr.shape, np.nan, dtype=np.float64)
        age_map = {
            "[0-10)": 5.0,
            "[10-20)": 15.0,
            "[20-30)": 25.0,
            "[30-40)": 35.0,
            "[40-50)": 45.0,
            "[50-60)": 55.0,
            "[60-70)": 65.0,
            "[70-80)": 75.0,
            "[80-90)": 85.0,
            "[90-100)": 95.0
        }
        for k, v in age_map.items():
            res[age_arr == k] = v
        return res

    tr_data["age_midpoint"] = parse_age(train["age"])
    ev_data["age_midpoint"] = parse_age(eval_cols["age"])
    extra_num.append("age_midpoint")

    tr_data["weight_present"] = (~np.isin(train["weight"], [None, "None"])).astype(np.float64)
    ev_data["weight_present"] = (~np.isin(eval_cols["weight"], [None, "None"])).astype(np.float64)
    extra_num.append("weight_present")

    tr_data["a1c_performed"] = (~np.isin(train["A1Cresult"], [None, "None"])).astype(np.float64)
    ev_data["a1c_performed"] = (~np.isin(eval_cols["A1Cresult"], [None, "None"])).astype(np.float64)
    tr_data["glu_performed"] = (~np.isin(train["max_glu_serum"], [None, "None"])).astype(np.float64)
    ev_data["glu_performed"] = (~np.isin(eval_cols["max_glu_serum"], [None, "None"])).astype(np.float64)
    extra_num.extend(["a1c_performed", "glu_performed"])

    # 6. Diagnosis groups from icd9_group
    for i in (1, 2, 3):
        col = f"diag_{i}"
        tr_data[f"diag_group_{i}"] = np.array([icd9_group(x) for x in train[col]])
        ev_data[f"diag_group_{i}"] = np.array([icd9_group(x) for x in eval_cols[col]])
        extra_cat.append(f"diag_group_{i}")

    # 7. Medication active/change counts
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    any_col = next(iter(train.keys()))
    n_train_rows = len(train[any_col])
    any_col_ev = next(iter(eval_cols.keys()))
    n_eval_rows = len(eval_cols[any_col_ev])

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

    # Encode tabular features
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, tr_data, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_train = encoder.transform(tr_data)
    x_eval = encoder.transform(ev_data)

    # 8. Multi-seed gradient boosting ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.022,
            num_leaves=31,
            min_child_samples=180,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=1.0,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.026,
            num_leaves=24,
            min_child_samples=220,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=2.0,
            cat_smooth=50.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.020,
            num_leaves=38,
            min_child_samples=150,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.5,
            cat_smooth=50.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.018,
            num_leaves=20,
            min_child_samples=160,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.45,
            reg_lambda=3.0,
            reg_alpha=1.0,
            cat_smooth=35.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.024,
            num_leaves=45,
            min_child_samples=250,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=1.5,
            cat_smooth=75.0,
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
