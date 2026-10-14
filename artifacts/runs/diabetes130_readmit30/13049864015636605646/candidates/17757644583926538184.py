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
    from sklearn.decomposition import TruncatedSVD
    try:
        from clinical_toolkit import (
            TabularEncoder,
            icd9_comorbidities,
            icd9_group,
            oof_target_encode,
        )
    except ImportError:
        pass

    n_train = len(y_train)
    n_eval = len(eval_cols["time_in_hospital"])

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson comorbidities derived from diagnosis codes
    cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in sorted(cm_train.keys()):
        train_feats[k] = cm_train[k].astype(np.float64)
        eval_feats[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # 2. ICD-9 Chapter groupings (organ systems)
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        gcol = f"{diag_col}_group"
        train_feats[gcol] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_feats[gcol] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(gcol)

    # 3. Out-of-fold target encoding for high-cardinality nominal features
    te_columns = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "admission_type_id",
    ]
    for col in te_columns:
        if col in train and col in eval_cols:
            te_tr, te_ev = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
            )
            col_name = f"te_{col}"
            train_feats[col_name] = te_tr.astype(np.float64)
            eval_feats[col_name] = te_ev.astype(np.float64)
            extra_numeric.append(col_name)

    # 4. Ordinal continuous age parsing
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    train_feats["age_num"] = np.array([age_map.get(a, np.nan) for a in train["age"]], dtype=np.float64)
    eval_feats["age_num"] = np.array([age_map.get(a, np.nan) for a in eval_cols["age"]], dtype=np.float64)
    extra_numeric.append("age_num")

    # 5. Prior healthcare utilization & care intensity metrics
    t_hosp_tr = np.maximum(np.nan_to_num(train["time_in_hospital"], nan=1.0), 1.0)
    t_hosp_ev = np.maximum(np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0), 1.0)

    inpat_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inpat_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emerg_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emerg_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    outpat_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    outpat_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    prior_tot_tr = inpat_tr + emerg_tr + outpat_tr
    prior_tot_ev = inpat_ev + emerg_ev + outpat_ev

    util_feats = {
        "prior_visits_total": (prior_tot_tr, prior_tot_ev),
        "has_prior_inpatient": ((inpat_tr > 0).astype(np.float64), (inpat_ev > 0).astype(np.float64)),
        "has_prior_emergency": ((emerg_tr > 0).astype(np.float64), (emerg_ev > 0).astype(np.float64)),
        "has_prior_outpatient": ((outpat_tr > 0).astype(np.float64), (outpat_ev > 0).astype(np.float64)),
        "inpatient_ratio": (inpat_tr / (prior_tot_tr + 1.0), inpat_ev / (prior_tot_ev + 1.0)),
        "emergency_ratio": (emerg_tr / (prior_tot_tr + 1.0), emerg_ev / (prior_tot_ev + 1.0)),
        "inpatient_sq": (inpat_tr ** 2, inpat_ev ** 2),
        "med_per_day": (np.nan_to_num(train["num_medications"], nan=0.0) / t_hosp_tr,
                        np.nan_to_num(eval_cols["num_medications"], nan=0.0) / t_hosp_ev),
        "lab_per_day": (np.nan_to_num(train["num_lab_procedures"], nan=0.0) / t_hosp_tr,
                        np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0) / t_hosp_ev),
        "proc_per_day": (np.nan_to_num(train["num_procedures"], nan=0.0) / t_hosp_tr,
                         np.nan_to_num(eval_cols["num_procedures"], nan=0.0) / t_hosp_ev),
        "diag_per_day": (np.nan_to_num(train["number_diagnoses"], nan=0.0) / t_hosp_tr,
                         np.nan_to_num(eval_cols["number_diagnoses"], nan=0.0) / t_hosp_ev),
        "med_x_time": (np.nan_to_num(train["num_medications"], nan=0.0) * t_hosp_tr,
                       np.nan_to_num(eval_cols["num_medications"], nan=0.0) * t_hosp_ev),
        "total_procedures": (np.nan_to_num(train["num_lab_procedures"], nan=0.0) + np.nan_to_num(train["num_procedures"], nan=0.0),
                            np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0) + np.nan_to_num(eval_cols["num_procedures"], nan=0.0)),
    }
    for fname, (ftr, fev) in util_feats.items():
        train_feats[fname] = ftr.astype(np.float64)
        eval_feats[fname] = fev.astype(np.float64)
        extra_numeric.append(fname)

    # 6. Pharmacotherapy profile extraction & medication indicators
    med_cols = [
        col for col in task.categorical_columns
        if set(np.unique(train[col][train[col] != None])).issubset({"No", "Steady", "Up", "Down"})
    ]

    med_mat_tr = np.zeros((n_train, len(med_cols)), dtype=np.float64)
    med_mat_ev = np.zeros((n_eval, len(med_cols)), dtype=np.float64)
    for j, mcol in enumerate(med_cols):
        tr_c = train[mcol]
        ev_c = eval_cols[mcol]
        med_mat_tr[tr_c == "Steady", j] = 1.0
        med_mat_tr[tr_c == "Down", j] = -1.0
        med_mat_tr[tr_c == "Up", j] = 2.0
        med_mat_ev[ev_c == "Steady", j] = 1.0
        med_mat_ev[ev_c == "Down", j] = -1.0
        med_mat_ev[ev_c == "Up", j] = 2.0

    train_feats["med_active_count"] = np.sum(med_mat_tr != 0.0, axis=1, dtype=np.float64)
    eval_feats["med_active_count"] = np.sum(med_mat_ev != 0.0, axis=1, dtype=np.float64)
    extra_numeric.append("med_active_count")

    train_feats["med_up_count"] = np.sum(med_mat_tr == 2.0, axis=1, dtype=np.float64)
    eval_feats["med_up_count"] = np.sum(med_mat_ev == 2.0, axis=1, dtype=np.float64)
    extra_numeric.append("med_up_count")

    train_feats["med_down_count"] = np.sum(med_mat_tr == -1.0, axis=1, dtype=np.float64)
    eval_feats["med_down_count"] = np.sum(med_mat_ev == -1.0, axis=1, dtype=np.float64)
    extra_numeric.append("med_down_count")

    train_feats["med_change_count"] = np.sum((med_mat_tr == 2.0) | (med_mat_tr == -1.0), axis=1, dtype=np.float64)
    eval_feats["med_change_count"] = np.sum((med_mat_ev == 2.0) | (med_mat_ev == -1.0), axis=1, dtype=np.float64)
    extra_numeric.append("med_change_count")

    if "change" in train:
        train_feats["has_change"] = (train["change"] == "Ch").astype(np.float64)
        eval_feats["has_change"] = (eval_cols["change"] == "Ch").astype(np.float64)
        extra_numeric.append("has_change")

    if "diabetesMed" in train:
        train_feats["has_diabetes_med"] = (train["diabetesMed"] == "Yes").astype(np.float64)
        eval_feats["has_diabetes_med"] = (eval_cols["diabetesMed"] == "Yes").astype(np.float64)
        extra_numeric.append("has_diabetes_med")

    if "insulin" in train:
        ins_map = {"No": 0.0, "Steady": 1.0, "Down": 2.0, "Up": 3.0}
        train_feats["insulin_status"] = np.array([ins_map.get(v, 0.0) for v in train["insulin"]], dtype=np.float64)
        eval_feats["insulin_status"] = np.array([ins_map.get(v, 0.0) for v in eval_cols["insulin"]], dtype=np.float64)
        extra_numeric.append("insulin_status")

    # 7. Low-Rank Tensor/Matrix Decomposition of Multimodal Clinical Phenotypes
    cm_flag_keys = [k for k in sorted(cm_train.keys()) if k.startswith("cm_") and k != "cm_count"]
    cm_mat_tr = np.column_stack([cm_train[k] for k in cm_flag_keys])
    cm_mat_ev = np.column_stack([cm_eval[k] for k in cm_flag_keys])

    # Unfolded joint clinical state: Patient x (Pharmacotherapy + Comorbidities)
    clin_tr = np.column_stack([med_mat_tr, cm_mat_tr])
    clin_ev = np.column_stack([med_mat_ev, cm_mat_ev])

    n_svd = 6
    svd_clin = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_tr = svd_clin.fit_transform(clin_tr)
    svd_ev = svd_clin.transform(clin_ev)

    for k in range(n_svd):
        cname = f"svd_clin_{k}"
        train_feats[cname] = svd_tr[:, k].astype(np.float64)
        eval_feats[cname] = svd_ev[:, k].astype(np.float64)
        extra_numeric.append(cname)

    # Low-rank subspace reconstruction error: measures patient atypicality / clinical anomaly
    recon_tr = clin_tr - svd_clin.inverse_transform(svd_tr)
    recon_ev = clin_ev - svd_clin.inverse_transform(svd_ev)
    train_feats["clin_recon_err"] = np.sum(recon_tr ** 2, axis=1, dtype=np.float64)
    eval_feats["clin_recon_err"] = np.sum(recon_ev ** 2, axis=1, dtype=np.float64)
    extra_numeric.append("clin_recon_err")

    # Latent pharmacotherapy axes
    if med_mat_tr.shape[1] >= 4:
        svd_med = TruncatedSVD(n_components=3, random_state=42)
        svd_m_tr = svd_med.fit_transform(med_mat_tr)
        svd_m_ev = svd_med.transform(med_mat_ev)
        for k in range(3):
            mname = f"svd_med_{k}"
            train_feats[mname] = svd_m_tr[:, k].astype(np.float64)
            eval_feats[mname] = svd_m_ev[:, k].astype(np.float64)
            extra_numeric.append(mname)

    # 8. Encode features
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, train_feats, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 9. Diverse Multi-Topology Gradient-Boosted Decision Tree Ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.1,
        reg_lambda=1.5,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    model2 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.018,
        num_leaves=20,
        max_depth=5,
        min_child_samples=220,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=5.0,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )

    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)

    preds1 = model1.predict_proba(x_eval)[:, 1]
    preds2 = model2.predict_proba(x_eval)[:, 1]
    ensemble_preds = 0.5 * preds1 + 0.5 * preds2

    return np.clip(ensemble_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
