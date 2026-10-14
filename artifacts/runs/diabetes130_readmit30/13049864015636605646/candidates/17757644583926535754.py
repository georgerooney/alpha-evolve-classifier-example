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
    """Predict 30-day readmission risk using parsimonious low-rank representations and dual-scale GBDT optimization."""
    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    extra_numeric: List[str] = []
    extra_categorical: List[str] = []

    # 1. Low-Rank Comorbidity Decomposition
    if "icd9_comorbidities" in globals():
        cm_train = icd9_comorbidities(train.get("diag_1"), train.get("diag_2"), train.get("diag_3"))
        cm_eval = icd9_comorbidities(eval_cols.get("diag_1"), eval_cols.get("diag_2"), eval_cols.get("diag_3"))
        for k, v in cm_train.items():
            train_ext[k] = v.astype(np.float64)
            eval_ext[k] = cm_eval[k].astype(np.float64)
            extra_numeric.append(k)

    # 2. Compact Diagnosis Chapter Grouping
    if "icd9_group" in globals():
        for d_col in ["diag_1", "diag_2", "diag_3"]:
            if d_col in train and d_col in eval_cols:
                grp_col = f"{d_col}_grp"
                train_ext[grp_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
                eval_ext[grp_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
                extra_categorical.append(grp_col)

    # 3. Bayes-Optimal Target Encoding (Rank-1 Conditional Expectation)
    te_cols = [
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "diag_1",
        "diag_2",
        "diag_3",
        "medical_specialty",
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            te_name = f"te_{col}"
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
            )
            train_ext[te_name] = tr_te.astype(np.float64)
            eval_ext[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # 4. Care Intensity & Utilization Ratios (Non-linear Oblique Projections)
    def _extract_intensity(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        time_hosp = np.maximum(1.0, np.nan_to_num(d["time_in_hospital"], nan=1.0))
        labs = np.maximum(0.0, np.nan_to_num(d["num_lab_procedures"], nan=0.0))
        procs = np.maximum(0.0, np.nan_to_num(d["num_procedures"], nan=0.0))
        meds = np.maximum(0.0, np.nan_to_num(d["num_medications"], nan=0.0))
        outp = np.maximum(0.0, np.nan_to_num(d["number_outpatient"], nan=0.0))
        emerg = np.maximum(0.0, np.nan_to_num(d["number_emergency"], nan=0.0))
        inpat = np.maximum(0.0, np.nan_to_num(d["number_inpatient"], nan=0.0))

        total_prior = outp + emerg + inpat
        return {
            "feat_total_prior": total_prior,
            "feat_inpatient_ratio": (inpat + 0.05) / (total_prior + 0.1),
            "feat_emergency_ratio": (emerg + 0.05) / (total_prior + 0.1),
            "feat_service_intensity": (labs + procs + meds) / time_hosp,
            "feat_meds_per_day": meds / time_hosp,
            "feat_labs_per_day": labs / time_hosp,
            "feat_procs_per_day": procs / time_hosp,
        }

    tr_intens = _extract_intensity(train)
    ev_intens = _extract_intensity(eval_cols)
    for k in tr_intens:
        train_ext[k] = tr_intens[k].astype(np.float64)
        eval_ext[k] = ev_intens[k].astype(np.float64)
        extra_numeric.append(k)

    # 5. Medication Regimen Aggregates (Row-Sum Pooling)
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def _extract_meds(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(d["time_in_hospital"])
        active_cnt = np.zeros(n, dtype=np.float64)
        change_cnt = np.zeros(n, dtype=np.float64)
        for m in med_cols:
            if m in d:
                arr = d[m]
                active_cnt += np.isin(arr, ["Steady", "Up", "Down"]).astype(np.float64)
                change_cnt += np.isin(arr, ["Up", "Down"]).astype(np.float64)
        feats: Dict[str, np.ndarray] = {
            "feat_num_active_meds": active_cnt,
            "feat_num_med_changes": change_cnt,
        }
        if "insulin" in d:
            feats["feat_on_insulin"] = np.isin(d["insulin"], ["Steady", "Up", "Down"]).astype(np.float64)
        return feats

    tr_meds = _extract_meds(train)
    ev_meds = _extract_meds(eval_cols)
    for k in tr_meds:
        train_ext[k] = tr_meds[k].astype(np.float64)
        eval_ext[k] = ev_meds[k].astype(np.float64)
        extra_numeric.append(k)

    # 6. Monotonic Age Representation
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    train_ext["feat_age_num"] = np.array([age_map.get(a, 65.0) for a in train["age"]], dtype=np.float64)
    eval_ext["feat_age_num"] = np.array([age_map.get(a, 65.0) for a in eval_cols["age"]], dtype=np.float64)
    extra_numeric.append("feat_age_num")

    # 7. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_ext,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_ext)
    x_eval = encoder.transform(eval_ext)
    cats = encoder.categorical_indices

    # 8. Dual-Scale GBDT Optimization: Additive Main Effects (Depth-4) + Interaction (Depth-6)
    model1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=15,
        min_child_samples=160,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=4.0,
        reg_alpha=0.3,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=28,
        min_child_samples=140,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=3.0,
        reg_alpha=0.2,
        cat_smooth=25.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )

    model1.fit(x_train, y_train, categorical_feature=cats)
    model2.fit(x_train, y_train, categorical_feature=cats)

    p1 = model1.predict_proba(x_eval)[:, 1]
    p2 = model2.predict_proba(x_eval)[:, 1]

    pred = 0.50 * p1 + 0.50 * p2
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
