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
    """Predict 30-day readmission risk using low-rank clinical factors and a regularized dual GBDT."""
    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    extra_numeric: List[str] = []
    extra_categorical: List[str] = []

    # 1. Low-Rank Comorbidity Factor Projection (Charlson indices across all diagnosis slots)
    if "icd9_comorbidities" in globals():
        cm_train = icd9_comorbidities(train.get("diag_1"), train.get("diag_2"), train.get("diag_3"))
        cm_eval = icd9_comorbidities(eval_cols.get("diag_1"), eval_cols.get("diag_2"), eval_cols.get("diag_3"))
        for k, v in cm_train.items():
            train_ext[k] = v.astype(np.float64)
            eval_ext[k] = cm_eval[k].astype(np.float64)
            extra_numeric.append(k)

        if "cm_chf" in train_ext and "cm_renal" in train_ext:
            train_ext["feat_cardiorenal"] = (train_ext["cm_chf"] * train_ext["cm_renal"]).astype(np.float64)
            eval_ext["feat_cardiorenal"] = (eval_ext["cm_chf"] * eval_ext["cm_renal"]).astype(np.float64)
            extra_numeric.append("feat_cardiorenal")

    # 2. Primary Diagnosis Chapter (Reason for current admission)
    if "icd9_group" in globals() and "diag_1" in train:
        train_ext["diag_1_grp"] = np.array([icd9_group(c) for c in train["diag_1"]], dtype=object)
        eval_ext["diag_1_grp"] = np.array([icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
        extra_categorical.append("diag_1_grp")

    # 3. Target Encoding on High-Cardinality Admission/Discharge Transition Points
    te_cols = ["discharge_disposition_id", "admission_source_id", "diag_1", "medical_specialty"]
    for col in te_cols:
        if col in train and col in eval_cols:
            te_name = f"te_{col}"
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
            )
            train_ext[te_name] = tr_te.astype(np.float64)
            eval_ext[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # 4. EHR Utilization & Care Intensity Projection (Non-redundant representation)
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
            "feat_service_intensity": (labs + procs + meds) / time_hosp,
        }

    tr_intens = _extract_intensity(train)
    ev_intens = _extract_intensity(eval_cols)
    for k in tr_intens:
        train_ext[k] = tr_intens[k].astype(np.float64)
        eval_ext[k] = ev_intens[k].astype(np.float64)
        extra_numeric.append(k)

    # 5. Medication Regimen Complexity & Glycemic Volatility
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
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
            ins = d["insulin"]
            feats["feat_on_insulin"] = np.isin(ins, ["Steady", "Up", "Down"]).astype(np.float64)
            feats["feat_insulin_up"] = (ins == "Up").astype(np.float64)
            feats["feat_insulin_down"] = (ins == "Down").astype(np.float64)

        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            feats["feat_a1c_tested"] = (~np.isin(a1c, ["None", None])).astype(np.float64)

        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            feats["feat_glu_tested"] = (~np.isin(glu, ["None", None])).astype(np.float64)

        return feats

    tr_meds = _extract_meds(train)
    ev_meds = _extract_meds(eval_cols)
    for k in tr_meds:
        train_ext[k] = tr_meds[k].astype(np.float64)
        eval_ext[k] = ev_meds[k].astype(np.float64)
        extra_numeric.append(k)

    # 6. Age midpoint encoding
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    train_ext["feat_age_num"] = np.array([age_map.get(a, 65.0) for a in train["age"]], dtype=np.float64)
    eval_ext["feat_age_num"] = np.array([age_map.get(a, 65.0) for a in eval_cols["age"]], dtype=np.float64)
    extra_numeric.append("feat_age_num")

    # 7. Tabular encoding
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

    # 8. Structurally Complementary Dual GBDT Ensemble (Low-Rank Main Effects + Leaf-Wise Interactions)
    model_main = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=24,
        min_child_samples=160,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=3.0,
        reg_alpha=0.2,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_inter = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=140,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=30.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )

    model_main.fit(x_train, y_train, categorical_feature=cats)
    model_inter.fit(x_train, y_train, categorical_feature=cats)

    p1 = model_main.predict_proba(x_eval)[:, 1]
    p2 = model_inter.predict_proba(x_eval)[:, 1]

    pred = 0.50 * p1 + 0.50 * p2
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
