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


def _manual_oof_te(
    tr_col: np.ndarray,
    y: np.ndarray,
    ev_col: np.ndarray,
    n_folds: int = 5,
    smoothing: float = 25.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback smoothed out-of-fold target encoding if toolkit function is absent."""
    prior = float(np.mean(y))
    n = len(y)
    tr_out = np.full(n, prior, dtype=np.float64)
    ev_out = np.full(len(ev_col), prior, dtype=np.float64)

    s_dict: Dict[Any, float] = {}
    c_dict: Dict[Any, int] = {}
    for v, y_val in zip(tr_col, y):
        s_dict[v] = s_dict.get(v, 0.0) + y_val
        c_dict[v] = c_dict.get(v, 0) + 1
    for i, v in enumerate(ev_col):
        if v in s_dict:
            cnt = c_dict[v]
            s = s_dict[v]
            ev_out[i] = (s + smoothing * prior) / (cnt + smoothing)

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        f_s: Dict[Any, float] = {}
        f_c: Dict[Any, int] = {}
        for v, y_val in zip(tr_col[mask], y[mask]):
            f_s[v] = f_s.get(v, 0.0) + y_val
            f_c[v] = f_c.get(v, 0) + 1
        for idx in fold:
            v = tr_col[idx]
            if v in f_s:
                cnt = f_c[v]
                s = f_s[v]
                tr_out[idx] = (s + smoothing * prior) / (cnt + smoothing)
    return tr_out, ev_out


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit diverse clinical GBDTs on engineered domain features and score evaluation risk."""
    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    extra_numeric: List[str] = []
    extra_categorical: List[str] = []

    # 1. Comorbidities extraction (Charlson index flags)
    if "icd9_comorbidities" in globals():
        cm_train = icd9_comorbidities(train.get("diag_1"), train.get("diag_2"), train.get("diag_3"))
        cm_eval = icd9_comorbidities(eval_cols.get("diag_1"), eval_cols.get("diag_2"), eval_cols.get("diag_3"))
        for k, v in cm_train.items():
            train_ext[k] = v.astype(np.float64)
            eval_ext[k] = cm_eval[k].astype(np.float64)
            extra_numeric.append(k)

        # Cardiorenal clinical syndrome interaction
        if "cm_chf" in train_ext and "cm_renal" in train_ext:
            train_ext["feat_cardiorenal"] = (train_ext["cm_chf"] * train_ext["cm_renal"]).astype(np.float64)
            eval_ext["feat_cardiorenal"] = (eval_ext["cm_chf"] * eval_ext["cm_renal"]).astype(np.float64)
            extra_numeric.append("feat_cardiorenal")

    # 2. ICD-9 Chapter Organ-System Groupings
    if "icd9_group" in globals():
        for diag_col in ["diag_1", "diag_2", "diag_3"]:
            if diag_col in train:
                train_ext[f"{diag_col}_grp"] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
                eval_ext[f"{diag_col}_grp"] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)
                extra_categorical.append(f"{diag_col}_grp")

    # 3. Out-of-fold target encoding of high-cardinality codes
    te_candidates = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
    ]
    for col in te_candidates:
        if col in train and col in eval_cols:
            te_name = f"te_{col}"
            if "oof_target_encode" in globals():
                tr_te, ev_te = oof_target_encode(
                    train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
                )
            else:
                tr_te, ev_te = _manual_oof_te(
                    train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
                )
            train_ext[te_name] = tr_te.astype(np.float64)
            eval_ext[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # 4. EHR Utilization Velocity & Service Intensity
    def _extract_intensity(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        time_hosp = np.maximum(1.0, np.nan_to_num(d["time_in_hospital"], nan=1.0))
        labs = np.maximum(0.0, np.nan_to_num(d["num_lab_procedures"], nan=0.0))
        procs = np.maximum(0.0, np.nan_to_num(d["num_procedures"], nan=0.0))
        meds = np.maximum(0.0, np.nan_to_num(d["num_medications"], nan=0.0))
        outp = np.maximum(0.0, np.nan_to_num(d["number_outpatient"], nan=0.0))
        emerg = np.maximum(0.0, np.nan_to_num(d["number_emergency"], nan=0.0))
        inpat = np.maximum(0.0, np.nan_to_num(d["number_inpatient"], nan=0.0))
        diag = np.maximum(0.0, np.nan_to_num(d["number_diagnoses"], nan=0.0))

        total_prior = outp + emerg + inpat
        return {
            "feat_total_prior": total_prior,
            "feat_has_inpatient": (inpat > 0).astype(np.float64),
            "feat_has_emergency": (emerg > 0).astype(np.float64),
            "feat_log_inpatient": np.log1p(inpat),
            "feat_log_emergency": np.log1p(emerg),
            "feat_log_total_prior": np.log1p(total_prior),
            "feat_inpatient_ratio": (inpat + 0.05) / (total_prior + 0.1),
            "feat_lab_per_day": labs / time_hosp,
            "feat_med_per_day": meds / time_hosp,
            "feat_proc_per_day": procs / time_hosp,
            "feat_diag_per_day": diag / time_hosp,
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
            feats["feat_insulin_up"] = (ins == "Up").astype(np.float64)
            feats["feat_insulin_down"] = (ins == "Down").astype(np.float64)
            feats["feat_insulin_steady"] = (ins == "Steady").astype(np.float64)
            feats["feat_on_insulin"] = np.isin(ins, ["Steady", "Up", "Down"]).astype(np.float64)

        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            feats["feat_a1c_tested"] = (~np.isin(a1c, ["None", None])).astype(np.float64)
            feats["feat_a1c_high"] = np.isin(a1c, [">7", ">8"]).astype(np.float64)

        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            feats["feat_glu_tested"] = (~np.isin(glu, ["None", None])).astype(np.float64)
            feats["feat_glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

        return feats

    tr_meds = _extract_meds(train)
    ev_meds = _extract_meds(eval_cols)
    for k in tr_meds:
        train_ext[k] = tr_meds[k].astype(np.float64)
        eval_ext[k] = ev_meds[k].astype(np.float64)
        extra_numeric.append(k)

    # 6. Age conversion & Frailty Interaction
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    age_tr = np.array([age_map.get(a, 65.0) for a in train["age"]], dtype=np.float64)
    age_ev = np.array([age_map.get(a, 65.0) for a in eval_cols["age"]], dtype=np.float64)
    train_ext["feat_age_num"] = age_tr
    eval_ext["feat_age_num"] = age_ev
    extra_numeric.append("feat_age_num")

    train_ext["feat_frailty"] = (
        (age_tr / 50.0)
        * (train_ext["feat_log_inpatient"] + 0.5)
        * (np.nan_to_num(train["number_diagnoses"], nan=5.0) / 5.0)
    ).astype(np.float64)
    eval_ext["feat_frailty"] = (
        (age_ev / 50.0)
        * (eval_ext["feat_log_inpatient"] + 0.5)
        * (np.nan_to_num(eval_cols["number_diagnoses"], nan=5.0) / 5.0)
    ).astype(np.float64)
    extra_numeric.append("feat_frailty")

    # 7. Tabular encoding with granular threshold
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_ext,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_ext)
    x_eval = encoder.transform(eval_ext)

    # 8. Diverse GBDT Ensemble
    model_leaf = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_depth = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=25,
        min_child_samples=200,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=3.0,
        reg_alpha=0.5,
        cat_smooth=45.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model_fine = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=42,
        min_child_samples=100,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=2.0,
        reg_alpha=0.05,
        cat_smooth=25.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )

    cats = encoder.categorical_indices
    model_leaf.fit(x_train, y_train, categorical_feature=cats)
    model_depth.fit(x_train, y_train, categorical_feature=cats)
    model_fine.fit(x_train, y_train, categorical_feature=cats)

    p1 = model_leaf.predict_proba(x_eval)[:, 1]
    p2 = model_depth.predict_proba(x_eval)[:, 1]
    p3 = model_fine.predict_proba(x_eval)[:, 1]

    pred = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
