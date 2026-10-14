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

Strategy only: an untuned single LightGBM on the raw columns. Encoding mechanics live in `clinical_toolkit`.
"""

from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np

from clinical_models import TabularTask
from clinical_toolkit import TabularEncoder, icd9_group  # noqa: F401  (icd9_group available for feature ideas)


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Pre-imported in the sandbox: numpy, scipy, sklearn, lightgbm, xgboost. No file/network access.


def _compute_oof_target_encoding(
    train_cat: np.ndarray,
    eval_cat: np.ndarray,
    y: np.ndarray,
    folds: list,
    m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding for train and full-train Bayesian smoothed encoding for eval."""
    n_train = len(train_cat)
    n_eval = len(eval_cat)
    global_prior = float(np.mean(y))

    train_te = np.full(n_train, global_prior, dtype=np.float64)
    eval_te = np.full(n_eval, global_prior, dtype=np.float64)

    tr_keys = [str(c) if (c is not None and c == c) else "__MISSING__" for c in train_cat]
    ev_keys = [str(c) if (c is not None and c == c) else "__MISSING__" for c in eval_cat]

    for tr_idx, val_idx in folds:
        y_tr = y[tr_idx]
        fold_prior = float(np.mean(y_tr))

        stats: dict[str, list[int]] = {}
        for idx in tr_idx:
            k = tr_keys[idx]
            lbl = int(y[idx])
            if k not in stats:
                stats[k] = [0, 0]
            stats[k][0] += 1
            stats[k][1] += lbl

        for idx in val_idx:
            k = tr_keys[idx]
            if k in stats:
                cnt, sm = stats[k]
                train_te[idx] = (sm + m * fold_prior) / (cnt + m)
            else:
                train_te[idx] = fold_prior

    full_stats: dict[str, list[int]] = {}
    for idx in range(n_train):
        k = tr_keys[idx]
        lbl = int(y[idx])
        if k not in full_stats:
            full_stats[k] = [0, 0]
        full_stats[k][0] += 1
        full_stats[k][1] += lbl

    for idx in range(n_eval):
        k = ev_keys[idx]
        if k in full_stats:
            cnt, sm = full_stats[k]
            eval_te[idx] = (sm + m * global_prior) / (cnt + m)
        else:
            eval_te[idx] = global_prior

    return train_te, eval_te


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    from sklearn.model_selection import StratifiedKFold

    train_feat = dict(train)
    eval_feat = dict(eval_cols)

    extra_numeric: list[str] = []
    extra_categorical: list[str] = []

    # 1. Age as continuous midpoint
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    for d in (train_feat, eval_feat):
        d["age_num"] = np.array([age_map.get(str(a), np.nan) for a in d["age"]], dtype=np.float64)
    extra_numeric.append("age_num")

    # 2. Prior Healthcare Utilization
    for d in (train_feat, eval_feat):
        inp = np.nan_to_num(d["number_inpatient"], nan=0.0)
        outp = np.nan_to_num(d["number_outpatient"], nan=0.0)
        emg = np.nan_to_num(d["number_emergency"], nan=0.0)

        tot_prior = inp + outp + emg
        d["total_prior_visits"] = tot_prior
        d["has_prior_inpatient"] = (inp > 0).astype(np.float64)
        d["has_prior_emergency"] = (emg > 0).astype(np.float64)
        d["has_multiple_inpatient"] = (inp > 1).astype(np.float64)
        d["inpatient_ratio"] = inp / (tot_prior + 1.0)
        d["emergency_ratio"] = emg / (tot_prior + 1.0)
        d["weighted_prior_util"] = 3.0 * inp + 2.0 * emg + outp
        d["log_prior_visits"] = np.log1p(tot_prior)
        d["log_inpatient"] = np.log1p(inp)

    extra_numeric.extend([
        "total_prior_visits", "has_prior_inpatient", "has_prior_emergency",
        "has_multiple_inpatient", "inpatient_ratio", "emergency_ratio",
        "weighted_prior_util", "log_prior_visits", "log_inpatient"
    ])

    # 3. Hospital stay & care intensity
    for d in (train_feat, eval_feat):
        t_hosp = np.maximum(np.nan_to_num(d["time_in_hospital"], nan=1.0), 1.0)
        n_meds = np.nan_to_num(d["num_medications"], nan=0.0)
        n_labs = np.nan_to_num(d["num_lab_procedures"], nan=0.0)
        n_procs = np.nan_to_num(d["num_procedures"], nan=0.0)
        n_diags = np.nan_to_num(d["number_diagnoses"], nan=0.0)

        d["med_per_day"] = n_meds / t_hosp
        d["lab_per_day"] = n_labs / t_hosp
        d["proc_per_day"] = n_procs / t_hosp
        d["diag_per_day"] = n_diags / t_hosp
        d["total_procedures"] = n_labs + n_procs
        d["overall_care_intensity"] = (n_meds + n_labs + n_procs) / t_hosp

    extra_numeric.extend([
        "med_per_day", "lab_per_day", "proc_per_day", "diag_per_day",
        "total_procedures", "overall_care_intensity"
    ])

    # 4. Admission and discharge flags
    for d in (train_feat, eval_feat):
        disch = d["discharge_disposition_id"]
        adm_type = d["admission_type_id"]
        adm_src = d["admission_source_id"]

        d["discharge_home"] = (disch == "1").astype(np.float64)
        d["discharge_snf_care"] = np.isin(disch, ["3", "6", "18", "22", "28"]).astype(np.float64)
        d["discharge_ama"] = (disch == "7").astype(np.float64)
        d["is_emergency_adm"] = np.isin(adm_type, ["1", "2"]).astype(np.float64)
        d["is_elective_adm"] = (adm_type == "3").astype(np.float64)
        d["is_adm_from_er"] = (adm_src == "7").astype(np.float64)
        d["is_adm_transfer"] = np.isin(adm_src, ["4", "5", "6", "10", "22"]).astype(np.float64)

    extra_numeric.extend([
        "discharge_home", "discharge_snf_care", "discharge_ama",
        "is_emergency_adm", "is_elective_adm", "is_adm_from_er", "is_adm_transfer"
    ])

    # 5. Lab tests
    for d in (train_feat, eval_feat):
        a1c = d["A1Cresult"]
        glu = d["max_glu_serum"]

        d["a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
        d["a1c_abnormal"] = np.isin(a1c, [">7", ">8"]).astype(np.float64)
        d["a1c_very_high"] = (a1c == ">8").astype(np.float64)
        d["glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
        d["glu_abnormal"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

    extra_numeric.extend(["a1c_tested", "a1c_abnormal", "a1c_very_high", "glu_tested", "glu_abnormal"])

    # 6. Medications dynamics
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    for d in (train_feat, eval_feat):
        n_rows = len(d["time_in_hospital"])
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in d:
                col = d[m]
                active_cnt += ((col != "No") & (col != None)).astype(np.float64)
                up_cnt += (col == "Up").astype(np.float64)
                down_cnt += (col == "Down").astype(np.float64)

        d["num_active_meds"] = active_cnt
        d["num_meds_up"] = up_cnt
        d["num_meds_down"] = down_cnt
        d["num_meds_changed"] = up_cnt + down_cnt

        ins = d.get("insulin")
        if ins is not None:
            d["has_insulin"] = ((ins != "No") & (ins != None)).astype(np.float64)
            d["insulin_up"] = (ins == "Up").astype(np.float64)
            d["insulin_down"] = (ins == "Down").astype(np.float64)
            d["insulin_changed"] = np.isin(ins, ["Up", "Down"]).astype(np.float64)
        else:
            d["has_insulin"] = np.zeros(n_rows, dtype=np.float64)
            d["insulin_up"] = np.zeros(n_rows, dtype=np.float64)
            d["insulin_down"] = np.zeros(n_rows, dtype=np.float64)
            d["insulin_changed"] = np.zeros(n_rows, dtype=np.float64)

        ch = d.get("change")
        d["change_flag"] = (ch == "Ch").astype(np.float64) if ch is not None else np.zeros(n_rows, dtype=np.float64)
        dm = d.get("diabetesMed")
        d["diabetesMed_flag"] = (dm == "Yes").astype(np.float64) if dm is not None else np.zeros(n_rows, dtype=np.float64)

    extra_numeric.extend([
        "num_active_meds", "num_meds_up", "num_meds_down", "num_meds_changed",
        "has_insulin", "insulin_up", "insulin_down", "insulin_changed",
        "change_flag", "diabetesMed_flag"
    ])

    # 7. ICD-9 Chapter groups and diagnosis roots
    for d in (train_feat, eval_feat):
        d1 = d["diag_1"]
        d2 = d["diag_2"]
        d3 = d["diag_3"]

        g1 = np.array([icd9_group(c) if c is not None else "missing" for c in d1], dtype=object)
        g2 = np.array([icd9_group(c) if c is not None else "missing" for c in d2], dtype=object)
        g3 = np.array([icd9_group(c) if c is not None else "missing" for c in d3], dtype=object)

        d["diag_1_group"] = g1
        d["diag_2_group"] = g2
        d["diag_3_group"] = g3

        r1 = np.array([str(c).split(".")[0] if c is not None else "missing" for c in d1], dtype=object)
        r2 = np.array([str(c).split(".")[0] if c is not None else "missing" for c in d2], dtype=object)
        r3 = np.array([str(c).split(".")[0] if c is not None else "missing" for c in d3], dtype=object)

        d["diag_1_root"] = r1
        d["diag_2_root"] = r2
        d["diag_3_root"] = r3

        d["has_circulatory"] = ((g1 == "circulatory") | (g2 == "circulatory") | (g3 == "circulatory")).astype(np.float64)
        d["has_respiratory"] = ((g1 == "respiratory") | (g2 == "respiratory") | (g3 == "respiratory")).astype(np.float64)
        d["has_diabetes_diag"] = ((g1 == "diabetes") | (g2 == "diabetes") | (g3 == "diabetes")).astype(np.float64)
        d["has_digestive"] = ((g1 == "digestive") | (g2 == "digestive") | (g3 == "digestive")).astype(np.float64)
        d["has_genitourinary"] = ((g1 == "genitourinary") | (g2 == "genitourinary") | (g3 == "genitourinary")).astype(np.float64)
        d["has_neoplasms"] = ((g1 == "neoplasms") | (g2 == "neoplasms") | (g3 == "neoplasms")).astype(np.float64)

        n_rows = len(d1)
        distinct_comorbidities = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            s = {g1[i], g2[i], g3[i]} - {"missing", "other"}
            distinct_comorbidities[i] = len(s)
        d["num_distinct_comorbidities"] = distinct_comorbidities

    extra_categorical.extend([
        "diag_1_group", "diag_2_group", "diag_3_group",
        "diag_1_root", "diag_2_root", "diag_3_root"
    ])
    extra_numeric.extend([
        "has_circulatory", "has_respiratory", "has_diabetes_diag",
        "has_digestive", "has_genitourinary", "has_neoplasms",
        "num_distinct_comorbidities"
    ])

    # 8. Stratified Folds and Out-of-Fold Target Encoding
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = list(skf.split(np.zeros(len(y_train)), y_train))

    te_columns = [
        "diag_1", "diag_2", "diag_3", "diag_1_root",
        "discharge_disposition_id", "admission_source_id", "medical_specialty"
    ]
    for col in te_columns:
        if col in train_feat:
            tr_te, ev_te = _compute_oof_target_encoding(
                train_feat[col], eval_feat[col], y_train, folds=cv_folds, m=20.0
            )
            col_name = f"{col}_te"
            train_feat[col_name] = tr_te
            eval_feat[col_name] = ev_te
            extra_numeric.append(col_name)

    # Diagnosis target encoding summary interactions
    for d in (train_feat, eval_feat):
        d1_te = d["diag_1_te"]
        d2_te = d["diag_2_te"]
        d3_te = d["diag_3_te"]
        d["diag_te_max"] = np.maximum(np.maximum(d1_te, d2_te), d3_te)
        d["diag_te_min"] = np.minimum(np.minimum(d1_te, d2_te), d3_te)
        d["diag_te_mean"] = (d1_te + d2_te + d3_te) / 3.0

    extra_numeric.extend(["diag_te_max", "diag_te_min", "diag_te_mean"])

    # 9. Tabular Encoding
    encoder = TabularEncoder(min_count=15)
    encoder.fit(
        task,
        train_feat,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)
    cat_indices = encoder.categorical_indices

    # 10. Multi-Architecture Stratified K-Fold Ensemble
    eval_preds = np.zeros(len(x_eval), dtype=np.float64)

    models_config = [
        # Model A: Deep / interaction-focused
        dict(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model B: Shallow / highly-regularized
        dict(
            n_estimators=400,
            learning_rate=0.035,
            num_leaves=16,
            max_depth=4,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Model C: Moderate depth / high feature fraction
        dict(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            min_child_samples=30,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    total_fits = len(cv_folds) * len(models_config)
    for tr_idx, _ in cv_folds:
        x_tr = x_train[tr_idx]
        y_tr = y_train[tr_idx]

        for config in models_config:
            clf = lgb.LGBMClassifier(**config)
            clf.fit(x_tr, y_tr, categorical_feature=cat_indices)
            eval_preds += clf.predict_proba(x_eval)[:, 1]

    return eval_preds / float(total_fits)


# EVOLVE-BLOCK-END
# =====================================================================
