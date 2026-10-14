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
    train_feats = dict(train)
    eval_feats = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # 1. Charlson comorbidities from diagnosis ICD-9 codes
    cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in cm_train:
        train_feats[k] = cm_train[k]
        eval_feats[k] = cm_eval[k]
        extra_numeric.append(k)

    # 2. ICD-9 Chapter groups
    for diag_col in ("diag_1", "diag_2", "diag_3"):
        grp_name = f"{diag_col}_group"
        train_feats[grp_name] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_feats[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(grp_name)

    # 3. High-cardinality Target Encodings
    te_candidates = (
        "diag_1",
        "diag_2",
        "diag_3",
        "medical_specialty",
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
    )
    for col in te_candidates:
        if col in train and col in eval_cols:
            te_tr, te_ev = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            te_name = f"{col}_te"
            train_feats[te_name] = te_tr
            eval_feats[te_name] = te_ev
            extra_numeric.append(te_name)

    # 4. Numeric Age (midpoint of age bracket)
    def parse_age(age_arr):
        res = np.empty(len(age_arr), dtype=np.float64)
        for i, a in enumerate(age_arr):
            if a is None or not isinstance(a, str):
                res[i] = np.nan
            else:
                clean = a.strip("[]()")
                parts = clean.split("-")
                if len(parts) == 2:
                    try:
                        res[i] = 0.5 * (float(parts[0]) + float(parts[1]))
                    except (ValueError, TypeError):
                        res[i] = np.nan
                else:
                    res[i] = np.nan
        return res

    train_feats["age_num"] = parse_age(train["age"])
    eval_feats["age_num"] = parse_age(eval_cols["age"])
    extra_numeric.append("age_num")

    # 5. Utilisation and Stay Intensity Features
    for d in (train_feats, eval_feats):
        inpat = d["number_inpatient"].astype(np.float64)
        emerg = d["number_emergency"].astype(np.float64)
        outpat = d["number_outpatient"].astype(np.float64)
        los = d["time_in_hospital"].astype(np.float64)
        n_labs = d["num_lab_procedures"].astype(np.float64)
        n_procs = d["num_procedures"].astype(np.float64)
        n_meds = d["num_medications"].astype(np.float64)
        n_diags = d["number_diagnoses"].astype(np.float64)

        total_prior = inpat + emerg + outpat
        d["total_prior_visits"] = total_prior
        d["inpatient_ratio"] = inpat / (total_prior + 1.0)
        d["emergency_ratio"] = emerg / (total_prior + 1.0)
        d["has_inpatient"] = (inpat > 0).astype(np.float64)
        d["has_emergency"] = (emerg > 0).astype(np.float64)
        d["has_prior_visits"] = (total_prior > 0).astype(np.float64)

        safe_los = np.maximum(los, 1.0)
        d["meds_per_day"] = n_meds / safe_los
        d["labs_per_day"] = n_labs / safe_los
        d["procs_per_day"] = n_procs / safe_los
        d["diags_per_day"] = n_diags / safe_los
        d["total_procedures"] = n_labs + n_procs

        d["inpat_x_los"] = inpat * los
        d["cm_count_x_inpat"] = d["cm_count"] * inpat

    util_features = [
        "total_prior_visits",
        "inpatient_ratio",
        "emergency_ratio",
        "has_inpatient",
        "has_emergency",
        "has_prior_visits",
        "meds_per_day",
        "labs_per_day",
        "procs_per_day",
        "diags_per_day",
        "total_procedures",
        "inpat_x_los",
        "cm_count_x_inpat",
    ]
    extra_numeric.extend(util_features)

    # 6. Medication Patterns and Dosing Changes
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    active_med_names = [c for c in med_names if c in train]

    for d in (train_feats, eval_feats):
        n = len(d["time_in_hospital"])
        num_active = np.zeros(n, dtype=np.float64)
        num_up = np.zeros(n, dtype=np.float64)
        num_down = np.zeros(n, dtype=np.float64)
        for c in active_med_names:
            arr = d[c]
            num_active += ((arr == "Steady") | (arr == "Up") | (arr == "Down")).astype(np.float64)
            num_up += (arr == "Up").astype(np.float64)
            num_down += (arr == "Down").astype(np.float64)

        d["num_active_meds"] = num_active
        d["num_meds_up"] = num_up
        d["num_meds_down"] = num_down
        d["num_med_changes"] = num_up + num_down

        ins = d.get("insulin", np.array([None] * n))
        d["ins_active"] = ((ins == "Steady") | (ins == "Up") | (ins == "Down")).astype(np.float64)
        d["ins_up"] = (ins == "Up").astype(np.float64)
        d["ins_down"] = (ins == "Down").astype(np.float64)
        d["ins_change"] = ((ins == "Up") | (ins == "Down")).astype(np.float64)

        chg = d.get("change", np.array([None] * n))
        d["change_flag"] = (chg == "Ch").astype(np.float64)

        dm = d.get("diabetesMed", np.array([None] * n))
        d["diabetes_med_flag"] = (dm == "Yes").astype(np.float64)

        d["payer_code_missing"] = np.array(
            [1.0 if x is None else 0.0 for x in d["payer_code"]], dtype=np.float64
        )
        d["specialty_missing"] = np.array(
            [1.0 if x is None else 0.0 for x in d["medical_specialty"]], dtype=np.float64
        )

    med_features = [
        "num_active_meds",
        "num_meds_up",
        "num_meds_down",
        "num_med_changes",
        "ins_active",
        "ins_up",
        "ins_down",
        "ins_change",
        "change_flag",
        "diabetes_med_flag",
        "payer_code_missing",
        "specialty_missing",
    ]
    extra_numeric.extend(med_features)

    # 7. Lab Results: A1C and Glucose Test Status
    for d in (train_feats, eval_feats):
        a1c = d["A1Cresult"]
        d["a1c_tested"] = np.array(
            [0.0 if (x is None or x == "None") else 1.0 for x in a1c], dtype=np.float64
        )
        d["a1c_high"] = np.array(
            [1.0 if x in (">7", ">8") else 0.0 for x in a1c], dtype=np.float64
        )

        glu = d["max_glu_serum"]
        d["glu_tested"] = np.array(
            [0.0 if (x is None or x == "None") else 1.0 for x in glu], dtype=np.float64
        )
        d["glu_high"] = np.array(
            [1.0 if x in (">200", ">300") else 0.0 for x in glu], dtype=np.float64
        )

    lab_features = ["a1c_tested", "a1c_high", "glu_tested", "glu_high"]
    extra_numeric.extend(lab_features)

    # Encode with TabularEncoder
    encoder = TabularEncoder(min_count=60)
    encoder.fit(
        task,
        train_feats,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)
    cat_indices = encoder.categorical_indices

    # Train an ensemble of diverse LightGBM models
    m1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=180,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.1,
        reg_alpha=0.01,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)

    m2 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.025,
        num_leaves=20,
        max_depth=6,
        min_child_samples=220,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=70.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)

    m3 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.018,
        num_leaves=40,
        min_child_samples=150,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.05,
        reg_alpha=0.001,
        cat_smooth=35.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)

    preds = (
        m1.predict_proba(x_eval)[:, 1]
        + m2.predict_proba(x_eval)[:, 1]
        + m3.predict_proba(x_eval)[:, 1]
    ) / 3.0

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
