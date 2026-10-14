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
    # Retrieve frozen toolkit helpers
    try:
        from clinical_toolkit import (
            TabularEncoder,
            icd9_comorbidities,
            icd9_group,
            oof_target_encode,
        )
    except Exception:
        TabularEncoder = globals().get("TabularEncoder")
        icd9_comorbidities = globals().get("icd9_comorbidities")
        icd9_group = globals().get("icd9_group")
        oof_target_encode = globals().get("oof_target_encode")

    def extract_engineered_features(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        n = len(data_dict["time_in_hospital"])

        # Prior visits
        n_inpatient = np.nan_to_num(data_dict.get("number_inpatient", np.zeros(n)), nan=0.0)
        n_emergency = np.nan_to_num(data_dict.get("number_emergency", np.zeros(n)), nan=0.0)
        n_outpatient = np.nan_to_num(data_dict.get("number_outpatient", np.zeros(n)), nan=0.0)

        tot_visits = n_inpatient + n_emergency + n_outpatient
        res["total_prior_visits"] = tot_visits
        res["acute_prior_visits"] = n_inpatient + n_emergency
        res["has_prior_inpatient"] = (n_inpatient > 0).astype(np.float64)
        res["inpatient_ratio"] = (n_inpatient + 0.1) / (tot_visits + 1.0)

        # Inpatient stay intensity
        los = np.nan_to_num(data_dict.get("time_in_hospital", np.ones(n)), nan=1.0)
        n_meds = np.nan_to_num(data_dict.get("num_medications", np.zeros(n)), nan=0.0)
        n_labs = np.nan_to_num(data_dict.get("num_lab_procedures", np.zeros(n)), nan=0.0)
        n_procs = np.nan_to_num(data_dict.get("num_procedures", np.zeros(n)), nan=0.0)
        n_diags = np.nan_to_num(data_dict.get("number_diagnoses", np.ones(n)), nan=1.0)

        res["med_per_day"] = n_meds / (los + 0.5)
        res["lab_per_day"] = n_labs / (los + 0.5)
        res["proc_per_day"] = n_procs / (los + 0.5)
        res["meds_per_diag"] = n_meds / (n_diags + 0.5)
        res["los_times_inpatient"] = los * (n_inpatient + 1.0)

        # Age conversion
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0,
        }
        if "age" in data_dict:
            age_arr = data_dict["age"]
            age_num = np.array([age_map.get(a, 65.0) for a in age_arr], dtype=np.float64)
            res["age_num"] = age_num
            res["age_times_inpatient"] = age_num * (n_inpatient + 1.0)

        # Medication summary
        med_list = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]
        num_act = np.zeros(n, dtype=np.float64)
        num_up = np.zeros(n, dtype=np.float64)
        num_down = np.zeros(n, dtype=np.float64)

        for m in med_list:
            if m in data_dict:
                arr = data_dict[m]
                num_act += ((arr != "No") & (arr != None)).astype(np.float64)
                num_up += (arr == "Up").astype(np.float64)
                num_down += (arr == "Down").astype(np.float64)

        res["med_num_active"] = num_act
        res["med_num_up"] = num_up
        res["med_num_down"] = num_down
        res["med_num_changed"] = num_up + num_down

        if "insulin" in data_dict:
            ins = data_dict["insulin"]
            ins_act = ((ins != "No") & (ins != None)).astype(np.float64)
            res["insulin_active"] = ins_act
            res["insulin_up"] = (ins == "Up").astype(np.float64)
            res["insulin_down"] = (ins == "Down").astype(np.float64)
            res["insulin_plus_oral"] = ((ins_act > 0) & ((num_act - ins_act) > 0)).astype(np.float64)

        # Labs
        if "A1Cresult" in data_dict:
            a1c = data_dict["A1Cresult"]
            res["a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
            res["a1c_high"] = np.isin(a1c, [">8", ">7"]).astype(np.float64)
        if "max_glu_serum" in data_dict:
            glu = data_dict["max_glu_serum"]
            res["glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
            res["glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

        return res

    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities
    if icd9_comorbidities is not None and "diag_1" in train and "diag_2" in train and "diag_3" in train:
        cm_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        for k in cm_tr:
            train_dict[k] = cm_tr[k]
            eval_dict[k] = cm_ev[k]
            extra_numeric.append(k)

    # 2. ICD-9 Chapter groups
    if icd9_group is not None and "diag_1" in train and "diag_2" in train and "diag_3" in train:
        for d_col in ["diag_1", "diag_2", "diag_3"]:
            g_name = f"{d_col}_group"
            train_dict[g_name] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
            eval_dict[g_name] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
            extra_categorical.append(g_name)

    # 3. Domain features
    dom_tr = extract_engineered_features(train)
    dom_ev = extract_engineered_features(eval_cols)
    for k in dom_tr:
        train_dict[k] = dom_tr[k]
        eval_dict[k] = dom_ev[k]
        extra_numeric.append(k)

    # 4. Out-of-fold target encoding
    te_specs = [
        ("diag_1", 30.0),
        ("diag_2", 30.0),
        ("diag_3", 30.0),
        ("discharge_disposition_id", 20.0),
        ("admission_source_id", 20.0),
        ("admission_type_id", 20.0),
        ("medical_specialty", 25.0),
        ("payer_code", 25.0),
    ]
    if oof_target_encode is not None:
        for col_n, sm in te_specs:
            if col_n in train and col_n in eval_cols:
                tr_te, ev_te = oof_target_encode(
                    train[col_n], y_train, eval_cols[col_n], n_folds=5, smoothing=sm, seed=42
                )
                te_k = f"te_{col_n}"
                train_dict[te_k] = tr_te
                eval_dict[te_k] = ev_te
                extra_numeric.append(te_k)

        if "te_diag_1" in train_dict and "te_diag_2" in train_dict and "te_diag_3" in train_dict:
            train_dict["te_diag_max"] = np.maximum(
                train_dict["te_diag_1"], np.maximum(train_dict["te_diag_2"], train_dict["te_diag_3"])
            )
            eval_dict["te_diag_max"] = np.maximum(
                eval_dict["te_diag_1"], np.maximum(eval_dict["te_diag_2"], eval_dict["te_diag_3"])
            )
            extra_numeric.append("te_diag_max")

            train_dict["te_diag_mean"] = (
                train_dict["te_diag_1"] + train_dict["te_diag_2"] + train_dict["te_diag_3"]
            ) / 3.0
            eval_dict["te_diag_mean"] = (
                eval_dict["te_diag_1"] + eval_dict["te_diag_2"] + eval_dict["te_diag_3"]
            ) / 3.0
            extra_numeric.append("te_diag_mean")

    # Fit tabular encoder
    enc_cls = TabularEncoder if TabularEncoder is not None else globals().get("TabularEncoder")
    encoder = enc_cls(min_count=25)
    encoder.fit(task, train_dict, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)
    cat_idx = encoder.categorical_indices

    # Low-rank multi-scale ensemble:
    # Restricts interaction order via bounded tree depth (rank 2 at depth 4, rank 3 at depth 5, rank 4 at depth 6)
    # and enforces strong L2 shrinkage and min_child_samples to prevent fitting high-rank noise.
    models = [
        # Scale 1: Low-rank additive model (focuses on robust main & pairwise effects)
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.025,
            max_depth=4,
            num_leaves=15,
            min_child_samples=160,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=1.5,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Scale 2: Balanced interaction model (captures non-linear pairwise moderation)
        lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.022,
            max_depth=5,
            num_leaves=22,
            min_child_samples=180,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=2.0,
            cat_smooth=50.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        # Scale 3: Bounded depth model with higher sample support
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.020,
            max_depth=6,
            num_leaves=28,
            min_child_samples=220,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.0,
            cat_smooth=60.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(eval_cols["time_in_hospital"]), dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=cat_idx)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
