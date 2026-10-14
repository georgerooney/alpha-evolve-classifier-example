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
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities via icd9_comorbidities
    cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in cm_train:
        train_feat[k] = cm_train[k]
        eval_feat[k] = cm_eval[k]
        extra_numeric.append(k)

    # Cardiorenal syndrome interaction
    cr_train = (((cm_train["cm_chf"] == 1.0) | (cm_train["cm_mi"] == 1.0)) & (cm_train["cm_renal"] == 1.0)).astype(np.float64)
    cr_eval = (((cm_eval["cm_chf"] == 1.0) | (cm_eval["cm_mi"] == 1.0)) & (cm_eval["cm_renal"] == 1.0)).astype(np.float64)
    train_feat["cm_cardiorenal"] = cr_train
    eval_feat["cm_cardiorenal"] = cr_eval
    extra_numeric.append("cm_cardiorenal")

    # 2. ICD-9 Chapter groups
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        grp_col = f"{d_col}_grp"
        train_feat[grp_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_feat[grp_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(grp_col)

    # 3. Out-of-fold smoothed target encoding of high-cardinality codes
    te_cols = ["diag_1", "diag_2", "diag_3", "discharge_disposition_id", "admission_source_id", "medical_specialty"]
    for col in te_cols:
        te_name = f"{col}_te"
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
        train_feat[te_name] = tr_te.astype(np.float64)
        eval_feat[te_name] = ev_te.astype(np.float64)
        extra_numeric.append(te_name)

    # Aggregate diagnosis risk
    train_feat["diag_te_sum"] = train_feat["diag_1_te"] + train_feat["diag_2_te"] + train_feat["diag_3_te"]
    eval_feat["diag_te_sum"] = eval_feat["diag_1_te"] + eval_feat["diag_2_te"] + eval_feat["diag_3_te"]
    extra_numeric.append("diag_te_sum")

    train_feat["diag_te_max"] = np.maximum(np.maximum(train_feat["diag_1_te"], train_feat["diag_2_te"]), train_feat["diag_3_te"])
    eval_feat["diag_te_max"] = np.maximum(np.maximum(eval_feat["diag_1_te"], eval_feat["diag_2_te"]), eval_feat["diag_3_te"])
    extra_numeric.append("diag_te_max")

    # 4. Domain & Utilisation Feature Engineering Helper
    def add_domain_features(cols_dict, is_train=True):
        inpat = cols_dict["number_inpatient"].astype(np.float64)
        emg = cols_dict["number_emergency"].astype(np.float64)
        outpat = cols_dict["number_outpatient"].astype(np.float64)
        time_h = cols_dict["time_in_hospital"].astype(np.float64)
        num_med = cols_dict["num_medications"].astype(np.float64)
        num_lab = cols_dict["num_lab_procedures"].astype(np.float64)
        num_proc = cols_dict["num_procedures"].astype(np.float64)
        num_diag = cols_dict["number_diagnoses"].astype(np.float64)

        total_vis = inpat + emg + outpat
        has_inpat = (inpat > 0.0).astype(np.float64)
        has_emg = (emg > 0.0).astype(np.float64)
        freq_inpat = (inpat >= 2.0).astype(np.float64)
        inpat_ratio = inpat / (total_vis + 1.0)
        emg_ratio = emg / (total_vis + 1.0)

        encounter_int = (num_lab + num_proc + num_med) / (time_h + 1.0)
        meds_per_day = num_med / (time_h + 1.0)
        labs_per_day = num_lab / (time_h + 1.0)
        procs_per_day = num_proc / (time_h + 1.0)
        diags_per_day = num_diag / (time_h + 1.0)

        inpat_x_time = inpat * time_h
        inpat_x_meds = inpat * num_med

        # Age mapping to numeric
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0,
        }
        age_num = np.array([age_map.get(a, np.nan) for a in cols_dict["age"]], dtype=np.float64)
        is_elderly = (age_num >= 70.0).astype(np.float64)

        # Lab tests
        a1c = cols_dict["A1Cresult"]
        a1c_tested = (a1c != "None").astype(np.float64)
        a1c_high = np.isin(a1c, [">7", ">8"]).astype(np.float64)
        glu = cols_dict["max_glu_serum"]
        glu_tested = (glu != "None").astype(np.float64)
        glu_high = np.isin(glu, [">200", ">300"]).astype(np.float64)

        # Missingness indicators
        spec_missing = np.array([(x is None or x == "?") for x in cols_dict["medical_specialty"]], dtype=np.float64)
        payer_missing = np.array([(x is None or x == "?") for x in cols_dict["payer_code"]], dtype=np.float64)

        # Medication features
        med_names = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
        ]
        active_meds = np.zeros(len(time_h), dtype=np.float64)
        changed_meds = np.zeros(len(time_h), dtype=np.float64)
        for med in med_names:
            if med in cols_dict:
                m_arr = cols_dict[med]
                is_up = m_arr == "Up"
                is_down = m_arr == "Down"
                is_steady = m_arr == "Steady"
                active_meds += is_up | is_down | is_steady
                changed_meds += is_up | is_down

        ins = cols_dict["insulin"]
        insulin_on = (ins != "No").astype(np.float64)
        insulin_up = (ins == "Up").astype(np.float64)
        insulin_down = (ins == "Down").astype(np.float64)
        insulin_steady = (ins == "Steady").astype(np.float64)

        change_flag = (cols_dict["change"] == "Ch").astype(np.float64)
        diab_med = (cols_dict["diabetesMed"] == "Yes").astype(np.float64)

        features = {
            "total_visits": total_vis,
            "has_inpatient": has_inpat,
            "has_emergency": has_emg,
            "freq_inpatient": freq_inpat,
            "inpatient_ratio": inpat_ratio,
            "emergency_ratio": emg_ratio,
            "encounter_intensity": encounter_int,
            "meds_per_day": meds_per_day,
            "labs_per_day": labs_per_day,
            "procs_per_day": procs_per_day,
            "diags_per_day": diags_per_day,
            "inpat_x_time": inpat_x_time,
            "inpat_x_meds": inpat_x_meds,
            "age_num": age_num,
            "is_elderly": is_elderly,
            "a1c_tested": a1c_tested,
            "a1c_high": a1c_high,
            "glu_tested": glu_tested,
            "glu_high": glu_high,
            "spec_missing": spec_missing,
            "payer_missing": payer_missing,
            "active_meds_count": active_meds,
            "changed_meds_count": changed_meds,
            "insulin_on": insulin_on,
            "insulin_up": insulin_up,
            "insulin_down": insulin_down,
            "insulin_steady": insulin_steady,
            "change_flag": change_flag,
            "diab_med_flag": diab_med,
        }
        return features

    tr_domain = add_domain_features(train, is_train=True)
    ev_domain = add_domain_features(eval_cols, is_train=False)
    for fname in tr_domain:
        train_feat[fname] = tr_domain[fname]
        eval_feat[fname] = ev_domain[fname]
        extra_numeric.append(fname)

    # 5. Fit TabularEncoder with tuned min_count=25
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)
    cat_indices = encoder.categorical_indices

    # 6. Ensemble of diverse LightGBM models
    m1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=180,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.5,
        reg_alpha=0.1,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    m2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.018,
        num_leaves=18,
        max_depth=5,
        min_child_samples=250,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=2.0,
        reg_alpha=0.5,
        cat_smooth=60.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    m3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.022,
        num_leaves=45,
        min_child_samples=120,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=1.0,
        reg_alpha=0.2,
        cat_smooth=25.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # 7. XGBoost histogram model
    pxgb = None
    try:
        xgb_mod = globals().get("xgb") or globals().get("xgboost")
        if xgb_mod is not None:
            m_xgb = xgb_mod.XGBClassifier(
                n_estimators=220,
                learning_rate=0.025,
                max_depth=5,
                subsample=0.80,
                colsample_bytree=0.60,
                reg_alpha=0.2,
                reg_lambda=1.5,
                tree_method="hist",
                random_state=42,
                n_jobs=1,
                eval_metric="logloss",
            )
            m_xgb.fit(x_train, y_train)
            pxgb = m_xgb.predict_proba(x_eval)[:, 1]
    except Exception:
        pxgb = None

    if pxgb is not None:
        preds = 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * pxgb
    else:
        preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3

    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
