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
    # 1. Resolve toolkit functions
    try:
        from clinical_toolkit import (
            TabularEncoder,
            icd9_group,
            icd9_comorbidities,
            oof_target_encode,
        )
    except Exception:
        TabularEncoder = globals().get("TabularEncoder")
        icd9_group = globals().get("icd9_group")
        icd9_comorbidities = globals().get("icd9_comorbidities")
        oof_target_encode = globals().get("oof_target_encode")

    trn_dict = dict(train)
    evl_dict = dict(eval_cols)
    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])

    extra_num_cols = []
    extra_cat_cols = []

    # 2. Charlson Comorbidities Mapping
    if icd9_comorbidities is not None:
        cm_trn = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_evl = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        for k in cm_trn:
            trn_dict[k] = cm_trn[k].astype(np.float64)
            evl_dict[k] = cm_evl[k].astype(np.float64)
            extra_num_cols.append(k)
        cm_count_tr = cm_trn["cm_count"]
        cm_count_ev = cm_evl["cm_count"]
    else:
        cm_count_tr = np.zeros(n_tr, dtype=np.float64)
        cm_count_ev = np.zeros(n_ev, dtype=np.float64)

    # 3. ICD-9 Chapter Groups & 3-Digit Categories
    if icd9_group is not None:
        trn_g1 = np.array([icd9_group(c) for c in train["diag_1"]], dtype=object)
        evl_g1 = np.array([icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
        trn_g2 = np.array([icd9_group(c) for c in train["diag_2"]], dtype=object)
        evl_g2 = np.array([icd9_group(c) for c in eval_cols["diag_2"]], dtype=object)
        trn_g3 = np.array([icd9_group(c) for c in train["diag_3"]], dtype=object)
        evl_g3 = np.array([icd9_group(c) for c in eval_cols["diag_3"]], dtype=object)

        trn_dict["diag_1_group"] = trn_g1
        evl_dict["diag_1_group"] = evl_g1
        trn_dict["diag_2_group"] = trn_g2
        evl_dict["diag_2_group"] = evl_g2
        trn_dict["diag_3_group"] = trn_g3
        evl_dict["diag_3_group"] = evl_g3

        trn_pair = np.array([f"{g1}_{g2}" for g1, g2 in zip(trn_g1, trn_g2)], dtype=object)
        evl_pair = np.array([f"{g1}_{g2}" for g1, g2 in zip(evl_g1, evl_g2)], dtype=object)
        trn_dict["diag_pair"] = trn_pair
        evl_dict["diag_pair"] = evl_pair
        extra_cat_cols.extend(["diag_1_group", "diag_2_group", "diag_3_group", "diag_pair"])

    def _icd3(codes):
        out = np.empty(len(codes), dtype=object)
        for i, c in enumerate(codes):
            if c is None or c == "?" or not str(c).strip():
                out[i] = "missing"
            else:
                out[i] = str(c).strip().split(".")[0]
        return out

    trn_d1_cat = _icd3(train["diag_1"])
    evl_d1_cat = _icd3(eval_cols["diag_1"])
    trn_dict["diag_1_cat"] = trn_d1_cat
    evl_dict["diag_1_cat"] = evl_d1_cat
    extra_cat_cols.append("diag_1_cat")

    # 4. Out-of-Fold Smoothed Target Encodings
    if oof_target_encode is not None:
        te_targets = [
            ("discharge_disposition_id", 20.0),
            ("admission_source_id", 20.0),
            ("medical_specialty", 30.0),
            ("diag_1_cat", 25.0),
        ]
        if "diag_pair" in trn_dict:
            te_targets.append(("diag_pair", 20.0))

        for col_name, smooth in te_targets:
            te_col = f"te_{col_name}"
            trn_enc, evl_enc = oof_target_encode(
                trn_dict[col_name], y_train, evl_dict[col_name], smoothing=smooth, seed=42
            )
            trn_dict[te_col] = trn_enc.astype(np.float64)
            evl_dict[te_col] = evl_enc.astype(np.float64)
            extra_num_cols.append(te_col)

    # 5. Prior Healthcare Utilization Trajectory
    trn_inp = np.nan_to_num(np.asarray(train["number_inpatient"], dtype=np.float64), nan=0.0)
    evl_inp = np.nan_to_num(np.asarray(eval_cols["number_inpatient"], dtype=np.float64), nan=0.0)
    trn_emg = np.nan_to_num(np.asarray(train["number_emergency"], dtype=np.float64), nan=0.0)
    evl_emg = np.nan_to_num(np.asarray(eval_cols["number_emergency"], dtype=np.float64), nan=0.0)
    trn_out = np.nan_to_num(np.asarray(train["number_outpatient"], dtype=np.float64), nan=0.0)
    evl_out = np.nan_to_num(np.asarray(eval_cols["number_outpatient"], dtype=np.float64), nan=0.0)

    trn_tot = trn_inp + trn_emg + trn_out
    evl_tot = evl_inp + evl_emg + evl_out

    trn_dict["has_prior_inpatient"] = (trn_inp > 0.0).astype(np.float64)
    evl_dict["has_prior_inpatient"] = (evl_inp > 0.0).astype(np.float64)
    trn_dict["high_prior_inpatient"] = (trn_inp >= 2.0).astype(np.float64)
    evl_dict["high_prior_inpatient"] = (evl_inp >= 2.0).astype(np.float64)
    trn_dict["total_prior_visits"] = trn_tot
    evl_dict["total_prior_visits"] = evl_tot
    trn_dict["inpatient_ratio"] = trn_inp / (trn_tot + 1e-4)
    evl_dict["inpatient_ratio"] = evl_inp / (evl_tot + 1e-4)
    trn_dict["ambulatory_vulnerability"] = trn_emg / (trn_out + 1.0)
    evl_dict["ambulatory_vulnerability"] = evl_emg / (evl_out + 1.0)
    trn_dict["log_inpatient"] = np.log1p(trn_inp)
    evl_dict["log_inpatient"] = np.log1p(evl_inp)
    trn_dict["log_total_visits"] = np.log1p(trn_tot)
    evl_dict["log_total_visits"] = np.log1p(evl_tot)

    extra_num_cols.extend([
        "has_prior_inpatient",
        "high_prior_inpatient",
        "total_prior_visits",
        "inpatient_ratio",
        "ambulatory_vulnerability",
        "log_inpatient",
        "log_total_visits",
    ])

    if "te_discharge_disposition_id" in trn_dict:
        trn_dict["prior_inp_x_te_discharge"] = trn_dict["log_inpatient"] * trn_dict["te_discharge_disposition_id"]
        evl_dict["prior_inp_x_te_discharge"] = evl_dict["log_inpatient"] * evl_dict["te_discharge_disposition_id"]
        extra_num_cols.append("prior_inp_x_te_discharge")

    # 6. Inpatient Care Density & Diagnostic Intensity
    trn_hosp = np.maximum(1.0, np.nan_to_num(np.asarray(train["time_in_hospital"], dtype=np.float64), nan=1.0))
    evl_hosp = np.maximum(1.0, np.nan_to_num(np.asarray(eval_cols["time_in_hospital"], dtype=np.float64), nan=1.0))
    trn_labs = np.nan_to_num(np.asarray(train["num_lab_procedures"], dtype=np.float64), nan=0.0)
    evl_labs = np.nan_to_num(np.asarray(eval_cols["num_lab_procedures"], dtype=np.float64), nan=0.0)
    trn_meds = np.nan_to_num(np.asarray(train["num_medications"], dtype=np.float64), nan=0.0)
    evl_meds = np.nan_to_num(np.asarray(eval_cols["num_medications"], dtype=np.float64), nan=0.0)
    trn_proc = np.nan_to_num(np.asarray(train["num_procedures"], dtype=np.float64), nan=0.0)
    evl_proc = np.nan_to_num(np.asarray(eval_cols["num_procedures"], dtype=np.float64), nan=0.0)
    trn_diag = np.maximum(1.0, np.nan_to_num(np.asarray(train["number_diagnoses"], dtype=np.float64), nan=1.0))
    evl_diag = np.maximum(1.0, np.nan_to_num(np.asarray(eval_cols["number_diagnoses"], dtype=np.float64), nan=1.0))

    trn_dict["labs_per_day"] = trn_labs / trn_hosp
    evl_dict["labs_per_day"] = evl_labs / evl_hosp
    trn_dict["meds_per_day"] = trn_meds / trn_hosp
    evl_dict["meds_per_day"] = evl_meds / evl_hosp
    trn_dict["procs_per_day"] = trn_proc / trn_hosp
    evl_dict["procs_per_day"] = evl_proc / evl_hosp
    trn_dict["meds_per_diag"] = trn_meds / trn_diag
    evl_dict["meds_per_diag"] = evl_meds / evl_diag
    extra_num_cols.extend(["labs_per_day", "meds_per_day", "procs_per_day", "meds_per_diag"])

    # 7. Demographics & Multimorbidity Frailty
    def _parse_age_mid(arr):
        out = np.full(len(arr), np.nan, dtype=np.float64)
        for i, v in enumerate(arr):
            if v is not None and isinstance(v, str):
                s = v.strip("[]() ")
                parts = s.split("-")
                if len(parts) == 2:
                    try:
                        out[i] = (float(parts[0]) + float(parts[1])) / 2.0
                    except ValueError:
                        pass
        return out

    trn_age = _parse_age_mid(train.get("age", []))
    evl_age = _parse_age_mid(eval_cols.get("age", []))
    trn_dict["age_num"] = trn_age
    evl_dict["age_num"] = evl_age
    extra_num_cols.append("age_num")

    trn_dict["frailty_index"] = (
        cm_count_tr
        + (np.nan_to_num(trn_age, nan=50.0) > 70.0).astype(np.float64) * 1.5
        + (trn_inp > 0.0).astype(np.float64) * 2.0
    )
    evl_dict["frailty_index"] = (
        cm_count_ev
        + (np.nan_to_num(evl_age, nan=50.0) > 70.0).astype(np.float64) * 1.5
        + (evl_inp > 0.0).astype(np.float64) * 2.0
    )
    extra_num_cols.append("frailty_index")

    # 8. Medication Regimen Dynamics & Testing Discordance
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]

    def _summarize_meds(src, size):
        active = np.zeros(size, dtype=np.float64)
        changed = np.zeros(size, dtype=np.float64)
        for m in med_cols:
            if m in src:
                arr = src[m]
                for i in range(size):
                    val = arr[i]
                    if val is not None and val != "No":
                        active[i] += 1.0
                        if val in ("Up", "Down"):
                            changed[i] += 1.0
        return active, changed

    trn_act_med, trn_chg_med = _summarize_meds(train, n_tr)
    evl_act_med, evl_chg_med = _summarize_meds(eval_cols, n_ev)
    trn_dict["med_active_count"] = trn_act_med
    evl_dict["med_active_count"] = evl_act_med
    trn_dict["med_change_count"] = trn_chg_med
    evl_dict["med_change_count"] = evl_chg_med

    ins_tr = train.get("insulin", np.array([None] * n_tr))
    ins_ev = eval_cols.get("insulin", np.array([None] * n_ev))
    trn_ins_chg = np.array([1.0 if v in ("Up", "Down") else 0.0 for v in ins_tr], dtype=np.float64)
    evl_ins_chg = np.array([1.0 if v in ("Up", "Down") else 0.0 for v in ins_ev], dtype=np.float64)
    trn_dict["insulin_change"] = trn_ins_chg
    evl_dict["insulin_change"] = evl_ins_chg
    trn_dict["insulin_active"] = np.array([1.0 if (v is not None and v != "No") else 0.0 for v in ins_tr], dtype=np.float64)
    evl_dict["insulin_active"] = np.array([1.0 if (v is not None and v != "No") else 0.0 for v in ins_ev], dtype=np.float64)

    a1c_tr = train.get("A1Cresult", np.array([None] * n_tr))
    a1c_ev = eval_cols.get("A1Cresult", np.array([None] * n_ev))
    trn_a1c_tested = np.array([0.0 if (v is None or v == "None") else 1.0 for v in a1c_tr], dtype=np.float64)
    evl_a1c_tested = np.array([0.0 if (v is None or v == "None") else 1.0 for v in a1c_ev], dtype=np.float64)
    trn_dict["a1c_tested"] = trn_a1c_tested
    evl_dict["a1c_tested"] = evl_a1c_tested
    trn_dict["a1c_high"] = np.array([1.0 if v == ">8" else 0.0 for v in a1c_tr], dtype=np.float64)
    evl_dict["a1c_high"] = np.array([1.0 if v == ">8" else 0.0 for v in a1c_ev], dtype=np.float64)

    # Volatility discordance: insulin titration without A1C verification
    trn_dict["unmonitored_insulin_change"] = trn_ins_chg * (1.0 - trn_a1c_tested)
    evl_dict["unmonitored_insulin_change"] = evl_ins_chg * (1.0 - evl_a1c_tested)

    extra_num_cols.extend([
        "med_active_count",
        "med_change_count",
        "insulin_change",
        "insulin_active",
        "a1c_tested",
        "a1c_high",
        "unmonitored_insulin_change",
    ])

    # 9. Encode Features
    encoder = TabularEncoder(min_count=25)
    x_train = encoder.fit_transform(
        task,
        trn_dict,
        extra_numeric=extra_num_cols,
        extra_categorical=extra_cat_cols,
    )
    x_eval = encoder.transform(evl_dict)

    # 10. Fit Heterogeneous Ensemble
    # Model 1: Primary Leaf-Wise LightGBM
    lgb1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        reg_lambda=1.0,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_lgb1 = lgb1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative Regularized LightGBM
    lgb2 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.02,
        num_leaves=16,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.5,
        reg_lambda=3.0,
        cat_smooth=60.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    lgb2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_lgb2 = lgb2.predict_proba(x_eval)[:, 1]

    # Model 3: Depth-Wise Histogram XGBoost
    pred_xgb = None
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        xgb_clf.fit(x_train, y_train)
        pred_xgb = xgb_clf.predict_proba(x_eval)[:, 1]
    except Exception:
        pass

    if pred_xgb is not None:
        p_final = 0.50 * pred_lgb1 + 0.25 * pred_lgb2 + 0.25 * pred_xgb
    else:
        p_final = 0.65 * pred_lgb1 + 0.35 * pred_lgb2

    p_final = np.clip(p_final, 0.0, 1.0)
    if np.isnan(p_final).any():
        p_final = np.nan_to_num(p_final, nan=float(np.mean(y_train)))

    return p_final


# EVOLVE-BLOCK-END
# =====================================================================
