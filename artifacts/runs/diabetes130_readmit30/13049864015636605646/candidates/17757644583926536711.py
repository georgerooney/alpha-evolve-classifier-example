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
        import lightgbm as lgb
    except ImportError:
        pass
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None

    _icd9_comorbidities = globals().get("icd9_comorbidities")
    _icd9_group = globals().get("icd9_group")
    _oof_target_encode = globals().get("oof_target_encode")

    train_data = dict(train)
    eval_data = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    def register_num(name: str, tr_arr: np.ndarray, ev_arr: np.ndarray):
        train_data[name] = np.asarray(tr_arr, dtype=np.float64)
        eval_data[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_numeric.append(name)

    # 1. Comorbidities
    if _icd9_comorbidities is not None:
        diag_cols_tr = [train_data[c] for c in ["diag_1", "diag_2", "diag_3"] if c in train_data]
        diag_cols_ev = [eval_data[c] for c in ["diag_1", "diag_2", "diag_3"] if c in eval_data]
        cm_tr = _icd9_comorbidities(*diag_cols_tr)
        cm_ev = _icd9_comorbidities(*diag_cols_ev)
        for k, v in cm_tr.items():
            register_num(k, v, cm_ev[k])

    # 2. ICD-9 Chapter Groups
    if _icd9_group is not None:
        for c in ["diag_1", "diag_2", "diag_3"]:
            if c in train_data:
                grp_name = f"{c}_group"
                train_data[grp_name] = np.array([_icd9_group(v) for v in train_data[c]], dtype=object)
                eval_data[grp_name] = np.array([_icd9_group(v) for v in eval_data[c]], dtype=object)
                extra_categorical.append(grp_name)

    # 3. Out-of-fold target encoding
    if _oof_target_encode is not None:
        te_cols = [
            "diag_1", "diag_2", "diag_3",
            "medical_specialty",
            "discharge_disposition_id",
            "admission_source_id",
            "admission_type_id",
        ]
        te_tr_map = {}
        te_ev_map = {}
        for c in te_cols:
            if c in train_data:
                tr_enc, ev_enc = _oof_target_encode(
                    train_data[c], y_train, eval_data[c],
                    n_folds=5, smoothing=25.0, seed=42
                )
                te_name = f"te_{c}"
                register_num(te_name, tr_enc, ev_enc)
                te_tr_map[c] = tr_enc
                te_ev_map[c] = ev_enc

        if "diag_1" in te_tr_map and "diag_2" in te_tr_map and "diag_3" in te_tr_map:
            tr_max_diag = np.maximum(te_tr_map["diag_1"], np.maximum(te_tr_map["diag_2"], te_tr_map["diag_3"]))
            ev_max_diag = np.maximum(te_ev_map["diag_1"], np.maximum(te_ev_map["diag_2"], te_ev_map["diag_3"]))
            register_num("te_diag_max", tr_max_diag, ev_max_diag)

    # 4. Prior Utilisation & Encounter Intensities
    num_inp_tr = train_data["number_inpatient"]
    num_inp_ev = eval_data["number_inpatient"]
    num_emg_tr = train_data["number_emergency"]
    num_emg_ev = eval_data["number_emergency"]
    num_out_tr = train_data["number_outpatient"]
    num_out_ev = eval_data["number_outpatient"]

    tot_visits_tr = num_inp_tr + num_emg_tr + num_out_tr
    tot_visits_ev = num_inp_ev + num_emg_ev + num_out_ev
    register_num("total_prior_visits", tot_visits_tr, tot_visits_ev)
    register_num("has_prior_inpatient", (num_inp_tr > 0).astype(np.float64), (num_inp_ev > 0).astype(np.float64))
    register_num("has_multiple_inpatient", (num_inp_tr >= 2).astype(np.float64), (num_inp_ev >= 2).astype(np.float64))
    register_num("has_prior_emergency", (num_emg_tr > 0).astype(np.float64), (num_emg_ev > 0).astype(np.float64))

    safe_tot_tr = np.maximum(tot_visits_tr, 1e-5)
    safe_tot_ev = np.maximum(tot_visits_ev, 1e-5)
    register_num("inpatient_ratio", np.where(tot_visits_tr > 0, num_inp_tr / safe_tot_tr, 0.0),
                                   np.where(tot_visits_ev > 0, num_inp_ev / safe_tot_ev, 0.0))
    register_num("emergency_ratio", np.where(tot_visits_tr > 0, num_emg_tr / safe_tot_tr, 0.0),
                                   np.where(tot_visits_ev > 0, num_emg_ev / safe_tot_ev, 0.0))

    # 5. In-Hospital Care Intensity & Ratios
    t_hosp_tr = train_data["time_in_hospital"]
    t_hosp_ev = eval_data["time_in_hospital"]
    n_med_tr = train_data["num_medications"]
    n_med_ev = eval_data["num_medications"]
    n_lab_tr = train_data["num_lab_procedures"]
    n_lab_ev = eval_data["num_lab_procedures"]
    n_proc_tr = train_data["num_procedures"]
    n_proc_ev = eval_data["num_procedures"]
    n_diag_tr = train_data["number_diagnoses"]
    n_diag_ev = eval_data["number_diagnoses"]

    register_num("meds_per_day", n_med_tr / (t_hosp_tr + 1e-5), n_med_ev / (t_hosp_ev + 1e-5))
    register_num("labs_per_day", n_lab_tr / (t_hosp_tr + 1e-5), n_lab_ev / (t_hosp_ev + 1e-5))
    register_num("procs_per_day", n_proc_tr / (t_hosp_tr + 1e-5), n_proc_ev / (t_hosp_ev + 1e-5))
    register_num("clinical_complexity",
                 n_med_tr + 2.0 * n_proc_tr + 0.3 * n_lab_tr + 2.0 * n_diag_tr,
                 n_med_ev + 2.0 * n_proc_ev + 0.3 * n_lab_ev + 2.0 * n_diag_ev)

    # 6. Medication Titration & Insulin Patterns
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    present_meds = [m for m in med_names if m in train_data]

    n_tr = len(y_train)
    n_ev = len(t_hosp_ev)
    num_active_tr = np.zeros(n_tr, dtype=np.float64)
    num_active_ev = np.zeros(n_ev, dtype=np.float64)
    num_up_tr = np.zeros(n_tr, dtype=np.float64)
    num_up_ev = np.zeros(n_ev, dtype=np.float64)
    num_down_tr = np.zeros(n_tr, dtype=np.float64)
    num_down_ev = np.zeros(n_ev, dtype=np.float64)

    for m in present_meds:
        tr_m = train_data[m]
        ev_m = eval_data[m]
        num_active_tr += (tr_m != "No") & (tr_m != None)
        num_active_ev += (ev_m != "No") & (ev_m != None)
        num_up_tr += (tr_m == "Up")
        num_up_ev += (ev_m == "Up")
        num_down_tr += (tr_m == "Down")
        num_down_ev += (ev_m == "Down")

    register_num("num_active_meds", num_active_tr, num_active_ev)
    register_num("num_meds_up", num_up_tr, num_up_ev)
    register_num("num_meds_down", num_down_tr, num_down_ev)
    register_num("num_meds_dosage_changed", num_up_tr + num_down_tr, num_up_ev + num_down_ev)

    if "insulin" in train_data:
        ins_tr = train_data["insulin"]
        ins_ev = eval_data["insulin"]
        register_num("on_insulin", ((ins_tr != "No") & (ins_tr != None)).astype(np.float64),
                                   ((ins_ev != "No") & (ins_ev != None)).astype(np.float64))
        register_num("insulin_up", (ins_tr == "Up").astype(np.float64), (ins_ev == "Up").astype(np.float64))
        register_num("insulin_down", (ins_tr == "Down").astype(np.float64), (ins_ev == "Down").astype(np.float64))
        register_num("insulin_steady", (ins_tr == "Steady").astype(np.float64), (ins_ev == "Steady").astype(np.float64))
        register_num("insulin_changed", ((ins_tr == "Up") | (ins_tr == "Down")).astype(np.float64),
                                        ((ins_ev == "Up") | (ins_ev == "Down")).astype(np.float64))

    if "change" in train_data:
        register_num("change_flag", (train_data["change"] == "Ch").astype(np.float64),
                                    (eval_data["change"] == "Ch").astype(np.float64))
    if "diabetesMed" in train_data:
        register_num("diabetes_med_flag", (train_data["diabetesMed"] == "Yes").astype(np.float64),
                                          (eval_data["diabetesMed"] == "Yes").astype(np.float64))

    # 7. Lab Results Indicators
    if "A1Cresult" in train_data:
        a1c_tr = train_data["A1Cresult"]
        a1c_ev = eval_data["A1Cresult"]
        register_num("a1c_tested", ((a1c_tr != "None") & (a1c_tr != None)).astype(np.float64),
                                   ((a1c_ev != "None") & (a1c_ev != None)).astype(np.float64))
        register_num("a1c_high", (a1c_tr == ">8").astype(np.float64), (a1c_ev == ">8").astype(np.float64))
        register_num("a1c_elevated", ((a1c_tr == ">7") | (a1c_tr == ">8")).astype(np.float64),
                                     ((a1c_ev == ">7") | (a1c_ev == ">8")).astype(np.float64))

    if "max_glu_serum" in train_data:
        glu_tr = train_data["max_glu_serum"]
        glu_ev = eval_data["max_glu_serum"]
        register_num("glu_tested", ((glu_tr != "None") & (glu_tr != None)).astype(np.float64),
                                   ((glu_ev != "None") & (glu_ev != None)).astype(np.float64))
        register_num("glu_high", ((glu_tr == ">200") | (glu_tr == ">300")).astype(np.float64),
                                 ((glu_ev == ">200") | (glu_ev == ">300")).astype(np.float64))

    # 8. Age Midpoint Mapping
    if "age" in train_data:
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        age_tr = np.array([age_map.get(str(x), np.nan) for x in train_data["age"]], dtype=np.float64)
        age_ev = np.array([age_map.get(str(x), np.nan) for x in eval_data["age"]], dtype=np.float64)
        register_num("age_numeric", age_tr, age_ev)
        register_num("is_elderly", (age_tr >= 75.0).astype(np.float64), (age_ev >= 75.0).astype(np.float64))

    # 9. Discharge Disposition Clinical Flags
    if "discharge_disposition_id" in train_data:
        disp_tr = train_data["discharge_disposition_id"]
        disp_ev = eval_data["discharge_disposition_id"]
        snf_codes = {"2", "3", "4", "5", "22", "23", "24"}
        register_num("discharged_home", (disp_tr == "1").astype(np.float64), (disp_ev == "1").astype(np.float64))
        register_num("discharged_facility",
                     np.array([1.0 if x in snf_codes else 0.0 for x in disp_tr], dtype=np.float64),
                     np.array([1.0 if x in snf_codes else 0.0 for x in disp_ev], dtype=np.float64))
        register_num("discharged_home_health", (disp_tr == "6").astype(np.float64), (disp_ev == "6").astype(np.float64))
        register_num("discharged_ama", (disp_tr == "7").astype(np.float64), (disp_ev == "7").astype(np.float64))

    # 10. Comorbidity & Utilisation Interactions
    if "cm_diabetes_complicated" in train_data:
        register_num("diab_comp_x_inpatient",
                     train_data["cm_diabetes_complicated"] * num_inp_tr,
                     eval_data["cm_diabetes_complicated"] * num_inp_ev)
    if "cm_count" in train_data:
        register_num("cm_count_x_inpatient",
                     train_data["cm_count"] * num_inp_tr,
                     eval_data["cm_count"] * num_inp_ev)

    # Encode tabular features
    encoder = TabularEncoder(min_count=35)
    encoder.fit(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # Model 1: Deep Leaf-Wise LightGBM
    model_lgb1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb1 = model_lgb1.predict_proba(x_eval)[:, 1]

    # Model 2: Depth-Constrained Regularized LightGBM
    model_lgb2 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.018,
        num_leaves=19,
        max_depth=5,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.2,
        reg_lambda=3.0,
        cat_smooth=30.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb2 = model_lgb2.predict_proba(x_eval)[:, 1]

    # Model 3: Depth-Wise XGBoost
    if xgb is not None:
        model_xgb = xgb.XGBClassifier(
            n_estimators=220,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=8,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        )
        model_xgb.fit(x_train, y_train)
        p_xgb = model_xgb.predict_proba(x_eval)[:, 1]
        p_eval = 0.50 * p_lgb1 + 0.30 * p_lgb2 + 0.20 * p_xgb
    else:
        p_eval = 0.60 * p_lgb1 + 0.40 * p_lgb2

    p_eval = np.clip(p_eval, 0.0, 1.0)
    if np.isnan(p_eval).any():
        p_eval = np.nan_to_num(p_eval, nan=float(np.mean(y_train)))

    return p_eval


# EVOLVE-BLOCK-END
# =====================================================================
