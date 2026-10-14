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
    """Multi-Scale Clinical Regimen Dynamics, Comorbidity Resonance & Heterogeneous GBDT Ensemble."""
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    n_train = len(y_train)
    n_eval = len(eval_cols["time_in_hospital"])

    extra_numeric = []
    extra_categorical = []

    # 1. Continuous Numerical Representation of Age Band Midpoints
    age_map = {
        "[0-10)": 5.0,
        "[10-20)": 15.0,
        "[20-30)": 25.0,
        "[30-40)": 35.0,
        "[40-50)": 45.0,
        "[50-60)": 55.0,
        "[60-70)": 65.0,
        "[70-80)": 75.0,
        "[80-90)": 85.0,
        "[90-100)": 95.0,
    }
    train_dict["age_numeric"] = np.array([age_map.get(str(x), np.nan) for x in train["age"]], dtype=np.float64)
    eval_dict["age_numeric"] = np.array([age_map.get(str(x), np.nan) for x in eval_cols["age"]], dtype=np.float64)
    extra_numeric.append("age_numeric")

    # 2. Charlson Comorbidity Matrix & Organ System Syndrome Dynamics
    train_cms = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    eval_cms = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

    cci_weights = {
        "cm_mi": 1.0,
        "cm_chf": 1.0,
        "cm_pvd": 1.0,
        "cm_cerebrovascular": 1.0,
        "cm_dementia": 1.0,
        "cm_pulmonary": 1.0,
        "cm_rheumatic": 1.0,
        "cm_peptic_ulcer": 1.0,
        "cm_liver": 1.0,
        "cm_diabetes_complicated": 2.0,
        "cm_hemiplegia": 2.0,
        "cm_renal": 2.0,
        "cm_cancer": 2.0,
        "cm_metastatic": 6.0,
        "cm_hiv": 6.0,
    }

    train_cci = np.zeros(n_train, dtype=np.float64)
    eval_cci = np.zeros(n_eval, dtype=np.float64)
    for cm_name, arr in train_cms.items():
        train_dict[cm_name] = arr
        eval_dict[cm_name] = eval_cms[cm_name]
        extra_numeric.append(cm_name)
        if cm_name in cci_weights:
            w = cci_weights[cm_name]
            train_cci += arr * w
            eval_cci += eval_cms[cm_name] * w

    train_dict["charlson_score"] = train_cci
    eval_dict["charlson_score"] = eval_cci
    extra_numeric.append("charlson_score")

    # Cardiorenal & vascular interaction syndromes
    train_dict["cardiorenal_synergy"] = (train_cms["cm_chf"] * train_cms["cm_renal"]).astype(np.float64)
    eval_dict["cardiorenal_synergy"] = (eval_cms["cm_chf"] * eval_cms["cm_renal"]).astype(np.float64)
    train_dict["vascular_burden"] = (train_cms["cm_mi"] + train_cms["cm_pvd"] + train_cms["cm_cerebrovascular"]).astype(np.float64)
    eval_dict["vascular_burden"] = (eval_cms["cm_mi"] + eval_cms["cm_pvd"] + eval_cms["cm_cerebrovascular"]).astype(np.float64)
    extra_numeric.extend(["cardiorenal_synergy", "vascular_burden"])

    # 3. ICD-9 Chapter Topology & Multi-Organ Chapter Entropy
    def compute_chapter_features(d1, d2, d3):
        n = len(d1)
        g1 = np.array([icd9_group(c) for c in d1], dtype=object)
        g2 = np.array([icd9_group(c) for c in d2], dtype=object)
        g3 = np.array([icd9_group(c) for c in d3], dtype=object)

        unique_count = np.zeros(n, dtype=np.float64)
        has_circ = np.zeros(n, dtype=np.float64)
        has_resp = np.zeros(n, dtype=np.float64)
        has_dm = np.zeros(n, dtype=np.float64)
        pair = np.empty(n, dtype=object)

        for i in range(n):
            s = {g1[i], g2[i], g3[i]}
            s.discard("missing")
            unique_count[i] = len(s)
            has_circ[i] = 1.0 if "circulatory" in s else 0.0
            has_resp[i] = 1.0 if "respiratory" in s else 0.0
            has_dm[i] = 1.0 if "diabetes" in s else 0.0
            pair[i] = f"{g1[i]}_{g2[i]}"

        return g1, pair, unique_count, has_circ, has_resp, has_dm

    tr_g1, tr_pair, tr_uniq, tr_circ, tr_resp, tr_dm = compute_chapter_features(
        train["diag_1"], train["diag_2"], train["diag_3"]
    )
    ev_g1, ev_pair, ev_uniq, ev_circ, ev_resp, ev_dm = compute_chapter_features(
        eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"]
    )

    train_dict["diag1_group"] = tr_g1
    eval_dict["diag1_group"] = ev_g1
    train_dict["diag1_diag2_pair"] = tr_pair
    eval_dict["diag1_diag2_pair"] = ev_pair
    extra_categorical.extend(["diag1_group", "diag1_diag2_pair"])

    train_dict["unique_diag_chapters"] = tr_uniq
    eval_dict["unique_diag_chapters"] = ev_uniq
    train_dict["has_circulatory_diag"] = tr_circ
    eval_dict["has_circulatory_diag"] = ev_circ
    train_dict["has_respiratory_diag"] = tr_resp
    eval_dict["has_respiratory_diag"] = ev_resp
    train_dict["has_diabetes_diag"] = tr_dm
    eval_dict["has_diabetes_diag"] = ev_dm
    extra_numeric.extend([
        "unique_diag_chapters",
        "has_circulatory_diag",
        "has_respiratory_diag",
        "has_diabetes_diag",
    ])

    # 4. Pharmacological Regimen Dynamics & Titration Instability
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citagliptin", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def extract_med_dynamics(cols_dict, n):
        n_active = np.zeros(n, dtype=np.float64)
        n_up = np.zeros(n, dtype=np.float64)
        n_down = np.zeros(n, dtype=np.float64)
        n_steady = np.zeros(n, dtype=np.float64)
        insulin_state = np.zeros(n, dtype=np.float64)
        n_orals = np.zeros(n, dtype=np.float64)

        for med in med_cols:
            if med not in cols_dict:
                continue
            arr = cols_dict[med]
            is_insulin = (med == "insulin")
            for i in range(n):
                val = arr[i]
                if val is not None and val != "No":
                    n_active[i] += 1.0
                    if is_insulin:
                        if val == "Steady":
                            insulin_state[i] = 1.0
                        elif val == "Down":
                            insulin_state[i] = 2.0
                        elif val == "Up":
                            insulin_state[i] = 3.0
                    else:
                        n_orals[i] += 1.0

                    if val == "Up":
                        n_up[i] += 1.0
                    elif val == "Down":
                        n_down[i] += 1.0
                    elif val == "Steady":
                        n_steady[i] += 1.0

        med_volatility = n_up + n_down
        med_titration_bias = n_up - n_down
        insulin_plus_orals = ((insulin_state > 0) & (n_orals > 0)).astype(np.float64)
        insulin_failure = ((insulin_state > 0) & (n_orals >= 2)).astype(np.float64)
        ch_arr = cols_dict.get("change")
        ch_flag = np.array([1.0 if x == "Ch" else 0.0 for x in ch_arr], dtype=np.float64)

        return {
            "num_active_meds": n_active,
            "num_meds_up": n_up,
            "num_meds_down": n_down,
            "num_meds_steady": n_steady,
            "med_volatility": med_volatility,
            "med_titration_bias": med_titration_bias,
            "insulin_state": insulin_state,
            "insulin_plus_orals": insulin_plus_orals,
            "insulin_failure": insulin_failure,
            "med_change_flag": ch_flag,
        }

    tr_meds = extract_med_dynamics(train, n_train)
    ev_meds = extract_med_dynamics(eval_cols, n_eval)
    for k, v in tr_meds.items():
        train_dict[k] = v
        eval_dict[k] = ev_meds[k]
        extra_numeric.append(k)

    # 5. Acute Care Gravitational Velocity & Daily Intensity Rates
    def extract_care_velocity(cols_dict):
        inp = cols_dict["number_inpatient"]
        emg = cols_dict["number_emergency"]
        out = cols_dict["number_outpatient"]
        hosp_days = cols_dict["time_in_hospital"]
        n_labs = cols_dict["num_lab_procedures"]
        n_procs = cols_dict["num_procedures"]
        n_meds = cols_dict["num_medications"]
        n_diags = cols_dict["number_diagnoses"]

        total_prior = inp + emg + out
        has_prior_inp = (inp > 0).astype(np.float64)
        high_prior_inp = (inp >= 2).astype(np.float64)
        has_prior_emg = (emg > 0).astype(np.float64)
        acute_velocity = np.log1p(inp) * 2.5 + np.log1p(emg) * 1.5 + np.log1p(out) * 0.5
        inp_ratio = inp / (total_prior + 1.0)

        lab_rate = n_labs / (hosp_days + 0.5)
        med_rate = n_meds / (hosp_days + 0.5)
        proc_rate = n_procs / (hosp_days + 0.5)
        diag_rate = n_diags / (hosp_days + 0.5)
        complexity_vol = np.log1p(hosp_days * n_meds * (n_diags + 1.0))

        return {
            "total_prior_visits": total_prior,
            "has_prior_inpatient": has_prior_inp,
            "high_prior_inpatient": high_prior_inp,
            "has_prior_emergency": has_prior_emg,
            "acute_care_velocity": acute_velocity,
            "inpatient_util_ratio": inp_ratio,
            "lab_intensity_per_day": lab_rate,
            "med_intensity_per_day": med_rate,
            "proc_intensity_per_day": proc_rate,
            "diag_intensity_per_day": diag_rate,
            "encounter_complexity_volume": complexity_vol,
        }

    tr_vel = extract_care_velocity(train)
    ev_vel = extract_care_velocity(eval_cols)
    for k, v in tr_vel.items():
        train_dict[k] = v
        eval_dict[k] = ev_vel[k]
        extra_numeric.append(k)

    # 6. Discharge Transition Friction & Vulnerability Markers
    def extract_transition_friction(cols_dict, n):
        disch = cols_dict["discharge_disposition_id"]
        admiss = cols_dict["admission_source_id"]
        inp = cols_dict["number_inpatient"]

        cross_disch_src = np.array([f"{disch[i]}_{admiss[i]}" for i in range(n)], dtype=object)
        is_home = np.array([1.0 if disch[i] == "1" else 0.0 for i in range(n)], dtype=np.float64)
        home_high_risk = is_home * (inp >= 2).astype(np.float64)
        vulnerable_dest = np.array([1.0 if disch[i] in {"3", "4", "7", "22"} else 0.0 for i in range(n)], dtype=np.float64)

        return cross_disch_src, is_home, home_high_risk, vulnerable_dest

    tr_dcross, tr_home, tr_hhr, tr_vuln = extract_transition_friction(train, n_train)
    ev_dcross, ev_home, ev_hhr, ev_vuln = extract_transition_friction(eval_cols, n_eval)

    train_dict["discharge_x_admission"] = tr_dcross
    eval_dict["discharge_x_admission"] = ev_dcross
    extra_categorical.append("discharge_x_admission")

    train_dict["is_discharged_home"] = tr_home
    eval_dict["is_discharged_home"] = ev_home
    train_dict["home_high_utilizer_risk"] = tr_hhr
    eval_dict["home_high_utilizer_risk"] = ev_hhr
    train_dict["vulnerable_facility_discharge"] = tr_vuln
    eval_dict["vulnerable_facility_discharge"] = ev_vuln
    extra_numeric.extend([
        "is_discharged_home",
        "home_high_utilizer_risk",
        "vulnerable_facility_discharge",
    ])

    # 7. Multi-Scale Out-Of-Fold Bayesian Target Resonance
    te_configs = [
        ("diag_1", 15.0),
        ("diag_2", 20.0),
        ("diag_3", 20.0),
        ("discharge_disposition_id", 20.0),
        ("admission_source_id", 25.0),
        ("medical_specialty", 30.0),
        ("discharge_x_admission", 20.0),
    ]

    for col_name, smoothing_val in te_configs:
        te_name = f"te_{col_name}"
        tr_te, ev_te = oof_target_encode(
            train_dict[col_name],
            y_train,
            eval_dict[col_name],
            n_folds=5,
            smoothing=smoothing_val,
            seed=42,
        )
        train_dict[te_name] = tr_te
        eval_dict[te_name] = ev_te
        extra_numeric.append(te_name)

    # 8. Encode Tabular Features with Optimized Intermediate Category Support
    encoder = TabularEncoder(min_count=35)
    encoder.fit(
        task,
        train_dict,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # 9. Tri-Model Gradient Boosting Architecture Ensemble
    # Model 1: Balanced interaction specialist
    m1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.016,
        num_leaves=31,
        min_child_samples=180,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=0.05,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative regularized generalist (prevents subgroup overfitting)
    m2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.020,
        num_leaves=18,
        min_child_samples=320,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.50,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Granular deep tree (captures complex localized multi-morbidity interactions)
    m3 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.013,
        num_leaves=48,
        min_child_samples=130,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=0.01,
        cat_smooth=40.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Weighted ensemble blend
    preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
    preds = np.clip(np.nan_to_num(preds, nan=0.114), 0.0, 1.0)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
