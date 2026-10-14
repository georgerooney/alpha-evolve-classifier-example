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


def _extract_derived_features(
    cols: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Derive clinical, utilization, and diagnostic features from raw tabular dictionaries."""
    n = len(cols["time_in_hospital"])

    # 1. Healthcare Utilization (primary readmission drivers)
    inp = np.nan_to_num(cols.get("number_inpatient", np.zeros(n)).astype(float), nan=0.0)
    emg = np.nan_to_num(cols.get("number_emergency", np.zeros(n)).astype(float), nan=0.0)
    out = np.nan_to_num(cols.get("number_outpatient", np.zeros(n)).astype(float), nan=0.0)

    total_prior = inp + emg + out
    acute_prior = inp + emg
    has_inp = (inp > 0.0).astype(float)
    has_emg = (emg > 0.0).astype(float)
    has_out = (out > 0.0).astype(float)
    has_any_prior = (total_prior > 0.0).astype(float)
    high_inp = (inp >= 2.0).astype(float)
    freq_inp = (inp >= 3.0).astype(float)
    inp_ratio = np.where(total_prior > 0.0, inp / np.maximum(total_prior, 1.0), 0.0)

    # 2. Inpatient stay & care intensity
    time_hosp = np.nan_to_num(cols.get("time_in_hospital", np.ones(n)).astype(float), nan=1.0)
    num_labs = np.nan_to_num(cols.get("num_lab_procedures", np.zeros(n)).astype(float), nan=0.0)
    num_procs = np.nan_to_num(cols.get("num_procedures", np.zeros(n)).astype(float), nan=0.0)
    num_meds = np.nan_to_num(cols.get("num_medications", np.zeros(n)).astype(float), nan=0.0)
    num_diag = np.nan_to_num(cols.get("number_diagnoses", np.zeros(n)).astype(float), nan=0.0)

    safe_time = np.maximum(time_hosp, 1.0)
    labs_per_day = num_labs / safe_time
    meds_per_day = num_meds / safe_time
    procs_per_day = num_procs / safe_time
    diag_per_day = num_diag / safe_time
    total_procs = num_labs + num_procs
    inp_time_mult = inp * time_hosp
    diag_med_ratio = num_meds / np.maximum(num_diag, 1.0)
    service_intensity = num_labs + 5.0 * num_procs + 2.0 * num_meds

    # 3. Demographics & Age bands mapped to numeric
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    age_col = cols.get("age", np.array(["[50-60)"] * n))
    age_numeric = np.array([age_map.get(str(a), 55.0) for a in age_col], dtype=float)
    is_elderly = (age_numeric >= 75.0).astype(float)
    is_senior = (age_numeric >= 65.0).astype(float)
    is_young = (age_numeric < 45.0).astype(float)

    # 4. Diagnoses & ICD-9 clinical groupings
    def _safe_icd9(c):
        if c is None or c == "?" or c == "None":
            return "missing"
        try:
            return str(icd9_group(str(c)))
        except Exception:
            return "missing"

    d1 = cols.get("diag_1", np.array([None] * n))
    d2 = cols.get("diag_2", np.array([None] * n))
    d3 = cols.get("diag_3", np.array([None] * n))

    d1_grp = np.array([_safe_icd9(c) for c in d1], dtype=object)
    d2_grp = np.array([_safe_icd9(c) for c in d2], dtype=object)
    d3_grp = np.array([_safe_icd9(c) for c in d3], dtype=object)

    has_circ = ((d1_grp == "circulatory") | (d2_grp == "circulatory") | (d3_grp == "circulatory")).astype(float)
    has_resp = ((d1_grp == "respiratory") | (d2_grp == "respiratory") | (d3_grp == "respiratory")).astype(float)
    has_diab = ((d1_grp == "diabetes") | (d2_grp == "diabetes") | (d3_grp == "diabetes")).astype(float)
    has_digest = ((d1_grp == "digestive") | (d2_grp == "digestive") | (d3_grp == "digestive")).astype(float)
    has_genit = ((d1_grp == "genitourinary") | (d2_grp == "genitourinary") | (d3_grp == "genitourinary")).astype(float)
    has_inj = ((d1_grp == "injury") | (d2_grp == "injury") | (d3_grp == "injury")).astype(float)
    has_neopl = ((d1_grp == "neoplasms") | (d2_grp == "neoplasms") | (d3_grp == "neoplasms")).astype(float)

    comorb_count = has_circ + has_resp + has_diab + has_digest + has_genit + has_inj + has_neopl
    diab_and_circ = has_circ * has_diab
    primary_circ = (d1_grp == "circulatory").astype(float)
    primary_diab = (d1_grp == "diabetes").astype(float)
    primary_resp = (d1_grp == "respiratory").astype(float)

    # 5. Admission & Discharge disposition
    disc = cols.get("discharge_disposition_id", np.array([None] * n))
    adm_type = cols.get("admission_type_id", np.array([None] * n))
    adm_src = cols.get("admission_source_id", np.array([None] * n))

    discharged_home = np.array([x == "1" for x in disc], dtype=float)
    discharged_snf_icf = np.array([x in ("3", "4") for x in disc], dtype=float)
    discharged_facility = np.array([x in ("2", "3", "4", "5", "22") for x in disc], dtype=float)
    discharged_home_health = np.array([x == "6" for x in disc], dtype=float)
    discharged_ama = np.array([x == "7" for x in disc], dtype=float)

    adm_emergency = np.array([x == "1" for x in adm_type], dtype=float)
    adm_urgent = np.array([x == "2" for x in adm_type], dtype=float)
    adm_elective = np.array([x == "3" for x in adm_type], dtype=float)
    adm_from_er = np.array([x == "7" for x in adm_src], dtype=float)

    # 6. Medications & Insulin regimen
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    active_meds = np.zeros(n, dtype=float)
    med_changes = np.zeros(n, dtype=float)
    for med in med_names:
        if med in cols:
            m = cols[med]
            is_act = np.array([x in ("Steady", "Up", "Down") for x in m], dtype=float)
            is_chg = np.array([x in ("Up", "Down") for x in m], dtype=float)
            active_meds += is_act
            med_changes += is_chg

    if "insulin" in cols:
        ins = cols["insulin"]
        insulin_active = np.array([x in ("Steady", "Up", "Down") for x in ins], dtype=float)
        insulin_changed = np.array([x in ("Up", "Down") for x in ins], dtype=float)
        insulin_up = np.array([x == "Up" for x in ins], dtype=float)
        insulin_down = np.array([x == "Down" for x in ins], dtype=float)
    else:
        insulin_active = np.zeros(n, dtype=float)
        insulin_changed = np.zeros(n, dtype=float)
        insulin_up = np.zeros(n, dtype=float)
        insulin_down = np.zeros(n, dtype=float)

    if "metformin" in cols:
        met = cols["metformin"]
        metformin_active = np.array([x in ("Steady", "Up", "Down") for x in met], dtype=float)
    else:
        metformin_active = np.zeros(n, dtype=float)

    ch_col = cols.get("change", np.array([None] * n))
    dm_col = cols.get("diabetesMed", np.array([None] * n))
    has_change = np.array([x == "Ch" for x in ch_col], dtype=float)
    has_dm = np.array([x == "Yes" for x in dm_col], dtype=float)

    # 7. Glycemic tests & Missingness indicators
    glu = cols.get("max_glu_serum", np.array([None] * n))
    a1c = cols.get("A1Cresult", np.array([None] * n))
    glu_tested = np.array([x is not None and str(x) not in ("None", "?", "nan") for x in glu], dtype=float)
    glu_high = np.array([str(x) in (">200", ">300") for x in glu], dtype=float)
    a1c_tested = np.array([x is not None and str(x) not in ("None", "?", "nan") for x in a1c], dtype=float)
    a1c_high = np.array([str(x) in (">8", ">7") for x in a1c], dtype=float)
    a1c_normal = np.array([str(x) == "Norm" for x in a1c], dtype=float)

    spec = cols.get("medical_specialty", np.array([None] * n))
    has_specialty = np.array([x is not None and str(x) not in ("?", "None", "nan") for x in spec], dtype=float)
    payer = cols.get("payer_code", np.array([None] * n))
    has_payer = np.array([x is not None and str(x) not in ("?", "None", "nan") for x in payer], dtype=float)

    numeric_features = {
        "total_prior_visits": total_prior,
        "acute_prior_visits": acute_prior,
        "has_prior_inpatient": has_inp,
        "has_prior_emergency": has_emg,
        "has_prior_outpatient": has_out,
        "has_any_prior": has_any_prior,
        "high_inpatient": high_inp,
        "freq_inpatient": freq_inp,
        "inpatient_ratio": inp_ratio,
        "labs_per_day": labs_per_day,
        "meds_per_day": meds_per_day,
        "procs_per_day": procs_per_day,
        "diag_per_day": diag_per_day,
        "total_procs": total_procs,
        "inp_time_mult": inp_time_mult,
        "diag_med_ratio": diag_med_ratio,
        "service_intensity": service_intensity,
        "age_numeric": age_numeric,
        "is_elderly": is_elderly,
        "is_senior": is_senior,
        "is_young": is_young,
        "has_circ": has_circ,
        "has_resp": has_resp,
        "has_diab": has_diab,
        "has_digest": has_digest,
        "has_genit": has_genit,
        "has_inj": has_inj,
        "has_neopl": has_neopl,
        "comorb_count": comorb_count,
        "diab_and_circ": diab_and_circ,
        "primary_circ": primary_circ,
        "primary_diab": primary_diab,
        "primary_resp": primary_resp,
        "discharged_home": discharged_home,
        "discharged_snf_icf": discharged_snf_icf,
        "discharged_facility": discharged_facility,
        "discharged_home_health": discharged_home_health,
        "discharged_ama": discharged_ama,
        "adm_emergency": adm_emergency,
        "adm_urgent": adm_urgent,
        "adm_elective": adm_elective,
        "adm_from_er": adm_from_er,
        "active_meds": active_meds,
        "med_changes": med_changes,
        "insulin_active": insulin_active,
        "insulin_changed": insulin_changed,
        "insulin_up": insulin_up,
        "insulin_down": insulin_down,
        "metformin_active": metformin_active,
        "has_change": has_change,
        "has_dm": has_dm,
        "glu_tested": glu_tested,
        "glu_high": glu_high,
        "a1c_tested": a1c_tested,
        "a1c_high": a1c_high,
        "a1c_normal": a1c_normal,
        "has_specialty": has_specialty,
        "has_payer": has_payer,
    }

    categorical_features = {
        "diag_1_group": d1_grp,
        "diag_2_group": d2_grp,
        "diag_3_group": d3_grp,
    }

    return numeric_features, categorical_features


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    from sklearn.model_selection import StratifiedKFold

    train_data = dict(train)
    eval_data = dict(eval_cols)

    train_num, train_cat = _extract_derived_features(train_data)
    eval_num, eval_cat = _extract_derived_features(eval_data)

    for k, v in train_num.items():
        train_data[k] = v
    for k, v in train_cat.items():
        train_data[k] = v

    for k, v in eval_num.items():
        eval_data[k] = v
    for k, v in eval_cat.items():
        eval_data[k] = v

    extra_num_names = list(train_num.keys())
    extra_cat_names = list(train_cat.keys())

    encoder = TabularEncoder(min_count=10)
    encoder.fit(task, train_data, extra_numeric=extra_num_names, extra_categorical=extra_cat_names)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    cat_idx = encoder.categorical_indices
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    eval_preds = np.zeros(len(x_eval), dtype=float)

    for train_idx, _ in skf.split(x_train, y_train):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]

        # Model 1: Conservative, shallow trees for core effects
        m1 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=16,
            max_depth=5,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=3.0,
            cat_smooth=15.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(x_tr, y_tr, categorical_feature=cat_idx)
        eval_preds += m1.predict_proba(x_eval)[:, 1]

        # Model 2: Deeper interactions with strong feature-level regularisation
        m2 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=28,
            max_depth=7,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=5.0,
            cat_smooth=25.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(x_tr, y_tr, categorical_feature=cat_idx)
        eval_preds += m2.predict_proba(x_eval)[:, 1]

        # Model 3: Balanced depth with high min_child_samples for subgroup robustness
        m3 = lgb.LGBMClassifier(
            n_estimators=290,
            learning_rate=0.03,
            num_leaves=22,
            max_depth=6,
            min_child_samples=65,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=2.0,
            cat_smooth=10.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        m3.fit(x_tr, y_tr, categorical_feature=cat_idx)
        eval_preds += m3.predict_proba(x_eval)[:, 1]

    eval_preds /= (n_splits * 3.0)
    return np.clip(eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
