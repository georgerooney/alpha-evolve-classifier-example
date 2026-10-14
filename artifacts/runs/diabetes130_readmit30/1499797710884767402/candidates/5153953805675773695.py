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
    """Derive rich clinical, utilization, and diagnostic features from raw tabular dictionaries."""
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
    freq_emg = (emg >= 2.0).astype(float)
    inp_ratio = np.where(total_prior > 0.0, inp / np.maximum(total_prior, 1.0), 0.0)
    emg_ratio = emg / np.maximum(total_prior, 1.0)
    acute_ratio = acute_prior / np.maximum(total_prior, 1.0)
    inp_squared = inp ** 2
    acute_squared = acute_prior ** 2

    log_inp = np.log1p(inp)
    log_emg = np.log1p(emg)
    log_out = np.log1p(out)
    log_total_prior = np.log1p(total_prior)

    # 2. Inpatient stay & care intensity
    time_hosp = np.nan_to_num(cols.get("time_in_hospital", np.ones(n)).astype(float), nan=1.0)
    num_labs = np.nan_to_num(cols.get("num_lab_procedures", np.zeros(n)).astype(float), nan=0.0)
    num_procs = np.nan_to_num(cols.get("num_procedures", np.zeros(n)).astype(float), nan=0.0)
    num_meds = np.nan_to_num(cols.get("num_medications", np.zeros(n)).astype(float), nan=0.0)
    num_diag = np.nan_to_num(cols.get("number_diagnoses", np.zeros(n)).astype(float), nan=0.0)

    log_labs = np.log1p(num_labs)
    log_meds = np.log1p(num_meds)
    log_time = np.log1p(time_hosp)
    log_diag = np.log1p(num_diag)

    safe_time = np.maximum(time_hosp, 1.0)
    labs_per_day = num_labs / safe_time
    meds_per_day = num_meds / safe_time
    procs_per_day = num_procs / safe_time
    diag_per_day = num_diag / safe_time
    total_procs = num_labs + num_procs
    inp_time_mult = inp * time_hosp
    diag_med_ratio = num_meds / np.maximum(num_diag, 1.0)
    service_intensity = num_labs + 5.0 * num_procs + 2.0 * num_meds
    polypharmacy = (num_meds >= 16.0).astype(float)
    high_polypharmacy = (num_meds >= 25.0).astype(float)
    long_stay = (time_hosp >= 7.0).astype(float)
    high_labs = (num_labs >= 60.0).astype(float)
    high_diag = (num_diag >= 9.0).astype(float)

    # 3. Demographics & Age bands mapped to numeric
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    age_col = cols.get("age", np.array(["[50-60)"] * n))
    age_numeric = np.array([age_map.get(str(a), 55.0) for a in age_col], dtype=float)
    age_squared = (age_numeric / 10.0) ** 2
    is_elderly = (age_numeric >= 75.0).astype(float)
    is_senior = (age_numeric >= 65.0).astype(float)
    is_young = (age_numeric < 45.0).astype(float)
    is_very_elderly = (age_numeric >= 85.0).astype(float)

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

    circ_count = ((d1_grp == "circulatory").astype(float) + (d2_grp == "circulatory").astype(float) + (d3_grp == "circulatory").astype(float))
    resp_count = ((d1_grp == "respiratory").astype(float) + (d2_grp == "respiratory").astype(float) + (d3_grp == "respiratory").astype(float))
    diab_count = ((d1_grp == "diabetes").astype(float) + (d2_grp == "diabetes").astype(float) + (d3_grp == "diabetes").astype(float))

    comorb_count = has_circ + has_resp + has_diab + has_digest + has_genit + has_inj + has_neopl
    diab_and_circ = has_circ * has_diab
    primary_circ = (d1_grp == "circulatory").astype(float)
    primary_diab = (d1_grp == "diabetes").astype(float)
    primary_resp = (d1_grp == "respiratory").astype(float)
    primary_digest = (d1_grp == "digestive").astype(float)

    # High-risk chronic condition flags based on specific ICD-9 codes
    d1_str = [str(x).strip() if (x is not None and x != "?") else "" for x in d1]
    d2_str = [str(x).strip() if (x is not None and x != "?") else "" for x in d2]
    d3_str = [str(x).strip() if (x is not None and x != "?") else "" for x in d3]

    def get_combined_flag(prefixes):
        return np.array([
            (any(c1.startswith(p) for p in prefixes) or
             any(c2.startswith(p) for p in prefixes) or
             any(c3.startswith(p) for p in prefixes))
            for c1, c2, c3 in zip(d1_str, d2_str, d3_str)
        ], dtype=float)

    has_hf = get_combined_flag(("428",))
    has_ckd = get_combined_flag(("585", "586"))
    has_copd = get_combined_flag(("49",))
    has_htn = get_combined_flag(("40",))
    has_ihd = get_combined_flag(("41",))
    has_dysrhythmia = get_combined_flag(("427",))
    has_liver = get_combined_flag(("571", "572", "573"))
    has_stroke = get_combined_flag(("43",))
    has_pneumonia = get_combined_flag(("48",))
    has_sepsis = get_combined_flag(("038", "995.9"))
    has_fluid_electrolyte = get_combined_flag(("276",))
    has_skin_ulcer = get_combined_flag(("681", "682", "707"))

    # Complicated diabetes (ICD-9 250.1 - 250.9)
    def _is_comp_diab(c):
        return c.startswith("250.") and len(c) > 4 and c[4] != "0"

    has_diab_comp = np.array([
        (_is_comp_diab(c1) or _is_comp_diab(c2) or _is_comp_diab(c3))
        for c1, c2, c3 in zip(d1_str, d2_str, d3_str)
    ], dtype=float)

    has_cardiorenal = has_hf * has_ckd
    has_diab_hf = has_diab * has_hf
    has_diab_ckd = has_diab * has_ckd
    has_copd_hf = has_copd * has_hf
    charlson_score = (1.0 * has_hf + 1.0 * has_copd + 1.0 * has_diab_comp + 1.0 * has_stroke +
                      2.0 * has_ckd + 2.0 * has_liver + 2.0 * has_neopl)

    # 5. Admission & Discharge disposition
    disc = cols.get("discharge_disposition_id", np.array([None] * n))
    adm_type = cols.get("admission_type_id", np.array([None] * n))
    adm_src = cols.get("admission_source_id", np.array([None] * n))

    discharged_home = np.array([x == "1" for x in disc], dtype=float)
    discharged_snf_icf = np.array([x in ("3", "4") for x in disc], dtype=float)
    discharged_facility = np.array([x in ("2", "3", "4", "5", "22") for x in disc], dtype=float)
    discharged_home_health = np.array([x == "6" for x in disc], dtype=float)
    discharged_ama = np.array([x == "7" for x in disc], dtype=float)
    discharged_high_risk = np.array([x in ("2", "3", "4", "5", "6", "7", "22") for x in disc], dtype=float)

    adm_emergency = np.array([x == "1" for x in adm_type], dtype=float)
    adm_urgent = np.array([x == "2" for x in adm_type], dtype=float)
    adm_elective = np.array([x == "3" for x in adm_type], dtype=float)
    adm_from_er = np.array([x == "7" for x in adm_src], dtype=float)
    adm_transfer = np.array([x in ("4", "5", "6", "10", "22") for x in adm_src], dtype=float)
    adm_acute = np.maximum(adm_emergency, adm_from_er)

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
    meds_up = np.zeros(n, dtype=float)
    meds_down = np.zeros(n, dtype=float)
    meds_steady = np.zeros(n, dtype=float)

    for med in med_names:
        if med in cols:
            m = cols[med]
            is_act = np.array([x in ("Steady", "Up", "Down") for x in m], dtype=float)
            is_up = np.array([x == "Up" for x in m], dtype=float)
            is_dn = np.array([x == "Down" for x in m], dtype=float)
            is_st = np.array([x == "Steady" for x in m], dtype=float)
            active_meds += is_act
            meds_up += is_up
            meds_down += is_dn
            meds_steady += is_st
            med_changes += (is_up + is_dn)

    any_med_up = (meds_up > 0.0).astype(float)
    any_med_down = (meds_down > 0.0).astype(float)

    # Insulin status
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

    # Metformin status
    if "metformin" in cols:
        met = cols["metformin"]
        metformin_active = np.array([x in ("Steady", "Up", "Down") for x in met], dtype=float)
    else:
        metformin_active = np.zeros(n, dtype=float)

    # Sulfonylurea class
    su_names = ["glimepiride", "glipizide", "glyburide", "chlorpropamide", "tolbutamide", "tolazamide", "acetohexamide"]
    has_su = np.zeros(n, dtype=float)
    for su in su_names:
        if su in cols:
            has_su = np.maximum(has_su, np.array([x in ("Steady", "Up", "Down") for x in cols[su]], dtype=float))

    regimen_insulin_only = ((insulin_active > 0.0) & (active_meds == insulin_active)).astype(float)
    regimen_insulin_plus_oral = ((insulin_active > 0.0) & (active_meds > insulin_active)).astype(float)
    regimen_oral_only = ((insulin_active == 0.0) & (active_meds > 0.0)).astype(float)
    non_diab_meds = np.maximum(num_meds - active_meds, 0.0)

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

    w_col = cols.get("weight", np.array([None] * n))
    weight_measured = np.array([x is not None and str(x) not in ("?", "None", "nan") for x in w_col], dtype=float)

    # Demographic & severity interactions
    elderly_with_many_meds = is_elderly * num_meds
    senior_with_many_meds = is_senior * num_meds
    elderly_with_prior_inp = is_elderly * inp
    senior_with_prior_inp = is_senior * inp
    very_old_x_snf = is_very_elderly * discharged_snf_icf
    elderly_with_hf = is_elderly * has_hf
    elderly_with_ckd = is_elderly * has_ckd

    elderly_discharged_facility = is_elderly * discharged_facility
    senior_discharged_facility = is_senior * discharged_facility
    elderly_discharged_home = is_elderly * discharged_home
    inp_x_discharged_facility = inp * discharged_facility

    hosp_per_prior_visit = time_hosp / np.maximum(total_prior, 1.0)
    meds_per_prior_visit = num_meds / np.maximum(total_prior, 1.0)
    labs_per_prior_visit = num_labs / np.maximum(total_prior, 1.0)

    high_a1c_and_insulin = a1c_high * insulin_active
    high_glu_and_insulin = glu_high * insulin_active
    insulin_change = insulin_active * insulin_changed

    numeric_features = {
        "total_prior_visits": total_prior,
        "acute_prior_visits": acute_prior,
        "has_prior_inpatient": has_inp,
        "has_prior_emergency": has_emg,
        "has_prior_outpatient": has_out,
        "has_any_prior": has_any_prior,
        "high_inpatient": high_inp,
        "freq_inpatient": freq_inp,
        "freq_emergency": freq_emg,
        "inpatient_ratio": inp_ratio,
        "emg_ratio": emg_ratio,
        "acute_ratio": acute_ratio,
        "inp_squared": inp_squared,
        "acute_squared": acute_squared,
        "log_inpatient": log_inp,
        "log_emergency": log_emg,
        "log_outpatient": log_out,
        "log_total_prior": log_total_prior,
        "labs_per_day": labs_per_day,
        "meds_per_day": meds_per_day,
        "procs_per_day": procs_per_day,
        "diag_per_day": diag_per_day,
        "total_procs": total_procs,
        "inp_time_mult": inp_time_mult,
        "diag_med_ratio": diag_med_ratio,
        "service_intensity": service_intensity,
        "polypharmacy": polypharmacy,
        "high_polypharmacy": high_polypharmacy,
        "long_stay": long_stay,
        "high_labs": high_labs,
        "high_diag": high_diag,
        "log_labs": log_labs,
        "log_meds": log_meds,
        "log_time_in_hospital": log_time,
        "log_diagnoses": log_diag,
        "age_numeric": age_numeric,
        "age_squared": age_squared,
        "is_elderly": is_elderly,
        "is_senior": is_senior,
        "is_young": is_young,
        "is_very_elderly": is_very_elderly,
        "has_circ": has_circ,
        "has_resp": has_resp,
        "has_diab": has_diab,
        "has_digest": has_digest,
        "has_genit": has_genit,
        "has_inj": has_inj,
        "has_neopl": has_neopl,
        "circ_count": circ_count,
        "resp_count": resp_count,
        "diab_count": diab_count,
        "comorb_count": comorb_count,
        "diab_and_circ": diab_and_circ,
        "primary_circ": primary_circ,
        "primary_diab": primary_diab,
        "primary_resp": primary_resp,
        "primary_digest": primary_digest,
        "has_hf": has_hf,
        "has_ckd": has_ckd,
        "has_copd": has_copd,
        "has_htn": has_htn,
        "has_ihd": has_ihd,
        "has_dysrhythmia": has_dysrhythmia,
        "has_liver": has_liver,
        "has_stroke": has_stroke,
        "has_pneumonia": has_pneumonia,
        "has_sepsis": has_sepsis,
        "has_fluid_electrolyte": has_fluid_electrolyte,
        "has_skin_ulcer": has_skin_ulcer,
        "has_diab_comp": has_diab_comp,
        "has_cardiorenal": has_cardiorenal,
        "has_diab_hf": has_diab_hf,
        "has_diab_ckd": has_diab_ckd,
        "has_copd_hf": has_copd_hf,
        "charlson_score": charlson_score,
        "discharged_home": discharged_home,
        "discharged_snf_icf": discharged_snf_icf,
        "discharged_facility": discharged_facility,
        "discharged_home_health": discharged_home_health,
        "discharged_ama": discharged_ama,
        "discharged_high_risk": discharged_high_risk,
        "adm_emergency": adm_emergency,
        "adm_urgent": adm_urgent,
        "adm_elective": adm_elective,
        "adm_from_er": adm_from_er,
        "adm_transfer": adm_transfer,
        "adm_acute": adm_acute,
        "active_meds": active_meds,
        "med_changes": med_changes,
        "meds_up": meds_up,
        "meds_down": meds_down,
        "meds_steady": meds_steady,
        "any_med_up": any_med_up,
        "any_med_down": any_med_down,
        "insulin_active": insulin_active,
        "insulin_changed": insulin_changed,
        "insulin_up": insulin_up,
        "insulin_down": insulin_down,
        "metformin_active": metformin_active,
        "has_su": has_su,
        "regimen_insulin_only": regimen_insulin_only,
        "regimen_insulin_plus_oral": regimen_insulin_plus_oral,
        "regimen_oral_only": regimen_oral_only,
        "non_diab_meds": non_diab_meds,
        "has_change": has_change,
        "has_dm": has_dm,
        "glu_tested": glu_tested,
        "glu_high": glu_high,
        "a1c_tested": a1c_tested,
        "a1c_high": a1c_high,
        "a1c_normal": a1c_normal,
        "has_specialty": has_specialty,
        "has_payer": has_payer,
        "weight_measured": weight_measured,
        "elderly_with_many_meds": elderly_with_many_meds,
        "senior_with_many_meds": senior_with_many_meds,
        "elderly_with_prior_inp": elderly_with_prior_inp,
        "senior_with_prior_inp": senior_with_prior_inp,
        "very_old_x_snf": very_old_x_snf,
        "elderly_with_hf": elderly_with_hf,
        "elderly_with_ckd": elderly_with_ckd,
        "elderly_discharged_facility": elderly_discharged_facility,
        "senior_discharged_facility": senior_discharged_facility,
        "elderly_discharged_home": elderly_discharged_home,
        "inp_x_discharged_facility": inp_x_discharged_facility,
        "hosp_per_prior_visit": hosp_per_prior_visit,
        "meds_per_prior_visit": meds_per_prior_visit,
        "labs_per_prior_visit": labs_per_prior_visit,
        "high_a1c_and_insulin": high_a1c_and_insulin,
        "high_glu_and_insulin": high_glu_and_insulin,
        "insulin_change": insulin_change,
    }

    categorical_features = {
        "diag_1_group": d1_grp,
        "diag_2_group": d2_grp,
        "diag_3_group": d3_grp,
    }

    return numeric_features, categorical_features


def _compute_target_encodings(
    train_data: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_data: Dict[str, np.ndarray],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute out-of-fold target encodings for high-cardinality and categorical variables."""
    from sklearn.model_selection import StratifiedKFold

    n_tr = len(y_train)
    n_ev = len(eval_data["time_in_hospital"])
    global_prior = float(np.mean(y_train))

    te_train: Dict[str, np.ndarray] = {}
    te_eval: Dict[str, np.ndarray] = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 1. Single categorical columns
    cols_to_encode = [
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "medical_specialty",
        "payer_code",
    ]

    m_cat = 20.0
    for col in cols_to_encode:
        c_tr = np.array([str(x) if (x is not None and str(x) not in ("?", "None", "nan")) else "__MISSING__"
                         for x in train_data.get(col, [None] * n_tr)])
        c_ev = np.array([str(x) if (x is not None and str(x) not in ("?", "None", "nan")) else "__MISSING__"
                         for x in eval_data.get(col, [None] * n_ev)])

        oof = np.full(n_tr, global_prior, dtype=float)
        test = np.full(n_ev, global_prior, dtype=float)

        for tr_idx, val_idx in skf.split(c_tr, y_train):
            counts: Dict[str, int] = {}
            sums: Dict[str, float] = {}
            for i in tr_idx:
                v = c_tr[i]
                counts[v] = counts.get(v, 0) + 1
                sums[v] = sums.get(v, 0.0) + y_train[i]
            fold_prior = float(np.mean(y_train[tr_idx]))

            for i in val_idx:
                v = c_tr[i]
                if v in counts:
                    oof[i] = (sums[v] + m_cat * fold_prior) / (counts[v] + m_cat)
                else:
                    oof[i] = fold_prior

        # Full train for eval
        full_counts: Dict[str, int] = {}
        full_sums: Dict[str, float] = {}
        for i in range(n_tr):
            v = c_tr[i]
            full_counts[v] = full_counts.get(v, 0) + 1
            full_sums[v] = full_sums.get(v, 0.0) + y_train[i]

        for i in range(n_ev):
            v = c_ev[i]
            if v in full_counts:
                test[i] = (full_sums[v] + m_cat * global_prior) / (full_counts[v] + m_cat)
            else:
                test[i] = global_prior

        te_train[f"te_{col}"] = oof
        te_eval[f"te_{col}"] = test

    # 2. Pooled ICD-9 diagnosis codes across diag_1, diag_2, diag_3
    d1_tr = [str(x).strip() if (x is not None and str(x) not in ("?", "None", "nan")) else ""
             for x in train_data.get("diag_1", [None] * n_tr)]
    d2_tr = [str(x).strip() if (x is not None and str(x) not in ("?", "None", "nan")) else ""
             for x in train_data.get("diag_2", [None] * n_tr)]
    d3_tr = [str(x).strip() if (x is not None and str(x) not in ("?", "None", "nan")) else ""
             for x in train_data.get("diag_3", [None] * n_tr)]

    d1_ev = [str(x).strip() if (x is not None and str(x) not in ("?", "None", "nan")) else ""
             for x in eval_data.get("diag_1", [None] * n_ev)]
    d2_ev = [str(x).strip() if (x is not None and str(x) not in ("?", "None", "nan")) else ""
             for x in eval_data.get("diag_2", [None] * n_ev)]
    d3_ev = [str(x).strip() if (x is not None and str(x) not in ("?", "None", "nan")) else ""
             for x in eval_data.get("diag_3", [None] * n_ev)]

    te_d1_tr = np.full(n_tr, global_prior, dtype=float)
    te_d2_tr = np.full(n_tr, global_prior, dtype=float)
    te_d3_tr = np.full(n_tr, global_prior, dtype=float)
    te_dmax_tr = np.full(n_tr, global_prior, dtype=float)
    te_dmean_tr = np.full(n_tr, global_prior, dtype=float)

    m_diag = 25.0
    for tr_idx, val_idx in skf.split(d1_tr, y_train):
        counts_diag: Dict[str, int] = {}
        sums_diag: Dict[str, float] = {}
        for i in tr_idx:
            y = y_train[i]
            unique_codes = {c for c in (d1_tr[i], d2_tr[i], d3_tr[i]) if c}
            for c in unique_codes:
                counts_diag[c] = counts_diag.get(c, 0) + 1
                sums_diag[c] = sums_diag.get(c, 0.0) + y
        fold_prior = float(np.mean(y_train[tr_idx]))

        def _get_fold_risk(code):
            if code and code in counts_diag:
                return (sums_diag[code] + m_diag * fold_prior) / (counts_diag[code] + m_diag)
            return fold_prior

        for i in val_idx:
            r1 = _get_fold_risk(d1_tr[i])
            r2 = _get_fold_risk(d2_tr[i])
            r3 = _get_fold_risk(d3_tr[i])
            te_d1_tr[i] = r1
            te_d2_tr[i] = r2
            te_d3_tr[i] = r3
            te_dmax_tr[i] = max(r1, r2, r3)
            te_dmean_tr[i] = (r1 + r2 + r3) / 3.0

    # Full pooled diag counts on all train
    full_counts_diag: Dict[str, int] = {}
    full_sums_diag: Dict[str, float] = {}
    for i in range(n_tr):
        y = y_train[i]
        unique_codes = {c for c in (d1_tr[i], d2_tr[i], d3_tr[i]) if c}
        for c in unique_codes:
            full_counts_diag[c] = full_counts_diag.get(c, 0) + 1
            full_sums_diag[c] = full_sums_diag.get(c, 0.0) + y

    def _get_full_risk(code):
        if code and code in full_counts_diag:
            return (full_sums_diag[code] + m_diag * global_prior) / (full_counts_diag[code] + m_diag)
        return global_prior

    te_d1_ev = np.array([_get_full_risk(c) for c in d1_ev], dtype=float)
    te_d2_ev = np.array([_get_full_risk(c) for c in d2_ev], dtype=float)
    te_d3_ev = np.array([_get_full_risk(c) for c in d3_ev], dtype=float)
    te_dmax_ev = np.maximum(te_d1_ev, np.maximum(te_d2_ev, te_d3_ev))
    te_dmean_ev = (te_d1_ev + te_d2_ev + te_d3_ev) / 3.0

    te_train["te_diag_1"] = te_d1_tr
    te_train["te_diag_2"] = te_d2_tr
    te_train["te_diag_3"] = te_d3_tr
    te_train["te_diag_max"] = te_dmax_tr
    te_train["te_diag_mean"] = te_dmean_tr

    te_eval["te_diag_1"] = te_d1_ev
    te_eval["te_diag_2"] = te_d2_ev
    te_eval["te_diag_3"] = te_d3_ev
    te_eval["te_diag_max"] = te_dmax_ev
    te_eval["te_diag_mean"] = te_dmean_ev

    return te_train, te_eval


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

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

    cat_idx = encoder.categorical_indices

    # Compute out-of-fold target encodings for high-cardinality features
    te_train, te_eval = _compute_target_encodings(train_data, y_train, eval_data, n_splits=5, random_state=42)
    for k, v in te_train.items():
        train_data[k] = v
        extra_num_names.append(k)
    for k, v in te_eval.items():
        eval_data[k] = v

    encoder = TabularEncoder(min_count=8)
    encoder.fit(task, train_data, extra_numeric=extra_num_names, extra_categorical=extra_cat_names)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    cat_idx = encoder.categorical_indices

    # Train a complementary 4-model ensemble with diverse depths, leaves, and regularizations
    models = [
        # Model 1: Deeper interaction structure
        lgb.LGBMClassifier(
            n_estimators=520,
            learning_rate=0.022,
            num_leaves=31,
            max_depth=6,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=5.0,
            cat_smooth=20.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Conservative main-effects structure with high child samples
        lgb.LGBMClassifier(
            n_estimators=550,
            learning_rate=0.018,
            num_leaves=16,
            max_depth=4,
            min_child_samples=65,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=3.0,
            cat_smooth=15.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Heavily regularized model with tight feature subsampling
        lgb.LGBMClassifier(
            n_estimators=520,
            learning_rate=0.022,
            num_leaves=24,
            max_depth=5,
            min_child_samples=35,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=1.0,
            reg_lambda=12.0,
            cat_smooth=30.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: Mid-depth balanced structure with diverse seed
        lgb.LGBMClassifier(
            n_estimators=520,
            learning_rate=0.02,
            num_leaves=20,
            max_depth=5,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=6.0,
            cat_smooth=25.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(x_eval), dtype=float)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=cat_idx)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
