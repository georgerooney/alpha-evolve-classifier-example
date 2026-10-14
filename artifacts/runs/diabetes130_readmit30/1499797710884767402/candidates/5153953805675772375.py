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
from typing import Dict, Tuple, List, Sequence
import numpy as np


def extract_icd9_prefix(val) -> str:
    """Extract 3-digit ICD-9 disease category prefix."""
    if val is None or val == 'None' or val == '?' or str(val) == 'nan':
        return 'missing'
    s = str(val).strip()
    if not s:
        return 'missing'
    return s.split('.')[0]


def add_features(cols_dict: Dict[str, np.ndarray], med_cols: list) -> Dict[str, np.ndarray]:
    new_cols = dict(cols_dict)
    n = len(next(iter(cols_dict.values())))

    # 1. Age conversion
    age_arr = cols_dict.get('age')
    parsed_age = []
    if age_arr is not None:
        for val in age_arr:
            if val is None or not isinstance(val, str):
                parsed_age.append(np.nan)
            else:
                s = val.strip('[]()')
                parts = s.split('-')
                if len(parts) == 2:
                    try:
                        parsed_age.append((float(parts[0]) + float(parts[1])) / 2.0)
                    except ValueError:
                        parsed_age.append(np.nan)
                else:
                    parsed_age.append(np.nan)
    else:
        parsed_age = [np.nan] * n
    new_cols['age_num'] = np.array(parsed_age, dtype=np.float64)

    # 2. Prior utilization metrics
    outpatient = np.nan_to_num(cols_dict.get('number_outpatient', np.zeros(n)), nan=0.0)
    emergency = np.nan_to_num(cols_dict.get('number_emergency', np.zeros(n)), nan=0.0)
    inpatient = np.nan_to_num(cols_dict.get('number_inpatient', np.zeros(n)), nan=0.0)

    total_visits = outpatient + emergency + inpatient
    new_cols['total_visits'] = total_visits.astype(np.float64)
    new_cols['has_prior_visits'] = (total_visits > 0).astype(np.float64)
    new_cols['has_prior_inpatient'] = (inpatient > 0).astype(np.float64)
    new_cols['frequent_inpatient'] = (inpatient >= 2).astype(np.float64)
    new_cols['has_prior_emergency'] = (emergency > 0).astype(np.float64)
    new_cols['inpatient_ratio'] = (inpatient / (total_visits + 1.0)).astype(np.float64)
    new_cols['emergency_ratio'] = (emergency / (total_visits + 1.0)).astype(np.float64)
    new_cols['log_inpatient'] = np.log1p(inpatient).astype(np.float64)
    new_cols['log_emergency'] = np.log1p(emergency).astype(np.float64)
    new_cols['log_total_visits'] = np.log1p(total_visits).astype(np.float64)

    # 3. Hospital stay intensity & ratios
    time_hosp = np.nan_to_num(cols_dict.get('time_in_hospital', np.ones(n)), nan=1.0)
    num_labs = np.nan_to_num(cols_dict.get('num_lab_procedures', np.zeros(n)), nan=0.0)
    num_procs = np.nan_to_num(cols_dict.get('num_procedures', np.zeros(n)), nan=0.0)
    num_meds = np.nan_to_num(cols_dict.get('num_medications', np.zeros(n)), nan=0.0)
    num_diags = np.nan_to_num(cols_dict.get('number_diagnoses', np.zeros(n)), nan=0.0)

    new_cols['meds_per_day'] = (num_meds / (time_hosp + 0.1)).astype(np.float64)
    new_cols['labs_per_day'] = (num_labs / (time_hosp + 0.1)).astype(np.float64)
    new_cols['procs_per_day'] = (num_procs / (time_hosp + 0.1)).astype(np.float64)
    new_cols['diags_per_day'] = (num_diags / (time_hosp + 0.1)).astype(np.float64)
    new_cols['polypharmacy'] = (num_meds >= 15.0).astype(np.float64)
    new_cols['complexity_score'] = (num_diags * 2.0 + num_meds + time_hosp * 1.5 + inpatient * 5.0).astype(np.float64)
    new_cols['log_time_in_hospital'] = np.log1p(time_hosp).astype(np.float64)

    # 4. ICD-9 Chapter groupings & 3-Digit Prefixes
    diag_1 = cols_dict.get('diag_1')
    diag_2 = cols_dict.get('diag_2')
    diag_3 = cols_dict.get('diag_3')

    p1 = np.array([extract_icd9_prefix(x) for x in diag_1], dtype=object) if diag_1 is not None else np.array(['missing']*n, dtype=object)
    p2 = np.array([extract_icd9_prefix(x) for x in diag_2], dtype=object) if diag_2 is not None else np.array(['missing']*n, dtype=object)
    p3 = np.array([extract_icd9_prefix(x) for x in diag_3], dtype=object) if diag_3 is not None else np.array(['missing']*n, dtype=object)

    new_cols['diag_1_prefix'] = p1
    new_cols['diag_2_prefix'] = p2
    new_cols['diag_3_prefix'] = p3

    g1 = np.array([icd9_group(str(x)) if (x is not None and str(x) not in ('None', '?', 'nan')) else 'missing' for x in diag_1], dtype=object) if diag_1 is not None else np.array(['missing']*n, dtype=object)
    g2 = np.array([icd9_group(str(x)) if (x is not None and str(x) not in ('None', '?', 'nan')) else 'missing' for x in diag_2], dtype=object) if diag_2 is not None else np.array(['missing']*n, dtype=object)
    g3 = np.array([icd9_group(str(x)) if (x is not None and str(x) not in ('None', '?', 'nan')) else 'missing' for x in diag_3], dtype=object) if diag_3 is not None else np.array(['missing']*n, dtype=object)

    new_cols['diag_1_group'] = g1
    new_cols['diag_2_group'] = g2
    new_cols['diag_3_group'] = g3

    # 5. Comorbidities (Charlson/Elixhauser chronic disease indicators)
    chf_set = {'428', '398', '402', '404'}
    cad_set = {'410', '411', '412', '413', '414'}
    copd_set = {'490', '491', '492', '493', '494', '495', '496', '518'}
    ckd_set = {'585', '586', '584', '403', '404'}
    htn_set = {'401', '402', '403', '404', '405'}
    fluid_set = {'276'}
    infection_set = {'038', '995', '482', '486'}

    has_chf = np.zeros(n, dtype=np.float64)
    has_cad = np.zeros(n, dtype=np.float64)
    has_copd = np.zeros(n, dtype=np.float64)
    has_ckd = np.zeros(n, dtype=np.float64)
    has_htn = np.zeros(n, dtype=np.float64)
    has_fluid = np.zeros(n, dtype=np.float64)
    has_infection = np.zeros(n, dtype=np.float64)
    has_diabetes = np.zeros(n, dtype=np.float64)
    unique_groups = np.zeros(n, dtype=np.float64)

    for i in range(n):
        pref_set = {p1[i], p2[i], p3[i]}
        grp_set = {g1[i], g2[i], g3[i]} - {'missing'}
        unique_groups[i] = len(grp_set)
        if 'diabetes' in grp_set or '250' in pref_set:
            has_diabetes[i] = 1.0
        if bool(pref_set & chf_set):
            has_chf[i] = 1.0
        if bool(pref_set & cad_set):
            has_cad[i] = 1.0
        if bool(pref_set & copd_set):
            has_copd[i] = 1.0
        if bool(pref_set & ckd_set):
            has_ckd[i] = 1.0
        if bool(pref_set & htn_set):
            has_htn[i] = 1.0
        if bool(pref_set & fluid_set):
            has_fluid[i] = 1.0
        if bool(pref_set & infection_set):
            has_infection[i] = 1.0

    new_cols['num_uniq_diag_groups'] = unique_groups
    new_cols['has_diabetes_diag'] = has_diabetes
    new_cols['has_circulatory_diag'] = np.array([1.0 if 'circulatory' in {g1[i], g2[i], g3[i]} else 0.0 for i in range(n)], dtype=np.float64)
    new_cols['has_chf'] = has_chf
    new_cols['has_cad'] = has_cad
    new_cols['has_copd'] = has_copd
    new_cols['has_ckd'] = has_ckd
    new_cols['has_htn'] = has_htn
    new_cols['has_fluid_disorder'] = has_fluid
    new_cols['has_infection'] = has_infection
    new_cols['chronic_comorbidity_count'] = (has_chf + has_cad + has_copd + has_ckd + has_htn + has_fluid + has_infection).astype(np.float64)

    # 6. Medication features
    active = np.zeros(n, dtype=np.float64)
    up = np.zeros(n, dtype=np.float64)
    down = np.zeros(n, dtype=np.float64)
    change_cnt = np.zeros(n, dtype=np.float64)
    oral_active = np.zeros(n, dtype=np.float64)

    oral_med_names = {
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    }

    for col in med_cols:
        if col in cols_dict:
            arr_str = cols_dict[col].astype(str)
            is_act = (arr_str != 'No') & (arr_str != 'None') & (arr_str != 'nan') & (arr_str != '?')
            active += is_act
            up += (arr_str == 'Up')
            down += (arr_str == 'Down')
            change_cnt += (arr_str == 'Up') | (arr_str == 'Down')
            if col in oral_med_names:
                oral_active += is_act

    new_cols['num_active_meds'] = active
    new_cols['num_med_up'] = up
    new_cols['num_med_down'] = down
    new_cols['num_med_changes'] = change_cnt
    new_cols['num_oral_meds'] = oral_active

    insulin_arr = cols_dict.get('insulin')
    if insulin_arr is not None:
        ins_str = insulin_arr.astype(str)
        ins_act = ((ins_str != 'No') & (ins_str != 'None') & (ins_str != 'nan') & (ins_str != '?')).astype(np.float64)
        new_cols['insulin_active'] = ins_act
        new_cols['insulin_changed'] = ((ins_str == 'Up') | (ins_str == 'Down')).astype(np.float64)
        new_cols['insulin_up'] = (ins_str == 'Up').astype(np.float64)
        new_cols['insulin_down'] = (ins_str == 'Down').astype(np.float64)
        new_cols['insulin_steady'] = (ins_str == 'Steady').astype(np.float64)
        new_cols['insulin_and_oral'] = (ins_act * (oral_active > 0)).astype(np.float64)
    else:
        new_cols['insulin_active'] = np.zeros(n, dtype=np.float64)
        new_cols['insulin_changed'] = np.zeros(n, dtype=np.float64)
        new_cols['insulin_up'] = np.zeros(n, dtype=np.float64)
        new_cols['insulin_down'] = np.zeros(n, dtype=np.float64)
        new_cols['insulin_steady'] = np.zeros(n, dtype=np.float64)
        new_cols['insulin_and_oral'] = np.zeros(n, dtype=np.float64)

    metformin_arr = cols_dict.get('metformin')
    if metformin_arr is not None:
        met_str = metformin_arr.astype(str)
        new_cols['metformin_active'] = ((met_str != 'No') & (met_str != 'None') & (met_str != 'nan') & (met_str != '?')).astype(np.float64)
    else:
        new_cols['metformin_active'] = np.zeros(n, dtype=np.float64)

    change_arr = cols_dict.get('change')
    change_str = change_arr.astype(str) if change_arr is not None else np.array(['No']*n)
    new_cols['med_change_flag'] = (change_str == 'Ch').astype(np.float64)

    diab_med_arr = cols_dict.get('diabetesMed')
    diab_med_str = diab_med_arr.astype(str) if diab_med_arr is not None else np.array(['No']*n)
    new_cols['on_diabetes_med'] = (diab_med_str == 'Yes').astype(np.float64)

    # 7. Lab results & clinical inertia interaction
    max_glu = cols_dict.get('max_glu_serum')
    if max_glu is not None:
        glu_str = max_glu.astype(str)
        new_cols['max_glu_measured'] = ((glu_str != 'None') & (glu_str != 'nan') & (glu_str != '?')).astype(np.float64)
        new_cols['max_glu_high'] = ((glu_str == '>200') | (glu_str == '>300')).astype(np.float64)
    else:
        new_cols['max_glu_measured'] = np.zeros(n, dtype=np.float64)
        new_cols['max_glu_high'] = np.zeros(n, dtype=np.float64)

    a1c = cols_dict.get('A1Cresult')
    if a1c is not None:
        a1c_str = a1c.astype(str)
        a1c_meas = ((a1c_str != 'None') & (a1c_str != 'nan') & (a1c_str != '?')).astype(np.float64)
        a1c_high = ((a1c_str == '>7') | (a1c_str == '>8')).astype(np.float64)
        new_cols['a1c_measured'] = a1c_meas
        new_cols['a1c_high'] = a1c_high
        new_cols['a1c_uncontrolled_no_change'] = (a1c_high * (change_str != 'Ch')).astype(np.float64)
        new_cols['a1c_uncontrolled_changed'] = (a1c_high * (change_str == 'Ch')).astype(np.float64)
    else:
        new_cols['a1c_measured'] = np.zeros(n, dtype=np.float64)
        new_cols['a1c_high'] = np.zeros(n, dtype=np.float64)
        new_cols['a1c_uncontrolled_no_change'] = np.zeros(n, dtype=np.float64)
        new_cols['a1c_uncontrolled_changed'] = np.zeros(n, dtype=np.float64)

    # 8. Admission and Discharge status
    discharge = cols_dict.get('discharge_disposition_id')
    if discharge is not None:
        d_str = discharge.astype(str)
        new_cols['discharged_to_home'] = (d_str == '1').astype(np.float64)
        new_cols['discharged_to_home_health'] = (d_str == '6').astype(np.float64)
        facility_set = {'2', '3', '4', '5', '9', '10', '12', '15', '16', '17', '22', '23', '24', '27', '28', '29', '30'}
        new_cols['discharged_to_facility'] = np.array([1.0 if x in facility_set else 0.0 for x in d_str], dtype=np.float64)
        new_cols['discharged_to_snf'] = (d_str == '3').astype(np.float64)
        new_cols['discharged_left_ama'] = (d_str == '7').astype(np.float64)
    else:
        new_cols['discharged_to_home'] = np.zeros(n, dtype=np.float64)
        new_cols['discharged_to_home_health'] = np.zeros(n, dtype=np.float64)
        new_cols['discharged_to_facility'] = np.zeros(n, dtype=np.float64)
        new_cols['discharged_to_snf'] = np.zeros(n, dtype=np.float64)
        new_cols['discharged_left_ama'] = np.zeros(n, dtype=np.float64)

    adm_src = cols_dict.get('admission_source_id')
    if adm_src is not None:
        s_str = adm_src.astype(str)
        new_cols['admit_from_emergency'] = (s_str == '7').astype(np.float64)
        new_cols['admit_from_referral'] = ((s_str == '1') | (s_str == '2') | (s_str == '3')).astype(np.float64)
        new_cols['admit_from_transfer'] = np.array([1.0 if x in {'4', '5', '6', '10', '18', '22'} else 0.0 for x in s_str], dtype=np.float64)
    else:
        new_cols['admit_from_emergency'] = np.zeros(n, dtype=np.float64)
        new_cols['admit_from_referral'] = np.zeros(n, dtype=np.float64)
        new_cols['admit_from_transfer'] = np.zeros(n, dtype=np.float64)

    adm_type = cols_dict.get('admission_type_id')
    if adm_type is not None:
        t_str = adm_type.astype(str)
        new_cols['admit_type_emergency'] = (t_str == '1').astype(np.float64)
        new_cols['admit_type_urgent'] = (t_str == '2').astype(np.float64)
        new_cols['admit_type_elective'] = (t_str == '3').astype(np.float64)
    else:
        new_cols['admit_type_emergency'] = np.zeros(n, dtype=np.float64)
        new_cols['admit_type_urgent'] = np.zeros(n, dtype=np.float64)
        new_cols['admit_type_elective'] = np.zeros(n, dtype=np.float64)

    med_spec = cols_dict.get('medical_specialty')
    if med_spec is not None:
        spec_str = med_spec.astype(str)
        new_cols['has_medical_specialty'] = ((spec_str != 'None') & (spec_str != 'nan') & (spec_str != '?')).astype(np.float64)
    else:
        new_cols['has_medical_specialty'] = np.zeros(n, dtype=np.float64)

    return new_cols


def compute_oof_target_encoding(
    train_cat: np.ndarray,
    eval_cat: np.ndarray,
    y_train: np.ndarray,
    cv_splits,
    m: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes leak-free out-of-fold target encoding with Empirical Bayes smoothing."""
    n_tr = len(train_cat)
    n_ev = len(eval_cat)
    oof_train = np.zeros(n_tr, dtype=np.float64)
    eval_encoded = np.zeros(n_ev, dtype=np.float64)

    global_mean = float(np.mean(y_train))

    # 1. Out-of-fold for train set
    for tr_idx, val_idx in cv_splits:
        tr_c = train_cat[tr_idx]
        tr_y = y_train[tr_idx]
        val_c = train_cat[val_idx]

        sums = {}
        counts = {}
        for c_val, y_val in zip(tr_c, tr_y):
            sums[c_val] = sums.get(c_val, 0.0) + y_val
            counts[c_val] = counts.get(c_val, 0) + 1

        val_enc = np.empty(len(val_c), dtype=np.float64)
        for i, c_val in enumerate(val_c):
            cnt = counts.get(c_val, 0)
            sm = sums.get(c_val, 0.0)
            val_enc[i] = (sm + m * global_mean) / (cnt + m)
        oof_train[val_idx] = val_enc

    # 2. Full train statistics for eval set
    sums_full = {}
    counts_full = {}
    for c_val, y_val in zip(train_cat, y_train):
        sums_full[c_val] = sums_full.get(c_val, 0.0) + y_val
        counts_full[c_val] = counts_full.get(c_val, 0) + 1

    for i, c_val in enumerate(eval_cat):
        cnt = counts_full.get(c_val, 0)
        sm = sums_full.get(c_val, 0.0)
        eval_encoded[i] = (sm + m * global_mean) / (cnt + m)

    return oof_train, eval_encoded


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return readmission probabilities for `eval_cols`."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    # Identify medication columns
    med_cols = []
    for col, arr in train.items():
        if arr.dtype == object:
            unique_vals = set(np.unique(arr.astype(str)))
            if 'No' in unique_vals and 'Steady' in unique_vals:
                med_cols.append(col)

    # Add engineered features
    train_with_feats = add_features(train, med_cols)
    eval_with_feats = add_features(eval_cols, med_cols)

    # 5-fold split for both target encoding and model training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(train_with_feats['age_num'], y_train))

    # Apply Out-Of-Fold Smoothed Target Encoding to key categorical features
    te_features = [
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code'
    ]

    for feat in te_features:
        tr_raw = np.array([str(x) if x is not None else 'missing' for x in train_with_feats.get(feat, [])], dtype=object)
        ev_raw = np.array([str(x) if x is not None else 'missing' for x in eval_with_feats.get(feat, [])], dtype=object)
        oof_tr, enc_ev = compute_oof_target_encoding(tr_raw, ev_raw, y_train, cv_splits, m=30.0)
        train_with_feats['te_' + feat] = oof_tr
        eval_with_feats['te_' + feat] = enc_ev

    # Diagnosis risk summaries (max risk across all 3 diagnoses is clinically crucial)
    tr_d1, tr_d2, tr_d3 = train_with_feats['te_diag_1_prefix'], train_with_feats['te_diag_2_prefix'], train_with_feats['te_diag_3_prefix']
    ev_d1, ev_d2, ev_d3 = eval_with_feats['te_diag_1_prefix'], eval_with_feats['te_diag_2_prefix'], eval_with_feats['te_diag_3_prefix']

    train_with_feats['te_diag_max'] = np.maximum(tr_d1, np.maximum(tr_d2, tr_d3))
    train_with_feats['te_diag_mean'] = (tr_d1 + tr_d2 + tr_d3) / 3.0
    train_with_feats['te_diag_min'] = np.minimum(tr_d1, np.minimum(tr_d2, tr_d3))

    eval_with_feats['te_diag_max'] = np.maximum(ev_d1, np.maximum(ev_d2, ev_d3))
    eval_with_feats['te_diag_mean'] = (ev_d1 + ev_d2 + ev_d3) / 3.0
    eval_with_feats['te_diag_min'] = np.minimum(ev_d1, np.minimum(ev_d2, ev_d3))

    extra_numeric = [
        'age_num', 'total_visits', 'has_prior_visits', 'has_prior_inpatient', 
        'frequent_inpatient', 'has_prior_emergency', 'inpatient_ratio', 
        'emergency_ratio', 'log_inpatient', 'log_emergency', 'log_total_visits',
        'meds_per_day', 'labs_per_day', 'procs_per_day', 'diags_per_day', 
        'polypharmacy', 'complexity_score', 'log_time_in_hospital',
        'num_uniq_diag_groups', 'has_diabetes_diag', 'has_circulatory_diag',
        'has_chf', 'has_cad', 'has_copd', 'has_ckd', 'has_htn', 
        'has_fluid_disorder', 'has_infection', 'chronic_comorbidity_count',
        'num_active_meds', 'num_med_up', 'num_med_down', 'num_med_changes', 
        'num_oral_meds', 'insulin_active', 'insulin_changed', 'insulin_up', 
        'insulin_down', 'insulin_steady', 'insulin_and_oral', 'metformin_active', 
        'med_change_flag', 'on_diabetes_med', 'max_glu_measured', 'max_glu_high', 
        'a1c_measured', 'a1c_high', 'a1c_uncontrolled_no_change', 
        'a1c_uncontrolled_changed', 'discharged_to_home', 'discharged_to_home_health', 
        'discharged_to_facility', 'discharged_to_snf', 'discharged_left_ama',
        'admit_from_emergency', 'admit_from_referral', 'admit_from_transfer',
        'admit_type_emergency', 'admit_type_urgent', 'admit_type_elective',
        'has_medical_specialty',
        'te_diag_1_prefix', 'te_diag_2_prefix', 'te_diag_3_prefix',
        'te_discharge_disposition_id', 'te_admission_source_id',
        'te_admission_type_id', 'te_medical_specialty', 'te_payer_code',
        'te_diag_max', 'te_diag_mean', 'te_diag_min'
    ]

    extra_categorical = [
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix'
    ]

    encoder = TabularEncoder(min_count=10)
    encoder.fit(
        task, 
        train_with_feats, 
        extra_numeric=extra_numeric, 
        extra_categorical=extra_categorical
    )
    x_train = encoder.transform(train_with_feats)
    x_eval = encoder.transform(eval_with_feats)

    # Multi-architecture diverse LightGBM configuration
    configs = [
        # Configuration 1: Standard tree depth, moderate capacity
        {
            'n_estimators': 240,
            'learning_rate': 0.035,
            'num_leaves': 31,
            'min_child_samples': 45,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.7,
            'reg_lambda': 3.0,
            'reg_alpha': 0.5,
            'random_state': 100,
        },
        # Configuration 2: Shallow, highly regularized trees
        {
            'n_estimators': 260,
            'learning_rate': 0.035,
            'num_leaves': 15,
            'min_child_samples': 30,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.65,
            'reg_lambda': 5.0,
            'reg_alpha': 1.0,
            'random_state': 200,
        },
        # Configuration 3: Deep-constrained, feature-subsampled trees
        {
            'n_estimators': 280,
            'learning_rate': 0.03,
            'num_leaves': 22,
            'max_depth': 6,
            'min_child_samples': 50,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.55,
            'reg_lambda': 8.0,
            'reg_alpha': 2.0,
            'random_state': 300,
        }
    ]

    total_models = len(cv_splits) * len(configs)
    preds = np.zeros(len(x_eval), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        for config in configs:
            cfg = dict(config)
            cfg['random_state'] += fold
            cfg['n_jobs'] = 1
            cfg['verbose'] = -1

            model = lgb.LGBMClassifier(**cfg)
            model.fit(
                x_tr, 
                y_tr, 
                categorical_feature=encoder.categorical_indices
            )
            preds += model.predict_proba(x_eval)[:, 1] / total_models

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
