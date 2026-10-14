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


MED_COLS = (
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
)

AGE_MAP = {
    '[0-10)': 5.0,
    '[10-20)': 15.0,
    '[20-30)': 25.0,
    '[30-40)': 35.0,
    '[40-50)': 45.0,
    '[50-60)': 55.0,
    '[60-70)': 65.0,
    '[70-80)': 75.0,
    '[80-90)': 85.0,
    '[90-100)': 95.0,
}


def _extract_domain_features(cols: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract clinical, utilisation, intensity and demographic features."""
    n = len(cols['time_in_hospital'])
    num_feats: Dict[str, np.ndarray] = {}
    cat_feats: Dict[str, np.ndarray] = {}

    # 1. ICD-9 Chapter groups and 3-digit prefixes
    d1, d2, d3 = cols['diag_1'], cols['diag_2'], cols['diag_3']
    d1_grp = np.array([icd9_group(c) for c in d1], dtype=object)
    d2_grp = np.array([icd9_group(c) for c in d2], dtype=object)
    d3_grp = np.array([icd9_group(c) for c in d3], dtype=object)

    cat_feats['diag_1_group'] = d1_grp
    cat_feats['diag_2_group'] = d2_grp
    cat_feats['diag_3_group'] = d3_grp

    def get_prefix(codes):
        out = np.empty(n, dtype=object)
        for i, c in enumerate(codes):
            if c is not None and isinstance(c, str) and len(c) > 0:
                out[i] = c.split('.')[0]
            else:
                out[i] = None
        return out

    p1 = get_prefix(d1)
    p2 = get_prefix(d2)
    p3 = get_prefix(d3)
    cat_feats['diag_1_prefix'] = p1
    cat_feats['diag_2_prefix'] = p2
    cat_feats['diag_3_prefix'] = p3

    # Broad chapter comorbidity flags
    num_feats['has_circulatory'] = ((d1_grp == 'circulatory') | (d2_grp == 'circulatory') | (d3_grp == 'circulatory')).astype(np.float64)
    num_feats['has_respiratory'] = ((d1_grp == 'respiratory') | (d2_grp == 'respiratory') | (d3_grp == 'respiratory')).astype(np.float64)
    num_feats['has_digestive'] = ((d1_grp == 'digestive') | (d2_grp == 'digestive') | (d3_grp == 'digestive')).astype(np.float64)
    num_feats['has_diabetes_diag'] = ((d1_grp == 'diabetes') | (d2_grp == 'diabetes') | (d3_grp == 'diabetes')).astype(np.float64)
    num_feats['has_injury'] = ((d1_grp == 'injury') | (d2_grp == 'injury') | (d3_grp == 'injury')).astype(np.float64)
    num_feats['has_genitourinary'] = ((d1_grp == 'genitourinary') | (d2_grp == 'genitourinary') | (d3_grp == 'genitourinary')).astype(np.float64)
    num_feats['has_neoplasms'] = ((d1_grp == 'neoplasms') | (d2_grp == 'neoplasms') | (d3_grp == 'neoplasms')).astype(np.float64)

    # Specific high-risk clinical conditions
    def has_any_prefix(prefix_set):
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if p1[i] in prefix_set or p2[i] in prefix_set or p3[i] in prefix_set:
                out[i] = 1.0
        return out

    num_feats['primary_chf'] = (p1 == '428').astype(np.float64)
    num_feats['has_chf'] = has_any_prefix({'428'})
    num_feats['has_cad'] = has_any_prefix({'410', '411', '412', '413', '414'})
    num_feats['has_dysrhythmia'] = has_any_prefix({'427'})
    num_feats['has_hypertension'] = has_any_prefix({'401', '402', '403', '404', '405'})
    num_feats['has_cerebrovascular'] = has_any_prefix({'430', '431', '432', '433', '434', '435', '436', '437', '438'})
    num_feats['has_pneumonia'] = has_any_prefix({'480', '481', '482', '483', '484', '485', '486'})
    num_feats['has_copd'] = has_any_prefix({'491', '492', '496'})
    num_feats['has_renal'] = has_any_prefix({'584', '585', '586'})
    num_feats['has_fluid_electrolyte'] = has_any_prefix({'276'})
    num_feats['has_sepsis'] = has_any_prefix({'038'})
    num_feats['has_ulcer_skin'] = has_any_prefix({'707', '682'})
    num_feats['has_anemia'] = has_any_prefix({'280', '281', '285'})
    num_feats['has_pvd'] = has_any_prefix({'440', '441', '442', '443'})
    num_feats['has_liver'] = has_any_prefix({'571', '572'})

    # Subtype of diabetes complications: acute crises (DKA/coma), renal, neuro, circulatory
    def check_dm_subcode(target_digit):
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for code in (d1[i], d2[i], d3[i]):
                if code is not None and isinstance(code, str) and code.startswith('250.'):
                    parts = code.split('.')
                    if len(parts) > 1 and len(parts[1]) > 0 and parts[1][0] in target_digit:
                        out[i] = 1.0
                        break
        return out

    num_feats['dm_acute_crisis'] = check_dm_subcode({'1', '2', '3'})
    num_feats['dm_renal_comp'] = check_dm_subcode({'4'})
    num_feats['dm_neuro_comp'] = check_dm_subcode({'6'})
    num_feats['dm_circ_comp'] = check_dm_subcode({'7'})

    def is_complicated_diabetes(code):
        if code is not None and isinstance(code, str) and code.startswith('250'):
            if '.' in code:
                parts = code.split('.')
                if len(parts) > 1 and len(parts[1]) > 0 and parts[1][0] != '0':
                    return True
        return False

    comp_dm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if is_complicated_diabetes(d1[i]) or is_complicated_diabetes(d2[i]) or is_complicated_diabetes(d3[i]):
            comp_dm[i] = 1.0
    num_feats['has_complicated_diabetes'] = comp_dm

    # Charlson Comorbidity Index calculation
    cci_1pt = {
        '410', '412', '428', '440', '441', '442', '443', '444', '447',
        '430', '431', '432', '433', '434', '435', '436', '437', '438',
        '290', '490', '491', '492', '493', '494', '495', '496', '500',
        '501', '502', '503', '504', '505', '710', '714', '725', '531',
        '532', '533', '534', '571'
    }
    cci_2pt = {'342', '344', '582', '583', '585', '586'}
    charlson = np.zeros(n, dtype=np.float64)
    for i in range(n):
        score = 0.0
        patient_prefixes = {p1[i], p2[i], p3[i]} - {None}
        for p in patient_prefixes:
            if p in cci_1pt: score += 1.0
            if p in cci_2pt: score += 2.0
            if p in {'572'}: score += 3.0
            if p in {'196', '197', '198', '199', '042'}: score += 6.0
            # Any non-metastatic malignancy (140-172, 174-195, 200-208)
            if p.isdigit() and ((140 <= int(p) <= 172) or (174 <= int(p) <= 195) or (200 <= int(p) <= 208)):
                score += 2.0
        if comp_dm[i] > 0.0:
            score += 2.0
        elif (p1[i] == '250' or p2[i] == '250' or p3[i] == '250'):
            score += 1.0
        charlson[i] = score
    num_feats['charlson_score'] = charlson
    num_feats['high_charlson'] = (charlson >= 3.0).astype(np.float64)

    # Multi-morbidity count and syndromic interactions
    comorb_sum = (
        num_feats['has_chf'] + num_feats['has_cad'] + num_feats['has_hypertension'] +
        num_feats['has_copd'] + num_feats['has_renal'] + num_feats['has_cerebrovascular'] +
        num_feats['has_fluid_electrolyte'] + num_feats['has_ulcer_skin'] + comp_dm
    )
    num_feats['comorbidity_count'] = comorb_sum
    num_feats['cardiorenal'] = (num_feats['has_chf'] * num_feats['has_renal']).astype(np.float64)
    num_feats['diabetic_renal'] = (np.maximum(comp_dm, num_feats['dm_renal_comp']) * num_feats['has_renal']).astype(np.float64)
    num_feats['diabetic_chf'] = (comp_dm * num_feats['has_chf']).astype(np.float64)
    num_feats['copd_heart_disease'] = (num_feats['has_copd'] * np.maximum(num_feats['has_chf'], num_feats['has_cad'])).astype(np.float64)
    num_feats['acute_and_prior_inp'] = (np.maximum(num_feats['dm_acute_crisis'], num_feats['has_sepsis']) * (cols['number_inpatient'].astype(np.float64) > 0.0)).astype(np.float64)

    num_feats['primary_is_diabetes'] = (d1_grp == 'diabetes').astype(np.float64)
    num_feats['primary_is_circulatory'] = (d1_grp == 'circulatory').astype(np.float64)
    num_feats['primary_is_respiratory'] = (d1_grp == 'respiratory').astype(np.float64)

    distinct_grps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        valid = {d1_grp[i], d2_grp[i], d3_grp[i]} - {'missing', 'other', None}
        distinct_grps[i] = float(len(valid))
    num_feats['num_distinct_diag_groups'] = distinct_grps

    # 2. Age as continuous midpoint, polynomial and subgroup interaction terms
    age_raw = cols.get('age')
    age_num = np.empty(n, dtype=np.float64)
    if age_raw is not None:
        for i, a in enumerate(age_raw):
            age_num[i] = AGE_MAP.get(a, np.nan)
    else:
        age_num[:] = np.nan
    num_feats['age_num'] = age_num
    num_feats['is_elderly'] = (age_num >= 70.0).astype(np.float64)
    num_feats['is_very_old'] = (age_num >= 80.0).astype(np.float64)
    num_feats['is_oldest_old'] = (age_num >= 90.0).astype(np.float64)
    num_feats['is_young'] = (age_num < 50.0).astype(np.float64)
    num_feats['age_squared'] = (age_num / 10.0) ** 2

    # 3. Medications intensity, class combinations & dosing changes
    med_cols = [m for m in MED_COLS if m in cols]
    n_act = np.zeros(n, dtype=np.float64)
    n_up = np.zeros(n, dtype=np.float64)
    n_down = np.zeros(n, dtype=np.float64)
    n_steady = np.zeros(n, dtype=np.float64)
    n_chg = np.zeros(n, dtype=np.float64)

    for m in med_cols:
        arr = cols[m]
        for i, v in enumerate(arr):
            if v == 'Steady':
                n_act[i] += 1.0
                n_steady[i] += 1.0
            elif v == 'Up':
                n_act[i] += 1.0
                n_up[i] += 1.0
                n_chg[i] += 1.0
            elif v == 'Down':
                n_act[i] += 1.0
                n_down[i] += 1.0
                n_chg[i] += 1.0

    num_feats['n_active_meds'] = n_act
    num_feats['n_med_up'] = n_up
    num_feats['n_med_down'] = n_down
    num_feats['n_med_steady'] = n_steady
    num_feats['n_med_changes'] = n_chg
    num_feats['med_up_minus_down'] = n_up - n_down

    insulin = cols.get('insulin')
    ins_presc = np.zeros(n, dtype=np.float64)
    ins_chg = np.zeros(n, dtype=np.float64)
    ins_up = np.zeros(n, dtype=np.float64)
    ins_down = np.zeros(n, dtype=np.float64)
    ins_steady = np.zeros(n, dtype=np.float64)
    if insulin is not None:
        for i, v in enumerate(insulin):
            if v in ('Steady', 'Up', 'Down'):
                ins_presc[i] = 1.0
            if v in ('Up', 'Down'):
                ins_chg[i] = 1.0
            if v == 'Up':
                ins_up[i] = 1.0
            elif v == 'Down':
                ins_down[i] = 1.0
            elif v == 'Steady':
                ins_steady[i] = 1.0
    num_feats['ins_prescribed'] = ins_presc
    num_feats['ins_changed'] = ins_chg
    num_feats['ins_up'] = ins_up
    num_feats['ins_down'] = ins_down
    num_feats['ins_steady'] = ins_steady

    def any_med_active(m_list):
        out = np.zeros(n, dtype=np.float64)
        for m in m_list:
            if m in cols:
                for i, v in enumerate(cols[m]):
                    if v in ('Steady', 'Up', 'Down'):
                        out[i] = 1.0
        return out

    num_feats['has_metformin'] = any_med_active(['metformin', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone'])
    num_feats['has_sulfonylurea'] = any_med_active(['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide', 'glyburide-metformin', 'glipizide-metformin'])
    num_feats['has_tzd'] = any_med_active(['pioglitazone', 'rosiglitazone', 'troglitazone', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'])

    has_oral = ((num_feats['has_metformin'] > 0.0) | (num_feats['has_sulfonylurea'] > 0.0) | (num_feats['has_tzd'] > 0.0)).astype(np.float64)
    num_feats['has_oral_med'] = has_oral
    num_feats['insulin_and_oral'] = (ins_presc * has_oral).astype(np.float64)
    num_feats['insulin_only'] = (ins_presc * (1.0 - has_oral)).astype(np.float64)
    num_feats['oral_only'] = ((1.0 - ins_presc) * has_oral).astype(np.float64)

    change_raw = cols.get('change')
    num_feats['change_flag'] = np.array([1.0 if v == 'Ch' else 0.0 for v in change_raw], dtype=np.float64) if change_raw is not None else np.zeros(n)
    dm_raw = cols.get('diabetesMed')
    num_feats['dm_flag'] = np.array([1.0 if v == 'Yes' else 0.0 for v in dm_raw], dtype=np.float64) if dm_raw is not None else np.zeros(n)

    # 4. Utilisation & Hospital Stay Intensity
    t_hosp = np.maximum(cols['time_in_hospital'].astype(np.float64), 1.0)
    n_lab = cols['num_lab_procedures'].astype(np.float64)
    n_proc = cols['num_procedures'].astype(np.float64)
    n_med = cols['num_medications'].astype(np.float64)
    n_diag = cols['number_diagnoses'].astype(np.float64)

    n_out = cols['number_outpatient'].astype(np.float64)
    n_emg = cols['number_emergency'].astype(np.float64)
    n_inp = cols['number_inpatient'].astype(np.float64)

    total_prior = n_out + n_emg + n_inp
    acute_prior = n_inp + n_emg
    num_feats['total_prior_visits'] = total_prior
    num_feats['acute_prior_visits'] = acute_prior
    num_feats['has_prior_inp'] = (n_inp > 0.0).astype(np.float64)
    num_feats['has_prior_emg'] = (n_emg > 0.0).astype(np.float64)
    num_feats['has_prior_out'] = (n_out > 0.0).astype(np.float64)
    num_feats['has_any_prior'] = (total_prior > 0.0).astype(np.float64)
    num_feats['has_acute_prior'] = (acute_prior > 0.0).astype(np.float64)
    num_feats['inp_zero'] = (n_inp == 0.0).astype(np.float64)
    num_feats['inp_one'] = (n_inp == 1.0).astype(np.float64)
    num_feats['inp_two'] = (n_inp == 2.0).astype(np.float64)
    num_feats['inp_severe'] = (n_inp >= 2.0).astype(np.float64)
    num_feats['inp_three_plus'] = (n_inp >= 3.0).astype(np.float64)
    num_feats['inp_four_plus'] = (n_inp >= 4.0).astype(np.float64)
    num_feats['frequent_flyer'] = ((n_inp >= 1.0) & (n_emg >= 1.0)).astype(np.float64)
    num_feats['high_utilizer'] = (total_prior >= 3.0).astype(np.float64)
    num_feats['very_high_utilizer'] = (total_prior >= 6.0).astype(np.float64)
    num_feats['acute_prior_ratio'] = acute_prior / (total_prior + 1.0)

    num_feats['log_inp'] = np.log1p(n_inp)
    num_feats['log_emg'] = np.log1p(n_emg)
    num_feats['log_out'] = np.log1p(n_out)
    num_feats['log_total_prior'] = np.log1p(total_prior)

    num_feats['inp_ratio'] = n_inp / (total_prior + 1.0)
    num_feats['emg_ratio'] = n_emg / (total_prior + 1.0)
    num_feats['prior_score'] = 3.0 * n_inp + 2.0 * n_emg + n_out

    num_feats['med_rate'] = n_med / t_hosp
    num_feats['lab_rate'] = n_lab / t_hosp
    num_feats['proc_rate'] = n_proc / t_hosp
    num_feats['diag_rate'] = n_diag / t_hosp
    total_proc = n_lab + n_proc
    num_feats['total_proc'] = total_proc
    num_feats['total_intensity'] = (n_med + total_proc) / t_hosp
    num_feats['diag_per_med'] = n_diag / (n_med + 1.0)
    num_feats['inp_to_time'] = n_inp / t_hosp
    num_feats['lab_to_med'] = n_lab / (n_med + 1.0)

    num_feats['stay_x_inp'] = t_hosp * n_inp
    num_feats['stay_x_med'] = t_hosp * n_med
    num_feats['is_short_stay'] = (t_hosp <= 2.0).astype(np.float64)
    num_feats['is_long_stay'] = (t_hosp >= 7.0).astype(np.float64)
    num_feats['care_load'] = n_lab + 2.0 * n_proc + 1.5 * n_med
    num_feats['care_load_rate'] = (n_lab + 2.0 * n_proc + 1.5 * n_med) / t_hosp
    num_feats['complexity_score'] = (n_med / 10.0) + (n_lab / 20.0) + n_proc + (2.0 * n_inp)

    num_feats['age_x_inp'] = (age_num / 10.0) * n_inp
    num_feats['age_x_stay'] = (age_num / 10.0) * t_hosp

    gender = cols.get('gender')
    num_feats['is_female'] = np.array([1.0 if g == 'Female' else 0.0 for g in gender], dtype=np.float64) if gender is not None else np.zeros(n)
    num_feats['is_male'] = np.array([1.0 if g == 'Male' else 0.0 for g in gender], dtype=np.float64) if gender is not None else np.zeros(n)

    # 5. Lab results
    a1c = cols.get('A1Cresult')
    a1c_tested = np.zeros(n, dtype=np.float64)
    a1c_high = np.zeros(n, dtype=np.float64)
    a1c_norm = np.zeros(n, dtype=np.float64)
    if a1c is not None:
        for i, v in enumerate(a1c):
            if v is not None and v != 'None':
                a1c_tested[i] = 1.0
                if v in ('>7', '>8'):
                    a1c_high[i] = 1.0
                elif v == 'Norm':
                    a1c_norm[i] = 1.0
    num_feats['a1c_tested'] = a1c_tested
    num_feats['a1c_high'] = a1c_high
    num_feats['a1c_norm'] = a1c_norm

    glu = cols.get('max_glu_serum')
    glu_tested = np.zeros(n, dtype=np.float64)
    glu_high = np.zeros(n, dtype=np.float64)
    glu_norm = np.zeros(n, dtype=np.float64)
    if glu is not None:
        for i, v in enumerate(glu):
            if v is not None and v != 'None':
                glu_tested[i] = 1.0
                if v in ('>200', '>300'):
                    glu_high[i] = 1.0
                elif v == 'Norm':
                    glu_norm[i] = 1.0
    num_feats['glu_tested'] = glu_tested
    num_feats['glu_high'] = glu_high
    num_feats['glu_norm'] = glu_norm
    num_feats['no_lab_tested'] = ((a1c_tested == 0.0) & (glu_tested == 0.0)).astype(np.float64)

    # Strack et al. 2014 Glycemic Testing & Medication Change clinical interactions
    chg = num_feats['change_flag']
    num_feats['a1c_tested_and_med_changed'] = (a1c_tested * chg).astype(np.float64)
    num_feats['a1c_tested_and_med_not_changed'] = (a1c_tested * (1.0 - chg)).astype(np.float64)
    num_feats['a1c_high_and_med_not_changed'] = (a1c_high * (1.0 - chg)).astype(np.float64)
    num_feats['a1c_high_and_med_changed'] = (a1c_high * chg).astype(np.float64)
    num_feats['no_a1c_and_med_changed'] = ((1.0 - a1c_tested) * chg).astype(np.float64)
    num_feats['no_a1c_and_med_not_changed'] = ((1.0 - a1c_tested) * (1.0 - chg)).astype(np.float64)
    num_feats['glu_high_and_med_not_changed'] = (glu_high * (1.0 - chg)).astype(np.float64)

    # 6. Admission / Discharge / Provider indicators
    disp = cols.get('discharge_disposition_id')
    dis_home = np.zeros(n, dtype=np.float64)
    dis_snf = np.zeros(n, dtype=np.float64)
    dis_hh = np.zeros(n, dtype=np.float64)
    dis_ama = np.zeros(n, dtype=np.float64)
    dis_oth = np.zeros(n, dtype=np.float64)
    dis_not_home = np.zeros(n, dtype=np.float64)
    if disp is not None:
        for i, v in enumerate(disp):
            if v == '1':
                dis_home[i] = 1.0
            elif v == '3':
                dis_snf[i] = 1.0
            elif v == '6':
                dis_hh[i] = 1.0
            elif v == '7':
                dis_ama[i] = 1.0
            elif v in ('2', '4', '5', '22', '23', '24'):
                dis_oth[i] = 1.0
            if v is not None and v != '1':
                dis_not_home[i] = 1.0
    num_feats['discharged_home'] = dis_home
    num_feats['discharged_snf'] = dis_snf
    num_feats['discharged_home_health'] = dis_hh
    num_feats['discharged_ama'] = dis_ama
    num_feats['discharged_other_facility'] = dis_oth
    num_feats['discharged_not_home'] = dis_not_home

    # Disposition interactions with age and prior inpatient status
    num_feats['snf_elderly'] = (dis_snf * (age_num >= 70.0)).astype(np.float64)
    num_feats['hh_elderly'] = (dis_hh * (age_num >= 70.0)).astype(np.float64)
    num_feats['home_with_prior_inp'] = (dis_home * (n_inp > 0.0)).astype(np.float64)
    num_feats['not_home_with_prior_inp'] = (dis_not_home * (n_inp > 0.0)).astype(np.float64)

    adm_type = cols.get('admission_type_id')
    adm_emg = np.zeros(n, dtype=np.float64)
    adm_urg = np.zeros(n, dtype=np.float64)
    adm_elec = np.zeros(n, dtype=np.float64)
    if adm_type is not None:
        for i, v in enumerate(adm_type):
            if v == '1': adm_emg[i] = 1.0
            elif v == '2': adm_urg[i] = 1.0
            elif v == '3': adm_elec[i] = 1.0
    num_feats['adm_emergency'] = adm_emg
    num_feats['adm_urgent'] = adm_urg
    num_feats['adm_elective'] = adm_elec

    adm_src = cols.get('admission_source_id')
    adm_er = np.zeros(n, dtype=np.float64)
    adm_tr = np.zeros(n, dtype=np.float64)
    if adm_src is not None:
        for i, v in enumerate(adm_src):
            if v == '7': adm_er[i] = 1.0
            elif v in ('4', '5', '6', '10', '18', '22'): adm_tr[i] = 1.0
    num_feats['adm_from_er'] = adm_er
    num_feats['adm_transfer'] = adm_tr

    # Interaction: Emergency admission from ER
    num_feats['emg_adm_from_er'] = (adm_emg * adm_er).astype(np.float64)

    payer = cols.get('payer_code')
    num_feats['has_payer'] = np.array([1.0 if v is not None else 0.0 for v in payer], dtype=np.float64) if payer is not None else np.zeros(n)
    num_feats['payer_medicare'] = np.array([1.0 if v == 'MC' else 0.0 for v in payer], dtype=np.float64) if payer is not None else np.zeros(n)
    num_feats['payer_medicaid'] = np.array([1.0 if v == 'MD' else 0.0 for v in payer], dtype=np.float64) if payer is not None else np.zeros(n)
    num_feats['payer_selfpay'] = np.array([1.0 if v == 'SP' else 0.0 for v in payer], dtype=np.float64) if payer is not None else np.zeros(n)

    spec = cols.get('medical_specialty')
    num_feats['has_specialty'] = np.array([1.0 if v is not None else 0.0 for v in spec], dtype=np.float64) if spec is not None else np.zeros(n)

    return num_feats, cat_feats


def _compute_pooled_diag_te(
    train_c: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_c: Dict[str, np.ndarray],
    smooth: float = 30.0,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute unified out-of-fold diagnosis risk by pooling diag_1..3."""
    p1_tr = train_c['diag_1_prefix']
    p2_tr = train_c['diag_2_prefix']
    p3_tr = train_c['diag_3_prefix']
    p1_ev = eval_c['diag_1_prefix']
    p2_ev = eval_c['diag_2_prefix']
    p3_ev = eval_c['diag_3_prefix']

    n_tr = len(y_train)
    prior = float(np.mean(y_train))
    oof_d1 = np.empty(n_tr, dtype=np.float64)
    oof_d2 = np.empty(n_tr, dtype=np.float64)
    oof_d3 = np.empty(n_tr, dtype=np.float64)

    rng = np.random.RandomState(seed)
    indices = np.arange(n_tr)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        trn_idx = np.setdiff1d(indices, val_idx)
        stats: Dict[object, List[int]] = {}
        for idx in trn_idx:
            yi = y_train[idx]
            prefixes = {p1_tr[idx], p2_tr[idx], p3_tr[idx]} - {None}
            for p in prefixes:
                if p not in stats:
                    stats[p] = [0, 0]
                stats[p][0] += 1
                stats[p][1] += yi

        def get_risk(p):
            if p is not None and p in stats:
                cnt, sy = stats[p]
                return (sy + smooth * prior) / (cnt + smooth)
            return prior

        for idx in val_idx:
            oof_d1[idx] = get_risk(p1_tr[idx])
            oof_d2[idx] = get_risk(p2_tr[idx])
            oof_d3[idx] = get_risk(p3_tr[idx])

    full_stats: Dict[object, List[int]] = {}
    for idx in range(n_tr):
        yi = y_train[idx]
        prefixes = {p1_tr[idx], p2_tr[idx], p3_tr[idx]} - {None}
        for p in prefixes:
            if p not in full_stats:
                full_stats[p] = [0, 0]
            full_stats[p][0] += 1
            full_stats[p][1] += yi

    def get_full_risk(p):
        if p is not None and p in full_stats:
            cnt, sy = full_stats[p]
            return (sy + smooth * prior) / (cnt + smooth)
        return prior

    n_ev = len(p1_ev)
    ev_d1 = np.empty(n_ev, dtype=np.float64)
    ev_d2 = np.empty(n_ev, dtype=np.float64)
    ev_d3 = np.empty(n_ev, dtype=np.float64)
    for idx in range(n_ev):
        ev_d1[idx] = get_full_risk(p1_ev[idx])
        ev_d2[idx] = get_full_risk(p2_ev[idx])
        ev_d3[idx] = get_full_risk(p3_ev[idx])

    tr_diag_feats = {
        'te_pooled_diag1': oof_d1,
        'te_pooled_diag2': oof_d2,
        'te_pooled_diag3': oof_d3,
        'te_pooled_diag_max': np.maximum(oof_d1, np.maximum(oof_d2, oof_d3)),
        'te_pooled_diag_min': np.minimum(oof_d1, np.minimum(oof_d2, oof_d3)),
        'te_pooled_diag_mean': (oof_d1 + oof_d2 + oof_d3) / 3.0,
        'te_pooled_diag_sum': oof_d1 + oof_d2 + oof_d3,
    }
    ev_diag_feats = {
        'te_pooled_diag1': ev_d1,
        'te_pooled_diag2': ev_d2,
        'te_pooled_diag3': ev_d3,
        'te_pooled_diag_max': np.maximum(ev_d1, np.maximum(ev_d2, ev_d3)),
        'te_pooled_diag_min': np.minimum(ev_d1, np.minimum(ev_d2, ev_d3)),
        'te_pooled_diag_mean': (ev_d1 + ev_d2 + ev_d3) / 3.0,
        'te_pooled_diag_sum': ev_d1 + ev_d2 + ev_d3,
    }
    return tr_diag_feats, ev_diag_feats


def _compute_oof_te(
    train_cats: np.ndarray,
    y_train: np.ndarray,
    eval_cats: np.ndarray,
    smooth: float = 30.0,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Missingness-aware out-of-fold target encoding with shrinkage."""
    n_train = len(train_cats)
    n_eval = len(eval_cats)
    oof_train = np.empty(n_train, dtype=np.float64)
    oof_eval = np.empty(n_eval, dtype=np.float64)
    prior = float(np.mean(y_train))

    def normalize_val(v):
        if v is None:
            return '__MISSING__'
        if isinstance(v, float) and np.isnan(v):
            return '__MISSING__'
        return str(v)

    norm_tr = [normalize_val(v) for v in train_cats]
    norm_ev = [normalize_val(v) for v in eval_cats]

    rng = np.random.RandomState(seed)
    indices = np.arange(n_train)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        trn_idx = np.setdiff1d(indices, val_idx)
        stats: Dict[str, List[int]] = {}
        for idx in trn_idx:
            c = norm_tr[idx]
            if c not in stats:
                stats[c] = [0, 0]
            stats[c][0] += 1
            stats[c][1] += y_train[idx]

        for idx in val_idx:
            c = norm_tr[idx]
            if c in stats:
                cnt, sy = stats[c]
                oof_train[idx] = (sy + smooth * prior) / (cnt + smooth)
            else:
                oof_train[idx] = prior

    full_stats: Dict[str, List[int]] = {}
    for idx in range(n_train):
        c = norm_tr[idx]
        if c not in full_stats:
            full_stats[c] = [0, 0]
        full_stats[c][0] += 1
        full_stats[c][1] += y_train[idx]

    for idx in range(n_eval):
        c = norm_ev[idx]
        if c in full_stats:
            cnt, sy = full_stats[c]
            oof_eval[idx] = (sy + smooth * prior) / (cnt + smooth)
        else:
            oof_eval[idx] = prior

    return oof_train, oof_eval


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit engineered gradient-boosted ensemble and return readmission probabilities."""
    train_c = dict(train)
    eval_c = dict(eval_cols)

    # 1. Extract domain features
    tr_num, tr_cat = _extract_domain_features(train_c)
    ev_num, ev_cat = _extract_domain_features(eval_c)

    for k, v in tr_num.items():
        train_c[k] = v
        eval_c[k] = ev_num[k]

    for k, v in tr_cat.items():
        train_c[k] = v
        eval_c[k] = ev_cat[k]

    # 2. Pooled diagnosis target encoding
    tr_pooled_te, ev_pooled_te = _compute_pooled_diag_te(train_c, y_train, eval_c, smooth=30.0)
    for k, v in tr_pooled_te.items():
        train_c[k] = v
        eval_c[k] = ev_pooled_te[k]
        tr_num[k] = v

    # Interaction categorical columns
    train_c['disp_adm_type'] = np.array([
        f"{d}_{a}" if (d is not None and a is not None) else None
        for d, a in zip(train_c['discharge_disposition_id'], train_c['admission_type_id'])
    ], dtype=object)
    eval_c['disp_adm_type'] = np.array([
        f"{d}_{a}" if (d is not None and a is not None) else None
        for d, a in zip(eval_c['discharge_disposition_id'], eval_c['admission_type_id'])
    ], dtype=object)
    tr_cat['disp_adm_type'] = train_c['disp_adm_type']
    ev_cat['disp_adm_type'] = eval_c['disp_adm_type']

    train_c['age_disp'] = np.array([
        f"{a}_{d}" if (a is not None and d is not None) else None
        for a, d in zip(train_c['age'], train_c['discharge_disposition_id'])
    ], dtype=object)
    eval_c['age_disp'] = np.array([
        f"{a}_{d}" if (a is not None and d is not None) else None
        for a, d in zip(eval_c['age'], eval_c['discharge_disposition_id'])
    ], dtype=object)
    tr_cat['age_disp'] = train_c['age_disp']
    ev_cat['age_disp'] = eval_c['age_disp']

    # Interaction between discharge disposition and prior inpatient count
    train_c['disp_has_inp'] = np.array([
        f"{d}_inp{int(inp > 0)}" if d is not None else None
        for d, inp in zip(train_c['discharge_disposition_id'], train_c['number_inpatient'])
    ], dtype=object)
    eval_c['disp_has_inp'] = np.array([
        f"{d}_inp{int(inp > 0)}" if d is not None else None
        for d, inp in zip(eval_c['discharge_disposition_id'], eval_c['number_inpatient'])
    ], dtype=object)
    tr_cat['disp_has_inp'] = train_c['disp_has_inp']
    ev_cat['disp_has_inp'] = eval_c['disp_has_inp']

    # Interaction between primary diagnosis group and discharge disposition
    train_c['diag1_grp_disp'] = np.array([
        f"{g}_{d}" if (g is not None and d is not None) else None
        for g, d in zip(train_c['diag_1_group'], train_c['discharge_disposition_id'])
    ], dtype=object)
    eval_c['diag1_grp_disp'] = np.array([
        f"{g}_{d}" if (g is not None and d is not None) else None
        for g, d in zip(eval_c['diag_1_group'], eval_c['discharge_disposition_id'])
    ], dtype=object)
    tr_cat['diag1_grp_disp'] = train_c['diag1_grp_disp']
    ev_cat['diag1_grp_disp'] = eval_c['diag1_grp_disp']

    # Target encodings & frequency features
    te_cols = [
        ('diag_1_prefix', 'te_diag1', 30.0),
        ('diag_2_prefix', 'te_diag2', 30.0),
        ('diag_3_prefix', 'te_diag3', 30.0),
        ('discharge_disposition_id', 'te_disp', 25.0),
        ('admission_source_id', 'te_adm_src', 25.0),
        ('admission_type_id', 'te_adm_type', 25.0),
        ('medical_specialty', 'te_spec', 35.0),
        ('disp_adm_type', 'te_disp_adm_type', 35.0),
        ('age_disp', 'te_age_disp', 35.0),
        ('disp_has_inp', 'te_disp_has_inp', 30.0),
        ('diag1_grp_disp', 'te_diag1_grp_disp', 30.0),
        ('payer_code', 'te_payer', 30.0),
    ]

    for col_name, te_name, smooth in te_cols:
        tr_te, ev_te = _compute_oof_te(train_c[col_name], y_train, eval_c[col_name], smooth=smooth)
        train_c[te_name] = tr_te
        eval_c[te_name] = ev_te
        tr_num[te_name] = tr_te

    # Diagnosis risk aggregates from primary TE
    mean_te_tr = (train_c['te_diag1'] + train_c['te_diag2'] + train_c['te_diag3']) / 3.0
    mean_te_ev = (eval_c['te_diag1'] + eval_c['te_diag2'] + eval_c['te_diag3']) / 3.0
    max_te_tr = np.maximum(train_c['te_diag1'], np.maximum(train_c['te_diag2'], train_c['te_diag3']))
    max_te_ev = np.maximum(eval_c['te_diag1'], np.maximum(eval_c['te_diag2'], eval_c['te_diag3']))
    min_te_tr = np.minimum(train_c['te_diag1'], np.minimum(train_c['te_diag2'], train_c['te_diag3']))
    min_te_ev = np.minimum(eval_c['te_diag1'], np.minimum(eval_c['te_diag2'], eval_c['te_diag3']))

    train_c['te_diag_mean'] = mean_te_tr
    eval_c['te_diag_mean'] = mean_te_ev
    train_c['te_diag_max'] = max_te_tr
    eval_c['te_diag_max'] = max_te_ev
    train_c['te_diag_min'] = min_te_tr
    eval_c['te_diag_min'] = min_te_ev
    tr_num['te_diag_mean'] = mean_te_tr
    tr_num['te_diag_max'] = max_te_tr
    tr_num['te_diag_min'] = min_te_tr

    # Frequency encodings
    freq_cols = ['medical_specialty', 'diag_1_prefix', 'discharge_disposition_id', 'admission_source_id', 'disp_adm_type', 'payer_code', 'diag1_grp_disp']
    n_tr = len(y_train)
    for col_name in freq_cols:
        f_name = f'freq_{col_name}'
        freq_map: Dict[object, float] = {}
        for val in train_c[col_name]:
            if val is not None:
                freq_map[val] = freq_map.get(val, 0.0) + 1.0 / n_tr
        train_c[f_name] = np.array([freq_map.get(v, 0.0) for v in train_c[col_name]], dtype=np.float64)
        eval_c[f_name] = np.array([freq_map.get(v, 0.0) for v in eval_c[col_name]], dtype=np.float64)
        tr_num[f_name] = train_c[f_name]

    extra_numeric = list(tr_num.keys())
    extra_categorical = list(tr_cat.keys())

    # 3. Encode tabular features
    encoder = TabularEncoder(min_count=15)
    x_train = encoder.fit_transform(
        task,
        train_c,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    x_eval = encoder.transform(eval_c)

    # 4. Multi-model Bagged LightGBM Ensemble blending GBDT, Extra-Trees, and Class-Weighted structures
    model_configs = [
        {
            'n_estimators': 330,
            'learning_rate': 0.030,
            'num_leaves': 22,
            'max_depth': 5,
            'min_child_samples': 45,
            'subsample': 0.75,
            'subsample_freq': 1,
            'colsample_bytree': 0.65,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 42,
        },
        {
            'extra_trees': True,
            'n_estimators': 340,
            'learning_rate': 0.030,
            'num_leaves': 28,
            'max_depth': 6,
            'min_child_samples': 50,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.75,
            'reg_alpha': 1.0,
            'reg_lambda': 4.0,
            'cat_smooth': 20.0,
            'cat_l2': 10.0,
            'random_state': 123,
        },
        {
            'n_estimators': 310,
            'learning_rate': 0.035,
            'num_leaves': 15,
            'max_depth': 4,
            'min_child_samples': 60,
            'subsample': 0.70,
            'subsample_freq': 1,
            'colsample_bytree': 0.80,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'cat_smooth': 30.0,
            'cat_l2': 15.0,
            'random_state': 456,
        },
        {
            'extra_trees': True,
            'n_estimators': 320,
            'learning_rate': 0.030,
            'num_leaves': 20,
            'max_depth': 5,
            'min_child_samples': 40,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.60,
            'reg_alpha': 0.8,
            'reg_lambda': 5.0,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 789,
        },
        {
            'n_estimators': 350,
            'learning_rate': 0.025,
            'num_leaves': 30,
            'max_depth': 6,
            'min_child_samples': 65,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.58,
            'reg_alpha': 1.5,
            'reg_lambda': 6.0,
            'cat_smooth': 20.0,
            'cat_l2': 10.0,
            'random_state': 999,
        },
        {
            'n_estimators': 310,
            'learning_rate': 0.030,
            'scale_pos_weight': 1.25,
            'num_leaves': 24,
            'max_depth': 5,
            'min_child_samples': 45,
            'subsample': 0.78,
            'subsample_freq': 1,
            'colsample_bytree': 0.68,
            'reg_alpha': 0.7,
            'reg_lambda': 4.0,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 1101,
        },
        {
            'extra_trees': True,
            'n_estimators': 310,
            'learning_rate': 0.032,
            'num_leaves': 24,
            'max_depth': 5,
            'min_child_samples': 45,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.70,
            'reg_alpha': 0.6,
            'reg_lambda': 3.5,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 2024,
        },
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for cfg in model_configs:
        clf = lgb.LGBMClassifier(
            **cfg,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += clf.predict_proba(x_eval)[:, 1]

    preds /= len(model_configs)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
