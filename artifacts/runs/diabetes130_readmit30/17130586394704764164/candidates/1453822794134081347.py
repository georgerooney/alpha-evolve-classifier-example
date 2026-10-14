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
    n_tr = len(y_train)
    n_ev = len(eval_cols['time_in_hospital'])

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities via icd9_comorbidities
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in cm_tr:
        train_data[k] = cm_tr[k].astype(np.float64)
        eval_data[k] = cm_ev[k].astype(np.float64)
        extra_numeric.append(k)

    # Weighted Charlson comorbidity index & critical interactions
    def calc_weighted_cci(cm):
        return (
            cm['cm_mi'] + cm['cm_chf'] + cm['cm_pvd'] + cm['cm_cerebrovascular']
            + cm['cm_dementia'] + cm['cm_pulmonary'] + cm['cm_rheumatic']
            + cm['cm_peptic_ulcer'] + cm['cm_liver']
            + 2.0 * cm['cm_hemiplegia'] + 2.0 * cm['cm_renal']
            + 2.0 * cm['cm_cancer'] + 2.0 * cm['cm_diabetes_complicated']
            + 6.0 * cm['cm_metastatic'] + 6.0 * cm['cm_hiv']
        ).astype(np.float64)

    train_data['weighted_cci'] = calc_weighted_cci(cm_tr)
    eval_data['weighted_cci'] = calc_weighted_cci(cm_ev)
    extra_numeric.append('weighted_cci')

    train_data['cardiorenal'] = (cm_tr['cm_chf'] * cm_tr['cm_renal']).astype(np.float64)
    eval_data['cardiorenal'] = (cm_ev['cm_chf'] * cm_ev['cm_renal']).astype(np.float64)
    extra_numeric.append('cardiorenal')

    train_data['pulmonary_chf'] = (cm_tr['cm_pulmonary'] * cm_tr['cm_chf']).astype(np.float64)
    eval_data['pulmonary_chf'] = (cm_ev['cm_pulmonary'] * cm_ev['cm_chf']).astype(np.float64)
    extra_numeric.append('pulmonary_chf')

    # 2. ICD-9 Chapter groups & 3-digit clinical prefixes
    def get_diag_prefix(col):
        return np.array([str(c).split('.')[0] if (c is not None and c is not np.nan) else 'missing' for c in col], dtype=object)

    for diag_field in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f"{diag_field}_group"
        train_data[grp_name] = np.array([icd9_group(c) for c in train[diag_field]], dtype=object)
        eval_data[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_field]], dtype=object)
        extra_categorical.append(grp_name)

        pref_name = f"{diag_field}_prefix"
        train_data[pref_name] = get_diag_prefix(train[diag_field])
        eval_data[pref_name] = get_diag_prefix(eval_cols[diag_field])
        extra_categorical.append(pref_name)

    # 3. High-risk condition presence across all 3 diagnosis slots
    def extract_condition_flags(d):
        n = len(d['diag_1'])
        s_cols = [[str(c) if c is not None else '' for c in d[f]] for f in ['diag_1', 'diag_2', 'diag_3']]

        def has_any_prefix(prefixes):
            res = np.zeros(n, dtype=np.float64)
            for sc in s_cols:
                arr = np.array([any(s.startswith(p) for p in prefixes) for s in sc], dtype=np.float64)
                res = np.maximum(res, arr)
            return res

        chf = has_any_prefix(('428',))
        ckd_arf = has_any_prefix(('584', '585', '586'))
        copd = has_any_prefix(('491', '492', '493', '494', '496'))
        fluid_lyte = has_any_prefix(('276',))
        cad = has_any_prefix(('410', '411', '412', '413', '414'))
        pneumonia = has_any_prefix(('480', '481', '482', '483', '484', '485', '486'))
        htn = has_any_prefix(('401', '402', '403', '404', '405'))
        comp_dm = has_any_prefix(('250.4', '250.5', '250.6', '250.7', '250.8', '250.9'))

        return {
            'cond_has_chf': chf,
            'cond_has_ckd_arf': ckd_arf,
            'cond_has_copd': copd,
            'cond_has_fluid_lyte': fluid_lyte,
            'cond_has_cad': cad,
            'cond_has_pneumonia': pneumonia,
            'cond_has_htn': htn,
            'cond_has_comp_dm': comp_dm,
            'cond_chf_ckd': chf * ckd_arf,
            'cond_chf_copd': chf * copd,
        }

    cond_tr = extract_condition_flags(train)
    cond_ev = extract_condition_flags(eval_cols)
    for k in cond_tr:
        train_data[k] = cond_tr[k]
        eval_data[k] = cond_ev[k]
        extra_numeric.append(k)

    # 4. Multi-way Clinical Composites
    inp_tr_bkt = np.minimum(np.nan_to_num(train['number_inpatient'], nan=0.0).astype(int), 3)
    inp_ev_bkt = np.minimum(np.nan_to_num(eval_cols['number_inpatient'], nan=0.0).astype(int), 3)

    train_data['disch_admit_combo'] = np.array(
        [f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_type_id'])],
        dtype=object
    )
    eval_data['disch_admit_combo'] = np.array(
        [f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_type_id'])],
        dtype=object
    )
    extra_categorical.append('disch_admit_combo')

    train_data['disch_inp_combo'] = np.array(
        [f"{d}_{b}" for d, b in zip(train['discharge_disposition_id'], inp_tr_bkt)],
        dtype=object
    )
    eval_data['disch_inp_combo'] = np.array(
        [f"{d}_{b}" for d, b in zip(eval_cols['discharge_disposition_id'], inp_ev_bkt)],
        dtype=object
    )
    extra_categorical.append('disch_inp_combo')

    train_data['disch_diag1_combo'] = np.array(
        [f"{d}_{g}" for d, g in zip(train['discharge_disposition_id'], train_data['diag_1_group'])],
        dtype=object
    )
    eval_data['disch_diag1_combo'] = np.array(
        [f"{d}_{g}" for d, g in zip(eval_cols['discharge_disposition_id'], eval_data['diag_1_group'])],
        dtype=object
    )
    extra_categorical.append('disch_diag1_combo')

    train_data['age_disch_combo'] = np.array(
        [f"{a}_{d}" for a, d in zip(train['age'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_data['age_disch_combo'] = np.array(
        [f"{a}_{d}" for a, d in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    extra_categorical.append('age_disch_combo')

    # 5. Out-of-fold target encoding of high-cardinality and interaction columns
    te_columns = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code',
        'diag_1_group',
        'disch_admit_combo',
        'disch_inp_combo',
        'disch_diag1_combo',
        'age_disch_combo',
    ]
    for col in te_columns:
        te_tr, te_ev = oof_target_encode(
            train_data[col], y_train, eval_data[col], n_folds=5, smoothing=20.0, seed=42
        )
        col_te = f"{col}_te"
        train_data[col_te] = te_tr.astype(np.float64)
        eval_data[col_te] = te_ev.astype(np.float64)
        extra_numeric.append(col_te)

    # Summaries of diagnosis risk
    train_data['diag_te_max'] = np.maximum(
        train_data['diag_1_te'], np.maximum(train_data['diag_2_te'], train_data['diag_3_te'])
    )
    eval_data['diag_te_max'] = np.maximum(
        eval_data['diag_1_te'], np.maximum(eval_data['diag_2_te'], eval_data['diag_3_te'])
    )
    extra_numeric.append('diag_te_max')

    train_data['diag_pref_te_max'] = np.maximum(
        train_data['diag_1_prefix_te'], np.maximum(train_data['diag_2_prefix_te'], train_data['diag_3_prefix_te'])
    )
    eval_data['diag_pref_te_max'] = np.maximum(
        eval_data['diag_1_prefix_te'], np.maximum(eval_data['diag_2_prefix_te'], eval_data['diag_3_prefix_te'])
    )
    extra_numeric.append('diag_pref_te_max')

    train_data['diag_te_mean'] = (
        train_data['diag_1_te'] + train_data['diag_2_te'] + train_data['diag_3_te']
    ) / 3.0
    eval_data['diag_te_mean'] = (
        eval_data['diag_1_te'] + eval_data['diag_2_te'] + eval_data['diag_3_te']
    ) / 3.0
    extra_numeric.append('diag_te_mean')

    train_data['diag_pref_te_mean'] = (
        train_data['diag_1_prefix_te'] + train_data['diag_2_prefix_te'] + train_data['diag_3_prefix_te']
    ) / 3.0
    eval_data['diag_pref_te_mean'] = (
        eval_data['diag_1_prefix_te'] + eval_data['diag_2_prefix_te'] + eval_data['diag_3_prefix_te']
    ) / 3.0
    extra_numeric.append('diag_pref_te_mean')

    # 6. Demographics
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    age_tr = np.array([age_map.get(a, 60.0) for a in train['age']], dtype=np.float64)
    age_ev = np.array([age_map.get(a, 60.0) for a in eval_cols['age']], dtype=np.float64)
    train_data['age_num'] = age_tr
    eval_data['age_num'] = age_ev
    extra_numeric.append('age_num')

    train_data['is_elderly'] = (age_tr >= 75.0).astype(np.float64)
    eval_data['is_elderly'] = (age_ev >= 75.0).astype(np.float64)
    extra_numeric.append('is_elderly')

    train_data['gender_is_female'] = (train['gender'] == 'Female').astype(np.float64)
    eval_data['gender_is_female'] = (eval_cols['gender'] == 'Female').astype(np.float64)
    extra_numeric.append('gender_is_female')

    # 7. Prior utilisation features
    def make_utilization_features(d):
        inpatient = np.nan_to_num(d['number_inpatient'], nan=0.0).astype(np.float64)
        emergency = np.nan_to_num(d['number_emergency'], nan=0.0).astype(np.float64)
        outpatient = np.nan_to_num(d['number_outpatient'], nan=0.0).astype(np.float64)
        total_vis = inpatient + emergency + outpatient
        return {
            'has_prior_inpatient': (inpatient > 0).astype(np.float64),
            'frequent_inpatient': (inpatient >= 2).astype(np.float64),
            'has_emergency': (emergency > 0).astype(np.float64),
            'has_outpatient': (outpatient > 0).astype(np.float64),
            'total_prior_visits': total_vis,
            'log_inpatient': np.log1p(inpatient),
            'log_emergency': np.log1p(emergency),
            'log_outpatient': np.log1p(outpatient),
            'log_total_visits': np.log1p(total_vis),
            'inpatient_visit_fraction': (inpatient + 0.1) / (total_vis + 1.0),
            'emergency_visit_fraction': (emergency + 0.1) / (total_vis + 1.0),
            'emergency_to_outpatient_ratio': (emergency + 0.1) / (outpatient + 0.1),
        }

    util_tr = make_utilization_features(train)
    util_ev = make_utilization_features(eval_cols)
    for k in util_tr:
        train_data[k] = util_tr[k]
        eval_data[k] = util_ev[k]
        extra_numeric.append(k)

    # 8. Clinical LACE and LACE+ Readmission Risk Indices
    def compute_lace(los_arr, admit_arr, cci_arr, emer_arr, inp_arr, age_arr):
        l_pts = np.zeros_like(los_arr)
        l_pts[los_arr == 1.0] = 1.0
        l_pts[los_arr == 2.0] = 2.0
        l_pts[los_arr == 3.0] = 3.0
        l_pts[(los_arr >= 4.0) & (los_arr <= 6.0)] = 4.0
        l_pts[(los_arr >= 7.0) & (los_arr <= 13.0)] = 5.0
        l_pts[los_arr >= 14.0] = 7.0

        a_pts = np.isin(admit_arr, ['1', '2']).astype(np.float64) * 3.0
        c_pts = np.minimum(cci_arr, 3.0)
        c_pts[cci_arr >= 4.0] = 5.0
        e_pts = np.minimum(emer_arr, 4.0)

        lace = l_pts + a_pts + c_pts + e_pts
        lace_plus = lace + 2.0 * np.minimum(inp_arr, 4.0) + (age_arr >= 70.0).astype(np.float64) * 2.0
        return lace, lace_plus

    los_tr_raw = np.maximum(np.nan_to_num(train['time_in_hospital'], nan=1.0), 1.0).astype(np.float64)
    los_ev_raw = np.maximum(np.nan_to_num(eval_cols['time_in_hospital'], nan=1.0), 1.0).astype(np.float64)
    inp_tr_num = np.nan_to_num(train['number_inpatient'], nan=0.0).astype(np.float64)
    inp_ev_num = np.nan_to_num(eval_cols['number_inpatient'], nan=0.0).astype(np.float64)
    em_tr_num = np.nan_to_num(train['number_emergency'], nan=0.0).astype(np.float64)
    em_ev_num = np.nan_to_num(eval_cols['number_emergency'], nan=0.0).astype(np.float64)

    lace_tr, lace_plus_tr = compute_lace(
        los_tr_raw, train['admission_type_id'], train_data['weighted_cci'],
        em_tr_num, inp_tr_num, age_tr
    )
    lace_ev, lace_plus_ev = compute_lace(
        los_ev_raw, eval_cols['admission_type_id'], eval_data['weighted_cci'],
        em_ev_num, inp_ev_num, age_ev
    )
    train_data['lace_index'] = lace_tr
    eval_data['lace_index'] = lace_ev
    extra_numeric.append('lace_index')

    train_data['lace_plus_index'] = lace_plus_tr
    eval_data['lace_plus_index'] = lace_plus_ev
    extra_numeric.append('lace_plus_index')

    # 9. Stay intensity & complexity ratios
    def make_intensity_features(d):
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 1.0).astype(np.float64)
        meds = np.nan_to_num(d['num_medications'], nan=0.0).astype(np.float64)
        labs = np.nan_to_num(d['num_lab_procedures'], nan=0.0).astype(np.float64)
        procs = np.nan_to_num(d['num_procedures'], nan=0.0).astype(np.float64)
        diags = np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0).astype(np.float64)
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0).astype(np.float64)
        return {
            'meds_per_day': meds / los,
            'labs_per_day': labs / los,
            'procs_per_day': procs / los,
            'meds_per_diag': meds / diags,
            'labs_per_med': labs / np.maximum(meds, 1.0),
            'procs_per_med': procs / np.maximum(meds, 1.0),
            'los_times_inpatient': los * inp,
            'diags_times_inpatient': diags * inp,
            'inpatient_per_day': inp / los,
            'clinical_complexity': np.log1p(los) * np.log1p(meds) * np.log1p(diags),
        }

    int_tr = make_intensity_features(train)
    int_ev = make_intensity_features(eval_cols)
    for k in int_tr:
        train_data[k] = int_tr[k]
        eval_data[k] = int_ev[k]
        extra_numeric.append(k)

    # 10. Medications patterns & Regimen Complexity
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in med_names if m in train]
    sulfo_meds = [m for m in ['glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'acetohexamide', 'tolbutamide', 'tolazamide'] if m in train]

    def extract_med_features(d, n_rows):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        has_sulfo = np.zeros(n_rows, dtype=np.float64)

        for m in avail_meds:
            col = d[m]
            is_up = (col == 'Up')
            is_down = (col == 'Down')
            is_steady = (col == 'Steady')
            up_cnt += is_up.astype(np.float64)
            down_cnt += is_down.astype(np.float64)
            active_cnt += (is_up | is_down | is_steady).astype(np.float64)

        for s in sulfo_meds:
            col = d[s]
            has_sulfo = np.maximum(has_sulfo, np.isin(col, ['Steady', 'Up', 'Down']).astype(np.float64))

        ins = d['insulin']
        ins_up = (ins == 'Up').astype(np.float64)
        ins_down = (ins == 'Down').astype(np.float64)
        ins_steady = (ins == 'Steady').astype(np.float64)
        ins_active = ((ins == 'Up') | (ins == 'Down') | (ins == 'Steady')).astype(np.float64)
        ins_num = np.zeros(n_rows, dtype=np.float64)
        ins_num[ins == 'Down'] = -1.0
        ins_num[ins == 'Steady'] = 1.0
        ins_num[ins == 'Up'] = 2.0

        met = d['metformin']
        met_active = ((met == 'Up') | (met == 'Down') | (met == 'Steady')).astype(np.float64)

        chg = (d['change'] == 'Ch').astype(np.float64)
        dm_med = (d['diabetesMed'] == 'Yes').astype(np.float64)

        # Regimen: 0=none, 1=oral only, 2=insulin only, 3=both insulin & oral
        regimen = np.zeros(n_rows, dtype=np.float64)
        regimen[(ins_active == 0) & (active_cnt > 0)] = 1.0
        regimen[(ins_active == 1) & (active_cnt == 1)] = 2.0
        regimen[(ins_active == 1) & (active_cnt > 1)] = 3.0

        return {
            'num_active_meds': active_cnt,
            'num_meds_up': up_cnt,
            'num_meds_down': down_cnt,
            'num_meds_changed': up_cnt + down_cnt,
            'insulin_is_up': ins_up,
            'insulin_is_down': ins_down,
            'insulin_is_steady': ins_steady,
            'insulin_is_active': ins_active,
            'insulin_numeric': ins_num,
            'metformin_is_active': met_active,
            'has_sulfonylurea': has_sulfo,
            'insulin_sulfo_combo': ins_active * has_sulfo,
            'regimen_type': regimen,
            'med_change_flag': chg,
            'has_diabetes_med': dm_med,
        }

    med_tr = extract_med_features(train, n_tr)
    med_ev = extract_med_features(eval_cols, n_ev)
    for k in med_tr:
        train_data[k] = med_tr[k]
        eval_data[k] = med_ev[k]
        extra_numeric.append(k)

    # 11. Lab results, Strack glycemic interactions, and disposition flags
    def extract_lab_and_disp_features(d):
        a1c = d['A1Cresult']
        a1c_tested = ((a1c != 'None') & (a1c != None)).astype(np.float64)
        a1c_high = ((a1c == '>8') | (a1c == '>7')).astype(np.float64)
        a1c_gt8 = (a1c == '>8').astype(np.float64)
        chg = (d['change'] == 'Ch').astype(np.float64)

        glu = d['max_glu_serum']
        glu_tested = ((glu != 'None') & (glu != None)).astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)

        disch = d['discharge_disposition_id']
        disch_home = (disch == '1').astype(np.float64)
        disch_snf = (disch == '3').astype(np.float64)
        disch_homehealth = (disch == '6').astype(np.float64)
        disch_ama = (disch == '7').astype(np.float64)
        disch_transferred = np.isin(disch, ['2', '4', '5', '18', '22', '23', '24']).astype(np.float64)

        admit = d['admission_type_id']
        admit_er = (admit == '1').astype(np.float64)
        admit_elective = (admit == '3').astype(np.float64)

        source = d['admission_source_id']
        source_er = (source == '7').astype(np.float64)
        source_transfer = np.isin(source, ['4', '5', '6']).astype(np.float64)

        payer_miss = np.isin(d['payer_code'], [None, '?']).astype(np.float64)
        spec_miss = np.isin(d['medical_specialty'], [None, '?']).astype(np.float64)

        return {
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'a1c_gt8': a1c_gt8,
            'a1c_high_no_change': a1c_high * (1.0 - chg),
            'a1c_tested_and_changed': a1c_tested * chg,
            'glu_tested': glu_tested,
            'glu_high': glu_high,
            'disch_home': disch_home,
            'disch_snf': disch_snf,
            'disch_homehealth': disch_homehealth,
            'disch_ama': disch_ama,
            'disch_transferred': disch_transferred,
            'admit_er': admit_er,
            'admit_elective': admit_elective,
            'source_er': source_er,
            'source_transfer': source_transfer,
            'payer_miss': payer_miss,
            'spec_miss': spec_miss,
        }

    lab_tr = extract_lab_and_disp_features(train)
    lab_ev = extract_lab_and_disp_features(eval_cols)
    for k in lab_tr:
        train_data[k] = lab_tr[k]
        eval_data[k] = lab_ev[k]
        extra_numeric.append(k)

    # De-duplicate feature names
    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 12. Encoding with min_count=20
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 13. Diverse LightGBM ensemble combining Standard & Extremely Randomized Trees
    models = [
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.5,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=340,
            learning_rate=0.025,
            num_leaves=48,
            min_child_samples=90,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=5.0,
            reg_alpha=0.5,
            cat_smooth=70.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=250,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=0.5,
            reg_alpha=0.0,
            cat_smooth=35.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=28,
            min_child_samples=120,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.45,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=60.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=38,
            min_child_samples=120,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            extra_trees=True,
            reg_lambda=3.0,
            reg_alpha=0.2,
            cat_smooth=50.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(n_ev, dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]
    preds /= len(models)

    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
