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


def _fallback_icd9_group(code) -> str:
    if code is None or code == '?' or code == 'None' or code == '':
        return 'missing'
    s = str(code).strip()
    if not s or s == '?':
        return 'missing'
    if s.startswith('V') or s.startswith('E'):
        return 'other'
    try:
        num = float(s.split('.')[0])
    except ValueError:
        return 'other'
    if 390 <= num <= 459 or num == 785:
        return 'circulatory'
    if 460 <= num <= 519 or num == 786:
        return 'respiratory'
    if 520 <= num <= 579 or num == 787:
        return 'digestive'
    if 250 <= num < 251:
        return 'diabetes'
    if 800 <= num <= 999:
        return 'injury'
    if 710 <= num <= 739:
        return 'musculoskeletal'
    if 580 <= num <= 629 or num == 788:
        return 'genitourinary'
    if 140 <= num <= 239:
        return 'neoplasms'
    return 'other'


def _compute_pooled_te(
    prefixes_list: Sequence[np.ndarray],
    y: np.ndarray,
    prior: float,
    smooth: float = 30.0,
) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    sums: Dict[str, float] = {}
    for p_arr in prefixes_list:
        for p, y_val in zip(p_arr, y):
            if p == 'missing':
                continue
            counts[p] = counts.get(p, 0) + 1
            sums[p] = sums.get(p, 0.0) + float(y_val)
    return {
        p: (sums[p] + smooth * prior) / (cnt + smooth)
        for p, cnt in counts.items()
    }


def _compute_single_te(
    arr: np.ndarray,
    y: np.ndarray,
    prior: float,
    smooth: float = 30.0,
) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    sums: Dict[str, float] = {}
    for val, y_val in zip(arr, y):
        s = str(val) if (val is not None and val != '?' and val != '') else 'missing'
        counts[s] = counts.get(s, 0) + 1
        sums[s] = sums.get(s, 0.0) + float(y_val)
    return {
        s: (sums[s] + smooth * prior) / (cnt + smooth)
        for s, cnt in counts.items()
    }


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import sklearn.model_selection

    n_tr = len(y_train)
    n_ev = len(eval_cols['time_in_hospital'])
    global_prior = float(np.mean(y_train))

    train_data = dict(train)
    eval_data = dict(eval_cols)

    extra_num_cols: List[str] = []
    extra_cat_cols: List[str] = []

    # 1. Healthcare Utilization Intensity
    for name, cols, n_rows in [('train', train_data, n_tr), ('eval', eval_data, n_ev)]:
        inp = np.nan_to_num(cols['number_inpatient'].astype(np.float64), nan=0.0)
        emg = np.nan_to_num(cols['number_emergency'].astype(np.float64), nan=0.0)
        out = np.nan_to_num(cols['number_outpatient'].astype(np.float64), nan=0.0)

        total_prior = inp + emg + out
        cols['ext_total_prior_visits'] = total_prior
        cols['ext_has_prior_inpatient'] = (inp > 0).astype(np.float64)
        cols['ext_has_prior_emergency'] = (emg > 0).astype(np.float64)
        cols['ext_has_prior_outpatient'] = (out > 0).astype(np.float64)
        cols['ext_has_any_prior_visit'] = (total_prior > 0).astype(np.float64)
        cols['ext_inpatient_ratio'] = inp / (total_prior + 1.0)
        cols['ext_emergency_ratio'] = emg / (total_prior + 1.0)
        cols['ext_weighted_prior_visits'] = inp * 3.0 + emg * 1.5 + out * 0.5
        cols['ext_log_prior_inpatient'] = np.log1p(np.maximum(inp, 0.0))
        cols['ext_log_prior_emergency'] = np.log1p(np.maximum(emg, 0.0))
        cols['ext_log_total_prior'] = np.log1p(np.maximum(total_prior, 0.0))

    util_keys = [
        'ext_total_prior_visits', 'ext_has_prior_inpatient', 'ext_has_prior_emergency',
        'ext_has_prior_outpatient', 'ext_has_any_prior_visit', 'ext_inpatient_ratio',
        'ext_emergency_ratio', 'ext_weighted_prior_visits', 'ext_log_prior_inpatient',
        'ext_log_prior_emergency', 'ext_log_total_prior',
    ]
    extra_num_cols.extend(util_keys)

    # 2. In-Hospital Stay & Intervention Intensity
    for cols in (train_data, eval_data):
        los = np.maximum(np.nan_to_num(cols['time_in_hospital'].astype(np.float64), nan=1.0), 1.0)
        n_labs = np.nan_to_num(cols['num_lab_procedures'].astype(np.float64), nan=0.0)
        n_procs = np.nan_to_num(cols['num_procedures'].astype(np.float64), nan=0.0)
        n_meds = np.nan_to_num(cols['num_medications'].astype(np.float64), nan=0.0)
        n_diags = np.nan_to_num(cols['number_diagnoses'].astype(np.float64), nan=0.0)

        cols['ext_labs_per_day'] = n_labs / los
        cols['ext_meds_per_day'] = n_meds / los
        cols['ext_procs_per_day'] = n_procs / los
        cols['ext_diags_per_day'] = n_diags / los
        cols['ext_total_stay_intensity'] = n_labs + n_procs * 4.0 + n_meds * 2.0

    stay_keys = [
        'ext_labs_per_day', 'ext_meds_per_day', 'ext_procs_per_day',
        'ext_diags_per_day', 'ext_total_stay_intensity',
    ]
    extra_num_cols.extend(stay_keys)

    # 3. Medications & Regimen Dynamics
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]
    active_meds = [m for m in med_names if m in train_data]

    for cols, n_rows in ((train_data, n_tr), (eval_data, n_ev)):
        num_active = np.zeros(n_rows, dtype=np.float64)
        num_up = np.zeros(n_rows, dtype=np.float64)
        num_down = np.zeros(n_rows, dtype=np.float64)
        for m in active_meds:
            arr = cols[m]
            is_up = (arr == 'Up')
            is_down = (arr == 'Down')
            is_steady = (arr == 'Steady')
            num_active += (is_up | is_down | is_steady).astype(np.float64)
            num_up += is_up.astype(np.float64)
            num_down += is_down.astype(np.float64)

        cols['ext_med_num_active'] = num_active
        cols['ext_med_num_up'] = num_up
        cols['ext_med_num_down'] = num_down
        cols['ext_med_num_change'] = num_up + num_down

        if 'insulin' in cols:
            ins = cols['insulin']
            cols['ext_med_insulin_active'] = np.isin(ins, ['Steady', 'Up', 'Down']).astype(np.float64)
            cols['ext_med_insulin_up'] = (ins == 'Up').astype(np.float64)
            cols['ext_med_insulin_down'] = (ins == 'Down').astype(np.float64)
            cols['ext_med_insulin_steady'] = (ins == 'Steady').astype(np.float64)
        else:
            cols['ext_med_insulin_active'] = np.zeros(n_rows, dtype=np.float64)
            cols['ext_med_insulin_up'] = np.zeros(n_rows, dtype=np.float64)
            cols['ext_med_insulin_down'] = np.zeros(n_rows, dtype=np.float64)
            cols['ext_med_insulin_steady'] = np.zeros(n_rows, dtype=np.float64)

        ch_arr = cols.get('change', None)
        cols['ext_has_med_change'] = (ch_arr == 'Ch').astype(np.float64) if ch_arr is not None else np.zeros(n_rows, dtype=np.float64)

        dm_arr = cols.get('diabetesMed', None)
        cols['ext_on_diabetes_med'] = (dm_arr == 'Yes').astype(np.float64) if dm_arr is not None else np.zeros(n_rows, dtype=np.float64)

    med_keys = [
        'ext_med_num_active', 'ext_med_num_up', 'ext_med_num_down', 'ext_med_num_change',
        'ext_med_insulin_active', 'ext_med_insulin_up', 'ext_med_insulin_down',
        'ext_med_insulin_steady', 'ext_has_med_change', 'ext_on_diabetes_med',
    ]
    extra_num_cols.extend(med_keys)

    # 4. Demographics, Lab Results, and Admission/Discharge Indicators
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    for cols, n_rows in ((train_data, n_tr), (eval_data, n_ev)):
        age_arr = cols.get('age', None)
        if age_arr is not None:
            cols['ext_age_numeric'] = np.array([age_map.get(str(x), np.nan) for x in age_arr], dtype=np.float64)
        else:
            cols['ext_age_numeric'] = np.full(n_rows, np.nan, dtype=np.float64)

        a1c = cols.get('A1Cresult', None)
        if a1c is not None:
            a1c_tested = np.array([1.0 if x not in ('None', None, '', '?') else 0.0 for x in a1c], dtype=np.float64)
            a1c_abnormal = np.array([1.0 if x in ('>7', '>8') else 0.0 for x in a1c], dtype=np.float64)
            a1c_high = np.array([1.0 if x == '>8' else 0.0 for x in a1c], dtype=np.float64)
        else:
            a1c_tested = np.zeros(n_rows, dtype=np.float64)
            a1c_abnormal = np.zeros(n_rows, dtype=np.float64)
            a1c_high = np.zeros(n_rows, dtype=np.float64)

        cols['ext_a1c_tested'] = a1c_tested
        cols['ext_a1c_abnormal'] = a1c_abnormal
        cols['ext_a1c_high'] = a1c_high

        glu = cols.get('max_glu_serum', None)
        if glu is not None:
            glu_tested = np.array([1.0 if x not in ('None', None, '', '?') else 0.0 for x in glu], dtype=np.float64)
            glu_high = np.array([1.0 if x in ('>200', '>300') else 0.0 for x in glu], dtype=np.float64)
        else:
            glu_tested = np.zeros(n_rows, dtype=np.float64)
            glu_high = np.zeros(n_rows, dtype=np.float64)

        cols['ext_glu_tested'] = glu_tested
        cols['ext_glu_high'] = glu_high

        ch_arr = cols.get('change', None)
        no_change = (ch_arr == 'No').astype(np.float64) if ch_arr is not None else np.zeros(n_rows, dtype=np.float64)
        cols['ext_high_a1c_no_change'] = a1c_abnormal * no_change

        dis = cols.get('discharge_disposition_id', None)
        if dis is not None:
            cols['ext_discharge_home'] = (dis == '1').astype(np.float64)
            cols['ext_discharge_facility'] = np.isin(dis, ['2', '3', '4', '5', '6', '22', '23', '24', '28']).astype(np.float64)
        else:
            cols['ext_discharge_home'] = np.zeros(n_rows, dtype=np.float64)
            cols['ext_discharge_facility'] = np.zeros(n_rows, dtype=np.float64)

        adm_src = cols.get('admission_source_id', None)
        cols['ext_admission_from_er'] = (adm_src == '7').astype(np.float64) if adm_src is not None else np.zeros(n_rows, dtype=np.float64)

        adm_typ = cols.get('admission_type_id', None)
        if adm_typ is not None:
            cols['ext_admission_emergency'] = (adm_typ == '1').astype(np.float64)
            cols['ext_admission_elective'] = (adm_typ == '3').astype(np.float64)
        else:
            cols['ext_admission_emergency'] = np.zeros(n_rows, dtype=np.float64)
            cols['ext_admission_elective'] = np.zeros(n_rows, dtype=np.float64)

    demo_keys = [
        'ext_age_numeric', 'ext_a1c_tested', 'ext_a1c_abnormal', 'ext_a1c_high',
        'ext_glu_tested', 'ext_glu_high', 'ext_high_a1c_no_change',
        'ext_discharge_home', 'ext_discharge_facility', 'ext_admission_from_er',
        'ext_admission_emergency', 'ext_admission_elective',
    ]
    extra_num_cols.extend(demo_keys)

    # 5. Diagnosis Prefix, Chapter Grouping, and Chronic Comorbidity Flags
    icd9_mapper = globals().get('icd9_group', _fallback_icd9_group)

    p_train: Dict[str, np.ndarray] = {}
    p_eval: Dict[str, np.ndarray] = {}

    for d_col in ('diag_1', 'diag_2', 'diag_3'):
        p_key = f'ext_{d_col}_prefix'
        g_key = f'ext_{d_col}_group'

        for cols, n_rows, p_dict in ((train_data, n_tr, p_train), (eval_data, n_ev, p_eval)):
            raw = cols.get(d_col, None)
            if raw is None:
                p_arr = np.full(n_rows, 'missing', dtype=object)
                g_arr = np.full(n_rows, 'missing', dtype=object)
            else:
                p_list: List[str] = []
                g_list: List[str] = []
                for val in raw:
                    if val is None or val == '?' or val == 'None' or val == '':
                        p_list.append('missing')
                        g_list.append('missing')
                    else:
                        s = str(val).strip()
                        if not s or s == '?':
                            p_list.append('missing')
                            g_list.append('missing')
                        else:
                            p_list.append(s.split('.')[0])
                            g_list.append(icd9_mapper(s))
                p_arr = np.array(p_list, dtype=object)
                g_arr = np.array(g_list, dtype=object)

            cols[p_key] = p_arr
            cols[g_key] = g_arr
            p_dict[d_col] = p_arr

        extra_cat_cols.extend([p_key, g_key])

    # Chronic comorbidity disease flags across all 3 diagnoses
    chf_codes = {'428'}
    cad_codes = {'410', '411', '412', '413', '414'}
    htn_codes = {'401', '402', '403', '404', '405'}
    copd_codes = {'490', '491', '492', '493', '494', '495', '496'}
    renal_codes = {'582', '583', '585', '586', '588'}
    fluid_codes = {'276'}
    pna_codes = {'480', '481', '482', '483', '484', '485', '486'}
    stroke_codes = {'430', '431', '432', '433', '434', '435', '436', '437', '438'}

    for cols, n_rows, p_dict in ((train_data, n_tr, p_train), (eval_data, n_ev, p_eval)):
        p1, p2, p3 = p_dict['diag_1'], p_dict['diag_2'], p_dict['diag_3']

        f_dm = np.zeros(n_rows, dtype=np.float64)
        f_chf = np.zeros(n_rows, dtype=np.float64)
        f_cad = np.zeros(n_rows, dtype=np.float64)
        f_htn = np.zeros(n_rows, dtype=np.float64)
        f_copd = np.zeros(n_rows, dtype=np.float64)
        f_renal = np.zeros(n_rows, dtype=np.float64)
        f_fluid = np.zeros(n_rows, dtype=np.float64)
        f_pna = np.zeros(n_rows, dtype=np.float64)
        f_stroke = np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            d1, d2, d3 = p1[i], p2[i], p3[i]
            d_set = {d1, d2, d3}
            if d1.startswith('250') or d2.startswith('250') or d3.startswith('250'):
                f_dm[i] = 1.0
            if d_set & chf_codes:
                f_chf[i] = 1.0
            if d_set & cad_codes:
                f_cad[i] = 1.0
            if d_set & htn_codes:
                f_htn[i] = 1.0
            if d_set & copd_codes:
                f_copd[i] = 1.0
            if d_set & renal_codes:
                f_renal[i] = 1.0
            if d_set & fluid_codes:
                f_fluid[i] = 1.0
            if d_set & pna_codes:
                f_pna[i] = 1.0
            if d_set & stroke_codes:
                f_stroke[i] = 1.0

        cols['ext_flag_dm'] = f_dm
        cols['ext_flag_chf'] = f_chf
        cols['ext_flag_cad'] = f_cad
        cols['ext_flag_htn'] = f_htn
        cols['ext_flag_copd'] = f_copd
        cols['ext_flag_renal'] = f_renal
        cols['ext_flag_fluid'] = f_fluid
        cols['ext_flag_pna'] = f_pna
        cols['ext_flag_stroke'] = f_stroke
        cols['ext_chronic_comorbidity_count'] = (
            f_dm + f_chf + f_cad + f_htn + f_copd + f_renal + f_fluid + f_pna + f_stroke
        )

    flag_keys = [
        'ext_flag_dm', 'ext_flag_chf', 'ext_flag_cad', 'ext_flag_htn', 'ext_flag_copd',
        'ext_flag_renal', 'ext_flag_fluid', 'ext_flag_pna', 'ext_flag_stroke',
        'ext_chronic_comorbidity_count',
    ]
    extra_num_cols.extend(flag_keys)

    # 6. Out-of-Fold Target Encoding (Pooled Diagnoses, Discharge, Specialty)
    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    diag_te_tr_1 = np.full(n_tr, global_prior, dtype=np.float64)
    diag_te_tr_2 = np.full(n_tr, global_prior, dtype=np.float64)
    diag_te_tr_3 = np.full(n_tr, global_prior, dtype=np.float64)
    discharge_te_tr = np.full(n_tr, global_prior, dtype=np.float64)
    specialty_te_tr = np.full(n_tr, global_prior, dtype=np.float64)

    p_tr_1 = p_train['diag_1']
    p_tr_2 = p_train['diag_2']
    p_tr_3 = p_train['diag_3']
    dis_tr = train_data.get('discharge_disposition_id', np.array([]))
    spec_tr = train_data.get('medical_specialty', np.array([]))

    for trn_idx, val_idx in skf.split(np.zeros(n_tr), y_train):
        fold_y = y_train[trn_idx]
        fold_prior = float(np.mean(fold_y))

        d_map = _compute_pooled_te([p_tr_1[trn_idx], p_tr_2[trn_idx], p_tr_3[trn_idx]], fold_y, fold_prior, smooth=30.0)
        for i in val_idx:
            diag_te_tr_1[i] = d_map.get(p_tr_1[i], fold_prior)
            diag_te_tr_2[i] = d_map.get(p_tr_2[i], fold_prior)
            diag_te_tr_3[i] = d_map.get(p_tr_3[i], fold_prior)

        if len(dis_tr) == n_tr:
            dis_map = _compute_single_te(dis_tr[trn_idx], fold_y, fold_prior, smooth=30.0)
            for i in val_idx:
                s = str(dis_tr[i]) if (dis_tr[i] is not None and dis_tr[i] != '?' and dis_tr[i] != '') else 'missing'
                discharge_te_tr[i] = dis_map.get(s, fold_prior)

        if len(spec_tr) == n_tr:
            spec_map = _compute_single_te(spec_tr[trn_idx], fold_y, fold_prior, smooth=30.0)
            for i in val_idx:
                s = str(spec_tr[i]) if (spec_tr[i] is not None and spec_tr[i] != '?' and spec_tr[i] != '') else 'missing'
                specialty_te_tr[i] = spec_map.get(s, fold_prior)

    # Full train target encoding maps for eval
    full_d_map = _compute_pooled_te([p_tr_1, p_tr_2, p_tr_3], y_train, global_prior, smooth=30.0)
    p_ev_1 = p_eval['diag_1']
    p_ev_2 = p_eval['diag_2']
    p_ev_3 = p_eval['diag_3']

    diag_te_ev_1 = np.array([full_d_map.get(p, global_prior) for p in p_ev_1], dtype=np.float64)
    diag_te_ev_2 = np.array([full_d_map.get(p, global_prior) for p in p_ev_2], dtype=np.float64)
    diag_te_ev_3 = np.array([full_d_map.get(p, global_prior) for p in p_ev_3], dtype=np.float64)

    dis_ev = eval_data.get('discharge_disposition_id', np.array([]))
    if len(dis_tr) == n_tr and len(dis_ev) == n_ev:
        full_dis_map = _compute_single_te(dis_tr, y_train, global_prior, smooth=30.0)
        discharge_te_ev = np.array(
            [full_dis_map.get(str(x) if (x is not None and x != '?' and x != '') else 'missing', global_prior) for x in dis_ev],
            dtype=np.float64,
        )
    else:
        discharge_te_ev = np.full(n_ev, global_prior, dtype=np.float64)

    spec_ev = eval_data.get('medical_specialty', np.array([]))
    if len(spec_tr) == n_tr and len(spec_ev) == n_ev:
        full_spec_map = _compute_single_te(spec_tr, y_train, global_prior, smooth=30.0)
        specialty_te_ev = np.array(
            [full_spec_map.get(str(x) if (x is not None and x != '?' and x != '') else 'missing', global_prior) for x in spec_ev],
            dtype=np.float64,
        )
    else:
        specialty_te_ev = np.full(n_ev, global_prior, dtype=np.float64)

    train_data['ext_diag_1_te'] = diag_te_tr_1
    train_data['ext_diag_2_te'] = diag_te_tr_2
    train_data['ext_diag_3_te'] = diag_te_tr_3
    train_data['ext_diag_max_te'] = np.maximum(diag_te_tr_1, np.maximum(diag_te_tr_2, diag_te_tr_3))
    train_data['ext_diag_mean_te'] = (diag_te_tr_1 + diag_te_tr_2 + diag_te_tr_3) / 3.0
    train_data['ext_discharge_te'] = discharge_te_tr
    train_data['ext_specialty_te'] = specialty_te_tr

    eval_data['ext_diag_1_te'] = diag_te_ev_1
    eval_data['ext_diag_2_te'] = diag_te_ev_2
    eval_data['ext_diag_3_te'] = diag_te_ev_3
    eval_data['ext_diag_max_te'] = np.maximum(diag_te_ev_1, np.maximum(diag_te_ev_2, diag_te_ev_3))
    eval_data['ext_diag_mean_te'] = (diag_te_ev_1 + diag_te_ev_2 + diag_te_ev_3) / 3.0
    eval_data['ext_discharge_te'] = discharge_te_ev
    eval_data['ext_specialty_te'] = specialty_te_ev

    te_keys = [
        'ext_diag_1_te', 'ext_diag_2_te', 'ext_diag_3_te', 'ext_diag_max_te',
        'ext_diag_mean_te', 'ext_discharge_te', 'ext_specialty_te',
    ]
    extra_num_cols.extend(te_keys)

    # 7. Tabular Encoding
    encoder = TabularEncoder(min_count=15)
    encoder.fit(
        task,
        train_data,
        extra_numeric=tuple(extra_num_cols),
        extra_categorical=tuple(extra_cat_cols),
    )
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 8. 5-Fold Stratified LightGBM Ensemble + Full Model
    cv_preds = np.zeros(n_ev, dtype=np.float64)

    for fold, (trn_idx, _) in enumerate(skf.split(x_train, y_train)):
        model_fold = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=25,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.3,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_fold.fit(x_train[trn_idx], y_train[trn_idx], categorical_feature=cat_indices)
        cv_preds += model_fold.predict_proba(x_eval)[:, 1] / 5.0

    full_model = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=28,
        max_depth=6,
        min_child_samples=55,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.3,
        reg_lambda=4.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    full_model.fit(x_train, y_train, categorical_feature=cat_indices)
    full_preds = full_model.predict_proba(x_eval)[:, 1]

    final_preds = 0.75 * cv_preds + 0.25 * full_preds
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
