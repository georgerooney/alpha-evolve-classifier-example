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


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    from sklearn.model_selection import StratifiedKFold

    # 1. Prepare shallow copies of dictionary to register derived features
    tr = dict(train)
    ev = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    def register_num(name: str, arr_tr: np.ndarray, arr_ev: np.ndarray):
        tr[name] = arr_tr.astype(np.float64)
        ev[name] = arr_ev.astype(np.float64)
        extra_numeric.append(name)

    def register_cat(name: str, arr_tr: np.ndarray, arr_ev: np.ndarray):
        tr[name] = arr_tr.astype(object)
        ev[name] = arr_ev.astype(object)
        extra_categorical.append(name)

    n_train = len(y_train)
    n_eval = len(eval_cols['time_in_hospital'])

    # 2. Age parsing into numeric midpoint
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0
    }
    def parse_age(age_arr):
        return np.array([age_map.get(str(x), 65.0) for x in age_arr], dtype=np.float64)

    age_tr = parse_age(train['age'])
    age_ev = parse_age(eval_cols['age'])
    register_num('age_num', age_tr, age_ev)
    register_num('is_elderly', (age_tr >= 65.0).astype(np.float64), (age_ev >= 65.0).astype(np.float64))

    gender_female_tr = (train.get('gender') == 'Female').astype(np.float64)
    gender_female_ev = (eval_cols.get('gender') == 'Female').astype(np.float64)
    register_num('gender_female', gender_female_tr, gender_female_ev)
    register_num('elderly_female', ((age_tr >= 65.0) & (gender_female_tr == 1.0)).astype(np.float64), ((age_ev >= 65.0) & (gender_female_ev == 1.0)).astype(np.float64))
    register_num('elderly_male', ((age_tr >= 65.0) & (gender_female_tr == 0.0)).astype(np.float64), ((age_ev >= 65.0) & (gender_female_ev == 0.0)).astype(np.float64))

    # 3. ICD-9 Chapter Groupings and Clinical Comorbidities
    def get_diag_groups(codes):
        res = []
        for c in codes:
            if c is None or (isinstance(c, float) and np.isnan(c)):
                res.append('missing')
            else:
                res.append(icd9_group(str(c)))
        return np.array(res, dtype=object)

    d1_tr, d1_ev = get_diag_groups(train['diag_1']), get_diag_groups(eval_cols['diag_1'])
    d2_tr, d2_ev = get_diag_groups(train['diag_2']), get_diag_groups(eval_cols['diag_2'])
    d3_tr, d3_ev = get_diag_groups(train['diag_3']), get_diag_groups(eval_cols['diag_3'])

    register_cat('diag_1_group', d1_tr, d1_ev)
    register_cat('diag_2_group', d2_tr, d2_ev)
    register_cat('diag_3_group', d3_tr, d3_ev)

    def get_3digit_prefix(codes):
        res = []
        for c in codes:
            if c is None or (isinstance(c, float) and np.isnan(c)):
                res.append('missing')
            else:
                res.append(str(c).strip().split('.')[0])
        return np.array(res, dtype=object)

    d1_pfx_tr, d1_pfx_ev = get_3digit_prefix(train['diag_1']), get_3digit_prefix(eval_cols['diag_1'])
    register_cat('diag_1_3digit', d1_pfx_tr, d1_pfx_ev)

    def extract_diag_flags(d1, d2, d3):
        circ = ((d1 == 'circulatory') | (d2 == 'circulatory') | (d3 == 'circulatory')).astype(np.float64)
        diab = ((d1 == 'diabetes') | (d2 == 'diabetes') | (d3 == 'diabetes')).astype(np.float64)
        resp = ((d1 == 'respiratory') | (d2 == 'respiratory') | (d3 == 'respiratory')).astype(np.float64)
        genit = ((d1 == 'genitourinary') | (d2 == 'genitourinary') | (d3 == 'genitourinary')).astype(np.float64)
        digest = ((d1 == 'digestive') | (d2 == 'digestive') | (d3 == 'digestive')).astype(np.float64)
        primary_diab = (d1 == 'diabetes').astype(np.float64)
        primary_circ = (d1 == 'circulatory').astype(np.float64)
        comorb = np.array([len({g1, g2, g3} - {'missing'}) for g1, g2, g3 in zip(d1, d2, d3)], dtype=np.float64)
        circ_cnt = ((d1 == 'circulatory').astype(np.float64) + (d2 == 'circulatory').astype(np.float64) + (d3 == 'circulatory').astype(np.float64))
        diab_cnt = ((d1 == 'diabetes').astype(np.float64) + (d2 == 'diabetes').astype(np.float64) + (d3 == 'diabetes').astype(np.float64))
        resp_cnt = ((d1 == 'respiratory').astype(np.float64) + (d2 == 'respiratory').astype(np.float64) + (d3 == 'respiratory').astype(np.float64))
        genit_cnt = ((d1 == 'genitourinary').astype(np.float64) + (d2 == 'genitourinary').astype(np.float64) + (d3 == 'genitourinary').astype(np.float64))
        return circ, diab, resp, genit, digest, primary_diab, primary_circ, comorb, circ_cnt, diab_cnt, resp_cnt, genit_cnt

    c_tr, db_tr, r_tr, g_tr, dg_tr, pd_tr, pc_tr, cm_tr, cc_tr, dc_tr, rc_tr, gc_tr = extract_diag_flags(d1_tr, d2_tr, d3_tr)
    c_ev, db_ev, r_ev, g_ev, dg_ev, pd_ev, pc_ev, cm_ev, cc_ev, dc_ev, rc_ev, gc_ev = extract_diag_flags(d1_ev, d2_ev, d3_ev)

    register_num('has_diag_circulatory', c_tr, c_ev)
    register_num('has_diag_diabetes', db_tr, db_ev)
    register_num('has_diag_respiratory', r_tr, r_ev)
    register_num('has_diag_genitourinary', g_tr, g_ev)
    register_num('has_diag_digestive', dg_tr, dg_ev)
    register_num('primary_is_diabetes', pd_tr, pd_ev)
    register_num('primary_is_circulatory', pc_tr, pc_ev)
    register_num('diag_comorbidity_count', cm_tr, cm_ev)
    register_num('circulatory_count', cc_tr, cc_ev)
    register_num('diabetes_count', dc_tr, dc_ev)
    register_num('respiratory_count', rc_tr, rc_ev)
    register_num('genitourinary_count', gc_tr, gc_ev)

    # Specific High-Risk Clinical Comorbidities (Charlson/Elixhauser)
    def extract_clinical_conditions(raw1, raw2, raw3):
        n = len(raw1)
        has_chf = np.zeros(n, dtype=np.float64)
        has_renal = np.zeros(n, dtype=np.float64)
        has_cad = np.zeros(n, dtype=np.float64)
        has_copd = np.zeros(n, dtype=np.float64)
        has_fluid = np.zeros(n, dtype=np.float64)
        has_dm_comp = np.zeros(n, dtype=np.float64)
        has_htn = np.zeros(n, dtype=np.float64)
        has_cerebro = np.zeros(n, dtype=np.float64)
        has_pvd = np.zeros(n, dtype=np.float64)
        has_infection = np.zeros(n, dtype=np.float64)

        for i in range(n):
            for code in (raw1[i], raw2[i], raw3[i]):
                if code is None or (isinstance(code, float) and np.isnan(code)):
                    continue
                s = str(code).strip()
                if s.startswith('428'):
                    has_chf[i] = 1.0
                if s.startswith(('584', '585', '586', '582', '583', '588')):
                    has_renal[i] = 1.0
                if s.startswith(('410', '411', '412', '413', '414')):
                    has_cad[i] = 1.0
                if s.startswith(('490', '491', '492', '493', '494', '495', '496')):
                    has_copd[i] = 1.0
                if s.startswith('276'):
                    has_fluid[i] = 1.0
                if s.startswith(('250.4', '250.5', '250.6', '250.7', '250.8', '250.9')):
                    has_dm_comp[i] = 1.0
                if s.startswith(('401', '402', '403', '404', '405')):
                    has_htn[i] = 1.0
                if s.startswith(('430', '431', '432', '433', '434', '435', '436', '437', '438')):
                    has_cerebro[i] = 1.0
                if s.startswith(('440', '441', '442', '443', '444', '445')):
                    has_pvd[i] = 1.0
                if s.startswith(('480', '481', '482', '483', '484', '485', '486', '038')):
                    has_infection[i] = 1.0

        cardiorenal = ((has_chf == 1.0) & (has_renal == 1.0)).astype(np.float64)
        chf_copd = ((has_chf == 1.0) & (has_copd == 1.0)).astype(np.float64)
        charlson = (
            2.0 * has_chf + 2.0 * has_renal + 1.0 * has_cad + 1.0 * has_copd +
            1.0 * has_htn + 1.0 * has_cerebro + 1.0 * has_pvd + 1.0 * has_dm_comp
        )
        return (has_chf, has_renal, has_cad, has_copd, has_fluid, has_dm_comp,
                has_htn, has_cerebro, has_pvd, has_infection, cardiorenal, chf_copd, charlson)

        has_dm_acute = np.zeros(n, dtype=np.float64)
        has_dysrhythmia = np.zeros(n, dtype=np.float64)
        has_skin_ulcer = np.zeros(n, dtype=np.float64)
        has_anemia = np.zeros(n, dtype=np.float64)

        for i in range(n):
            for code in (raw1[i], raw2[i], raw3[i]):
                if code is None or (isinstance(code, float) and np.isnan(code)):
                    continue
                s = str(code).strip()
                if s.startswith(('250.1', '250.2', '250.3')):
                    has_dm_acute[i] = 1.0
                if s.startswith('427'):
                    has_dysrhythmia[i] = 1.0
                if s.startswith(('682', '707')):
                    has_skin_ulcer[i] = 1.0
                if s.startswith(('280', '281', '285')):
                    has_anemia[i] = 1.0

        cardiorenal_dm_triad = ((has_chf == 1.0) & (has_renal == 1.0) & (has_dm_comp == 1.0)).astype(np.float64)
        renal_fluid = ((has_renal == 1.0) & (has_fluid == 1.0)).astype(np.float64)

        return (has_chf, has_renal, has_cad, has_copd, has_fluid, has_dm_comp,
                has_htn, has_cerebro, has_pvd, has_infection, cardiorenal, chf_copd, charlson,
                has_dm_acute, has_dysrhythmia, has_skin_ulcer, has_anemia, cardiorenal_dm_triad, renal_fluid)

    (chf_tr, ren_tr, cad_tr, copd_tr, fld_tr, dmc_tr,
     htn_tr, cer_tr, pvd_tr, inf_tr, cr_tr, cc_copd_tr, chs_tr,
     dma_tr, dys_tr, sku_tr, ane_tr, cdm_tr, rf_tr) = extract_clinical_conditions(train['diag_1'], train['diag_2'], train['diag_3'])
    (chf_ev, ren_ev, cad_ev, copd_ev, fld_ev, dmc_ev,
     htn_ev, cer_ev, pvd_ev, inf_ev, cr_ev, cc_copd_ev, chs_ev,
     dma_ev, dys_ev, sku_ev, ane_ev, cdm_ev, rf_ev) = extract_clinical_conditions(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])

    register_num('has_chf', chf_tr, chf_ev)
    register_num('has_renal_disease', ren_tr, ren_ev)
    register_num('has_cad', cad_tr, cad_ev)
    register_num('has_copd', copd_tr, copd_ev)
    register_num('has_fluid_electrolyte', fld_tr, fld_ev)
    register_num('has_dm_complications', dmc_tr, dmc_ev)
    register_num('has_hypertension', htn_tr, htn_ev)
    register_num('has_cerebrovascular', cer_tr, cer_ev)
    register_num('has_pvd', pvd_tr, pvd_ev)
    register_num('has_pneumonia_sepsis', inf_tr, inf_ev)
    register_num('cardiorenal_syndrome', cr_tr, cr_ev)
    register_num('chf_and_copd', cc_copd_tr, cc_copd_ev)
    register_num('clinical_charlson_score', chs_tr, chs_ev)
    register_num('has_dm_acute', dma_tr, dma_ev)
    register_num('has_dysrhythmia', dys_tr, dys_ev)
    register_num('has_skin_ulcer', sku_tr, sku_ev)
    register_num('has_anemia', ane_tr, ane_ev)
    register_num('cardiorenal_dm_triad', cdm_tr, cdm_ev)
    register_num('renal_fluid_syndrome', rf_tr, rf_ev)

    # 4. Prior Utilization and Resource Intensity
    def extract_utilization(cols):
        inp = np.nan_to_num(cols['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(cols['number_emergency'], nan=0.0)
        out = np.nan_to_num(cols['number_outpatient'], nan=0.0)
        los = np.maximum(np.nan_to_num(cols['time_in_hospital'], nan=1.0), 1.0)
        meds = np.nan_to_num(cols['num_medications'], nan=0.0)
        labs = np.nan_to_num(cols['num_lab_procedures'], nan=0.0)
        procs = np.nan_to_num(cols['num_procedures'], nan=0.0)
        diags = np.maximum(np.nan_to_num(cols['number_diagnoses'], nan=1.0), 1.0)

        tot_prior = inp + emg + out
        has_inp = (inp > 0).astype(np.float64)
        has_emg = (emg > 0).astype(np.float64)
        has_out = (out > 0).astype(np.float64)
        high_inp = (inp >= 2.0).astype(np.float64)
        inp_ratio = inp / (tot_prior + 1.0)
        emg_ratio = emg / (tot_prior + 1.0)
        acute_prior = inp + emg

        meds_per_day = meds / los
        labs_per_day = labs / los
        procs_per_day = procs / los
        meds_per_diag = meds / diags
        labs_per_diag = labs / diags
        procs_per_diag = procs / diags
        service_density = (meds + labs + procs) / los

        inp_x_los = inp * los
        inp_x_diag = inp * diags
        inp_x_meds = inp * meds
        emg_x_inp = emg * inp
        labs_per_med = labs / (meds + 1.0)
        procs_per_med = procs / (meds + 1.0)
        util_index = 3.0 * inp + 1.5 * emg + 0.5 * out
        many_diagnoses = (diags >= 9.0).astype(np.float64)
        polypharmacy = (meds >= 16.0).astype(np.float64)
        purely_medical = (procs == 0.0).astype(np.float64)
        extreme_polypharmacy = (meds >= 20.0).astype(np.float64)

        log_inp = np.log1p(inp)
        log_emg = np.log1p(emg)
        log_out = np.log1p(out)
        log_los = np.log1p(los)
        log_meds = np.log1p(meds)
        log_labs = np.log1p(labs)
        log_tot = np.log1p(tot_prior)

        log_inp_x_los = log_inp * log_los
        log_inp_x_meds = log_inp * log_meds

        return {
            'total_prior_visits': tot_prior,
            'has_prior_inpatient': has_inp,
            'has_prior_emergency': has_emg,
            'has_prior_outpatient': has_out,
            'high_inpatient': high_inp,
            'frequent_inpatient': (inp >= 3.0).astype(np.float64),
            'inpatient_ratio': inp_ratio,
            'emergency_ratio': emg_ratio,
            'acute_prior': acute_prior,
            'meds_per_day': meds_per_day,
            'labs_per_day': labs_per_day,
            'procs_per_day': procs_per_day,
            'meds_per_diag': meds_per_diag,
            'labs_per_diag': labs_per_diag,
            'procs_per_diag': procs_per_diag,
            'service_density': service_density,
            'inpatient_x_los': inp_x_los,
            'inpatient_x_diag': inp_x_diag,
            'inpatient_x_meds': inp_x_meds,
            'emergency_x_inpatient': emg_x_inp,
            'labs_per_med': labs_per_med,
            'procs_per_med': procs_per_med,
            'utilization_index': util_index,
            'many_diagnoses': many_diagnoses,
            'polypharmacy': polypharmacy,
            'purely_medical_stay': purely_medical,
            'extreme_polypharmacy': extreme_polypharmacy,
            'log_inpatient': log_inp,
            'log_emergency': log_emg,
            'log_outpatient': log_out,
            'log_los': log_los,
            'log_medications': log_meds,
            'log_labs': log_labs,
            'log_prior_visits': log_tot,
            'log_inp_x_los': log_inp_x_los,
            'log_inp_x_meds': log_inp_x_meds,
        }

    util_tr = extract_utilization(train)
    util_ev = extract_utilization(eval_cols)
    for k in util_tr:
        register_num(k, util_tr[k], util_ev[k])

    register_num('inpatient_x_age', util_tr['has_prior_inpatient'] * age_tr, util_ev['has_prior_inpatient'] * age_ev)
    register_num('elderly_prior_inpatient', ((age_tr >= 80.0) & (util_tr['has_prior_inpatient'] == 1.0)).astype(np.float64), ((age_ev >= 80.0) & (util_ev['has_prior_inpatient'] == 1.0)).astype(np.float64))

    # LACE Readmission Risk Index (L: Length of Stay, A: Acuity, C: Charlson, E: Emergency)
    def compute_lace(los_arr, adm_type_arr, charlson_arr, emg_arr):
        L = np.zeros_like(los_arr)
        L[los_arr == 1.0] = 1.0
        L[los_arr == 2.0] = 2.0
        L[los_arr == 3.0] = 3.0
        L[(los_arr >= 4.0) & (los_arr <= 6.0)] = 4.0
        L[(los_arr >= 7.0) & (los_arr <= 13.0)] = 5.0
        L[los_arr >= 14.0] = 7.0

        A = np.isin(adm_type_arr, ['1', '2']).astype(np.float64) * 3.0

        C = np.minimum(charlson_arr, 4.0)
        C[charlson_arr >= 4.0] = 5.0

        E = np.minimum(emg_arr, 4.0)
        lace = L + A + C + E
        high_lace = (lace >= 10.0).astype(np.float64)
        return lace, high_lace

    lace_tr, hlace_tr = compute_lace(train['time_in_hospital'], train['admission_type_id'], chs_tr, train['number_emergency'])
    lace_ev, hlace_ev = compute_lace(eval_cols['time_in_hospital'], eval_cols['admission_type_id'], chs_ev, eval_cols['number_emergency'])
    register_num('lace_score', lace_tr, lace_ev)
    register_num('high_lace', hlace_tr, hlace_ev)

    # 5. Medication and Lab Dynamics
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]

    def extract_meds_and_labs(cols, n_rows):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        change_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in cols:
                arr = cols[m]
                is_up = (arr == 'Up')
                is_down = (arr == 'Down')
                is_steady = (arr == 'Steady')
                active_cnt += (is_up | is_down | is_steady).astype(np.float64)
                change_cnt += (is_up | is_down).astype(np.float64)
                up_cnt += is_up.astype(np.float64)
                down_cnt += is_down.astype(np.float64)

        ins = cols.get('insulin')
        ins_status = np.zeros(n_rows, dtype=np.float64)
        ins_up_flag = np.zeros(n_rows, dtype=np.float64)
        ins_down_flag = np.zeros(n_rows, dtype=np.float64)
        ins_steady_flag = np.zeros(n_rows, dtype=np.float64)
        if ins is not None:
            ins_status[ins == 'Steady'] = 1.0
            ins_status[ins == 'Down'] = 2.0
            ins_status[ins == 'Up'] = 3.0
            ins_active = (ins_status > 0).astype(np.float64)
            ins_changed = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
            ins_up_flag = (ins == 'Up').astype(np.float64)
            ins_down_flag = (ins == 'Down').astype(np.float64)
            ins_steady_flag = (ins == 'Steady').astype(np.float64)
        else:
            ins_active = np.zeros(n_rows, dtype=np.float64)
            ins_changed = np.zeros(n_rows, dtype=np.float64)

        ch = cols.get('change')
        ch_flag = (ch == 'Ch').astype(np.float64) if ch is not None else np.zeros(n_rows, dtype=np.float64)

        dm = cols.get('diabetesMed')
        dm_flag = (dm == 'Yes').astype(np.float64) if dm is not None else np.zeros(n_rows, dtype=np.float64)

        stable_regimen = ((dm_flag == 1.0) & (ch_flag == 0.0)).astype(np.float64)
        modified_regimen = ((dm_flag == 1.0) & (ch_flag == 1.0)).astype(np.float64)
        no_diabetes_med = (dm_flag == 0.0).astype(np.float64)
        insulin_monotherapy = ((ins_active == 1.0) & (active_cnt == 1.0)).astype(np.float64)
        insulin_plus_oral = ((ins_active == 1.0) & (active_cnt > 1.0)).astype(np.float64)
        oral_only = ((ins_active == 0.0) & (active_cnt > 0.0)).astype(np.float64)

        comb_pill_cnt = np.zeros(n_rows, dtype=np.float64)
        comb_cols = ['glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        for m in comb_cols:
            if m in cols:
                arr = cols[m]
                comb_pill_cnt += (np.isin(arr, ['Steady', 'Up', 'Down'])).astype(np.float64)

        a1c = cols.get('A1Cresult')
        if a1c is not None:
            a1c_tested = np.array([1.0 if x not in ('None', None) and not (isinstance(x, float) and np.isnan(x)) else 0.0 for x in a1c], dtype=np.float64)
            a1c_high = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        else:
            a1c_tested = np.zeros(n_rows, dtype=np.float64)
            a1c_high = np.zeros(n_rows, dtype=np.float64)

        glu = cols.get('max_glu_serum')
        if glu is not None:
            glu_tested = np.array([1.0 if x not in ('None', None) and not (isinstance(x, float) and np.isnan(x)) else 0.0 for x in glu], dtype=np.float64)
            glu_high = np.isin(glu, ['>200', '>300']).astype(np.float64)
        else:
            glu_tested = np.zeros(n_rows, dtype=np.float64)
            glu_high = np.zeros(n_rows, dtype=np.float64)

        clinical_inertia = ((a1c_high == 1.0) & (ch_flag == 0.0)).astype(np.float64)
        glycemic_uncontrolled = ((a1c_high == 1.0) | (glu_high == 1.0)).astype(np.float64)

        return {
            'num_active_meds': active_cnt,
            'num_med_changes': change_cnt,
            'num_med_up': up_cnt,
            'num_med_down': down_cnt,
            'insulin_status': ins_status,
            'insulin_active': ins_active,
            'insulin_changed': ins_changed,
            'insulin_up_flag': ins_up_flag,
            'insulin_down_flag': ins_down_flag,
            'insulin_steady_flag': ins_steady_flag,
            'change_flag': ch_flag,
            'diabetes_med_flag': dm_flag,
            'stable_regimen': stable_regimen,
            'modified_regimen': modified_regimen,
            'no_diabetes_med': no_diabetes_med,
            'insulin_monotherapy': insulin_monotherapy,
            'insulin_plus_oral': insulin_plus_oral,
            'oral_only': oral_only,
            'on_comb_pill': comb_pill_cnt,
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'glu_tested': glu_tested,
            'glu_high': glu_high,
            'clinical_inertia': clinical_inertia,
            'glycemic_uncontrolled': glycemic_uncontrolled,
        }

    med_tr = extract_meds_and_labs(train, n_train)
    med_ev = extract_meds_and_labs(eval_cols, n_eval)
    for k in med_tr:
        register_num(k, med_tr[k], med_ev[k])

    # 6. Admission, Discharge, and Missingness Indicators
    def extract_context(cols, n_rows):
        disch = cols.get('discharge_disposition_id')
        if disch is not None:
            disch_home = (disch == '1').astype(np.float64)
            disch_snf = (disch == '3').astype(np.float64)
            disch_hh = (disch == '6').astype(np.float64)
            disch_ama = (disch == '7').astype(np.float64)
            disch_rehab = (disch == '22').astype(np.float64)
            disch_fac = np.isin(disch, ['2', '4', '5', '22', '28']).astype(np.float64)
            disch_needs_support = np.isin(disch, ['3', '6']).astype(np.float64)
        else:
            disch_home = np.zeros(n_rows, dtype=np.float64)
            disch_snf = np.zeros(n_rows, dtype=np.float64)
            disch_hh = np.zeros(n_rows, dtype=np.float64)
            disch_ama = np.zeros(n_rows, dtype=np.float64)
            disch_rehab = np.zeros(n_rows, dtype=np.float64)
            disch_fac = np.zeros(n_rows, dtype=np.float64)
            disch_needs_support = np.zeros(n_rows, dtype=np.float64)

        src = cols.get('admission_source_id')
        adm_er = (src == '7').astype(np.float64) if src is not None else np.zeros(n_rows, dtype=np.float64)
        if src is not None:
            adm_referral = np.isin(src, ['1', '2', '3']).astype(np.float64)
            adm_transfer = np.isin(src, ['4', '5', '6', '10', '18', '22', '25']).astype(np.float64)
        else:
            adm_referral = np.zeros(n_rows, dtype=np.float64)
            adm_transfer = np.zeros(n_rows, dtype=np.float64)

        adm_type = cols.get('admission_type_id')
        adm_emerg = (adm_type == '1').astype(np.float64) if adm_type is not None else np.zeros(n_rows, dtype=np.float64)
        adm_urgent = (adm_type == '2').astype(np.float64) if adm_type is not None else np.zeros(n_rows, dtype=np.float64)
        adm_elective = (adm_type == '3').astype(np.float64) if adm_type is not None else np.zeros(n_rows, dtype=np.float64)
        adm_acute = np.isin(adm_type, ['1', '2']).astype(np.float64) if adm_type is not None else np.zeros(n_rows, dtype=np.float64)
        disch_high_risk = np.isin(disch, ['3', '6', '7', '22', '28']).astype(np.float64) if disch is not None else np.zeros(n_rows, dtype=np.float64)

        spec = cols.get('medical_specialty')
        spec_arr = spec if spec is not None else [None]*n_rows
        missing_spec = np.array([1.0 if x is None or str(x).strip() in ('?', '', 'None') else 0.0 for x in spec_arr], dtype=np.float64)
        spec_cardio = np.array([1.0 if x is not None and 'Cardiology' in str(x) else 0.0 for x in spec_arr], dtype=np.float64)
        spec_nephro = np.array([1.0 if x is not None and 'Nephrology' in str(x) else 0.0 for x in spec_arr], dtype=np.float64)
        spec_emerg = np.array([1.0 if x is not None and 'Emergency' in str(x) else 0.0 for x in spec_arr], dtype=np.float64)

        payer = cols.get('payer_code')
        missing_payer = np.array([1.0 if x is None or str(x).strip() in ('?', '', 'None') else 0.0 for x in (payer if payer is not None else [None]*n_rows)], dtype=np.float64)

        return {
            'disch_home': disch_home,
            'disch_snf': disch_snf,
            'disch_home_health': disch_hh,
            'disch_ama': disch_ama,
            'disch_rehab': disch_rehab,
            'disch_other_facility': disch_fac,
            'disch_needs_support': disch_needs_support,
            'adm_er': adm_er,
            'adm_referral_type': adm_referral,
            'adm_transfer_type': adm_transfer,
            'adm_emergency_type': adm_emerg,
            'adm_urgent_type': adm_urgent,
            'adm_elective_type': adm_elective,
            'adm_acute_type': adm_acute,
            'spec_cardiology': spec_cardio,
            'spec_nephrology': spec_nephro,
            'spec_emergency': spec_emerg,
            'missing_specialty': missing_spec,
            'missing_payer': missing_payer,
            'disch_high_risk': disch_high_risk,
        }

    ctx_tr = extract_context(train, n_train)
    ctx_ev = extract_context(eval_cols, n_eval)
    for k in ctx_tr:
        register_num(k, ctx_tr[k], ctx_ev[k])

    # 7. Out-of-fold Target Encoding with Bayesian Smoothing
    def target_encode(train_arr, y_arr, eval_arr, n_splits=5, smoothing=20.0, random_state=42):
        global_mean = float(np.mean(y_arr))
        def clean_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "__MISSING__"
            s = str(v).strip()
            return s if s and s != "?" else "__MISSING__"

        tr_clean = [clean_val(x) for x in train_arr]
        ev_clean = [clean_val(x) for x in eval_arr]

        train_encoded = np.full(len(train_arr), global_mean, dtype=np.float64)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for tr_idx, val_idx in skf.split(np.zeros(len(y_arr)), y_arr):
            counts = {}
            sums = {}
            for idx in tr_idx:
                v = tr_clean[idx]
                counts[v] = counts.get(v, 0) + 1
                sums[v] = sums.get(v, 0.0) + y_arr[idx]

            for idx in val_idx:
                v = tr_clean[idx]
                cnt = counts.get(v, 0)
                s = sums.get(v, 0.0)
                train_encoded[idx] = (s + smoothing * global_mean) / (cnt + smoothing)

        counts = {}
        sums = {}
        for v, y in zip(tr_clean, y_arr):
            counts[v] = counts.get(v, 0) + 1
            sums[v] = sums.get(v, 0.0) + y

        eval_encoded = np.array([
            (sums.get(v, 0.0) + smoothing * global_mean) / (counts.get(v, 0) + smoothing)
            for v in ev_clean
        ], dtype=np.float64)

        return train_encoded, eval_encoded

    te_cols = [
        ('diag_1', train['diag_1'], eval_cols['diag_1'], 25.0),
        ('diag_2', train['diag_2'], eval_cols['diag_2'], 25.0),
        ('diag_3', train['diag_3'], eval_cols['diag_3'], 25.0),
        ('discharge_disposition_id', train['discharge_disposition_id'], eval_cols['discharge_disposition_id'], 15.0),
        ('admission_source_id', train['admission_source_id'], eval_cols['admission_source_id'], 15.0),
        ('medical_specialty', train['medical_specialty'], eval_cols['medical_specialty'], 20.0),
        ('payer_code', train['payer_code'], eval_cols['payer_code'], 15.0),
        ('diag_1_group', d1_tr, d1_ev, 20.0),
        ('diag_1_3digit', d1_pfx_tr, d1_pfx_ev, 20.0),
    ]

    disch_adm_tr = np.array([f"{d}_{s}" for d, s in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    disch_adm_ev = np.array([f"{d}_{s}" for d, s in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)

    disch_diag_tr = np.array([f"{d}_{g}" for d, g in zip(train['discharge_disposition_id'], d1_tr)], dtype=object)
    disch_diag_ev = np.array([f"{d}_{g}" for d, g in zip(eval_cols['discharge_disposition_id'], d1_ev)], dtype=object)

    diag_123_tr = np.array([f"{g1}_{g2}_{g3}" for g1, g2, g3 in zip(d1_tr, d2_tr, d3_tr)], dtype=object)
    diag_123_ev = np.array([f"{g1}_{g2}_{g3}" for g1, g2, g3 in zip(d1_ev, d2_ev, d3_ev)], dtype=object)

    disch_inp_tr = np.array([f"{d}_{util_tr['has_prior_inpatient'][i] > 0}" for i, d in enumerate(train['discharge_disposition_id'])], dtype=object)
    disch_inp_ev = np.array([f"{d}_{util_ev['has_prior_inpatient'][i] > 0}" for i, d in enumerate(eval_cols['discharge_disposition_id'])], dtype=object)

    diag1_inp_tr = np.array([f"{g}_{util_tr['has_prior_inpatient'][i] > 0}" for i, g in enumerate(d1_tr)], dtype=object)
    diag1_inp_ev = np.array([f"{g}_{util_ev['has_prior_inpatient'][i] > 0}" for i, g in enumerate(d1_ev)], dtype=object)

    te_cols.extend([
        ('disch_adm', disch_adm_tr, disch_adm_ev, 25.0),
        ('disch_diag', disch_diag_tr, disch_diag_ev, 25.0),
        ('disch_inp', disch_inp_tr, disch_inp_ev, 20.0),
        ('diag1_inp', diag1_inp_tr, diag1_inp_ev, 20.0),
        ('diag_123', diag_123_tr, diag_123_ev, 30.0),
        ('admission_type_id', train['admission_type_id'], eval_cols['admission_type_id'], 15.0),
    ])

    for col_name, tr_arr, ev_arr, sm in te_cols:
        te_tr, te_ev = target_encode(tr_arr, y_train, ev_arr, n_splits=5, smoothing=sm)
        register_num(f'te_{col_name}', te_tr, te_ev)

    register_num('high_risk_disch_prior_inp', ctx_tr['disch_high_risk'] * util_tr['has_prior_inpatient'], ctx_ev['disch_high_risk'] * util_ev['has_prior_inpatient'])
    register_num('er_adm_prior_inp', ctx_tr['adm_er'] * util_tr['has_prior_inpatient'], ctx_ev['adm_er'] * util_ev['has_prior_inpatient'])

    # 7b. Twin-Patient Latent Manifold Risk & Phenotype Resonance
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors

    twin_features = [
        'age_num', 'log_inpatient', 'log_emergency', 'log_outpatient',
        'log_los', 'log_medications', 'log_labs', 'acute_prior', 'utilization_index',
        'clinical_charlson_score', 'lace_score', 'has_chf', 'has_renal_disease',
        'has_cad', 'has_copd', 'has_dm_complications', 'has_hypertension',
        'num_active_meds', 'num_med_changes', 'insulin_active',
        'disch_home', 'disch_snf', 'disch_home_health', 'adm_er',
        'te_discharge_disposition_id', 'te_diag_1', 'te_disch_adm'
    ]

    m_tr_matrix = np.column_stack([tr[f] for f in twin_features])
    m_ev_matrix = np.column_stack([ev[f] for f in twin_features])

    med_vals = np.nanmedian(m_tr_matrix, axis=0)
    nan_tr_idx = np.where(np.isnan(m_tr_matrix))
    m_tr_matrix[nan_tr_idx] = np.take(med_vals, nan_tr_idx[1])
    nan_ev_idx = np.where(np.isnan(m_ev_matrix))
    m_ev_matrix[nan_ev_idx] = np.take(med_vals, nan_ev_idx[1])

    mean_vals = np.mean(m_tr_matrix, axis=0)
    std_vals = np.std(m_tr_matrix, axis=0)
    std_vals[std_vals == 0.0] = 1.0

    m_tr_norm = (m_tr_matrix - mean_vals) / std_vals
    m_ev_norm = (m_ev_matrix - mean_vals) / std_vals

    svd = TruncatedSVD(n_components=8, random_state=42)
    z_tr = svd.fit_transform(m_tr_norm)
    z_ev = svd.transform(m_ev_norm)

    for comp_idx in range(8):
        register_num(f'svd_archetype_{comp_idx}', z_tr[:, comp_idx], z_ev[:, comp_idx])

    # Out-of-fold Twin-Patient Readmission Density Estimation
    skf_twin = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    twin_risk_k15_tr = np.zeros(n_train, dtype=np.float64)
    twin_risk_k45_tr = np.zeros(n_train, dtype=np.float64)
    twin_dist_tr = np.zeros(n_train, dtype=np.float64)

    for f_tr_idx, f_val_idx in skf_twin.split(z_tr, y_train):
        nn_fold = NearestNeighbors(n_neighbors=46, metric='euclidean', algorithm='kd_tree')
        nn_fold.fit(z_tr[f_tr_idx])
        dists_fold, inds_fold = nn_fold.kneighbors(z_tr[f_val_idx])
        y_sub = y_train[f_tr_idx]
        neigh_labels = y_sub[inds_fold]
        twin_risk_k15_tr[f_val_idx] = np.mean(neigh_labels[:, :15], axis=1)
        twin_risk_k45_tr[f_val_idx] = np.mean(neigh_labels[:, :45], axis=1)
        twin_dist_tr[f_val_idx] = np.mean(dists_fold[:, :15], axis=1)

    nn_full = NearestNeighbors(n_neighbors=45, metric='euclidean', algorithm='kd_tree')
    nn_full.fit(z_tr)
    dists_ev, inds_ev = nn_full.kneighbors(z_ev)
    neigh_labels_ev = y_train[inds_ev]
    twin_risk_k15_ev = np.mean(neigh_labels_ev[:, :15], axis=1)
    twin_risk_k45_ev = np.mean(neigh_labels_ev[:, :45], axis=1)
    twin_dist_ev = np.mean(dists_ev[:, :15], axis=1)

    register_num('twin_readmission_risk_k15', twin_risk_k15_tr, twin_risk_k15_ev)
    register_num('twin_readmission_risk_k45', twin_risk_k45_tr, twin_risk_k45_ev)
    register_num('twin_readmission_density_ratio',
                 twin_risk_k15_tr / (twin_risk_k45_tr + 0.02),
                 twin_risk_k15_ev / (twin_risk_k45_ev + 0.02))
    register_num('twin_archetype_distance', twin_dist_tr, twin_dist_ev)

    # 8. Encode full feature matrices using TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        tr,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_indices = encoder.categorical_indices

    # 9. Diverse Multi-Depth LightGBM and XGBoost Ensemble with ExtraTrees and Calibrated Linear Model
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import rankdata

    m1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=24,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    m2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.035,
        max_depth=6,
        num_leaves=30,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )

    m3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.045,
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.05,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    m4_xt = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=28,
        extra_trees=True,
        min_child_samples=45,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=3.0,
        reg_alpha=0.2,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )

    m_xgb1 = xgb.XGBClassifier(
        n_estimators=260,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=4.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
        tree_method='hist',
    )

    m_xgb2 = xgb.XGBClassifier(
        n_estimators=280,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_lambda=6.0,
        reg_alpha=0.5,
        random_state=101,
        n_jobs=1,
        eval_metric='logloss',
        tree_method='hist',
    )

    m_lr = LogisticRegression(C=0.08, max_iter=300, random_state=42)
    m_lr.fit(m_tr_norm, y_train)

    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    m4_xt.fit(x_train, y_train, categorical_feature=cat_indices)
    m_xgb1.fit(x_train, y_train)
    m_xgb2.fit(x_train, y_train)

    p1 = m1.predict_proba(x_eval)[:, 1]
    p2 = m2.predict_proba(x_eval)[:, 1]
    p3 = m3.predict_proba(x_eval)[:, 1]
    p4 = m4_xt.predict_proba(x_eval)[:, 1]
    p_xgb1 = m_xgb1.predict_proba(x_eval)[:, 1]
    p_xgb2 = m_xgb2.predict_proba(x_eval)[:, 1]
    p_lr = m_lr.predict_proba(m_ev_norm)[:, 1]

    # Rank Normalization for AUC maximization across diverse model families
    def to_rank(probs: np.ndarray) -> np.ndarray:
        return rankdata(probs) / len(probs)

    r1 = to_rank(p1)
    r2 = to_rank(p2)
    r3 = to_rank(p3)
    r4 = to_rank(p4)
    r_xgb1 = to_rank(p_xgb1)
    r_xgb2 = to_rank(p_xgb2)
    r_lr = to_rank(p_lr)

    preds = (
        0.19 * r1 +
        0.19 * r2 +
        0.14 * r3 +
        0.14 * r4 +
        0.14 * r_xgb1 +
        0.14 * r_xgb2 +
        0.06 * r_lr
    )

    preds = np.nan_to_num(preds, nan=0.11)
    preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
