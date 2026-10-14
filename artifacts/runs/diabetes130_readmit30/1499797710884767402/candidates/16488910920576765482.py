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
    try:
        from clinical_toolkit import icd9_group, TabularEncoder
    except ImportError:
        pass
    from sklearn.model_selection import StratifiedKFold

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Clean continuous age
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for col_dict in [train_data, eval_data]:
        raw_age = col_dict.get('age', np.array([]))
        col_dict['num_age'] = np.array([age_map.get(a, np.nan) for a in raw_age], dtype=np.float64)
    extra_numeric.append('num_age')

    # 2. Non-redundant Utilisation & Hospital Stay intensity + Clinical domain features
    for col_dict in [train_data, eval_data]:
        inp = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        outp = np.nan_to_num(col_dict.get('number_outpatient', 0.0), nan=0.0)
        tot_visits = inp + emg + outp

        col_dict['util_total_visits'] = tot_visits
        col_dict['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        col_dict['has_prior_emergency'] = (emg > 0.0).astype(np.float64)
        col_dict['frequent_inpatient'] = (inp >= 2.0).astype(np.float64)
        col_dict['log_number_inpatient'] = np.log1p(inp)
        col_dict['log_total_visits'] = np.log1p(tot_visits)
        col_dict['util_inpatient_ratio'] = inp / (tot_visits + 1.0)
        col_dict['util_emergency_ratio'] = emg / (tot_visits + 1.0)
        col_dict['util_visit_severity'] = inp * 3.0 + emg * 1.5 + outp * 0.5
        col_dict['util_inp_to_outp'] = inp / (outp + 1.0)
        col_dict['util_emg_to_outp'] = emg / (outp + 1.0)

        time_safe = np.maximum(np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0), 1.0)
        num_labs = np.nan_to_num(col_dict.get('num_lab_procedures', 0.0), nan=0.0)
        num_proc = np.nan_to_num(col_dict.get('num_procedures', 0.0), nan=0.0)
        num_meds = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)
        num_diag = np.nan_to_num(col_dict.get('number_diagnoses', 1.0), nan=1.0)

        col_dict['hosp_labs_per_day'] = num_labs / time_safe
        col_dict['hosp_meds_per_day'] = num_meds / time_safe
        col_dict['hosp_service_intensity'] = (num_labs + num_proc + num_meds) / time_safe
        col_dict['hosp_proc_to_labs'] = num_proc / (num_labs + 1.0)
        col_dict['hosp_meds_to_labs'] = num_meds / (num_labs + 1.0)
        col_dict['hosp_interaction_log'] = np.log1p(num_labs * num_proc * num_meds)
        col_dict['hosp_diag_to_days'] = num_diag / time_safe
        col_dict['hosp_meds_to_diag'] = num_meds / (num_diag + 1.0)
        col_dict['inter_inp_time'] = inp * time_safe

        # Discharge and Admission grouping
        disp = col_dict.get('discharge_disposition_id', np.array([]))
        adm_s = col_dict.get('admission_source_id', np.array([]))
        adm_t = col_dict.get('admission_type_id', np.array([]))

        col_dict['feat_disp_home'] = np.isin(disp, ['1', '8']).astype(np.float64)
        col_dict['feat_disp_home_health'] = (disp == '6').astype(np.float64)
        col_dict['feat_disp_ama'] = (disp == '7').astype(np.float64)
        col_dict['feat_disp_facility'] = np.isin(disp, ['2', '3', '4', '5', '22', '23', '24', '25']).astype(np.float64)
        col_dict['feat_adm_emergency'] = (np.isin(adm_s, ['7']) | np.isin(adm_t, ['1'])).astype(np.float64)
        col_dict['feat_adm_elective'] = (adm_t == '3').astype(np.float64)
        col_dict['feat_disp_adm_comb'] = np.array([f"{d}_{s}" for d, s in zip(disp, adm_s)], dtype=object)

        # Lab results
        a1c = col_dict.get('A1Cresult', np.array([]))
        glu = col_dict.get('max_glu_serum', np.array([]))
        col_dict['feat_high_a1c'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        col_dict['feat_high_glucose'] = np.isin(glu, ['>200', '>300']).astype(np.float64)
        col_dict['feat_performed_a1c'] = (a1c != 'None').astype(np.float64)
        col_dict['feat_performed_glucose'] = (glu != 'None').astype(np.float64)

        # Missingness features
        spec = col_dict.get('medical_specialty', np.array([]))
        payer = col_dict.get('payer_code', np.array([]))
        weight = col_dict.get('weight', np.array([]))
        col_dict['feat_spec_missing'] = np.isin(spec, [None, 'None', '?', 'missing']).astype(np.float64)
        col_dict['feat_payer_missing'] = np.isin(payer, [None, 'None', '?', 'missing']).astype(np.float64)
        col_dict['feat_weight_missing'] = np.isin(weight, [None, 'None', '?', 'missing']).astype(np.float64)

    extra_categorical.append('feat_disp_adm_comb')
    extra_numeric.extend([
        'util_total_visits', 'has_prior_inpatient', 'has_prior_emergency', 'frequent_inpatient',
        'log_number_inpatient', 'log_total_visits', 'util_inpatient_ratio', 'util_emergency_ratio',
        'util_visit_severity', 'util_inp_to_outp', 'util_emg_to_outp',
        'hosp_labs_per_day', 'hosp_meds_per_day', 'hosp_service_intensity',
        'hosp_proc_to_labs', 'hosp_meds_to_labs', 'hosp_interaction_log',
        'hosp_diag_to_days', 'hosp_meds_to_diag', 'inter_inp_time',
        'feat_disp_home', 'feat_disp_home_health', 'feat_disp_ama', 'feat_disp_facility',
        'feat_adm_emergency', 'feat_adm_elective',
        'feat_high_a1c', 'feat_high_glucose', 'feat_performed_a1c', 'feat_performed_glucose',
        'feat_spec_missing', 'feat_payer_missing', 'feat_weight_missing'
    ])

    # 3. Medication summary features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col_dict in [train_data, eval_data]:
        n_rows = len(col_dict['time_in_hospital'])
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in col_dict:
                arr = col_dict[m]
                active += np.isin(arr, ['Steady', 'Up', 'Down']).astype(np.float64)
                up += (arr == 'Up').astype(np.float64)
                down += (arr == 'Down').astype(np.float64)

        insulin_arr = col_dict.get('insulin', np.full(n_rows, 'No', dtype=object))
        ins_active = np.isin(insulin_arr, ['Steady', 'Up', 'Down']).astype(np.float64)
        ins_up = (insulin_arr == 'Up').astype(np.float64)
        ins_down = (insulin_arr == 'Down').astype(np.float64)
        oral_meds = np.maximum(0.0, active - ins_active)

        col_dict['med_num_active'] = active
        col_dict['med_num_changed'] = up + down
        col_dict['med_insulin_active'] = ins_active
        col_dict['med_insulin_up'] = ins_up
        col_dict['med_insulin_down'] = ins_down
        col_dict['med_insulin_plus_oral'] = ((ins_active > 0.0) & (oral_meds > 0.0)).astype(np.float64)
        col_dict['med_only_insulin'] = ((ins_active > 0.0) & (oral_meds == 0.0)).astype(np.float64)
        col_dict['med_only_oral'] = ((ins_active == 0.0) & (oral_meds > 0.0)).astype(np.float64)
        col_dict['med_multiple_oral'] = (oral_meds > 1.0).astype(np.float64)
        col_dict['med_no_active'] = (active == 0.0).astype(np.float64)

        met_arr = col_dict.get('metformin', np.full(n_rows, 'No', dtype=object))
        col_dict['med_metformin_active'] = np.isin(met_arr, ['Steady', 'Up', 'Down']).astype(np.float64)
        col_dict['med_insulin_changed'] = ((ins_up + ins_down) > 0.0).astype(np.float64)

        glip = col_dict.get('glipizide', np.full(n_rows, 'No', dtype=object))
        glyb = col_dict.get('glyburide', np.full(n_rows, 'No', dtype=object))
        glim = col_dict.get('glimepiride', np.full(n_rows, 'No', dtype=object))
        has_su = (np.isin(glip, ['Steady', 'Up', 'Down']) |
                  np.isin(glyb, ['Steady', 'Up', 'Down']) |
                  np.isin(glim, ['Steady', 'Up', 'Down'])).astype(np.float64)
        col_dict['med_su_active'] = has_su
        col_dict['med_insulin_and_su'] = ((ins_active > 0.0) & (has_su > 0.0)).astype(np.float64)

        dm = col_dict.get('diabetesMed', np.array([]))
        ch = col_dict.get('change', np.array([]))
        col_dict['feat_diabetesMed_num'] = (dm == 'Yes').astype(np.float64)
        col_dict['feat_change_num'] = (ch == 'Ch').astype(np.float64)

    extra_numeric.extend([
        'med_num_active', 'med_num_changed', 'med_insulin_active',
        'med_insulin_up', 'med_insulin_down', 'med_insulin_plus_oral',
        'med_only_insulin', 'med_only_oral', 'med_multiple_oral', 'med_no_active',
        'med_metformin_active', 'med_insulin_changed', 'med_su_active', 'med_insulin_and_su',
        'feat_diabetesMed_num', 'feat_change_num'
    ])

    # 4. Vectorized diagnosis grouping & comorbidity patterns
    def extract_prefix(code):
        if code is None or code == '?' or code == 'None':
            return 'missing'
        s = str(code).strip()
        return s.split('.')[0] if s else 'missing'

    for col_dict in [train_data, eval_data]:
        d1 = col_dict.get('diag_1', np.array([]))
        d2 = col_dict.get('diag_2', np.array([]))
        d3 = col_dict.get('diag_3', np.array([]))

        g1 = np.array([icd9_group(c) for c in d1], dtype=object)
        g2 = np.array([icd9_group(c) for c in d2], dtype=object)
        g3 = np.array([icd9_group(c) for c in d3], dtype=object)
        p1 = np.array([extract_prefix(c) for c in d1], dtype=object)
        p2 = np.array([extract_prefix(c) for c in d2], dtype=object)
        p3 = np.array([extract_prefix(c) for c in d3], dtype=object)

        col_dict['diag_1_group'] = g1
        col_dict['diag_2_group'] = g2
        col_dict['diag_3_group'] = g3
        col_dict['diag_1_prefix'] = p1

        circ = (g1 == 'circulatory') | (g2 == 'circulatory') | (g3 == 'circulatory')
        diab = (g1 == 'diabetes') | (g2 == 'diabetes') | (g3 == 'diabetes')
        resp = (g1 == 'respiratory') | (g2 == 'respiratory') | (g3 == 'respiratory')
        digest = (g1 == 'digestive') | (g2 == 'digestive') | (g3 == 'digestive')
        genit = (g1 == 'genitourinary') | (g2 == 'genitourinary') | (g3 == 'genitourinary')

        chf = (p1 == '428') | (p2 == '428') | (p3 == '428')
        ckd = (p1 == '585') | (p2 == '585') | (p3 == '585') | (p1 == '586') | (p2 == '586') | (p3 == '586')
        
        ht = np.isin(p1, ['401', '402', '403', '404', '405']) | \
             np.isin(p2, ['401', '402', '403', '404', '405']) | \
             np.isin(p3, ['401', '402', '403', '404', '405'])
             
        copd = np.isin(p1, ['490', '491', '492', '493', '494', '495', '496']) | \
               np.isin(p2, ['490', '491', '492', '493', '494', '495', '496']) | \
               np.isin(p3, ['490', '491', '492', '493', '494', '495', '496'])

        aki = (p1 == '584') | (p2 == '584') | (p3 == '584')
        stroke = np.isin(p1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | \
                 np.isin(p2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | \
                 np.isin(p3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])

        cad = np.isin(p1, ['410', '411', '412', '413', '414']) | \
              np.isin(p2, ['410', '411', '412', '413', '414']) | \
              np.isin(p3, ['410', '411', '412', '413', '414'])

        fluid = (p1 == '276') | (p2 == '276') | (p3 == '276')
        dysr = (p1 == '427') | (p2 == '427') | (p3 == '427')

        def is_diab_comp(c):
            if c is None or c == '?' or c == 'None':
                return False
            s = str(c).strip()
            return s.startswith(('250.4', '250.5', '250.6', '250.7', '250.8', '250.9', '250.1', '250.2', '250.3'))

        diab_comp = np.array([is_diab_comp(c) for c in d1], dtype=bool) | \
                    np.array([is_diab_comp(c) for c in d2], dtype=bool) | \
                    np.array([is_diab_comp(c) for c in d3], dtype=bool)

        col_dict['comorb_has_circ'] = circ.astype(np.float64)
        col_dict['comorb_has_diab'] = diab.astype(np.float64)
        col_dict['comorb_has_resp'] = resp.astype(np.float64)
        col_dict['comorb_has_genit'] = genit.astype(np.float64)
        col_dict['comorb_has_chf'] = chf.astype(np.float64)
        col_dict['comorb_has_ckd'] = ckd.astype(np.float64)
        col_dict['comorb_has_ht'] = ht.astype(np.float64)
        col_dict['comorb_has_copd'] = copd.astype(np.float64)
        col_dict['comorb_has_aki'] = aki.astype(np.float64)
        col_dict['comorb_has_stroke'] = stroke.astype(np.float64)
        col_dict['comorb_has_cad'] = cad.astype(np.float64)
        col_dict['comorb_has_fluid_electrolyte'] = fluid.astype(np.float64)
        col_dict['comorb_has_dysrhythmia'] = dysr.astype(np.float64)
        col_dict['comorb_diab_complicated'] = diab_comp.astype(np.float64)
        col_dict['comorb_num_systems'] = (
            circ.astype(int) + diab.astype(int) + resp.astype(int) + digest.astype(int) + genit.astype(int)
        ).astype(np.float64)

    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group', 'diag_1_prefix'])
    extra_numeric.extend([
        'comorb_has_circ', 'comorb_has_diab', 'comorb_has_resp', 'comorb_has_genit',
        'comorb_has_chf', 'comorb_has_ckd', 'comorb_has_ht', 'comorb_has_copd', 'comorb_num_systems',
        'comorb_has_aki', 'comorb_has_stroke', 'comorb_has_cad', 'comorb_has_fluid_electrolyte',
        'comorb_has_dysrhythmia', 'comorb_diab_complicated'
    ])


    # 5. Rockwood Clinical Frailty & Deficit Saturation Index
    for col_dict in [train_data, eval_data]:
        age_val = np.nan_to_num(col_dict.get('num_age', 50.0), nan=50.0)
        time_val = np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0)
        inp_val = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg_val = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        diag_val = np.nan_to_num(col_dict.get('number_diagnoses', 1.0), nan=1.0)
        med_val = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)
        tot_visits = np.nan_to_num(col_dict.get('util_total_visits', 0.0), nan=0.0)

        d1 = (age_val >= 70.0).astype(np.float64)
        d2 = (time_val >= 6.0).astype(np.float64)
        d3 = (inp_val >= 1.0).astype(np.float64)
        d4 = (emg_val >= 1.0).astype(np.float64)
        d5 = (diag_val >= 8.0).astype(np.float64)
        d6 = (med_val >= 15.0).astype(np.float64)
        d7 = col_dict.get('comorb_has_chf', np.zeros_like(age_val))
        d8 = col_dict.get('comorb_has_ckd', np.zeros_like(age_val))
        d9 = col_dict.get('comorb_has_copd', np.zeros_like(age_val))
        d10 = (col_dict.get('comorb_num_systems', np.zeros_like(age_val)) >= 3.0).astype(np.float64)
        d11 = (col_dict.get('med_num_changed', np.zeros_like(age_val)) >= 1.0).astype(np.float64)
        d12 = col_dict.get('feat_disp_facility', np.zeros_like(age_val))
        d13 = col_dict.get('feat_adm_emergency', np.zeros_like(age_val))
        d14 = ((col_dict.get('feat_high_a1c', np.zeros_like(age_val)) == 1.0) |
               (col_dict.get('feat_high_glucose', np.zeros_like(age_val)) == 1.0)).astype(np.float64)
        d15 = ((col_dict.get('med_insulin_up', np.zeros_like(age_val)) == 1.0) |
               (col_dict.get('med_insulin_down', np.zeros_like(age_val)) == 1.0)).astype(np.float64)

        deficits = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14 + d15
        col_dict['frailty_deficit_count'] = deficits
        col_dict['frailty_deficit_index'] = deficits / 15.0
        col_dict['frailty_saturation'] = np.exp(2.0 * col_dict['frailty_deficit_index']) - 1.0
        col_dict['cardiorenal_syndrome'] = (col_dict.get('comorb_has_chf', 0.0) * col_dict.get('comorb_has_ckd', 0.0)).astype(np.float64)
        col_dict['triad_cardiorenal_diab'] = (col_dict['cardiorenal_syndrome'] * col_dict.get('comorb_diab_complicated', 0.0)).astype(np.float64)

        # LACE Index (Length of stay, Acuity, Comorbidities, Emergency visits)
        l_pts = np.where(time_val >= 14.0, 7.0,
                np.where(time_val >= 7.0, 5.0,
                np.where(time_val >= 4.0, 4.0, time_val)))
        a_pts = col_dict.get('feat_adm_emergency', np.zeros_like(time_val)) * 3.0
        c_pts = np.minimum(deficits, 6.0)
        e_pts = np.minimum(inp_val + emg_val, 4.0)
        lace = l_pts + a_pts + c_pts + e_pts
        col_dict['lace_score'] = lace.astype(np.float64)
        col_dict['lace_high_risk'] = (lace >= 10.0).astype(np.float64)

        col_dict['inter_age_visits'] = age_val * tot_visits
        col_dict['inter_age_diagnoses'] = age_val * diag_val
        col_dict['inter_visits_diagnoses'] = tot_visits * diag_val

        # Geriatric vulnerability features to mitigate elderly subgroup gap
        is_elderly = (age_val >= 80.0).astype(np.float64)
        is_very_elderly = (age_val >= 90.0).astype(np.float64)
        col_dict['feat_elderly_facility'] = is_elderly * col_dict.get('feat_disp_facility', 0.0)
        col_dict['feat_elderly_med_burden'] = is_elderly * (med_val >= 15.0).astype(np.float64)
        col_dict['feat_elderly_cardiorenal'] = is_elderly * col_dict.get('cardiorenal_syndrome', 0.0)
        col_dict['feat_elderly_emergency'] = is_elderly * col_dict.get('feat_adm_emergency', 0.0)
        col_dict['feat_very_elderly'] = is_very_elderly
        col_dict['feat_very_elderly_med_change'] = is_very_elderly * (col_dict.get('med_num_changed', np.zeros_like(age_val)) >= 1.0).astype(np.float64)
        col_dict['feat_very_elderly_home_health'] = is_very_elderly * col_dict.get('feat_disp_home_health', np.zeros_like(age_val))
        col_dict['feat_very_elderly_aki'] = is_very_elderly * col_dict.get('comorb_has_aki', np.zeros_like(age_val))

    extra_numeric.extend([
        'frailty_deficit_count', 'frailty_deficit_index', 'frailty_saturation', 'cardiorenal_syndrome',
        'triad_cardiorenal_diab', 'lace_score', 'lace_high_risk',
        'inter_age_visits', 'inter_age_diagnoses', 'inter_visits_diagnoses',
        'feat_elderly_facility', 'feat_elderly_med_burden', 'feat_elderly_cardiorenal',
        'feat_elderly_emergency', 'feat_very_elderly',
        'feat_very_elderly_med_change', 'feat_very_elderly_home_health', 'feat_very_elderly_aki'
    ])

    # 6. Latent Clinical Phenotyping via TruncatedSVD on Clinical Bag-of-Entities
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    n_tr = len(train_data['time_in_hospital'])
    n_ev = len(eval_data['time_in_hospital'])

    def build_clinical_tokens(col_dict, n_rows):
        tokens = [[] for _ in range(n_rows)]
        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            if diag_col in col_dict:
                arr = col_dict[diag_col]
                for i, code in enumerate(arr):
                    if code is not None and code != '?' and code != 'None' and str(code).strip() != '':
                        s = str(code).strip()
                        tokens[i].append(f'{diag_col}_{s}')
                        tokens[i].append(f'{diag_col}_p_{s.split(".")[0]}')
                        tokens[i].append(f'{diag_col}_g_{icd9_group(s)}')
        for m in med_cols:
            if m in col_dict:
                arr = col_dict[m]
                for i, v in enumerate(arr):
                    if v in ('Steady', 'Up', 'Down'):
                        tokens[i].append(f'med_{m}_{v}')
        disp = col_dict.get('discharge_disposition_id', np.array([]))
        adm_s = col_dict.get('admission_source_id', np.array([]))
        for i in range(n_rows):
            if i < len(disp) and disp[i] is not None:
                tokens[i].append(f'disp_{disp[i]}')
            if i < len(adm_s) and adm_s[i] is not None:
                tokens[i].append(f'adm_{adm_s[i]}')
        return [' '.join(t) for t in tokens]

    tr_text = build_clinical_tokens(train_data, n_tr)
    ev_text = build_clinical_tokens(eval_data, n_ev)

    bow_vec = CountVectorizer(min_df=8, binary=True, token_pattern=r'\S+')
    X_bow_tr = bow_vec.fit_transform(tr_text)
    X_bow_ev = bow_vec.transform(ev_text)

    svd = TruncatedSVD(n_components=6, random_state=42)
    X_svd_tr = svd.fit_transform(X_bow_tr)
    X_svd_ev = svd.transform(X_bow_ev)

    for k in range(6):
        col_name = f'svd_pheno_{k}'
        train_data[col_name] = X_svd_tr[:, k].astype(np.float64)
        eval_data[col_name] = X_svd_ev[:, k].astype(np.float64)
        extra_numeric.append(col_name)

    # 7. Taxonomy-Aware Hierarchical Empirical Bayes Target Encoding for Diagnoses
    def compute_hierarchical_diag_te(tr_d, ev_d, y_tr, n_splits=5, s_glob=40.0, s_grp=15.0, s_pref=8.0, seed=42):
        tr_clean = np.array([str(x).strip() if x is not None and x != '?' and x != 'None' and str(x).strip() != '' else 'missing' for x in tr_d], dtype=object)
        ev_clean = np.array([str(x).strip() if x is not None and x != '?' and x != 'None' and str(x).strip() != '' else 'missing' for x in ev_d], dtype=object)

        tr_grp = np.array([icd9_group(c) for c in tr_clean], dtype=object)
        ev_grp = np.array([icd9_group(c) for c in ev_clean], dtype=object)

        tr_pref = np.array([c.split('.')[0] if c != 'missing' else 'missing' for c in tr_clean], dtype=object)
        ev_pref = np.array([c.split('.')[0] if c != 'missing' else 'missing' for c in ev_clean], dtype=object)

        prior = float(np.mean(y_tr))
        tr_te = np.full(len(tr_d), prior, dtype=np.float64)
        ev_te = np.full(len(ev_d), prior, dtype=np.float64)

        def build_hier_rates(codes, prefs, grps, y, prior_val):
            grp_counts, grp_sums = {}, {}
            pref_counts, pref_sums, pref_to_grp = {}, {}, {}
            code_counts, code_sums, code_to_pref = {}, {}, {}

            for c, p, g, val in zip(codes, prefs, grps, y):
                grp_counts[g] = grp_counts.get(g, 0) + 1
                grp_sums[g] = grp_sums.get(g, 0.0) + val

                pref_counts[p] = pref_counts.get(p, 0) + 1
                pref_sums[p] = pref_sums.get(p, 0.0) + val
                pref_to_grp[p] = g

                code_counts[c] = code_counts.get(c, 0) + 1
                code_sums[c] = code_sums.get(c, 0.0) + val
                code_to_pref[c] = p

            grp_rate = {g: (grp_sums[g] + s_glob * prior_val) / (grp_counts[g] + s_glob) for g in grp_counts}
            pref_rate = {p: (pref_sums[p] + s_grp * grp_rate.get(pref_to_grp[p], prior_val)) / (pref_counts[p] + s_grp) for p in pref_counts}
            code_rate = {c: (code_sums[c] + s_pref * pref_rate.get(code_to_pref[c], prior_val)) / (code_counts[c] + s_pref) for c in code_counts}
            return code_rate, pref_rate, grp_rate

        def lookup(c, p, g, code_rate, pref_rate, grp_rate, prior_val):
            if c in code_rate:
                return code_rate[c]
            if p in pref_rate:
                return pref_rate[p]
            if g in grp_rate:
                return grp_rate[g]
            return prior_val

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_clean, y_tr):
            sub_c, sub_p, sub_g, sub_y = tr_clean[train_idx], tr_pref[train_idx], tr_grp[train_idx], y_tr[train_idx]
            v_c, v_p, v_g = tr_clean[val_idx], tr_pref[val_idx], tr_grp[val_idx]
            sub_prior = float(np.mean(sub_y))

            c_r, p_r, g_r = build_hier_rates(sub_c, sub_p, sub_g, sub_y, sub_prior)
            tr_te[val_idx] = [lookup(c, p, g, c_r, p_r, g_r, sub_prior) for c, p, g in zip(v_c, v_p, v_g)]

        full_c_r, full_p_r, full_g_r = build_hier_rates(tr_clean, tr_pref, tr_grp, y_tr, prior)
        ev_te[:] = [lookup(c, p, g, full_c_r, full_p_r, full_g_r, prior) for c, p, g in zip(ev_clean, ev_pref, ev_grp)]
        return tr_te, ev_te

        # Compute hierarchical TE on diag_1, diag_2, diag_3
    h_tr1, h_ev1 = compute_hierarchical_diag_te(train_data['diag_1'], eval_data['diag_1'], y_train)
    h_tr2, h_ev2 = compute_hierarchical_diag_te(train_data['diag_2'], eval_data['diag_2'], y_train)
    h_tr3, h_ev3 = compute_hierarchical_diag_te(train_data['diag_3'], eval_data['diag_3'], y_train)

    train_data['hier_te_diag_1'] = h_tr1
    eval_data['hier_te_diag_1'] = h_ev1
    train_data['hier_te_diag_2'] = h_tr2
    eval_data['hier_te_diag_2'] = h_ev2
    train_data['hier_te_diag_3'] = h_tr3
    eval_data['hier_te_diag_3'] = h_ev3

    train_data['hier_te_diag_max'] = np.maximum(np.maximum(h_tr1, h_tr2), h_tr3)
    eval_data['hier_te_diag_max'] = np.maximum(np.maximum(h_ev1, h_ev2), h_ev3)
    train_data['hier_te_diag_mean'] = (h_tr1 + h_tr2 + h_tr3) / 3.0
    eval_data['hier_te_diag_mean'] = (h_ev1 + h_ev2 + h_ev3) / 3.0

    extra_numeric.extend(['hier_te_diag_1', 'hier_te_diag_2', 'hier_te_diag_3', 'hier_te_diag_max', 'hier_te_diag_mean'])

    # Standard OOF target encoding on non-hierarchical high-cardinality features
    def compute_oof_te(tr_arr, y_tr, ev_arr, smoothing=25.0, n_splits=5, seed=42):
        tr_arr_str = np.array([str(x) if x is not None else 'missing' for x in tr_arr], dtype=object)
        ev_arr_str = np.array([str(x) if x is not None else 'missing' for x in ev_arr], dtype=object)

        prior = float(np.mean(y_tr))
        tr_te = np.full(len(tr_arr), prior, dtype=np.float64)
        ev_te = np.full(len(ev_arr), prior, dtype=np.float64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_arr_str, y_tr):
            sub_c = tr_arr_str[train_idx]
            sub_y = y_tr[train_idx]
            v_c = tr_arr_str[val_idx]

            counts = {}
            sums = {}
            for c, y in zip(sub_c, sub_y):
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y

            vals = np.empty(len(val_idx), dtype=np.float64)
            for i, c in enumerate(v_c):
                cnt = counts.get(c, 0)
                vals[i] = (sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior
            tr_te[val_idx] = vals

        full_counts = {}
        full_sums = {}
        for c, y in zip(tr_arr_str, y_tr):
            full_counts[c] = full_counts.get(c, 0) + 1
            full_sums[c] = full_sums.get(c, 0.0) + y

        for i, c in enumerate(ev_arr_str):
            cnt = full_counts.get(c, 0)
            ev_te[i] = (full_sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior

        return tr_te, ev_te

    te_targets = [
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'medical_specialty',
        'payer_code',
        'feat_disp_adm_comb'
    ]
    for c_name in te_targets:
        if c_name in train_data:
            tr_te, ev_te = compute_oof_te(train_data[c_name], y_train, eval_data[c_name], smoothing=25.0)
            te_col_name = f'te_{c_name}'
            train_data[te_col_name] = tr_te
            eval_data[te_col_name] = ev_te
            extra_numeric.append(te_col_name)

    # 8. Non-parametric Topological Manifold Geometry (OOF k-NN Density & Contrastive Pos/Neg Boundary)
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    manifold_cols = [
        'num_age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_diagnoses', 'util_total_visits',
        'util_inpatient_ratio', 'util_visit_severity', 'frailty_deficit_index',
        'hier_te_diag_max', 'svd_pheno_0', 'svd_pheno_1'
    ]

    def extract_manifold_matrix(col_dict, cols, n_rows):
        mat = np.zeros((n_rows, len(cols)), dtype=np.float64)
        for j, col in enumerate(cols):
            arr = np.nan_to_num(col_dict.get(col, np.zeros(n_rows)), nan=0.0)
            mat[:, j] = arr
        return mat

    X_m_tr_raw = extract_manifold_matrix(train_data, manifold_cols, n_tr)
    X_m_ev_raw = extract_manifold_matrix(eval_data, manifold_cols, n_ev)

    scaler = StandardScaler()
    X_m_tr = scaler.fit_transform(X_m_tr_raw)
    X_m_ev = scaler.transform(X_m_ev_raw)

    knn_risk_oof = np.full(n_tr, np.mean(y_train), dtype=np.float64)
    knn_contrastive_oof = np.zeros(n_tr, dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(X_m_tr, y_train):
        sub_X = X_m_tr[tr_idx]
        sub_y = y_train[tr_idx]
        v_X = X_m_tr[val_idx]

        nn_all = NearestNeighbors(n_neighbors=30, algorithm='kd_tree', metric='euclidean', n_jobs=1)
        nn_all.fit(sub_X)
        indices = nn_all.kneighbors(v_X, return_distance=False)
        knn_risk_oof[val_idx] = np.mean(sub_y[indices], axis=1)

        nn_pos = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
        nn_pos.fit(sub_X[sub_y == 1])
        d_pos = nn_pos.kneighbors(v_X, return_distance=True)[0].mean(axis=1)

        nn_neg = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
        nn_neg.fit(sub_X[sub_y == 0])
        d_neg = nn_neg.kneighbors(v_X, return_distance=True)[0].mean(axis=1)

        knn_contrastive_oof[val_idx] = (d_neg - d_pos) / (d_neg + d_pos + 1e-4)

    nn_all_full = NearestNeighbors(n_neighbors=30, algorithm='kd_tree', metric='euclidean', n_jobs=1)
    nn_all_full.fit(X_m_tr)
    ev_indices = nn_all_full.kneighbors(X_m_ev, return_distance=False)
    knn_risk_ev = np.mean(y_train[ev_indices], axis=1)

    nn_pos_full = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
    nn_pos_full.fit(X_m_tr[y_train == 1])
    ev_d_pos = nn_pos_full.kneighbors(X_m_ev, return_distance=True)[0].mean(axis=1)

    nn_neg_full = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
    nn_neg_full.fit(X_m_tr[y_train == 0])
    ev_d_neg = nn_neg_full.kneighbors(X_m_ev, return_distance=True)[0].mean(axis=1)
    knn_contrastive_ev = (ev_d_neg - ev_d_pos) / (ev_d_neg + ev_d_pos + 1e-4)

    train_data['knn_risk_density'] = knn_risk_oof
    eval_data['knn_risk_density'] = knn_risk_ev
    train_data['knn_contrastive_ratio'] = knn_contrastive_oof
    eval_data['knn_contrastive_ratio'] = knn_contrastive_ev
    extra_numeric.extend(['knn_risk_density', 'knn_contrastive_ratio'])

    # 9. Multilinear Tensor Decomposition (HOSVD / Tucker) on 3-Way Clinical Interaction Tensors
    from collections import Counter

    def hosvd_decomposition(T: np.ndarray, ranks: tuple):
        """Higher-Order Singular Value Decomposition (HOSVD) for low-rank multilinear tensor approximation."""
        d1, d2, d3 = T.shape
        r1, r2, r3 = min(ranks[0], d1), min(ranks[1], d2), min(ranks[2], d3)

        # Mode-1 unfolding SVD
        M1 = T.reshape(d1, d2 * d3)
        u1, _, _ = np.linalg.svd(M1, full_matrices=False)
        U1 = u1[:, :r1]

        # Mode-2 unfolding SVD
        M2 = np.transpose(T, (1, 0, 2)).reshape(d2, d1 * d3)
        u2, _, _ = np.linalg.svd(M2, full_matrices=False)
        U2 = u2[:, :r2]

        # Mode-3 unfolding SVD
        M3 = np.transpose(T, (2, 0, 1)).reshape(d3, d1 * d2)
        u3, _, _ = np.linalg.svd(M3, full_matrices=False)
        U3 = u3[:, :r3]

        # Core tensor G = T x1 U1^T x2 U2^T x3 U3^T
        core = np.einsum('ijk,ia,jb,kc->abc', T, U1, U2, U3)
        # Low-rank tensor reconstruction
        T_recon = np.einsum('abc,ia,jb,kc->ijk', core, U1, U2, U3)
        return U1, U2, U3, core, T_recon

    def compute_empirical_tensor(i_arr, j_arr, k_arr, y_arr, d1, d2, d3, smoothing=20.0):
        prior = float(np.mean(y_arr))
        flat_idx = i_arr * (d2 * d3) + j_arr * d3 + k_arr
        counts = np.bincount(flat_idx, minlength=d1 * d2 * d3).astype(np.float64)
        sums = np.bincount(flat_idx, weights=y_arr, minlength=d1 * d2 * d3).astype(np.float64)
        T_flat = (sums + smoothing * prior) / (counts + smoothing)
        return T_flat.reshape(d1, d2, d3)

    # Tensor 1: Primary Diagnosis (26) x Secondary Diagnosis (26) x Discharge Disposition (16)
    p1_counts = Counter(train_data['diag_1_prefix'])
    top_p1 = [p for p, _ in p1_counts.most_common(25) if p != 'missing']
    p1_map = {p: i + 1 for i, p in enumerate(top_p1)}

    p2_clean_tr = [extract_prefix(c) for c in train_data['diag_2']]
    p2_clean_ev = [extract_prefix(c) for c in eval_data['diag_2']]
    top_p2 = [p for p, _ in Counter(p2_clean_tr).most_common(25) if p != 'missing']
    p2_map = {p: i + 1 for i, p in enumerate(top_p2)}

    disp_tr = [str(x) if x is not None else 'missing' for x in train_data['discharge_disposition_id']]
    disp_ev = [str(x) if x is not None else 'missing' for x in eval_data['discharge_disposition_id']]
    top_disp = [d for d, _ in Counter(disp_tr).most_common(15) if d != 'missing']
    disp_map = {d: i + 1 for i, d in enumerate(top_disp)}

    t1_i_tr = np.array([p1_map.get(p, 0) for p in train_data['diag_1_prefix']], dtype=np.int64)
    t1_i_ev = np.array([p1_map.get(p, 0) for p in eval_data['diag_1_prefix']], dtype=np.int64)
    t1_j_tr = np.array([p2_map.get(p, 0) for p in p2_clean_tr], dtype=np.int64)
    t1_j_ev = np.array([p2_map.get(p, 0) for p in p2_clean_ev], dtype=np.int64)
    t1_k_tr = np.array([disp_map.get(d, 0) for d in disp_tr], dtype=np.int64)
    t1_k_ev = np.array([disp_map.get(d, 0) for d in disp_ev], dtype=np.int64)

    D1_1, D1_2, D1_3 = len(top_p1) + 1, len(top_p2) + 1, len(top_disp) + 1

    # Tensor 2: Diag 1 Group (10) x Med Regimen (24) x Utilization Strata (18)
    grp_list = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other', 'missing']
    grp_map = {g: i for i, g in enumerate(grp_list)}

    def build_med_code(col_dict):
        ins = col_dict.get('insulin', np.array([]))
        ins_code = np.array([3 if x == 'Down' else (2 if x == 'Up' else (1 if x == 'Steady' else 0)) for x in ins], dtype=np.int64)
        active = col_dict.get('med_num_active', np.zeros(len(ins_code)))
        ins_act = col_dict.get('med_insulin_active', np.zeros(len(ins_code)))
        oral = np.maximum(0.0, active - ins_act)
        oral_code = np.array([0 if o == 0 else (1 if o == 1 else 2) for o in oral], dtype=np.int64)
        ch = col_dict.get('change', np.array([]))
        ch_code = np.array([1 if x == 'Ch' else 0 for x in ch], dtype=np.int64)
        return ins_code * 6 + oral_code * 2 + ch_code

    def build_util_code(col_dict):
        inp = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        los = np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0)
        inp_code = np.array([0 if x == 0 else (1 if x == 1 else 2) for x in inp], dtype=np.int64)
        emg_code = np.array([0 if x == 0 else 1 for x in emg], dtype=np.int64)
        los_code = np.array([0 if x <= 3 else (1 if x <= 6 else 2) for x in los], dtype=np.int64)
        return inp_code * 6 + emg_code * 3 + los_code

    t2_i_tr = np.array([grp_map.get(g, 9) for g in train_data['diag_1_group']], dtype=np.int64)
    t2_i_ev = np.array([grp_map.get(g, 9) for g in eval_data['diag_1_group']], dtype=np.int64)
    t2_j_tr = build_med_code(train_data)
    t2_j_ev = build_med_code(eval_data)
    t2_k_tr = build_util_code(train_data)
    t2_k_ev = build_util_code(eval_data)

    D2_1, D2_2, D2_3 = 10, 24, 18

    # OOF Computation for Training Set
    oof_t1_recon = np.zeros(n_tr, dtype=np.float64)
    oof_t1_d1_0 = np.zeros(n_tr, dtype=np.float64)
    oof_t1_d1_1 = np.zeros(n_tr, dtype=np.float64)
    oof_t1_d2_0 = np.zeros(n_tr, dtype=np.float64)
    oof_t1_disp_0 = np.zeros(n_tr, dtype=np.float64)
    oof_t1_interaction = np.zeros(n_tr, dtype=np.float64)

    oof_t2_recon = np.zeros(n_tr, dtype=np.float64)
    oof_t2_med_0 = np.zeros(n_tr, dtype=np.float64)
    oof_t2_util_0 = np.zeros(n_tr, dtype=np.float64)

    skf_tensor = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_tensor.split(t1_i_tr, y_train):
        sub_y = y_train[tr_idx]

        # Fold Tensor 1
        T1 = compute_empirical_tensor(t1_i_tr[tr_idx], t1_j_tr[tr_idx], t1_k_tr[tr_idx], sub_y, D1_1, D1_2, D1_3, smoothing=20.0)
        U1, U2, U3, _, T1_recon = hosvd_decomposition(T1, ranks=(5, 5, 4))
        vi, vj, vk = t1_i_tr[val_idx], t1_j_tr[val_idx], t1_k_tr[val_idx]
        oof_t1_recon[val_idx] = T1_recon[vi, vj, vk]
        oof_t1_d1_0[val_idx] = U1[vi, 0]
        oof_t1_d1_1[val_idx] = U1[vi, 1]
        oof_t1_d2_0[val_idx] = U2[vj, 0]
        oof_t1_disp_0[val_idx] = U3[vk, 0]
        oof_t1_interaction[val_idx] = np.sum(U1[vi, :3] * U2[vj, :3] * U3[vk, :3], axis=1)

        # Fold Tensor 2
        T2 = compute_empirical_tensor(t2_i_tr[tr_idx], t2_j_tr[tr_idx], t2_k_tr[tr_idx], sub_y, D2_1, D2_2, D2_3, smoothing=15.0)
        U2_1, U2_2, U2_3, _, T2_recon = hosvd_decomposition(T2, ranks=(4, 4, 4))
        v2i, v2j, v2k = t2_i_tr[val_idx], t2_j_tr[val_idx], t2_k_tr[val_idx]
        oof_t2_recon[val_idx] = T2_recon[v2i, v2j, v2k]
        oof_t2_med_0[val_idx] = U2_2[v2j, 0]
        oof_t2_util_0[val_idx] = U2_3[v2k, 0]

    # Full Tensor Evaluation for Eval Set
    T1_full = compute_empirical_tensor(t1_i_tr, t1_j_tr, t1_k_tr, y_train, D1_1, D1_2, D1_3, smoothing=20.0)
    U1_f, U2_f, U3_f, _, T1_recon_f = hosvd_decomposition(T1_full, ranks=(5, 5, 4))
    evi, evj, evk = t1_i_ev, t1_j_ev, t1_k_ev
    ev_t1_recon = T1_recon_f[evi, evj, evk]
    ev_t1_d1_0 = U1_f[evi, 0]
    ev_t1_d1_1 = U1_f[evi, 1]
    ev_t1_d2_0 = U2_f[evj, 0]
    ev_t1_disp_0 = U3_f[evk, 0]
    ev_t1_interaction = np.sum(U1_f[evi, :3] * U2_f[evj, :3] * U3_f[evk, :3], axis=1)

    T2_full = compute_empirical_tensor(t2_i_tr, t2_j_tr, t2_k_tr, y_train, D2_1, D2_2, D2_3, smoothing=15.0)
    U2_1_f, U2_2_f, U2_3_f, _, T2_recon_f = hosvd_decomposition(T2_full, ranks=(4, 4, 4))
    ev2i, ev2j, ev2k = t2_i_ev, t2_j_ev, t2_k_ev
    ev_t2_recon = T2_recon_f[ev2i, ev2j, ev2k]
    ev_t2_med_0 = U2_2_f[ev2j, 0]
    ev_t2_util_0 = U2_3_f[ev2k, 0]

    train_data['hosvd_t1_recon'] = oof_t1_recon
    eval_data['hosvd_t1_recon'] = ev_t1_recon
    train_data['hosvd_t1_d1_0'] = oof_t1_d1_0
    eval_data['hosvd_t1_d1_0'] = ev_t1_d1_0
    train_data['hosvd_t1_d1_1'] = oof_t1_d1_1
    eval_data['hosvd_t1_d1_1'] = ev_t1_d1_1
    train_data['hosvd_t1_d2_0'] = oof_t1_d2_0
    eval_data['hosvd_t1_d2_0'] = ev_t1_d2_0
    train_data['hosvd_t1_disp_0'] = oof_t1_disp_0
    eval_data['hosvd_t1_disp_0'] = ev_t1_disp_0
    train_data['hosvd_t1_interaction'] = oof_t1_interaction
    eval_data['hosvd_t1_interaction'] = ev_t1_interaction

    train_data['hosvd_t2_recon'] = oof_t2_recon
    eval_data['hosvd_t2_recon'] = ev_t2_recon
    train_data['hosvd_t2_med_0'] = oof_t2_med_0
    eval_data['hosvd_t2_med_0'] = ev_t2_med_0
    train_data['hosvd_t2_util_0'] = oof_t2_util_0
    eval_data['hosvd_t2_util_0'] = ev_t2_util_0

    extra_numeric.extend([
        'hosvd_t1_recon', 'hosvd_t1_d1_0', 'hosvd_t1_d1_1', 'hosvd_t1_d2_0', 'hosvd_t1_disp_0',
        'hosvd_t1_interaction', 'hosvd_t2_recon', 'hosvd_t2_med_0', 'hosvd_t2_util_0'
    ])

    # 10. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 11. Diverse Ensemble with Node-Subspace Optimization and Weighted Rank Consensus
    import lightgbm as lgb
    import xgboost as xgb
    from scipy.stats import rankdata

    models = [
        # Model 1: Moderate depth, conservative with node subspace sampling and path smoothing
        (lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.02,
            num_leaves=18,
            min_child_samples=40,
            path_smooth=5.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            colsample_bynode=0.8,
            reg_alpha=0.2,
            reg_lambda=8.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ), 1.15),
        # Model 2: Extremely Randomized Trees mode (extra_trees=True) with leaf regularization
        (lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.025,
            num_leaves=24,
            extra_trees=True,
            min_child_samples=50,
            path_smooth=8.0,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.7,
            colsample_bynode=0.8,
            reg_alpha=0.5,
            reg_lambda=10.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ), 1.10),
        # Model 3: Deeper, heavily regularized, slower learning
        (lgb.LGBMClassifier(
            n_estimators=420,
            learning_rate=0.016,
            num_leaves=31,
            min_child_samples=75,
            path_smooth=10.0,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            colsample_bynode=0.8,
            reg_alpha=1.5,
            reg_lambda=20.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ), 1.10),
        # Model 4: Shallow, high bagging, fast learning
        (lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.038,
            num_leaves=12,
            min_child_samples=30,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ), 0.95),
        # Model 5: Cost-sensitive minority focusing (scale_pos_weight=1.4)
        (lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.02,
            num_leaves=20,
            scale_pos_weight=1.4,
            min_child_samples=60,
            path_smooth=5.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            colsample_bynode=0.8,
            reg_alpha=1.0,
            reg_lambda=12.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ), 1.05),
        # Model 6: XGBoost Model with node feature randomization
        (xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.6,
            colsample_bynode=0.8,
            reg_alpha=0.5,
            reg_lambda=10.0,
            random_state=101,
            n_jobs=1,
            verbosity=0,
        ), 0.85),
        # Model 7: Shallow XGBoost
        (xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=202,
            n_jobs=1,
            verbosity=0,
        ), 0.80)
    ]

    n_eval_rows = len(x_eval)
    rank_sum = np.zeros(n_eval_rows, dtype=np.float64)
    raw_prob_sum = np.zeros(n_eval_rows, dtype=np.float64)
    total_weight = 0.0

    for model, weight in models:
        if isinstance(model, lgb.LGBMClassifier):
            model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        else:
            model.fit(x_train, y_train)
        p = model.predict_proba(x_eval)[:, 1]
        r = rankdata(p) / float(n_eval_rows)
        rank_sum += weight * r
        raw_prob_sum += weight * p
        total_weight += weight

    avg_rank = rank_sum / total_weight
    mean_raw_prob = raw_prob_sum / total_weight

    # Quantile matching: enforce exact rank-consensus ordering with calibrated probabilities
    sorted_raw = np.sort(mean_raw_prob)
    rank_indices = np.argsort(np.argsort(avg_rank))
    calibrated_preds = sorted_raw[rank_indices]

    calibrated_preds = np.nan_to_num(calibrated_preds, nan=0.114, posinf=1.0, neginf=0.0)
    return np.clip(calibrated_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
