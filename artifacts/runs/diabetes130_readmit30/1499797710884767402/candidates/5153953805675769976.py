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
    """Fit on `train`/`y_train` using Clinical LACE Index, Phenotyping & Quad-GBDT Simplex Ensembling."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize
    import xgboost as xgb

    non_med_cols = {
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    }
    med_cols = [col for col in task.categorical_columns if col not in non_med_cols]

    oral_med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]

    age_midpoints = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }

    def safe_str(val):
        if val is None or val == '?' or val == '':
            return '__missing__'
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        return str(code).strip().split('.')[0]

    # 1. Feature Engineering: Utilization, LACE Index, Comorbidities, Medications
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        # Hospital Utilization & Intensity Metrics
        total_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['total_visits'] = total_v
        d['inpatient_ratio'] = d['number_inpatient'] / (total_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (total_v + 1.0)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)

        stay = d['time_in_hospital'] + 1.0
        d['med_per_day'] = d['num_medications'] / stay
        d['lab_per_day'] = d['num_lab_procedures'] / stay
        d['proc_per_day'] = d['num_procedures'] / stay
        d['diag_per_day'] = d['number_diagnoses'] / stay
        d['service_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / stay

        d['log_inpatient'] = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_outpatient'] = np.log1p(np.maximum(0.0, d['number_outpatient']))
        d['log_total_visits'] = np.log1p(np.maximum(0.0, total_v))

        acute_v = d['number_inpatient'] + d['number_emergency']
        d['acute_visits'] = acute_v
        d['acute_ratio'] = acute_v / (total_v + 1.0)
        d['log_acute_visits'] = np.log1p(np.maximum(0.0, acute_v))
        d['inp_per_day'] = d['number_inpatient'] / stay
        d['has_prior_emergency'] = (d['number_emergency'] > 0).astype(float)
        d['med_proc_ratio'] = d['num_medications'] / (d['num_procedures'] + 1.0)
        d['lab_med_ratio'] = d['num_lab_procedures'] / (d['num_medications'] + 1.0)

        # Medications: Classes, Combinations & Instability
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        med_inc = np.zeros(n_rows, dtype=float)
        med_dec = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
                med_inc += (val == 'Up').astype(float)
                med_dec += (val == 'Down').astype(float)
        d['active_meds'] = active
        d['num_med_changes'] = changes
        d['med_dose_increased'] = (med_inc > 0).astype(float)
        d['med_dose_decreased'] = (med_dec > 0).astype(float)

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        has_ins = (insulin != 'No').astype(float)
        d['has_insulin'] = has_ins
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        num_oral = np.zeros(n_rows, dtype=float)
        for m in oral_med_names:
            if m in d:
                num_oral += (d[m] != 'No').astype(float)
        d['num_oral_meds'] = num_oral
        d['on_insulin'] = has_ins
        d['insulin_only'] = ((has_ins > 0) & (num_oral == 0)).astype(float)
        d['oral_only'] = ((has_ins == 0) & (num_oral > 0)).astype(float)
        d['insulin_plus_oral'] = ((has_ins > 0) & (num_oral > 0)).astype(float)

        d['on_metformin'] = (d.get('metformin', np.array(['No'] * n_rows)) != 'No').astype(float)

        on_sulf = np.zeros(n_rows, dtype=bool)
        for s in ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'tolazamide']:
            if s in d:
                on_sulf |= (d[s] != 'No')
        d['on_sulfonylurea'] = on_sulf.astype(float)

        on_tzd = np.zeros(n_rows, dtype=bool)
        for t in ['pioglitazone', 'rosiglitazone', 'troglitazone']:
            if t in d:
                on_tzd |= (d[t] != 'No')
        d['on_tzd'] = on_tzd.astype(float)

        # Biomarkers & Clinical Guidelines
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['a1c_high_no_change'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)

        # Admission & Discharge Dynamics
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        src = np.array([safe_str(x) for x in d['admission_source_id']])
        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])

        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf'] = (disp == '3').astype(float)
        d['disch_is_hha'] = (disp == '6').astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5']).astype(float)
        d['disch_is_snf_hh'] = np.isin(disp, ['3', '6', '22', '28']).astype(float)
        high_risk_disch = np.isin(disp, ['3', '6', '7', '22', '23', '24', '28'])
        d['high_risk_discharge'] = high_risk_disch.astype(float)
        d['unstable_discharge'] = (high_risk_disch & (d['num_med_changes'] > 0)).astype(float)

        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_urgent_adm'] = (adm_type == '2').astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)

        d['is_er_source'] = (src == '7').astype(float)
        d['is_referral_source'] = np.isin(src, ['1', '2']).astype(float)
        d['is_transfer_source'] = np.isin(src, ['4', '5', '6']).astype(float)

        d['adm_er_to_home'] = ((src == '7') & (disp == '1')).astype(float)
        d['adm_er_to_snf'] = ((src == '7') & (disp == '3')).astype(float)

        d['payer_is_missing'] = np.isin(d['payer_code'], [None, '?', '']).astype(float)
        d['specialty_is_missing'] = np.isin(d['medical_specialty'], [None, '?', '']).astype(float)
        d['weight_is_recorded'] = (~np.isin(d['weight'], [None, '?', ''])).astype(float)

        # Diagnoses: groups, 3-digit categories & comorbidity profile
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_ckd = np.isin(c1, ['585', '586']) | np.isin(c2, ['585', '586']) | np.isin(c3, ['585', '586'])
        has_aki = (c1 == '584') | (c2 == '584') | (c3 == '584')
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_pvd = np.isin(c1, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c2, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c3, ['440', '441', '442', '443', '444', '445', '446', '447'])
        has_stroke = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')
        has_pneumonia = np.isin(c1, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c2, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c3, ['480', '481', '482', '483', '484', '485', '486'])
        has_dysrhythmia = (c1 == '427') | (c2 == '427') | (c3 == '427')
        has_sepsis = (c1 == '038') | (c2 == '038') | (c3 == '038')
        has_skin_ulcer = np.isin(c1, ['707', '682']) | np.isin(c2, ['707', '682']) | np.isin(c3, ['707', '682'])
        has_gi_bleed = (c1 == '578') | (c2 == '578') | (c3 == '578')
        has_liver_disease = (c1 == '571') | (c2 == '571') | (c3 == '571')
        has_neoplasm = (d['diag_1_grp'] == 'neoplasms') | (d['diag_2_grp'] == 'neoplasms') | (d['diag_3_grp'] == 'neoplasms')

        has_diab_comp = np.zeros(n_rows, dtype=bool)
        has_diab_acute = np.zeros(n_rows, dtype=bool)
        has_uncontrolled_diab = np.zeros(n_rows, dtype=bool)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            has_diab_comp |= np.array([str(x).startswith('250') and str(x) not in ['250', '250.0', '250.00', '250.02', '250.1', '250.2', '250.3'] for x in d[col]])
            has_diab_acute |= np.array([str(x).startswith(('250.1', '250.2', '250.3')) for x in d[col]])
            has_uncontrolled_diab |= np.array([str(x).startswith('250') and (str(x).endswith('2') or str(x).endswith('3')) for x in d[col]])

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_aki'] = has_aki.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_pvd'] = has_pvd.astype(float)
        d['has_stroke'] = has_stroke.astype(float)
        d['has_fluid_elect'] = has_fluid_elect.astype(float)
        d['has_pneumonia'] = has_pneumonia.astype(float)
        d['has_dysrhythmia'] = has_dysrhythmia.astype(float)
        d['has_sepsis'] = has_sepsis.astype(float)
        d['has_skin_ulcer'] = has_skin_ulcer.astype(float)
        d['has_gi_bleed'] = has_gi_bleed.astype(float)
        d['has_liver_disease'] = has_liver_disease.astype(float)
        d['has_neoplasm'] = has_neoplasm.astype(float)
        d['has_diab_comp'] = has_diab_comp.astype(float)
        d['has_diab_acute'] = has_diab_acute.astype(float)
        d['has_uncontrolled_diab'] = has_uncontrolled_diab.astype(float)

        d['cardiorenal_risk'] = ((has_chf & has_ckd) | (has_chf & has_aki)).astype(float)
        d['cardiopulmonary_risk'] = (has_chf & has_copd).astype(float)
        d['triad_risk'] = (has_chf & has_ckd & has_diab_comp).astype(float)

        # Drug-Disease Interactions (FDA Contraindications)
        d['tzd_with_chf'] = (d['on_tzd'] * d['has_chf']).astype(float)
        d['sulfonylurea_with_ckd'] = (d['on_sulfonylurea'] * np.maximum(d['has_ckd'], d['has_aki'])).astype(float)
        d['metformin_with_ckd'] = (d['on_metformin'] * np.maximum(d['has_ckd'], d['has_aki'])).astype(float)

        charlson = (
            1.0 * has_cad.astype(float) +
            1.0 * has_copd.astype(float) +
            1.0 * has_pvd.astype(float) +
            1.0 * has_stroke.astype(float) +
            1.0 * has_htn.astype(float) +
            1.0 * (has_diab_comp & ~has_diab_acute).astype(float) +
            2.0 * has_chf.astype(float) +
            2.0 * has_ckd.astype(float) +
            2.0 * has_liver_disease.astype(float) +
            2.0 * has_sepsis.astype(float) +
            2.0 * has_diab_acute.astype(float)
        )
        d['charlson_score'] = charlson
        d['comorbidity_count'] = (
            d['has_chf'] + d['has_ckd'] + d['has_cad'] + d['has_htn'] + d['has_copd'] +
            d['has_pvd'] + d['has_stroke'] + d['has_fluid_elect'] + d['has_pneumonia'] + d['has_diab_comp']
        )
        d['frailty_index'] = d['comorbidity_count'] * (d['age_num'] / 50.0) * np.log1p(d['number_inpatient'] + 1.0)

        # LACE Readmission Risk Index (Length of stay, Acuity, Comorbidities, Emergency visits)
        stay_raw = d['time_in_hospital']
        lace_l = np.where(stay_raw >= 14, 7.0,
                 np.where(stay_raw >= 7, 5.0,
                 np.where(stay_raw >= 4, 4.0,
                 np.where(stay_raw >= 3, 3.0,
                 np.where(stay_raw >= 2, 2.0, 1.0)))))
        lace_a = np.where(adm_type == '1', 3.0, np.where(adm_type == '2', 2.0, 0.0))
        lace_c = np.clip(charlson, 0.0, 5.0)
        lace_e = np.clip(d['number_emergency'], 0.0, 4.0)
        lace_index = lace_l + lace_a + lace_c + lace_e
        d['lace_l'] = lace_l
        d['lace_a'] = lace_a
        d['lace_c'] = lace_c
        d['lace_e'] = lace_e
        d['lace_index'] = lace_index
        d['lace_squared'] = lace_index ** 2.0

        # Cross Interaction strings for Joint Target Encoding
        d['disp_x_source'] = np.array([f"{a}_{b}" for a, b in zip(disp, src)], dtype=object)
        d['diag1_grp_x_disp'] = np.array([f"{a}_{b}" for a, b in zip(d['diag_1_grp'], disp)], dtype=object)
        d['diag1_grp_x_adm'] = np.array([f"{a}_{b}" for a, b in zip(d['diag_1_grp'], adm_type)], dtype=object)

    # 2. Bayesian Smoothed Target Encoding (Out-of-Fold within train)
    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

    def to_odds(p):
        p_c = np.clip(p, 0.005, 0.995)
        return np.log(p_c / (1.0 - p_c))

    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def compute_freq_map(cats):
        counts = {}
        for c in cats:
            c_s = safe_str(c)
            counts[c_s] = counts.get(c_s, 0) + 1
        n = float(len(cats))
        return {k: v / n for k, v in counts.items()}

    # Frequency prevalence features
    for col_key, freq_name in [
        ('diag_1_3d', 'freq_diag_1'),
        ('discharge_disposition_id', 'freq_discharge'),
        ('medical_specialty', 'freq_specialty'),
        ('admission_source_id', 'freq_adm_source'),
        ('payer_code', 'freq_payer'),
    ]:
        f_map = compute_freq_map(train[col_key])
        train[freq_name] = np.array([f_map.get(safe_str(c), 0.0) for c in train[col_key]], dtype=float)
        eval_cols[freq_name] = np.array([f_map.get(safe_str(c), 0.0) for c in eval_cols[col_key]], dtype=float)

    tr_diag_1 = np.zeros(n_tr, dtype=float)
    tr_diag_1_grp = np.zeros(n_tr, dtype=float)
    tr_diag_max = np.zeros(n_tr, dtype=float)
    tr_diag_mean = np.zeros(n_tr, dtype=float)
    tr_diag_sum = np.zeros(n_tr, dtype=float)
    tr_disp = np.zeros(n_tr, dtype=float)
    tr_spec = np.zeros(n_tr, dtype=float)
    tr_src = np.zeros(n_tr, dtype=float)
    tr_adm_type = np.zeros(n_tr, dtype=float)
    tr_payer = np.zeros(n_tr, dtype=float)
    tr_disp_x_src = np.zeros(n_tr, dtype=float)
    tr_diag1_grp_x_disp = np.zeros(n_tr, dtype=float)
    tr_diag1_grp_x_adm = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf.split(train['gender'], y_train):
        d1_map = compute_smoothed_map(train['diag_1_3d'][tr_idx], y_train[tr_idx], smooth=25.0, prior=prior_glob)
        tr_diag_1[val_idx] = np.array([d1_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])

        d1_grp_map = compute_smoothed_map(train['diag_1_grp'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_diag_1_grp[val_idx] = np.array([d1_grp_map.get(safe_str(c), prior_glob) for c in train['diag_1_grp'][val_idx]])

        pooled_diag = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_y = np.concatenate([y_train[tr_idx], y_train[tr_idx], y_train[tr_idx]])
        d_map = compute_smoothed_map(pooled_diag, pooled_y, smooth=25.0, prior=prior_glob)
        d1 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        d2 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        d3 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        tr_diag_max[val_idx] = np.maximum(np.maximum(d1, d2), d3)
        tr_diag_mean[val_idx] = (d1 + d2 + d3) / 3.0
        tr_diag_sum[val_idx] = d1 + d2 + d3

        disp_map = compute_smoothed_map(train['discharge_disposition_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp[val_idx] = np.array([disp_map.get(safe_str(c), prior_glob) for c in train['discharge_disposition_id'][val_idx]])

        spec_map = compute_smoothed_map(train['medical_specialty'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_spec[val_idx] = np.array([spec_map.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

        src_map = compute_smoothed_map(train['admission_source_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_src[val_idx] = np.array([src_map.get(safe_str(c), prior_glob) for c in train['admission_source_id'][val_idx]])

        adm_map = compute_smoothed_map(train['admission_type_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_adm_type[val_idx] = np.array([adm_map.get(safe_str(c), prior_glob) for c in train['admission_type_id'][val_idx]])

        payer_map = compute_smoothed_map(train['payer_code'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_payer[val_idx] = np.array([payer_map.get(safe_str(c), prior_glob) for c in train['payer_code'][val_idx]])

        dxs_map = compute_smoothed_map(train['disp_x_source'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp_x_src[val_idx] = np.array([dxs_map.get(safe_str(c), prior_glob) for c in train['disp_x_source'][val_idx]])

        gxd_map = compute_smoothed_map(train['diag1_grp_x_disp'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_diag1_grp_x_disp[val_idx] = np.array([gxd_map.get(safe_str(c), prior_glob) for c in train['diag1_grp_x_disp'][val_idx]])

        gxa_map = compute_smoothed_map(train['diag1_grp_x_adm'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_diag1_grp_x_adm[val_idx] = np.array([gxa_map.get(safe_str(c), prior_glob) for c in train['diag1_grp_x_adm'][val_idx]])

    train['te_diag_1'] = tr_diag_1
    train['te_diag_1_grp'] = tr_diag_1_grp
    train['te_diag_max'] = tr_diag_max
    train['te_diag_mean'] = tr_diag_mean
    train['te_diag_sum'] = tr_diag_sum
    train['te_discharge'] = tr_disp
    train['te_specialty'] = tr_spec
    train['te_adm_source'] = tr_src
    train['te_adm_type'] = tr_adm_type
    train['te_payer'] = tr_payer
    train['te_disp_x_src'] = tr_disp_x_src
    train['te_diag1_grp_x_disp'] = tr_diag1_grp_x_disp
    train['te_diag1_grp_x_adm'] = tr_diag1_grp_x_adm
    train['te_disp_odds'] = to_odds(tr_disp)
    train['te_diag_max_odds'] = to_odds(tr_diag_max)
    train['te_disp_x_src_odds'] = to_odds(tr_disp_x_src)

    # Full train-to-eval target mappings
    d1_map_all = compute_smoothed_map(train['diag_1_3d'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_diag_1'] = np.array([d1_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])

    d1_grp_map_all = compute_smoothed_map(train['diag_1_grp'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_diag_1_grp'] = np.array([d1_grp_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_grp']])

    pooled_diag_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    d_map_all = compute_smoothed_map(pooled_diag_all, pooled_y_all, smooth=25.0, prior=prior_glob)
    ed1 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    ed2 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    ed3 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_max'] = np.maximum(np.maximum(ed1, ed2), ed3)
    eval_cols['te_diag_mean'] = (ed1 + ed2 + ed3) / 3.0
    eval_cols['te_diag_sum'] = ed1 + ed2 + ed3

    disp_map_all = compute_smoothed_map(train['discharge_disposition_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_discharge'] = np.array([disp_map_all.get(safe_str(c), prior_glob) for c in eval_cols['discharge_disposition_id']])

    spec_map_all = compute_smoothed_map(train['medical_specialty'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([spec_map_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']])

    src_map_all = compute_smoothed_map(train['admission_source_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_source'] = np.array([src_map_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_source_id']])

    adm_map_all = compute_smoothed_map(train['admission_type_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_type'] = np.array([adm_map_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_type_id']])

    payer_map_all = compute_smoothed_map(train['payer_code'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_payer'] = np.array([payer_map_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_code']])

    dxs_map_all = compute_smoothed_map(train['disp_x_source'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_disp_x_src'] = np.array([dxs_map_all.get(safe_str(c), prior_glob) for c in eval_cols['disp_x_source']])

    gxd_map_all = compute_smoothed_map(train['diag1_grp_x_disp'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_diag1_grp_x_disp'] = np.array([gxd_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_grp_x_disp']])

    gxa_map_all = compute_smoothed_map(train['diag1_grp_x_adm'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_diag1_grp_x_adm'] = np.array([gxa_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_grp_x_adm']])

    eval_cols['te_disp_odds'] = to_odds(eval_cols['te_discharge'])
    eval_cols['te_diag_max_odds'] = to_odds(eval_cols['te_diag_max'])
    eval_cols['te_disp_x_src_odds'] = to_odds(eval_cols['te_disp_x_src'])

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'frequent_inpatient', 'has_prior_emergency',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'diag_per_day', 'service_intensity',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits',
        'acute_visits', 'acute_ratio', 'log_acute_visits',
        'inp_per_day', 'med_proc_ratio', 'lab_med_ratio',
        'active_meds', 'num_med_changes', 'med_dose_increased', 'med_dose_decreased',
        'has_insulin', 'insulin_up', 'insulin_down', 'change_flag', 'diabetes_med_flag',
        'num_oral_meds', 'on_insulin', 'insulin_only', 'oral_only', 'insulin_plus_oral',
        'on_metformin', 'on_sulfonylurea', 'on_tzd', 'tzd_with_chf', 'sulfonylurea_with_ckd', 'metformin_with_ckd',
        'a1c_tested', 'a1c_high', 'a1c_normal', 'a1c_high_no_change', 'glu_high', 'glu_tested',
        'disch_is_home', 'disch_is_snf', 'disch_is_hha', 'disch_is_snf_hh', 'disch_is_ama', 'disch_is_transfer',
        'high_risk_discharge', 'unstable_discharge',
        'is_emergency_adm', 'is_urgent_adm', 'is_elective_adm',
        'is_er_source', 'is_referral_source', 'is_transfer_source', 'adm_er_to_home', 'adm_er_to_snf',
        'payer_is_missing', 'specialty_is_missing', 'weight_is_recorded',
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_dysrhythmia', 'has_sepsis', 'has_skin_ulcer',
        'has_gi_bleed', 'has_liver_disease', 'has_neoplasm', 'has_diab_acute', 'has_diab_comp', 'has_uncontrolled_diab',
        'cardiorenal_risk', 'cardiopulmonary_risk', 'triad_risk',
        'comorbidity_count', 'charlson_score', 'frailty_index',
        'lace_l', 'lace_a', 'lace_c', 'lace_e', 'lace_index', 'lace_squared',
        'te_diag_1', 'te_diag_1_grp', 'te_diag_max', 'te_diag_mean', 'te_diag_sum',
        'te_discharge', 'te_specialty', 'te_adm_source', 'te_adm_type', 'te_payer',
        'te_disp_x_src', 'te_diag1_grp_x_disp', 'te_diag1_grp_x_adm',
        'te_disp_odds', 'te_diag_max_odds', 'te_disp_x_src_odds',
        'freq_diag_1', 'freq_discharge', 'freq_specialty', 'freq_adm_source', 'freq_payer',
    ]

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'disp_x_source', 'diag1_grp_x_disp'
    ]

    # 4. Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 5. Quad-GBDT 5-Fold Ensemble: Leaf-Wise LGBM, Depth-Constrained LGBM, ExtraTrees LGBM, and Hist-XGBoost
    n_splits = 5
    skf_models = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_tr, dtype=float)
    oof_lgb_dc = np.zeros(n_tr, dtype=float)
    oof_lgb_xt = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_lgb = np.zeros(n_eval_rows, dtype=float)
    eval_lgb_dc = np.zeros(n_eval_rows, dtype=float)
    eval_lgb_xt = np.zeros(n_eval_rows, dtype=float)
    eval_xgb = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # 1. Standard Leaf-Wise LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.6,
            reg_lambda=3.5,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb[val_idx] = model_lgb.predict_proba(x_va)[:, 1]
        eval_lgb += model_lgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 2. Depth-Constrained LightGBM (Level-Wise Tree Regularization)
        model_lgb_dc = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            max_depth=5,
            num_leaves=24,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb_dc.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb_dc[val_idx] = model_lgb_dc.predict_proba(x_va)[:, 1]
        eval_lgb_dc += model_lgb_dc.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 3. Extremely Randomized LightGBM (Orthogonal Random Splits)
        model_lgb_xt = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=28,
            max_depth=6,
            extra_trees=True,
            min_child_samples=45,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=242 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb_xt.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb_xt[val_idx] = model_lgb_xt.predict_proba(x_va)[:, 1]
        eval_lgb_xt += model_lgb_xt.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 4. XGBoost (Exact Second-Order Hessian Gain & Hist Binning)
        model_xgb = xgb.XGBClassifier(
            n_estimators=210,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        model_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = model_xgb.predict_proba(x_va)[:, 1]
        eval_xgb += model_xgb.predict_proba(x_eval)[:, 1] / float(n_splits)

    # 6. Regularized Simplex Optimization in Log-Odds Space for ROC-AUC
    def to_logit(p, eps=1e-6):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    z_oof_lgb = to_logit(oof_lgb)
    z_oof_lgb_dc = to_logit(oof_lgb_dc)
    z_oof_lgb_xt = to_logit(oof_lgb_xt)
    z_oof_xgb = to_logit(oof_xgb)

    def blend_neg_auc(weights):
        w = np.maximum(0.0, weights)
        s = np.sum(w)
        if s == 0:
            return 0.0
        w = w / s
        blend = w[0] * z_oof_lgb + w[1] * z_oof_lgb_dc + w[2] * z_oof_lgb_xt + w[3] * z_oof_xgb
        return -roc_auc_score(y_train, blend)

    init_w = np.array([0.35, 0.25, 0.20, 0.20])
    opt_res = minimize(
        blend_neg_auc,
        init_w,
        method='Nelder-Mead',
        options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.0001}
    )
    w_opt = np.maximum(0.0, opt_res.x)
    if np.sum(w_opt) == 0:
        w_opt = init_w
    w_norm = w_opt / np.sum(w_opt)
    # Bayesian shrinkage toward uniform prior: prevents overfitting OOF quirks
    best_w = 0.85 * w_norm + 0.15 * np.array([0.25, 0.25, 0.25, 0.25])

    z_eval = (
        best_w[0] * to_logit(eval_lgb) +
        best_w[1] * to_logit(eval_lgb_dc) +
        best_w[2] * to_logit(eval_lgb_xt) +
        best_w[3] * to_logit(eval_xgb)
    )
    pred = 1.0 / (1.0 + np.exp(-z_eval))
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
