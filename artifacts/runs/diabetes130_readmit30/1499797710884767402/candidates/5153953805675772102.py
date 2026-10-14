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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize
    from scipy.special import ndtri, ndtr
    from scipy.stats import rankdata
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

    def make_pair_str(a, b):
        sa, sb = safe_str(a), safe_str(b)
        return f"{sa}_{sb}" if sa <= sb else f"{sb}_{sa}"

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
        d['has_diab_comp'] = has_diab_comp.astype(float)
        d['has_diab_acute'] = has_diab_acute.astype(float)
        d['has_uncontrolled_diab'] = has_uncontrolled_diab.astype(float)

        d['cardiorenal_risk'] = ((has_chf & has_ckd) | (has_chf & has_aki)).astype(float)
        d['cardiopulmonary_risk'] = (has_chf & has_copd).astype(float)
        d['triad_risk'] = (has_chf & has_ckd & has_diab_comp).astype(float)

        # Drug-Disease Interactions (FDA Contraindications)
        d['tzd_with_chf'] = (d['on_tzd'] * d['has_chf']).astype(float)
        d['sulfonylurea_with_ckd'] = (d['on_sulfonylurea'] * np.maximum(d['has_ckd'], d['has_aki'])).astype(float)

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
        # Additional Granular Comorbidities
        has_anemia = np.isin(c1, ['280', '281', '282', '283', '284', '285']) | np.isin(c2, ['280', '281', '282', '283', '284', '285']) | np.isin(c3, ['280', '281', '282', '283', '284', '285'])
        has_depression_psych = np.isin(c1, ['296', '300', '309', '311']) | np.isin(c2, ['296', '300', '309', '311']) | np.isin(c3, ['296', '300', '309', '311'])
        has_substance_abuse = np.isin(c1, ['303', '304', '305']) | np.isin(c2, ['303', '304', '305']) | np.isin(c3, ['303', '304', '305'])
        has_obesity = (c1 == '278') | (c2 == '278') | (c3 == '278')
        has_lipid_disorder = (c1 == '272') | (c2 == '272') | (c3 == '272')
        has_neoplasm = (d['diag_1_grp'] == 'neoplasms') | (d['diag_2_grp'] == 'neoplasms') | (d['diag_3_grp'] == 'neoplasms')

        has_diab_nephropathy = np.zeros(n_rows, dtype=bool)
        has_diab_neuropathy = np.zeros(n_rows, dtype=bool)
        has_diab_retinopathy = np.zeros(n_rows, dtype=bool)
        has_diab_pvd = np.zeros(n_rows, dtype=bool)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            has_diab_nephropathy |= np.array([str(x).startswith('250.4') for x in d[col]])
            has_diab_neuropathy |= np.array([str(x).startswith('250.6') for x in d[col]])
            has_diab_retinopathy |= np.array([str(x).startswith('250.5') for x in d[col]])
            has_diab_pvd |= np.array([str(x).startswith('250.7') for x in d[col]])

        d['has_anemia'] = has_anemia.astype(float)
        d['has_depression_psych'] = has_depression_psych.astype(float)
        d['has_substance_abuse'] = has_substance_abuse.astype(float)
        d['has_obesity'] = has_obesity.astype(float)
        d['has_lipid_disorder'] = has_lipid_disorder.astype(float)
        d['has_neoplasm'] = has_neoplasm.astype(float)
        d['has_diab_nephropathy'] = has_diab_nephropathy.astype(float)
        d['has_diab_neuropathy'] = has_diab_neuropathy.astype(float)
        d['has_diab_retinopathy'] = has_diab_retinopathy.astype(float)
        d['has_diab_pvd'] = has_diab_pvd.astype(float)

        d['diag_1_is_circulatory'] = (d['diag_1_grp'] == 'circulatory').astype(float)
        d['diag_1_is_respiratory'] = (d['diag_1_grp'] == 'respiratory').astype(float)
        d['diag_1_is_diabetes'] = (d['diag_1_grp'] == 'diabetes').astype(float)
        d['diag_1_is_digestive'] = (d['diag_1_grp'] == 'digestive').astype(float)
        d['diag_1_is_injury'] = (d['diag_1_grp'] == 'injury').astype(float)
        d['diag_1_is_genitourinary'] = (d['diag_1_grp'] == 'genitourinary').astype(float)

        d['same_diag_1_2'] = ((c1 == c2) & (c1 != 'missing')).astype(float)
        d['same_grp_1_2'] = ((d['diag_1_grp'] == d['diag_2_grp']) & (d['diag_1_grp'] != 'missing')).astype(float)
        d['same_grp_all_3'] = ((d['diag_1_grp'] == d['diag_2_grp']) & (d['diag_2_grp'] == d['diag_3_grp']) & (d['diag_1_grp'] != 'missing')).astype(float)

        # Additional Primary Care Utilization Ratios & Rates
        d['inpatient_to_outpatient_ratio'] = d['number_inpatient'] / (d['number_outpatient'] + 1.0)
        d['emergency_to_outpatient_ratio'] = d['number_emergency'] / (d['number_outpatient'] + 1.0)
        d['has_prior_emergency'] = (d['number_emergency'] > 0).astype(float)
        d['has_prior_outpatient'] = (d['number_outpatient'] > 0).astype(float)
        d['frequent_emergency'] = (d['number_emergency'] >= 2).astype(float)
        d['prior_inpatient_gte_3'] = (d['number_inpatient'] >= 3).astype(float)
        d['super_utilizer'] = (d['number_inpatient'] >= 5).astype(float)
        d['zero_prior_visits'] = (total_v == 0).astype(float)
        d['annual_inpatient_rate'] = d['number_inpatient'] / (d['age_num'] / 10.0 + 1.0)
        d['inp_stay_interaction'] = np.log1p(d['number_inpatient']) * np.log1p(d['time_in_hospital'])
        d['meds_per_diag'] = d['num_medications'] / (d['number_diagnoses'] + 1.0)
        d['labs_per_diag'] = d['num_lab_procedures'] / (d['number_diagnoses'] + 1.0)
        d['procedures_per_diag'] = d['num_procedures'] / (d['number_diagnoses'] + 1.0)
        d['is_purely_medical'] = ((d['num_procedures'] == 0) & (adm_type == '1')).astype(float)
        d['diabetes_med_share'] = d['active_meds'] / (d['num_medications'] + 1.0)
        d['metformin_with_ckd'] = (d['on_metformin'] * np.maximum(d['has_ckd'], d['has_aki'])).astype(float)

        d['a1c_normal'] = (a1c_res == 'Norm').astype(float)
        d['a1c_mod_high'] = (a1c_res == '>7').astype(float)
        d['a1c_not_tested'] = (a1c_res == 'None').astype(float)

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

        # HOSPITAL Readmission Risk Score (Donzé et al., JAMA Intern Med 2013)
        hosp_prev_adm = np.where(d['number_inpatient'] >= 5, 5.0, np.where(d['number_inpatient'] >= 1, 2.0, 0.0))
        d['hospital_score'] = (
            1.0 * has_anemia.astype(float) +
            2.0 * has_neoplasm.astype(float) +
            1.0 * d['has_fluid_elect'] +
            1.0 * (d['num_procedures'] == 0).astype(float) +
            1.0 * np.isin(adm_type, ['1', '2']).astype(float) +
            hosp_prev_adm +
            2.0 * (stay_raw >= 5).astype(float)
        )

        # Biophysical Destabilization Hamiltonian & Glycemic Conflict
        h_inertia = (((a1c_res == '>8') | np.isin(glu_res, ['>200', '>300'])) & (d['num_med_changes'] == 0) & (has_ins == 0)).astype(float)
        h_fragility = ((d['age_num'] >= 70.0) & (has_ckd | has_aki) & (has_ins > 0) & (d['on_sulfonylurea'] > 0)).astype(float)
        h_met_tox = (d['on_metformin'] * (has_ckd | has_aki)).astype(float)
        h_tzd_tox = (d['on_tzd'] * has_chf).astype(float)
        h_sulf_tox = (d['on_sulfonylurea'] * (has_ckd | has_aki)).astype(float)
        h_revolving = (d['number_inpatient'] * stay) / (d['age_num'] / 10.0 + 1.0)
        h_friction = d['high_risk_discharge'] * np.log1p(acute_v)

        d['h_inertia'] = h_inertia
        d['h_fragility'] = h_fragility
        d['h_metformin_tox'] = h_met_tox
        d['h_tzd_tox'] = h_tzd_tox
        d['h_sulf_tox'] = h_sulf_tox
        d['h_revolving_door'] = h_revolving
        d['h_discharge_friction'] = h_friction
        d['h_destabilization_total'] = (
            2.5 * h_inertia + 3.0 * h_fragility + 2.0 * h_met_tox +
            2.5 * h_tzd_tox + 2.0 * h_sulf_tox + 1.5 * h_friction +
            1.0 * np.log1p(np.maximum(0.0, h_revolving))
        )

        # Regimen Flux, Density & Fragmentation
        d['medication_flux'] = d['num_med_changes'] / (d['num_medications'] + 1.0)
        d['metformin_and_sulfonylurea'] = (d['on_metformin'] * d['on_sulfonylurea']).astype(float)
        d['triple_therapy'] = (d['on_metformin'] * d['on_sulfonylurea'] * has_ins).astype(float)
        d['sulf_monotherapy'] = (d['on_sulfonylurea'] * (1.0 - d['on_metformin']) * (1.0 - has_ins)).astype(float)
        d['bed_days_accum'] = (d['number_inpatient'] + 1.0) * d['time_in_hospital']
        d['diagnostic_density'] = d['number_diagnoses'] * np.log1p(stay)
        d['care_fragmentation'] = acute_v / (d['number_outpatient'] + 1.0)
        d['stay_per_diagnosis'] = d['time_in_hospital'] / (d['number_diagnoses'] + 1.0)
        d['lab_per_procedure'] = d['num_lab_procedures'] / (d['num_procedures'] + 1.0)

        # Diagnosis Synergistic Pairs
        d['pair_12'] = np.array([make_pair_str(a, b) for a, b in zip(c1, c2)], dtype=object)
        d['pair_13'] = np.array([make_pair_str(a, b) for a, b in zip(c1, c3)], dtype=object)
        d['pair_23'] = np.array([make_pair_str(a, b) for a, b in zip(c2, c3)], dtype=object)
        d['grp_pair_12'] = np.array([make_pair_str(a, b) for a, b in zip(d['diag_1_grp'], d['diag_2_grp'])], dtype=object)
        d['grp_pair_13'] = np.array([make_pair_str(a, b) for a, b in zip(d['diag_1_grp'], d['diag_3_grp'])], dtype=object)
        d['grp_pair_23'] = np.array([make_pair_str(a, b) for a, b in zip(d['diag_2_grp'], d['diag_3_grp'])], dtype=object)

        # Cross Interaction strings for Joint Target Encoding
        d['disp_x_source'] = np.array([f"{a}_{b}" for a, b in zip(disp, src)], dtype=object)
        d['diag1_grp_x_disp'] = np.array([f"{a}_{b}" for a, b in zip(d['diag_1_grp'], disp)], dtype=object)
        d['diag1_grp_x_adm'] = np.array([f"{a}_{b}" for a, b in zip(d['diag_1_grp'], adm_type)], dtype=object)
        d['disp_x_age'] = np.array([f"{a}_{b}" for a, b in zip(disp, d['age'])], dtype=object)
        d['disp_x_adm'] = np.array([f"{a}_{b}" for a, b in zip(disp, adm_type)], dtype=object)
        d['spec_x_disp'] = np.array([f"{safe_str(a)}_{b}" for a, b in zip(d['medical_specialty'], disp)], dtype=object)

    # Organ System Multimorbidity Entropy & NPMI Discordance
    grp_counts = {}
    total_grps = 0
    pair_counts = {}
    total_pairs = 0
    for i in range(len(y_train)):
        items = [train['diag_1_grp'][i], train['diag_2_grp'][i], train['diag_3_grp'][i]]
        valid = [x for x in items if x != 'missing']
        for g in valid:
            grp_counts[g] = grp_counts.get(g, 0) + 1
            total_grps += 1
        for p in [train['grp_pair_12'][i], train['grp_pair_13'][i], train['grp_pair_23'][i]]:
            if 'missing' not in str(p):
                pair_counts[p] = pair_counts.get(p, 0) + 1
                total_pairs += 1

    p_grp = {k: v / max(1.0, float(total_grps)) for k, v in grp_counts.items()}
    p_pair = {k: v / max(1.0, float(total_pairs)) for k, v in pair_counts.items()}

    def get_npmi(p_key, g_a, g_b):
        if 'missing' in str(p_key):
            return 0.0
        p_ab = p_pair.get(p_key, 1e-5)
        pa = p_grp.get(g_a, 1e-4)
        pb = p_grp.get(g_b, 1e-4)
        pmi = np.log(p_ab / (pa * pb + 1e-9) + 1e-9)
        norm = -np.log(p_ab + 1e-9)
        return float(np.clip(pmi / max(1e-6, norm), -1.0, 1.0))

    for d in [train, eval_cols]:
        n = len(d['gender'])
        g1_a, g2_a, g3_a = d['diag_1_grp'], d['diag_2_grp'], d['diag_3_grp']
        p12_a, p13_a, p23_a = d['grp_pair_12'], d['grp_pair_13'], d['grp_pair_23']
        u_counts = np.zeros(n, dtype=float)
        ents = np.zeros(n, dtype=float)
        min_npmi_a = np.zeros(n, dtype=float)
        mean_npmi_a = np.zeros(n, dtype=float)
        for i in range(n):
            items = [g1_a[i], g2_a[i], g3_a[i]]
            valid = [x for x in items if x != 'missing']
            if valid:
                u_counts[i] = float(len(set(valid)))
                m = float(len(valid))
                cd = {}
                for x in valid:
                    cd[x] = cd.get(x, 0) + 1
                e = 0.0
                for cnt in cd.values():
                    p = cnt / m
                    e -= p * np.log2(p)
                ents[i] = e

            v12 = get_npmi(p12_a[i], g1_a[i], g2_a[i])
            v13 = get_npmi(p13_a[i], g1_a[i], g3_a[i])
            v23 = get_npmi(p23_a[i], g2_a[i], g3_a[i])
            min_npmi_a[i] = min(v12, v13, v23)
            mean_npmi_a[i] = (v12 + v13 + v23) / 3.0

        d['multisystem_organ_count'] = u_counts
        d['organ_entropy'] = ents
        d['is_tri_systemic'] = (u_counts >= 3.0).astype(float)
        d['is_focal_disease'] = (u_counts == 1.0).astype(float)
        d['min_grp_npmi'] = min_npmi_a
        d['mean_grp_npmi'] = mean_npmi_a

    # Frequency Encoding (Count Rarity)
    def get_freq_map(arr):
        counts = {}
        for x in arr:
            s = safe_str(x)
            counts[s] = counts.get(s, 0) + 1
        n = float(len(arr))
        return {k: v / n for k, v in counts.items()}

    freq_d1_map = get_freq_map(train['diag_1_3d'])
    freq_d2_map = get_freq_map(train['diag_2_3d'])
    freq_d3_map = get_freq_map(train['diag_3_3d'])
    freq_spec_map = get_freq_map(train['medical_specialty'])
    freq_disp_map = get_freq_map(train['discharge_disposition_id'])

    for src_dict in [train, eval_cols]:
        src_dict['freq_diag_1'] = np.array([freq_d1_map.get(safe_str(x), 0.0) for x in src_dict['diag_1_3d']], dtype=float)
        src_dict['freq_diag_2'] = np.array([freq_d2_map.get(safe_str(x), 0.0) for x in src_dict['diag_2_3d']], dtype=float)
        src_dict['freq_diag_3'] = np.array([freq_d3_map.get(safe_str(x), 0.0) for x in src_dict['diag_3_3d']], dtype=float)
        src_dict['freq_specialty'] = np.array([freq_spec_map.get(safe_str(x), 0.0) for x in src_dict['medical_specialty']], dtype=float)
        src_dict['freq_discharge'] = np.array([freq_disp_map.get(safe_str(x), 0.0) for x in src_dict['discharge_disposition_id']], dtype=float)

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
    tr_disp_x_age = np.zeros(n_tr, dtype=float)
    tr_disp_x_adm = np.zeros(n_tr, dtype=float)
    tr_spec_x_disp = np.zeros(n_tr, dtype=float)
    tr_dpair_max = np.zeros(n_tr, dtype=float)
    tr_dpair_mean = np.zeros(n_tr, dtype=float)
    tr_gpair_max = np.zeros(n_tr, dtype=float)
    tr_gpair_mean = np.zeros(n_tr, dtype=float)

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

        dxs_map = compute_smoothed_map(train['disp_x_source'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp_x_src[val_idx] = np.array([dxs_map.get(safe_str(c), prior_glob) for c in train['disp_x_source'][val_idx]])

        gxd_map = compute_smoothed_map(train['diag1_grp_x_disp'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_diag1_grp_x_disp[val_idx] = np.array([gxd_map.get(safe_str(c), prior_glob) for c in train['diag1_grp_x_disp'][val_idx]])

        gxa_map = compute_smoothed_map(train['diag1_grp_x_adm'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_diag1_grp_x_adm[val_idx] = np.array([gxa_map.get(safe_str(c), prior_glob) for c in train['diag1_grp_x_adm'][val_idx]])

        payer_map = compute_smoothed_map(train['payer_code'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_payer[val_idx] = np.array([payer_map.get(safe_str(c), prior_glob) for c in train['payer_code'][val_idx]])

        dxa_map = compute_smoothed_map(train['disp_x_age'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp_x_age[val_idx] = np.array([dxa_map.get(safe_str(c), prior_glob) for c in train['disp_x_age'][val_idx]])

        dxadm_map = compute_smoothed_map(train['disp_x_adm'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp_x_adm[val_idx] = np.array([dxadm_map.get(safe_str(c), prior_glob) for c in train['disp_x_adm'][val_idx]])

        sxd_map = compute_smoothed_map(train['spec_x_disp'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_spec_x_disp[val_idx] = np.array([sxd_map.get(safe_str(c), prior_glob) for c in train['spec_x_disp'][val_idx]])

        pooled_d_pairs = np.concatenate([train['pair_12'][tr_idx], train['pair_13'][tr_idx], train['pair_23'][tr_idx]])
        dpair_map = compute_smoothed_map(pooled_d_pairs, pooled_y, smooth=30.0, prior=prior_glob)
        dp12 = np.array([dpair_map.get(safe_str(c), prior_glob) for c in train['pair_12'][val_idx]])
        dp13 = np.array([dpair_map.get(safe_str(c), prior_glob) for c in train['pair_13'][val_idx]])
        dp23 = np.array([dpair_map.get(safe_str(c), prior_glob) for c in train['pair_23'][val_idx]])
        tr_dpair_max[val_idx] = np.maximum(np.maximum(dp12, dp13), dp23)
        tr_dpair_mean[val_idx] = (dp12 + dp13 + dp23) / 3.0

        pooled_g_pairs = np.concatenate([train['grp_pair_12'][tr_idx], train['grp_pair_13'][tr_idx], train['grp_pair_23'][tr_idx]])
        gpair_map = compute_smoothed_map(pooled_g_pairs, pooled_y, smooth=20.0, prior=prior_glob)
        gp12 = np.array([gpair_map.get(safe_str(c), prior_glob) for c in train['grp_pair_12'][val_idx]])
        gp13 = np.array([gpair_map.get(safe_str(c), prior_glob) for c in train['grp_pair_13'][val_idx]])
        gp23 = np.array([gpair_map.get(safe_str(c), prior_glob) for c in train['grp_pair_23'][val_idx]])
        tr_gpair_max[val_idx] = np.maximum(np.maximum(gp12, gp13), gp23)
        tr_gpair_mean[val_idx] = (gp12 + gp13 + gp23) / 3.0

    train['te_diag_1'] = tr_diag_1
    train['te_diag_1_grp'] = tr_diag_1_grp
    train['te_diag_max'] = tr_diag_max
    train['te_diag_mean'] = tr_diag_mean
    train['te_diag_sum'] = tr_diag_sum
    train['te_discharge'] = tr_disp
    train['te_specialty'] = tr_spec
    train['te_adm_source'] = tr_src
    train['te_adm_type'] = tr_adm_type
    train['te_disp_x_src'] = tr_disp_x_src
    train['te_diag1_grp_x_disp'] = tr_diag1_grp_x_disp
    train['te_diag1_grp_x_adm'] = tr_diag1_grp_x_adm
    train['te_payer'] = tr_payer
    train['te_disp_x_age'] = tr_disp_x_age
    train['te_disp_x_adm'] = tr_disp_x_adm
    train['te_spec_x_disp'] = tr_spec_x_disp
    train['te_dpair_max'] = tr_dpair_max
    train['te_dpair_mean'] = tr_dpair_mean
    train['te_gpair_max'] = tr_gpair_max
    train['te_gpair_mean'] = tr_gpair_mean
    train['te_disp_odds'] = to_odds(tr_disp)
    train['te_diag_max_odds'] = to_odds(tr_diag_max)
    train['te_disp_x_src_odds'] = to_odds(tr_disp_x_src)
    train['te_payer_odds'] = to_odds(tr_payer)

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

    dxs_map_all = compute_smoothed_map(train['disp_x_source'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_disp_x_src'] = np.array([dxs_map_all.get(safe_str(c), prior_glob) for c in eval_cols['disp_x_source']])

    gxd_map_all = compute_smoothed_map(train['diag1_grp_x_disp'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_diag1_grp_x_disp'] = np.array([gxd_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_grp_x_disp']])

    gxa_map_all = compute_smoothed_map(train['diag1_grp_x_adm'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_diag1_grp_x_adm'] = np.array([gxa_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_grp_x_adm']])

    payer_map_all = compute_smoothed_map(train['payer_code'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_payer'] = np.array([payer_map_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_code']])

    dxa_map_all = compute_smoothed_map(train['disp_x_age'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_disp_x_age'] = np.array([dxa_map_all.get(safe_str(c), prior_glob) for c in eval_cols['disp_x_age']])

    dxadm_map_all = compute_smoothed_map(train['disp_x_adm'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_disp_x_adm'] = np.array([dxadm_map_all.get(safe_str(c), prior_glob) for c in eval_cols['disp_x_adm']])

    sxd_map_all = compute_smoothed_map(train['spec_x_disp'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_spec_x_disp'] = np.array([sxd_map_all.get(safe_str(c), prior_glob) for c in eval_cols['spec_x_disp']])

    pooled_d_pairs_all = np.concatenate([train['pair_12'], train['pair_13'], train['pair_23']])
    dpair_map_all = compute_smoothed_map(pooled_d_pairs_all, pooled_y_all, smooth=30.0, prior=prior_glob)
    edp12 = np.array([dpair_map_all.get(safe_str(c), prior_glob) for c in eval_cols['pair_12']])
    edp13 = np.array([dpair_map_all.get(safe_str(c), prior_glob) for c in eval_cols['pair_13']])
    edp23 = np.array([dpair_map_all.get(safe_str(c), prior_glob) for c in eval_cols['pair_23']])
    eval_cols['te_dpair_max'] = np.maximum(np.maximum(edp12, edp13), edp23)
    eval_cols['te_dpair_mean'] = (edp12 + edp13 + edp23) / 3.0

    pooled_g_pairs_all = np.concatenate([train['grp_pair_12'], train['grp_pair_13'], train['grp_pair_23']])
    gpair_map_all = compute_smoothed_map(pooled_g_pairs_all, pooled_y_all, smooth=20.0, prior=prior_glob)
    egp12 = np.array([gpair_map_all.get(safe_str(c), prior_glob) for c in eval_cols['grp_pair_12']])
    egp13 = np.array([gpair_map_all.get(safe_str(c), prior_glob) for c in eval_cols['grp_pair_13']])
    egp23 = np.array([gpair_map_all.get(safe_str(c), prior_glob) for c in eval_cols['grp_pair_23']])
    eval_cols['te_gpair_max'] = np.maximum(np.maximum(egp12, egp13), egp23)
    eval_cols['te_gpair_mean'] = (egp12 + egp13 + egp23) / 3.0

    eval_cols['te_disp_odds'] = to_odds(eval_cols['te_discharge'])
    eval_cols['te_diag_max_odds'] = to_odds(eval_cols['te_diag_max'])
    eval_cols['te_disp_x_src_odds'] = to_odds(eval_cols['te_disp_x_src'])
    eval_cols['te_payer_odds'] = to_odds(eval_cols['te_payer'])

    # 3. Clinical Phenotyping (NMF) & Narrative Semantic Topics (LSA)
    def build_phenotype_matrix(d):
        n = len(d['gender'])
        med_intensity_map = {'No': 0.0, 'Steady': 1.0, 'Down': 1.5, 'Up': 2.0}
        ins_val = np.array([med_intensity_map.get(str(x), 0.0) for x in d.get('insulin', ['No'] * n)])
        met_val = np.array([med_intensity_map.get(str(x), 0.0) for x in d.get('metformin', ['No'] * n)])

        sulf_val = np.zeros(n, dtype=float)
        for s_col in ['glipizide', 'glyburide', 'glimepiride']:
            if s_col in d:
                s_arr = np.array([med_intensity_map.get(str(x), 0.0) for x in d[s_col]])
                sulf_val = np.maximum(sulf_val, s_arr)

        log_inp = np.log1p(np.maximum(0.0, d['number_inpatient']))
        log_er = np.log1p(np.maximum(0.0, d['number_emergency']))
        stay_norm = d['time_in_hospital'] / 14.0

        cols = [
            d['has_chf'], d['has_ckd'], d['has_aki'], d['has_cad'], d['has_htn'], d['has_copd'],
            d['has_pvd'], d['has_stroke'], d['has_fluid_elect'], d['has_pneumonia'],
            d['has_anemia'], d['has_depression_psych'], d['has_neoplasm'],
            d['has_diab_acute'], d['has_diab_comp'], d['has_diab_nephropathy'],
            d['cardiorenal_risk'],
            d['lace_index'] / 19.0, d['hospital_score'] / 15.0, d['charlson_score'] / 10.0,
            ins_val, met_val, sulf_val,
            d['active_meds'] / 10.0, d['num_med_changes'] / 4.0,
            d['disch_is_snf_hh'], d['disch_is_ama'], d['disch_is_transfer'],
            d['is_emergency_adm'], log_inp, log_er, stay_norm,
            d['has_chf'] * ins_val,
            d['has_ckd'] * ins_val,
            d['has_chf'] * log_inp,
            d['has_ckd'] * log_inp,
        ]
        mat = np.column_stack(cols).astype(float)
        return np.maximum(0.0, np.nan_to_num(mat, nan=0.0))

    X_pheno_tr = build_phenotype_matrix(train)
    X_pheno_ev = build_phenotype_matrix(eval_cols)

    nmf = NMF(n_components=8, init='nndsvda', max_iter=200, random_state=42)
    tr_nmf = nmf.fit_transform(X_pheno_tr)
    ev_nmf = nmf.transform(X_pheno_ev)

    pheno_names = [f'pheno_{k}' for k in range(8)]
    for k in range(8):
        train[f'pheno_{k}'] = tr_nmf[:, k].astype(float)
        eval_cols[f'pheno_{k}'] = ev_nmf[:, k].astype(float)

    def build_narrative_corpus(d):
        n = len(d['gender'])
        med_pairs = [(m, d[m]) for m in med_cols if m in d]
        c = []
        for i in range(n):
            w = [
                f"d1_{d['diag_1_3d'][i]}", f"d2_{d['diag_2_3d'][i]}", f"d3_{d['diag_3_3d'][i]}",
                f"g1_{d['diag_1_grp'][i]}", f"g2_{d['diag_2_grp'][i]}", f"g3_{d['diag_3_grp'][i]}",
                f"disp_{safe_str(d['discharge_disposition_id'][i])}",
                f"src_{safe_str(d['admission_source_id'][i])}",
                f"spec_{safe_str(d['medical_specialty'][i])}",
                f"adm_{safe_str(d['admission_type_id'][i])}",
                f"age_{safe_str(d['age'][i])}",
            ]
            if d['number_inpatient'][i] > 0:
                w.append("prior_inp")
            if d['number_inpatient'][i] >= 2:
                w.append("freq_inp")
            if d['number_emergency'][i] > 0:
                w.append("prior_er")
            for m_name, m_vals in med_pairs:
                val = m_vals[i]
                if val is not None and val != 'No':
                    w.append(f"{m_name}_{val}")
            c.append(" ".join(w))
        return c

    tfidf = TfidfVectorizer(max_features=500, min_df=3, token_pattern=r'(?u)\b\S+\b')
    tr_tfidf = tfidf.fit_transform(build_narrative_corpus(train))
    ev_tfidf = tfidf.transform(build_narrative_corpus(eval_cols))

    svd = TruncatedSVD(n_components=6, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    topic_names = [f'topic_{k}' for k in range(6)]
    for k in range(6):
        train[f'topic_{k}'] = tr_svd[:, k].astype(float)
        eval_cols[f'topic_{k}'] = ev_svd[:, k].astype(float)

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'frequent_inpatient', 'prior_inpatient_gte_3', 'super_utilizer',
        'has_prior_emergency', 'has_prior_outpatient', 'frequent_emergency', 'zero_prior_visits',
        'inpatient_to_outpatient_ratio', 'emergency_to_outpatient_ratio', 'annual_inpatient_rate', 'inp_stay_interaction',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'diag_per_day', 'service_intensity',
        'meds_per_diag', 'labs_per_diag', 'procedures_per_diag', 'is_purely_medical',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits',
        'acute_visits', 'acute_ratio', 'log_acute_visits',
        'inp_per_day', 'med_proc_ratio', 'lab_med_ratio',
        'active_meds', 'num_med_changes', 'med_dose_increased', 'med_dose_decreased', 'diabetes_med_share',
        'has_insulin', 'insulin_up', 'insulin_down', 'change_flag', 'diabetes_med_flag',
        'num_oral_meds', 'on_insulin', 'insulin_only', 'oral_only', 'insulin_plus_oral',
        'on_metformin', 'on_sulfonylurea', 'on_tzd', 'tzd_with_chf', 'sulfonylurea_with_ckd', 'metformin_with_ckd',
        'a1c_tested', 'a1c_high', 'a1c_normal', 'a1c_mod_high', 'a1c_not_tested', 'a1c_high_no_change', 'glu_high',
        'disch_is_home', 'disch_is_snf', 'disch_is_hha', 'disch_is_snf_hh', 'disch_is_ama', 'disch_is_transfer',
        'high_risk_discharge', 'unstable_discharge',
        'is_emergency_adm', 'is_urgent_adm', 'is_elective_adm',
        'is_er_source', 'is_referral_source', 'is_transfer_source', 'adm_er_to_home', 'adm_er_to_snf',
        'payer_is_missing', 'specialty_is_missing', 'weight_is_recorded',
        'diag_1_is_circulatory', 'diag_1_is_respiratory', 'diag_1_is_diabetes', 'diag_1_is_digestive', 'diag_1_is_injury', 'diag_1_is_genitourinary',
        'same_diag_1_2', 'same_grp_1_2', 'same_grp_all_3',
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_dysrhythmia', 'has_sepsis', 'has_skin_ulcer',
        'has_gi_bleed', 'has_liver_disease', 'has_anemia', 'has_depression_psych', 'has_substance_abuse', 'has_obesity', 'has_lipid_disorder', 'has_neoplasm',
        'has_diab_acute', 'has_diab_comp', 'has_uncontrolled_diab',
        'has_diab_nephropathy', 'has_diab_neuropathy', 'has_diab_retinopathy', 'has_diab_pvd',
        'cardiorenal_risk', 'cardiopulmonary_risk', 'triad_risk',
        'comorbidity_count', 'charlson_score', 'frailty_index',
        'lace_l', 'lace_a', 'lace_c', 'lace_e', 'lace_index', 'lace_squared', 'hospital_score',
        'freq_diag_1', 'freq_diag_2', 'freq_diag_3', 'freq_specialty', 'freq_discharge',
        'h_inertia', 'h_fragility', 'h_metformin_tox', 'h_tzd_tox', 'h_sulf_tox', 'h_revolving_door', 'h_discharge_friction', 'h_destabilization_total',
        'medication_flux', 'metformin_and_sulfonylurea', 'triple_therapy', 'sulf_monotherapy',
        'bed_days_accum', 'diagnostic_density', 'care_fragmentation', 'stay_per_diagnosis', 'lab_per_procedure',
        'multisystem_organ_count', 'organ_entropy', 'is_tri_systemic', 'is_focal_disease', 'min_grp_npmi', 'mean_grp_npmi',
        'te_diag_1', 'te_diag_1_grp', 'te_diag_max', 'te_diag_mean', 'te_diag_sum',
        'te_dpair_max', 'te_dpair_mean', 'te_gpair_max', 'te_gpair_mean',
        'te_discharge', 'te_specialty', 'te_adm_source', 'te_adm_type', 'te_payer',
        'te_disp_x_src', 'te_diag1_grp_x_disp', 'te_diag1_grp_x_adm',
        'te_disp_x_age', 'te_disp_x_adm', 'te_spec_x_disp',
        'te_disp_odds', 'te_diag_max_odds', 'te_disp_x_src_odds', 'te_payer_odds',
    ] + pheno_names + topic_names

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'grp_pair_12', 'grp_pair_13', 'grp_pair_23',
        'disp_x_source', 'diag1_grp_x_disp', 'disp_x_age', 'disp_x_adm'
    ]

    # 4. Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 5. Quad-GBDT Ensemble: Leaf-Wise LGBM, ExtraTrees LGBM, HistGradientBoosting, and Hist-XGBoost (5 Folds)
    n_folds = 5
    skf_models = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_tr, dtype=float)
    oof_lgb_xt = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)
    oof_hgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_lgb = np.zeros(n_eval_rows, dtype=float)
    eval_lgb_xt = np.zeros(n_eval_rows, dtype=float)
    eval_xgb = np.zeros(n_eval_rows, dtype=float)
    eval_hgb = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # 1. Standard Leaf-Wise LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.030,
            num_leaves=31,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb[val_idx] = model_lgb.predict_proba(x_va)[:, 1]
        eval_lgb += model_lgb.predict_proba(x_eval)[:, 1] / float(n_folds)

        # 2. Extremely Randomized Depth-Constrained LightGBM (Orthogonal Splits)
        model_lgb_xt = lgb.LGBMClassifier(
            n_estimators=230,
            learning_rate=0.032,
            num_leaves=24,
            max_depth=6,
            extra_trees=True,
            min_child_samples=50,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=1.2,
            reg_lambda=5.0,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb_xt.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb_xt[val_idx] = model_lgb_xt.predict_proba(x_va)[:, 1]
        eval_lgb_xt += model_lgb_xt.predict_proba(x_eval)[:, 1] / float(n_folds)

        # 3. XGBoost (Depth-Wise Histogram Binning)
        model_xgb = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.030,
            max_depth=5,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        model_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = model_xgb.predict_proba(x_va)[:, 1]
        eval_xgb += model_xgb.predict_proba(x_eval)[:, 1] / float(n_folds)

        # 4. HistGradientBoostingClassifier (Bin-based Histogram)
        model_hgb = HistGradientBoostingClassifier(
            max_iter=190,
            learning_rate=0.032,
            max_leaf_nodes=31,
            min_samples_leaf=45,
            l2_regularization=4.0,
            categorical_features=encoder.categorical_indices,
            random_state=42 + fold,
        )
        model_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = model_hgb.predict_proba(x_va)[:, 1]
        eval_hgb += model_hgb.predict_proba(x_eval)[:, 1] / float(n_folds)

    # 6. Gaussian Copula (Van der Waerden Score) Rank Ensembling with Exact Mean Calibration
    def to_normal_scores(arr):
        n = len(arr)
        r = (rankdata(arr) - 0.5) / float(n)
        return ndtri(np.clip(r, 1e-5, 1.0 - 1e-5))

    z_oof_lgb = to_normal_scores(oof_lgb)
    z_oof_lgb_xt = to_normal_scores(oof_lgb_xt)
    z_oof_xgb = to_normal_scores(oof_xgb)
    z_oof_hgb = to_normal_scores(oof_hgb)

    def blend_neg_auc(weights):
        w = np.maximum(0.0, weights)
        s = np.sum(w)
        if s == 0:
            return 0.0
        w = w / s
        blend = w[0] * z_oof_lgb + w[1] * z_oof_lgb_xt + w[2] * z_oof_xgb + w[3] * z_oof_hgb
        return -roc_auc_score(y_train, blend)

    init_w = np.array([0.35, 0.20, 0.25, 0.20])
    opt_res = minimize(
        blend_neg_auc,
        init_w,
        method='Nelder-Mead',
        options={'maxiter': 140, 'xatol': 0.005, 'fatol': 0.0001}
    )
    w_opt = np.maximum(0.0, opt_res.x)
    if np.sum(w_opt) == 0:
        w_opt = init_w
    best_w = w_opt / np.sum(w_opt)

    z_eval = (
        best_w[0] * to_normal_scores(eval_lgb) +
        best_w[1] * to_normal_scores(eval_lgb_xt) +
        best_w[2] * to_normal_scores(eval_xgb) +
        best_w[3] * to_normal_scores(eval_hgb)
    )
    # Gaussian Integral shift: centers expected probability at prior_glob while perfectly preserving ROC-AUC ranks
    z_std = float(np.std(z_eval))
    z_normed = (z_eval - float(np.mean(z_eval))) / (z_std if z_std > 1e-6 else 1.0)
    calib_shift = np.sqrt(2.0) * ndtri(prior_glob)
    pred = ndtr(z_normed + calib_shift)
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
