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
    """Fit on `train`/`y_train` using Clinical LACE+ Scoring, Comorbidity Indices & Simplex Quad-GBDTs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize
    import xgboost as xgb
    import lightgbm as lgb

    non_med_cols = {
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    }
    med_cols = [col for col in task.categorical_columns if col not in non_med_cols]

    age_midpoints = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }

    def safe_str(val):
        if val is None or val == '?' or val == '':
            return '__missing__'
        try:
            if isinstance(val, float) and np.isnan(val):
                return '__missing__'
        except Exception:
            pass
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        try:
            if isinstance(code, float) and np.isnan(code):
                return 'missing'
        except Exception:
            pass
        return str(code).strip().split('.')[0]

    def check_neoplasm(codes):
        res = np.zeros(len(codes), dtype=bool)
        for idx, c in enumerate(codes):
            try:
                v = int(c)
                if 140 <= v <= 208:
                    res[idx] = True
            except Exception:
                pass
        return res

    # 1. Domain Feature Engineering: Clinical Scores, Biomarkers, and Dynamics
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        # Diagnoses cleaning & chapter groupings
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']

        # Clinical Condition Flags (ICD-9)
        has_mi = np.isin(c1, ['410', '412']) | np.isin(c2, ['410', '412']) | np.isin(c3, ['410', '412'])
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_htn_comp = np.isin(c1, ['402', '403', '404', '405']) | np.isin(c2, ['402', '403', '404', '405']) | np.isin(c3, ['402', '403', '404', '405'])
        has_pvd = np.isin(c1, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c2, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c3, ['440', '441', '442', '443', '444', '445', '446', '447'])
        has_stroke = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])
        has_dementia = np.isin(c1, ['290', '294', '331']) | np.isin(c2, ['290', '294', '331']) | np.isin(c3, ['290', '294', '331'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_rheum = np.isin(c1, ['710', '714', '725']) | np.isin(c2, ['710', '714', '725']) | np.isin(c3, ['710', '714', '725'])
        has_pud = np.isin(c1, ['531', '532', '533', '534']) | np.isin(c2, ['531', '532', '533', '534']) | np.isin(c3, ['531', '532', '533', '534'])
        has_mild_liver = (c1 == '571') | (c2 == '571') | (c3 == '571')
        has_sev_liver = np.isin(c1, ['572', '456']) | np.isin(c2, ['572', '456']) | np.isin(c3, ['572', '456'])
        has_hemi = np.isin(c1, ['342', '344']) | np.isin(c2, ['342', '344']) | np.isin(c3, ['342', '344'])
        has_ckd = np.isin(c1, ['582', '583', '585', '586', '588']) | np.isin(c2, ['582', '583', '585', '586', '588']) | np.isin(c3, ['582', '583', '585', '586', '588'])
        has_aki = (c1 == '584') | (c2 == '584') | (c3 == '584')
        has_renal = has_ckd | has_aki
        has_met_cancer = np.isin(c1, ['196', '197', '198', '199']) | np.isin(c2, ['196', '197', '198', '199']) | np.isin(c3, ['196', '197', '198', '199'])
        has_hiv = (c1 == '042') | (c2 == '042') | (c3 == '042')
        has_neoplasm = check_neoplasm(c1) | check_neoplasm(c2) | check_neoplasm(c3)
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')
        has_pneumonia = np.isin(c1, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c2, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c3, ['480', '481', '482', '483', '484', '485', '486'])
        has_uti = (c1 == '599') | (c2 == '599') | (c3 == '599')
        has_sepsis = np.isin(c1, ['038', '995']) | np.isin(c2, ['038', '995']) | np.isin(c3, ['038', '995'])
        has_anemia = np.isin(c1, ['280', '281', '282', '283', '284', '285']) | np.isin(c2, ['280', '281', '282', '283', '284', '285']) | np.isin(c3, ['280', '281', '282', '283', '284', '285'])
        has_afib = (c1 == '427') | (c2 == '427') | (c3 == '427')
        has_valvular = np.isin(c1, ['394', '395', '396', '397', '424']) | np.isin(c2, ['394', '395', '396', '397', '424']) | np.isin(c3, ['394', '395', '396', '397', '424'])
        has_pulm_circ = np.isin(c1, ['416', '417']) | np.isin(c2, ['416', '417']) | np.isin(c3, ['416', '417'])
        has_depression = np.isin(c1, ['296', '300', '309', '311']) | np.isin(c2, ['296', '300', '309', '311']) | np.isin(c3, ['296', '300', '309', '311'])
        has_substance = np.isin(c1, ['291', '292', '303', '304', '305']) | np.isin(c2, ['291', '292', '303', '304', '305']) | np.isin(c3, ['291', '292', '303', '304', '305'])
        has_obesity = (c1 == '278') | (c2 == '278') | (c3 == '278')
        has_ulcer_skin = np.isin(c1, ['680', '681', '682', '683', '684', '685', '686', '707']) | np.isin(c2, ['680', '681', '682', '683', '684', '685', '686', '707']) | np.isin(c3, ['680', '681', '682', '683', '684', '685', '686', '707'])

        # Detailed Diabetic Manifestations (5th-digit ICD-9 schema)
        is_diab_type1 = np.zeros(n_rows, dtype=float)
        is_diab_uncontrolled = np.zeros(n_rows, dtype=float)
        has_diab_uncomp = np.zeros(n_rows, dtype=float)
        has_diab_acute = np.zeros(n_rows, dtype=float)
        has_diab_comp = np.zeros(n_rows, dtype=float)
        has_diab_renal = np.zeros(n_rows, dtype=float)
        has_diab_retino = np.zeros(n_rows, dtype=float)
        has_diab_neuro = np.zeros(n_rows, dtype=float)
        has_diab_vasc = np.zeros(n_rows, dtype=float)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            for idx, raw in enumerate(d[col]):
                if raw is None or raw == '?' or raw == '':
                    continue
                s = str(raw).strip()
                if s.startswith('250'):
                    if s.endswith(('1', '3')):
                        is_diab_type1[idx] = 1.0
                    if s.endswith(('2', '3')):
                        is_diab_uncontrolled[idx] = 1.0
                    if s.startswith(('250.1', '250.2', '250.3')):
                        has_diab_acute[idx] = 1.0
                    elif s.startswith('250.4'):
                        has_diab_renal[idx] = 1.0
                        has_diab_comp[idx] = 1.0
                    elif s.startswith('250.5'):
                        has_diab_retino[idx] = 1.0
                        has_diab_comp[idx] = 1.0
                    elif s.startswith('250.6'):
                        has_diab_neuro[idx] = 1.0
                        has_diab_comp[idx] = 1.0
                    elif s.startswith('250.7'):
                        has_diab_vasc[idx] = 1.0
                        has_diab_comp[idx] = 1.0
                    elif s.startswith(('250.8', '250.9')):
                        has_diab_comp[idx] = 1.0
                    else:
                        has_diab_uncomp[idx] = 1.0

        d['is_diab_type1'] = is_diab_type1
        d['is_diab_uncontrolled'] = is_diab_uncontrolled
        d['has_diab_uncomp'] = has_diab_uncomp
        d['has_diab_acute'] = has_diab_acute
        d['has_diab_comp'] = has_diab_comp
        d['has_diab_renal'] = has_diab_renal
        d['has_diab_retino'] = has_diab_retino
        d['has_diab_neuro'] = has_diab_neuro
        d['has_diab_vasc'] = has_diab_vasc
        d['num_diab_organs'] = (has_diab_renal + has_diab_retino + has_diab_neuro + has_diab_vasc)

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_aki'] = has_aki.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_htn_comp'] = has_htn_comp.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_pvd'] = has_pvd.astype(float)
        d['has_stroke'] = has_stroke.astype(float)
        d['has_fluid_elect'] = has_fluid_elect.astype(float)
        d['has_pneumonia'] = has_pneumonia.astype(float)
        d['has_uti'] = has_uti.astype(float)
        d['has_sepsis'] = has_sepsis.astype(float)
        d['has_anemia'] = has_anemia.astype(float)
        d['has_afib'] = has_afib.astype(float)
        d['has_valvular'] = has_valvular.astype(float)
        d['has_pulm_circ'] = has_pulm_circ.astype(float)
        d['has_depression'] = has_depression.astype(float)
        d['has_substance'] = has_substance.astype(float)
        d['has_obesity'] = has_obesity.astype(float)
        d['has_ulcer_skin'] = has_ulcer_skin.astype(float)

        # Primary Admitting Diagnosis Context
        d['diag1_is_chf'] = (c1 == '428').astype(float)
        d['diag1_is_circ'] = (d['diag_1_grp'] == 'circulatory').astype(float)
        d['diag1_is_resp'] = (d['diag_1_grp'] == 'respiratory').astype(float)
        d['diag1_is_diab'] = (d['diag_1_grp'] == 'diabetes').astype(float)
        d['diag1_is_digestive'] = (d['diag_1_grp'] == 'digestive').astype(float)
        d['diag1_is_infection'] = np.isin(c1, ['038', '480', '481', '482', '485', '486', '599', '682']).astype(float)
        d['diag1_is_injury'] = (d['diag_1_grp'] == 'injury').astype(float)

        # Multimorbid Complications & Syndrome Bundles
        d['cardiorenal_risk'] = ((has_chf | has_cad) & has_renal).astype(float)
        d['cardiorenal_anemia'] = ((d['cardiorenal_risk'] > 0.5) & has_anemia).astype(float)
        d['cardiorespiratory'] = (has_chf & has_copd).astype(float)
        d['metabolic_sepsis'] = (((has_diab_acute > 0.5) | has_fluid_elect) & (has_sepsis | has_pneumonia | has_uti)).astype(float)
        d['vascular_burden'] = (has_cad.astype(float) + has_pvd.astype(float) + has_stroke.astype(float) + has_htn.astype(float))

        # Elixhauser Comorbidity Index (Tailored for 30-Day Readmission)
        elixhauser_count = (
            has_chf.astype(float) + has_cad.astype(float) + has_htn_comp.astype(float) +
            has_pvd.astype(float) + has_stroke.astype(float) + has_copd.astype(float) +
            has_renal.astype(float) + has_sev_liver.astype(float) + has_neoplasm.astype(float) +
            has_met_cancer.astype(float) + has_fluid_elect.astype(float) + has_anemia.astype(float) +
            has_afib.astype(float) + has_valvular.astype(float) + has_pulm_circ.astype(float) +
            has_depression.astype(float) + has_substance.astype(float) + has_obesity.astype(float) +
            has_ulcer_skin.astype(float) + has_diab_comp
        )
        d['elixhauser_count'] = elixhauser_count

        # Charlson Comorbidity Index (CCI)
        age_pts = np.clip(np.floor((d['age_num'] - 40.0) / 10.0), 0.0, 4.0)
        charlson = (
            1.0 * (has_mi | has_chf | has_pvd | has_stroke | has_dementia | has_copd | has_rheum | has_pud | has_mild_liver | (has_diab_uncomp > 0.5)).astype(float)
            + 1.0 * has_diab_comp
            + 2.0 * (has_hemi | has_renal | has_neoplasm).astype(float)
            + 3.0 * has_sev_liver.astype(float)
            + 6.0 * (has_met_cancer | has_hiv).astype(float)
            + age_pts
        )
        d['charlson_score'] = charlson

        # Admission & Discharge Granularity
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf'] = (disp == '3').astype(float)
        d['disch_is_home_health'] = np.isin(disp, ['6', '8']).astype(float)
        d['disch_is_short_hosp'] = (disp == '2').astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5', '27', '29', '30']).astype(float)
        d['disch_is_rehab'] = np.isin(disp, ['22', '27']).astype(float)
        d['disch_is_ama'] = (disp == '7').astype(float)
        d['disch_is_psych'] = np.isin(disp, ['25', '28']).astype(float)
        d['disch_is_facility'] = np.isin(disp, ['2', '3', '4', '5', '6', '8', '22', '23', '24', '27', '28', '29', '30']).astype(float)

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_urgent_adm'] = (adm_type == '2').astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)
        d['is_acute_adm'] = np.isin(adm_type, ['1', '2', '7']).astype(float)

        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])
        d['adm_from_er'] = (adm_src == '7').astype(float)
        d['adm_from_referral'] = np.isin(adm_src, ['1', '2', '3']).astype(float)
        d['adm_from_transfer'] = np.isin(adm_src, ['4', '5', '6', '10', '17', '20']).astype(float)
        d['adm_from_snf'] = (adm_src == '5').astype(float)

        d['er_to_facility'] = (d['adm_from_er'] * d['disch_is_facility']).astype(float)
        d['er_to_snf'] = (d['adm_from_er'] * d['disch_is_snf']).astype(float)
        d['er_to_home_health'] = (d['adm_from_er'] * d['disch_is_home_health']).astype(float)
        d['transfer_in_and_out'] = (d['adm_from_transfer'] * d['disch_is_facility']).astype(float)
        d['elective_to_home'] = (d['is_elective_adm'] * d['disch_is_home']).astype(float)

        # Missingness Signals
        payer_str = np.array([safe_str(x) for x in d['payer_code']])
        spec_str = np.array([safe_str(x) for x in d['medical_specialty']])
        weight_str = np.array([safe_str(x) for x in d['weight']])
        d['payer_missing'] = np.isin(payer_str, ['__missing__', '?']).astype(float)
        d['spec_missing'] = np.isin(spec_str, ['__missing__', '?']).astype(float)
        d['weight_missing'] = np.isin(weight_str, ['__missing__', '?']).astype(float)

        # LACE & LACE+ Index
        stay = np.maximum(0.0, d['time_in_hospital'])
        l_score = np.zeros(n_rows, dtype=float)
        l_score += (stay == 1).astype(float) * 1.0
        l_score += (stay == 2).astype(float) * 2.0
        l_score += (stay == 3).astype(float) * 3.0
        l_score += ((stay >= 4) & (stay <= 6)).astype(float) * 4.0
        l_score += ((stay >= 7) & (stay <= 13)).astype(float) * 5.0
        l_score += (stay >= 14).astype(float) * 7.0

        a_score = d['is_acute_adm'] * 3.0
        c_score = np.where(charlson >= 4.0, 5.0, np.clip(charlson, 0.0, 3.0))
        e_score = np.clip(d['number_emergency'], 0.0, 4.0)

        lace = l_score + a_score + c_score + e_score
        d['lace_score'] = lace
        d['lace_plus'] = lace + 2.0 * np.log1p(d['number_inpatient']) + 0.5 * (d['age_num'] / 10.0) + 1.5 * d['disch_is_facility']

        # Stay Interval Indicators
        d['stay_1'] = (stay == 1).astype(float)
        d['stay_2_3'] = ((stay >= 2) & (stay <= 3)).astype(float)
        d['stay_4_7'] = ((stay >= 4) & (stay <= 7)).astype(float)
        d['stay_ge8'] = (stay >= 8).astype(float)

        # Utilization & Intensity Dynamics
        tot_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        acute_v = d['number_emergency'] + d['number_inpatient']
        d['total_visits'] = tot_v.astype(float)
        d['acute_visits'] = acute_v.astype(float)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)
        d['super_utilizer'] = (acute_v >= 3).astype(float)
        d['zero_prior_visits'] = (tot_v == 0).astype(float)

        d['inpatient_ge1'] = (d['number_inpatient'] >= 1).astype(float)
        d['inpatient_ge2'] = (d['number_inpatient'] >= 2).astype(float)
        d['inpatient_ge3'] = (d['number_inpatient'] >= 3).astype(float)
        d['emergency_ge1'] = (d['number_emergency'] >= 1).astype(float)
        d['emergency_ge2'] = (d['number_emergency'] >= 2).astype(float)
        d['outpatient_ge1'] = (d['number_outpatient'] >= 1).astype(float)
        d['no_prior_visits'] = (tot_v == 0).astype(float)
        d['prior_visit_diversity'] = (d['inpatient_ge1'] + d['emergency_ge1'] + d['outpatient_ge1'])

        d['inpatient_ratio'] = d['number_inpatient'] / (tot_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (tot_v + 1.0)
        d['outpatient_ratio'] = d['number_outpatient'] / (tot_v + 1.0)

        log_inp = np.log1p(np.maximum(0.0, d['number_inpatient']))
        log_er = np.log1p(np.maximum(0.0, d['number_emergency']))
        log_out = np.log1p(np.maximum(0.0, d['number_outpatient']))
        log_tot = np.log1p(np.maximum(0.0, tot_v))

        d['log_inpatient'] = log_inp
        d['log_emergency'] = log_er
        d['log_outpatient'] = log_out
        d['log_total_visits'] = log_tot
        d['inpatient_squared'] = (d['number_inpatient'] ** 2).astype(float)

        d['med_per_day'] = d['num_medications'] / (stay + 1.0)
        d['lab_per_day'] = d['num_lab_procedures'] / (stay + 1.0)
        d['proc_per_day'] = d['num_procedures'] / (stay + 1.0)
        d['diag_per_day'] = d['number_diagnoses'] / (stay + 1.0)
        d['clinical_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / (stay + 1.0)
        d['lab_per_med'] = d['num_lab_procedures'] / (d['num_medications'] + 1.0)
        d['lab_per_diag'] = d['num_lab_procedures'] / (d['number_diagnoses'] + 1.0)

        d['no_proc_long_stay'] = ((d['num_procedures'] == 0) & (stay >= 4)).astype(float)
        d['high_lab_no_proc'] = ((d['num_lab_procedures'] >= 50) & (d['num_procedures'] == 0)).astype(float)
        d['diag_maxed'] = (d['number_diagnoses'] >= 9).astype(float)

        d['high_util_long_stay'] = ((d['number_inpatient'] > 0) & (stay >= 7)).astype(float)
        d['short_stay_acute'] = ((stay <= 2) & (d['is_emergency_adm'] > 0.5)).astype(float)
        d['utilization_x_stay'] = log_inp * np.log1p(stay)
        d['utilization_x_facility'] = log_inp * d['disch_is_facility']
        d['utilization_x_charlson'] = log_inp * (charlson / 5.0)
        d['utilization_x_age'] = log_inp * (d['age_num'] / 50.0)

        # Elderly Subgroup Specific Indicators
        d['elderly_ge75'] = (d['age_num'] >= 75.0).astype(float)
        d['elderly_snf'] = d['elderly_ge75'] * d['disch_is_snf']
        d['elderly_home'] = d['elderly_ge75'] * d['disch_is_home']
        d['elderly_inpatient'] = d['elderly_ge75'] * log_inp
        d['elderly_polypharmacy'] = d['elderly_ge75'] * (d['num_medications'] >= 15).astype(float)

        # Medication Profile & Regimen Complexity
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        ups = np.zeros(n_rows, dtype=float)
        downs = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
                ups += (val == 'Up').astype(float)
                downs += (val == 'Down').astype(float)

        d['active_med_count'] = active
        d['med_change_count'] = changes
        d['med_up_count'] = ups
        d['med_down_count'] = downs

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['on_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['insulin_steady'] = (insulin == 'Steady').astype(float)

        met = d.get('metformin', np.array(['No'] * n_rows))
        d['on_metformin'] = (met != 'No').astype(float)

        sulf = np.zeros(n_rows, dtype=bool)
        for s_col in ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']:
            if s_col in d:
                sulf |= (d[s_col] != 'No')
        d['on_sulfonylurea'] = sulf.astype(float)

        tzd = np.zeros(n_rows, dtype=bool)
        for t_col in ['pioglitazone', 'rosiglitazone', 'troglitazone']:
            if t_col in d:
                tzd |= (d[t_col] != 'No')
        d['on_tzd'] = tzd.astype(float)

        other_oral = np.zeros(n_rows, dtype=bool)
        for o_col in ['repaglinide', 'nateglinide', 'acarbose', 'miglitol']:
            if o_col in d:
                other_oral |= (d[o_col] != 'No')
        d['on_other_oral'] = other_oral.astype(float)

        d['num_drug_classes'] = d['on_insulin'] + d['on_metformin'] + d['on_sulfonylurea'] + d['on_tzd'] + d['on_other_oral']
        d['insulin_plus_oral'] = ((d['on_insulin'] > 0.5) & ((d['on_metformin'] + d['on_sulfonylurea'] + d['on_tzd'] + d['on_other_oral']) > 0.5)).astype(float)
        d['metformin_ckd_warning'] = (d['on_metformin'] * has_ckd).astype(float)
        d['sulfonylurea_elderly_warning'] = (d['on_sulfonylurea'] * (d['age_num'] >= 65.0).astype(float)).astype(float)
        d['insulin_renal_warning'] = (d['on_insulin'] * (has_ckd | has_aki)).astype(float)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(float)
        d['high_med_intensity'] = ((d['num_medications'] / (stay + 1.0)) >= 4.0).astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        d['regimen_insulin_only'] = ((d['on_insulin'] > 0.5) & (d['num_drug_classes'] == 1)).astype(float)
        d['regimen_oral_only'] = ((d['on_insulin'] == 0) & (d['num_drug_classes'] >= 1)).astype(float)
        d['regimen_no_meds'] = (d['num_drug_classes'] == 0).astype(float)
        d['regimen_insulin_sulf'] = ((d['on_insulin'] > 0.5) & (d['on_sulfonylurea'] > 0.5)).astype(float)
        d['regimen_med_change_rate'] = d['med_change_count'] / (d['active_med_count'] + 1.0)

        # Biomarkers & Care Quality
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['a1c_normal'] = (a1c_res == 'Norm').astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)
        d['glu_tested'] = (glu_res != 'None').astype(float)
        d['therapeutic_inertia'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)
        d['a1c_high_med_changed'] = (d['a1c_high'] * d['change_flag']).astype(float)
        d['a1c_high_med_not_changed'] = (d['a1c_high'] * (1.0 - d['change_flag'])).astype(float)
        d['a1c_norm_med_steady'] = (d['a1c_normal'] * (1.0 - d['change_flag'])).astype(float)

        # Piette-Kerr Discordant Multimorbidity Architecture (Diabetes Care Framework)
        concordant_burden = (
            has_cad.astype(float) + has_chf.astype(float) + has_htn_comp.astype(float) +
            has_pvd.astype(float) + has_stroke.astype(float) + has_renal.astype(float) +
            has_diab_comp + has_diab_renal + has_diab_retino + has_diab_neuro + has_diab_vasc
        )
        discordant_burden = (
            has_copd.astype(float) + has_neoplasm.astype(float) + has_met_cancer.astype(float) +
            has_dementia.astype(float) + has_depression.astype(float) + has_substance.astype(float) +
            has_rheum.astype(float) + has_pud.astype(float) + has_sev_liver.astype(float) +
            has_ulcer_skin.astype(float) + has_sepsis.astype(float) + has_pneumonia.astype(float) +
            has_valvular.astype(float)
        )
        d['concordant_burden'] = concordant_burden
        d['discordant_burden'] = discordant_burden
        d['discordance_ratio'] = discordant_burden / (concordant_burden + 1.0)
        d['discordance_dominance'] = (discordant_burden > concordant_burden).astype(float)
        d['high_discordance_risk'] = ((discordant_burden >= 2.0) & (concordant_burden >= 1.0)).astype(float)

        # Pharmacoclinical Conflict & Adverse Cross-Organ Triggers
        d['copd_steroid_glycemic_risk'] = (has_copd * ((d['on_insulin'] > 0.5) | (d['a1c_high'] > 0.5) | (d['glu_high'] > 0.5))).astype(float)
        d['cardiorenal_diuretic_conflict'] = (has_chf * has_renal * (stay >= 3.0)).astype(float)
        d['hypoglycemia_lethal_triad'] = ((d['age_num'] >= 70.0) * ((d['on_insulin'] > 0.5) | (d['on_sulfonylurea'] > 0.5)) * has_renal).astype(float)
        d['metformin_contraindicated'] = (d['on_metformin'] * (has_renal | has_sepsis | has_sev_liver)).astype(float)
        d['tzd_chf_risk'] = (d['on_tzd'] * has_chf).astype(float)
        d['dementia_complex_med_risk'] = (has_dementia * ((d['on_insulin'] > 0.5) | (d['active_med_count'] >= 3.0))).astype(float)

        # Geriatric Frailty Dynamics & Nonagenarian Subgroup Calibration
        is_nonagenarian = (d['age_num'] >= 85.0).astype(float)
        d['is_nonagenarian'] = is_nonagenarian
        d['geriatric_frailty_score'] = (
            (d['age_num'] >= 75.0).astype(float) * 1.5 +
            has_dementia.astype(float) * 2.0 +
            has_ulcer_skin.astype(float) * 1.5 +
            has_fluid_elect.astype(float) * 1.0 +
            has_anemia.astype(float) * 1.0 +
            (d['num_medications'] >= 15).astype(float) * 1.0 +
            d['disch_is_facility'] * 1.5 +
            (d['number_inpatient'] >= 1).astype(float) * 2.0
        )
        d['geriatric_snf_readmit'] = is_nonagenarian * d['disch_is_snf']
        d['geriatric_inpatient_bounce'] = is_nonagenarian * (d['number_inpatient'] >= 1).astype(float)
        d['geriatric_polypharmacy_risk'] = is_nonagenarian * (d['num_medications'] >= 12).astype(float)
        d['geriatric_fall_injury'] = is_nonagenarian * d['diag1_is_injury']
        d['geriatric_low_mobility'] = is_nonagenarian * (has_stroke | has_hemi | has_dementia).astype(float)
        d['elderly_short_stay_home'] = ((d['age_num'] >= 80.0) & (stay <= 2) & (d['disch_is_home'] > 0.5)).astype(float)
        d['elderly_unstable_discharge'] = ((d['age_num'] >= 80.0) & (d['med_change_count'] > 0) & (d['disch_is_home'] > 0.5)).astype(float)
        d['elderly_acute_inpatient'] = ((d['age_num'] >= 80.0) & (d['number_inpatient'] >= 1)).astype(float)
        d['elderly_er_bounce'] = ((d['age_num'] >= 80.0) & (d['adm_from_er'] > 0.5) & (d['number_emergency'] >= 1)).astype(float)

        # Utilization Tier & Velocity Dynamics
        u_tier = np.zeros(n_rows, dtype=float)
        u_tier = np.where(d['number_outpatient'] > 0, 1.0, u_tier)
        u_tier = np.where(d['number_emergency'] > 0, 2.0, u_tier)
        u_tier = np.where(d['number_inpatient'] == 1, 3.0, u_tier)
        u_tier = np.where(d['number_inpatient'] >= 2, 4.0, u_tier)
        d['utilization_tier'] = u_tier

        d['acute_utilization_fraction'] = (d['number_emergency'] + d['number_inpatient']) / (tot_v + 1.0)
        d['inpatient_to_emergency_ratio'] = d['number_inpatient'] / (d['number_emergency'] + 1.0)
        d['elective_decompensation'] = (d['is_elective_adm'] * d['disch_is_snf']).astype(float)
        d['chronic_snf_loop'] = (d['adm_from_snf'] * d['disch_is_snf']).astype(float)
        d['rehab_transfer'] = (d['disch_is_rehab'] * (stay >= 5)).astype(float)
        d['acute_glycemic_stress'] = ((d['glu_high'] > 0.5) & (d['change_flag'] == 0.0)).astype(float)
        d['care_quality_deficit'] = ((d['a1c_tested'] == 0.0) & (d['is_emergency_adm'] > 0.5) & (stay >= 4)).astype(float)

        # Medication Regimen Entropy
        act_safe = np.maximum(d['active_med_count'], 1.0)
        p_up = d['med_up_count'] / act_safe
        p_down = d['med_down_count'] / act_safe
        p_steady = np.maximum(0.0, act_safe - d['med_up_count'] - d['med_down_count']) / act_safe
        d['med_regimen_entropy'] = -(
            p_up * np.log(p_up + 1e-6) +
            p_down * np.log(p_down + 1e-6) +
            p_steady * np.log(p_steady + 1e-6)
        )

        # Clinical Risk Interactions
        d['insulin_change_acute_adm'] = (((d['insulin_up'] > 0.5) | (d['insulin_down'] > 0.5)) * d['is_acute_adm']).astype(float)
        d['metformin_renal_risk'] = (d['on_metformin'] * (has_ckd | has_aki)).astype(float)
        d['glycemic_instability'] = (1.5 * d['a1c_high'] + 1.5 * d['glu_high'] + d['change_flag'] + d['on_insulin'] + 2.0 * d['therapeutic_inertia']).astype(float)
        d['unstable_home_discharge'] = (d['disch_is_home_health'] * d['is_acute_adm'] * (charlson >= 3.0).astype(float)).astype(float)

        # Pathway Interaction Strings
        d['diag1_disch'] = np.array([f"{g}_{p}" for g, p in zip(d['diag_1_grp'], disp)], dtype=object)
        d['adm_disch'] = np.array([f"{a}_{p}" for a, p in zip(adm_src, disp)], dtype=object)
        d['age_disch'] = np.array([f"{a}_{p}" for a, p in zip(d['age'], disp)], dtype=object)
        d['disch_spec'] = np.array([f"{p}_{s}" for p, s in zip(disp, spec_str)], dtype=object)
        d['spec_adm'] = np.array([f"{s}_{a}" for s, a in zip(spec_str, adm_type)], dtype=object)
        d['payer_disch'] = np.array([f"{p}_{v}" for p, v in zip(payer_str, disp)], dtype=object)
        d['diag1_age'] = np.array([f"{g}_{a}" for g, a in zip(d['diag_1_grp'], d['age'])], dtype=object)
        d['spec_diag1_grp'] = np.array([f"{s}_{g}" for s, g in zip(spec_str, d['diag_1_grp'])], dtype=object)
        d['adm_type_disch'] = np.array([f"{a}_{p}" for a, p in zip(adm_type, disp)], dtype=object)

    # 2. Out-of-Fold Bayesian Smoothed Target Encodings & Frequency Encodings
    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)

    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

    def compute_freq_map(arr):
        n = len(arr)
        counts = {}
        for x in arr:
            s = safe_str(x)
            counts[s] = counts.get(s, 0) + 1
        return {k: v / n for k, v in counts.items()}

    # Frequency mappings
    for f_name, col_key in [
        ('freq_diag_1', 'diag_1_3d'),
        ('freq_specialty', 'medical_specialty'),
        ('freq_discharge', 'discharge_disposition_id'),
        ('freq_adm_source', 'admission_source_id'),
        ('freq_payer', 'payer_code'),
    ]:
        f_map = compute_freq_map(train[col_key])
        train[f_name] = np.array([f_map.get(safe_str(x), 0.0) for x in train[col_key]], dtype=float)
        eval_cols[f_name] = np.array([f_map.get(safe_str(x), 0.0) for x in eval_cols[col_key]], dtype=float)

    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    te_features = [
        'te_diag_1', 'te_diag_2', 'te_diag_3', 'te_diag_1_grp', 'te_diag_max', 'te_diag_mean', 'te_diag_sum', 'te_diag_min',
        'te_discharge', 'te_specialty', 'te_adm_source', 'te_adm_type', 'te_payer',
        'te_diag1_disch', 'te_adm_disch', 'te_age_disch', 'te_disch_spec',
        'te_spec_adm', 'te_payer_disch', 'te_diag1_age', 'te_spec_diag1_grp', 'te_adm_type_disch'
    ]
    for tf in te_features:
        train[tf] = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf_te.split(train['gender'], y_train):
        y_tr_f = y_train[tr_idx]

        m_d1 = compute_smoothed_map(train['diag_1_3d'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_diag_1'][val_idx] = np.array([m_d1.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])

        m_g1 = compute_smoothed_map(train['diag_1_grp'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_diag_1_grp'][val_idx] = np.array([m_g1.get(safe_str(c), prior_glob) for c in train['diag_1_grp'][val_idx]])

        pooled_d = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_y = np.concatenate([y_tr_f, y_tr_f, y_tr_f])
        m_pool = compute_smoothed_map(pooled_d, pooled_y, smooth=25.0, prior=prior_glob)

        pd1 = np.array([m_pool.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        pd2 = np.array([m_pool.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        pd3 = np.array([m_pool.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        train['te_diag_2'][val_idx] = pd2
        train['te_diag_3'][val_idx] = pd3
        train['te_diag_max'][val_idx] = np.maximum(np.maximum(pd1, pd2), pd3)
        train['te_diag_min'][val_idx] = np.minimum(np.minimum(pd1, pd2), pd3)
        train['te_diag_mean'][val_idx] = (pd1 + pd2 + pd3) / 3.0
        train['te_diag_sum'][val_idx] = pd1 + pd2 + pd3

        m_disp = compute_smoothed_map(train['discharge_disposition_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_discharge'][val_idx] = np.array([m_disp.get(safe_str(c), prior_glob) for c in train['discharge_disposition_id'][val_idx]])

        m_spec = compute_smoothed_map(train['medical_specialty'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_specialty'][val_idx] = np.array([m_spec.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

        m_src = compute_smoothed_map(train['admission_source_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_adm_source'][val_idx] = np.array([m_src.get(safe_str(c), prior_glob) for c in train['admission_source_id'][val_idx]])

        m_adm = compute_smoothed_map(train['admission_type_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_adm_type'][val_idx] = np.array([m_adm.get(safe_str(c), prior_glob) for c in train['admission_type_id'][val_idx]])

        m_pay = compute_smoothed_map(train['payer_code'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_payer'][val_idx] = np.array([m_pay.get(safe_str(c), prior_glob) for c in train['payer_code'][val_idx]])

        m_d1_disch = compute_smoothed_map(train['diag1_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_diag1_disch'][val_idx] = np.array([m_d1_disch.get(safe_str(c), prior_glob) for c in train['diag1_disch'][val_idx]])

        m_adm_disch = compute_smoothed_map(train['adm_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_adm_disch'][val_idx] = np.array([m_adm_disch.get(safe_str(c), prior_glob) for c in train['adm_disch'][val_idx]])

        m_age_disch = compute_smoothed_map(train['age_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_age_disch'][val_idx] = np.array([m_age_disch.get(safe_str(c), prior_glob) for c in train['age_disch'][val_idx]])

        m_ds = compute_smoothed_map(train['disch_spec'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_disch_spec'][val_idx] = np.array([m_ds.get(safe_str(c), prior_glob) for c in train['disch_spec'][val_idx]])

        m_sa = compute_smoothed_map(train['spec_adm'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_spec_adm'][val_idx] = np.array([m_sa.get(safe_str(c), prior_glob) for c in train['spec_adm'][val_idx]])

        m_pd = compute_smoothed_map(train['payer_disch'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_payer_disch'][val_idx] = np.array([m_pd.get(safe_str(c), prior_glob) for c in train['payer_disch'][val_idx]])

        m_da = compute_smoothed_map(train['diag1_age'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_diag1_age'][val_idx] = np.array([m_da.get(safe_str(c), prior_glob) for c in train['diag1_age'][val_idx]])

        m_sd = compute_smoothed_map(train['spec_diag1_grp'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_spec_diag1_grp'][val_idx] = np.array([m_sd.get(safe_str(c), prior_glob) for c in train['spec_diag1_grp'][val_idx]])

        m_atd = compute_smoothed_map(train['adm_type_disch'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_adm_type_disch'][val_idx] = np.array([m_atd.get(safe_str(c), prior_glob) for c in train['adm_type_disch'][val_idx]])

    # Full train-to-eval target mappings
    m_d1_all = compute_smoothed_map(train['diag_1_3d'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_diag_1'] = np.array([m_d1_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']], dtype=float)

    m_g1_all = compute_smoothed_map(train['diag_1_grp'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_diag_1_grp'] = np.array([m_g1_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_grp']], dtype=float)

    pooled_d_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    m_pool_all = compute_smoothed_map(pooled_d_all, pooled_y_all, smooth=25.0, prior=prior_glob)
    epd1 = np.array([m_pool_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    epd2 = np.array([m_pool_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    epd3 = np.array([m_pool_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_2'] = epd2
    eval_cols['te_diag_3'] = epd3
    eval_cols['te_diag_max'] = np.maximum(np.maximum(epd1, epd2), epd3)
    eval_cols['te_diag_min'] = np.minimum(np.minimum(epd1, epd2), epd3)
    eval_cols['te_diag_mean'] = (epd1 + epd2 + epd3) / 3.0
    eval_cols['te_diag_sum'] = epd1 + epd2 + epd3

    m_disp_all = compute_smoothed_map(train['discharge_disposition_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_discharge'] = np.array([m_disp_all.get(safe_str(c), prior_glob) for c in eval_cols['discharge_disposition_id']], dtype=float)

    m_spec_all = compute_smoothed_map(train['medical_specialty'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([m_spec_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']], dtype=float)

    m_src_all = compute_smoothed_map(train['admission_source_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_source'] = np.array([m_src_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_source_id']], dtype=float)

    m_adm_all = compute_smoothed_map(train['admission_type_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_type'] = np.array([m_adm_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_type_id']], dtype=float)

    m_pay_all = compute_smoothed_map(train['payer_code'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_payer'] = np.array([m_pay_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_code']], dtype=float)

    m_d1_disch_all = compute_smoothed_map(train['diag1_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_diag1_disch'] = np.array([m_d1_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_disch']], dtype=float)

    m_adm_disch_all = compute_smoothed_map(train['adm_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_adm_disch'] = np.array([m_adm_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['adm_disch']], dtype=float)

    m_age_disch_all = compute_smoothed_map(train['age_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_age_disch'] = np.array([m_age_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['age_disch']], dtype=float)

    m_ds_all = compute_smoothed_map(train['disch_spec'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_disch_spec'] = np.array([m_ds_all.get(safe_str(c), prior_glob) for c in eval_cols['disch_spec']], dtype=float)

    m_sa_all = compute_smoothed_map(train['spec_adm'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_spec_adm'] = np.array([m_sa_all.get(safe_str(c), prior_glob) for c in eval_cols['spec_adm']], dtype=float)

    m_pd_all = compute_smoothed_map(train['payer_disch'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_payer_disch'] = np.array([m_pd_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_disch']], dtype=float)

    m_da_all = compute_smoothed_map(train['diag1_age'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_diag1_age'] = np.array([m_da_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_age']], dtype=float)

    m_sd_all = compute_smoothed_map(train['spec_diag1_grp'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_spec_diag1_grp'] = np.array([m_sd_all.get(safe_str(c), prior_glob) for c in eval_cols['spec_diag1_grp']], dtype=float)

    m_atd_all = compute_smoothed_map(train['adm_type_disch'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_adm_type_disch'] = np.array([m_atd_all.get(safe_str(c), prior_glob) for c in eval_cols['adm_type_disch']], dtype=float)

    # 2b. Multilinear 4-Way Risk-Synergy Tensor Decomposition via Higher-Order Orthogonal Iteration (HOOI)
    def extract_4way_modes(dataset):
        D = np.column_stack([
            dataset['has_chf'],
            ((dataset['has_cad'] > 0.5) | (dataset['has_pvd'] > 0.5) | (dataset['has_stroke'] > 0.5)).astype(float),
            ((dataset['has_ckd'] > 0.5) | (dataset['has_aki'] > 0.5)).astype(float),
            dataset['has_copd'],
            ((dataset['has_diab_comp'] > 0.5) | (dataset['has_diab_acute'] > 0.5)).astype(float),
            ((dataset['has_sepsis'] > 0.5) | (dataset['has_pneumonia'] > 0.5) | (dataset['has_uti'] > 0.5)).astype(float),
            dataset['has_fluid_elect'],
            dataset['has_anemia'],
            ((dataset['has_sev_liver'] > 0.5) | (dataset['has_neoplasm'] > 0.5)).astype(float),
            ((dataset['charlson_score'] >= 3.0) | (dataset['elixhauser_count'] >= 3.0)).astype(float),
        ])
        M = np.column_stack([
            dataset['on_insulin'],
            ((dataset['insulin_up'] > 0.5) | (dataset['insulin_down'] > 0.5)).astype(float),
            dataset['on_metformin'],
            dataset['on_sulfonylurea'],
            (dataset['num_drug_classes'] >= 2.0).astype(float),
            dataset['change_flag'],
            dataset['polypharmacy'],
            ((dataset['a1c_high'] > 0.5) | (dataset['glu_high'] > 0.5)).astype(float),
        ])
        U = np.column_stack([
            dataset['inpatient_ge1'],
            dataset['inpatient_ge2'],
            dataset['emergency_ge1'],
            dataset['emergency_ge2'],
            dataset['outpatient_ge1'],
            dataset['super_utilizer'],
            dataset['no_prior_visits'],
        ])
        V = np.column_stack([
            ((dataset['is_emergency_adm'] > 0.5) | (dataset['adm_from_er'] > 0.5)).astype(float),
            dataset['disch_is_snf'],
            dataset['disch_is_facility'],
            dataset['disch_is_home_health'],
            dataset['disch_is_home'],
            (dataset['time_in_hospital'] <= 2.0).astype(float),
            (dataset['time_in_hospital'] >= 7.0).astype(float),
            dataset['elderly_ge75'],
        ])
        return D, M, U, V

    D_tr_raw, M_tr_raw, U_tr_raw, V_tr_raw = extract_4way_modes(train)
    D_ev_raw, M_ev_raw, U_ev_raw, V_ev_raw = extract_4way_modes(eval_cols)

    def standardize_mode(tr_mat, ev_mat):
        mu = np.mean(tr_mat, axis=0)
        std = np.std(tr_mat, axis=0) + 1e-6
        return (tr_mat - mu) / std, (ev_mat - mu) / std

    D_tr, D_ev = standardize_mode(D_tr_raw, D_ev_raw)
    M_tr, M_ev = standardize_mode(M_tr_raw, M_ev_raw)
    U_tr, U_ev = standardize_mode(U_tr_raw, U_ev_raw)
    V_tr, V_ev = standardize_mode(V_tr_raw, V_ev_raw)

    y_dev = (y_train.astype(float) - prior_glob) / (np.std(y_train) + 1e-6)
    DM_tr = np.einsum('ni,nj->nij', D_tr, M_tr)
    UV_tr = np.einsum('nk,nl->nkl', U_tr, V_tr)
    yDM_tr = y_dev[:, None, None] * DM_tr
    T_risk = np.einsum('nij,nkl->ijkl', yDM_tr, UV_tr) / float(n_tr)

    I1, I2, I3, I4 = T_risk.shape
    R1, R2, R3, R4 = 4, 3, 3, 3

    # HOSVD Factor Matrix Initialization
    T_mode1 = T_risk.reshape(I1, -1)
    T_mode2 = np.transpose(T_risk, (1, 0, 2, 3)).reshape(I2, -1)
    T_mode3 = np.transpose(T_risk, (2, 0, 1, 3)).reshape(I3, -1)
    T_mode4 = np.transpose(T_risk, (3, 0, 1, 2)).reshape(I4, -1)

    u1, _, _ = np.linalg.svd(T_mode1, full_matrices=False)
    u2, _, _ = np.linalg.svd(T_mode2, full_matrices=False)
    u3, _, _ = np.linalg.svd(T_mode3, full_matrices=False)
    u4, _, _ = np.linalg.svd(T_mode4, full_matrices=False)

    V_D = u1[:, :R1]
    V_M = u2[:, :R2]
    V_U = u3[:, :R3]
    V_V = u4[:, :R4]

    # HOOI Alternating Least Squares Projections
    for _ in range(4):
        Y_D = np.einsum('ijkl,jq,kr,ls->iqrs', T_risk, V_M, V_U, V_V)
        u, _, _ = np.linalg.svd(Y_D.reshape(I1, -1), full_matrices=False)
        V_D = u[:, :R1]

        Y_M = np.einsum('ijkl,ip,kr,ls->jqps', T_risk, V_D, V_U, V_V)
        u, _, _ = np.linalg.svd(Y_M.reshape(I2, -1), full_matrices=False)
        V_M = u[:, :R2]

        Y_U = np.einsum('ijkl,ip,jq,ls->krpq', T_risk, V_D, V_M, V_V)
        u, _, _ = np.linalg.svd(Y_U.reshape(I3, -1), full_matrices=False)
        V_U = u[:, :R3]

        Y_V = np.einsum('ijkl,ip,jq,kr->lspq', T_risk, V_D, V_M, V_U)
        u, _, _ = np.linalg.svd(Y_V.reshape(I4, -1), full_matrices=False)
        V_V = u[:, :R4]

    # Core Interaction Tensor
    G = np.einsum('ijkl,ip,jq,kr,ls->pqrs', T_risk, V_D, V_M, V_U, V_V)
    G_mat = G.reshape(R1 * R2, R3 * R4)

    for d_set, (d_m, m_m, u_m, v_m) in [(train, (D_tr, M_tr, U_tr, V_tr)), (eval_cols, (D_ev, M_ev, U_ev, V_ev))]:
        pD = d_m @ V_D
        pM = m_m @ V_M
        pU = u_m @ V_U
        pV = v_m @ V_V

        for r in range(R1):
            d_set[f'hooi_d_{r}'] = pD[:, r]
        for r in range(R2):
            d_set[f'hooi_m_{r}'] = pM[:, r]
        for r in range(R3):
            d_set[f'hooi_u_{r}'] = pU[:, r]
            d_set[f'hooi_v_{r}'] = pV[:, r]

        pDM = np.einsum('np,nq->npq', pD, pM).reshape(-1, R1 * R2)
        pUV = np.einsum('nr,ns->nrs', pU, pV).reshape(-1, R3 * R4)
        d_set['hooi_4way_synergy'] = np.sum((pDM @ G_mat) * pUV, axis=1)

        d_set['hooi_dm_risk'] = np.einsum('pq,np,nq->n', G[:, :, 0, 0], pD, pM)
        d_set['hooi_du_risk'] = np.einsum('pr,np,nr->n', G[:, 0, :, 0], pD, pU)
        d_set['hooi_uv_risk'] = np.einsum('rs,nr,ns->n', G[0, 0, :, :], pU, pV)
        d_set['hooi_mv_risk'] = np.einsum('qs,nq,ns->n', G[0, :, 0, :], pM, pV)
        d_set['hooi_dv_risk'] = np.einsum('ps,np,ns->n', G[:, 0, 0, :], pD, pV)

        ctx_DMU = np.einsum('pqrs,nq,nr,ns->np', G, pM, pU, pV)
        d_set['hooi_comorbidity_context_intensity'] = np.sum(pD * ctx_DMU, axis=1)

        ctx_M = np.einsum('pqrs,np,nr,ns->nq', G, pD, pU, pV)
        d_set['hooi_medication_context_intensity'] = np.sum(pM * ctx_M, axis=1)

        ctx_V = np.einsum('pqrs,np,nq,nr->ns', G, pD, pM, pU)
        d_set['hooi_acuity_context_intensity'] = np.sum(pV * ctx_V, axis=1)

    hooi_features = (
        [f'hooi_d_{r}' for r in range(R1)] +
        [f'hooi_m_{r}' for r in range(R2)] +
        [f'hooi_u_{r}' for r in range(R3)] +
        [f'hooi_v_{r}' for r in range(R4)] +
        ['hooi_4way_synergy', 'hooi_dm_risk', 'hooi_du_risk', 'hooi_uv_risk', 'hooi_mv_risk', 'hooi_dv_risk',
         'hooi_comorbidity_context_intensity', 'hooi_medication_context_intensity', 'hooi_acuity_context_intensity']
    )

    # 2c. Spectral Disease Co-occurrence Manifold (Pheno-SVD)
    diag_counts = {}
    for col in ['diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
        for c in train[col]:
            if c != 'missing':
                diag_counts[c] = diag_counts.get(c, 0) + 1
    top_codes = [c for c, cnt in sorted(diag_counts.items(), key=lambda x: x[1], reverse=True)[:100]]
    code_to_idx = {c: i for i, c in enumerate(top_codes)}
    k_codes = len(top_codes)
    n_ev = len(eval_cols['gender'])

    def build_incidence_matrix(dataset, n_r):
        B = np.zeros((n_r, k_codes), dtype=float)
        rows = np.arange(n_r)
        for col in ['diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
            idxs = np.array([code_to_idx.get(c, -1) for c in dataset[col]], dtype=int)
            valid = idxs >= 0
            B[rows[valid], idxs[valid]] += 1.0
        return B

    B_tr = build_incidence_matrix(train, n_tr)
    B_ev = build_incidence_matrix(eval_cols, n_ev)

    C_co = B_tr.T @ B_tr
    diag_f = np.diag(C_co) + 1.0
    expected = np.outer(diag_f, diag_f) / float(n_tr)
    ratio = (C_co + 1.0) / (expected + 1e-5)
    ppmi = np.maximum(0.0, np.log(ratio))

    U_p, _, _ = np.linalg.svd(ppmi, full_matrices=False)
    n_pdims = min(6, U_p.shape[1])
    W_codes = U_p[:, :n_pdims]

    P_tr = B_tr @ W_codes
    P_ev = B_ev @ W_codes

    for k in range(n_pdims):
        train[f'pheno_dim_{k}'] = P_tr[:, k]
        eval_cols[f'pheno_dim_{k}'] = P_ev[:, k]

    r_excess = B_tr.T @ (y_train.astype(float) - prior_glob)
    u_risk = W_codes.T @ r_excess
    u_risk_norm = u_risk / (np.linalg.norm(u_risk) + 1e-8)

    train['pheno_risk_latent'] = P_tr @ u_risk_norm
    eval_cols['pheno_risk_latent'] = P_ev @ u_risk_norm
    train['pheno_risk_norm'] = np.linalg.norm(P_tr, axis=1)
    eval_cols['pheno_risk_norm'] = np.linalg.norm(P_ev, axis=1)

    def get_code_embeddings(d_col, n_r):
        idxs = np.array([code_to_idx.get(c, -1) for c in d_col], dtype=int)
        emb = np.zeros((n_r, n_pdims), dtype=float)
        valid = idxs >= 0
        emb[valid] = W_codes[idxs[valid]]
        return emb, valid

    for d_set, n_rows_set in [(train, n_tr), (eval_cols, n_ev)]:
        e1, v1 = get_code_embeddings(d_set['diag_1_3d'], n_rows_set)
        e2, v2 = get_code_embeddings(d_set['diag_2_3d'], n_rows_set)
        e3, v3 = get_code_embeddings(d_set['diag_3_3d'], n_rows_set)

        d12 = np.where(v1 & v2, np.linalg.norm(e1 - e2, axis=1), 0.0)
        d13 = np.where(v1 & v3, np.linalg.norm(e1 - e3, axis=1), 0.0)
        d23 = np.where(v2 & v3, np.linalg.norm(e2 - e3, axis=1), 0.0)

        d_set['pheno_dispersion_12'] = d12
        d_set['pheno_dispersion_13'] = d13
        d_set['pheno_dispersion_max'] = np.maximum(np.maximum(d12, d13), d23)
        cnt = (v1 & v2).astype(float) + (v1 & v3).astype(float) + (v2 & v3).astype(float)
        d_set['pheno_dispersion_mean'] = np.where(cnt > 0, (d12 + d13 + d23) / np.maximum(cnt, 1.0), 0.0)

    pheno_features = [f'pheno_dim_{k}' for k in range(n_pdims)] + [
        'pheno_risk_latent', 'pheno_risk_norm', 'pheno_dispersion_12',
        'pheno_dispersion_13', 'pheno_dispersion_max', 'pheno_dispersion_mean'
    ]

    extra_num = hooi_features + pheno_features + [
        'age_num', 'total_visits', 'acute_visits', 'has_prior_inpatient', 'frequent_inpatient', 'super_utilizer', 'zero_prior_visits',
        'inpatient_ge1', 'inpatient_ge2', 'inpatient_ge3', 'emergency_ge1', 'emergency_ge2', 'outpatient_ge1', 'no_prior_visits', 'prior_visit_diversity',
        'inpatient_ratio', 'emergency_ratio', 'outpatient_ratio',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits', 'inpatient_squared',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'diag_per_day', 'clinical_intensity',
        'lab_per_med', 'lab_per_diag', 'high_util_long_stay', 'short_stay_acute',
        'no_proc_long_stay', 'high_lab_no_proc', 'diag_maxed',
        'stay_1', 'stay_2_3', 'stay_4_7', 'stay_ge8',
        'utilization_x_stay', 'utilization_x_facility', 'utilization_x_charlson', 'utilization_x_age',
        'active_med_count', 'med_change_count', 'med_up_count', 'med_down_count',
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady',
        'on_metformin', 'on_sulfonylurea', 'on_tzd', 'on_other_oral',
        'num_drug_classes', 'insulin_plus_oral', 'metformin_ckd_warning', 'sulfonylurea_elderly_warning', 'insulin_renal_warning', 'polypharmacy', 'high_med_intensity',
        'change_flag', 'diabetes_med_flag',
        'regimen_insulin_only', 'regimen_oral_only', 'regimen_no_meds', 'regimen_insulin_sulf', 'regimen_med_change_rate',
        'a1c_tested', 'a1c_high', 'a1c_normal', 'glu_high', 'glu_tested', 'therapeutic_inertia',
        'a1c_high_med_changed', 'a1c_high_med_not_changed', 'a1c_norm_med_steady',
        'disch_is_home', 'disch_is_snf', 'disch_is_home_health', 'disch_is_short_hosp', 'disch_is_facility',
        'disch_is_transfer', 'disch_is_rehab', 'disch_is_ama', 'disch_is_psych',
        'is_emergency_adm', 'is_urgent_adm', 'is_elective_adm', 'is_acute_adm',
        'adm_from_er', 'adm_from_referral', 'adm_from_transfer', 'adm_from_snf',
        'er_to_facility', 'transfer_in_and_out', 'elective_to_home', 'er_to_snf', 'er_to_home_health',
        'payer_missing', 'spec_missing', 'weight_missing',
        'elderly_ge75', 'elderly_snf', 'elderly_home', 'elderly_inpatient', 'elderly_polypharmacy',
        'diag1_is_chf', 'diag1_is_circ', 'diag1_is_resp', 'diag1_is_diab', 'diag1_is_digestive', 'diag1_is_infection', 'diag1_is_injury',
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_htn_comp', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_uti', 'has_sepsis', 'has_anemia', 'has_afib', 'has_ulcer_skin',
        'has_valvular', 'has_pulm_circ', 'has_depression', 'has_substance', 'has_obesity',
        'is_diab_type1', 'is_diab_uncontrolled', 'has_diab_uncomp', 'has_diab_comp', 'has_diab_acute', 'has_diab_renal', 'has_diab_retino', 'has_diab_neuro', 'has_diab_vasc', 'num_diab_organs',
        'cardiorenal_risk', 'cardiorenal_anemia', 'cardiorespiratory', 'metabolic_sepsis', 'vascular_burden',
        'charlson_score', 'elixhauser_count', 'lace_score', 'lace_plus',
        'concordant_burden', 'discordant_burden', 'discordance_ratio', 'discordance_dominance', 'high_discordance_risk',
        'copd_steroid_glycemic_risk', 'cardiorenal_diuretic_conflict', 'hypoglycemia_lethal_triad',
        'metformin_contraindicated', 'tzd_chf_risk', 'dementia_complex_med_risk',
        'is_nonagenarian', 'geriatric_frailty_score', 'geriatric_snf_readmit', 'geriatric_inpatient_bounce',
        'geriatric_polypharmacy_risk', 'geriatric_fall_injury', 'geriatric_low_mobility',
        'elderly_short_stay_home', 'elderly_unstable_discharge', 'elderly_acute_inpatient', 'elderly_er_bounce',
        'utilization_tier', 'acute_utilization_fraction', 'inpatient_to_emergency_ratio',
        'elective_decompensation', 'chronic_snf_loop', 'rehab_transfer',
        'acute_glycemic_stress', 'care_quality_deficit', 'med_regimen_entropy',
        'freq_diag_1', 'freq_specialty', 'freq_discharge', 'freq_adm_source', 'freq_payer',
        'insulin_change_acute_adm', 'metformin_renal_risk', 'glycemic_instability', 'unstable_home_discharge'
    ] + te_features

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag1_disch', 'adm_disch', 'age_disch', 'disch_spec',
        'spec_adm', 'payer_disch', 'diag1_age', 'spec_diag1_grp', 'adm_type_disch'
    ]

    # 3. Encoding Matrix Assembly
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 3b. Non-Parametric Manifold Patient Cohesion (NMPC) Feature Representation
    knn_feature_names = (
        [f'hooi_d_{r}' for r in range(R1)] +
        [f'hooi_m_{r}' for r in range(R2)] +
        [f'hooi_u_{r}' for r in range(R3)] +
        [f'hooi_v_{r}' for r in range(R4)] +
        ['hooi_4way_synergy', 'hooi_dm_risk', 'hooi_uv_risk'] +
        [f'pheno_dim_{k}' for k in range(n_pdims)] +
        ['pheno_risk_latent', 'lace_plus', 'charlson_score', 'log_inpatient']
    )
    Z_tr_list, Z_ev_list = [], []
    for f in knn_feature_names:
        v_tr = train[f]
        v_ev = eval_cols[f]
        m = np.mean(v_tr)
        s = np.std(v_tr) + 1e-8
        Z_tr_list.append((v_tr - m) / s)
        Z_ev_list.append((v_ev - m) / s)
    Z_tr = np.column_stack(Z_tr_list)
    Z_ev = np.column_stack(Z_ev_list)

    # 4. Quad-Engine GBDT Models across 4 Stratified Folds
    skf_models = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    oof_lgb1 = np.zeros(n_tr, dtype=float)
    oof_lgb2 = np.zeros(n_tr, dtype=float)
    oof_hgb = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_logits_lgb1 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_lgb2 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_hgb = np.zeros(n_eval_rows, dtype=float)
    eval_logits_xgb = np.zeros(n_eval_rows, dtype=float)

    def to_logit(p, eps=1e-6):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        # Fit NearestNeighbors on training fold to build a patient similarity manifold
        from sklearn.neighbors import NearestNeighbors
        Z_tr_fold = Z_tr[tr_idx]
        Z_va_fold = Z_tr[val_idx]
        
        knn = NearestNeighbors(n_neighbors=80, metric='minkowski', p=2, n_jobs=1)
        knn.fit(Z_tr_fold)

        dist_tr, idx_tr = knn.kneighbors(Z_tr_fold, n_neighbors=81)
        dist_tr, idx_tr = dist_tr[:, 1:], idx_tr[:, 1:]

        dist_va, idx_va = knn.kneighbors(Z_va_fold, n_neighbors=80)
        dist_ev, idx_ev = knn.kneighbors(Z_ev, n_neighbors=80)

        def compute_nmpc(dists, indices, y_source):
            features = []
            for K in [12, 35, 80]:
                d_k = dists[:, :K]
                idx_k = indices[:, :K]
                y_k = y_source[idx_k]
                
                w = 1.0 / (d_k + 1e-4)
                w_sum = np.sum(w, axis=1, keepdims=True)
                w = w / w_sum
                
                risk = np.sum(w * y_k, axis=1)
                features.append(risk)
                
                risk_std = np.sqrt(np.sum(w * (y_k - risk[:, None])**2, axis=1))
                features.append(risk_std)
                
                mean_dist = np.mean(d_k, axis=1)
                features.append(mean_dist)
                
            return np.column_stack(features)

        y_tr_source = y_train[tr_idx]
        nmpc_tr = compute_nmpc(dist_tr, idx_tr, y_tr_source)
        nmpc_va = compute_nmpc(dist_va, idx_va, y_tr_source)
        nmpc_ev = compute_nmpc(dist_ev, idx_ev, y_tr_source)

        x_tr = np.column_stack([x_train[tr_idx], nmpc_tr])
        x_va = np.column_stack([x_train[val_idx], nmpc_va])
        x_ev = np.column_stack([x_eval, nmpc_ev])

        y_tr = y_train[tr_idx]

        # Model 1: Standard Leaf-Wise LightGBM
        m_lgb1 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_split_gain=0.01,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_lgb1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb1[val_idx] = m_lgb1.predict_proba(x_va)[:, 1]
        eval_logits_lgb1 += to_logit(m_lgb1.predict_proba(x_ev)[:, 1]) / 4.0

        # Model 2: Depth-Constrained Conservative LightGBM
        m_lgb2 = lgb.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.038,
            max_depth=5,
            num_leaves=22,
            min_child_samples=70,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=2.5,
            reg_lambda=10.0,
            min_split_gain=0.02,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_lgb2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb2[val_idx] = m_lgb2.predict_proba(x_va)[:, 1]
        eval_logits_lgb2 += to_logit(m_lgb2.predict_proba(x_ev)[:, 1]) / 4.0

        # Model 3: HistGradientBoostingClassifier (Exact Histogram Binning)
        m_hgb = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.045,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=5.0,
            categorical_features=encoder.categorical_indices,
            random_state=42 + fold,
        )
        m_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = m_hgb.predict_proba(x_va)[:, 1]
        eval_logits_hgb += to_logit(m_hgb.predict_proba(x_ev)[:, 1]) / 4.0

        # Model 4: XGBoost with histogram tree method
        m_xgb = xgb.XGBClassifier(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=25,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        m_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(x_va)[:, 1]
        eval_logits_xgb += to_logit(m_xgb.predict_proba(x_ev)[:, 1]) / 4.0

    # 5. Dual-Space Simplex Lattice Optimization & Monotonic Logistic Calibration
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression

    r_lgb1 = rankdata(oof_lgb1) / (n_tr + 1.0)
    r_lgb2 = rankdata(oof_lgb2) / (n_tr + 1.0)
    r_hgb = rankdata(oof_hgb) / (n_tr + 1.0)
    r_xgb = rankdata(oof_xgb) / (n_tr + 1.0)
    r_mat = np.column_stack([r_lgb1, r_lgb2, r_hgb, r_xgb])

    z_oofs = [to_logit(oof_lgb1), to_logit(oof_lgb2), to_logit(oof_hgb), to_logit(oof_xgb)]
    z_mat = np.column_stack(z_oofs)
    z_means = np.mean(z_mat, axis=0)
    z_stds = np.std(z_mat, axis=0) + 1e-8
    z_norm_mat = (z_mat - z_means) / z_stds

    r_ev_lgb1 = rankdata(eval_logits_lgb1) / (n_eval_rows + 1.0)
    r_ev_lgb2 = rankdata(eval_logits_lgb2) / (n_eval_rows + 1.0)
    r_ev_hgb = rankdata(eval_logits_hgb) / (n_eval_rows + 1.0)
    r_ev_xgb = rankdata(eval_logits_xgb) / (n_eval_rows + 1.0)
    r_ev_mat = np.column_stack([r_ev_lgb1, r_ev_lgb2, r_ev_hgb, r_ev_xgb])

    eval_z_mat = np.column_stack([eval_logits_lgb1, eval_logits_lgb2, eval_logits_hgb, eval_logits_xgb])
    eval_z_norm_mat = (eval_z_mat - z_means) / z_stds

    # Global uniform lattice search over 4-simplex
    lattice_weights = []
    for k1 in range(11):
        for k2 in range(11 - k1):
            for k3 in range(11 - k1 - k2):
                k4 = 10 - k1 - k2 - k3
                lattice_weights.append(np.array([k1, k2, k3, k4], dtype=float) / 10.0)

    best_score = -1.0
    best_space = 'rank'
    best_w = np.full(4, 0.25)

    for w in lattice_weights:
        s_r = r_mat @ w
        auc_r = roc_auc_score(y_train, s_r)
        if auc_r > best_score:
            best_score = auc_r
            best_space = 'rank'
            best_w = w

        s_z = z_norm_mat @ w
        auc_z = roc_auc_score(y_train, s_z)
        if auc_z > best_score:
            best_score = auc_z
            best_space = 'logit'
            best_w = w

    # Local coordinate refinement around best lattice point
    refined_w = best_w.copy()
    step = 0.025
    active_mat = r_mat if best_space == 'rank' else z_norm_mat

    for _ in range(2):
        for i in range(4):
            for j in range(4):
                if i != j and refined_w[j] >= step:
                    cand_w = refined_w.copy()
                    cand_w[i] += step
                    cand_w[j] -= step
                    auc_cand = roc_auc_score(y_train, active_mat @ cand_w)
                    if auc_cand > best_score:
                        best_score = auc_cand
                        refined_w = cand_w

    if best_space == 'rank':
        oof_final_score = r_mat @ refined_w
        eval_final_score = r_ev_mat @ refined_w
    else:
        oof_final_score = z_norm_mat @ refined_w
        eval_final_score = eval_z_norm_mat @ refined_w

    calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200)
    calibrator.fit(oof_final_score.reshape(-1, 1), y_train)
    pred = calibrator.predict_proba(eval_final_score.reshape(-1, 1))[:, 1]
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
