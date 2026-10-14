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

        has_htn_renal = has_htn_comp & has_renal
        has_infection = has_sepsis | has_pneumonia | has_uti | has_ulcer_skin

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

        has_diab_hypo = np.zeros(n_rows, dtype=float)
        for col in ['diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
            has_diab_hypo = np.maximum(has_diab_hypo, np.isin(d[col], ['251']).astype(float))

        d['has_diab_hypo'] = has_diab_hypo
        d['has_microvascular'] = ((has_diab_renal > 0.5) & (has_diab_retino > 0.5)).astype(float)
        d['has_foot_ulcer'] = (has_ulcer_skin & (has_diab_comp > 0.5)).astype(float)
        d['has_htn_renal'] = has_htn_renal.astype(float)
        d['has_infection'] = has_infection.astype(float)

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
        d['diag1_is_ckd'] = (c1 == '585').astype(float)
        d['diag1_is_copd'] = np.isin(c1, ['491', '492', '493', '494', '496']).astype(float)
        d['diag1_is_pneumonia'] = np.isin(c1, ['480', '481', '482', '483', '484', '485', '486']).astype(float)
        d['diag1_is_fluid_elect'] = (c1 == '276').astype(float)
        d['diag1_is_stroke'] = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']).astype(float)
        d['diag1_is_cad'] = np.isin(c1, ['410', '411', '412', '413', '414']).astype(float)

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
        d['is_trauma_adm'] = (adm_type == '7').astype(float)

        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])
        d['adm_from_er'] = (adm_src == '7').astype(float)
        d['adm_from_referral'] = np.isin(adm_src, ['1', '2', '3']).astype(float)
        d['adm_from_transfer'] = np.isin(adm_src, ['4', '5', '6', '10', '17', '20']).astype(float)
        d['adm_from_snf'] = (adm_src == '5').astype(float)
        d['adm_from_physician'] = (adm_src == '1').astype(float)
        d['adm_from_clinic'] = (adm_src == '2').astype(float)

        d['er_to_facility'] = (d['adm_from_er'] * d['disch_is_facility']).astype(float)
        d['er_to_snf'] = (d['adm_from_er'] * d['disch_is_snf']).astype(float)
        d['er_to_home_health'] = (d['adm_from_er'] * d['disch_is_home_health']).astype(float)
        d['transfer_in_and_out'] = (d['adm_from_transfer'] * d['disch_is_facility']).astype(float)
        d['elective_to_home'] = (d['is_elective_adm'] * d['disch_is_home']).astype(float)
        d['sick_disch_home'] = (d['disch_is_home'] * (charlson >= 4.0)).astype(float)
        d['home_health_cardiorenal'] = (d['disch_is_home_health'] * d['cardiorenal_risk']).astype(float)
        d['disch_ama_emergency'] = (d['disch_is_ama'] * d['is_emergency_adm']).astype(float)

        # Missingness Signals & Demographics
        payer_str = np.array([safe_str(x) for x in d['payer_code']])
        spec_str = np.array([safe_str(x) for x in d['medical_specialty']])
        weight_str = np.array([safe_str(x) for x in d['weight']])
        race_str = np.array([safe_str(x) for x in d['race']])
        d['payer_missing'] = np.isin(payer_str, ['__missing__', '?']).astype(float)
        d['spec_missing'] = np.isin(spec_str, ['__missing__', '?']).astype(float)
        d['weight_missing'] = np.isin(weight_str, ['__missing__', '?']).astype(float)
        d['race_missing'] = np.isin(race_str, ['__missing__', '?']).astype(float)

        # Granular Payer & Specialty Groupings
        d['payer_medicare'] = (payer_str == 'MC').astype(float)
        d['payer_medicaid'] = (payer_str == 'MD').astype(float)
        d['payer_commercial'] = np.isin(payer_str, ['BC', 'HM', 'CP', 'OG', 'CI']).astype(float)
        d['payer_selfpay'] = np.isin(payer_str, ['SP', 'UN']).astype(float)

        d['spec_cardio'] = np.isin(spec_str, ['Cardiology', 'Cardiology-Pediatric', 'Surgeon-Cardiovascular', 'Surgery-Cardiovascular/Thoracic']).astype(float)
        d['spec_internal'] = (spec_str == 'InternalMedicine').astype(float)
        d['spec_family'] = np.isin(spec_str, ['Family/GeneralPractice', 'GeneralPractice']).astype(float)
        d['spec_emergency'] = (spec_str == 'Emergency/Trauma').astype(float)
        d['spec_nephro'] = (spec_str == 'Nephrology').astype(float)
        d['spec_surgery'] = np.array(['Surgery' in s or 'Surgeon' in s for s in spec_str], dtype=float)
        d['spec_pulm'] = (spec_str == 'Pulmonology').astype(float)
        d['spec_psych'] = np.isin(spec_str, ['Psychiatry', 'Psychiatry-Child/Adolescent', 'Psychology']).astype(float)

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
        d['inpatient_ge4'] = (d['number_inpatient'] >= 4).astype(float)
        d['inpatient_ge5'] = (d['number_inpatient'] >= 5).astype(float)
        d['emergency_ge1'] = (d['number_emergency'] >= 1).astype(float)
        d['emergency_ge2'] = (d['number_emergency'] >= 2).astype(float)
        d['emergency_ge3'] = (d['number_emergency'] >= 3).astype(float)
        d['outpatient_ge1'] = (d['number_outpatient'] >= 1).astype(float)
        d['no_prior_visits'] = (tot_v == 0).astype(float)
        d['prior_visit_diversity'] = (d['inpatient_ge1'] + d['emergency_ge1'] + d['outpatient_ge1'])

        d['inpatient_ratio'] = d['number_inpatient'] / (tot_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (tot_v + 1.0)
        d['outpatient_ratio'] = d['number_outpatient'] / (tot_v + 1.0)
        d['acute_visit_ratio'] = acute_v / (tot_v + 1.0)

        weighted_util = 3.5 * d['number_inpatient'] + 1.5 * d['number_emergency'] + 0.5 * d['number_outpatient']
        d['weighted_prior_util'] = weighted_util
        d['log_weighted_prior_util'] = np.log1p(np.maximum(0.0, weighted_util))

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
        d['total_interventions'] = (d['num_lab_procedures'] + d['num_procedures'] + d['num_medications']).astype(float)
        d['interventions_per_day'] = d['total_interventions'] / (stay + 1.0)
        d['diag_workup_ratio'] = d['num_lab_procedures'] / (d['num_procedures'] + 1.0)
        d['log_lab_procedures'] = np.log1p(d['num_lab_procedures'])
        d['log_num_medications'] = np.log1p(d['num_medications'])
        d['log_time_in_hospital'] = np.log1p(stay)

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
        d['emergent_unconnected'] = ((d['is_emergency_adm'] > 0.5) & (d['number_outpatient'] == 0)).astype(float)
        d['ed_reliant'] = ((d['number_emergency'] >= 2) & (d['number_inpatient'] == 0)).astype(float)
        d['complex_utilizer'] = ((charlson >= 3.0) & (acute_v >= 2)).astype(float)

        # Elderly Subgroup & Age-Targeted Specific Indicators
        d['is_age_90plus'] = (d['age'] == '[90-100)').astype(float)
        d['is_age_80plus'] = np.isin(d['age'], ['[80-90)', '[90-100)']).astype(float)
        d['is_age_70plus'] = np.isin(d['age'], ['[70-80)', '[80-90)', '[90-100)']).astype(float)
        d['is_age_under40'] = np.isin(d['age'], ['[0-10)', '[10-20)', '[20-30)', '[30-40)']).astype(float)
        d['age_squared'] = ((d['age_num'] / 10.0) ** 2).astype(float)
        d['age_x_inpatient'] = (d['age_num'] / 10.0) * log_inp
        d['age_x_charlson'] = (d['age_num'] / 10.0) * charlson
        d['age_x_stay'] = (d['age_num'] / 10.0) * np.log1p(stay)

        d['elderly_ge75'] = (d['age_num'] >= 75.0).astype(float)
        d['elderly_snf'] = d['elderly_ge75'] * d['disch_is_snf']
        d['elderly_home'] = d['elderly_ge75'] * d['disch_is_home']
        d['elderly_inpatient'] = d['elderly_ge75'] * log_inp
        d['elderly_polypharmacy'] = d['elderly_ge75'] * (d['num_medications'] >= 15).astype(float)

        d['age_gt75_snf'] = d['elderly_ge75'] * d['disch_is_snf']
        d['age_gt75_home'] = d['elderly_ge75'] * d['disch_is_home']
        d['age_gt75_dementia'] = d['elderly_ge75'] * has_dementia.astype(float)
        d['age_gt75_chf'] = d['elderly_ge75'] * has_chf.astype(float)
        d['age_gt75_renal'] = d['elderly_ge75'] * has_renal.astype(float)
        d['age_gt75_util'] = d['elderly_ge75'] * log_inp

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
        d['insulin_unstable'] = np.isin(insulin, ['Up', 'Down']).astype(float)

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
        d['sulfonylurea_renal_warning'] = (d['on_sulfonylurea'] * has_renal).astype(float)
        d['insulin_renal_warning'] = (d['on_insulin'] * (has_ckd | has_aki)).astype(float)
        d['insulin_and_sulf'] = ((d['on_insulin'] > 0.5) & (d['on_sulfonylurea'] > 0.5)).astype(float)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(float)
        d['high_med_intensity'] = ((d['num_medications'] / (stay + 1.0)) >= 4.0).astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        d['regimen_insulin_only'] = ((d['on_insulin'] > 0.5) & (d['num_drug_classes'] == 1)).astype(float)
        d['regimen_oral_only'] = ((d['on_insulin'] == 0) & (d['num_drug_classes'] >= 1)).astype(float)
        d['regimen_no_meds'] = (d['num_drug_classes'] == 0).astype(float)
        d['regimen_insulin_sulf'] = ((d['on_insulin'] > 0.5) & (d['on_sulfonylurea'] > 0.5)).astype(float)
        d['regimen_med_change_rate'] = d['med_change_count'] / (d['active_med_count'] + 1.0)
        d['med_up_ratio'] = d['med_up_count'] / (d['active_med_count'] + 1.0)
        d['med_down_ratio'] = d['med_down_count'] / (d['active_med_count'] + 1.0)

        # Biomarkers & Care Quality
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_untested'] = (a1c_res == 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['a1c_gt7'] = (a1c_res == '>7').astype(float)
        d['a1c_normal'] = (a1c_res == 'Norm').astype(float)
        d['a1c_abnormal'] = np.isin(a1c_res, ['>7', '>8']).astype(float)

        d['glu_tested'] = (glu_res != 'None').astype(float)
        d['glu_untested'] = (glu_res == 'None').astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)
        d['glu_gt200'] = (glu_res == '>200').astype(float)
        d['glu_gt300'] = (glu_res == '>300').astype(float)
        d['glu_normal'] = (glu_res == 'Norm').astype(float)

        d['therapeutic_inertia'] = ((d['a1c_high'] > 0.5) & (d['change'] == 'No')).astype(float)
        d['uncontrolled_no_change'] = (((d['a1c_abnormal'] > 0.5) | (d['glu_high'] > 0.5)) & (d['change'] == 'No')).astype(float)
        d['a1c_high_med_changed'] = (d['a1c_high'] * d['change_flag']).astype(float)
        d['a1c_high_med_not_changed'] = (d['a1c_high'] * (1.0 - d['change_flag'])).astype(float)
        d['a1c_norm_med_steady'] = (d['a1c_normal'] * (1.0 - d['change_flag'])).astype(float)

        # Pathway & Chapter Interaction Strings
        d['diag1_disch'] = np.array([f"{g}_{p}" for g, p in zip(d['diag_1_grp'], disp)], dtype=object)
        d['adm_disch'] = np.array([f"{a}_{p}" for a, p in zip(adm_src, disp)], dtype=object)
        d['age_disch'] = np.array([f"{a}_{p}" for a, p in zip(d['age'], disp)], dtype=object)
        d['disch_spec'] = np.array([f"{p}_{s}" for p, s in zip(disp, spec_str)], dtype=object)
        d['diag_pair_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(d['diag_1_grp'], d['diag_2_grp'])], dtype=object)
        d['diag1_adm'] = np.array([f"{g}_{a}" for g, a in zip(d['diag_1_grp'], adm_type)], dtype=object)
        d['payer_disch'] = np.array([f"{py}_{dp}" for py, dp in zip(payer_str, disp)], dtype=object)

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
        'te_diag_pair_grp', 'te_diag1_adm', 'te_payer_disch'
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

        m_dp = compute_smoothed_map(train['diag_pair_grp'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_diag_pair_grp'][val_idx] = np.array([m_dp.get(safe_str(c), prior_glob) for c in train['diag_pair_grp'][val_idx]])

        m_d1a = compute_smoothed_map(train['diag1_adm'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_diag1_adm'][val_idx] = np.array([m_d1a.get(safe_str(c), prior_glob) for c in train['diag1_adm'][val_idx]])

        m_pd = compute_smoothed_map(train['payer_disch'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_payer_disch'][val_idx] = np.array([m_pd.get(safe_str(c), prior_glob) for c in train['payer_disch'][val_idx]])

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

    m_dp_all = compute_smoothed_map(train['diag_pair_grp'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_diag_pair_grp'] = np.array([m_dp_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_pair_grp']], dtype=float)

    m_d1a_all = compute_smoothed_map(train['diag1_adm'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_diag1_adm'] = np.array([m_d1a_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_adm']], dtype=float)

    m_pd_all = compute_smoothed_map(train['payer_disch'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_payer_disch'] = np.array([m_pd_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_disch']], dtype=float)

    # 2b. Dimensionality Reduction & Latent Patient Profile Extraction via PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Medication Matrix
    med_map = {'No': 0.0, 'Steady': 1.0, 'Down': -1.0, 'Up': 2.0}
    train_med_mat = np.zeros((len(train['gender']), len(med_cols)), dtype=float)
    eval_med_mat = np.zeros((len(eval_cols['gender']), len(med_cols)), dtype=float)
    for col_idx, col in enumerate(med_cols):
        train_med_mat[:, col_idx] = np.array([med_map.get(x, 0.0) for x in train[col]], dtype=float)
        eval_med_mat[:, col_idx] = np.array([med_map.get(x, 0.0) for x in eval_cols[col]], dtype=float)

    scaler_med = StandardScaler()
    train_med_scaled = scaler_med.fit_transform(train_med_mat)
    eval_med_scaled = scaler_med.transform(eval_med_mat)
    pca_med = PCA(n_components=4, random_state=42)
    train_med_pca = pca_med.fit_transform(train_med_scaled)
    eval_med_pca = pca_med.transform(eval_med_scaled)

    # Comorbidity Matrix
    comorb_cols = [
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_htn_comp', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_uti', 'has_sepsis', 'has_anemia', 'has_afib', 'has_ulcer_skin',
        'has_valvular', 'has_pulm_circ', 'has_depression', 'has_substance', 'has_obesity',
        'is_diab_type1', 'is_diab_uncontrolled', 'has_diab_uncomp', 'has_diab_comp', 'has_diab_acute', 
        'has_diab_renal', 'has_diab_retino', 'has_diab_neuro', 'has_diab_vasc'
    ]
    train_comorb_mat = np.column_stack([train[col] for col in comorb_cols])
    eval_comorb_mat = np.column_stack([eval_cols[col] for col in comorb_cols])

    scaler_comorb = StandardScaler()
    train_comorb_scaled = scaler_comorb.fit_transform(train_comorb_mat)
    eval_comorb_scaled = scaler_comorb.transform(eval_comorb_mat)
    pca_comorb = PCA(n_components=4, random_state=42)
    train_comorb_pca = pca_comorb.fit_transform(train_comorb_scaled)
    eval_comorb_pca = pca_comorb.transform(eval_comorb_scaled)

    # Target-Encoded Risk Matrix
    train_te_mat = np.column_stack([train[col] for col in te_features])
    eval_te_mat = np.column_stack([eval_cols[col] for col in te_features])

    scaler_te = StandardScaler()
    train_te_scaled = scaler_te.fit_transform(train_te_mat)
    eval_te_scaled = scaler_te.transform(eval_te_mat)
    pca_te = PCA(n_components=3, random_state=42)
    train_te_pca = pca_te.fit_transform(train_te_scaled)
    eval_te_pca = pca_te.transform(eval_te_scaled)

    # Assign to dictionaries
    for i in range(4):
        train[f'med_pca_{i}'] = train_med_pca[:, i]
        eval_cols[f'med_pca_{i}'] = eval_med_pca[:, i]
        train[f'comorb_pca_{i}'] = train_comorb_pca[:, i]
        eval_cols[f'comorb_pca_{i}'] = eval_comorb_pca[:, i]
    for i in range(3):
        train[f'te_pca_{i}'] = train_te_pca[:, i]
        eval_cols[f'te_pca_{i}'] = eval_te_pca[:, i]

    extra_num = [
        'med_pca_0', 'med_pca_1', 'med_pca_2', 'med_pca_3',
        'comorb_pca_0', 'comorb_pca_1', 'comorb_pca_2', 'comorb_pca_3',
        'te_pca_0', 'te_pca_1', 'te_pca_2',
        'age_num', 'total_visits', 'acute_visits', 'has_prior_inpatient', 'frequent_inpatient', 'super_utilizer', 'zero_prior_visits',
        'inpatient_ge1', 'inpatient_ge2', 'inpatient_ge3', 'inpatient_ge4', 'inpatient_ge5',
        'emergency_ge1', 'emergency_ge2', 'emergency_ge3', 'outpatient_ge1', 'no_prior_visits', 'prior_visit_diversity',
        'inpatient_ratio', 'emergency_ratio', 'outpatient_ratio', 'acute_visit_ratio',
        'weighted_prior_util', 'log_weighted_prior_util',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits', 'inpatient_squared',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'diag_per_day', 'clinical_intensity',
        'total_interventions', 'interventions_per_day', 'diag_workup_ratio',
        'log_lab_procedures', 'log_num_medications', 'log_time_in_hospital',
        'lab_per_med', 'lab_per_diag', 'high_util_long_stay', 'short_stay_acute',
        'no_proc_long_stay', 'high_lab_no_proc', 'diag_maxed',
        'stay_1', 'stay_2_3', 'stay_4_7', 'stay_ge8',
        'utilization_x_stay', 'utilization_x_facility', 'utilization_x_charlson', 'utilization_x_age',
        'emergent_unconnected', 'ed_reliant', 'complex_utilizer',
        'active_med_count', 'med_change_count', 'med_up_count', 'med_down_count',
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_unstable',
        'on_metformin', 'on_sulfonylurea', 'on_tzd', 'on_other_oral',
        'num_drug_classes', 'insulin_plus_oral', 'metformin_ckd_warning', 'sulfonylurea_elderly_warning',
        'sulfonylurea_renal_warning', 'insulin_renal_warning', 'insulin_and_sulf', 'polypharmacy', 'high_med_intensity',
        'change_flag', 'diabetes_med_flag',
        'regimen_insulin_only', 'regimen_oral_only', 'regimen_no_meds', 'regimen_insulin_sulf',
        'regimen_med_change_rate', 'med_up_ratio', 'med_down_ratio',
        'a1c_tested', 'a1c_untested', 'a1c_high', 'a1c_gt7', 'a1c_normal', 'a1c_abnormal',
        'glu_tested', 'glu_untested', 'glu_high', 'glu_gt200', 'glu_gt300', 'glu_normal',
        'therapeutic_inertia', 'uncontrolled_no_change',
        'a1c_high_med_changed', 'a1c_high_med_not_changed', 'a1c_norm_med_steady',
        'disch_is_home', 'disch_is_snf', 'disch_is_home_health', 'disch_is_short_hosp', 'disch_is_facility',
        'disch_is_transfer', 'disch_is_rehab', 'disch_is_ama', 'disch_is_psych',
        'is_emergency_adm', 'is_urgent_adm', 'is_elective_adm', 'is_acute_adm', 'is_trauma_adm',
        'adm_from_er', 'adm_from_referral', 'adm_from_transfer', 'adm_from_snf', 'adm_from_physician', 'adm_from_clinic',
        'er_to_facility', 'transfer_in_and_out', 'elective_to_home', 'er_to_snf', 'er_to_home_health',
        'sick_disch_home', 'home_health_cardiorenal', 'disch_ama_emergency',
        'payer_missing', 'spec_missing', 'weight_missing', 'race_missing',
        'payer_medicare', 'payer_medicaid', 'payer_commercial', 'payer_selfpay',
        'spec_cardio', 'spec_internal', 'spec_family', 'spec_emergency', 'spec_nephro', 'spec_surgery', 'spec_pulm', 'spec_psych',
        'is_age_90plus', 'is_age_80plus', 'is_age_70plus', 'is_age_under40', 'age_squared',
        'age_x_inpatient', 'age_x_charlson', 'age_x_stay',
        'elderly_ge75', 'elderly_snf', 'elderly_home', 'elderly_inpatient', 'elderly_polypharmacy',
        'age_gt75_snf', 'age_gt75_home', 'age_gt75_dementia', 'age_gt75_chf', 'age_gt75_renal', 'age_gt75_util',
        'diag1_is_chf', 'diag1_is_circ', 'diag1_is_resp', 'diag1_is_diab', 'diag1_is_digestive', 'diag1_is_infection', 'diag1_is_injury',
        'diag1_is_ckd', 'diag1_is_copd', 'diag1_is_pneumonia', 'diag1_is_fluid_elect', 'diag1_is_stroke', 'diag1_is_cad',
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_htn_comp', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_uti', 'has_sepsis', 'has_anemia', 'has_afib', 'has_ulcer_skin',
        'has_valvular', 'has_pulm_circ', 'has_depression', 'has_substance', 'has_obesity',
        'has_htn_renal', 'has_infection', 'has_foot_ulcer', 'has_microvascular', 'has_diab_hypo',
        'is_diab_type1', 'is_diab_uncontrolled', 'has_diab_uncomp', 'has_diab_comp', 'has_diab_acute', 'has_diab_renal', 'has_diab_retino', 'has_diab_neuro', 'has_diab_vasc', 'num_diab_organs',
        'cardiorenal_risk', 'cardiorenal_anemia', 'cardiorespiratory', 'metabolic_sepsis', 'vascular_burden',
        'charlson_score', 'elixhauser_count', 'lace_score', 'lace_plus',
        'freq_diag_1', 'freq_specialty', 'freq_discharge', 'freq_adm_source', 'freq_payer',
    ] + te_features

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag1_disch', 'adm_disch', 'age_disch', 'disch_spec',
        'diag_pair_grp', 'diag1_adm', 'payer_disch'
    ]

    # 3. Encoding Matrix Assembly
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 4. Penta-Engine GBDT Models across 4 Stratified Folds
    skf_models = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    oof_lgb1 = np.zeros(n_tr, dtype=float)
    oof_lgb2 = np.zeros(n_tr, dtype=float)
    oof_lgb3 = np.zeros(n_tr, dtype=float)
    oof_hgb = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_logits_lgb1 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_lgb2 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_lgb3 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_hgb = np.zeros(n_eval_rows, dtype=float)
    eval_logits_xgb = np.zeros(n_eval_rows, dtype=float)

    def to_logit(p, eps=1e-6):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    # Restrict HGB categorical features to moderate cardinality to avoid combinatorial bottleneck
    hgb_cat_indices = [idx for idx in encoder.categorical_indices if len(np.unique(x_train[:, idx])) <= 30]

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # Model 1: Standard Leaf-Wise LightGBM
        m_lgb1 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.03,
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
        eval_logits_lgb1 += to_logit(m_lgb1.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 2: Depth-Constrained Conservative LightGBM
        m_lgb2 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.032,
            max_depth=5,
            num_leaves=20,
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
        eval_logits_lgb2 += to_logit(m_lgb2.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 3: Extremely Randomized Trees LightGBM (Stochastic Split Diversity)
        m_lgb3 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=3.0,
            extra_trees=True,
            random_state=242 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_lgb3.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb3[val_idx] = m_lgb3.predict_proba(x_va)[:, 1]
        eval_logits_lgb3 += to_logit(m_lgb3.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 4: HistGradientBoostingClassifier (Exact Histogram Binning with Filtered Categoricals)
        m_hgb = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=28,
            min_samples_leaf=50,
            l2_regularization=5.0,
            categorical_features=hgb_cat_indices,
            random_state=42 + fold,
        )
        m_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = m_hgb.predict_proba(x_va)[:, 1]
        eval_logits_hgb += to_logit(m_hgb.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 5: XGBoost with histogram tree method
        m_xgb = xgb.XGBClassifier(
            n_estimators=210,
            learning_rate=0.03,
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
        eval_logits_xgb += to_logit(m_xgb.predict_proba(x_eval)[:, 1]) / 4.0

    # 5. Standardized Convex Warm-Started Simplex Logit-Space Ensembling
    raw_z_oofs = [to_logit(oof_lgb1), to_logit(oof_lgb2), to_logit(oof_lgb3), to_logit(oof_hgb), to_logit(oof_xgb)]
    raw_eval_logits = [eval_logits_lgb1, eval_logits_lgb2, eval_logits_lgb3, eval_logits_hgb, eval_logits_xgb]
    n_models = len(raw_z_oofs)

    # Standardize logit distributions to eliminate scale discrepancies across tree architectures
    z_means = [float(np.mean(z)) for z in raw_z_oofs]
    z_stds = [float(np.std(z)) + 1e-7 for z in raw_z_oofs]

    z_oofs_norm = [(z - m) / s for z, m, s in zip(raw_z_oofs, z_means, z_stds)]
    eval_z_norm = [(z - m) / s for z, m, s in zip(raw_eval_logits, z_means, z_stds)]
    z_mat = np.column_stack(z_oofs_norm)

    # Step 5a: Smooth strictly-convex binary cross-entropy warm start
    def log_loss_obj(w_raw):
        w = np.exp(w_raw - np.max(w_raw))
        w = w / np.sum(w)
        z_blend = z_mat @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z_blend, -15.0, 15.0)))
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y_train * np.log(p) + (1.0 - y_train) * np.log(1.0 - p))

    res_ll = minimize(log_loss_obj, np.zeros(n_models), method='BFGS', options={'maxiter': 50})
    w_start = res_ll.x

    # Step 5b: Simplex Nelder-Mead refinement directly on empirical ROC-AUC
    def ensemble_loss(raw_w):
        w = np.exp(raw_w - np.max(raw_w))
        w = w / np.sum(w)
        blend = sum(wi * zi for wi, zi in zip(w, z_oofs_norm))
        return -roc_auc_score(y_train, blend)

    res = minimize(ensemble_loss, w_start, method='Nelder-Mead', options={'maxiter': 100})
    best_w = np.exp(res.x - np.max(res.x)) / np.sum(np.exp(res.x - np.max(res.x)))

    # Reconstruct final probability prediction in standardized logit space
    z_final = sum(wi * zi for wi, zi in zip(best_w, eval_z_norm))
    pred = 1.0 / (1.0 + np.exp(-z_final))
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
