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
    """Fit on `train`/`y_train` and return readmission probabilities using Clinical Holographic Topics & Tri-GBDTs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
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
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        return str(code).strip().split('.')[0]

    # 1. Feature Engineering on both sets
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        # Missingness flags
        d['weight_missing'] = np.isin(d['weight'], [None, '?', '']).astype(float)
        d['payer_missing'] = np.isin(d['payer_code'], [None, '?', '']).astype(float)
        d['specialty_missing'] = np.isin(d['medical_specialty'], [None, '?', '']).astype(float)

        # Utilization & clinical intensity metrics
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

        # Polypharmacy & Medication instability
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
        d['active_meds'] = active
        d['num_med_changes'] = changes

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['has_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Clinical Guidelines & Biomarkers (Strack et al. 2014)
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['a1c_high_no_change'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)

        # Discharge and Admission risk factors
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf_hh'] = np.isin(disp, ['3', '6', '22', '28']).astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5']).astype(float)

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)

        # Diagnoses: groups, 3-digit categories & clinical comorbidity profile
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        # Vectorized parsing for all 29 Elixhauser Comorbidities
        chf = np.zeros(n_rows, dtype=float)
        arrhythmia = np.zeros(n_rows, dtype=float)
        valvular = np.zeros(n_rows, dtype=float)
        pulmonary_circ = np.zeros(n_rows, dtype=float)
        pvd = np.zeros(n_rows, dtype=float)
        htn = np.zeros(n_rows, dtype=float)
        paralysis = np.zeros(n_rows, dtype=float)
        neuro = np.zeros(n_rows, dtype=float)
        copd = np.zeros(n_rows, dtype=float)
        dm_uncomp = np.zeros(n_rows, dtype=float)
        dm_comp = np.zeros(n_rows, dtype=float)
        renal = np.zeros(n_rows, dtype=float)
        liver = np.zeros(n_rows, dtype=float)
        pud = np.zeros(n_rows, dtype=float)
        hiv = np.zeros(n_rows, dtype=float)
        lymphoma = np.zeros(n_rows, dtype=float)
        metastatic = np.zeros(n_rows, dtype=float)
        tumor = np.zeros(n_rows, dtype=float)
        rheumatoid = np.zeros(n_rows, dtype=float)
        coagulopathy = np.zeros(n_rows, dtype=float)
        obesity = np.zeros(n_rows, dtype=float)
        weight_loss = np.zeros(n_rows, dtype=float)
        fluid_electrolyte = np.zeros(n_rows, dtype=float)
        anemia_blood_loss = np.zeros(n_rows, dtype=float)
        anemia_deficiency = np.zeros(n_rows, dtype=float)
        alcohol = np.zeros(n_rows, dtype=float)
        drug = np.zeros(n_rows, dtype=float)
        psychoses = np.zeros(n_rows, dtype=float)
        depression = np.zeros(n_rows, dtype=float)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            for i, x in enumerate(d[col]):
                if x is None or x == '?' or x == '':
                    continue
                code = str(x).strip()
                parts = code.split('.')
                main = parts[0]
                if main.startswith('V'):
                    if code.startswith(('V42.0', 'V45.1', 'V56')):
                        renal[i] = 1.0
                    elif code.startswith(('V42.2', 'V43.3')):
                        valvular[i] = 1.0
                    elif code.startswith('V43.4'):
                        pvd[i] = 1.0
                    elif code.startswith('V45.0'):
                        arrhythmia[i] = 1.0
                    continue
                elif main.startswith('E'):
                    continue
                try:
                    num = int(main)
                except ValueError:
                    continue
                if num == 428:
                    chf[i] = 1.0
                elif num == 398 and code.endswith('91'):
                    chf[i] = 1.0
                elif num in (402, 404):
                    if len(parts) > 1 and parts[1] in ('01', '11', '91', '03', '13', '93'):
                        chf[i] = 1.0
                    if num == 403:
                        if len(parts) > 1 and parts[1] in ('01', '11', '91'):
                            renal[i] = 1.0
                    elif num == 404:
                        if len(parts) > 1 and parts[1] in ('02', '03', '12', '13', '92', '93'):
                            renal[i] = 1.0
                elif num == 425:
                    if len(parts) > 1 and parts[1] in ('4', '5', '6', '7', '8', '9'):
                        chf[i] = 1.0
                elif num == 427:
                    if len(parts) > 1 and parts[1] in ('0', '1', '2', '3', '4', '6', '7', '8', '9'):
                        arrhythmia[i] = 1.0
                elif num == 426:
                    if len(parts) > 1 and parts[1] in ('0', '13', '7', '9'):
                        arrhythmia[i] = 1.0
                elif num == 785 and code.endswith('0'):
                    arrhythmia[i] = 1.0
                if 394 <= num <= 397 or num == 424:
                    valvular[i] = 1.0
                if num in (416, 417) or (num == 415 and len(parts) > 1 and parts[1] in ('0', '1')):
                    pulmonary_circ[i] = 1.0
                if num in (440, 441, 442, 443, 449):
                    pvd[i] = 1.0
                elif num == 447 and len(parts) > 1 and parts[1] == '1':
                    pvd[i] = 1.0
                if 401 <= num <= 405:
                    htn[i] = 1.0
                if 342 <= num <= 344:
                    paralysis[i] = 1.0
                if num in (332, 334, 335, 340, 341, 345):
                    neuro[i] = 1.0
                elif num == 331 and len(parts) > 1 and parts[1] == '9':
                    neuro[i] = 1.0
                elif num == 333 and len(parts) > 1 and parts[1] in ('4', '5'):
                    neuro[i] = 1.0
                elif num == 348 and len(parts) > 1 and parts[1] in ('1', '3'):
                    neuro[i] = 1.0
                elif num == 780 and len(parts) > 1 and parts[1] == '3':
                    neuro[i] = 1.0
                elif num == 784 and len(parts) > 1 and parts[1] == '3':
                    neuro[i] = 1.0
                if 490 <= num <= 496 or 500 <= num <= 505:
                    copd[i] = 1.0
                elif num == 506 and len(parts) > 1 and parts[1] == '4':
                    copd[i] = 1.0
                if num == 250:
                    if len(parts) > 1:
                        dec = parts[1]
                        if dec in ('00', '02', '0'):
                            dm_uncomp[i] = 1.0
                        else:
                            dm_comp[i] = 1.0
                    else:
                        dm_uncomp[i] = 1.0
                if num in (585, 586):
                    renal[i] = 1.0
                if 571 <= num <= 573:
                    liver[i] = 1.0
                if 531 <= num <= 534:
                    pud[i] = 1.0
                if 42 <= num <= 44:
                    hiv[i] = 1.0
                if 200 <= num <= 202:
                    lymphoma[i] = 1.0
                elif num == 203 and len(parts) > 1 and parts[1] == '0':
                    lymphoma[i] = 1.0
                if 196 <= num <= 199:
                    metastatic[i] = 1.0
                if 140 <= num <= 195:
                    tumor[i] = 1.0
                if num in (710, 714, 720, 725):
                    rheumatoid[i] = 1.0
                if num in (286, 287):
                    coagulopathy[i] = 1.0
                if num == 278:
                    if len(parts) > 1 and parts[1].startswith('0'):
                        obesity[i] = 1.0
                if 260 <= num <= 263:
                    weight_loss[i] = 1.0
                if num == 276:
                    fluid_electrolyte[i] = 1.0
                if num == 280 and len(parts) > 1 and parts[1] == '0':
                    anemia_blood_loss[i] = 1.0
                if num == 281 or (num == 280 and len(parts) > 1 and parts[1] != '0'):
                    anemia_deficiency[i] = 1.0
                if num in (291, 303) or (num == 305 and len(parts) > 1 and parts[1] == '0'):
                    alcohol[i] = 1.0
                if num in (292, 304) or (num == 305 and len(parts) > 1 and parts[1] != '0'):
                    drug[i] = 1.0
                if 295 <= num <= 298:
                    psychoses[i] = 1.0
                if num in (309, 311) or (num == 300 and len(parts) > 1 and parts[1] == '4'):
                    depression[i] = 1.0

        d['elix_chf'] = chf
        d['elix_arrhythmia'] = arrhythmia
        d['elix_valvular'] = valvular
        d['elix_pulmonary_circ'] = pulmonary_circ
        d['elix_pvd'] = pvd
        d['elix_htn'] = htn
        d['elix_paralysis'] = paralysis
        d['elix_neuro'] = neuro
        d['elix_copd'] = copd
        d['elix_dm_uncomp'] = dm_uncomp
        d['elix_dm_comp'] = dm_comp
        d['elix_renal'] = renal
        d['elix_liver'] = liver
        d['elix_pud'] = pud
        d['elix_hiv'] = hiv
        d['elix_lymphoma'] = lymphoma
        d['elix_metastatic'] = metastatic
        d['elix_tumor'] = tumor
        d['elix_rheumatoid'] = rheumatoid
        d['elix_coagulopathy'] = coagulopathy
        d['elix_obesity'] = obesity
        d['elix_weight_loss'] = weight_loss
        d['elix_fluid_electrolyte'] = fluid_electrolyte
        d['elix_anemia_blood_loss'] = anemia_blood_loss
        d['elix_anemia_deficiency'] = anemia_deficiency
        d['elix_alcohol'] = alcohol
        d['elix_drug'] = drug
        d['elix_psychoses'] = psychoses
        d['elix_depression'] = depression

        # Van Walraven score
        d['van_walraven_score'] = (
            7.0 * chf +
            5.0 * arrhythmia +
            -1.0 * valvular +
            4.0 * pulmonary_circ +
            2.0 * pvd +
            7.0 * paralysis +
            6.0 * neuro +
            3.0 * copd +
            -3.0 * dm_comp +
            6.0 * renal +
            11.0 * liver +
            9.0 * lymphoma +
            12.0 * metastatic +
            4.0 * tumor +
            11.0 * coagulopathy +
            -4.0 * obesity +
            6.0 * weight_loss +
            5.0 * fluid_electrolyte +
            -2.0 * anemia_blood_loss +
            -2.0 * anemia_deficiency +
            -7.0 * drug +
            -5.0 * psychoses +
            -3.0 * depression
        )

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_ckd = np.isin(c1, ['585', '586']) | np.isin(c2, ['585', '586']) | np.isin(c3, ['585', '586'])
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_pvd = np.isin(c1, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c2, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c3, ['440', '441', '442', '443', '444', '445', '446', '447'])
        has_stroke = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')
        has_pneumonia = np.isin(c1, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c2, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c3, ['480', '481', '482', '483', '484', '485', '486'])

        has_diab_comp = np.zeros(n_rows, dtype=bool)
        has_diab_acute = np.zeros(n_rows, dtype=bool)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            has_diab_comp |= np.array([str(x).startswith('250') and str(x) not in ['250', '250.0', '250.00', '250.02', '250.1', '250.2', '250.3'] for x in d[col]])
            has_diab_acute |= np.array([str(x).startswith(('250.1', '250.2', '250.3')) for x in d[col]])

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_pvd'] = has_pvd.astype(float)
        d['has_stroke'] = has_stroke.astype(float)
        d['has_fluid_elect'] = has_fluid_elect.astype(float)
        d['has_pneumonia'] = has_pneumonia.astype(float)
        d['has_diab_comp'] = has_diab_comp.astype(float)
        d['has_diab_acute'] = has_diab_acute.astype(float)
        d['cardiorenal_risk'] = (has_chf & has_ckd).astype(float)

        comorb = (
            d['has_chf'] + d['has_ckd'] + d['has_cad'] + d['has_htn'] + d['has_copd'] +
            d['has_pvd'] + d['has_stroke'] + d['has_fluid_elect'] + d['has_pneumonia'] + d['has_diab_comp']
        )
        d['comorbidity_count'] = comorb
        d['frailty_index'] = comorb * (d['age_num'] / 50.0) * np.log1p(d['number_inpatient'] + 1.0)

        # Medication class counts
        oral_agents = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citagliptin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        num_oral_active = np.zeros(n_rows, dtype=float)
        for col in oral_agents:
            if col in d:
                num_oral_active += (d[col] != 'No').astype(float)
        d['num_oral_active'] = num_oral_active

        has_biguanide = np.zeros(n_rows, dtype=bool)
        for m in ['metformin', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone']:
            if m in d:
                has_biguanide |= (d[m] != 'No')

        has_sulfonylurea = np.zeros(n_rows, dtype=bool)
        for m in ['glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'tolazamide', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone']:
            if m in d:
                has_sulfonylurea |= (d[m] != 'No')

        has_meglitinide = np.zeros(n_rows, dtype=bool)
        for m in ['repaglinide', 'nateglinide']:
            if m in d:
                has_meglitinide |= (d[m] != 'No')

        has_tzd = np.zeros(n_rows, dtype=bool)
        for m in ['pioglitazone', 'rosiglitazone', 'troglitazone', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']:
            if m in d:
                has_tzd |= (d[m] != 'No')

        has_agi = np.zeros(n_rows, dtype=bool)
        for m in ['acarbose', 'miglitol']:
            if m in d:
                has_agi |= (d[m] != 'No')

        d['has_biguanide'] = has_biguanide.astype(float)
        d['has_sulfonylurea'] = has_sulfonylurea.astype(float)
        d['has_meglitinide'] = has_meglitinide.astype(float)
        d['has_tzd'] = has_tzd.astype(float)
        d['has_agi'] = has_agi.astype(float)

        d['num_med_classes'] = (
            d['has_biguanide'] + d['has_sulfonylurea'] + d['has_meglitinide'] +
            d['has_tzd'] + d['has_agi'] + d['has_insulin']
        )

        # LACE Score calculation
        stay_days = d['time_in_hospital']
        l_score = np.zeros(n_rows, dtype=float)
        l_score[stay_days == 1] = 1.0
        l_score[stay_days == 2] = 2.0
        l_score[stay_days == 3] = 3.0
        l_score[(stay_days >= 4) & (stay_days <= 6)] = 4.0
        l_score[(stay_days >= 7) & (stay_days <= 13)] = 5.0
        l_score[stay_days >= 14] = 7.0

        a_score = (adm_type == '1').astype(float) * 3.0

        c_score = np.zeros(n_rows, dtype=float)
        c_score[comorb == 1] = 1.0
        c_score[comorb == 2] = 2.0
        c_score[comorb == 3] = 3.0
        c_score[comorb >= 4] = 5.0

        er_visits = d['number_emergency']
        e_score = np.zeros(n_rows, dtype=float)
        e_score[er_visits == 1] = 1.0
        e_score[er_visits == 2] = 2.0
        e_score[er_visits == 3] = 3.0
        e_score[er_visits >= 4] = 4.0

        d['lace_score'] = l_score + a_score + c_score + e_score

        # Interactions
        d['age_med_interaction'] = d['age_num'] * d['num_medications']
        d['hospital_lab_interaction'] = d['time_in_hospital'] * d['num_lab_procedures']
        d['inpatient_diagnoses_interaction'] = d['number_inpatient'] * d['number_diagnoses']
        d['inpatient_lace_interaction'] = d['number_inpatient'] * d['lace_score']
        d['comorb_stay_interaction'] = d['comorbidity_count'] * d['time_in_hospital']

    # 2. Bayesian Smoothed Target Encoding (Out-of-Fold within train)
    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

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

    for tr_idx, val_idx in skf.split(train['gender'], y_train):
        # Primary diagnosis and chapter group mapping
        d1_map = compute_smoothed_map(train['diag_1_3d'][tr_idx], y_train[tr_idx], smooth=25.0, prior=prior_glob)
        tr_diag_1[val_idx] = np.array([d1_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])

        d1_grp_map = compute_smoothed_map(train['diag_1_grp'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_diag_1_grp[val_idx] = np.array([d1_grp_map.get(safe_str(c), prior_glob) for c in train['diag_1_grp'][val_idx]])

        # Pooled secondary diagnoses mapping
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

    train['te_diag_1'] = tr_diag_1
    train['te_diag_1_grp'] = tr_diag_1_grp
    train['te_diag_max'] = tr_diag_max
    train['te_diag_mean'] = tr_diag_mean
    train['te_diag_sum'] = tr_diag_sum
    train['te_discharge'] = tr_disp
    train['te_specialty'] = tr_spec
    train['te_adm_source'] = tr_src
    train['te_adm_type'] = tr_adm_type

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

    # 3. Non-Negative Clinical Phenotyping Matrix Factorization (NMPF) & Narrative Topics
    from sklearn.decomposition import NMF

    def build_phenotype_matrix(d):
        n = len(d['gender'])
        med_intensity_map = {'No': 0.0, 'Steady': 1.0, 'Down': 1.5, 'Up': 2.0}
        ins_val = np.array([med_intensity_map.get(str(x), 0.0) for x in d.get('insulin', ['No'] * n)])
        met_val = np.array([med_intensity_map.get(str(x), 0.0) for x in d.get('metformin', ['No'] * n)])

        # Sulfonylurea class max
        sulf_val = np.zeros(n, dtype=float)
        for s_col in ['glipizide', 'glyburide', 'glimepiride']:
            if s_col in d:
                s_arr = np.array([med_intensity_map.get(str(x), 0.0) for x in d[s_col]])
                sulf_val = np.maximum(sulf_val, s_arr)

        log_inp = np.log1p(np.maximum(0.0, d['number_inpatient']))
        log_er = np.log1p(np.maximum(0.0, d['number_emergency']))
        stay_norm = d['time_in_hospital'] / 14.0

        # Multi-modal non-negative feature slices: Comorbidities, Meds, Trajectory, and Bilinear Interactions
        cols = [
            d['has_chf'], d['has_ckd'], d['has_cad'], d['has_htn'], d['has_copd'],
            d['has_pvd'], d['has_stroke'], d['has_fluid_elect'], d['has_pneumonia'],
            d['has_diab_acute'], d['has_diab_comp'], d['cardiorenal_risk'],
            ins_val, met_val, sulf_val,
            d['active_meds'] / 10.0, d['num_med_changes'] / 4.0,
            d['disch_is_snf_hh'], d['disch_is_ama'], d['disch_is_transfer'],
            d['is_emergency_adm'], log_inp, log_er, stay_norm,
            d['has_chf'] * ins_val,
            d['has_ckd'] * ins_val,
            d['has_cad'] * ins_val,
            d['has_diab_acute'] * (d['num_med_changes'] > 0).astype(float),
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

    # Topic SVD on narrative categorical corpus
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

    tfidf = TfidfVectorizer(max_features=100, min_df=5, token_pattern=r'(?u)\b\S+\b')
    tr_tfidf = tfidf.fit_transform(build_narrative_corpus(train))
    ev_tfidf = tfidf.transform(build_narrative_corpus(eval_cols))

    svd = TruncatedSVD(n_components=4, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    topic_names = [f'topic_{k}' for k in range(4)]
    for k in range(4):
        train[f'topic_{k}'] = tr_svd[:, k].astype(float)
        eval_cols[f'topic_{k}'] = ev_svd[:, k].astype(float)

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'frequent_inpatient', 'med_per_day', 'lab_per_day',
        'proc_per_day', 'diag_per_day', 'service_intensity',
        'active_meds', 'num_med_changes', 'has_insulin', 'insulin_up', 'insulin_down',
        'change_flag', 'diabetes_med_flag',
        'a1c_tested', 'a1c_high', 'a1c_high_no_change', 'glu_high',
        'disch_is_home', 'disch_is_snf_hh', 'disch_is_ama', 'disch_is_transfer',
        'is_emergency_adm', 'is_elective_adm',
        'has_chf', 'has_ckd', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_diab_acute', 'has_diab_comp', 'cardiorenal_risk',
        'comorbidity_count', 'frailty_index',
        'te_diag_1', 'te_diag_1_grp', 'te_diag_max', 'te_diag_mean', 'te_diag_sum',
        'te_discharge', 'te_specialty', 'te_adm_source', 'te_adm_type',
    ] + pheno_names + topic_names

    extra_cat = ['diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d']

    # 4. Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 5. Tri-Engine GBDT Models with 4-Fold Cross-Validation and Simplex ROC-AUC Optimization
    from sklearn.metrics import roc_auc_score

    skf_models = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)
    oof_hgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_lgb = np.zeros(n_eval_rows, dtype=float)
    eval_xgb = np.zeros(n_eval_rows, dtype=float)
    eval_hgb = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # LightGBM (Leaf-wise expansion)
        model_lgb = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=60,
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
        eval_lgb += model_lgb.predict_proba(x_eval)[:, 1] / 4.0

        # HistGradientBoosting (Histogram binning)
        model_hgb = HistGradientBoostingClassifier(
            max_iter=170,
            learning_rate=0.035,
            max_leaf_nodes=31,
            min_samples_leaf=60,
            l2_regularization=4.0,
            categorical_features=encoder.categorical_indices,
            random_state=42 + fold,
        )
        model_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = model_hgb.predict_proba(x_va)[:, 1]
        eval_hgb += model_hgb.predict_proba(x_eval)[:, 1] / 4.0

        # XGBoost (Depth-wise expansion with histogram splitting)
        model_xgb = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        model_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = model_xgb.predict_proba(x_va)[:, 1]
        eval_xgb += model_xgb.predict_proba(x_eval)[:, 1] / 4.0

    # 6. Bayesian Logit Space Simplex Optimization for Maximum ROC-AUC
    def to_logit(p, eps=1e-6):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    z_oof_lgb = to_logit(oof_lgb)
    z_oof_xgb = to_logit(oof_xgb)
    z_oof_hgb = to_logit(oof_hgb)

    best_auc = -1.0
    best_w = (0.45, 0.30, 0.25)

    # Exhaustive fine-simplex grid search over blend weights
    for w1 in np.linspace(0.1, 0.8, 15):
        for w2 in np.linspace(0.1, 0.8, 15):
            if w1 + w2 < 0.95:
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    blend_oof = w1 * z_oof_lgb + w2 * z_oof_xgb + w3 * z_oof_hgb
                    auc = roc_auc_score(y_train, blend_oof)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = (w1, w2, w3)

    w1, w2, w3 = best_w
    z_eval = w1 * to_logit(eval_lgb) + w2 * to_logit(eval_xgb) + w3 * to_logit(eval_hgb)
    pred = 1.0 / (1.0 + np.exp(-z_eval))
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
