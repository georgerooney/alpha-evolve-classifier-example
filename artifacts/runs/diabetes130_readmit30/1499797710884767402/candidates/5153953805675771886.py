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


def target_encode(train_col, y_train, eval_col, alpha=15.0, n_splits=5):
    from sklearn.model_selection import KFold
    tr_col_clean = np.array([str(x) if x is not None else "None" for x in train_col])
    ev_col_clean = np.array([str(x) if x is not None else "None" for x in eval_col])
    
    global_sum = {}
    global_cnt = {}
    for val, y in zip(tr_col_clean, y_train):
        global_sum[val] = global_sum.get(val, 0.0) + y
        global_cnt[val] = global_cnt.get(val, 0.0) + 1.0
        
    global_mean = float(np.mean(y_train))
    eval_encoded = np.array([(global_sum.get(v, 0.0) + alpha * global_mean) / (global_cnt.get(v, 0.0) + alpha) for v in ev_col_clean])
    
    train_encoded = np.zeros(len(tr_col_clean), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for _, val_idx in kf.split(tr_col_clean):
        val_sum = {}
        val_cnt = {}
        for idx in val_idx:
            v = tr_col_clean[idx]
            val_sum[v] = val_sum.get(v, 0.0) + y_train[idx]
            val_cnt[v] = val_cnt.get(v, 0.0) + 1.0
        for idx in val_idx:
            v = tr_col_clean[idx]
            s = global_sum[v] - val_sum[v]
            c = global_cnt[v] - val_cnt[v]
            train_encoded[idx] = (s + alpha * global_mean) / (c + alpha)
            
    return train_encoded, eval_encoded


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    train_copy = dict(train)
    eval_copy = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # 1. Clean ICD-9 Prefix extractor
    def get_icd_prefix(val):
        if val is None or not isinstance(val, str) or val in ('?', 'None'):
            return 'missing'
        s = val.strip()
        if not s:
            return 'missing'
        if s[0] in ('V', 'v'):
            return s[:3]
        if s[0] in ('E', 'e'):
            return s[:4]
        dot_idx = s.find('.')
        if dot_idx != -1:
            return s[:dot_idx]
        return s[:3]

    for col in ('diag_1', 'diag_2', 'diag_3'):
        p_col = f'{col}_prefix'
        train_copy[p_col] = np.array([get_icd_prefix(x) for x in train_copy[col]])
        eval_copy[p_col] = np.array([get_icd_prefix(x) for x in eval_copy[col]])
        extra_categorical.append(p_col)

    # 2. Age parsing
    def parse_age(age_str):
        if age_str is None:
            return np.nan
        s = age_str.replace('[', '').replace(')', '').replace(']', '').replace('(', '')
        parts = s.split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                return np.nan
        return np.nan

    train_copy['age_numeric'] = np.array([parse_age(x) for x in train_copy['age']])
    eval_copy['age_numeric'] = np.array([parse_age(x) for x in eval_copy['age']])
    extra_numeric.append('age_numeric')

    # 3. Clinical Comorbidities & Charlson Scoring
    def extract_comorbidities(d):
        n = len(d['time_in_hospital'])
        chf = np.zeros(n, dtype=float)
        ckd = np.zeros(n, dtype=float)
        cad = np.zeros(n, dtype=float)
        copd = np.zeros(n, dtype=float)
        stroke = np.zeros(n, dtype=float)
        diab_comp = np.zeros(n, dtype=float)
        pvd = np.zeros(n, dtype=float)
        htn = np.zeros(n, dtype=float)

        for diag_key in ('diag_1', 'diag_2', 'diag_3'):
            arr = d[diag_key]
            for i in range(n):
                val = arr[i]
                if val is None or val in ('?', 'None'):
                    continue
                code = str(val).strip()
                prefix = code.split('.')[0] if '.' in code else code[:3]
                if prefix == '428':
                    chf[i] = 1.0
                elif prefix in ('585', '586', '582', '583'):
                    ckd[i] = 1.0
                elif prefix in ('410', '411', '412', '414'):
                    cad[i] = 1.0
                elif prefix in ('491', '492', '493', '494', '496'):
                    copd[i] = 1.0
                elif prefix in ('430', '431', '432', '433', '434', '435', '436', '437', '438'):
                    stroke[i] = 1.0
                elif code.startswith('250.'):
                    suffix = code[4:]
                    if suffix and suffix[0] in ('4', '5', '6', '7', '8', '9'):
                        diab_comp[i] = 1.0
                elif prefix in ('440', '441', '442', '443'):
                    pvd[i] = 1.0
                elif prefix in ('401', '402', '403', '404', '405'):
                    htn[i] = 1.0

        cardiorenal = (chf * ckd).astype(float)
        diab_chf = (diab_comp * chf).astype(float)
        diab_ckd = (diab_comp * ckd).astype(float)
        charlson = chf * 2.0 + ckd * 2.0 + cad * 1.0 + copd * 1.0 + stroke * 1.0 + diab_comp * 2.0 + pvd * 1.0
        comorb_sum = chf + ckd + cad + copd + stroke + diab_comp + pvd + htn
        return {
            'comorb_chf': chf,
            'comorb_ckd': ckd,
            'comorb_cad': cad,
            'comorb_copd': copd,
            'comorb_stroke': stroke,
            'comorb_diab_comp': diab_comp,
            'comorb_pvd': pvd,
            'comorb_htn': htn,
            'comorb_cardiorenal': cardiorenal,
            'comorb_diab_chf': diab_chf,
            'comorb_diab_ckd': diab_ckd,
            'charlson_score': charlson,
            'comorb_sum': comorb_sum,
        }

    tr_comorb = extract_comorbidities(train_copy)
    ev_comorb = extract_comorbidities(eval_copy)
    for k, v in tr_comorb.items():
        train_copy[k] = v
        eval_copy[k] = ev_comorb[k]
        extra_numeric.append(k)

    # 4. ICD9 Groups and Multi-Organ System Discordance
    train_diag1_grp = np.array([icd9_group(x) for x in train_copy['diag_1']])
    train_diag2_grp = np.array([icd9_group(x) for x in train_copy['diag_2']])
    train_diag3_grp = np.array([icd9_group(x) for x in train_copy['diag_3']])

    eval_diag1_grp = np.array([icd9_group(x) for x in eval_copy['diag_1']])
    eval_diag2_grp = np.array([icd9_group(x) for x in eval_copy['diag_2']])
    eval_diag3_grp = np.array([icd9_group(x) for x in eval_copy['diag_3']])

    train_copy['diag_1_group'] = train_diag1_grp
    train_copy['diag_2_group'] = train_diag2_grp
    train_copy['diag_3_group'] = train_diag3_grp

    eval_copy['diag_1_group'] = eval_diag1_grp
    eval_copy['diag_2_group'] = eval_diag2_grp
    eval_copy['diag_3_group'] = eval_diag3_grp

    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group'])

    train_copy['primary_circulatory'] = (train_diag1_grp == 'circulatory').astype(float)
    train_copy['primary_respiratory'] = (train_diag1_grp == 'respiratory').astype(float)
    train_copy['primary_digestive'] = (train_diag1_grp == 'digestive').astype(float)
    train_copy['primary_diabetes'] = (train_diag1_grp == 'diabetes').astype(float)

    eval_copy['primary_circulatory'] = (eval_diag1_grp == 'circulatory').astype(float)
    eval_copy['primary_respiratory'] = (eval_diag1_grp == 'respiratory').astype(float)
    eval_copy['primary_digestive'] = (eval_diag1_grp == 'digestive').astype(float)
    eval_copy['primary_diabetes'] = (eval_diag1_grp == 'diabetes').astype(float)

    extra_numeric.extend(['primary_circulatory', 'primary_respiratory', 'primary_digestive', 'primary_diabetes'])

    def compute_discordance(g1_arr, g2_arr, g3_arr):
        n = len(g1_arr)
        disc = np.zeros(n, dtype=float)
        for i in range(n):
            valid = {g for g in (g1_arr[i], g2_arr[i], g3_arr[i]) if g not in ('missing', 'other')}
            disc[i] = float(len(valid))
        return disc

    train_copy['organ_system_discordance'] = compute_discordance(train_diag1_grp, train_diag2_grp, train_diag3_grp)
    eval_copy['organ_system_discordance'] = compute_discordance(eval_diag1_grp, eval_diag2_grp, eval_diag3_grp)
    extra_numeric.append('organ_system_discordance')

    all_groups = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms']
    for grp in all_groups:
        col_name = f'grp_cnt_{grp}'
        train_copy[col_name] = (train_diag1_grp == grp).astype(float) + (train_diag2_grp == grp).astype(float) + (train_diag3_grp == grp).astype(float)
        eval_copy[col_name] = (eval_diag1_grp == grp).astype(float) + (eval_diag2_grp == grp).astype(float) + (eval_diag3_grp == grp).astype(float)
        extra_numeric.append(col_name)

    # 5. Admission and Discharge Grouping
    def group_discharge(disp):
        if disp is None:
            return 'missing'
        disp = str(disp)
        if disp == '1':
            return 'home'
        elif disp in ('2', '3', '4', '5', '22', '23', '24'):
            return 'transfer_facility'
        elif disp in ('6', '8'):
            return 'home_health'
        elif disp == '7':
            return 'left_ama'
        else:
            return 'other'

    train_copy['discharge_group'] = np.array([group_discharge(x) for x in train_copy['discharge_disposition_id']])
    eval_copy['discharge_group'] = np.array([group_discharge(x) for x in eval_copy['discharge_disposition_id']])
    extra_categorical.append('discharge_group')

    def group_admission_source(src):
        if src is None:
            return 'missing'
        src = str(src)
        if src in ('1', '2', '3'):
            return 'referral'
        elif src == '7':
            return 'emergency'
        elif src in ('4', '5', '6', '10', '22', '25'):
            return 'transfer'
        else:
            return 'other'

    train_copy['admission_source_group'] = np.array([group_admission_source(x) for x in train_copy['admission_source_id']])
    eval_copy['admission_source_group'] = np.array([group_admission_source(x) for x in eval_copy['admission_source_id']])
    extra_categorical.append('admission_source_group')

    def group_admission_type(typ):
        if typ is None:
            return 'missing'
        typ = str(typ)
        if typ == '1':
            return 'emergency'
        elif typ == '2':
            return 'urgent'
        elif typ == '3':
            return 'elective'
        else:
            return 'other'

    train_copy['admission_type_group'] = np.array([group_admission_type(x) for x in train_copy['admission_type_id']])
    eval_copy['admission_type_group'] = np.array([group_admission_type(x) for x in eval_copy['admission_type_id']])
    extra_categorical.append('admission_type_group')

    # 6. Latent Clinical Phenotyping via TruncatedSVD (Low-Rank Decomposition)
    def build_clinical_tokens(d):
        n = len(d['time_in_hospital'])
        docs = []
        for i in range(n):
            p1 = d['diag_1_prefix'][i]
            p2 = d['diag_2_prefix'][i]
            p3 = d['diag_3_prefix'][i]
            d1 = str(d['diag_1'][i]) if d['diag_1'][i] is not None else 'None'
            d2 = str(d['diag_2'][i]) if d['diag_2'][i] is not None else 'None'
            d3 = str(d['diag_3'][i]) if d['diag_3'][i] is not None else 'None'
            disp = str(d['discharge_disposition_id'][i])
            adm = str(d['admission_source_id'][i])
            spec = str(d['medical_specialty'][i]) if d['medical_specialty'][i] is not None else 'None'
            docs.append(f"dx_{p1} dx_{p2} dx_{p3} raw_{d1} raw_{d2} raw_{d3} disp_{disp} adm_{adm} spec_{spec}")
        return docs

    tr_docs = build_clinical_tokens(train_copy)
    ev_docs = build_clinical_tokens(eval_copy)

    tfidf = TfidfVectorizer(max_features=500, lowercase=False, token_pattern=r'\S+')
    tr_tfidf = tfidf.fit_transform(tr_docs)
    ev_tfidf = tfidf.transform(ev_docs)

    svd = TruncatedSVD(n_components=8, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    for k in range(8):
        col_name = f'clin_svd_{k}'
        train_copy[col_name] = tr_svd[:, k]
        eval_copy[col_name] = ev_svd[:, k]
        extra_numeric.append(col_name)

    # 7. Healthcare Utilization Dynamics & Revolving Door Index
    for d in (train_copy, eval_copy):
        tot_u = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['total_utilization'] = tot_u
        d['revolving_door_index'] = 3.0 * d['number_inpatient'] + 1.5 * d['number_emergency'] + 0.5 * d['number_outpatient']
        d['inpatient_fraction'] = d['number_inpatient'] / (tot_u + 1.0)
        d['emergency_fraction'] = d['number_emergency'] / (tot_u + 1.0)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)
        d['has_emergency'] = (d['number_emergency'] > 0).astype(float)
        d['inpatient_to_emergency_ratio'] = d['number_inpatient'] / (d['number_emergency'] + 1.0)
        d['inpatient_to_outpatient_ratio'] = d['number_inpatient'] / (d['number_outpatient'] + 1.0)
        d['emergency_to_outpatient_ratio'] = d['number_emergency'] / (d['number_outpatient'] + 1.0)
        d['clinical_activity_index'] = d['num_lab_procedures'] + d['num_procedures'] + d['num_medications'] + d['number_diagnoses']

    extra_numeric.extend([
        'total_utilization', 'revolving_door_index', 'inpatient_fraction',
        'emergency_fraction', 'has_prior_inpatient', 'frequent_inpatient', 'has_emergency',
        'inpatient_to_emergency_ratio', 'inpatient_to_outpatient_ratio', 'emergency_to_outpatient_ratio',
        'clinical_activity_index'
    ])

    # 8. Clinical Service Ratios
    for d in (train_copy, eval_copy):
        hosp_days = d['time_in_hospital'] + 1e-4
        d['lab_intensity'] = d['num_lab_procedures'] / hosp_days
        d['proc_intensity'] = d['num_procedures'] / hosp_days
        d['med_intensity'] = d['num_medications'] / hosp_days
        d['diag_intensity'] = d['number_diagnoses'] / hosp_days
        d['med_per_diag'] = d['num_medications'] / (d['number_diagnoses'] + 1e-4)
        d['lab_per_med'] = d['num_lab_procedures'] / (d['num_medications'] + 1e-4)
        d['service_load'] = (d['num_lab_procedures'] + 2.0 * d['num_procedures'] + d['num_medications']) / hosp_days

    extra_numeric.extend([
        'lab_intensity', 'proc_intensity', 'med_intensity', 'diag_intensity',
        'med_per_diag', 'lab_per_med', 'service_load'
    ])

    # 9. Medication Regimen, Glycemic Control & Discordance
    non_med = {
        'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
    }
    med_cols = [c for c in task.categorical_columns if c not in non_med]
    oral_med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide'
    ]
    present_orals = [m for m in oral_med_names if m in task.categorical_columns]

    for d in (train_copy, eval_copy):
        n = len(d['time_in_hospital'])
        num_up = np.zeros(n)
        num_down = np.zeros(n)
        num_steady = np.zeros(n)
        num_taken = np.zeros(n)
        num_oral = np.zeros(n)

        for col in med_cols:
            arr = d[col]
            num_up += (arr == 'Up').astype(float)
            num_down += (arr == 'Down').astype(float)
            num_steady += (arr == 'Steady').astype(float)
            num_taken += ((arr != 'No') & (arr != None)).astype(float)

        for col in present_orals:
            arr = d[col]
            num_oral += ((arr != 'No') & (arr != None)).astype(float)

        d['num_meds_up'] = num_up
        d['num_meds_down'] = num_down
        d['num_meds_steady'] = num_steady
        d['num_meds_taken'] = num_taken
        d['med_change_count'] = num_up + num_down
        d['num_oral_meds'] = num_oral
        d['polypharmacy'] = (num_taken >= 3).astype(float)

        if 'insulin' in d:
            ins = d['insulin']
            d['insulin_change'] = ((ins == 'Up') | (ins == 'Down')).astype(float)
            d['insulin_up'] = (ins == 'Up').astype(float)
            d['insulin_down'] = (ins == 'Down').astype(float)
            d['insulin_steady'] = (ins == 'Steady').astype(float)
            d['insulin_taking'] = ((ins != 'No') & (ins != None)).astype(float)
            d['oral_and_insulin'] = (d['insulin_taking'] * (num_oral > 0)).astype(float)
        else:
            d['insulin_change'] = np.zeros(n)
            d['insulin_up'] = np.zeros(n)
            d['insulin_down'] = np.zeros(n)
            d['insulin_steady'] = np.zeros(n)
            d['insulin_taking'] = np.zeros(n)
            d['oral_and_insulin'] = np.zeros(n)

        # New exclusive treatment types
        d['treat_none'] = (num_taken == 0).astype(float)
        d['treat_oral_only'] = ((num_oral > 0) & (d['insulin_taking'] == 0)).astype(float)
        d['treat_insulin_only'] = ((num_oral == 0) & (d['insulin_taking'] > 0)).astype(float)
        d['treat_both'] = ((num_oral > 0) & (d['insulin_taking'] > 0)).astype(float)

        # Glycemic testing indicators
        glu = d['max_glu_serum']
        a1c = d['A1Cresult']
        d['glu_test_performed'] = (glu != 'None').astype(float)
        d['a1c_test_performed'] = (a1c != 'None').astype(float)
        d['untested_glycemia'] = ((glu == 'None') & (a1c == 'None')).astype(float)
        d['a1c_high'] = ((a1c == '>8') | (a1c == '>7')).astype(float)
        d['glu_high'] = ((glu == '>200') | (glu == '>300')).astype(float)

        # New glycemic indicators & clinical inertia
        d['a1c_normal'] = (a1c == 'Norm').astype(float)
        d['glu_normal'] = (glu == 'Norm').astype(float)
        d['both_tests_performed'] = (d['glu_test_performed'] * d['a1c_test_performed']).astype(float)
        
        has_change = (d['change'] == 'Ch').astype(float)
        d['clinical_inertia'] = (d['a1c_high'] * (1.0 - has_change)).astype(float)
        d['treatment_escalation'] = (d['a1c_high'] * has_change).astype(float)

    extra_numeric.extend([
        'num_meds_up', 'num_meds_down', 'num_meds_steady', 'num_meds_taken', 'med_change_count',
        'num_oral_meds', 'polypharmacy', 'insulin_change', 'insulin_up', 'insulin_down',
        'insulin_steady', 'insulin_taking', 'oral_and_insulin',
        'treat_none', 'treat_oral_only', 'treat_insulin_only', 'treat_both',
        'glu_test_performed', 'a1c_test_performed', 'untested_glycemia', 'a1c_high', 'glu_high',
        'a1c_normal', 'glu_normal', 'both_tests_performed', 'clinical_inertia', 'treatment_escalation'
    ])

    # 10. High-Risk Clinical Interactions
    for d in (train_copy, eval_copy):
        is_transfer_or_snf = np.isin(d['discharge_group'], ['home_health', 'transfer_facility']).astype(float)
        d['high_risk_disp_prior_inpatient'] = is_transfer_or_snf * (d['number_inpatient'] > 0).astype(float)
        d['chf_or_ckd_transfer'] = ((d['comorb_chf'] == 1.0) | (d['comorb_ckd'] == 1.0)) * is_transfer_or_snf
        d['ama_discharge'] = (d['discharge_group'] == 'left_ama').astype(float)

    extra_numeric.extend(['high_risk_disp_prior_inpatient', 'chf_or_ckd_transfer', 'ama_discharge'])

    # 11. Weight Measured
    train_copy['weight_measured'] = (train_copy['weight'] != None).astype(float)
    eval_copy['weight_measured'] = (eval_copy['weight'] != None).astype(float)
    extra_numeric.append('weight_measured')

    train_copy['medical_specialty_measured'] = (train_copy['medical_specialty'] != None).astype(float)
    eval_copy['medical_specialty_measured'] = (eval_copy['medical_specialty'] != None).astype(float)
    extra_numeric.append('medical_specialty_measured')

    train_copy['payer_code_measured'] = (train_copy['payer_code'] != None).astype(float)
    eval_copy['payer_code_measured'] = (eval_copy['payer_code'] != None).astype(float)
    extra_numeric.append('payer_code_measured')

    for d in (train_copy, eval_copy):
        d['discharge_to_snf'] = np.array([1.0 if x == '3' else 0.0 for x in d['discharge_disposition_id']])
        d['discharge_to_home'] = np.array([1.0 if x == '1' else 0.0 for x in d['discharge_disposition_id']])
        d['discharge_to_home_health'] = np.array([1.0 if x == '6' else 0.0 for x in d['discharge_disposition_id']])
        d['discharge_high_risk'] = np.array([1.0 if x in ('3', '5', '22', '25') else 0.0 for x in d['discharge_disposition_id']])
        d['admission_from_emergency'] = np.array([1.0 if x == '7' else 0.0 for x in d['admission_source_id']])

    extra_numeric.extend([
        'discharge_to_snf', 'discharge_to_home', 'discharge_to_home_health',
        'discharge_high_risk', 'admission_from_emergency'
    ])

    # 12. Out-of-Fold Regularized Target Encoding (Essential non-redundant features)
    te_target_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id',
        'payer_code'
    ]
    for col in te_target_cols:
        te_train, te_eval = target_encode(train_copy[col], y_train, eval_copy[col], alpha=15.0)
        new_col_name = f'te_{col}'
        train_copy[new_col_name] = te_train
        eval_copy[new_col_name] = te_eval
        extra_numeric.append(new_col_name)

    # Encode with TabularEncoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_copy,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical)
    )
    x_eval = encoder.transform(eval_copy)

    # 13. Heterogeneous Ensemble: 5-Fold Stratified CV with LightGBM & XGBoost
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    p_eval_lgb1 = np.zeros(len(x_eval))
    p_eval_lgb2 = np.zeros(len(x_eval))
    p_eval_xgb1 = np.zeros(len(x_eval))

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        X_tr, y_tr = x_train[train_idx], y_train[train_idx]

        # LightGBM Model 1: Balanced leaf-wise learner with native categorical handling
        lgb_model1 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=28,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=3.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model1.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)

        # LightGBM Model 2: Shallow regularized learner for robust main effects
        lgb_model2 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=16,
            min_child_samples=30,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=4.0,
            random_state=123 + fold,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model2.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)

        # XGBoost: Depth-wise histogram booster providing orthogonal inductive bias
        xgb_model1 = xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=4.0,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=3.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        xgb_model1.fit(X_tr, y_tr)

        # Accumulate predictions
        p_eval_lgb1 += lgb_model1.predict_proba(x_eval)[:, 1] / 5.0
        p_eval_lgb2 += lgb_model2.predict_proba(x_eval)[:, 1] / 5.0
        p_eval_xgb1 += xgb_model1.predict_proba(x_eval)[:, 1] / 5.0

    preds = 0.40 * p_eval_lgb1 + 0.25 * p_eval_lgb2 + 0.35 * p_eval_xgb1
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
