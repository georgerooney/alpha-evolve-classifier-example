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
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    
    # 1. Initialize lists for new features
    extra_numeric = []
    extra_categorical = []
    
    # Make shallow copies to avoid modifying caller's dict keys
    train = dict(train)
    eval_cols = dict(eval_cols)
    
    # 2. Numeric age representation (moved early)
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = age_str.replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')
    
    # 3. ICD-9 3-digit prefix extraction (gold standard clinical disease abstraction)
    def get_icd9_prefix(code):
        if code is None or code == '?':
            return None
        c = str(code).strip()
        if not c or c == 'None':
            return None
        return c.split('.')[0]

    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        pref_col = col + "_3digit"
        train[pref_col] = np.array([get_icd9_prefix(x) for x in train[col]], dtype=object)
        eval_cols[pref_col] = np.array([get_icd9_prefix(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(pref_col)

    # Specific High-Risk Inpatient Readmission Clinical Flags (CMS / HRRP targets)
    for d in (train, eval_cols):
        p1 = d['diag_1_3digit']
        p2 = d['diag_2_3digit']
        p3 = d['diag_3_3digit']
        d['diag1_chf'] = (p1 == '428').astype(np.float64)
        d['diag1_copd'] = np.isin(p1, ['491', '492', '494', '496']).astype(np.float64)
        d['diag1_pna'] = np.isin(p1, ['480', '481', '482', '485', '486']).astype(np.float64)
        d['diag1_cad'] = np.isin(p1, ['410', '411', '414']).astype(np.float64)
        d['diag1_diabetes'] = (p1 == '250').astype(np.float64)
        d['diag1_renal'] = np.isin(p1, ['584', '585']).astype(np.float64)
        d['diag1_infection'] = np.isin(p1, ['038', '599', '682', '486']).astype(np.float64)
        d['diag1_electrolyte'] = (p1 == '276').astype(np.float64)

        # Multi-diagnosis flags across all three diagnoses
        d['has_dysrhythmia'] = ((p1 == '427') | (p2 == '427') | (p3 == '427')).astype(np.float64)
        d['has_electrolyte'] = ((p1 == '276') | (p2 == '276') | (p3 == '276')).astype(np.float64)
        d['has_aki'] = ((p1 == '584') | (p2 == '584') | (p3 == '584')).astype(np.float64)
        d['has_ckd'] = ((p1 == '585') | (p2 == '585') | (p3 == '585')).astype(np.float64)
        d['has_uti'] = ((p1 == '599') | (p2 == '599') | (p3 == '599')).astype(np.float64)
        d['has_chf'] = ((p1 == '428') | (p2 == '428') | (p3 == '428')).astype(np.float64)
        d['has_copd'] = (np.isin(p1, ['491', '492', '494', '496']) | np.isin(p2, ['491', '492', '494', '496']) | np.isin(p3, ['491', '492', '494', '496'])).astype(np.float64)
        d['has_pna'] = (np.isin(p1, ['480', '481', '482', '485', '486']) | np.isin(p2, ['480', '481', '482', '485', '486']) | np.isin(p3, ['480', '481', '482', '485', '486'])).astype(np.float64)
        d['has_sepsis'] = ((p1 == '038') | (p2 == '038') | (p3 == '038')).astype(np.float64)
        d['has_gi_bleed'] = ((p1 == '578') | (p2 == '578') | (p3 == '578')).astype(np.float64)
        d['has_surgical_comp'] = (np.isin(p1, ['996', '997', '998', '999']) | np.isin(p2, ['996', '997', '998', '999']) | np.isin(p3, ['996', '997', '998', '999'])).astype(np.float64)
        d['has_ulcer_gangrene'] = (np.isin(p1, ['707', '682', '040', '730']) | np.isin(p2, ['707', '682', '040', '730']) | np.isin(p3, ['707', '682', '040', '730'])).astype(np.float64)
        d['has_cva'] = (np.isin(p1, ['431', '433', '434', '435', '436']) | np.isin(p2, ['431', '433', '434', '435', '436']) | np.isin(p3, ['431', '433', '434', '435', '436'])).astype(np.float64)
        d['has_htn_comp'] = (np.isin(p1, ['402', '403', '404', '405']) | np.isin(p2, ['402', '403', '404', '405']) | np.isin(p3, ['402', '403', '404', '405'])).astype(np.float64)

    extra_numeric.extend([
        'diag1_chf', 'diag1_copd', 'diag1_pna', 'diag1_cad', 'diag1_diabetes',
        'diag1_renal', 'diag1_infection', 'diag1_electrolyte',
        'has_dysrhythmia', 'has_electrolyte', 'has_aki', 'has_ckd', 'has_uti',
        'has_chf', 'has_copd', 'has_pna', 'has_sepsis', 'has_gi_bleed',
        'has_surgical_comp', 'has_ulcer_gangrene', 'has_cva', 'has_htn_comp'
    ])

    # 3a. Fine-grained diabetes complication and acuity flags from ICD-9 decimal digits
    def extract_detailed_dm_flags(d, n_rows):
        dm_type1 = np.zeros(n_rows, dtype=np.float64)
        dm_type2 = np.zeros(n_rows, dtype=np.float64)
        dm_uncontrolled = np.zeros(n_rows, dtype=np.float64)
        dm_acute_crisis = np.zeros(n_rows, dtype=np.float64)
        dm_renal = np.zeros(n_rows, dtype=np.float64)
        dm_neuro = np.zeros(n_rows, dtype=np.float64)
        dm_circ_ulcer = np.zeros(n_rows, dtype=np.float64)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in d:
                arr = d[col]
                for i in range(n_rows):
                    val = arr[i]
                    if val is not None and str(val).startswith('250'):
                        c_str = str(val).strip()
                        if '.' in c_str:
                            sub = c_str.split('.')[1]
                            d4 = sub[0] if len(sub) > 0 else '0'
                            d5 = sub[1] if len(sub) > 1 else '0'
                        else:
                            d4 = '0'
                            d5 = '0'
                        if d5 in ('1', '3'): dm_type1[i] = 1.0
                        if d5 in ('0', '2'): dm_type2[i] = 1.0
                        if d5 in ('2', '3'): dm_uncontrolled[i] = 1.0
                        if d4 in ('1', '2', '3'): dm_acute_crisis[i] = 1.0
                        if d4 == '4': dm_renal[i] = 1.0
                        if d4 == '6': dm_neuro[i] = 1.0
                        if d4 in ('7', '8'): dm_circ_ulcer[i] = 1.0
        return {
            'dm_type1': dm_type1, 'dm_type2': dm_type2, 'dm_uncontrolled': dm_uncontrolled,
            'dm_acute_crisis': dm_acute_crisis, 'dm_renal': dm_renal, 'dm_neuro': dm_neuro,
            'dm_circ_ulcer': dm_circ_ulcer
        }

    dm_train = extract_detailed_dm_flags(train, len(y_train))
    dm_eval = extract_detailed_dm_flags(eval_cols, len(eval_cols['insulin']))
    for k in dm_train:
        train[k] = dm_train[k]
        eval_cols[k] = dm_eval[k]
        extra_numeric.append(k)

    # 3b. Extract comorbidities using icd9_comorbidities
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # 4. Grouped ICD-9 chapters & combinatorial groupings
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

    train['diag_12_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag_12_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag_12_grp')

    train['diag_13_grp'] = np.array([f"{g1}_{g3}" for g1, g3 in zip(train['diag_1_grp'], train['diag_3_grp'])], dtype=object)
    eval_cols['diag_13_grp'] = np.array([f"{g1}_{g3}" for g1, g3 in zip(eval_cols['diag_1_grp'], eval_cols['diag_3_grp'])], dtype=object)
    extra_categorical.append('diag_13_grp')

    train['diag_23_grp'] = np.array([f"{g2}_{g3}" for g2, g3 in zip(train['diag_2_grp'], train['diag_3_grp'])], dtype=object)
    eval_cols['diag_23_grp'] = np.array([f"{g2}_{g3}" for g2, g3 in zip(eval_cols['diag_2_grp'], eval_cols['diag_3_grp'])], dtype=object)
    extra_categorical.append('diag_23_grp')

    # 5. High-order categorical interactions
    train['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch')

    train['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admsrc_disch')

    train['admtype_disch'] = np.array([f"{t}_{disp}" for t, disp in zip(train['admission_type_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admtype_disch'] = np.array([f"{t}_{disp}" for t, disp in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admtype_disch')

    train['disch_spec'] = np.array([f"{disp}_{spec}" for disp, spec in zip(train['discharge_disposition_id'], train['medical_specialty'])], dtype=object)
    eval_cols['disch_spec'] = np.array([f"{disp}_{spec}" for disp, spec in zip(eval_cols['discharge_disposition_id'], eval_cols['medical_specialty'])], dtype=object)
    extra_categorical.append('disch_spec')

    train['adm_spec'] = np.array([f"{adm}_{spec}" for adm, spec in zip(train['admission_source_id'], train['medical_specialty'])], dtype=object)
    eval_cols['adm_spec'] = np.array([f"{adm}_{spec}" for adm, spec in zip(eval_cols['admission_source_id'], eval_cols['medical_specialty'])], dtype=object)
    extra_categorical.append('adm_spec')

    train['disch_age'] = np.array([f"{disp}_{age}" for disp, age in zip(train['discharge_disposition_id'], train['age'])], dtype=object)
    eval_cols['disch_age'] = np.array([f"{disp}_{age}" for disp, age in zip(eval_cols['discharge_disposition_id'], eval_cols['age'])], dtype=object)
    extra_categorical.append('disch_age')

    train['diag1_3digit_disch'] = np.array([f"{p}_{disp}" for p, disp in zip(train['diag_1_3digit'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_3digit_disch'] = np.array([f"{p}_{disp}" for p, disp in zip(eval_cols['diag_1_3digit'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_3digit_disch')

    train['payer_disch'] = np.array([f"{p}_{disp}" for p, disp in zip(train['payer_code'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['payer_disch'] = np.array([f"{p}_{disp}" for p, disp in zip(eval_cols['payer_code'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('payer_disch')

    train['admtype_admsrc'] = np.array([f"{t}_{s}" for t, s in zip(train['admission_type_id'], train['admission_source_id'])], dtype=object)
    eval_cols['admtype_admsrc'] = np.array([f"{t}_{s}" for t, s in zip(eval_cols['admission_type_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('admtype_admsrc')

    # Age and Admission crosses with primary diagnosis
    train['diag1_age'] = np.array([f"{g}_{a}" for g, a in zip(train['diag_1_grp'], train['age'])], dtype=object)
    eval_cols['diag1_age'] = np.array([f"{g}_{a}" for g, a in zip(eval_cols['diag_1_grp'], eval_cols['age'])], dtype=object)
    extra_categorical.append('diag1_age')

    train['diag1_admtype'] = np.array([f"{g}_{t}" for g, t in zip(train['diag_1_grp'], train['admission_type_id'])], dtype=object)
    eval_cols['diag1_admtype'] = np.array([f"{g}_{t}" for g, t in zip(eval_cols['diag_1_grp'], eval_cols['admission_type_id'])], dtype=object)
    extra_categorical.append('diag1_admtype')

    def make_inpatient_bucket(arr):
        return np.array(['0' if x == 0 else ('1' if x == 1 else ('2' if x == 2 else '3+')) for x in arr], dtype=object)

    train_inp_b = make_inpatient_bucket(train['number_inpatient'])
    eval_inp_b = make_inpatient_bucket(eval_cols['number_inpatient'])
    train['inpatient_disch'] = np.array([f"{b}_{disp}" for b, disp in zip(train_inp_b, train['discharge_disposition_id'])], dtype=object)
    eval_cols['inpatient_disch'] = np.array([f"{b}_{disp}" for b, disp in zip(eval_inp_b, eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('inpatient_disch')

    # Frequency / Count encodings for high cardinality categorical features
    freq_cols = ['medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'diag_1_3digit']
    n_tr_rows = len(y_train)
    for col in freq_cols:
        counts = {}
        for val in train[col]:
            counts[val] = counts.get(val, 0) + 1
        f_col = col + "_freq"
        train[f_col] = np.array([counts.get(x, 0) / float(n_tr_rows) for x in train[col]], dtype=np.float64)
        eval_cols[f_col] = np.array([counts.get(x, 0) / float(n_tr_rows) for x in eval_cols[col]], dtype=np.float64)
        extra_numeric.append(f_col)

    # 6. Out-of-fold target encoding for high-cardinality features
    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_3digit', 'diag_2_3digit', 'diag_3_3digit',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code',
        'admission_type_id', 'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag1_disch', 'diag1_3digit_disch', 'admsrc_disch',
        'diag_12_grp', 'diag_13_grp', 'diag_23_grp', 'admtype_disch', 'disch_spec', 'adm_spec', 'disch_age',
        'payer_disch', 'admtype_admsrc', 'diag1_age', 'diag1_admtype', 'inpatient_disch', 'race', 'gender', 'insulin_change'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 7. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')
    
    # 8. Medication active, change, up, down, and stable counts
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    train_active_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_active_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_change_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_change_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_up_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_up_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_down_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_down_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_steady_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_steady_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            train_up_meds += (train[col] == "Up")
            eval_up_meds += (eval_cols[col] == "Up")
            train_down_meds += (train[col] == "Down")
            eval_down_meds += (eval_cols[col] == "Down")
            train_steady_meds += (train[col] == "Steady")
            eval_steady_meds += (eval_cols[col] == "Steady")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')

    train['num_up_meds'] = train_up_meds
    eval_cols['num_up_meds'] = eval_up_meds
    extra_numeric.append('num_up_meds')

    train['num_down_meds'] = train_down_meds
    eval_cols['num_down_meds'] = eval_down_meds
    extra_numeric.append('num_down_meds')

    train['num_steady_meds'] = train_steady_meds
    eval_cols['num_steady_meds'] = eval_steady_meds
    extra_numeric.append('num_steady_meds')

    # Oral diabetes medications count and clinical regimen complexity
    oral_med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'glyburide-metformin', 'glipizide-metformin'
    ]
    train_oral_count = np.zeros(len(y_train), dtype=np.float64)
    eval_oral_count = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    for col in oral_med_names:
        if col in train:
            train_oral_count += (train[col] != "No") & (train[col] != None)
            eval_oral_count += (eval_cols[col] != "No") & (eval_cols[col] != None)
    train['num_oral_meds'] = train_oral_count
    eval_cols['num_oral_meds'] = eval_oral_count
    extra_numeric.append('num_oral_meds')

    train_has_ins = (train['insulin'] != 'No').astype(np.float64)
    eval_has_ins = (eval_cols['insulin'] != 'No').astype(np.float64)
    train['on_insulin_and_oral'] = ((train_has_ins == 1.0) & (train_oral_count >= 1.0)).astype(np.float64)
    eval_cols['on_insulin_and_oral'] = ((eval_has_ins == 1.0) & (eval_oral_count >= 1.0)).astype(np.float64)
    train['on_insulin_only'] = ((train_has_ins == 1.0) & (train_oral_count == 0.0)).astype(np.float64)
    eval_cols['on_insulin_only'] = ((eval_has_ins == 1.0) & (eval_oral_count == 0.0)).astype(np.float64)
    extra_numeric.extend(['on_insulin_and_oral', 'on_insulin_only'])

    su_meds = ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
    train_has_su = np.zeros(len(y_train), dtype=np.float64)
    eval_has_su = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    for col in su_meds:
        if col in train:
            train_has_su = np.maximum(train_has_su, (train[col] != "No").astype(np.float64))
            eval_has_su = np.maximum(eval_has_su, (eval_cols[col] != "No").astype(np.float64))
    train['su_renal_risk'] = train_has_su * np.maximum(train_cm['cm_renal'], train['has_aki'])
    eval_cols['su_renal_risk'] = eval_has_su * np.maximum(eval_cm['cm_renal'], eval_cols['has_aki'])
    train['metformin_renal_risk'] = (train['metformin'] != 'No').astype(np.float64) * np.maximum(train_cm['cm_renal'], train['has_aki'])
    eval_cols['metformin_renal_risk'] = (eval_cols['metformin'] != 'No').astype(np.float64) * np.maximum(eval_cm['cm_renal'], eval_cols['has_aki'])
    extra_numeric.extend(['su_renal_risk', 'metformin_renal_risk'])

    train['regimen_complexity_score'] = train_active_meds + 2.0 * train_change_meds + 3.0 * train_has_ins
    eval_cols['regimen_complexity_score'] = eval_active_meds + 2.0 * eval_change_meds + 3.0 * eval_has_ins
    train['su_insulin_combo'] = train_has_su * train_has_ins
    eval_cols['su_insulin_combo'] = eval_has_su * eval_has_ins
    train['regimen_instability'] = ((train_up_meds > 0) & (train_down_meds > 0)).astype(np.float64)
    eval_cols['regimen_instability'] = ((eval_up_meds > 0) & (eval_down_meds > 0)).astype(np.float64)
    extra_numeric.extend(['regimen_complexity_score', 'su_insulin_combo', 'regimen_instability'])
    
    # 9. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    # 10. Comorbidity weighted Charlson proxy and syndromic interactions
    train_charlson = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] +
        train_cm['cm_peptic_ulcer'] + train_cm['cm_liver'] +
        2.0 * (train_cm['cm_hemiplegia'] + train_cm['cm_renal'] +
               train_cm['cm_diabetes_complicated'] + train_cm['cm_cancer']) +
        4.0 * (train_cm['cm_metastatic'] + train_cm['cm_hiv'])
    )
    eval_charlson = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] +
        eval_cm['cm_peptic_ulcer'] + eval_cm['cm_liver'] +
        2.0 * (eval_cm['cm_hemiplegia'] + eval_cm['cm_renal'] +
               eval_cm['cm_diabetes_complicated'] + eval_cm['cm_cancer']) +
        4.0 * (eval_cm['cm_metastatic'] + eval_cm['cm_hiv'])
    )
    train['charlson_proxy'] = train_charlson
    eval_cols['charlson_proxy'] = eval_charlson
    extra_numeric.append('charlson_proxy')

    for d, cm in [(train, train_cm), (eval_cols, eval_cm)]:
        d['cm_chf_renal'] = cm['cm_chf'] * cm['cm_renal']
        d['cm_chf_dm'] = cm['cm_chf'] * cm['cm_diabetes_complicated']
        d['cm_renal_dm'] = cm['cm_renal'] * cm['cm_diabetes_complicated']
        d['cm_pulmonary_chf'] = cm['cm_pulmonary'] * cm['cm_chf']
    extra_numeric.extend(['cm_chf_renal', 'cm_chf_dm', 'cm_renal_dm', 'cm_pulmonary_chf'])

    # 11. Prior utilization details & acute care ratios
    for d in (train, eval_cols):
        tot = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['has_inpatient'] = (d['number_inpatient'] > 0).astype(np.float64)
        d['has_emergency'] = (d['number_emergency'] > 0).astype(np.float64)
        d['has_outpatient'] = (d['number_outpatient'] > 0).astype(np.float64)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(np.float64)
        d['emergency_ratio'] = d['number_emergency'] / (tot + 1.0)
        d['acute_ratio'] = (d['number_inpatient'] + d['number_emergency']) / (tot + 1.0)
        d['weighted_prior_visits'] = 3.0 * d['number_inpatient'] + 2.0 * d['number_emergency'] + 1.0 * d['number_outpatient']
        d['log_inpatient'] = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_total_visits'] = np.log1p(np.maximum(0.0, tot))
        
        # Risk ratio and age-comorbidity interactions
        d['emergency_to_inpatient_ratio'] = d['number_emergency'] / (d['number_inpatient'] + 1.0)
        d['outpatient_to_inpatient_ratio'] = d['number_outpatient'] / (d['number_inpatient'] + 1.0)
        d['age_comorbidity_interaction'] = d['age_num'] * d['charlson_proxy']
        d['age_visits_interaction'] = d['age_num'] * d['log_total_visits']
        d['age_over_65'] = (d['age_num'] >= 65.0).astype(np.float64)
        d['age_over_75'] = (d['age_num'] >= 75.0).astype(np.float64)
        d['high_utilizer'] = ((d['number_inpatient'] >= 2) | (tot >= 5)).astype(np.float64)
        
        # High-order inpatient crosses
        d['inpatient_sq'] = d['number_inpatient'] ** 2
        d['inpatient_gte_3'] = (d['number_inpatient'] >= 3).astype(np.float64)
        d['inpatient_x_los'] = d['number_inpatient'] * d['time_in_hospital']
        d['inpatient_x_charlson'] = d['number_inpatient'] * d['charlson_proxy']
        d['inpatient_x_emergency'] = d['number_inpatient'] * d['number_emergency']
        d['inpatient_x_age'] = d['number_inpatient'] * d['age_num']
        d['inpatient_x_meds'] = d['number_inpatient'] * d['num_medications']
        d['acute_visits_per_los'] = (d['number_inpatient'] + d['number_emergency']) / np.maximum(1.0, d['time_in_hospital'])

    extra_numeric.extend([
        'has_inpatient', 'has_emergency', 'has_outpatient', 'frequent_inpatient',
        'emergency_ratio', 'acute_ratio', 'weighted_prior_visits',
        'log_inpatient', 'log_emergency', 'log_total_visits',
        'emergency_to_inpatient_ratio', 'outpatient_to_inpatient_ratio',
        'age_comorbidity_interaction', 'age_visits_interaction',
        'age_over_65', 'age_over_75', 'high_utilizer',
        'inpatient_sq', 'inpatient_gte_3', 'inpatient_x_los', 'inpatient_x_charlson',
        'inpatient_x_emergency', 'inpatient_x_age', 'inpatient_x_meds', 'acute_visits_per_los'
    ])

    # 12. Hospital stay intensity & procedures ratios
    for d in (train, eval_cols):
        t_hosp = np.maximum(1.0, d['time_in_hospital'])
        d['labs_per_day'] = d['num_lab_procedures'] / t_hosp
        d['meds_per_day'] = d['num_medications'] / t_hosp
        d['procs_per_day'] = d['num_procedures'] / t_hosp
        d['diagnoses_per_day'] = d['number_diagnoses'] / t_hosp
        d['total_procedures'] = d['num_lab_procedures'] + d['num_procedures']
        d['meds_per_diag'] = d['num_medications'] / (d['number_diagnoses'] + 1.0)
        d['procs_per_med'] = d['total_procedures'] / (d['num_medications'] + 1.0)
        d['long_stay'] = (d['time_in_hospital'] >= 7).astype(np.float64)
    extra_numeric.extend([
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'diagnoses_per_day',
        'total_procedures', 'meds_per_diag', 'procs_per_med', 'long_stay'
    ])

    # 13. Discharge and admission indicators
    for d in (train, eval_cols):
        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = (disp == '3').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_transfer'] = np.isin(disp, ['2', '3', '4', '5', '22']).astype(np.float64)
        d['disch_revisit_inpatient'] = (disp == '18').astype(np.float64)

        adm_typ = d['admission_type_id']
        adm_src = d['admission_source_id']
        d['adm_is_emergency'] = np.isin(adm_typ, ['1', '2']).astype(np.float64)
        d['adm_from_er'] = (adm_src == '7').astype(np.float64)
        d['adm_high_acuity'] = (d['adm_is_emergency'] * d['adm_from_er']).astype(np.float64)
        d['adm_from_transfer'] = np.isin(adm_src, ['4', '5', '6']).astype(np.float64)
        d['adm_elective'] = (adm_typ == '3').astype(np.float64)
        d['snf_prior_inpatient'] = d['disch_snf'] * d['has_inpatient']
        d['homehealth_prior_inpatient'] = d['disch_home_health'] * d['has_inpatient']
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'disch_revisit_inpatient', 'adm_is_emergency', 'adm_from_er', 'adm_high_acuity',
        'adm_from_transfer', 'adm_elective', 'snf_prior_inpatient', 'homehealth_prior_inpatient'
    ])

    # 14. LACE Index bedside score proxy & clinical severity score
    def calc_lace(los, is_er, charlson, ed_visits):
        l_score = np.zeros_like(los, dtype=np.float64)
        l_score[los == 1] = 1.0
        l_score[los == 2] = 2.0
        l_score[los == 3] = 3.0
        l_score[(los >= 4) & (los <= 6)] = 4.0
        l_score[(los >= 7) & (los <= 13)] = 5.0
        l_score[los >= 14] = 7.0
        a_score = 3.0 * is_er
        c_score = np.zeros_like(charlson, dtype=np.float64)
        c_score[charlson == 1] = 1.0
        c_score[charlson == 2] = 2.0
        c_score[charlson == 3] = 3.0
        c_score[charlson >= 4] = 5.0
        e_score = np.clip(ed_visits, 0.0, 4.0)
        return l_score + a_score + c_score + e_score

    train['lace_score'] = calc_lace(train['time_in_hospital'], train['adm_is_emergency'], train_charlson, train['number_emergency'])
    eval_cols['lace_score'] = calc_lace(eval_cols['time_in_hospital'], eval_cols['adm_is_emergency'], eval_charlson, eval_cols['number_emergency'])
    train['lace_high_risk'] = (train['lace_score'] >= 10.0).astype(np.float64)
    eval_cols['lace_high_risk'] = (eval_cols['lace_score'] >= 10.0).astype(np.float64)
    extra_numeric.extend(['lace_score', 'lace_high_risk'])

    for d in (train, eval_cols):
        d['clinical_severity_score'] = (
            d['lace_score'] * 1.5 +
            d['inpatient_ratio'] * 10.0 +
            d['charlson_proxy'] * 2.0 +
            d['log_total_visits'] * 3.0 +
            (d['time_in_hospital'] * d['num_medications'] / 100.0)
        )
    extra_numeric.append('clinical_severity_score')

    # 15. Lab test indicators & Glycemic control clinical inertia
    for d in (train, eval_cols):
        a1c = d['A1Cresult']
        glu = d['max_glu_serum']
        chg = d['change']

        a1c_tested = (a1c != 'None').astype(np.float64)
        a1c_high = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
        glu_tested = (glu != 'None').astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)
        
        a1c_normal = (a1c == 'Norm').astype(np.float64)
        glu_normal = (glu == 'Norm').astype(np.float64)

        d['a1c_tested'] = a1c_tested
        d['a1c_high'] = a1c_high
        d['glu_tested'] = glu_tested
        d['glu_high'] = glu_high
        d['a1c_normal'] = a1c_normal
        d['glu_normal'] = glu_normal
        d['good_glycemic_control'] = ((a1c_normal == 1.0) | (glu_normal == 1.0)).astype(np.float64)
        d['diabetes_uncontrolled'] = ((a1c_high == 1.0) | (glu_high == 1.0)).astype(np.float64)
        d['high_a1c_no_med_change'] = ((a1c_high == 1.0) & (chg == 'No')).astype(np.float64)
        d['high_a1c_med_change'] = ((a1c_high == 1.0) & (chg == 'Ch')).astype(np.float64)
        d['a1c_tested_and_med_change'] = ((a1c_tested == 1.0) & (chg == 'Ch')).astype(np.float64)
        d['a1c_tested_no_med_change'] = ((a1c_tested == 1.0) & (chg == 'No')).astype(np.float64)
        
        # Ordinal encoding of lab severity
        d['a1c_ordinal'] = np.array([0.0 if x is None or x == 'None' or x == '?' else (1.0 if x == 'Norm' else (2.0 if x == '>7' else 3.0)) for x in a1c], dtype=np.float64)
        d['glu_ordinal'] = np.array([0.0 if x is None or x == 'None' or x == '?' else (1.0 if x == 'Norm' else (2.0 if x == '>200' else 3.0)) for x in glu], dtype=np.float64)

    extra_numeric.extend([
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high', 'a1c_normal', 'glu_normal',
        'good_glycemic_control', 'diabetes_uncontrolled', 'high_a1c_no_med_change', 'high_a1c_med_change',
        'a1c_tested_and_med_change', 'a1c_tested_no_med_change', 'a1c_ordinal', 'glu_ordinal'
    ])

    # 16. Medication directionality & polypharmacy
    for d in (train, eval_cols):
        ins = d['insulin']
        d['on_insulin'] = (ins != 'No').astype(np.float64)
        d['insulin_up'] = (ins == 'Up').astype(np.float64)
        d['insulin_down'] = (ins == 'Down').astype(np.float64)
        d['insulin_steady'] = (ins == 'Steady').astype(np.float64)
        d['insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        d['on_metformin'] = (d['metformin'] != 'No').astype(np.float64)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(np.float64)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(np.float64)
        d['med_change_flag'] = (d['change'] == 'Ch').astype(np.float64)
        
        # Oral meds
        d['on_glipizide'] = (d['glipizide'] != 'No').astype(np.float64)
        d['on_glyburide'] = (d['glyburide'] != 'No').astype(np.float64)
        d['on_pioglitazone'] = (d['pioglitazone'] != 'No').astype(np.float64)
        d['on_rosiglitazone'] = (d['rosiglitazone'] != 'No').astype(np.float64)
        
        # Elderly polypharmacy risk
        d['elderly_polypharmacy_change'] = ((d['age_num'] >= 70.0) & (d['polypharmacy'] == 1.0) & (d['med_change_flag'] == 1.0)).astype(np.float64)

    extra_numeric.extend([
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'on_metformin', 'polypharmacy', 'diabetes_med_flag', 'med_change_flag',
        'on_glipizide', 'on_glyburide', 'on_pioglitazone', 'on_rosiglitazone',
        'elderly_polypharmacy_change'
    ])

    # 17. Missingness, Payer and Specialty Domain Features
    for d in (train, eval_cols):
        d['missing_specialty'] = (d['medical_specialty'] == None).astype(np.float64)
        d['missing_payer'] = (d['payer_code'] == None).astype(np.float64)
        d['has_weight'] = ((d['weight'] != None) & (d['weight'] != '?')).astype(np.float64)
        
        p_code = d['payer_code']
        d['is_medicare'] = (p_code == 'MC').astype(np.float64)
        d['is_medicaid'] = (p_code == 'MD').astype(np.float64)
        d['is_commercial'] = np.isin(p_code, ['BC', 'HM', 'CP', 'CI']).astype(np.float64)
        d['is_self_pay'] = np.isin(p_code, ['SP', 'UN']).astype(np.float64)

        spec = d['medical_specialty']
        d['spec_cardio'] = np.isin(spec, ['Cardiology', 'Cardiology-Pediatric']).astype(np.float64)
        d['spec_internal'] = (spec == 'InternalMedicine').astype(np.float64)
        d['spec_emergency'] = (spec == 'Emergency/Trauma').astype(np.float64)
        d['spec_family'] = (spec == 'Family/GeneralPractice').astype(np.float64)
        d['spec_nephro'] = (spec == 'Nephrology').astype(np.float64)
        d['spec_surgery'] = np.array([1.0 if s is not None and 'Surgery' in str(s) else 0.0 for s in spec], dtype=np.float64)

    extra_numeric.extend([
        'missing_specialty', 'missing_payer', 'has_weight',
        'is_medicare', 'is_medicaid', 'is_commercial', 'is_self_pay',
        'spec_cardio', 'spec_internal', 'spec_emergency', 'spec_family', 'spec_nephro', 'spec_surgery'
    ])

    # 17.5 Latent Clinical Phenotypes and Treatment Regimens via SVD (Matrix Decomposition)
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    # 17.5.1 Latent Diagnosis Phenotypes (SVD on ICD-9 codes and 3-digit prefixes)
    diag_tokens = set()
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in train:
            for val in train[col]:
                if val is not None and val != '?':
                    diag_tokens.add(val)
                    pref = val.split('.')[0]
                    diag_tokens.add(f"pref_{pref}")
    diag_codes = sorted(list(diag_tokens))
    code_to_idx = {code: idx for idx, code in enumerate(diag_codes)}
    num_codes = len(diag_codes)

    def build_sparse_diag_matrix(data_dict, num_rows):
        rows = []
        cols = []
        vals = []
        for r_idx in range(num_rows):
            seen_in_row = set()
            for col_name in ['diag_1', 'diag_2', 'diag_3']:
                if col_name in data_dict:
                    val = data_dict[col_name][r_idx]
                    if val is not None and val != '?':
                        c_val = str(val).strip()
                        if c_val in code_to_idx:
                            c_idx = code_to_idx[c_val]
                            if c_idx not in seen_in_row:
                                rows.append(r_idx)
                                cols.append(c_idx)
                                vals.append(1.0)
                                seen_in_row.add(c_idx)
                        pref = f"pref_{c_val.split('.')[0]}"
                        if pref in code_to_idx:
                            p_idx = code_to_idx[pref]
                            if p_idx not in seen_in_row:
                                rows.append(r_idx)
                                cols.append(p_idx)
                                vals.append(1.0)
                                seen_in_row.add(p_idx)
        return csr_matrix((vals, (rows, cols)), shape=(num_rows, num_codes), dtype=np.float64)

    n_train_rows = len(y_train)
    n_eval_rows = len(eval_cols['insulin'])

    if num_codes > 0:
        sparse_train = build_sparse_diag_matrix(train, n_train_rows)
        sparse_eval = build_sparse_diag_matrix(eval_cols, n_eval_rows)

        n_diag_components = min(10, num_codes)
        svd_diag = TruncatedSVD(n_components=n_diag_components, random_state=42)
        train_diag_svd = svd_diag.fit_transform(sparse_train)
        eval_diag_svd = svd_diag.transform(sparse_eval)

        for i in range(n_diag_components):
            col_name = f'diag_svd_{i}'
            train[col_name] = train_diag_svd[:, i]
            eval_cols[col_name] = eval_diag_svd[:, i]
            extra_numeric.append(col_name)

    # 17.5.2 Latent Treatment Regimens (SVD on Medication space)
    med_mapping = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
    def build_med_matrix(data_dict, num_rows):
        mat = np.zeros((num_rows, len(med_cols)), dtype=np.float64)
        for c_idx, col in enumerate(med_cols):
            if col in data_dict:
                arr = data_dict[col]
                for r_idx in range(num_rows):
                    val = arr[r_idx]
                    mat[r_idx, c_idx] = med_mapping.get(val, 0)
        return mat

    med_train = build_med_matrix(train, n_train_rows)
    med_eval = build_med_matrix(eval_cols, n_eval_rows)

    n_med_components = min(6, len(med_cols))
    svd_med = TruncatedSVD(n_components=n_med_components, random_state=42)
    train_med_svd = svd_med.fit_transform(med_train)
    eval_med_svd = svd_med.transform(med_eval)

    for i in range(n_med_components):
        col_name = f'med_svd_{i}'
        train[col_name] = train_med_svd[:, i]
        eval_cols[col_name] = eval_med_svd[:, i]
        extra_numeric.append(col_name)

    # 17.6 "Patient-Like-Me" Non-Parametric Manifold KNN Risk Engine
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    knn_dim_cols = [
        'age_num', 'charlson_proxy', 'lace_score', 'clinical_severity_score',
        'log_total_visits', 'log_inpatient', 'inpatient_ratio',
        'time_in_hospital', 'num_medications', 'num_lab_procedures', 'num_procedures', 'number_diagnoses',
        'diag_svd_0', 'diag_svd_1', 'diag_svd_2', 'diag_svd_3',
        'med_svd_0', 'med_svd_1'
    ]
    avail_knn_cols = [c for c in knn_dim_cols if c in train and c in eval_cols]
    if len(avail_knn_cols) >= 5:
        X_knn_tr = np.zeros((n_train_rows, len(avail_knn_cols)), dtype=np.float64)
        X_knn_ev = np.zeros((n_eval_rows, len(avail_knn_cols)), dtype=np.float64)
        for idx, col in enumerate(avail_knn_cols):
            tr_v = train[col].astype(np.float64)
            ev_v = eval_cols[col].astype(np.float64)
            m_val = np.nanmean(tr_v)
            if np.isnan(m_val):
                m_val = 0.0
            X_knn_tr[:, idx] = np.nan_to_num(tr_v, nan=m_val)
            X_knn_ev[:, idx] = np.nan_to_num(ev_v, nan=m_val)

        knn_scaler = StandardScaler()
        X_knn_tr_s = knn_scaler.fit_transform(X_knn_tr)
        X_knn_ev_s = knn_scaler.transform(X_knn_ev)

        train_knn_risk_25 = np.zeros(n_train_rows, dtype=np.float64)
        train_knn_wrisk_25 = np.zeros(n_train_rows, dtype=np.float64)
        train_knn_risk_100 = np.zeros(n_train_rows, dtype=np.float64)
        train_knn_dist_25 = np.zeros(n_train_rows, dtype=np.float64)

        skf_knn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, va_idx in skf_knn.split(X_knn_tr_s, y_train):
            nn = NearestNeighbors(n_neighbors=100, algorithm='kd_tree', n_jobs=1)
            nn.fit(X_knn_tr_s[tr_idx])
            dists, inds = nn.kneighbors(X_knn_tr_s[va_idx])
            y_fold_knn = y_train[tr_idx][inds[:, :25]]
            train_knn_risk_25[va_idx] = np.mean(y_fold_knn, axis=1)
            train_knn_risk_100[va_idx] = np.mean(y_train[tr_idx][inds[:, :100]], axis=1)
            train_knn_dist_25[va_idx] = np.mean(dists[:, :25], axis=1)
            w_25 = 1.0 / (dists[:, :25] + 1.0)
            train_knn_wrisk_25[va_idx] = np.sum(w_25 * y_fold_knn, axis=1) / np.sum(w_25, axis=1)

        nn_full = NearestNeighbors(n_neighbors=100, algorithm='kd_tree', n_jobs=1)
        nn_full.fit(X_knn_tr_s)
        dists_ev, inds_ev = nn_full.kneighbors(X_knn_ev_s)
        y_ev_knn = y_train[inds_ev[:, :25]]
        eval_knn_risk_25 = np.mean(y_ev_knn, axis=1)
        eval_knn_risk_100 = np.mean(y_train[inds_ev[:, :100]], axis=1)
        eval_knn_dist_25 = np.mean(dists_ev[:, :25], axis=1)
        w_ev_25 = 1.0 / (dists_ev[:, :25] + 1.0)
        eval_knn_wrisk_25 = np.sum(w_ev_25 * y_ev_knn, axis=1) / np.sum(w_ev_25, axis=1)

        train['knn_risk_25'] = train_knn_risk_25
        eval_cols['knn_risk_25'] = eval_knn_risk_25
        train['knn_wrisk_25'] = train_knn_wrisk_25
        eval_cols['knn_wrisk_25'] = eval_knn_wrisk_25
        train['knn_risk_100'] = train_knn_risk_100
        eval_cols['knn_risk_100'] = eval_knn_risk_100
        train['knn_dist_25'] = train_knn_dist_25
        eval_cols['knn_dist_25'] = eval_knn_dist_25
        train['knn_risk_diff'] = train_knn_risk_25 - train_knn_risk_100
        eval_cols['knn_risk_diff'] = eval_knn_risk_25 - eval_knn_risk_100

        extra_numeric.extend(['knn_risk_25', 'knn_wrisk_25', 'knn_risk_100', 'knn_dist_25', 'knn_risk_diff'])

    # 18. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)

    # 19. Heterogeneous 4-Family 5-Fold Ensemble: Deep LGBM + Depth-Wise XGBoost + Shallow LGBM + Regularized Logistic Regression
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    from scipy.optimize import minimize

    # Filter out integer category codes for XGBoost and linear model
    xgb_feature_mask = np.ones(x_train.shape[1], dtype=bool)
    xgb_feature_mask[encoder.categorical_indices] = False
    x_train_xgb = x_train[:, xgb_feature_mask]
    x_eval_xgb = x_eval[:, xgb_feature_mask]

    # Pre-impute medians for numeric arrays for the linear model
    med_tr = np.nanmedian(x_train_xgb, axis=0)
    med_tr = np.nan_to_num(med_tr, nan=0.0)
    tr_xgb_imp = np.where(np.isnan(x_train_xgb), med_tr, x_train_xgb)
    ev_xgb_imp = np.where(np.isnan(x_eval_xgb), med_tr, x_eval_xgb)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds_lgb1 = np.zeros(x_eval.shape[0])
    eval_preds_xgb = np.zeros(x_eval.shape[0])
    eval_preds_lgb2 = np.zeros(x_eval.shape[0])
    eval_preds_lr = np.zeros(x_eval.shape[0])

    oof_lgb1 = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_lgb2 = np.zeros(len(y_train))
    oof_lr = np.zeros(len(y_train))

    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        fold_x_val, fold_y_val = x_train[val_idx], y_train[val_idx]

        # Model 1: LightGBM Deep Leaf-Wise (captures complex multi-way interactions)
        lgb1 = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.022,
            num_leaves=36,
            min_child_samples=80,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.3,
            reg_lambda=1.5,
            cat_smooth=25.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb1.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        oof_lgb1[val_idx] = lgb1.predict_proba(fold_x_val)[:, 1]
        eval_preds_lgb1 += lgb1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: XGBoost Depth-Wise Hist (independent tree construction & regularization)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.028,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=1.5,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(x_train_xgb[train_idx], fold_y_train)
        oof_xgb[val_idx] = xgb_model.predict_proba(x_train_xgb[val_idx])[:, 1]
        eval_preds_xgb += xgb_model.predict_proba(x_eval_xgb)[:, 1] / 5.0

        # Model 3: LightGBM Shallow Constrained (stabilizes main effects and low-order interactions)
        lgb2 = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.032,
            max_depth=4,
            num_leaves=14,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.0,
            cat_smooth=20.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        )
        lgb2.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        oof_lgb2[val_idx] = lgb2.predict_proba(fold_x_val)[:, 1]
        eval_preds_lgb2 += lgb2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 4: L2 Logistic Regression on standard-scaled features (linear baseline)
        f_scaler = StandardScaler()
        f_tr_lr = f_scaler.fit_transform(tr_xgb_imp[train_idx])
        f_va_lr = f_scaler.transform(tr_xgb_imp[val_idx])
        lr_model = LogisticRegression(C=0.03, max_iter=150, solver='lbfgs', random_state=42)
        lr_model.fit(f_tr_lr, fold_y_train)
        oof_lr[val_idx] = lr_model.predict_proba(f_va_lr)[:, 1]
        eval_preds_lr += lr_model.predict_proba(f_scaler.transform(ev_xgb_imp))[:, 1] / 5.0

    # 20. Multi-Start Simplex Nelder-Mead Optimization for Probability and Rank Ensembles
    n_tr = float(len(y_train))
    n_ev = float(x_eval.shape[0])

    oof_r1 = (rankdata(oof_lgb1) - 0.5) / n_tr
    oof_r2 = (rankdata(oof_xgb) - 0.5) / n_tr
    oof_r3 = (rankdata(oof_lgb2) - 0.5) / n_tr
    oof_r4 = (rankdata(oof_lr) - 0.5) / n_tr

    ev_r1 = (rankdata(eval_preds_lgb1) - 0.5) / n_ev
    ev_r2 = (rankdata(eval_preds_xgb) - 0.5) / n_ev
    ev_r3 = (rankdata(eval_preds_lgb2) - 0.5) / n_ev
    ev_r4 = (rankdata(eval_preds_lr) - 0.5) / n_ev

    oof_prob_matrix = np.column_stack([oof_lgb1, oof_xgb, oof_lgb2, oof_lr])
    oof_rank_matrix = np.column_stack([oof_r1, oof_r2, oof_r3, oof_r4])
    ev_prob_matrix = np.column_stack([eval_preds_lgb1, eval_preds_xgb, eval_preds_lgb2, eval_preds_lr])
    ev_rank_matrix = np.column_stack([ev_r1, ev_r2, ev_r3, ev_r4])

    best_score = -1.0
    best_mode = 'prob'
    best_w = np.array([0.4, 0.3, 0.2, 0.1])

    init_points = [
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.5, 0.25, 0.2, 0.05]),
        np.array([0.3, 0.3, 0.3, 0.1]),
        np.array([0.35, 0.35, 0.2, 0.1])
    ]

    for mode, mat in [('prob', oof_prob_matrix), ('rank', oof_rank_matrix)]:
        for i in range(4):
            sc = roc_auc_score(y_train, mat[:, i])
            if sc > best_score:
                best_score = sc
                best_mode = mode
                best_w = np.eye(4)[i]

        def obj(raw_w, m=mat):
            w = np.maximum(0.0, raw_w)
            s = np.sum(w)
            if s < 1e-7:
                return 0.0
            w = w / s
            preds = m @ w
            return -roc_auc_score(y_train, preds)

        for init_w in init_points:
            res = minimize(obj, init_w, method='Nelder-Mead', options={'maxiter': 100, 'disp': False})
            sc = -res.fun
            if sc > best_score:
                best_score = sc
                best_mode = mode
                w_opt = np.maximum(0.0, res.x)
                best_w = w_opt / np.sum(w_opt)

    if best_mode == 'rank':
        final_preds = ev_rank_matrix @ best_w
    else:
        final_preds = ev_prob_matrix @ best_w

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
