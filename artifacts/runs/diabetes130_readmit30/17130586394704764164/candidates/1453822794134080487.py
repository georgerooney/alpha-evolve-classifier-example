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
    
    # 3. Extract comorbidities using icd9_comorbidities
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # 4. Grouped ICD-9 chapters & 3-digit disease category truncation
    def get_icd9_cat(code):
        if code is None:
            return "missing"
        s = str(code).strip()
        if not s or s == "?":
            return "missing"
        return s.split(".")[0]

    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

        cat_col = col + "_cat"
        train[cat_col] = np.array([get_icd9_cat(x) for x in train[col]], dtype=object)
        eval_cols[cat_col] = np.array([get_icd9_cat(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(cat_col)

    train['diag_12_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag_12_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag_12_grp')

    train['diag_13_grp'] = np.array([f"{g1}_{g3}" for g1, g3 in zip(train['diag_1_grp'], train['diag_3_grp'])], dtype=object)
    eval_cols['diag_13_grp'] = np.array([f"{g1}_{g3}" for g1, g3 in zip(eval_cols['diag_1_grp'], eval_cols['diag_3_grp'])], dtype=object)
    extra_categorical.append('diag_13_grp')

    train['diag_12_cat'] = np.array([f"{c1}_{c2}" for c1, c2 in zip(train['diag_1_cat'], train['diag_2_cat'])], dtype=object)
    eval_cols['diag_12_cat'] = np.array([f"{c1}_{c2}" for c1, c2 in zip(eval_cols['diag_1_cat'], eval_cols['diag_2_cat'])], dtype=object)
    extra_categorical.append('diag_12_cat')

    # 5. Targeted clinical syndrome flags across diagnoses
    def extract_clinical_flags(d1_arr, d2_arr, d3_arr):
        n = len(d1_arr)
        aki = np.zeros(n, dtype=np.float64)
        fluid = np.zeros(n, dtype=np.float64)
        uti = np.zeros(n, dtype=np.float64)
        pneu = np.zeros(n, dtype=np.float64)
        cad = np.zeros(n, dtype=np.float64)
        htn = np.zeros(n, dtype=np.float64)
        sepsis = np.zeros(n, dtype=np.float64)
        dka = np.zeros(n, dtype=np.float64)
        pneu_pre = ('480', '481', '482', '483', '484', '485', '486')
        cad_pre = ('410', '411', '412', '413', '414')
        htn_pre = ('401', '402', '403', '404', '405')
        for i in range(n):
            for c in (d1_arr[i], d2_arr[i], d3_arr[i]):
                if c is None:
                    continue
                s = str(c).strip()
                if not s or s == '?':
                    continue
                if s.startswith('584'):
                    aki[i] = 1.0
                if s.startswith('276'):
                    fluid[i] = 1.0
                if s.startswith('599'):
                    uti[i] = 1.0
                if s.startswith(pneu_pre):
                    pneu[i] = 1.0
                if s.startswith(cad_pre):
                    cad[i] = 1.0
                if s.startswith(htn_pre):
                    htn[i] = 1.0
                if s.startswith(('038', '995.9')):
                    sepsis[i] = 1.0
                if s.startswith(('250.1', '250.2', '250.3')):
                    dka[i] = 1.0
        return {
            'flag_aki': aki,
            'flag_fluid_electrolyte': fluid,
            'flag_uti': uti,
            'flag_pneumonia': pneu,
            'flag_cad': cad,
            'flag_hypertension': htn,
            'flag_sepsis': sepsis,
            'flag_dka': dka,
        }

    train_flags = extract_clinical_flags(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_flags = extract_clinical_flags(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in train_flags:
        train[k] = train_flags[k]
        eval_cols[k] = eval_flags[k]
        extra_numeric.append(k)

    # 6. High-order categorical interactions
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

    train['diag1_age'] = np.array([f"{g}_{a}" for g, a in zip(train['diag_1_grp'], train['age'])], dtype=object)
    eval_cols['diag1_age'] = np.array([f"{g}_{a}" for g, a in zip(eval_cols['diag_1_grp'], eval_cols['age'])], dtype=object)
    extra_categorical.append('diag1_age')

    train['disch_inpatient'] = np.array([f"{disp}_{int(inp > 0)}" for disp, inp in zip(train['discharge_disposition_id'], train['number_inpatient'])], dtype=object)
    eval_cols['disch_inpatient'] = np.array([f"{disp}_{int(inp > 0)}" for disp, inp in zip(eval_cols['discharge_disposition_id'], eval_cols['number_inpatient'])], dtype=object)
    extra_categorical.append('disch_inpatient')

    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    # 7. Frequency encoding of high-cardinality features
    freq_cols = ['diag_1', 'diag_1_cat', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'diag_1_grp']
    for col in freq_cols:
        if col in train:
            vals, counts = np.unique([x for x in train[col] if x is not None], return_counts=True)
            freq_map = dict(zip(vals, counts / float(len(train[col]))))
            f_col = col + '_freq'
            train[f_col] = np.array([freq_map.get(x, 0.0) if x is not None else 0.0 for x in train[col]], dtype=np.float64)
            eval_cols[f_col] = np.array([freq_map.get(x, 0.0) if x is not None else 0.0 for x in eval_cols[col]], dtype=np.float64)
            extra_numeric.append(f_col)

    # 8. Target encoding with double seed-averaging
    def double_oof_target_encode(train_col, y_train, eval_col, n_folds=5, smoothing=20.0):
        tr1, ev1 = oof_target_encode(train_col, y_train, eval_col, n_folds=n_folds, smoothing=smoothing, seed=42)
        tr2, ev2 = oof_target_encode(train_col, y_train, eval_col, n_folds=n_folds, smoothing=smoothing, seed=2023)
        return 0.5 * (tr1 + tr2), 0.5 * (ev1 + ev2)

    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_cat', 'diag_2_cat', 'diag_3_cat', 'diag_12_cat',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code',
        'admission_type_id', 'diag_1_grp', 'diag1_disch', 'admsrc_disch', 'diag_12_grp', 'diag_13_grp',
        'admtype_disch', 'disch_spec', 'adm_spec', 'disch_age', 'insulin_change', 'diag1_age',
        'disch_inpatient', 'age', 'race'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = double_oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0)
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
    
    # 9. Sulfonylurea & complex diabetes medication combinations
    sulf_cols = ['chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'tolazamide']
    train_sulf = np.zeros(len(y_train), dtype=np.float64)
    eval_sulf = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    for c in sulf_cols:
        if c in train:
            train_sulf += ((train[c] != 'No') & (train[c] != None)).astype(np.float64)
            eval_sulf += ((eval_cols[c] != 'No') & (eval_cols[c] != None)).astype(np.float64)

    for d, s in [(train, train_sulf), (eval_cols, eval_sulf)]:
        d['sulfonylurea_count'] = s
        ins_flag = (d['insulin'] != 'No').astype(np.float64)
        d['hypo_risk_regimen'] = ((ins_flag == 1.0) & (s > 0.0)).astype(np.float64)
        oral_cnt = np.maximum(0.0, d['active_meds_count'] - ins_flag)
        d['oral_meds_count'] = oral_cnt
        d['complex_med_regimen'] = ((ins_flag == 1.0) & (oral_cnt > 0.0)).astype(np.float64)
        on_met = (d['metformin'] != 'No').astype(np.float64)
        d['on_metformin_only'] = ((on_met == 1.0) & (ins_flag == 0.0) & (oral_cnt == 1.0)).astype(np.float64)
        d['metformin_and_insulin'] = ((on_met == 1.0) & (ins_flag == 1.0)).astype(np.float64)
    extra_numeric.extend(['sulfonylurea_count', 'hypo_risk_regimen', 'oral_meds_count', 'complex_med_regimen', 'on_metformin_only', 'metformin_and_insulin'])

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

        d['no_prior_visits'] = (tot == 0).astype(np.float64)
        d['frequent_prior_visits'] = (tot >= 4).astype(np.float64)
        d['inpatient_times_stay'] = d['number_inpatient'] * d['time_in_hospital']
        d['inpatient_times_charlson'] = d['number_inpatient'] * d['charlson_proxy']
        d['inpatient_times_meds'] = d['number_inpatient'] * d['num_medications']
        d['emergency_times_inpatient'] = d['number_inpatient'] * d['number_emergency']
        d['inpatient_sq'] = d['number_inpatient'] ** 2
        d['age_times_stay'] = d['age_num'] * d['time_in_hospital']
        d['age_times_inpatient'] = d['age_num'] * d['number_inpatient']

    extra_numeric.extend([
        'has_inpatient', 'has_emergency', 'has_outpatient', 'frequent_inpatient',
        'emergency_ratio', 'acute_ratio', 'weighted_prior_visits',
        'log_inpatient', 'log_emergency', 'log_total_visits',
        'emergency_to_inpatient_ratio', 'outpatient_to_inpatient_ratio',
        'age_comorbidity_interaction', 'age_visits_interaction',
        'no_prior_visits', 'frequent_prior_visits',
        'inpatient_times_stay', 'inpatient_times_charlson', 'inpatient_times_meds',
        'emergency_times_inpatient', 'inpatient_sq',
        'age_times_stay', 'age_times_inpatient'
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
        d['lab_to_med_ratio'] = d['num_lab_procedures'] / (d['num_medications'] + 1.0)
        d['diag_to_med_ratio'] = d['number_diagnoses'] / (d['num_medications'] + 1.0)
        d['time_per_diagnosis'] = d['time_in_hospital'] / (d['number_diagnoses'] + 1.0)
        d['procedure_intensity'] = (d['num_procedures'] + d['num_lab_procedures']) / (t_hosp + 1.0)
    extra_numeric.extend([
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'diagnoses_per_day',
        'total_procedures', 'meds_per_diag', 'procs_per_med', 'long_stay',
        'lab_to_med_ratio', 'diag_to_med_ratio', 'time_per_diagnosis', 'procedure_intensity'
    ])

    # 13. Discharge and admission indicators
    for d in (train, eval_cols):
        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = (disp == '3').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_transfer'] = np.isin(disp, ['2', '3', '4', '5', '22']).astype(np.float64)

        adm_typ = d['admission_type_id']
        adm_src = d['admission_source_id']
        d['adm_is_emergency'] = np.isin(adm_typ, ['1', '2']).astype(np.float64)
        d['adm_is_elective'] = (adm_typ == '3').astype(np.float64)
        d['adm_from_er'] = (adm_src == '7').astype(np.float64)
        d['adm_from_referral'] = np.isin(adm_src, ['1', '2', '3']).astype(np.float64)
        d['adm_from_transfer'] = np.isin(adm_src, ['4', '5', '6', '10', '18', '22']).astype(np.float64)
        d['adm_high_acuity'] = (d['adm_is_emergency'] * d['adm_from_er']).astype(np.float64)
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_is_emergency', 'adm_is_elective', 'adm_from_er', 'adm_from_referral',
        'adm_from_transfer', 'adm_high_acuity'
    ])

    # 14. LACE Index bedside score proxy & clinical severity score
    def calc_lace(los, is_er, charlson, ed_visits):
        l_score = np.zeros_like(los, dtype=np.float64)
        l_score[los == 2] = 1.0
        l_score[los == 3] = 2.0
        l_score[(los >= 4) & (los <= 6)] = 3.0
        l_score[(los >= 7) & (los <= 13)] = 4.0
        l_score[los >= 14] = 5.0
        a_score = 3.0 * is_er
        c_score = np.clip(charlson, 0.0, 5.0)
        e_score = np.clip(ed_visits, 0.0, 4.0)
        return l_score + a_score + c_score + e_score

    train['lace_score'] = calc_lace(train['time_in_hospital'], train['adm_is_emergency'], train_charlson, train['number_emergency'])
    eval_cols['lace_score'] = calc_lace(eval_cols['time_in_hospital'], eval_cols['adm_is_emergency'], eval_charlson, eval_cols['number_emergency'])
    extra_numeric.append('lace_score')

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
    extra_numeric.extend([
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high', 'a1c_normal', 'glu_normal',
        'good_glycemic_control', 'diabetes_uncontrolled', 'high_a1c_no_med_change', 'high_a1c_med_change'
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

    # 17. Missingness flags
    for d in (train, eval_cols):
        d['missing_specialty'] = (d['medical_specialty'] == None).astype(np.float64)
        d['missing_payer'] = (d['payer_code'] == None).astype(np.float64)
    extra_numeric.extend(['missing_specialty', 'missing_payer'])

    # 18. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)

    # 19. Heterogeneous Stratified 5-Fold Ensemble: LightGBM (x2) + XGBoost
    import xgboost as xgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds_lgb1 = np.zeros(x_eval.shape[0])
    eval_preds_lgb2 = np.zeros(x_eval.shape[0])
    eval_preds_xgb = np.zeros(x_eval.shape[0])

    oof_lgb1 = np.zeros(len(y_train))
    oof_lgb2 = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))

    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        fold_x_val, fold_y_val = x_train[val_idx], y_train[val_idx]

        # Model 1: LightGBM (standard leaf-wise, seed 42)
        lgb_model1 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=1.5,
            cat_smooth=30.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model1.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        oof_lgb1[val_idx] = lgb_model1.predict_proba(fold_x_val)[:, 1]
        eval_preds_lgb1 += lgb_model1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: LightGBM (shallow depth-constrained, strong regularizer, seed 2023)
        lgb_model2 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.025,
            max_depth=4,
            num_leaves=15,
            min_child_samples=140,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=2.5,
            cat_smooth=35.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model2.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        oof_lgb2[val_idx] = lgb_model2.predict_proba(fold_x_val)[:, 1]
        eval_preds_lgb2 += lgb_model2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: XGBoost (depth-wise histogram, seed 42)
        xgb_model = xgb.XGBClassifier(
            n_estimators=220,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=2.0,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(fold_x_train, fold_y_train)
        oof_xgb[val_idx] = xgb_model.predict_proba(fold_x_val)[:, 1]
        eval_preds_xgb += xgb_model.predict_proba(x_eval)[:, 1] / 5.0

    # 20. High-Resolution Out-of-fold Grid Search for Optimal Blending Weights
    from sklearn.metrics import roc_auc_score
    best_auc = 0.0
    best_w = (0.45, 0.20, 0.35)
    
    w_steps = np.linspace(0.0, 1.0, 51)
    for w1 in w_steps:
        rem = 1.0 - w1
        if rem <= 0:
            w2_vals = [0.0]
        else:
            n_w2 = int(np.round(rem * 50)) + 1
            w2_vals = np.linspace(0.0, rem, n_w2)
        for w2 in w2_vals:
            w3 = max(0.0, 1.0 - w1 - w2)
            total = w1 + w2 + w3
            if total > 0:
                w1_n, w2_n, w3_n = w1 / total, w2 / total, w3 / total
            else:
                w1_n, w2_n, w3_n = 0.33, 0.33, 0.34
            blend_oof = w1_n * oof_lgb1 + w2_n * oof_lgb2 + w3_n * oof_xgb
            auc = roc_auc_score(y_train, blend_oof)
            if auc > best_auc:
                best_auc = auc
                best_w = (w1_n, w2_n, w3_n)

    # Blend the predictions using the optimal weights
    eval_preds = best_w[0] * eval_preds_lgb1 + best_w[1] * eval_preds_lgb2 + best_w[2] * eval_preds_xgb
    return np.clip(eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
