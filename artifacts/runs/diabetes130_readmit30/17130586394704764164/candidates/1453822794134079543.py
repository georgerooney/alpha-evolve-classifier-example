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
    import xgboost as xgb
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    # 1. Initialize feature registration lists and dict copies
    extra_numeric = []
    extra_categorical = []

    train = dict(train)
    eval_cols = dict(eval_cols)
    n_train_rows = len(y_train)
    n_eval_rows = len(eval_cols['time_in_hospital'])

    # 2. Extract comorbidities using icd9_comorbidities
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]

    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)

    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)

    # 3. Grouped ICD-9 chapters
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

    # 4. Standard 3-digit ICD-9 chapter extraction
    def extract_3digit(code_array):
        res = []
        for x in code_array:
            if x is None:
                res.append("None")
            else:
                s = str(x).strip()
                if not s or s == "None" or s == "?":
                    res.append("None")
                else:
                    res.append(s.split('.')[0][:3])
        return np.array(res, dtype=object)

    diag_1_3d = extract_3digit(train['diag_1'])
    diag_2_3d = extract_3digit(train['diag_2'])
    diag_3_3d = extract_3digit(train['diag_3'])
    ev_diag_1_3d = extract_3digit(eval_cols['diag_1'])
    ev_diag_2_3d = extract_3digit(eval_cols['diag_2'])
    ev_diag_3_3d = extract_3digit(eval_cols['diag_3'])

    train['diag_1_3d'] = diag_1_3d
    train['diag_2_3d'] = diag_2_3d
    train['diag_3_3d'] = diag_3_3d
    eval_cols['diag_1_3d'] = ev_diag_1_3d
    eval_cols['diag_2_3d'] = ev_diag_2_3d
    eval_cols['diag_3_3d'] = ev_diag_3_3d
    extra_categorical.extend(['diag_1_3d', 'diag_2_3d', 'diag_3_3d'])

    # 5. Multi-diagnosis chronic condition flags across all 3 diagnoses
    for d, d1_3d, d2_3d, d3_3d in [
        (train, diag_1_3d, diag_2_3d, diag_3_3d),
        (eval_cols, ev_diag_1_3d, ev_diag_2_3d, ev_diag_3_3d)
    ]:
        d['any_heart_failure'] = ((d1_3d == '428') | (d2_3d == '428') | (d3_3d == '428')).astype(np.float64)
        d['any_copd'] = (np.isin(d1_3d, ['491', '492', '493', '494', '496']) |
                         np.isin(d2_3d, ['491', '492', '493', '494', '496']) |
                         np.isin(d3_3d, ['491', '492', '493', '494', '496'])).astype(np.float64)
        d['any_renal_failure'] = (np.isin(d1_3d, ['584', '585', '586']) |
                                  np.isin(d2_3d, ['584', '585', '586']) |
                                  np.isin(d3_3d, ['584', '585', '586'])).astype(np.float64)
        d['any_cad'] = (np.isin(d1_3d, ['410', '411', '412', '414']) |
                        np.isin(d2_3d, ['410', '411', '412', '414']) |
                        np.isin(d3_3d, ['410', '411', '412', '414'])).astype(np.float64)
        d['any_pneumonia'] = (np.isin(d1_3d, ['480', '481', '482', '485', '486']) |
                              np.isin(d2_3d, ['480', '481', '482', '485', '486']) |
                              np.isin(d3_3d, ['480', '481', '482', '485', '486'])).astype(np.float64)
        d['any_fluid_electrolyte'] = ((d1_3d == '276') | (d2_3d == '276') | (d3_3d == '276')).astype(np.float64)
        d['any_dysrhythmia'] = ((d1_3d == '427') | (d2_3d == '427') | (d3_3d == '427')).astype(np.float64)
        d['any_diabetes'] = ((d1_3d == '250') | (d2_3d == '250') | (d3_3d == '250')).astype(np.float64)
    extra_numeric.extend([
        'any_heart_failure', 'any_copd', 'any_renal_failure', 'any_cad',
        'any_pneumonia', 'any_fluid_electrolyte', 'any_dysrhythmia', 'any_diabetes'
    ])

    # 6. Specific uncontrolled diabetes ICD-9 flag (5th digit 2 or 3)
    def check_dm_uncontrolled(d1, d2, d3):
        res = np.zeros(len(d1), dtype=np.float64)
        for i, (c1, c2, c3) in enumerate(zip(d1, d2, d3)):
            for c in (c1, c2, c3):
                if c is not None:
                    s = str(c).strip()
                    if s.startswith('250') and (s.endswith('2') or s.endswith('3')):
                        res[i] = 1.0
                        break
        return res

    train['dm_uncontrolled'] = check_dm_uncontrolled(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cols['dm_uncontrolled'] = check_dm_uncontrolled(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    extra_numeric.append('dm_uncontrolled')

    # 7. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')

    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')

    # 8. Medication active, changes, direction, and pharmacotherapy interactions
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    sulfonylurea_cols = [
        'glimepiride', 'glipizide', 'glyburide', 'chlorpropamide',
        'acetohexamide', 'tolbutamide', 'tolazamide',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone'
    ]

    for d in (train, eval_cols):
        n_rows = len(d['time_in_hospital'])
        act_meds = np.zeros(n_rows, dtype=np.float64)
        chg_meds = np.zeros(n_rows, dtype=np.float64)
        up_meds = np.zeros(n_rows, dtype=np.float64)
        down_meds = np.zeros(n_rows, dtype=np.float64)
        on_sulfo = np.zeros(n_rows, dtype=np.float64)

        for col in med_cols:
            if col in d:
                val = d[col]
                act_meds += ((val != "No") & (val != None)).astype(np.float64)
                chg_meds += ((val == "Up") | (val == "Down")).astype(np.float64)
                up_meds += (val == "Up").astype(np.float64)
                down_meds += (val == "Down").astype(np.float64)

        for col in sulfonylurea_cols:
            if col in d:
                on_sulfo = np.maximum(on_sulfo, ((d[col] != "No") & (d[col] != None)).astype(np.float64))

        d['active_meds_count'] = act_meds
        d['change_meds_count'] = chg_meds
        d['meds_up_count'] = up_meds
        d['meds_down_count'] = down_meds
        d['net_med_change'] = up_meds - down_meds
        d['on_sulfonylurea'] = on_sulfo
        d['insulin_plus_sulfo'] = ((d['insulin'] != "No") & (on_sulfo == 1.0)).astype(np.float64)

    extra_numeric.extend([
        'active_meds_count', 'change_meds_count', 'meds_up_count',
        'meds_down_count', 'net_med_change', 'on_sulfonylurea', 'insulin_plus_sulfo'
    ])

    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    # 9. Numeric age representation & geriatric acuity
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = str(age_str).replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 10. Charlson proxy and syndromic interactions
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
    extra_numeric.extend([
        'has_inpatient', 'has_emergency', 'has_outpatient', 'frequent_inpatient',
        'emergency_ratio', 'acute_ratio', 'weighted_prior_visits',
        'log_inpatient', 'log_emergency', 'log_total_visits'
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
        d['complexity_index'] = np.log1p(d['time_in_hospital'] * d['number_diagnoses'] * (d['number_inpatient'] + 1.0))
    extra_numeric.extend([
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'diagnoses_per_day',
        'total_procedures', 'meds_per_diag', 'procs_per_med', 'long_stay', 'complexity_index'
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
        d['adm_from_er'] = (adm_src == '7').astype(np.float64)
        d['adm_high_acuity'] = (d['adm_is_emergency'] * d['adm_from_er']).astype(np.float64)

        elderly = (d['age_num'] >= 80).astype(np.float64)
        d['elderly'] = elderly
        d['elderly_snf'] = elderly * d['disch_snf']
        d['elderly_high_util'] = elderly * (d['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_is_emergency', 'adm_from_er', 'adm_high_acuity',
        'elderly', 'elderly_snf', 'elderly_high_util'
    ])

    # 14. LACE Index bedside score proxy
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

    # 15. Lab test indicators & Glycemic control clinical inertia
    for d in (train, eval_cols):
        a1c = d['A1Cresult']
        glu = d['max_glu_serum']
        chg = d['change']

        a1c_tested = (a1c != 'None').astype(np.float64)
        a1c_high = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
        glu_tested = (glu != 'None').astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)

        d['a1c_tested'] = a1c_tested
        d['a1c_high'] = a1c_high
        d['glu_tested'] = glu_tested
        d['glu_high'] = glu_high
        d['diabetes_uncontrolled'] = ((a1c_high == 1.0) | (glu_high == 1.0)).astype(np.float64)
        d['high_a1c_no_med_change'] = ((a1c_high == 1.0) & (chg == 'No')).astype(np.float64)
        d['high_a1c_med_change'] = ((a1c_high == 1.0) & (chg == 'Ch')).astype(np.float64)
    extra_numeric.extend([
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high',
        'diabetes_uncontrolled', 'high_a1c_no_med_change', 'high_a1c_med_change'
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
        d['missing_specialty'] = (d['medical_specialty'] == None).astype(np.float64)
        d['missing_payer'] = (d['payer_code'] == None).astype(np.float64)
        d['weight_recorded'] = ((d['weight'] != None) & (d['weight'] != '?')).astype(np.float64)
    extra_numeric.extend([
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'on_metformin', 'polypharmacy', 'diabetes_med_flag', 'med_change_flag',
        'missing_specialty', 'missing_payer', 'weight_recorded'
    ])

    # 17. High-order categorical interactions
    train['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch')

    train['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admsrc_disch')

    train['age_disch'] = np.array([f"{a}_{disp}" for a, disp in zip(train['age'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['age_disch'] = np.array([f"{a}_{disp}" for a, disp in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('age_disch')

    train['spec_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(train['medical_specialty'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['spec_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(eval_cols['medical_specialty'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('spec_disch')

    # 18. Frequency encodings
    freq_cols = ['medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'diag_1_3d', 'diag1_disch']
    for col in freq_cols:
        val_counts = {}
        for x in train[col]:
            val_counts[x] = val_counts.get(x, 0) + 1
        n_tr = float(len(train[col]))
        col_freq = col + "_freq"
        train[col_freq] = np.array([val_counts.get(x, 0) / n_tr for x in train[col]], dtype=np.float64)
        eval_cols[col_freq] = np.array([val_counts.get(x, 0) / n_tr for x in eval_cols[col]], dtype=np.float64)
        extra_numeric.append(col_freq)

    # 19. Out-of-fold target encoding
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code', 'diag_1_grp',
        'diag1_disch', 'admsrc_disch', 'age_disch', 'spec_disch'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)

    # 20. Summary diagnosis risk across all 3 diagnoses
    train['max_diag_risk'] = np.maximum(train['diag_1_te'], np.maximum(train['diag_2_te'], train['diag_3_te']))
    eval_cols['max_diag_risk'] = np.maximum(eval_cols['diag_1_te'], np.maximum(eval_cols['diag_2_te'], eval_cols['diag_3_te']))
    extra_numeric.append('max_diag_risk')

    train['mean_diag_risk'] = (train['diag_1_te'] + train['diag_2_te'] + train['diag_3_te']) / 3.0
    eval_cols['mean_diag_risk'] = (eval_cols['diag_1_te'] + eval_cols['diag_2_te'] + eval_cols['diag_3_te']) / 3.0
    extra_numeric.append('mean_diag_risk')

    train['max_diag3d_risk'] = np.maximum(train['diag_1_3d_te'], np.maximum(train['diag_2_3d_te'], train['diag_3_3d_te']))
    eval_cols['max_diag3d_risk'] = np.maximum(eval_cols['diag_1_3d_te'], np.maximum(eval_cols['diag_2_3d_te'], eval_cols['diag_3_3d_te']))
    extra_numeric.append('max_diag3d_risk')

    # 21. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)

    # 22. Heterogeneous 5-Fold Ensemble: LightGBM + XGBoost + Deep-Regularized LightGBM
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_1 = np.zeros(n_train_rows, dtype=np.float64)
    oof_preds_2 = np.zeros(n_train_rows, dtype=np.float64)
    oof_preds_3 = np.zeros(n_train_rows, dtype=np.float64)

    eval_preds_1 = np.zeros(n_eval_rows, dtype=np.float64)
    eval_preds_2 = np.zeros(n_eval_rows, dtype=np.float64)
    eval_preds_3 = np.zeros(n_eval_rows, dtype=np.float64)

    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        fold_x_val = x_train[val_idx]

        # Model 1: LightGBM (leaf-wise, categorical optimal splits)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            cat_smooth=35.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        oof_preds_1[val_idx] = lgb_model.predict_proba(fold_x_val)[:, 1]
        eval_preds_1 += lgb_model.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: XGBoost (depth-wise histogram trees)
        xgb_model = xgb.XGBClassifier(
            n_estimators=220,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=1.5,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(fold_x_train, fold_y_train)
        oof_preds_2[val_idx] = xgb_model.predict_proba(fold_x_val)[:, 1]
        eval_preds_2 += xgb_model.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: Deep-Regularized LightGBM (higher leaf capacity with stronger shrinkage & feature subsampling)
        lgb2_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.025,
            num_leaves=48,
            min_child_samples=140,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.5,
            reg_lambda=3.0,
            cat_smooth=45.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        )
        lgb2_model.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        oof_preds_3[val_idx] = lgb2_model.predict_proba(fold_x_val)[:, 1]
        eval_preds_3 += lgb2_model.predict_proba(x_eval)[:, 1] / 5.0

    # 23. Out-Of-Fold Rank Normalization & Fast Grid Search for Optimal Blend Weights
    rank_oof_1 = rankdata(oof_preds_1) / float(n_train_rows)
    rank_oof_2 = rankdata(oof_preds_2) / float(n_train_rows)
    rank_oof_3 = rankdata(oof_preds_3) / float(n_train_rows)

    rank_test_1 = rankdata(eval_preds_1) / float(n_eval_rows)
    rank_test_2 = rankdata(eval_preds_2) / float(n_eval_rows)
    rank_test_3 = rankdata(eval_preds_3) / float(n_eval_rows)

    best_auc = -1.0
    best_w = (0.45, 0.35, 0.20)
    for w1_i in range(0, 21):
        w1 = w1_i * 0.05
        for w2_i in range(0, 21 - w1_i):
            w2 = w2_i * 0.05
            w3 = max(0.0, round(1.0 - w1 - w2, 4))
            blend_oof = w1 * rank_oof_1 + w2 * rank_oof_2 + w3 * rank_oof_3
            auc = roc_auc_score(y_train, blend_oof)
            if auc > best_auc:
                best_auc = auc
                best_w = (w1, w2, w3)

    w1, w2, w3 = best_w
    final_preds = w1 * rank_test_1 + w2 * rank_test_2 + w3 * rank_test_3
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
