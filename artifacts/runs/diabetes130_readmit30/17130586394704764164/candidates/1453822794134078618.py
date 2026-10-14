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
        
    # 4. Target encoding of high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 5. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')
    
    # 6. Medication active and change counts
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
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')
    
    # 7. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')
    
    # 8. Numeric age representation
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

    # 9. Comorbidity weighted Charlson proxy and syndromic interactions
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

    # 10. Prior utilization details & acute care ratios
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

    # 11. Hospital stay intensity & procedures ratios
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

    # 12. Discharge and admission indicators
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
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_is_emergency', 'adm_from_er', 'adm_high_acuity'
    ])

    # 13. LACE Index bedside score proxy
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

    # 14. Lab test indicators & Glycemic control clinical inertia
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

    # 15. Medication directionality & polypharmacy
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
    extra_numeric.extend([
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'on_metformin', 'polypharmacy', 'diabetes_med_flag', 'med_change_flag'
    ])

    # 16. Missingness flags
    for d in (train, eval_cols):
        d['missing_specialty'] = (d['medical_specialty'] == None).astype(np.float64)
        d['missing_payer'] = (d['payer_code'] == None).astype(np.float64)
    extra_numeric.extend(['missing_specialty', 'missing_payer'])

    # 17. High-order categorical interactions & additional target encodings
    train['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch')

    train['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admsrc_disch')

    more_te_cols = ['admission_type_id', 'diag_1_grp', 'diag1_disch', 'admsrc_disch']
    for col in more_te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)

    # 18. Deep ICD-9 250.xx Diabetes Micro-Phenotyping
    def extract_dm_subtypes(d, n_rows):
        diag_c = ['diag_1', 'diag_2', 'diag_3']
        t1 = np.zeros(n_rows, dtype=np.float64)
        uncontrolled = np.zeros(n_rows, dtype=np.float64)
        keto = np.zeros(n_rows, dtype=np.float64)
        hyper = np.zeros(n_rows, dtype=np.float64)
        renal = np.zeros(n_rows, dtype=np.float64)
        ophthalmic = np.zeros(n_rows, dtype=np.float64)
        neuro = np.zeros(n_rows, dtype=np.float64)
        pvd = np.zeros(n_rows, dtype=np.float64)
        ulcer = np.zeros(n_rows, dtype=np.float64)

        for col_name in diag_c:
            arr = d[col_name]
            for i in range(n_rows):
                val = arr[i]
                if val is not None and str(val).startswith('250'):
                    s = str(val)
                    parts = s.split('.')
                    sub = parts[1] if len(parts) > 1 else '0'
                    x = sub[0] if len(sub) > 0 else '0'
                    y = sub[1] if len(sub) > 1 else '0'

                    if y in ('1', '3'):
                        t1[i] = 1.0
                    if y in ('2', '3'):
                        uncontrolled[i] = 1.0
                    if x == '1':
                        keto[i] = 1.0
                    elif x == '2':
                        hyper[i] = 1.0
                    elif x == '4':
                        renal[i] = 1.0
                    elif x == '5':
                        ophthalmic[i] = 1.0
                    elif x == '6':
                        neuro[i] = 1.0
                    elif x == '7':
                        pvd[i] = 1.0
                    elif x == '8':
                        ulcer[i] = 1.0

        return {
            'dm_type1': t1,
            'dm_uncontrolled': uncontrolled,
            'dm_ketoacidosis': keto,
            'dm_hyperosmolar': hyper,
            'dm_renal': renal,
            'dm_ophthalmic': ophthalmic,
            'dm_neuro': neuro,
            'dm_pvd': pvd,
            'dm_ulcer': ulcer,
            'dm_microvascular': renal + ophthalmic + neuro,
            'dm_metabolic_crisis': np.maximum(keto, hyper),
        }

    n_train_rows = len(y_train)
    n_eval_rows = len(eval_cols['insulin'])
    train_dm = extract_dm_subtypes(train, n_train_rows)
    eval_dm = extract_dm_subtypes(eval_cols, n_eval_rows)

    for k in train_dm:
        train[k] = train_dm[k]
        eval_cols[k] = eval_dm[k]
        extra_numeric.append(k)

    # Cardiorenal-metabolic triad
    train['cardiorenal_dm'] = ((train_cm['cm_chf'] == 1.0) & (train_cm['cm_renal'] == 1.0) & ((train_dm['dm_uncontrolled'] == 1.0) | (train_cm['cm_diabetes_complicated'] == 1.0))).astype(np.float64)
    eval_cols['cardiorenal_dm'] = ((eval_cm['cm_chf'] == 1.0) & (eval_cm['cm_renal'] == 1.0) & ((eval_dm['dm_uncontrolled'] == 1.0) | (eval_cm['cm_diabetes_complicated'] == 1.0))).astype(np.float64)
    extra_numeric.append('cardiorenal_dm')

    # 19. Pharmacotherapy Regimen Complexity & Hypoglycemia Risk
    sulfonylureas = ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
    oral_classes = ['metformin', 'repaglinide', 'nateglinide', 'glipizide', 'glyburide', 'glimepiride', 'pioglitazone', 'rosiglitazone', 'acarbose']

    for d, dm_feats in [(train, train_dm), (eval_cols, eval_dm)]:
        has_sulfo = np.zeros(len(d['insulin']), dtype=np.float64)
        for s_col in sulfonylureas:
            if s_col in d:
                has_sulfo = np.maximum(has_sulfo, ((d[s_col] != 'No') & (d[s_col] != None)).astype(np.float64))
        d['has_sulfonylurea'] = has_sulfo
        d['insulin_plus_sulfonylurea'] = ((d['insulin'] != 'No') & (has_sulfo == 1.0)).astype(np.float64)
        d['no_metformin_high_meds'] = ((d['metformin'] == 'No') & (d['num_medications'] >= 15.0)).astype(np.float64)

        oral_cnt = np.zeros(len(d['insulin']), dtype=np.float64)
        for o_col in oral_classes:
            if o_col in d:
                oral_cnt += ((d[o_col] != 'No') & (d[o_col] != None)).astype(np.float64)
        d['oral_drug_count'] = oral_cnt
        d['glycemic_inertia'] = ((d['a1c_high'] == 1.0) & (d['insulin'] == 'Steady') & (d['change'] == 'No')).astype(np.float64)

    extra_numeric.extend(['has_sulfonylurea', 'insulin_plus_sulfonylurea', 'no_metformin_high_meds', 'oral_drug_count', 'glycemic_inertia'])

    # 20. EHR Latent Semantic Phenomics via TruncatedSVD
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    def extract_concept_lists(d, n_rows):
        concepts = [[] for _ in range(n_rows)]
        for col_name in diag_cols:
            arr = d[col_name]
            for i in range(n_rows):
                val = arr[i]
                if val is not None and val != '?':
                    pfx = str(val).split('.')[0]
                    if pfx:
                        concepts[i].append(f"dx_{pfx}")

        disp_arr = d['discharge_disposition_id']
        src_arr = d['admission_source_id']
        typ_arr = d['admission_type_id']
        for i in range(n_rows):
            if disp_arr[i] is not None:
                concepts[i].append(f"disp_{disp_arr[i]}")
            if src_arr[i] is not None:
                concepts[i].append(f"src_{src_arr[i]}")
            if typ_arr[i] is not None:
                concepts[i].append(f"typ_{typ_arr[i]}")

        for m in med_cols:
            if m in d:
                m_arr = d[m]
                for i in range(n_rows):
                    val = m_arr[i]
                    if val is not None and val != 'No':
                        concepts[i].append(f"med_{m}_{val}")

        a1c_arr = d['A1Cresult']
        glu_arr = d['max_glu_serum']
        chg_arr = d['change']
        for i in range(n_rows):
            if a1c_arr[i] in ('>7', '>8'):
                concepts[i].append("a1c_high")
            if glu_arr[i] in ('>200', '>300'):
                concepts[i].append("glu_high")
            if chg_arr[i] == 'Ch':
                concepts[i].append("med_chg")

        return concepts

    train_concepts = extract_concept_lists(train, n_train_rows)
    eval_concepts = extract_concept_lists(eval_cols, n_eval_rows)

    c_counts = {}
    for clist in train_concepts:
        for c in clist:
            c_counts[c] = c_counts.get(c, 0) + 1

    svd_vocab = {c: idx for idx, (c, cnt) in enumerate(c_counts.items()) if cnt >= 20}

    r_tr, col_tr = [], []
    for i, clist in enumerate(train_concepts):
        for c in clist:
            idx = svd_vocab.get(c)
            if idx is not None:
                r_tr.append(i)
                col_tr.append(idx)
    train_mat = csr_matrix((np.ones(len(r_tr), dtype=np.float32), (r_tr, col_tr)), shape=(n_train_rows, len(svd_vocab)))

    r_ev, col_ev = [], []
    for i, clist in enumerate(eval_concepts):
        for c in clist:
            idx = svd_vocab.get(c)
            if idx is not None:
                r_ev.append(i)
                col_ev.append(idx)
    eval_mat = csr_matrix((np.ones(len(r_ev), dtype=np.float32), (r_ev, col_ev)), shape=(n_eval_rows, len(svd_vocab)))

    n_svd_comp = min(6, max(1, len(svd_vocab) - 1))
    svd = TruncatedSVD(n_components=n_svd_comp, random_state=42)
    train_svd = svd.fit_transform(train_mat)
    eval_svd = svd.transform(eval_mat)

    for comp in range(n_svd_comp):
        col_name = f'svd_pheno_{comp}'
        train[col_name] = train_svd[:, comp].astype(np.float64)
        eval_cols[col_name] = eval_svd[:, comp].astype(np.float64)
        extra_numeric.append(col_name)

    # 21. Care Trajectory Sequence & Transition Encodings
    train['trajectory_type'] = np.array([f"{t}_{s}_{disp}" for t, s, disp in zip(train['admission_type_id'], train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['trajectory_type'] = np.array([f"{t}_{s}_{disp}" for t, s, disp in zip(eval_cols['admission_type_id'], eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('trajectory_type')

    train['diag1_diag2'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag1_diag2'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag1_diag2')

    def bucket_stay(days):
        b = np.empty(len(days), dtype=object)
        b[days <= 2] = 'short'
        b[(days >= 3) & (days <= 5)] = 'med'
        b[(days >= 6) & (days <= 8)] = 'long'
        b[days >= 9] = 'vlong'
        return b

    train['stay_disch'] = np.array([f"{sb}_{disp}" for sb, disp in zip(bucket_stay(train['time_in_hospital']), train['discharge_disposition_id'])], dtype=object)
    eval_cols['stay_disch'] = np.array([f"{sb}_{disp}" for sb, disp in zip(bucket_stay(eval_cols['time_in_hospital']), eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('stay_disch')

    traj_te_cols = ['trajectory_type', 'diag1_diag2', 'stay_disch']
    for col in traj_te_cols:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
        te_col = col + "_te"
        train[te_col] = tr_enc
        eval_cols[te_col] = ev_enc
        extra_numeric.append(te_col)

    # 22. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)

    # 23. Heterogeneous 5-Fold Ensemble with True OOF Tracking
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds_lgb = np.zeros(x_eval.shape[0])
    eval_preds_xgb = np.zeros(x_eval.shape[0])
    oof_preds_lgb = np.zeros(x_train.shape[0])
    oof_preds_xgb = np.zeros(x_train.shape[0])

    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        fold_x_val = x_train[val_idx]

        # Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=120,
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
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(fold_x_val)[:, 1]
        eval_preds_lgb += lgb_model.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
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
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(fold_x_val)[:, 1]
        eval_preds_xgb += xgb_model.predict_proba(x_eval)[:, 1] / 5.0

    # 24. Dynamic OOF-Optimal Model Blending & Quantile Adaptation
    best_w = 0.65
    best_auc = -1.0
    best_mode = 'prob'

    for w in np.linspace(0.35, 0.85, 11):
        score = roc_auc_score(y_train, w * oof_preds_lgb + (1.0 - w) * oof_preds_xgb)
        if score > best_auc:
            best_auc = score
            best_w = w
            best_mode = 'prob'

    oof_r_lgb = rankdata(oof_preds_lgb) / len(oof_preds_lgb)
    oof_r_xgb = rankdata(oof_preds_xgb) / len(oof_preds_xgb)
    for w in np.linspace(0.35, 0.85, 11):
        score = roc_auc_score(y_train, w * oof_r_lgb + (1.0 - w) * oof_r_xgb)
        if score > best_auc:
            best_auc = score
            best_w = w
            best_mode = 'rank'

    if best_mode == 'rank':
        eval_r_lgb = rankdata(eval_preds_lgb) / len(eval_preds_lgb)
        eval_r_xgb = rankdata(eval_preds_xgb) / len(eval_preds_xgb)
        eval_preds = best_w * eval_r_lgb + (1.0 - best_w) * eval_r_xgb
        min_p = min(float(np.min(eval_preds_lgb)), float(np.min(eval_preds_xgb)))
        max_p = max(float(np.max(eval_preds_lgb)), float(np.max(eval_preds_xgb)))
        eval_preds = min_p + eval_preds * (max_p - min_p)
    else:
        eval_preds = best_w * eval_preds_lgb + (1.0 - best_w) * eval_preds_xgb

    return np.clip(eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
