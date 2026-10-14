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
    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson Comorbidities & Cardiorenal Syndromes
    cm_train = icd9_comorbidities(train_data['diag_1'], train_data['diag_2'], train_data['diag_3'])
    cm_eval = icd9_comorbidities(eval_data['diag_1'], eval_data['diag_2'], eval_data['diag_3'])
    for k, v in cm_train.items():
        train_data[k] = v
        eval_data[k] = cm_eval[k]
        extra_numeric.append(k)

    train_data['cm_cardiorenal'] = ((cm_train['cm_chf'] > 0) & (cm_train['cm_renal'] > 0)).astype(np.float64)
    eval_data['cm_cardiorenal'] = ((cm_eval['cm_chf'] > 0) & (cm_eval['cm_renal'] > 0)).astype(np.float64)
    extra_numeric.append('cm_cardiorenal')

    # 2. ICD-9 Chapter Groupings & 3-Digit Category Extraction
    def extract_icd9_3digit(diag_col):
        out = np.empty(len(diag_col), dtype=object)
        for i, c in enumerate(diag_col):
            if c is None:
                out[i] = 'missing'
            else:
                s = str(c).strip()
                out[i] = s.split('.')[0] if (s and s != '?') else 'missing'
        return out

    train_data['diag_1_group'] = np.array([icd9_group(c) for c in train_data['diag_1']], dtype=object)
    eval_data['diag_1_group'] = np.array([icd9_group(c) for c in eval_data['diag_1']], dtype=object)
    train_data['diag_2_group'] = np.array([icd9_group(c) for c in train_data['diag_2']], dtype=object)
    eval_data['diag_2_group'] = np.array([icd9_group(c) for c in eval_data['diag_2']], dtype=object)
    train_data['diag_3_group'] = np.array([icd9_group(c) for c in train_data['diag_3']], dtype=object)
    eval_data['diag_3_group'] = np.array([icd9_group(c) for c in eval_data['diag_3']], dtype=object)

    train_data['diag_1_3digit'] = extract_icd9_3digit(train_data['diag_1'])
    eval_data['diag_1_3digit'] = extract_icd9_3digit(eval_data['diag_1'])
    train_data['diag_2_3digit'] = extract_icd9_3digit(train_data['diag_2'])
    eval_data['diag_2_3digit'] = extract_icd9_3digit(eval_data['diag_2'])
    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group', 'diag_1_3digit', 'diag_2_3digit'])

    # Acute Readmission Precipitants (Electrolyte Imbalance, Pneumonia, Sepsis)
    def extract_prefix_flag(d1, d2, d3, prefixes):
        pref_tuple = tuple(prefixes)
        n = len(d1)
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            c1 = str(d1[i]).strip() if d1[i] is not None else ''
            c2 = str(d2[i]).strip() if d2[i] is not None else ''
            c3 = str(d3[i]).strip() if d3[i] is not None else ''
            if c1.startswith(pref_tuple) or c2.startswith(pref_tuple) or c3.startswith(pref_tuple):
                res[i] = 1.0
        return res

    for d, d_ref in ((train_data, train_data), (eval_data, eval_data)):
        d['cm_electrolyte'] = extract_prefix_flag(d_ref['diag_1'], d_ref['diag_2'], d_ref['diag_3'], ['276'])
        d['cm_pneumonia'] = extract_prefix_flag(d_ref['diag_1'], d_ref['diag_2'], d_ref['diag_3'], ['480', '481', '482', '483', '484', '485', '486'])
        d['cm_sepsis'] = extract_prefix_flag(d_ref['diag_1'], d_ref['diag_2'], d_ref['diag_3'], ['038', '995.9'])
    extra_numeric.extend(['cm_electrolyte', 'cm_pneumonia', 'cm_sepsis'])

    # 3. Validated Clinical LACE Readmission Risk Index
    for d in (train_data, eval_data):
        los = np.nan_to_num(d['time_in_hospital'], nan=0.0)
        l_score = np.select(
            [los < 1, los == 1, los == 2, los == 3,
             (los >= 4) & (los <= 6), (los >= 7) & (los <= 13), los >= 14],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
            default=0.0
        )
        a_score = np.where(d['admission_type_id'] == '1', 3.0, 0.0)
        c_score = np.minimum(d['cm_count'], 5.0)
        e_score = np.minimum(np.nan_to_num(d['number_emergency'], nan=0.0), 4.0)
        d['lace_index'] = l_score + a_score + c_score + e_score
    extra_numeric.append('lace_index')

    # 4. Out-of-Fold Target Encodings (Individual & Composite Care Transitions)
    te_features = ['diag_1', 'diag_2', 'diag_3', 'diag_1_3digit', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_features:
        te_tr, te_ev = oof_target_encode(train_data[col], y_train, eval_data[col], n_folds=5, smoothing=25.0, seed=42)
        te_name = f'te_{col}'
        train_data[te_name] = te_tr
        eval_data[te_name] = te_ev
        extra_numeric.append(te_name)

    # Composite: Discharge Disposition x Primary Diagnosis Group
    disp_diag_tr = np.array([f"{d}_{g}" for d, g in zip(train_data['discharge_disposition_id'], train_data['diag_1_group'])], dtype=object)
    disp_diag_ev = np.array([f"{d}_{g}" for d, g in zip(eval_data['discharge_disposition_id'], eval_data['diag_1_group'])], dtype=object)
    te_dd_tr, te_dd_ev = oof_target_encode(disp_diag_tr, y_train, disp_diag_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_disp_diag'] = te_dd_tr
    eval_data['te_disp_diag'] = te_dd_ev
    extra_numeric.append('te_disp_diag')

    # Composite: Admission Type x Admission Source
    adm_combo_tr = np.array([f"{t}_{s}" for t, s in zip(train_data['admission_type_id'], train_data['admission_source_id'])], dtype=object)
    adm_combo_ev = np.array([f"{t}_{s}" for t, s in zip(eval_data['admission_type_id'], eval_data['admission_source_id'])], dtype=object)
    te_adm_tr, te_adm_ev = oof_target_encode(adm_combo_tr, y_train, adm_combo_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_adm_combo'] = te_adm_tr
    eval_data['te_adm_combo'] = te_adm_ev
    extra_numeric.append('te_adm_combo')

    # Composite: Care Transition Pathway (Admission Source x Discharge Disposition)
    adm_disp_tr = np.array([f"{a}_{d}" for a, d in zip(train_data['admission_source_id'], train_data['discharge_disposition_id'])], dtype=object)
    adm_disp_ev = np.array([f"{a}_{d}" for a, d in zip(eval_data['admission_source_id'], eval_data['discharge_disposition_id'])], dtype=object)
    te_ad_tr, te_ad_ev = oof_target_encode(adm_disp_tr, y_train, adm_disp_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_adm_disp'] = te_ad_tr
    eval_data['te_adm_disp'] = te_ad_ev
    extra_numeric.append('te_adm_disp')

    # 5. Prior Healthcare Utilization Intensity & Encounter Velocity
    for d in (train_data, eval_data):
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(d['number_emergency'], nan=0.0)
        out = np.nan_to_num(d['number_outpatient'], nan=0.0)
        tot = inp + emg + out
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 0.5)

        d['total_prior_visits'] = tot
        d['log_total_visits'] = np.log1p(tot)
        d['log_inpatient'] = np.log1p(inp)
        d['inpatient_ratio'] = (inp + 0.1) / (tot + 1.0)
        d['emergency_ratio'] = (emg + 0.1) / (tot + 1.0)
        d['acute_utilization_score'] = 3.0 * inp + 2.0 * emg + out
        d['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d['freq_inpatient'] = (inp >= 2).astype(np.float64)
        d['high_inpatient'] = (inp >= 3).astype(np.float64)
        d['no_prior_visits'] = (tot == 0).astype(np.float64)

        d['labs_per_day'] = np.nan_to_num(d['num_lab_procedures'], nan=0.0) / los
        d['meds_per_day'] = np.nan_to_num(d['num_medications'], nan=0.0) / los
        d['procs_per_day'] = np.nan_to_num(d['num_procedures'], nan=0.0) / los
        d['meds_per_diag'] = np.nan_to_num(d['num_medications'], nan=0.0) / np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0)
        d['polypharmacy'] = (np.nan_to_num(d['num_medications'], nan=0.0) >= 15).astype(np.float64)
        d['multi_diagnoses'] = (np.nan_to_num(d['number_diagnoses'], nan=0.0) >= 9).astype(np.float64)

        d['inp_x_los'] = inp * los
        d['inp_x_cm'] = inp * d['cm_count']
        d['inp_x_lace'] = inp * d['lace_index']

    extra_numeric.extend([
        'total_prior_visits', 'log_total_visits', 'log_inpatient', 'inpatient_ratio', 'emergency_ratio',
        'acute_utilization_score', 'has_prior_inpatient', 'freq_inpatient', 'high_inpatient', 'no_prior_visits',
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'meds_per_diag', 'polypharmacy', 'multi_diagnoses',
        'inp_x_los', 'inp_x_cm', 'inp_x_lace'
    ])

    # 6. Age Continuous Mapping, Dispositions, & Clinical Lab Tests
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for d in (train_data, eval_data):
        d['age_numeric'] = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in d['age']], dtype=np.float64)
        d['age_x_inpatient'] = d['age_numeric'] * np.log1p(np.nan_to_num(d['number_inpatient'], nan=0.0))

        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = np.isin(disp, ['3', '4', '5', '22', '23', '24']).astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)

        # Glycemic Testing & Physician Responsiveness
        a1c = d['A1Cresult']
        d['a1c_tested'] = (a1c != 'None').astype(np.float64)
        d['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        d['a1c_very_high'] = (a1c == '>8').astype(np.float64)

        glu = d['max_glu_serum']
        d['glu_tested'] = (glu != 'None').astype(np.float64)
        d['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)

        ch = d.get('change', np.array(['No'] * len(disp)))
        d['a1c_high_no_change'] = ((d['a1c_high'] == 1.0) & (ch == 'No')).astype(np.float64)
        d['a1c_high_changed'] = ((d['a1c_high'] == 1.0) & (ch == 'Ch')).astype(np.float64)

        d['spec_is_missing'] = np.array([1.0 if s is None or s == '?' else 0.0 for s in d['medical_specialty']], dtype=np.float64)
        d['payer_is_missing'] = np.array([1.0 if p is None or p == '?' else 0.0 for p in d['payer_code']], dtype=np.float64)

    extra_numeric.extend([
        'age_numeric', 'age_x_inpatient', 'disch_home', 'disch_snf', 'disch_ama', 'disch_home_health',
        'a1c_tested', 'a1c_high', 'a1c_very_high', 'glu_tested', 'glu_high', 'a1c_high_no_change', 'a1c_high_changed',
        'spec_is_missing', 'payer_is_missing'
    ])

    # 7. Medication Regimen Complexity, Escalation vs De-escalation
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
        'insulin'
    ]
    avail_meds = [m for m in med_cols if m in task.categorical_columns]
    for d in (train_data, eval_data):
        n_rows = len(d['time_in_hospital'])
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        for m in avail_meds:
            col = d[m]
            active_cnt += np.isin(col, ['Steady', 'Up', 'Down']).astype(np.float64)
            up_cnt += (col == 'Up').astype(np.float64)
            down_cnt += (col == 'Down').astype(np.float64)
        d['num_active_meds'] = active_cnt
        d['num_meds_up'] = up_cnt
        d['num_meds_down'] = down_cnt
        d['num_changed_meds'] = up_cnt + down_cnt

        if 'insulin' in d:
            ins = d['insulin']
            d['has_insulin'] = (ins != 'No').astype(np.float64)
            d['insulin_status'] = np.select([ins == 'Steady', ins == 'Down', ins == 'Up'], [1.0, 2.0, 3.0], default=0.0)
            d['insulin_unstable'] = np.isin(ins, ['Up', 'Down']).astype(np.float64)

        if 'metformin' in d and 'insulin' in d:
            met = d['metformin']
            ins = d['insulin']
            d['insulin_only'] = ((ins != 'No') & (met == 'No')).astype(np.float64)
            d['insulin_and_met'] = ((ins != 'No') & (met != 'No')).astype(np.float64)

        if 'change' in d:
            d['med_change_flag'] = (d['change'] == 'Ch').astype(np.float64)
        if 'diabetesMed' in d:
            d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(np.float64)

    extra_numeric.extend(['num_active_meds', 'num_meds_up', 'num_meds_down', 'num_changed_meds', 'med_change_flag', 'diabetes_med_flag'])
    if 'insulin' in train_data:
        extra_numeric.extend(['has_insulin', 'insulin_status', 'insulin_unstable'])
    if 'metformin' in train_data and 'insulin' in train_data:
        extra_numeric.extend(['insulin_only', 'insulin_and_met'])

    train_data['diag_3_3digit'] = extract_icd9_3digit(train_data['diag_3'])
    eval_data['diag_3_3digit'] = extract_icd9_3digit(eval_data['diag_3'])
    extra_categorical.append('diag_3_3digit')

    # Interaction & Diagnostic Breadth Features
    for d in (train_data, eval_data):
        d['revolving_door_emergency'] = ((np.nan_to_num(d['number_emergency'], nan=0.0) > 0) &
                                         ((d['admission_source_id'] == '7') | (d['admission_type_id'] == '1'))).astype(np.float64)
        d['snf_x_age'] = d['disch_snf'] * d['age_numeric']
        d['disch_ama_x_prior'] = d['disch_ama'] * d['has_prior_inpatient']
        d['home_health_x_polypharmacy'] = d['disch_home_health'] * d['polypharmacy']
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 0.5)
        d['meds_density'] = np.nan_to_num(d['num_medications'], nan=0.0) / (los + 0.5)
        d['labs_per_diag'] = np.nan_to_num(d['num_lab_procedures'], nan=0.0) / np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0)
        if 'insulin' in d:
            d['insulin_up_with_oral'] = ((d['insulin'] == 'Up') & (d['num_active_meds'] > 1)).astype(np.float64)
            d['oral_meds_count'] = np.maximum(0.0, d['num_active_meds'] - d['has_insulin'])
        else:
            d['insulin_up_with_oral'] = np.zeros(len(los), dtype=np.float64)
            d['oral_meds_count'] = d['num_active_meds']

        # Diagnostic breadth: number of distinct clinical groups involved
        n_rows = len(los)
        breadth = np.empty(n_rows, dtype=np.float64)
        g1, g2, g3 = d['diag_1_group'], d['diag_2_group'], d['diag_3_group']
        for i in range(n_rows):
            s = {g for g in (g1[i], g2[i], g3[i]) if g not in ('missing', 'other', None)}
            breadth[i] = float(len(s))
        d['unique_diag_groups'] = breadth

    extra_numeric.extend([
        'revolving_door_emergency', 'snf_x_age', 'disch_ama_x_prior', 'home_health_x_polypharmacy',
        'meds_density', 'labs_per_diag', 'insulin_up_with_oral', 'oral_meds_count', 'unique_diag_groups'
    ])

    # Target Encode diag_3_3digit & Age-Stratified Primary Diagnosis
    te_tr_d3, te_ev_d3 = oof_target_encode(train_data['diag_3_3digit'], y_train, eval_data['diag_3_3digit'], n_folds=5, smoothing=25.0, seed=42)
    train_data['te_diag_3_3digit'] = te_tr_d3
    eval_data['te_diag_3_3digit'] = te_ev_d3
    extra_numeric.append('te_diag_3_3digit')

    diag1_age_tr = np.array([f"{g}_{a}" for g, a in zip(train_data['diag_1_group'], train_data['age'])], dtype=object)
    diag1_age_ev = np.array([f"{g}_{a}" for g, a in zip(eval_data['diag_1_group'], eval_data['age'])], dtype=object)
    te_da_tr, te_da_ev = oof_target_encode(diag1_age_tr, y_train, diag1_age_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_diag1_age'] = te_da_tr
    eval_data['te_diag1_age'] = te_da_ev
    extra_numeric.append('te_diag1_age')

    # 8. Care Discordance & Systemic Fragility Scoring (Isolation Forest)
    from sklearn.ensemble import IsolationForest
    iso_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_inpatient', 'number_emergency', 'number_diagnoses', 'age_numeric', 'lace_index'
    ]
    X_iso_tr = np.column_stack([np.nan_to_num(train_data[c], nan=0.0) for c in iso_cols])
    X_iso_ev = np.column_stack([np.nan_to_num(eval_data[c], nan=0.0) for c in iso_cols])
    iso = IsolationForest(n_estimators=80, max_samples=1000, random_state=42, n_jobs=1)
    iso.fit(X_iso_tr)
    train_data['care_discordance_score'] = iso.decision_function(X_iso_tr).astype(np.float64)
    eval_data['care_discordance_score'] = iso.decision_function(X_iso_ev).astype(np.float64)
    extra_numeric.append('care_discordance_score')

    # 9. Latent Syndromic Phenotyping via Bag-of-Clinical-Tokens SVD (EHR-LSA)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    def build_encounter_corpus(d):
        n = len(d['time_in_hospital'])
        corpus = []
        g1, g2 = d['diag_1_group'], d['diag_2_group']
        d1, d2, d3 = d['diag_1_3digit'], d['diag_2_3digit'], d['diag_3_3digit']
        disp = d['discharge_disposition_id']
        adm = d['admission_source_id']
        for i in range(n):
            toks = []
            if d1[i] != 'missing': toks.append(f"d1_{d1[i]}")
            if d2[i] != 'missing': toks.append(f"d2_{d2[i]}")
            if d3[i] != 'missing': toks.append(f"d3_{d3[i]}")
            if g1[i] != 'missing': toks.append(f"g1_{g1[i]}")
            if g2[i] != 'missing': toks.append(f"g2_{g2[i]}")
            if disp[i] is not None: toks.append(f"disp_{disp[i]}")
            if adm[i] is not None: toks.append(f"adm_{adm[i]}")
            for m in avail_meds:
                v = d[m][i]
                if v in ('Up', 'Down', 'Steady'):
                    toks.append(f"m_{m}_{v}")
            corpus.append(" ".join(toks))
        return corpus

    corpus_tr = build_encounter_corpus(train_data)
    corpus_ev = build_encounter_corpus(eval_data)
    tfidf = TfidfVectorizer(max_features=180, min_df=5, token_pattern=r'\S+')
    X_tfidf_tr = tfidf.fit_transform(corpus_tr)
    X_tfidf_ev = tfidf.transform(corpus_ev)

    n_svd_comp = 6
    svd = TruncatedSVD(n_components=n_svd_comp, random_state=42)
    X_svd_tr = svd.fit_transform(X_tfidf_tr)
    X_svd_ev = svd.transform(X_tfidf_ev)
    for k in range(n_svd_comp):
        c_name = f'pheno_svd_{k}'
        train_data[c_name] = X_svd_tr[:, k].astype(np.float64)
        eval_data[c_name] = X_svd_ev[:, k].astype(np.float64)
        extra_numeric.append(c_name)

    # 10. Case-Based Retrieval Memory (Out-of-Fold k-NN Doppelganger Priors)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    knn_feature_cols = [
        'log_inpatient', 'log_total_visits', 'emergency_ratio', 'lace_index',
        'time_in_hospital', 'age_numeric', 'cm_count', 'number_diagnoses',
        'num_medications', 'te_diag_1', 'te_disp_diag', 'care_discordance_score'
    ]
    X_knn_raw = np.column_stack([np.nan_to_num(train_data[c], nan=0.0) for c in knn_feature_cols])
    X_knn_eval_raw = np.column_stack([np.nan_to_num(eval_data[c], nan=0.0) for c in knn_feature_cols])

    scaler_full = StandardScaler()
    X_knn_tr_scaled = scaler_full.fit_transform(X_knn_raw)
    X_knn_ev_scaled = scaler_full.transform(X_knn_eval_raw)

    n_train_rows = len(y_train)
    oof_knn_risk_15 = np.zeros(n_train_rows, dtype=np.float64)
    oof_knn_risk_30 = np.zeros(n_train_rows, dtype=np.float64)
    oof_knn_dist_ratio = np.zeros(n_train_rows, dtype=np.float64)

    rng_knn = np.random.RandomState(42)
    perm_knn = rng_knn.permutation(n_train_rows)
    folds_knn = np.array_split(perm_knn, 5)

    for val_idx in folds_knn:
        trn_mask = np.ones(n_train_rows, dtype=bool)
        trn_mask[val_idx] = False
        trn_idx = np.where(trn_mask)[0]

        sc = StandardScaler()
        X_f_tr = sc.fit_transform(X_knn_raw[trn_idx])
        X_f_val = sc.transform(X_knn_raw[val_idx])

        nn = NearestNeighbors(n_neighbors=30, algorithm='kd_tree', n_jobs=1)
        nn.fit(X_f_tr)
        distances, indices = nn.kneighbors(X_f_val)

        y_trn_f = y_train[trn_idx]
        nbr_y = y_trn_f[indices]
        oof_knn_risk_15[val_idx] = np.mean(nbr_y[:, :15], axis=1)
        oof_knn_risk_30[val_idx] = np.mean(nbr_y, axis=1)

        pos_dist = np.where(nbr_y == 1, distances, np.inf)
        neg_dist = np.where(nbr_y == 0, distances, np.inf)
        min_pos = np.min(pos_dist, axis=1)
        min_neg = np.min(neg_dist, axis=1)
        min_pos = np.where(np.isinf(min_pos), 10.0, min_pos)
        min_neg = np.where(np.isinf(min_neg), 1.0, min_neg)
        oof_knn_dist_ratio[val_idx] = (min_pos + 1e-3) / (min_neg + 1e-3)

    nn_full = NearestNeighbors(n_neighbors=30, algorithm='kd_tree', n_jobs=1)
    nn_full.fit(X_knn_tr_scaled)
    distances_ev, indices_ev = nn_full.kneighbors(X_knn_ev_scaled)

    nbr_y_ev = y_train[indices_ev]
    eval_knn_risk_15 = np.mean(nbr_y_ev[:, :15], axis=1)
    eval_knn_risk_30 = np.mean(nbr_y_ev, axis=1)

    pos_dist_ev = np.where(nbr_y_ev == 1, distances_ev, np.inf)
    neg_dist_ev = np.where(nbr_y_ev == 0, distances_ev, np.inf)
    min_pos_ev = np.min(pos_dist_ev, axis=1)
    min_neg_ev = np.min(neg_dist_ev, axis=1)
    min_pos_ev = np.where(np.isinf(min_pos_ev), 10.0, min_pos_ev)
    min_neg_ev = np.where(np.isinf(min_neg_ev), 1.0, min_neg_ev)
    eval_knn_dist_ratio = (min_pos_ev + 1e-3) / (min_neg_ev + 1e-3)

    train_data['knn_readmit_risk_15'] = oof_knn_risk_15
    eval_data['knn_readmit_risk_15'] = eval_knn_risk_15
    train_data['knn_readmit_risk_30'] = oof_knn_risk_30
    eval_data['knn_readmit_risk_30'] = eval_knn_risk_30
    train_data['knn_dist_ratio'] = oof_knn_dist_ratio
    eval_data['knn_dist_ratio'] = eval_knn_dist_ratio
    extra_numeric.extend(['knn_readmit_risk_15', 'knn_readmit_risk_30', 'knn_dist_ratio'])

    # 11. Encode Features
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_data, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 12. Diverse Quad-LightGBM Acuity Ensemble (including Extremely Randomized Trees Boosting)
    model_balanced = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_balanced.fit(x_train, y_train, categorical_feature=cat_indices)

    model_interaction = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.015,
        num_leaves=48,
        max_depth=7,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.52,
        reg_lambda=3.5,
        reg_alpha=0.4,
        cat_smooth=35.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    model_interaction.fit(x_train, y_train, categorical_feature=cat_indices)

    model_conservative = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.022,
        num_leaves=18,
        max_depth=4,
        min_child_samples=250,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.68,
        reg_lambda=5.0,
        cat_smooth=75.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model_conservative.fit(x_train, y_train, categorical_feature=cat_indices)

    pred_extra = None
    try:
        model_extra = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.020,
            num_leaves=36,
            extra_trees=True,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=40.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        )
        model_extra.fit(x_train, y_train, categorical_feature=cat_indices)
        pred_extra = model_extra.predict_proba(x_eval)[:, 1]
    except Exception:
        pred_extra = None

    pred1 = model_balanced.predict_proba(x_eval)[:, 1]
    pred2 = model_interaction.predict_proba(x_eval)[:, 1]
    pred3 = model_conservative.predict_proba(x_eval)[:, 1]

    # 13. Diverse Depth-Wise Model Integration (XGBoost)
    pred_xgb = None
    xgb_pkg = globals().get('xgb', globals().get('xgboost', None))
    if xgb_pkg is not None and hasattr(xgb_pkg, 'XGBClassifier'):
        try:
            model_xgb = xgb_pkg.XGBClassifier(
                n_estimators=200,
                learning_rate=0.022,
                max_depth=5,
                min_child_weight=8,
                subsample=0.80,
                colsample_bytree=0.55,
                reg_lambda=3.0,
                reg_alpha=0.2,
                random_state=314,
                n_jobs=1,
                eval_metric='logloss',
            )
            model_xgb.fit(x_train, y_train)
            pred_xgb = model_xgb.predict_proba(x_eval)[:, 1]
        except Exception:
            pred_xgb = None

    # 14. Percentile-Rank Consensus & Probability Ensembling
    def to_percentile_rank(pred: np.ndarray) -> np.ndarray:
        n = len(pred)
        if n <= 1:
            return pred
        return np.argsort(np.argsort(pred)).astype(np.float64) / float(n - 1)

    models_preds = [pred1, pred2, pred3]
    weights_prob = [0.35, 0.30, 0.18]
    weights_rank = [0.35, 0.30, 0.18]

    if pred_extra is not None:
        models_preds.append(pred_extra)
        weights_prob.append(0.12)
        weights_rank.append(0.12)

    if pred_xgb is not None:
        models_preds.append(pred_xgb)
        weights_prob.append(0.15)
        weights_rank.append(0.15)

    w_p = np.array(weights_prob) / sum(weights_prob)
    w_r = np.array(weights_rank) / sum(weights_rank)

    prob_blend = np.zeros_like(pred1)
    rank_blend = np.zeros_like(pred1)
    for p, wp, wr in zip(models_preds, w_p, w_r):
        prob_blend += wp * p
        rank_blend += wr * to_percentile_rank(p)

    final_pred = 0.60 * rank_blend + 0.40 * prob_blend
    return np.clip(final_pred, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
