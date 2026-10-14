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
    try:
        from sklearn.decomposition import TruncatedSVD
    except Exception:
        import sklearn.decomposition
        TruncatedSVD = sklearn.decomposition.TruncatedSVD

    try:
        from scipy.stats import rankdata
    except Exception:
        def rankdata(a):
            return np.argsort(np.argsort(a)).astype(np.float64) + 1.0

    try:
        import xgboost as xgb
    except Exception:
        xgb = globals().get('xgboost')

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []
    n_train_rows = len(y_train)
    n_eval_rows = len(eval_data['time_in_hospital'])

    # 1. Charlson Comorbidities & Cardiorenal Syndromes
    cm_train = icd9_comorbidities(train_data['diag_1'], train_data['diag_2'], train_data['diag_3'])
    cm_eval = icd9_comorbidities(eval_data['diag_1'], eval_data['diag_2'], eval_data['diag_3'])
    for k, v in cm_train.items():
        train_data[k] = v
        eval_data[k] = cm_eval[k]
        extra_numeric.append(k)

    # Cardiorenal & Cardiopulmonary syndromic interactions
    train_data['cm_cardiorenal'] = ((cm_train['cm_chf'] > 0) & (cm_train['cm_renal'] > 0)).astype(np.float64)
    eval_data['cm_cardiorenal'] = ((cm_eval['cm_chf'] > 0) & (cm_eval['cm_renal'] > 0)).astype(np.float64)
    train_data['cm_cardiopulmonary'] = ((cm_train['cm_chf'] > 0) & (cm_train['cm_pulmonary'] > 0)).astype(np.float64)
    eval_data['cm_cardiopulmonary'] = ((cm_eval['cm_chf'] > 0) & (cm_eval['cm_pulmonary'] > 0)).astype(np.float64)
    train_data['cm_chf_dm'] = ((cm_train['cm_chf'] > 0) & (cm_train['cm_diabetes_complicated'] > 0)).astype(np.float64)
    eval_data['cm_chf_dm'] = ((cm_eval['cm_chf'] > 0) & (cm_eval['cm_diabetes_complicated'] > 0)).astype(np.float64)
    extra_numeric.extend(['cm_cardiorenal', 'cm_cardiopulmonary', 'cm_chf_dm'])

    # 2. ICD-9 Chapter & 3-Digit Clinical Hierarchical Codes
    def _extract_prefix(val):
        if val is None or not isinstance(val, str) or len(val) == 0:
            return 'missing'
        s = val.split('.')[0]
        return s[:4] if s.startswith('E') else s[:3]

    for diag in ('diag_1', 'diag_2', 'diag_3'):
        train_data[f'{diag}_group'] = np.array([icd9_group(c) for c in train_data[diag]], dtype=object)
        eval_data[f'{diag}_group'] = np.array([icd9_group(c) for c in eval_data[diag]], dtype=object)
        train_data[f'{diag}_prefix'] = np.array([_extract_prefix(c) for c in train_data[diag]], dtype=object)
        eval_data[f'{diag}_prefix'] = np.array([_extract_prefix(c) for c in eval_data[diag]], dtype=object)
        extra_categorical.extend([f'{diag}_group', f'{diag}_prefix'])

    # 3. Clinical LACE Readmission Risk Index (Acuity adjusted)
    for d in (train_data, eval_data):
        los = np.nan_to_num(d['time_in_hospital'], nan=0.0)
        l_score = np.select(
            [los < 1, los == 1, los == 2, los == 3,
             (los >= 4) & (los <= 6), (los >= 7) & (los <= 13), los >= 14],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
            default=0.0
        )
        adm_type = d['admission_type_id']
        adm_src = d['admission_source_id']
        a_score = np.where(np.isin(adm_type, ['1', '2']) | (adm_src == '7'), 3.0, 0.0)
        c_score = np.minimum(d['cm_count'], 5.0)
        e_score = np.minimum(np.nan_to_num(d['number_emergency'], nan=0.0), 4.0)
        lace = l_score + a_score + c_score + e_score
        d['lace_index'] = lace
        d['lace_high'] = (lace >= 10.0).astype(np.float64)
    extra_numeric.extend(['lace_index', 'lace_high'])

    # 4. Out-of-Fold Target Encodings (Individual & Care Continuum Pathways)
    te_features = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_prefix', 'diag_2_prefix',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code'
    ]
    for col in te_features:
        te_tr, te_ev = oof_target_encode(train_data[col], y_train, eval_data[col], n_folds=5, smoothing=20.0, seed=42)
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

    # Composite: Payer x Discharge Disposition
    payer_disp_tr = np.array([f"{p}_{d}" for p, d in zip(train_data['payer_code'], train_data['discharge_disposition_id'])], dtype=object)
    payer_disp_ev = np.array([f"{p}_{d}" for p, d in zip(eval_data['payer_code'], eval_data['discharge_disposition_id'])], dtype=object)
    te_pd_tr, te_pd_ev = oof_target_encode(payer_disp_tr, y_train, payer_disp_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_payer_disp'] = te_pd_tr
    eval_data['te_payer_disp'] = te_pd_ev
    extra_numeric.append('te_payer_disp')

    # Composite Care Pathway: Admission Source x Discharge Disposition x Diagnosis Group
    pathway_tr = np.array([f"{s}_{d}_{g}" for s, d, g in zip(train_data['admission_source_id'], train_data['discharge_disposition_id'], train_data['diag_1_group'])], dtype=object)
    pathway_ev = np.array([f"{s}_{d}_{g}" for s, d, g in zip(eval_data['admission_source_id'], eval_data['discharge_disposition_id'], eval_data['diag_1_group'])], dtype=object)
    te_pw_tr, te_pw_ev = oof_target_encode(pathway_tr, y_train, pathway_ev, n_folds=5, smoothing=30.0, seed=42)
    train_data['te_care_pathway'] = te_pw_tr
    eval_data['te_care_pathway'] = te_pw_ev
    extra_numeric.append('te_care_pathway')

    # 5. Low-Rank Tensor Decomposition on Multi-Modal Clinical Interactions
    cm_keys = list(cm_train.keys())
    cm_tr_mat = np.column_stack([cm_train[k] for k in cm_keys])
    cm_ev_mat = np.column_stack([cm_eval[k] for k in cm_keys])

    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
        'insulin'
    ]
    avail_meds = [m for m in med_cols if m in task.categorical_columns]

    def _encode_med(d, m):
        col = d[m]
        return np.select([col == 'Steady', col == 'Up', col == 'Down'], [1.0, 2.0, -1.0], default=0.0)

    if avail_meds:
        med_tr_mat = np.column_stack([_encode_med(train_data, m) for m in avail_meds])
        med_ev_mat = np.column_stack([_encode_med(eval_data, m) for m in avail_meds])

        # Mode-1 Tensor Unfolding: Patient x (Comorbidity (15) x Medication (23))
        T_tr = (cm_tr_mat[:, :, None] * med_tr_mat[:, None, :]).reshape(n_train_rows, -1)
        T_ev = (cm_ev_mat[:, :, None] * med_ev_mat[:, None, :]).reshape(n_eval_rows, -1)

        svd_cm_med = TruncatedSVD(n_components=6, algorithm='randomized', n_iter=5, random_state=42)
        fact_cm_med_tr = svd_cm_med.fit_transform(T_tr)
        fact_cm_med_ev = svd_cm_med.transform(T_ev)
        for i in range(6):
            name = f'tensor_cm_med_{i}'
            train_data[name] = fact_cm_med_tr[:, i]
            eval_data[name] = fact_cm_med_ev[:, i]
            extra_numeric.append(name)
        del T_tr, T_ev

        # Latent Pharmacotherapy Regimen Decomposition
        svd_med = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=5, random_state=42)
        med_fact_tr = svd_med.fit_transform(med_tr_mat)
        med_fact_ev = svd_med.transform(med_ev_mat)
        for i in range(3):
            name = f'tensor_med_{i}'
            train_data[name] = med_fact_tr[:, i]
            eval_data[name] = med_fact_ev[:, i]
            extra_numeric.append(name)

    # Multi-Diagnosis Chapter Co-occurrence Syndromes
    chapter_list = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other']
    ch_tr_mat = np.column_stack([
        ((train_data['diag_1_group'] == ch).astype(np.float64) +
         (train_data['diag_2_group'] == ch).astype(np.float64) +
         (train_data['diag_3_group'] == ch).astype(np.float64))
        for ch in chapter_list
    ])
    ch_ev_mat = np.column_stack([
        ((eval_data['diag_1_group'] == ch).astype(np.float64) +
         (eval_data['diag_2_group'] == ch).astype(np.float64) +
         (eval_data['diag_3_group'] == ch).astype(np.float64))
        for ch in chapter_list
    ])

    svd_diag = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=5, random_state=42)
    diag_fact_tr = svd_diag.fit_transform(ch_tr_mat)
    diag_fact_ev = svd_diag.transform(ch_ev_mat)
    for i in range(3):
        name = f'tensor_diag_{i}'
        train_data[name] = diag_fact_tr[:, i]
        eval_data[name] = diag_fact_ev[:, i]
        extra_numeric.append(name)

    # Key Chapter Counts
    for ch_name in ['circulatory', 'respiratory', 'diabetes', 'genitourinary']:
        ch_idx = chapter_list.index(ch_name)
        col_name = f'ch_{ch_name}_cnt'
        train_data[col_name] = ch_tr_mat[:, ch_idx]
        eval_data[col_name] = ch_ev_mat[:, ch_idx]
        extra_numeric.append(col_name)

    # Multi-Diagnosis 3-Digit Prefix Syndromic Phenotyping via SVD
    all_tr_prefixes = np.concatenate([train_data['diag_1_prefix'], train_data['diag_2_prefix'], train_data['diag_3_prefix']])
    mask_valid = (all_tr_prefixes != 'missing')
    unq_prefixes, p_counts = np.unique(all_tr_prefixes[mask_valid], return_counts=True)
    sort_idx = np.argsort(-p_counts)
    top_prefixes = [unq_prefixes[i] for i in sort_idx[:75] if p_counts[i] >= 25]

    if top_prefixes:
        p_to_idx = {p: i for i, p in enumerate(top_prefixes)}
        n_p = len(top_prefixes)
        diag_tr_sparse = np.zeros((n_train_rows, n_p), dtype=np.float64)
        diag_ev_sparse = np.zeros((n_eval_rows, n_p), dtype=np.float64)
        for col in ('diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix'):
            for p, idx in p_to_idx.items():
                diag_tr_sparse[:, idx] = np.maximum(diag_tr_sparse[:, idx], (train_data[col] == p).astype(np.float64))
                diag_ev_sparse[:, idx] = np.maximum(diag_ev_sparse[:, idx], (eval_data[col] == p).astype(np.float64))

        n_comp = min(6, n_p)
        svd_prefix = TruncatedSVD(n_components=n_comp, algorithm='randomized', n_iter=5, random_state=42)
        fact_prefix_tr = svd_prefix.fit_transform(diag_tr_sparse)
        fact_prefix_ev = svd_prefix.transform(diag_ev_sparse)
        for i in range(n_comp):
            col_name = f'svd_diag_syndrome_{i}'
            train_data[col_name] = fact_prefix_tr[:, i]
            eval_data[col_name] = fact_prefix_ev[:, i]
            extra_numeric.append(col_name)

    # 6. Prior Healthcare Utilization Intensity & Encounter Dynamics
    for d in (train_data, eval_data):
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(d['number_emergency'], nan=0.0)
        out = np.nan_to_num(d['number_outpatient'], nan=0.0)
        tot = inp + emg + out
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 0.5)

        d['total_prior_visits'] = tot
        d['inpatient_ratio'] = (inp + 0.1) / (tot + 1.0)
        d['emergency_ratio'] = (emg + 0.1) / (tot + 1.0)
        d['acute_utilization_score'] = 3.0 * inp + 2.0 * emg + out
        d['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d['freq_inpatient'] = (inp >= 2).astype(np.float64)
        d['prior_inp_sq'] = inp ** 2
        d['log_total_visits'] = np.log1p(tot)
        d['super_utilizer'] = ((inp >= 2) & (emg >= 1)).astype(np.float64)

        d['labs_per_day'] = np.nan_to_num(d['num_lab_procedures'], nan=0.0) / los
        d['meds_per_day'] = np.nan_to_num(d['num_medications'], nan=0.0) / los
        d['procs_per_day'] = np.nan_to_num(d['num_procedures'], nan=0.0) / los
        d['meds_per_diag'] = np.nan_to_num(d['num_medications'], nan=0.0) / np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0)
        d['polypharmacy'] = (np.nan_to_num(d['num_medications'], nan=0.0) >= 15).astype(np.float64)
        d['multi_diagnoses'] = (np.nan_to_num(d['number_diagnoses'], nan=0.0) >= 9).astype(np.float64)
        d['diagnostic_intensity'] = (np.nan_to_num(d['num_lab_procedures'], nan=0.0) + np.nan_to_num(d['num_procedures'], nan=0.0)) / los
        d['prior_inp_per_los'] = inp / los
        d['inpatient_los_product'] = inp * los

        # Clinical Inertia & Management Dynamics
        has_change = (d['change'] == 'Ch') if 'change' in d else np.zeros(len(los), dtype=bool)
        if 'A1Cresult' in d:
            a1c_t = (d['A1Cresult'] != 'None')
            a1c_h = np.isin(d['A1Cresult'], ['>7', '>8'])
            d['a1c_tested'] = a1c_t.astype(np.float64)
            d['a1c_high'] = a1c_h.astype(np.float64)
            d['a1c_inertia'] = (a1c_h & ~has_change).astype(np.float64)
            d['a1c_adjusted'] = (a1c_h & has_change).astype(np.float64)
        if 'max_glu_serum' in d:
            glu_t = (d['max_glu_serum'] != 'None')
            glu_h = np.isin(d['max_glu_serum'], ['>200', '>300'])
            d['glu_tested'] = glu_t.astype(np.float64)
            d['glu_high'] = glu_h.astype(np.float64)
            d['glu_inertia'] = (glu_h & ~has_change).astype(np.float64)
            d['glu_adjusted'] = (glu_h & has_change).astype(np.float64)
        if 'change' in d:
            d['med_changed'] = has_change.astype(np.float64)
        if 'diabetesMed' in d:
            d['on_diabetes_med'] = (d['diabetesMed'] == 'Yes').astype(np.float64)

    extra_numeric.extend([
        'total_prior_visits', 'inpatient_ratio', 'emergency_ratio', 'acute_utilization_score',
        'has_prior_inpatient', 'freq_inpatient', 'prior_inp_sq', 'log_total_visits', 'super_utilizer',
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'meds_per_diag', 'polypharmacy',
        'multi_diagnoses', 'diagnostic_intensity', 'prior_inp_per_los', 'inpatient_los_product'
    ])
    if 'A1Cresult' in train_data:
        extra_numeric.extend(['a1c_tested', 'a1c_high', 'a1c_inertia', 'a1c_adjusted'])
    if 'max_glu_serum' in train_data:
        extra_numeric.extend(['glu_tested', 'glu_high', 'glu_inertia', 'glu_adjusted'])
    if 'change' in train_data:
        extra_numeric.append('med_changed')
    if 'diabetesMed' in train_data:
        extra_numeric.append('on_diabetes_med')

    # 7. Age Continuous Mapping & High-Risk Dispositions
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for d in (train_data, eval_data):
        d['age_numeric'] = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in d['age']], dtype=np.float64)
        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = np.isin(disp, ['3', '4', '5', '22', '23', '24']).astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)
        d['fragile_elderly_discharge'] = ((d['age_numeric'] >= 70.0) & (d['disch_snf'] == 1.0)).astype(np.float64)
    extra_numeric.extend(['age_numeric', 'disch_home', 'disch_snf', 'disch_ama', 'disch_home_health', 'fragile_elderly_discharge'])

    # 8. Medication Regimen Complexity & Insulin Dynamics
    for d in (train_data, eval_data):
        n_rows = len(d['time_in_hospital'])
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        changed_cnt = np.zeros(n_rows, dtype=np.float64)
        for m in avail_meds:
            col = d[m]
            active_cnt += np.isin(col, ['Steady', 'Up', 'Down']).astype(np.float64)
            changed_cnt += np.isin(col, ['Up', 'Down']).astype(np.float64)
        d['num_active_meds'] = active_cnt
        d['num_changed_meds'] = changed_cnt

        if 'insulin' in d:
            ins = d['insulin']
            d['insulin_status'] = np.select([ins == 'Steady', ins == 'Down', ins == 'Up'], [1.0, 2.0, 3.0], default=0.0)
            d['insulin_unstable'] = np.isin(ins, ['Up', 'Down']).astype(np.float64)
    extra_numeric.extend(['num_active_meds', 'num_changed_meds'])
    if 'insulin' in train_data:
        extra_numeric.extend(['insulin_status', 'insulin_unstable'])

    # 9. Tabular Encoding
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_data, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 10. Heterogeneous Ensembling & Multi-Objective Gradient Tree Optimization
    # Model 1: Balanced Moderate LightGBM
    model_balanced = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=2.0,
        reg_alpha=0.1,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_balanced.fit(x_train, y_train, categorical_feature=cat_indices)

    # Model 2: Deeper Non-Linear Interaction LightGBM
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

    # Model 3: Risk-Weighted Minority-Focused LightGBM
    model_focused = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=24,
        max_depth=5,
        min_child_samples=200,
        scale_pos_weight=1.25,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=4.0,
        reg_alpha=0.2,
        cat_smooth=50.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    model_focused.fit(x_train, y_train, categorical_feature=cat_indices)

    # Model 4: Extremely Randomized GBDT LightGBM (Variance-reducing orthogonal tree splits)
    try:
        model_et = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.018,
            num_leaves=36,
            extra_trees=True,
            min_child_samples=120,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=3.0,
            reg_alpha=0.2,
            cat_smooth=35.0,
            random_state=333,
            n_jobs=1,
            verbose=-1,
        )
        model_et.fit(x_train, y_train, categorical_feature=cat_indices)
    except Exception:
        model_et = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.018,
            num_leaves=36,
            min_child_samples=120,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=3.0,
            reg_alpha=0.2,
            cat_smooth=35.0,
            random_state=333,
            n_jobs=1,
            verbose=-1,
        )
        model_et.fit(x_train, y_train, categorical_feature=cat_indices)

    # Model 5: Exact Second-Order Taylor Histogram XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=220,
        learning_rate=0.025,
        max_depth=5,
        subsample=0.80,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=2.5,
        tree_method='hist',
        random_state=2024,
        n_jobs=1,
        eval_metric='logloss',
        verbosity=0,
    )
    model_xgb.fit(x_train, y_train)

    p_bal = model_balanced.predict_proba(x_eval)[:, 1]
    p_deep = model_interaction.predict_proba(x_eval)[:, 1]
    p_foc = model_focused.predict_proba(x_eval)[:, 1]
    p_et = model_et.predict_proba(x_eval)[:, 1]
    p_xgb = model_xgb.predict_proba(x_eval)[:, 1]

    # 11. Optimal Rank-Concordance Ensembling with Quantile Calibration
    r_bal = rankdata(p_bal) / n_eval_rows
    r_deep = rankdata(p_deep) / n_eval_rows
    r_foc = rankdata(p_foc) / n_eval_rows
    r_et = rankdata(p_et) / n_eval_rows
    r_xgb = rankdata(p_xgb) / n_eval_rows

    final_rank = 0.25 * r_bal + 0.25 * r_deep + 0.16 * r_foc + 0.16 * r_et + 0.18 * r_xgb
    p_raw = 0.25 * p_bal + 0.25 * p_deep + 0.16 * p_foc + 0.16 * p_et + 0.18 * p_xgb

    # Rank-preserving monotonic quantile mapping: preserves optimal ROC-AUC
    # while calibrating probabilities to the true empirical prevalence distribution
    r_order = np.argsort(final_rank)
    calibrated_pred = np.empty_like(p_raw)
    calibrated_pred[r_order] = np.sort(p_raw)

    return calibrated_pred


# EVOLVE-BLOCK-END
# =====================================================================
