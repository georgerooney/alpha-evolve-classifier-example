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
    # Create working copies of column dictionaries
    train_c = dict(train)
    eval_c = dict(eval_cols)

    extra_num = []
    extra_cat = []

    # 1. Charlson-style clinical comorbidities and weighted index
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for cm_key, cm_arr in cm_tr.items():
        train_c[cm_key] = cm_arr
        eval_c[cm_key] = cm_ev[cm_key]
        extra_num.append(cm_key)

    charlson_score_tr = (
        cm_tr['cm_mi'] + cm_tr['cm_chf'] + cm_tr['cm_pvd'] + cm_tr['cm_cerebrovascular'] +
        cm_tr['cm_dementia'] + cm_tr['cm_pulmonary'] + cm_tr['cm_rheumatic'] + cm_tr['cm_peptic_ulcer'] +
        cm_tr['cm_liver'] + 2.0 * cm_tr['cm_hemiplegia'] + 2.0 * cm_tr['cm_renal'] +
        2.0 * cm_tr['cm_cancer'] + 4.0 * cm_tr['cm_metastatic'] + 4.0 * cm_tr['cm_hiv'] +
        2.0 * cm_tr['cm_diabetes_complicated']
    )
    charlson_score_ev = (
        cm_ev['cm_mi'] + cm_ev['cm_chf'] + cm_ev['cm_pvd'] + cm_ev['cm_cerebrovascular'] +
        cm_ev['cm_dementia'] + cm_ev['cm_pulmonary'] + cm_ev['cm_rheumatic'] + cm_ev['cm_peptic_ulcer'] +
        cm_ev['cm_liver'] + 2.0 * cm_ev['cm_hemiplegia'] + 2.0 * cm_ev['cm_renal'] +
        2.0 * cm_ev['cm_cancer'] + 4.0 * cm_ev['cm_metastatic'] + 4.0 * cm_ev['cm_hiv'] +
        2.0 * cm_ev['cm_diabetes_complicated']
    )
    train_c['charlson_score'] = charlson_score_tr
    eval_c['charlson_score'] = charlson_score_ev
    extra_num.append('charlson_score')

    train_c['cardiorenal'] = (cm_tr['cm_chf'] * cm_tr['cm_renal']).astype(np.float64)
    eval_c['cardiorenal'] = (cm_ev['cm_chf'] * cm_ev['cm_renal']).astype(np.float64)
    extra_num.append('cardiorenal')

    train_c['vascular_cluster'] = np.clip(cm_tr['cm_mi'] + cm_tr['cm_chf'] + cm_tr['cm_pvd'] + cm_tr['cm_cerebrovascular'], 0.0, 1.0)
    eval_c['vascular_cluster'] = np.clip(cm_ev['cm_mi'] + cm_ev['cm_chf'] + cm_ev['cm_pvd'] + cm_ev['cm_cerebrovascular'], 0.0, 1.0)
    extra_num.append('vascular_cluster')

    # 2. ICD-9 Chapter Groups & 3-Digit Category Prefix mapping
    def get_icd9_prefix(val):
        if val is None or not isinstance(val, str) or val == '?' or val == '':
            return 'missing'
        return val.split('.')[0]

    for d_col in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f"{d_col}_group"
        train_c[grp_name] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_c[grp_name] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_cat.append(grp_name)

        pfx_name = f"{d_col}_prefix"
        train_c[pfx_name] = np.array([get_icd9_prefix(c) for c in train[d_col]], dtype=object)
        eval_c[pfx_name] = np.array([get_icd9_prefix(c) for c in eval_cols[d_col]], dtype=object)
        extra_cat.append(pfx_name)

    # 3. Utilization intensity features
    inp_tr = np.nan_to_num(train['number_inpatient'], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols['number_inpatient'], nan=0.0)
    emg_tr = np.nan_to_num(train['number_emergency'], nan=0.0)
    emg_ev = np.nan_to_num(eval_cols['number_emergency'], nan=0.0)
    out_tr = np.nan_to_num(train['number_outpatient'], nan=0.0)
    out_ev = np.nan_to_num(eval_cols['number_outpatient'], nan=0.0)

    total_prior_tr = inp_tr + emg_tr + out_tr
    total_prior_ev = inp_ev + emg_ev + out_ev
    train_c['total_prior_visits'] = total_prior_tr
    eval_c['total_prior_visits'] = total_prior_ev
    extra_num.append('total_prior_visits')

    train_c['has_prior_inpatient'] = (inp_tr > 0).astype(np.float64)
    eval_c['has_prior_inpatient'] = (inp_ev > 0).astype(np.float64)
    extra_num.append('has_prior_inpatient')

    train_c['frequent_inpatient'] = (inp_tr >= 2.0).astype(np.float64)
    eval_c['frequent_inpatient'] = (inp_ev >= 2.0).astype(np.float64)
    extra_num.append('frequent_inpatient')

    train_c['has_prior_emergency'] = (emg_tr > 0).astype(np.float64)
    eval_c['has_prior_emergency'] = (emg_ev > 0).astype(np.float64)
    extra_num.append('has_prior_emergency')

    train_c['inpatient_ratio'] = inp_tr / (total_prior_tr + 1.0)
    eval_c['inpatient_ratio'] = inp_ev / (total_prior_ev + 1.0)
    extra_num.append('inpatient_ratio')

    train_c['emergency_ratio'] = emg_tr / (total_prior_tr + 1.0)
    eval_c['emergency_ratio'] = emg_ev / (total_prior_ev + 1.0)
    extra_num.append('emergency_ratio')

    train_c['inp_x_charlson'] = inp_tr * charlson_score_tr
    eval_c['inp_x_charlson'] = inp_ev * charlson_score_ev
    extra_num.append('inp_x_charlson')

    # 4. Discharge disposition & admission clinical flags and cross-interactions
    disp_tr_str = np.array([str(d) if d is not None else 'missing' for d in train['discharge_disposition_id']], dtype=object)
    disp_ev_str = np.array([str(d) if d is not None else 'missing' for d in eval_cols['discharge_disposition_id']], dtype=object)

    disp_x_inp_tr = np.array([f"{d}_{int(h)}" for d, h in zip(disp_tr_str, inp_tr > 0)], dtype=object)
    disp_x_inp_ev = np.array([f"{d}_{int(h)}" for d, h in zip(disp_ev_str, inp_ev > 0)], dtype=object)
    train_c['disp_x_has_inp'] = disp_x_inp_tr
    eval_c['disp_x_has_inp'] = disp_x_inp_ev
    extra_cat.append('disp_x_has_inp')

    disp_facility = {'2', '3', '4', '5', '22', '23', '24'}
    train_c['disp_is_home'] = np.array([1.0 if d == '1' else 0.0 for d in disp_tr_str], dtype=np.float64)
    eval_c['disp_is_home'] = np.array([1.0 if d == '1' else 0.0 for d in disp_ev_str], dtype=np.float64)
    extra_num.append('disp_is_home')

    train_c['disp_is_facility'] = np.array([1.0 if d in disp_facility else 0.0 for d in disp_tr_str], dtype=np.float64)
    eval_c['disp_is_facility'] = np.array([1.0 if d in disp_facility else 0.0 for d in disp_ev_str], dtype=np.float64)
    extra_num.append('disp_is_facility')

    train_c['disp_is_home_health'] = np.array([1.0 if d == '6' else 0.0 for d in disp_tr_str], dtype=np.float64)
    eval_c['disp_is_home_health'] = np.array([1.0 if d == '6' else 0.0 for d in disp_ev_str], dtype=np.float64)
    extra_num.append('disp_is_home_health')

    adm_src_tr_str = np.array([str(s) if s is not None else 'missing' for s in train['admission_source_id']], dtype=object)
    adm_src_ev_str = np.array([str(s) if s is not None else 'missing' for s in eval_cols['admission_source_id']], dtype=object)
    train_c['adm_is_er'] = np.array([1.0 if s == '7' else 0.0 for s in adm_src_tr_str], dtype=np.float64)
    eval_c['adm_is_er'] = np.array([1.0 if s == '7' else 0.0 for s in adm_src_ev_str], dtype=np.float64)
    extra_num.append('adm_is_er')

    adm_typ_tr_str = np.array([str(t) if t is not None else 'missing' for t in train['admission_type_id']], dtype=object)
    adm_typ_ev_str = np.array([str(t) if t is not None else 'missing' for t in eval_cols['admission_type_id']], dtype=object)
    train_c['adm_is_emergency'] = np.array([1.0 if t in ('1', '2', '7') else 0.0 for t in adm_typ_tr_str], dtype=np.float64)
    eval_c['adm_is_emergency'] = np.array([1.0 if t in ('1', '2', '7') else 0.0 for t in adm_typ_ev_str], dtype=np.float64)
    extra_num.append('adm_is_emergency')

    # Out-of-fold smoothed target encoding for high-cardinality codes
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty',
        'diag_1_group', 'diag_2_group', 'disp_x_has_inp', 'payer_code'
    ]
    prior_mean = float(np.mean(y_train))
    for col in te_cols:
        if col in train_c and col in eval_c:
            tr_te, ev_te = oof_target_encode(
                train_c[col], y_train, eval_c[col],
                n_folds=5, smoothing=25.0, seed=42
            )
            te_name = f"{col}_te"
            train_c[te_name] = np.nan_to_num(tr_te, nan=prior_mean).astype(np.float64)
            eval_c[te_name] = np.nan_to_num(ev_te, nan=prior_mean).astype(np.float64)
            extra_num.append(te_name)

    # 5. Encounter daily intensity ratios
    los_tr = np.maximum(np.nan_to_num(train['time_in_hospital'], nan=1.0), 1.0)
    los_ev = np.maximum(np.nan_to_num(eval_cols['time_in_hospital'], nan=1.0), 1.0)

    train_c['lab_per_day'] = np.nan_to_num(train['num_lab_procedures'], nan=0.0) / los_tr
    eval_c['lab_per_day'] = np.nan_to_num(eval_cols['num_lab_procedures'], nan=0.0) / los_ev
    extra_num.append('lab_per_day')

    train_c['proc_per_day'] = np.nan_to_num(train['num_procedures'], nan=0.0) / los_tr
    eval_c['proc_per_day'] = np.nan_to_num(eval_cols['num_procedures'], nan=0.0) / los_ev
    extra_num.append('proc_per_day')

    train_c['meds_per_day'] = np.nan_to_num(train['num_medications'], nan=0.0) / los_tr
    eval_c['meds_per_day'] = np.nan_to_num(eval_cols['num_medications'], nan=0.0) / los_ev
    extra_num.append('meds_per_day')

    train_c['diag_per_day'] = np.nan_to_num(train['number_diagnoses'], nan=0.0) / los_tr
    eval_c['diag_per_day'] = np.nan_to_num(eval_cols['number_diagnoses'], nan=0.0) / los_ev
    extra_num.append('diag_per_day')

    # 6. Medication pharmacotherapy regimen metrics
    med_list = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    available_meds = [m for m in med_list if m in train and m in eval_cols]

    act_tr = np.zeros(len(y_train), dtype=np.float64)
    act_ev = np.zeros(len(los_ev), dtype=np.float64)
    chg_tr = np.zeros(len(y_train), dtype=np.float64)
    chg_ev = np.zeros(len(los_ev), dtype=np.float64)

    for med in available_meds:
        m_tr = train[med]
        m_ev = eval_cols[med]
        act_tr += ((m_tr != 'No') & (m_tr != None)).astype(np.float64)
        act_ev += ((m_ev != 'No') & (m_ev != None)).astype(np.float64)
        chg_tr += ((m_tr == 'Up') | (m_tr == 'Down')).astype(np.float64)
        chg_ev += ((m_ev == 'Up') | (m_ev == 'Down')).astype(np.float64)

    train_c['num_active_meds'] = act_tr
    eval_c['num_active_meds'] = act_ev
    extra_num.append('num_active_meds')

    train_c['num_changed_meds'] = chg_tr
    eval_c['num_changed_meds'] = chg_ev
    extra_num.append('num_changed_meds')

    if 'insulin' in train:
        ins_tr = train['insulin']
        ins_ev = eval_cols['insulin']
        train_c['insulin_active'] = ((ins_tr != 'No') & (ins_tr != None)).astype(np.float64)
        eval_c['insulin_active'] = ((ins_ev != 'No') & (ins_ev != None)).astype(np.float64)
        extra_num.append('insulin_active')

        train_c['insulin_up'] = (ins_tr == 'Up').astype(np.float64)
        eval_c['insulin_up'] = (ins_ev == 'Up').astype(np.float64)
        extra_num.append('insulin_up')

        train_c['insulin_down'] = (ins_tr == 'Down').astype(np.float64)
        eval_c['insulin_down'] = (ins_ev == 'Down').astype(np.float64)
        extra_num.append('insulin_down')

    # Lab and clinical test flags
    if 'A1Cresult' in train and 'A1Cresult' in eval_cols:
        a1c_tr = train['A1Cresult']
        a1c_ev = eval_cols['A1Cresult']
        train_c['a1c_tested'] = np.array([0.0 if (v is None or v == 'None') else 1.0 for v in a1c_tr], dtype=np.float64)
        eval_c['a1c_tested'] = np.array([0.0 if (v is None or v == 'None') else 1.0 for v in a1c_ev], dtype=np.float64)
        extra_num.append('a1c_tested')

        train_c['a1c_high'] = np.array([1.0 if v in ('>7', '>8') else 0.0 for v in a1c_tr], dtype=np.float64)
        eval_c['a1c_high'] = np.array([1.0 if v in ('>7', '>8') else 0.0 for v in a1c_ev], dtype=np.float64)
        extra_num.append('a1c_high')

    if 'max_glu_serum' in train and 'max_glu_serum' in eval_cols:
        glu_tr = train['max_glu_serum']
        glu_ev = eval_cols['max_glu_serum']
        train_c['glu_tested'] = np.array([0.0 if (v is None or v == 'None') else 1.0 for v in glu_tr], dtype=np.float64)
        eval_c['glu_tested'] = np.array([0.0 if (v is None or v == 'None') else 1.0 for v in glu_ev], dtype=np.float64)
        extra_num.append('glu_tested')

    if 'diabetesMed' in train and 'diabetesMed' in eval_cols:
        train_c['has_diabetes_med'] = np.array([1.0 if v == 'Yes' else 0.0 for v in train['diabetesMed']], dtype=np.float64)
        eval_c['has_diabetes_med'] = np.array([1.0 if v == 'Yes' else 0.0 for v in eval_cols['diabetesMed']], dtype=np.float64)
        extra_num.append('has_diabetes_med')

    if 'change' in train and 'change' in eval_cols:
        train_c['has_med_change'] = np.array([1.0 if v == 'Ch' else 0.0 for v in train['change']], dtype=np.float64)
        eval_c['has_med_change'] = np.array([1.0 if v == 'Ch' else 0.0 for v in eval_cols['change']], dtype=np.float64)
        extra_num.append('has_med_change')

    for m_field in ['medical_specialty', 'payer_code']:
        if m_field in train and m_field in eval_cols:
            f_name = f"{m_field}_missing"
            train_c[f_name] = np.array([1.0 if (v is None or v == '?') else 0.0 for v in train[m_field]], dtype=np.float64)
            eval_c[f_name] = np.array([1.0 if (v is None or v == '?') else 0.0 for v in eval_cols[m_field]], dtype=np.float64)
            extra_num.append(f_name)

    # 7. Monotonic numeric age representation
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    if 'age' in train:
        train_c['age_num'] = np.array([age_map.get(a, 65.0) for a in train['age']], dtype=np.float64)
        eval_c['age_num'] = np.array([age_map.get(a, 65.0) for a in eval_cols['age']], dtype=np.float64)
        extra_num.append('age_num')

    # 8. Low-rank decomposition (PCA / SVD) of multi-morbidity and pharmacotherapy matrix
    decomp_features_tr = []
    decomp_features_ev = []

    # 15 Charlson binary indicators
    for cm_k in [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic',
        'cm_hiv', 'cm_diabetes_complicated'
    ]:
        decomp_features_tr.append(cm_tr[cm_k])
        decomp_features_ev.append(cm_ev[cm_k])

    # Medication active & titration status
    for med in available_meds:
        m_tr = train[med]
        m_ev = eval_cols[med]
        decomp_features_tr.append(((m_tr != 'No') & (m_tr != None)).astype(np.float64))
        decomp_features_ev.append(((m_ev != 'No') & (m_ev != None)).astype(np.float64))
        decomp_features_tr.append(((m_tr == 'Up') | (m_tr == 'Down')).astype(np.float64))
        decomp_features_ev.append(((m_ev == 'Up') | (m_ev == 'Down')).astype(np.float64))

    # Core utilization metrics
    for u_col in ['number_inpatient', 'number_emergency', 'number_outpatient', 'time_in_hospital', 'num_medications', 'number_diagnoses']:
        decomp_features_tr.append(np.log1p(np.maximum(0.0, np.nan_to_num(train[u_col]))))
        decomp_features_ev.append(np.log1p(np.maximum(0.0, np.nan_to_num(eval_cols[u_col]))))

    X_pheno_tr = np.column_stack(decomp_features_tr)
    X_pheno_ev = np.column_stack(decomp_features_ev)

    mu_pheno = np.mean(X_pheno_tr, axis=0)
    std_pheno = np.std(X_pheno_tr, axis=0)
    std_pheno[std_pheno < 1e-6] = 1.0

    X_pheno_tr_norm = (X_pheno_tr - mu_pheno) / std_pheno
    X_pheno_ev_norm = (X_pheno_ev - mu_pheno) / std_pheno

    # Exact eigendecomposition of the d x d correlation matrix
    N_tr = len(X_pheno_tr_norm)
    cov_pheno = (X_pheno_tr_norm.T @ X_pheno_tr_norm) / N_tr
    eigvals, eigvecs = np.linalg.eigh(cov_pheno)
    sort_idx = np.argsort(eigvals)[::-1]
    k_components = 6
    top_eigenvectors = eigvecs[:, sort_idx[:k_components]]

    Z_tr = X_pheno_tr_norm @ top_eigenvectors
    Z_ev = X_pheno_ev_norm @ top_eigenvectors

    for j in range(k_components):
        comp_name = f'svd_pheno_{j}'
        train_c[comp_name] = Z_tr[:, j]
        eval_c[comp_name] = Z_ev[:, j]
        extra_num.append(comp_name)

    # 9. Encoding and Feature Preprocessing
    extra_num = list(dict.fromkeys(extra_num))
    extra_cat = list(dict.fromkeys(extra_cat))

    encoder = TabularEncoder(min_count=30)
    x_train = encoder.fit_transform(task, train_c, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_c)

    # 10. Multi-Topology GBDT Ensemble
    # Model 1: Deep tree (depth 6, 31 leaves) for higher-order interactions
    m1 = lgb.LGBMClassifier(
        n_estimators=340,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=130,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        reg_alpha=0.2,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Shallow, robust tree (depth 4, 16 leaves) for main effects
    m2 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.022,
        num_leaves=16,
        max_depth=4,
        min_child_samples=160,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        cat_smooth=50.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Extra-trees LightGBM (Extremely randomized tree splits for structural diversity)
    m3 = lgb.LGBMClassifier(
        n_estimators=330,
        learning_rate=0.024,
        num_leaves=28,
        max_depth=6,
        min_child_samples=110,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.0,
        cat_smooth=30.0,
        extra_trees=True,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: Heavily regularized orthogonal subspace tree
    m4 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.02,
        num_leaves=22,
        max_depth=5,
        min_child_samples=180,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=3.5,
        reg_alpha=1.0,
        cat_smooth=55.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p4 = m4.predict_proba(x_eval)[:, 1]

    # Weighted ensemble blend
    final_preds = 0.32 * p1 + 0.28 * p2 + 0.22 * p3 + 0.18 * p4
    return np.clip(np.nan_to_num(final_preds, nan=prior_mean), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
