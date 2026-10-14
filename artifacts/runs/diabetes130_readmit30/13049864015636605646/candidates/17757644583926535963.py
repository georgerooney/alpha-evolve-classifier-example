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
    # Shallow copy dicts so caller structures are not mutated
    train = dict(train)
    eval_cols = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities extraction via clinical mapping
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for cm_key in cm_train:
        train[cm_key] = cm_train[cm_key].astype(np.float64)
        eval_cols[cm_key] = cm_eval[cm_key].astype(np.float64)
        extra_numeric.append(cm_key)

    # 2. ICD-9 Chapter Grouping
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f'{diag_col}_group'
        train[grp_name] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_cols[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(grp_name)

    for grp in ['circulatory', 'diabetes', 'respiratory', 'genitourinary']:
        flag_name = f'has_diag_grp_{grp}'
        train[flag_name] = (
            (train['diag_1_group'] == grp)
            | (train['diag_2_group'] == grp)
            | (train['diag_3_group'] == grp)
        ).astype(np.float64)
        eval_cols[flag_name] = (
            (eval_cols['diag_1_group'] == grp)
            | (eval_cols['diag_2_group'] == grp)
            | (eval_cols['diag_3_group'] == grp)
        ).astype(np.float64)
        extra_numeric.append(flag_name)

    # 3. Smoothed Out-Of-Fold Target Encodings of High-Cardinality Categoricals
    te_columns = [
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'medical_specialty',
        'diag_1',
        'diag_2',
        'diag_3',
        'payer_code',
    ]
    for col in te_columns:
        te_tr, te_ev = oof_target_encode(
            train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
        )
        te_col_name = f'{col}_te'
        train[te_col_name] = te_tr.astype(np.float64)
        eval_cols[te_col_name] = te_ev.astype(np.float64)
        extra_numeric.append(te_col_name)

    # Aggregate diagnosis risk severity
    train['diag_te_mean'] = (train['diag_1_te'] + train['diag_2_te'] + train['diag_3_te']) / 3.0
    eval_cols['diag_te_mean'] = (eval_cols['diag_1_te'] + eval_cols['diag_2_te'] + eval_cols['diag_3_te']) / 3.0
    extra_numeric.append('diag_te_mean')

    train['diag_te_max'] = np.maximum(np.maximum(train['diag_1_te'], train['diag_2_te']), train['diag_3_te'])
    eval_cols['diag_te_max'] = np.maximum(np.maximum(eval_cols['diag_1_te'], eval_cols['diag_2_te']), eval_cols['diag_3_te'])
    extra_numeric.append('diag_te_max')

    # Interaction target encodings: clinical pathway and discharge disposition
    diag1_disc_tr = np.array([f'{g}_{d}' for g, d in zip(train['diag_1_group'], train['discharge_disposition_id'])], dtype=object)
    diag1_disc_ev = np.array([f'{g}_{d}' for g, d in zip(eval_cols['diag_1_group'], eval_cols['discharge_disposition_id'])], dtype=object)
    te_d1d_tr, te_d1d_ev = oof_target_encode(diag1_disc_tr, y_train, diag1_disc_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_diag1_disc'] = te_d1d_tr.astype(np.float64)
    eval_cols['te_diag1_disc'] = te_d1d_ev.astype(np.float64)
    extra_numeric.append('te_diag1_disc')

    adm_disc_tr = np.array([f'{a}_{d}' for a, d in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    adm_disc_ev = np.array([f'{a}_{d}' for a, d in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    te_ad_tr, te_ad_ev = oof_target_encode(adm_disc_tr, y_train, adm_disc_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_adm_disc'] = te_ad_tr.astype(np.float64)
    eval_cols['te_adm_disc'] = te_ad_ev.astype(np.float64)
    extra_numeric.append('te_adm_disc')

    # 4. Low-Rank Tensor Factor Decomposition on Clinical Condition & Medication Space
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]

    # Find frequent ICD-9 diagnosis codes across all 3 diagnosis positions
    diag_counts = {}
    for col in ['diag_1', 'diag_2', 'diag_3']:
        for val in train[col]:
            if val is not None and val != '?':
                diag_counts[val] = diag_counts.get(val, 0) + 1
    sorted_diags = sorted(diag_counts.items(), key=lambda x: x[1], reverse=True)
    top_diags = [code for code, count in sorted_diags[:50] if count >= 20]

    tensor_slice_tr = []
    tensor_slice_ev = []
    for code in top_diags:
        tensor_slice_tr.append(
            ((train['diag_1'] == code) | (train['diag_2'] == code) | (train['diag_3'] == code)).astype(np.float64)
        )
        tensor_slice_ev.append(
            ((eval_cols['diag_1'] == code) | (eval_cols['diag_2'] == code) | (eval_cols['diag_3'] == code)).astype(np.float64)
        )

    for med in med_cols:
        tensor_slice_tr.append(((train[med] != 'No') & (train[med] is not None)).astype(np.float64))
        tensor_slice_ev.append(((eval_cols[med] != 'No') & (eval_cols[med] is not None)).astype(np.float64))

    for cm_key in ['cm_mi', 'cm_chf', 'cm_pvd', 'cm_pulmonary', 'cm_renal', 'cm_diabetes_complicated']:
        tensor_slice_tr.append(train[cm_key])
        tensor_slice_ev.append(eval_cols[cm_key])

    M_tr = np.column_stack(tensor_slice_tr)
    M_ev = np.column_stack(tensor_slice_ev)
    M_mean = np.mean(M_tr, axis=0, keepdims=True)
    M_tr_c = M_tr - M_mean
    M_ev_c = M_ev - M_mean

    # Low-rank factor projection via sample covariance eigendecomposition
    cov_mat = (M_tr_c.T @ M_tr_c) / max(len(M_tr) - 1.0, 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx_sorted = np.argsort(eigvals)[::-1]
    n_latent = min(8, M_tr.shape[1])
    V_factors = eigvecs[:, idx_sorted[:n_latent]]

    latents_tr = M_tr_c @ V_factors
    latents_ev = M_ev_c @ V_factors
    for j in range(n_latent):
        latent_name = f'latent_phenotype_{j}'
        train[latent_name] = latents_tr[:, j].astype(np.float64)
        eval_cols[latent_name] = latents_ev[:, j].astype(np.float64)
        extra_numeric.append(latent_name)

    # 5. Prior Healthcare Utilization Intensity Signals
    inp_tr = np.nan_to_num(train['number_inpatient'], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols['number_inpatient'], nan=0.0)
    emg_tr = np.nan_to_num(train['number_emergency'], nan=0.0)
    emg_ev = np.nan_to_num(eval_cols['number_emergency'], nan=0.0)
    otp_tr = np.nan_to_num(train['number_outpatient'], nan=0.0)
    otp_ev = np.nan_to_num(eval_cols['number_outpatient'], nan=0.0)

    tot_vis_tr = inp_tr + emg_tr + otp_tr
    tot_vis_ev = inp_ev + emg_ev + otp_ev

    train['has_inpatient'] = (inp_tr > 0).astype(np.float64)
    eval_cols['has_inpatient'] = (inp_ev > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train['has_emergency'] = (emg_tr > 0).astype(np.float64)
    eval_cols['has_emergency'] = (emg_ev > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    train['total_prior_visits'] = tot_vis_tr.astype(np.float64)
    eval_cols['total_prior_visits'] = tot_vis_ev.astype(np.float64)
    extra_numeric.append('total_prior_visits')

    train['inpatient_ratio'] = (inp_tr / (tot_vis_tr + 1.0)).astype(np.float64)
    eval_cols['inpatient_ratio'] = (inp_ev / (tot_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('inpatient_ratio')

    train['log_inpatient'] = np.log1p(np.maximum(inp_tr, 0.0)).astype(np.float64)
    eval_cols['log_inpatient'] = np.log1p(np.maximum(inp_ev, 0.0)).astype(np.float64)
    extra_numeric.append('log_inpatient')

    train['log_total_visits'] = np.log1p(np.maximum(tot_vis_tr, 0.0)).astype(np.float64)
    eval_cols['log_total_visits'] = np.log1p(np.maximum(tot_vis_ev, 0.0)).astype(np.float64)
    extra_numeric.append('log_total_visits')

    # 6. Stay and Encounter Intensity Rates
    los_tr = np.maximum(np.nan_to_num(train['time_in_hospital'], nan=1.0), 1.0)
    los_ev = np.maximum(np.nan_to_num(eval_cols['time_in_hospital'], nan=1.0), 1.0)
    labs_tr = np.nan_to_num(train['num_lab_procedures'], nan=0.0)
    labs_ev = np.nan_to_num(eval_cols['num_lab_procedures'], nan=0.0)
    procs_tr = np.nan_to_num(train['num_procedures'], nan=0.0)
    procs_ev = np.nan_to_num(eval_cols['num_procedures'], nan=0.0)
    meds_tr = np.nan_to_num(train['num_medications'], nan=0.0)
    meds_ev = np.nan_to_num(eval_cols['num_medications'], nan=0.0)
    diags_tr = np.nan_to_num(train['number_diagnoses'], nan=0.0)
    diags_ev = np.nan_to_num(eval_cols['number_diagnoses'], nan=0.0)

    train['lab_per_day'] = (labs_tr / los_tr).astype(np.float64)
    eval_cols['lab_per_day'] = (labs_ev / los_ev).astype(np.float64)
    extra_numeric.append('lab_per_day')

    train['med_per_day'] = (meds_tr / los_tr).astype(np.float64)
    eval_cols['med_per_day'] = (meds_ev / los_ev).astype(np.float64)
    extra_numeric.append('med_per_day')

    train['diag_per_day'] = (diags_tr / los_tr).astype(np.float64)
    eval_cols['diag_per_day'] = (diags_ev / los_ev).astype(np.float64)
    extra_numeric.append('diag_per_day')

    train['proc_per_day'] = (procs_tr / los_tr).astype(np.float64)
    eval_cols['proc_per_day'] = (procs_ev / los_ev).astype(np.float64)
    extra_numeric.append('proc_per_day')

    train['meds_per_lab'] = (meds_tr / np.maximum(labs_tr, 1.0)).astype(np.float64)
    eval_cols['meds_per_lab'] = (meds_ev / np.maximum(labs_ev, 1.0)).astype(np.float64)
    extra_numeric.append('meds_per_lab')

    # 7. Medication Modification Dynamics & Insulin Indicators
    n_tr = len(los_tr)
    n_ev = len(los_ev)
    changes_tr = np.zeros(n_tr, dtype=np.float64)
    changes_ev = np.zeros(n_ev, dtype=np.float64)
    med_up_tr = np.zeros(n_tr, dtype=np.float64)
    med_up_ev = np.zeros(n_ev, dtype=np.float64)
    med_down_tr = np.zeros(n_tr, dtype=np.float64)
    med_down_ev = np.zeros(n_ev, dtype=np.float64)
    med_active_tr = np.zeros(n_tr, dtype=np.float64)
    med_active_ev = np.zeros(n_ev, dtype=np.float64)

    for m in med_cols:
        changes_tr += np.isin(train[m], ['Up', 'Down']).astype(np.float64)
        changes_ev += np.isin(eval_cols[m], ['Up', 'Down']).astype(np.float64)
        med_up_tr += (train[m] == 'Up').astype(np.float64)
        med_up_ev += (eval_cols[m] == 'Up').astype(np.float64)
        med_down_tr += (train[m] == 'Down').astype(np.float64)
        med_down_ev += (eval_cols[m] == 'Down').astype(np.float64)
        med_active_tr += ((train[m] != 'No') & (train[m] is not None)).astype(np.float64)
        med_active_ev += ((eval_cols[m] != 'No') & (eval_cols[m] is not None)).astype(np.float64)

    train['num_med_changes'] = changes_tr
    eval_cols['num_med_changes'] = changes_ev
    extra_numeric.append('num_med_changes')

    train['num_med_up'] = med_up_tr
    eval_cols['num_med_up'] = med_up_ev
    extra_numeric.append('num_med_up')

    train['num_med_down'] = med_down_tr
    eval_cols['num_med_down'] = med_down_ev
    extra_numeric.append('num_med_down')

    train['num_active_meds'] = med_active_tr
    eval_cols['num_active_meds'] = med_active_ev
    extra_numeric.append('num_active_meds')

    train['insulin_is_up'] = (train['insulin'] == 'Up').astype(np.float64)
    eval_cols['insulin_is_up'] = (eval_cols['insulin'] == 'Up').astype(np.float64)
    extra_numeric.append('insulin_is_up')

    train['insulin_is_down'] = (train['insulin'] == 'Down').astype(np.float64)
    eval_cols['insulin_is_down'] = (eval_cols['insulin'] == 'Down').astype(np.float64)
    extra_numeric.append('insulin_is_down')

    train['insulin_is_active'] = (train['insulin'] != 'No').astype(np.float64)
    eval_cols['insulin_is_active'] = (eval_cols['insulin'] != 'No').astype(np.float64)
    extra_numeric.append('insulin_is_active')

    # Lab testing indicators
    train['a1c_tested'] = (train['A1Cresult'] != 'None').astype(np.float64)
    eval_cols['a1c_tested'] = (eval_cols['A1Cresult'] != 'None').astype(np.float64)
    extra_numeric.append('a1c_tested')

    train['glu_serum_tested'] = (train['max_glu_serum'] != 'None').astype(np.float64)
    eval_cols['glu_serum_tested'] = (eval_cols['max_glu_serum'] != 'None').astype(np.float64)
    extra_numeric.append('glu_serum_tested')

    train['discharged_home'] = (train['discharge_disposition_id'] == '1').astype(np.float64)
    eval_cols['discharged_home'] = (eval_cols['discharge_disposition_id'] == '1').astype(np.float64)
    extra_numeric.append('discharged_home')

    # Deduplicate extra feature names
    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 8. Encode Tabular Data
    encoder = TabularEncoder(min_count=30)
    encoder.fit(
        task,
        train,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    cat_indices = encoder.categorical_indices

    # 9. Diverse Gradient-Boosted Model Ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=160,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.05,
        reg_lambda=1.5,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m2 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.022,
        num_leaves=20,
        min_child_samples=250,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.2,
        reg_lambda=5.0,
        cat_smooth=75.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m3 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.028,
        num_leaves=42,
        min_child_samples=120,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.01,
        reg_lambda=0.8,
        cat_smooth=30.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    m3.fit(x_train, y_train, categorical_feature=cat_indices)

    p1 = m1.predict_proba(x_eval)[:, 1]
    p2 = m2.predict_proba(x_eval)[:, 1]
    p3 = m3.predict_proba(x_eval)[:, 1]

    preds = (p1 + p2 + p3) / 3.0
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
