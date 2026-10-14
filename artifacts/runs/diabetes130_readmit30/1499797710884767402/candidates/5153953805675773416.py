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
    """Radically simplified pipeline: Empirical-Bayes diagnosis pooling & fast dual-structure LightGBMs."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)
    n_ev = len(eval_cols['gender'])

    age_midpoints = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }

    def safe_str(val):
        if val is None or val == '?' or val == '':
            return '__missing__'
        try:
            if isinstance(val, float) and np.isnan(val):
                return '__missing__'
        except Exception:
            pass
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        try:
            if isinstance(code, float) and np.isnan(code):
                return 'missing'
        except Exception:
            pass
        return str(code).strip().split('.')[0]

    # 1. Clean Core Features & Dynamics (No bloated manual condition booleans)
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])

        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf'] = (disp == '3').astype(float)
        d['disch_is_home_health'] = np.isin(disp, ['6', '8']).astype(float)
        d['disch_is_facility'] = np.isin(disp, ['2', '3', '4', '5', '6', '8', '22', '23', '24', '27', '28', '29', '30']).astype(float)

        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_acute_adm'] = np.isin(adm_type, ['1', '2', '7']).astype(float)
        d['adm_from_er'] = (adm_src == '7').astype(float)
        d['er_to_facility'] = (d['adm_from_er'] * d['disch_is_facility']).astype(float)

        stay = np.maximum(0.0, d['time_in_hospital'])
        tot_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        acute_v = d['number_emergency'] + d['number_inpatient']

        d['total_visits'] = tot_v.astype(float)
        d['acute_visits'] = acute_v.astype(float)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)
        d['super_utilizer'] = (acute_v >= 3).astype(float)

        d['log_inpatient'] = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_total_visits'] = np.log1p(np.maximum(0.0, tot_v))

        d['inpatient_ratio'] = d['number_inpatient'] / (tot_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (tot_v + 1.0)

        d['med_per_day'] = d['num_medications'] / (stay + 1.0)
        d['lab_per_day'] = d['num_lab_procedures'] / (stay + 1.0)
        d['proc_per_day'] = d['num_procedures'] / (stay + 1.0)
        d['diag_per_day'] = d['number_diagnoses'] / (stay + 1.0)
        d['clinical_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / (stay + 1.0)

        d['high_util_long_stay'] = ((d['number_inpatient'] > 0) & (stay >= 6)).astype(float)
        d['utilization_x_stay'] = d['log_inpatient'] * np.log1p(stay)
        d['utilization_x_facility'] = d['log_inpatient'] * d['disch_is_facility']

        # Medication summary
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        for col in [c for c in task.categorical_columns if c not in {
            'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
            'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
        }]:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)

        d['active_med_count'] = active
        d['med_change_count'] = changes

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['on_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['insulin_steady'] = (insulin == 'Steady').astype(float)

        met = d.get('metformin', np.array(['No'] * n_rows))
        d['on_metformin'] = (met != 'No').astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(float)

        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)
        d['therapeutic_inertia'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)

        spec_str = np.array([safe_str(x) for x in d['medical_specialty']])
        payer_str = np.array([safe_str(x) for x in d['payer_code']])
        d['payer_missing'] = np.isin(payer_str, ['__missing__', '?']).astype(float)
        d['spec_missing'] = np.isin(spec_str, ['__missing__', '?']).astype(float)

        # Primary pathways
        d['diag1_disch'] = np.array([f"{g}_{p}" for g, p in zip(d['diag_1_grp'], disp)], dtype=object)
        d['adm_disch'] = np.array([f"{a}_{p}" for a, p in zip(adm_src, disp)], dtype=object)

    # 2. Universal Empirical-Bayes Pooled Diagnosis Hazard & Target Encodings
    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

    def compute_freq_map(arr):
        n = len(arr)
        counts = {}
        for x in arr:
            s = safe_str(x)
            counts[s] = counts.get(s, 0) + 1
        return {k: v / n for k, v in counts.items()}

    for f_name, col_key in [
        ('freq_diag_1', 'diag_1_3d'),
        ('freq_specialty', 'medical_specialty'),
        ('freq_discharge', 'discharge_disposition_id'),
        ('freq_adm_source', 'admission_source_id'),
    ]:
        f_map = compute_freq_map(train[col_key])
        train[f_name] = np.array([f_map.get(safe_str(x), 0.0) for x in train[col_key]], dtype=float)
        eval_cols[f_name] = np.array([f_map.get(safe_str(x), 0.0) for x in eval_cols[col_key]], dtype=float)

    te_features = [
        'te_diag_1', 'te_diag_max', 'te_diag_mean', 'te_diag_sum',
        'te_discharge', 'te_specialty', 'te_adm_source', 'te_adm_type', 'te_payer',
        'te_diag1_disch', 'te_adm_disch'
    ]
    for tf in te_features:
        train[tf] = np.zeros(n_tr, dtype=float)

    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_te.split(train['gender'], y_train):
        y_tr_f = y_train[tr_idx]

        m_d1 = compute_smoothed_map(train['diag_1_3d'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_diag_1'][val_idx] = np.array([m_d1.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])

        # Symmetrically pooled diagnosis hazard across all 3 diagnosis slots
        pooled_d = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_y = np.concatenate([y_tr_f, y_tr_f, y_tr_f])
        m_pool = compute_smoothed_map(pooled_d, pooled_y, smooth=25.0, prior=prior_glob)

        pd1 = np.array([m_pool.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        pd2 = np.array([m_pool.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        pd3 = np.array([m_pool.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        train['te_diag_max'][val_idx] = np.maximum(np.maximum(pd1, pd2), pd3)
        train['te_diag_mean'][val_idx] = (pd1 + pd2 + pd3) / 3.0
        train['te_diag_sum'][val_idx] = pd1 + pd2 + pd3

        m_disp = compute_smoothed_map(train['discharge_disposition_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_discharge'][val_idx] = np.array([m_disp.get(safe_str(c), prior_glob) for c in train['discharge_disposition_id'][val_idx]])

        m_spec = compute_smoothed_map(train['medical_specialty'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_specialty'][val_idx] = np.array([m_spec.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

        m_src = compute_smoothed_map(train['admission_source_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_adm_source'][val_idx] = np.array([m_src.get(safe_str(c), prior_glob) for c in train['admission_source_id'][val_idx]])

        m_adm = compute_smoothed_map(train['admission_type_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_adm_type'][val_idx] = np.array([m_adm.get(safe_str(c), prior_glob) for c in train['admission_type_id'][val_idx]])

        m_pay = compute_smoothed_map(train['payer_code'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_payer'][val_idx] = np.array([m_pay.get(safe_str(c), prior_glob) for c in train['payer_code'][val_idx]])

        m_d1_disch = compute_smoothed_map(train['diag1_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_diag1_disch'][val_idx] = np.array([m_d1_disch.get(safe_str(c), prior_glob) for c in train['diag1_disch'][val_idx]])

        m_adm_disch = compute_smoothed_map(train['adm_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_adm_disch'][val_idx] = np.array([m_adm_disch.get(safe_str(c), prior_glob) for c in train['adm_disch'][val_idx]])

    # Full train-to-eval mappings
    m_d1_all = compute_smoothed_map(train['diag_1_3d'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_diag_1'] = np.array([m_d1_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']], dtype=float)

    pooled_d_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    m_pool_all = compute_smoothed_map(pooled_d_all, pooled_y_all, smooth=25.0, prior=prior_glob)
    epd1 = np.array([m_pool_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    epd2 = np.array([m_pool_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    epd3 = np.array([m_pool_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_max'] = np.maximum(np.maximum(epd1, epd2), epd3)
    eval_cols['te_diag_mean'] = (epd1 + epd2 + epd3) / 3.0
    eval_cols['te_diag_sum'] = epd1 + epd2 + epd3

    m_disp_all = compute_smoothed_map(train['discharge_disposition_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_discharge'] = np.array([m_disp_all.get(safe_str(c), prior_glob) for c in eval_cols['discharge_disposition_id']], dtype=float)

    m_spec_all = compute_smoothed_map(train['medical_specialty'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([m_spec_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']], dtype=float)

    m_src_all = compute_smoothed_map(train['admission_source_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_source'] = np.array([m_src_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_source_id']], dtype=float)

    m_adm_all = compute_smoothed_map(train['admission_type_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_type'] = np.array([m_adm_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_type_id']], dtype=float)

    m_pay_all = compute_smoothed_map(train['payer_code'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_payer'] = np.array([m_pay_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_code']], dtype=float)

    m_d1_disch_all = compute_smoothed_map(train['diag1_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_diag1_disch'] = np.array([m_d1_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_disch']], dtype=float)

    m_adm_disch_all = compute_smoothed_map(train['adm_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_adm_disch'] = np.array([m_adm_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['adm_disch']], dtype=float)

    extra_num = [
        'age_num', 'total_visits', 'acute_visits', 'has_prior_inpatient', 'frequent_inpatient', 'super_utilizer',
        'log_inpatient', 'log_emergency', 'log_total_visits', 'inpatient_ratio', 'emergency_ratio',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'diag_per_day', 'clinical_intensity',
        'high_util_long_stay', 'utilization_x_stay', 'utilization_x_facility',
        'active_med_count', 'med_change_count', 'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady',
        'on_metformin', 'change_flag', 'diabetes_med_flag', 'polypharmacy',
        'a1c_tested', 'a1c_high', 'glu_high', 'therapeutic_inertia',
        'disch_is_home', 'disch_is_snf', 'disch_is_home_health', 'disch_is_facility',
        'is_emergency_adm', 'is_acute_adm', 'adm_from_er', 'er_to_facility',
        'payer_missing', 'spec_missing',
        'freq_diag_1', 'freq_specialty', 'freq_discharge', 'freq_adm_source',
    ] + te_features

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag1_disch', 'adm_disch'
    ]

    # 3. Clean Matrix Transformation
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 4. Multi-Architectural Fast LightGBM Ensemble across 5 Stratified Folds
    skf_models = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_1 = np.zeros(n_tr, dtype=float)
    oof_2 = np.zeros(n_tr, dtype=float)
    oof_3 = np.zeros(n_tr, dtype=float)

    eval_1 = np.zeros(n_ev, dtype=float)
    eval_2 = np.zeros(n_ev, dtype=float)
    eval_3 = np.zeros(n_ev, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # Model 1: Deep Leaf-Wise Explorer
        m1 = lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_split_gain=0.01,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_1[val_idx] = m1.predict_proba(x_va)[:, 1]
        eval_1 += m1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: Depth-Constrained Conservative Tree
        m2 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=22,
            min_child_samples=60,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=2.5,
            reg_lambda=10.0,
            min_split_gain=0.02,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_2[val_idx] = m2.predict_proba(x_va)[:, 1]
        eval_2 += m2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: Broad Heavy-Regularization Smoothing
        m3 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.035,
            max_depth=6,
            num_leaves=28,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=4.0,
            reg_lambda=15.0,
            min_split_gain=0.03,
            random_state=242 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m3.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_3[val_idx] = m3.predict_proba(x_va)[:, 1]
        eval_3 += m3.predict_proba(x_eval)[:, 1] / 5.0

    # 5. Exact Out-of-Fold Simplex Grid-Search (Instant, deterministic AUC optimization)
    best_score = -1.0
    best_weights = (0.4, 0.4, 0.2)
    for i in range(21):
        w1 = i * 0.05
        for j in range(21 - i):
            w2 = j * 0.05
            w3 = round(1.0 - w1 - w2, 4)
            if w3 >= -1e-6:
                blend_oof = w1 * oof_1 + w2 * oof_2 + max(0.0, w3) * oof_3
                score = roc_auc_score(y_train, blend_oof)
                if score > best_score:
                    best_score = score
                    best_weights = (w1, w2, max(0.0, w3))

    w1, w2, w3 = best_weights
    final_pred = w1 * eval_1 + w2 * eval_2 + w3 * eval_3
    return np.clip(np.nan_to_num(final_pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
