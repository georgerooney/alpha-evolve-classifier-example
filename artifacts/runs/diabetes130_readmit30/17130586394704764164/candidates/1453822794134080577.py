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
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson-style comorbidities from ICD-9 diagnoses
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_feat[k] = v.astype(np.float64)
        eval_feat[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # High-risk cardiorenal and multi-morbidity interactions
    train_feat['cm_chf_and_renal'] = (train_feat['cm_chf'] * train_feat['cm_renal']).astype(np.float64)
    eval_feat['cm_chf_and_renal'] = (eval_feat['cm_chf'] * eval_feat['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_chf_and_renal')

    train_feat['cm_chf_and_diabetes'] = (train_feat['cm_chf'] * train_feat['cm_diabetes_complicated']).astype(np.float64)
    eval_feat['cm_chf_and_diabetes'] = (eval_feat['cm_chf'] * eval_feat['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_chf_and_diabetes')

    # 2. ICD-9 Chapter groups
    for d_col in ['diag_1', 'diag_2', 'diag_3']:
        g_col = f'{d_col}_group'
        train_feat[g_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_feat[g_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(g_col)

    # 3. Out-of-fold target encoding for high-cardinality & key categorical features
    te_columns = [
        'diag_1', 'diag_2', 'diag_3',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code',
        'diag_1_group', 'diag_2_group', 'diag_3_group',
    ]
    for col in te_columns:
        if col in train_feat and col in eval_feat:
            tr_enc, ev_enc = oof_target_encode(
                train_feat[col], y_train, eval_feat[col],
                n_folds=5, smoothing=25.0, seed=42
            )
            te_name = f'te_{col}'
            train_feat[te_name] = tr_enc.astype(np.float64)
            eval_feat[te_name] = ev_enc.astype(np.float64)
            extra_numeric.append(te_name)

    # Composite diagnosis risk indicators
    train_feat['te_diag_max'] = np.maximum(
        train_feat['te_diag_1'],
        np.maximum(train_feat['te_diag_2'], train_feat['te_diag_3'])
    )
    eval_feat['te_diag_max'] = np.maximum(
        eval_feat['te_diag_1'],
        np.maximum(eval_feat['te_diag_2'], eval_feat['te_diag_3'])
    )
    extra_numeric.append('te_diag_max')

    train_feat['te_diag_mean'] = (
        train_feat['te_diag_1'] + train_feat['te_diag_2'] + train_feat['te_diag_3']
    ) / 3.0
    eval_feat['te_diag_mean'] = (
        eval_feat['te_diag_1'] + eval_feat['te_diag_2'] + eval_feat['te_diag_3']
    ) / 3.0
    extra_numeric.append('te_diag_mean')

    # 4. Derived clinical features: healthcare utilization, intensity, medications, age
    def compute_clinical_features(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        n = len(data['time_in_hospital'])

        # Prior visits
        inp = np.nan_to_num(data.get('number_inpatient', np.zeros(n)), nan=0.0)
        emg = np.nan_to_num(data.get('number_emergency', np.zeros(n)), nan=0.0)
        outp = np.nan_to_num(data.get('number_outpatient', np.zeros(n)), nan=0.0)

        tot_visits = inp + emg + outp
        res['total_visits'] = tot_visits
        res['has_inpatient'] = (inp > 0).astype(np.float64)
        res['has_emergency'] = (emg > 0).astype(np.float64)
        res['multiple_inpatient'] = (inp >= 2).astype(np.float64)
        res['high_utilizer'] = ((inp >= 2) | (tot_visits >= 4)).astype(np.float64)
        res['inpatient_ratio'] = inp / (tot_visits + 1.0)
        res['emergency_ratio'] = emg / (tot_visits + 1.0)
        res['util_intensity'] = inp * 3.0 + emg * 2.0 + outp

        # Length of stay & encounter intensity
        los = np.maximum(np.nan_to_num(data.get('time_in_hospital', np.ones(n)), nan=1.0), 1.0)
        n_labs = np.nan_to_num(data.get('num_lab_procedures', np.zeros(n)), nan=0.0)
        n_procs = np.nan_to_num(data.get('num_procedures', np.zeros(n)), nan=0.0)
        n_meds = np.nan_to_num(data.get('num_medications', np.zeros(n)), nan=0.0)
        n_diag = np.maximum(np.nan_to_num(data.get('number_diagnoses', np.ones(n)), nan=1.0), 1.0)

        res['labs_per_day'] = n_labs / los
        res['meds_per_day'] = n_meds / los
        res['procs_per_day'] = n_procs / los
        res['total_procedures'] = n_labs + n_procs
        res['meds_per_diag'] = n_meds / n_diag
        res['labs_per_med'] = n_labs / np.maximum(n_meds, 1.0)
        res['overall_intensity'] = (n_labs + 5.0 * n_procs + 2.0 * n_meds) / los

        # Age mapping to numeric
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0,
        }
        raw_age = data.get('age', np.full(n, None))
        age_num = np.array([age_map.get(str(a), 65.0) for a in raw_age], dtype=np.float64)
        res['age_numeric'] = age_num
        res['age_x_inpatient'] = age_num * inp
        res['age_x_los'] = age_num * los

        # Lab tests: Glucose & A1C
        glu = data.get('max_glu_serum', np.full(n, None))
        res['glu_measured'] = np.isin(glu, ['Norm', '>200', '>300']).astype(np.float64)
        res['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)

        a1c = data.get('A1Cresult', np.full(n, None))
        res['a1c_measured'] = np.isin(a1c, ['Norm', '>7', '>8']).astype(np.float64)
        res['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        res['a1c_very_high'] = (a1c == '>8').astype(np.float64)

        # Medication changes & status
        ch = data.get('change', np.full(n, None))
        res['med_changed'] = (ch == 'Ch').astype(np.float64)
        dm = data.get('diabetesMed', np.full(n, None))
        res['on_diabetes_med'] = (dm == 'Yes').astype(np.float64)

        # Medication count analysis
        med_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
        ]
        active_count = np.zeros(n, dtype=np.float64)
        up_count = np.zeros(n, dtype=np.float64)
        down_count = np.zeros(n, dtype=np.float64)
        steady_count = np.zeros(n, dtype=np.float64)

        for m in med_cols:
            if m in data:
                m_val = data[m]
                active_count += np.isin(m_val, ['Steady', 'Up', 'Down'])
                up_count += (m_val == 'Up')
                down_count += (m_val == 'Down')
                steady_count += (m_val == 'Steady')

        res['med_active_count'] = active_count
        res['med_up_count'] = up_count
        res['med_down_count'] = down_count
        res['med_change_count'] = up_count + down_count
        res['med_steady_count'] = steady_count

        # Insulin details
        ins = data.get('insulin', np.full(n, None))
        res['insulin_active'] = np.isin(ins, ['Steady', 'Up', 'Down']).astype(np.float64)
        res['insulin_up'] = (ins == 'Up').astype(np.float64)
        res['insulin_down'] = (ins == 'Down').astype(np.float64)
        res['insulin_changed'] = np.isin(ins, ['Up', 'Down']).astype(np.float64)

        return res

    tr_clin = compute_clinical_features(train)
    ev_clin = compute_clinical_features(eval_cols)
    for col_name, col_arr in tr_clin.items():
        train_feat[col_name] = col_arr
        eval_feat[col_name] = ev_clin[col_name]
        extra_numeric.append(col_name)

    # 5. Encode all features
    encoder = TabularEncoder(min_count=25)
    encoder.fit(
        task,
        train_feat,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)
    cat_idx = encoder.categorical_indices

    # 6. Ensemble: Diverse LightGBM architectures + XGBoost
    # Model 1: Balanced LightGBM
    m1 = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.022,
        num_leaves=31,
        max_depth=6,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        reg_alpha=0.2,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_idx)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Regularized Shallow LightGBM
    m2 = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=18,
        max_depth=5,
        min_child_samples=250,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=4.5,
        reg_alpha=0.5,
        cat_smooth=75.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_idx)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Interaction-focused Deeper LightGBM
    m3 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.022,
        num_leaves=40,
        max_depth=7,
        min_child_samples=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=25.0,
        random_state=789,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_idx)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: Histogram-based XGBoost
    xgb_preds = None
    try:
        xgb_mod = globals().get('xgb', globals().get('xgboost', None))
        if xgb_mod is not None:
            m_xgb = xgb_mod.XGBClassifier(
                n_estimators=250,
                learning_rate=0.025,
                max_depth=5,
                min_child_weight=15.0,
                subsample=0.75,
                colsample_bytree=0.6,
                reg_lambda=3.0,
                reg_alpha=0.2,
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                eval_metric='auc',
            )
            m_xgb.fit(x_train, y_train)
            xgb_preds = m_xgb.predict_proba(x_eval)[:, 1]
    except Exception:
        xgb_preds = None

    if xgb_preds is not None:
        preds = 0.35 * p1 + 0.30 * p2 + 0.20 * p3 + 0.15 * xgb_preds
    else:
        preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3

    # Safe clipping to guarantee valid probabilities
    prior = float(np.mean(y_train))
    preds = np.nan_to_num(preds, nan=prior)
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
