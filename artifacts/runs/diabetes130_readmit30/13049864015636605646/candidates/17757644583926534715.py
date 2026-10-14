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
    import numpy as np
    import lightgbm as lgb

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric_cols = []
    extra_categorical_cols = []

    # 1. Comorbidities from clinical mapping
    train_cm = icd9_comorbidities(train_feats['diag_1'], train_feats['diag_2'], train_feats['diag_3'])
    eval_cm = icd9_comorbidities(eval_feats['diag_1'], eval_feats['diag_2'], eval_feats['diag_3'])
    for k, v in train_cm.items():
        train_feats[k] = v
        eval_feats[k] = eval_cm[k]
        extra_numeric_cols.append(k)

    # Comorbidity interactions
    for df, cm in [(train_feats, train_cm), (eval_feats, eval_cm)]:
        df['cm_cardiorenal'] = cm['cm_chf'] * cm['cm_renal']
        df['cm_chf_pulmonary'] = cm['cm_chf'] * cm['cm_pulmonary']
        df['cm_high_count'] = (cm['cm_count'] >= 2.0).astype(np.float64)
    extra_numeric_cols.extend(['cm_cardiorenal', 'cm_chf_pulmonary', 'cm_high_count'])

    # 2. ICD-9 Chapter groups & 3-digit prefixes
    def extract_prefix(val):
        if val is None or val == 'None':
            return 'missing'
        s = str(val).strip()
        if not s or s == '?':
            return 'missing'
        return s.split('.')[0]

    for col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col}_group"
        pref_col = f"{col}_prefix"
        train_feats[grp_col] = np.array([icd9_group(x) for x in train_feats[col]], dtype=object)
        eval_feats[grp_col] = np.array([icd9_group(x) for x in eval_feats[col]], dtype=object)
        train_feats[pref_col] = np.array([extract_prefix(x) for x in train_feats[col]], dtype=object)
        eval_feats[pref_col] = np.array([extract_prefix(x) for x in eval_feats[col]], dtype=object)
        extra_categorical_cols.extend([grp_col, pref_col])

    # 3. Target encoding for high-cardinality & key categorical features
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code',
    ]
    for col in te_cols:
        te_col = f"{col}_te"
        tr_te, ev_te = oof_target_encode(
            train_feats[col], y_train, eval_feats[col], n_folds=5, smoothing=20.0, seed=42
        )
        train_feats[te_col] = tr_te
        eval_feats[te_col] = ev_te
        extra_numeric_cols.append(te_col)

    # 4. Domain & Clinical Feature Engineering
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }

    new_numeric = [
        'total_visits', 'has_prior_inpatient', 'multiple_prior_inpatient',
        'has_prior_emergency', 'inpatient_ratio', 'emergency_ratio', 'weighted_utilization',
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'meds_per_diag', 'stay_intensity',
        'age_num', 'is_elderly',
        'num_active_meds', 'num_meds_up', 'num_meds_down', 'num_med_changes',
        'has_med_change', 'has_diabetes_med', 'insulin_prescribed', 'insulin_changed',
        'insulin_up', 'insulin_down',
        'a1c_tested', 'a1c_high', 'a1c_high_no_change', 'a1c_high_change',
        'glu_tested', 'glu_high',
        'discharge_to_home', 'discharge_to_snf', 'discharge_to_home_health',
        'admission_emergency', 'admission_from_er',
        'has_payer_code', 'has_medical_specialty',
    ]

    for df in [train_feats, eval_feats]:
        n_rows = len(df['diag_1'])

        # Prior utilization
        inp = np.nan_to_num(df['number_inpatient'], nan=0.0)
        er = np.nan_to_num(df['number_emergency'], nan=0.0)
        out = np.nan_to_num(df['number_outpatient'], nan=0.0)
        tot = inp + er + out
        df['total_visits'] = tot
        df['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        df['multiple_prior_inpatient'] = (inp >= 2.0).astype(np.float64)
        df['has_prior_emergency'] = (er > 0.0).astype(np.float64)
        df['inpatient_ratio'] = (inp + 0.01) / (tot + 0.1)
        df['emergency_ratio'] = (er + 0.01) / (tot + 0.1)
        df['weighted_utilization'] = 3.0 * inp + 1.5 * er + 0.5 * out

        # Stay intensity
        stay = np.maximum(np.nan_to_num(df['time_in_hospital'], nan=1.0), 1.0)
        meds = np.nan_to_num(df['num_medications'], nan=0.0)
        labs = np.nan_to_num(df['num_lab_procedures'], nan=0.0)
        procs = np.nan_to_num(df['num_procedures'], nan=0.0)
        diags = np.maximum(np.nan_to_num(df['number_diagnoses'], nan=1.0), 1.0)

        df['labs_per_day'] = labs / stay
        df['meds_per_day'] = meds / stay
        df['procs_per_day'] = procs / stay
        df['meds_per_diag'] = meds / diags
        df['stay_intensity'] = (labs + 4.0 * procs + meds) / stay

        # Continuous age representation
        df['age_num'] = np.array([age_map.get(str(x), np.nan) for x in df['age']], dtype=np.float64)
        df['is_elderly'] = np.array(
            [1.0 if str(x) in ('[70-80)', '[80-90)', '[90-100)') else 0.0 for x in df['age']],
            dtype=np.float64,
        )

        # Medication features
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)
        for m in med_cols:
            if m in df:
                active += np.isin(df[m], ['Steady', 'Up', 'Down'])
                up += (df[m] == 'Up')
                down += (df[m] == 'Down')
        df['num_active_meds'] = active
        df['num_meds_up'] = up
        df['num_meds_down'] = down
        df['num_med_changes'] = up + down
        df['has_med_change'] = (df['change'] == 'Ch').astype(np.float64)
        df['has_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(np.float64)

        insulin = df['insulin']
        df['insulin_prescribed'] = np.array(
            [0.0 if (x is None or str(x) in ('No', 'None', '?')) else 1.0 for x in insulin],
            dtype=np.float64,
        )
        df['insulin_changed'] = np.isin(insulin, ['Up', 'Down']).astype(np.float64)
        df['insulin_up'] = (insulin == 'Up').astype(np.float64)
        df['insulin_down'] = (insulin == 'Down').astype(np.float64)

        # Lab results & Strack et al. interaction
        a1c_tested = np.array(
            [0.0 if (x is None or str(x) in ('None', '?')) else 1.0 for x in df['A1Cresult']],
            dtype=np.float64,
        )
        a1c_high = np.isin(df['A1Cresult'], ['>7', '>8']).astype(np.float64)
        df['a1c_tested'] = a1c_tested
        df['a1c_high'] = a1c_high
        df['a1c_high_no_change'] = (a1c_high == 1.0) & (df['has_med_change'] == 0.0)
        df['a1c_high_change'] = (a1c_high == 1.0) & (df['has_med_change'] == 1.0)

        df['glu_tested'] = np.array(
            [0.0 if (x is None or str(x) in ('None', '?')) else 1.0 for x in df['max_glu_serum']],
            dtype=np.float64,
        )
        df['glu_high'] = np.isin(df['max_glu_serum'], ['>200', '>300']).astype(np.float64)

        # Admission & Discharge
        df['discharge_to_home'] = (df['discharge_disposition_id'] == '1').astype(np.float64)
        df['discharge_to_snf'] = (df['discharge_disposition_id'] == '3').astype(np.float64)
        df['discharge_to_home_health'] = (df['discharge_disposition_id'] == '6').astype(np.float64)
        df['admission_emergency'] = (df['admission_type_id'] == '1').astype(np.float64)
        df['admission_from_er'] = (df['admission_source_id'] == '7').astype(np.float64)

        df['has_payer_code'] = np.array(
            [0.0 if (x is None or str(x) in ('None', '?')) else 1.0 for x in df['payer_code']],
            dtype=np.float64,
        )
        df['has_medical_specialty'] = np.array(
            [0.0 if (x is None or str(x) in ('None', '?')) else 1.0 for x in df['medical_specialty']],
            dtype=np.float64,
        )

    extra_numeric_cols.extend(new_numeric)

    # 5. Fit TabularEncoder
    extra_numeric_cols = list(dict.fromkeys(extra_numeric_cols))
    extra_categorical_cols = list(dict.fromkeys(extra_categorical_cols))

    encoder = TabularEncoder(min_count=20, one_hot=False)
    encoder.fit(task, train_feats, extra_numeric=extra_numeric_cols, extra_categorical=extra_categorical_cols)
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 6. Multi-capacity, multi-seed LightGBM ensemble
    configs = [
        # Family 1: Balanced medium-depth trees
        {
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 70,
            'learning_rate': 0.02,
            'n_estimators': 280,
            'colsample_bytree': 0.65,
            'subsample': 0.80,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'cat_smooth': 50.0,
            'seeds': [42, 100, 2024],
        },
        # Family 2: Deeper, interaction-focused trees
        {
            'num_leaves': 50,
            'max_depth': 8,
            'min_child_samples': 50,
            'learning_rate': 0.018,
            'n_estimators': 260,
            'colsample_bytree': 0.55,
            'subsample': 0.75,
            'reg_lambda': 2.0,
            'reg_alpha': 0.3,
            'cat_smooth': 35.0,
            'seeds': [7, 999],
        },
        # Family 3: Shallower, regularized anchor trees
        {
            'num_leaves': 18,
            'max_depth': 5,
            'min_child_samples': 120,
            'learning_rate': 0.025,
            'n_estimators': 280,
            'colsample_bytree': 0.75,
            'subsample': 0.85,
            'reg_lambda': 0.5,
            'reg_alpha': 0.05,
            'cat_smooth': 70.0,
            'seeds': [123, 777],
        },
    ]

    preds = np.zeros(len(eval_cols['diag_1']), dtype=np.float64)
    total_models = 0

    for cfg in configs:
        for seed in cfg['seeds']:
            model = lgb.LGBMClassifier(
                n_estimators=cfg['n_estimators'],
                learning_rate=cfg['learning_rate'],
                num_leaves=cfg['num_leaves'],
                max_depth=cfg['max_depth'],
                min_child_samples=cfg['min_child_samples'],
                subsample=cfg['subsample'],
                subsample_freq=1,
                colsample_bytree=cfg['colsample_bytree'],
                reg_lambda=cfg['reg_lambda'],
                reg_alpha=cfg['reg_alpha'],
                cat_smooth=cfg['cat_smooth'],
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
            model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
            preds += model.predict_proba(x_eval)[:, 1]
            total_models += 1

    return preds / total_models


# EVOLVE-BLOCK-END
# =====================================================================
