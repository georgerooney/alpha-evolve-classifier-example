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
    import xgboost as xgb
    from sklearn.decomposition import TruncatedSVD

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

    # 2. ICD-9 Chapter groups and 2-way diagnosis interaction
    for col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col}_group"
        train_feats[grp_col] = np.array([icd9_group(x) for x in train_feats[col]], dtype=object)
        eval_feats[grp_col] = np.array([icd9_group(x) for x in eval_feats[col]], dtype=object)
        extra_categorical_cols.append(grp_col)

    # 2-way diagnosis group interaction tensor
    train_feats['diag_1_2_group'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(train_feats['diag_1_group'], train_feats['diag_2_group'])],
        dtype=object,
    )
    eval_feats['diag_1_2_group'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(eval_feats['diag_1_group'], eval_feats['diag_2_group'])],
        dtype=object,
    )
    extra_categorical_cols.append('diag_1_2_group')

    # 3. Out-of-fold target encoding for primary high-cardinality and interaction categories
    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_2_group',
        'medical_specialty', 'discharge_disposition_id',
        'admission_source_id', 'admission_type_id', 'payer_code',
    ]
    for col in te_cols:
        te_col = f"{col}_te"
        tr_te, ev_te = oof_target_encode(train_feats[col], y_train, eval_feats[col], n_folds=5, smoothing=25.0)
        train_feats[te_col] = tr_te
        eval_feats[te_col] = ev_te
        extra_numeric_cols.append(te_col)

    # 4. Low-rank tensor / matrix decomposition of medication and comorbidity profiles
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]

    n_tr = len(train_feats['diag_1'])
    n_ev = len(eval_feats['diag_1'])

    med_mat_tr = np.zeros((n_tr, len(med_cols)), dtype=np.float32)
    med_mat_ev = np.zeros((n_ev, len(med_cols)), dtype=np.float32)

    for i, m in enumerate(med_cols):
        if m in train_feats:
            tr_m = train_feats[m]
            med_mat_tr[tr_m == 'Steady', i] = 1.0
            med_mat_tr[tr_m == 'Down', i] = 2.0
            med_mat_tr[tr_m == 'Up', i] = 3.0
        if m in eval_feats:
            ev_m = eval_feats[m]
            med_mat_ev[ev_m == 'Steady', i] = 1.0
            med_mat_ev[ev_m == 'Down', i] = 2.0
            med_mat_ev[ev_m == 'Up', i] = 3.0

    svd_med = TruncatedSVD(n_components=4, random_state=42)
    med_svd_tr = svd_med.fit_transform(med_mat_tr)
    med_svd_ev = svd_med.transform(med_mat_ev)
    for c in range(4):
        c_name = f'med_svd_{c}'
        train_feats[c_name] = med_svd_tr[:, c].astype(np.float64)
        eval_feats[c_name] = med_svd_ev[:, c].astype(np.float64)
        extra_numeric_cols.append(c_name)

    cm_flags = [k for k in train_cm.keys() if k != 'cm_count']
    cm_mat_tr = np.column_stack([train_cm[k] for k in cm_flags])
    cm_mat_ev = np.column_stack([eval_cm[k] for k in cm_flags])
    svd_cm = TruncatedSVD(n_components=3, random_state=42)
    cm_svd_tr = svd_cm.fit_transform(cm_mat_tr)
    cm_svd_ev = svd_cm.transform(cm_mat_ev)
    for c in range(3):
        c_name = f'cm_svd_{c}'
        train_feats[c_name] = cm_svd_tr[:, c].astype(np.float64)
        eval_feats[c_name] = cm_svd_ev[:, c].astype(np.float64)
        extra_numeric_cols.append(c_name)

    # 5. Continuous age representation and clinical care intensity
    def parse_age(age_str):
        if not isinstance(age_str, str):
            return np.nan
        s = age_str.replace('[', '').replace(')', '').replace(']', '')
        parts = s.split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                pass
        return np.nan

    train_feats['age_numeric'] = np.array([parse_age(x) for x in train_feats['age']], dtype=float)
    eval_feats['age_numeric'] = np.array([parse_age(x) for x in eval_feats['age']], dtype=float)
    extra_numeric_cols.append('age_numeric')

    for df in [train_feats, eval_feats]:
        outpatient = np.nan_to_num(df['number_outpatient'])
        emergency = np.nan_to_num(df['number_emergency'])
        inpatient = np.nan_to_num(df['number_inpatient'])
        time_hosp = np.maximum(np.nan_to_num(df['time_in_hospital']), 1.0)
        num_meds = np.nan_to_num(df['num_medications'])
        num_labs = np.nan_to_num(df['num_lab_procedures'])
        num_procs = np.nan_to_num(df['num_procedures'])
        num_diag = np.nan_to_num(df['number_diagnoses'])

        tot_util = outpatient + emergency + inpatient
        df['total_utilization'] = tot_util
        df['utilization_weighted'] = outpatient + 2.0 * emergency + 3.0 * inpatient
        df['has_prior_inpatient'] = (inpatient > 0).astype(float)
        df['has_prior_emergency'] = (emergency > 0).astype(float)
        df['frequent_inpatient'] = (inpatient >= 2.0).astype(float)
        df['inpatient_ratio'] = inpatient / (tot_util + 1.0)
        df['emergency_ratio'] = emergency / (tot_util + 1.0)

        df['labs_per_day'] = num_labs / time_hosp
        df['meds_per_day'] = num_meds / time_hosp
        df['procs_per_day'] = num_procs / time_hosp
        df['diag_per_day'] = num_diag / time_hosp

        active_meds = np.zeros(len(outpatient), dtype=float)
        changed_meds = np.zeros(len(outpatient), dtype=float)
        for m in med_cols:
            if m in df:
                is_active = np.isin(df[m], ['Steady', 'Up', 'Down'])
                is_changed = np.isin(df[m], ['Up', 'Down'])
                active_meds += is_active.astype(float)
                changed_meds += is_changed.astype(float)

        df['num_active_meds'] = active_meds
        df['num_changed_meds'] = changed_meds
        df['has_change'] = (df['change'] == 'Ch').astype(float)
        df['has_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(float)
        df['insulin_up'] = (df['insulin'] == 'Up').astype(float)
        df['insulin_down'] = (df['insulin'] == 'Down').astype(float)
        df['insulin_steady'] = (df['insulin'] == 'Steady').astype(float)
        df['insulin_changed'] = np.isin(df['insulin'], ['Up', 'Down']).astype(float)

        a1c_str = df['A1Cresult'].astype(str)
        chg_str = df['change'].astype(str)
        df['high_a1c_no_change'] = (np.isin(a1c_str, ['>8', '>7']) & (chg_str == 'No')).astype(float)

    extra_numeric_cols.extend([
        'total_utilization', 'utilization_weighted', 'has_prior_inpatient', 'has_prior_emergency',
        'frequent_inpatient', 'inpatient_ratio', 'emergency_ratio',
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'diag_per_day',
        'num_active_meds', 'num_changed_meds', 'has_change', 'has_diabetes_med',
        'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'high_a1c_no_change',
    ])

    # 6. Fit TabularEncoder
    extra_numeric_cols = list(dict.fromkeys(extra_numeric_cols))
    extra_categorical_cols = list(dict.fromkeys(extra_categorical_cols))

    encoder = TabularEncoder(min_count=50, one_hot=False)
    encoder.fit(task, train_feats, extra_numeric=extra_numeric_cols, extra_categorical=extra_categorical_cols)
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 7. Heterogeneous model ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=190,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=160,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=0.08,
        reg_alpha=0.02,
        cat_smooth=64.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    model2 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.012,
        num_leaves=46,
        min_child_samples=240,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=0.15,
        reg_alpha=0.05,
        cat_smooth=80.0,
        random_state=2023,
        n_jobs=1,
        verbose=-1,
    )

    model3 = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.018,
        num_leaves=20,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=0.2,
        reg_alpha=0.02,
        cat_smooth=50.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=140,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=15,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_lambda=1.8,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    # 8. Train models and compute weighted blend
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    preds1 = model1.predict_proba(x_eval)[:, 1]

    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    preds2 = model2.predict_proba(x_eval)[:, 1]

    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    preds3 = model3.predict_proba(x_eval)[:, 1]

    xgb_model.fit(x_train, y_train)
    preds4 = xgb_model.predict_proba(x_eval)[:, 1]

    return 0.40 * preds1 + 0.35 * preds2 + 0.10 * preds3 + 0.15 * preds4


# EVOLVE-BLOCK-END
# =====================================================================
