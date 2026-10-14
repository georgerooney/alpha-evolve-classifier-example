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
    from sklearn.decomposition import TruncatedSVD

    try:
        _icd9_comorbidities = icd9_comorbidities
    except NameError:
        from clinical_toolkit import icd9_comorbidities as _icd9_comorbidities

    try:
        _icd9_group = icd9_group
    except NameError:
        from clinical_toolkit import icd9_group as _icd9_group

    try:
        _oof_target_encode = oof_target_encode
    except NameError:
        from clinical_toolkit import oof_target_encode as _oof_target_encode

    # Make shallow copies of dictionary structures
    train_dict = {k: v for k, v in train.items()}
    eval_dict = {k: v for k, v in eval_cols.items()}

    n_tr = len(y_train)
    n_ev = len(eval_cols['time_in_hospital'])

    extra_numeric_keys = []
    extra_categorical_keys = []

    # 1. Comorbidities extraction (Charlson mapping)
    train_cm = _icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = _icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])

    cm_condition_names = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv',
        'cm_diabetes_complicated'
    ]
    for k, v in train_cm.items():
        train_dict[k] = np.asarray(v, dtype=np.float64)
        eval_dict[k] = np.asarray(eval_cm[k], dtype=np.float64)
        extra_numeric_keys.append(k)

    cm_count_tr = np.asarray(train_cm['cm_count'], dtype=np.float64)
    cm_count_ev = np.asarray(eval_cm['cm_count'], dtype=np.float64)

    # 2. ICD-9 Chapter groups for primary, secondary, and tertiary diagnoses
    for col_diag in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col_diag}_group"
        train_dict[grp_col] = np.array([_icd9_group(c) for c in train[col_diag]], dtype=object)
        eval_dict[grp_col] = np.array([_icd9_group(c) for c in eval_cols[col_diag]], dtype=object)
        extra_categorical_keys.append(grp_col)

    # 3. Smoothed Out-Of-Fold Target Encodings
    te_target_cols = [
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'medical_specialty',
        'diag_1',
        'diag_2',
        'diag_3',
    ]
    for col in te_target_cols:
        if col in train and col in eval_cols:
            te_tr, te_ev = _oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
            )
            te_name = f"te_{col}"
            train_dict[te_name] = np.asarray(te_tr, dtype=np.float64)
            eval_dict[te_name] = np.asarray(te_ev, dtype=np.float64)
            extra_numeric_keys.append(te_name)

    # 4. Utilization & Intensity Engineering
    def get_num(cols_dict, col_name):
        return np.asarray(cols_dict[col_name], dtype=np.float64)

    inp_tr, inp_ev = get_num(train, 'number_inpatient'), get_num(eval_cols, 'number_inpatient')
    em_tr, em_ev = get_num(train, 'number_emergency'), get_num(eval_cols, 'number_emergency')
    out_tr, out_ev = get_num(train, 'number_outpatient'), get_num(eval_cols, 'number_outpatient')
    los_tr, los_ev = get_num(train, 'time_in_hospital'), get_num(eval_cols, 'time_in_hospital')
    meds_tr, meds_ev = get_num(train, 'num_medications'), get_num(eval_cols, 'num_medications')
    labs_tr, labs_ev = get_num(train, 'num_lab_procedures'), get_num(eval_cols, 'num_lab_procedures')
    procs_tr, procs_ev = get_num(train, 'num_procedures'), get_num(eval_cols, 'num_procedures')
    diags_tr, diags_ev = get_num(train, 'number_diagnoses'), get_num(eval_cols, 'number_diagnoses')

    def build_util_dict(inp, em, out, los, meds, labs, procs, diags, cm_cnt):
        total_prior = inp + em + out
        return {
            'total_prior_visits': total_prior,
            'has_prior_inpatient': (inp > 0.0).astype(np.float64),
            'has_multiple_inpatient': (inp >= 2.0).astype(np.float64),
            'has_prior_emergency': (em > 0.0).astype(np.float64),
            'inpatient_ratio': inp / (total_prior + 1.0),
            'emergency_ratio': em / (total_prior + 1.0),
            'service_intensity': (labs + procs + meds) / (los + 1.0),
            'med_per_day': meds / (los + 1.0),
            'labs_per_day': labs / (los + 1.0),
            'diags_per_day': diags / (los + 1.0),
            'inp_per_day': inp / (los + 1.0),
            'complexity_index': diags * total_prior,
            'inp_x_cm_count': inp * cm_cnt,
            'total_prior_x_cm_count': total_prior * cm_cnt,
        }

    util_tr = build_util_dict(inp_tr, em_tr, out_tr, los_tr, meds_tr, labs_tr, procs_tr, diags_tr, cm_count_tr)
    util_ev = build_util_dict(inp_ev, em_ev, out_ev, los_ev, meds_ev, labs_ev, procs_ev, diags_ev, cm_count_ev)

    for k in util_tr:
        train_dict[k] = util_tr[k]
        eval_dict[k] = util_ev[k]
        extra_numeric_keys.append(k)

    # 5. Age continuous midpoint parsing
    def parse_age(age_val):
        if age_val is None or not isinstance(age_val, str):
            return np.nan
        try:
            clean = age_val.strip('[]() ')
            parts = clean.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    age_tr = np.array([parse_age(a) for a in train.get('age', [])], dtype=np.float64)
    age_ev = np.array([parse_age(a) for a in eval_cols.get('age', [])], dtype=np.float64)
    if not np.all(np.isnan(age_tr)):
        train_dict['age_numeric'] = age_tr
        eval_dict['age_numeric'] = age_ev
        extra_numeric_keys.append('age_numeric')

    # 6. Specific high-risk discharge destinations
    train_dict['is_discharged_home'] = (train['discharge_disposition_id'] == '1').astype(np.float64)
    eval_dict['is_discharged_home'] = (eval_cols['discharge_disposition_id'] == '1').astype(np.float64)
    extra_numeric_keys.append('is_discharged_home')

    train_dict['is_discharged_snf'] = (train['discharge_disposition_id'] == '3').astype(np.float64)
    eval_dict['is_discharged_snf'] = (eval_cols['discharge_disposition_id'] == '3').astype(np.float64)
    extra_numeric_keys.append('is_discharged_snf')

    train_dict['is_discharged_home_health'] = (train['discharge_disposition_id'] == '6').astype(np.float64)
    eval_dict['is_discharged_home_health'] = (eval_cols['discharge_disposition_id'] == '6').astype(np.float64)
    extra_numeric_keys.append('is_discharged_home_health')

    # 7. Medication Regimen Matrix Representation
    known_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_cols = [c for c in known_meds if c in train]
    M_train = np.zeros((n_tr, len(med_cols)), dtype=np.float64)
    M_eval = np.zeros((n_ev, len(med_cols)), dtype=np.float64)

    for idx, col in enumerate(med_cols):
        tr_v = train[col]
        ev_v = eval_cols[col]
        M_train[tr_v == 'Steady', idx] = 1.0
        M_train[tr_v == 'Down', idx] = -1.0
        M_train[tr_v == 'Up', idx] = 2.0
        M_eval[ev_v == 'Steady', idx] = 1.0
        M_eval[ev_v == 'Down', idx] = -1.0
        M_eval[ev_v == 'Up', idx] = 2.0

    train_dict['num_meds_active'] = (M_train != 0.0).sum(axis=1).astype(np.float64)
    eval_dict['num_meds_active'] = (M_eval != 0.0).sum(axis=1).astype(np.float64)
    extra_numeric_keys.append('num_meds_active')

    train_dict['num_meds_up'] = (M_train == 2.0).sum(axis=1).astype(np.float64)
    eval_dict['num_meds_up'] = (M_eval == 2.0).sum(axis=1).astype(np.float64)
    extra_numeric_keys.append('num_meds_up')

    train_dict['num_meds_down'] = (M_train == -1.0).sum(axis=1).astype(np.float64)
    eval_dict['num_meds_down'] = (M_eval == -1.0).sum(axis=1).astype(np.float64)
    extra_numeric_keys.append('num_meds_down')

    train_dict['num_meds_changed'] = ((M_train == 2.0) | (M_train == -1.0)).sum(axis=1).astype(np.float64)
    eval_dict['num_meds_changed'] = ((M_eval == 2.0) | (M_eval == -1.0)).sum(axis=1).astype(np.float64)
    extra_numeric_keys.append('num_meds_changed')

    if 'insulin' in med_cols:
        ins_idx = med_cols.index('insulin')
        train_dict['insulin_treated'] = (M_train[:, ins_idx] != 0.0).astype(np.float64)
        eval_dict['insulin_treated'] = (M_eval[:, ins_idx] != 0.0).astype(np.float64)
        extra_numeric_keys.append('insulin_treated')

        train_dict['insulin_up'] = (M_train[:, ins_idx] == 2.0).astype(np.float64)
        eval_dict['insulin_up'] = (M_eval[:, ins_idx] == 2.0).astype(np.float64)
        extra_numeric_keys.append('insulin_up')

        train_dict['insulin_down'] = (M_train[:, ins_idx] == -1.0).astype(np.float64)
        eval_dict['insulin_down'] = (M_eval[:, ins_idx] == -1.0).astype(np.float64)
        extra_numeric_keys.append('insulin_down')

    # 8. Low-Rank Singular Subspace & Tucker Tensor Core Decomposition
    D_train = np.column_stack([np.asarray(train_cm[k], dtype=np.float64) for k in cm_condition_names])
    D_eval = np.column_stack([np.asarray(eval_cm[k], dtype=np.float64) for k in cm_condition_names])

    k_m = 4
    k_d = 4

    svd_m = TruncatedSVD(n_components=k_m, random_state=42)
    Z_M_train = svd_m.fit_transform(M_train)
    Z_M_eval = svd_m.transform(M_eval)

    svd_d = TruncatedSVD(n_components=k_d, random_state=42)
    Z_D_train = svd_d.fit_transform(D_train)
    Z_D_eval = svd_d.transform(D_eval)

    for i in range(k_m):
        key = f"latent_med_{i}"
        train_dict[key] = Z_M_train[:, i].astype(np.float64)
        eval_dict[key] = Z_M_eval[:, i].astype(np.float64)
        extra_numeric_keys.append(key)

    for j in range(k_d):
        key = f"latent_cm_{j}"
        train_dict[key] = Z_D_train[:, j].astype(np.float64)
        eval_dict[key] = Z_D_eval[:, j].astype(np.float64)
        extra_numeric_keys.append(key)

    # Tensor outer product Tucker core: Z_M (N x 4) \otimes Z_D (N x 4) -> (N x 16)
    Z_tensor_tr = (Z_M_train[:, :, None] * Z_D_train[:, None, :]).reshape(n_tr, k_m * k_d)
    Z_tensor_ev = (Z_M_eval[:, :, None] * Z_D_eval[:, None, :]).reshape(n_ev, k_m * k_d)

    for k in range(k_m * k_d):
        key = f"tensor_core_{k}"
        train_dict[key] = Z_tensor_tr[:, k].astype(np.float64)
        eval_dict[key] = Z_tensor_ev[:, k].astype(np.float64)
        extra_numeric_keys.append(key)

    # Interactions between Inpatient Utilization and Dominant Latent Factors
    train_dict['inp_x_lat_cm0'] = (inp_tr * Z_D_train[:, 0]).astype(np.float64)
    eval_dict['inp_x_lat_cm0'] = (inp_ev * Z_D_eval[:, 0]).astype(np.float64)
    extra_numeric_keys.append('inp_x_lat_cm0')

    train_dict['inp_x_lat_med0'] = (inp_tr * Z_M_train[:, 0]).astype(np.float64)
    eval_dict['inp_x_lat_med0'] = (inp_ev * Z_M_eval[:, 0]).astype(np.float64)
    extra_numeric_keys.append('inp_x_lat_med0')

    # Deduplicate extra keys
    extra_numeric_keys = list(dict.fromkeys(extra_numeric_keys))
    extra_categorical_keys = list(dict.fromkeys(extra_categorical_keys))

    # Fit encoder with tuned min_count=20 to retain informative rare categories
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_dict,
        extra_numeric=tuple(extra_numeric_keys),
        extra_categorical=tuple(extra_categorical_keys),
    )
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # 9. Diverse Multi-Model Heterogeneous Ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        reg_alpha=0.2,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = model1.predict_proba(x_eval)[:, 1]

    model2 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.02,
        num_leaves=18,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=4.0,
        reg_alpha=0.5,
        cat_smooth=45.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = model2.predict_proba(x_eval)[:, 1]

    model3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.025,
        num_leaves=45,
        min_child_samples=60,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        reg_alpha=0.1,
        cat_smooth=25.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = model3.predict_proba(x_eval)[:, 1]

    blended_preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
    return np.clip(np.nan_to_num(blended_preds, nan=0.114), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
