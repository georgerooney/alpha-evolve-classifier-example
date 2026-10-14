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
    """Fit on `train`/`y_train` and return readmission probabilities using SVD/tensor decomposition,
    OOF target encoding, domain features, and a regularized ensemble."""
    n_train = len(train['time_in_hospital'])
    n_eval = len(eval_cols['time_in_hospital'])

    train_ext = dict(train)
    eval_ext = dict(eval_cols)

    # 1. ICD-9 Chapter Groupings
    for diag_col in ('diag_1', 'diag_2', 'diag_3'):
        grp_name = f"{diag_col}_group"
        train_ext[grp_name] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_ext[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)

    extra_cat_cols = ['diag_1_group', 'diag_2_group', 'diag_3_group']

    # 2. Comorbidity flags & multi-system burden
    for data, n_rows in ((train_ext, n_train), (eval_ext, n_eval)):
        g1 = data['diag_1_group']
        g2 = data['diag_2_group']
        g3 = data['diag_3_group']

        data['diag_has_diabetes'] = np.array([1.0 if 'diabetes' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_circulatory'] = np.array([1.0 if 'circulatory' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_respiratory'] = np.array([1.0 if 'respiratory' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_digestive'] = np.array([1.0 if 'digestive' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_genitourinary'] = np.array([1.0 if 'genitourinary' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_neoplasms'] = np.array([1.0 if 'neoplasms' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_num_comorbid_groups'] = np.array([float(len({x for x in (a, b, c) if x not in ('missing', 'other')})) for a, b, c in zip(g1, g2, g3)], dtype=np.float64)

    # 3. Latent Comorbidity Phenotype via SVD Decomposition
    prefix_counts = {}
    for col in ('diag_1', 'diag_2', 'diag_3'):
        if col in train:
            for c in train[col]:
                if c is not None and c not in ('', '?', 'None'):
                    p = str(c).split('.')[0].strip()
                    if p:
                        prefix_counts[p] = prefix_counts.get(p, 0) + 1

    top_prefixes = [p for p, _ in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:60]]
    prefix_to_idx = {p: i for i, p in enumerate(top_prefixes)}
    D_diag = len(top_prefixes)
    k_diag = min(6, D_diag)

    if D_diag > 0:
        B_train = np.zeros((n_train, D_diag), dtype=np.float32)
        B_eval = np.zeros((n_eval, D_diag), dtype=np.float32)

        for col in ('diag_1', 'diag_2', 'diag_3'):
            if col in train:
                for i, c in enumerate(train[col]):
                    if c is not None and c not in ('', '?', 'None'):
                        idx = prefix_to_idx.get(str(c).split('.')[0].strip())
                        if idx is not None:
                            B_train[i, idx] = 1.0
            if col in eval_cols:
                for i, c in enumerate(eval_cols[col]):
                    if c is not None and c not in ('', '?', 'None'):
                        idx = prefix_to_idx.get(str(c).split('.')[0].strip())
                        if idx is not None:
                            B_eval[i, idx] = 1.0

        diag_mean = np.mean(B_train, axis=0, keepdims=True)
        B_train_c = B_train - diag_mean
        B_eval_c = B_eval - diag_mean

        C_diag = (B_train_c.T @ B_train_c) / max(1, n_train)
        _, eigvecs_diag = np.linalg.eigh(C_diag)
        diag_components = eigvecs_diag[:, -k_diag:][:, ::-1]

        Z_diag_train = B_train_c @ diag_components
        Z_diag_eval = B_eval_c @ diag_components

        for d in range(k_diag):
            train_ext[f'diag_svd_{d}'] = Z_diag_train[:, d].astype(np.float64)
            eval_ext[f'diag_svd_{d}'] = Z_diag_eval[:, d].astype(np.float64)
    else:
        Z_diag_train = np.zeros((n_train, 1), dtype=np.float32)
        Z_diag_eval = np.zeros((n_eval, 1), dtype=np.float32)

    # 4. Latent Medication Regimen via SVD Decomposition
    all_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    available_meds = [m for m in all_meds if m in train and m in eval_cols]
    med_val_map = {'No': 0.0, 'Steady': 1.0, 'Up': 2.0, 'Down': -0.5}
    D_med = len(available_meds)
    k_med = min(4, D_med)

    if D_med > 0:
        M_train = np.zeros((n_train, D_med), dtype=np.float32)
        M_eval = np.zeros((n_eval, D_med), dtype=np.float32)

        for j, med in enumerate(available_meds):
            tr_col = train[med]
            for i in range(n_train):
                M_train[i, j] = med_val_map.get(tr_col[i], 0.0)
            ev_col = eval_cols[med]
            for i in range(n_eval):
                M_eval[i, j] = med_val_map.get(ev_col[i], 0.0)

        med_mean = np.mean(M_train, axis=0, keepdims=True)
        M_train_c = M_train - med_mean
        M_eval_c = M_eval - med_mean

        C_med = (M_train_c.T @ M_train_c) / max(1, n_train)
        _, eigvecs_med = np.linalg.eigh(C_med)
        med_components = eigvecs_med[:, -k_med:][:, ::-1]

        Z_med_train = M_train_c @ med_components
        Z_med_eval = M_eval_c @ med_components

        for m in range(k_med):
            train_ext[f'med_svd_{m}'] = Z_med_train[:, m].astype(np.float64)
            eval_ext[f'med_svd_{m}'] = Z_med_eval[:, m].astype(np.float64)
    else:
        Z_med_train = np.zeros((n_train, 1), dtype=np.float32)
        Z_med_eval = np.zeros((n_eval, 1), dtype=np.float32)

    # 5. Bilinear Tensor Interactions: Phenotype x Regimen Coupling
    n_inter_d = min(3, Z_diag_train.shape[1])
    n_inter_m = min(3, Z_med_train.shape[1])
    for d in range(n_inter_d):
        for m in range(n_inter_m):
            train_ext[f'tensor_inter_{d}_{m}'] = (Z_diag_train[:, d] * Z_med_train[:, m]).astype(np.float64)
            eval_ext[f'tensor_inter_{d}_{m}'] = (Z_diag_eval[:, d] * Z_med_eval[:, m]).astype(np.float64)

    # 6. Utilization, Prior Admissions & Hospital Stay Intensity
    for data, n_rows in ((train_ext, n_train), (eval_ext, n_eval)):
        num_inpatient = np.nan_to_num(data.get('number_inpatient', np.zeros(n_rows, dtype=np.float64)))
        num_emergency = np.nan_to_num(data.get('number_emergency', np.zeros(n_rows, dtype=np.float64)))
        num_outpatient = np.nan_to_num(data.get('number_outpatient', np.zeros(n_rows, dtype=np.float64)))
        time_hosp = np.nan_to_num(data.get('time_in_hospital', np.ones(n_rows, dtype=np.float64)), nan=1.0)
        time_hosp_safe = np.maximum(time_hosp, 1.0)
        num_meds = np.nan_to_num(data.get('num_medications', np.zeros(n_rows, dtype=np.float64)))
        num_labs = np.nan_to_num(data.get('num_lab_procedures', np.zeros(n_rows, dtype=np.float64)))
        num_procs = np.nan_to_num(data.get('num_procedures', np.zeros(n_rows, dtype=np.float64)))
        num_diags = np.nan_to_num(data.get('number_diagnoses', np.zeros(n_rows, dtype=np.float64)))

        total_prior = num_inpatient + num_emergency + num_outpatient
        data['total_prior_visits'] = total_prior
        data['has_prior_inpatient'] = (num_inpatient > 0).astype(np.float64)
        data['has_prior_emergency'] = (num_emergency > 0).astype(np.float64)
        data['has_prior_outpatient'] = (num_outpatient > 0).astype(np.float64)
        data['inpatient_ratio'] = num_inpatient / (total_prior + 1.0)
        data['emergency_ratio'] = num_emergency / (total_prior + 1.0)
        data['high_inpatient'] = (num_inpatient >= 2.0).astype(np.float64)

        data['service_intensity'] = (num_labs + num_procs + num_meds) / time_hosp_safe
        data['meds_per_day'] = num_meds / time_hosp_safe
        data['labs_per_day'] = num_labs / time_hosp_safe
        data['procedures_per_day'] = num_procs / time_hosp_safe
        data['diagnoses_per_day'] = num_diags / time_hosp_safe
        data['lab_to_med_ratio'] = num_labs / (num_meds + 1.0)

    # 7. Medication Regimen Intensity & Titration Details
    for data, n_rows in ((train_ext, n_train), (eval_ext, n_eval)):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        for med in available_meds:
            arr = data[med]
            for i in range(n_rows):
                v = arr[i]
                if v in ('Steady', 'Up', 'Down'):
                    active_cnt[i] += 1.0
                if v == 'Up':
                    up_cnt[i] += 1.0
                elif v == 'Down':
                    down_cnt[i] += 1.0
        data['num_active_meds'] = active_cnt
        data['num_meds_up'] = up_cnt
        data['num_meds_down'] = down_cnt
        data['num_meds_changed'] = up_cnt + down_cnt

        if 'insulin' in data:
            data['insulin_active'] = np.array([1.0 if v in ('Steady', 'Up', 'Down') else 0.0 for v in data['insulin']], dtype=np.float64)
            data['insulin_up'] = np.array([1.0 if v == 'Up' else 0.0 for v in data['insulin']], dtype=np.float64)
            data['insulin_down'] = np.array([1.0 if v == 'Down' else 0.0 for v in data['insulin']], dtype=np.float64)
        if 'metformin' in data:
            data['metformin_active'] = np.array([1.0 if v in ('Steady', 'Up', 'Down') else 0.0 for v in data['metformin']], dtype=np.float64)
        if 'insulin' in data and 'metformin' in data:
            data['insulin_and_metformin'] = data['insulin_active'] * data['metformin_active']

        if 'change' in data:
            data['change_flag'] = np.array([1.0 if v == 'Ch' else 0.0 for v in data['change']], dtype=np.float64)
        if 'diabetesMed' in data:
            data['diabetes_med_flag'] = np.array([1.0 if v == 'Yes' else 0.0 for v in data['diabetesMed']], dtype=np.float64)

    # 8. Demographics, Admission, Discharge & Lab Tests
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for data in (train_ext, eval_ext):
        if 'age' in data:
            data['age_num'] = np.array([age_map.get(a, 65.0) for a in data['age']], dtype=np.float64)
            data['age_elderly'] = (data['age_num'] >= 70.0).astype(np.float64)

        if 'discharge_disposition_id' in data:
            arr = data['discharge_disposition_id']
            data['discharge_home'] = np.array([1.0 if str(v) == '1' else 0.0 for v in arr], dtype=np.float64)
            data['discharge_facility'] = np.array([1.0 if str(v) in ('2', '3', '4', '5', '6', '22', '28') else 0.0 for v in arr], dtype=np.float64)

        if 'admission_type_id' in data:
            data['admission_emergency'] = np.array([1.0 if str(v) in ('1', '2') else 0.0 for v in data['admission_type_id']], dtype=np.float64)
        if 'admission_source_id' in data:
            data['admission_from_er'] = np.array([1.0 if str(v) == '7' else 0.0 for v in data['admission_source_id']], dtype=np.float64)

        if 'A1Cresult' in data:
            data['a1c_tested'] = np.array([0.0 if v in (None, 'None') else 1.0 for v in data['A1Cresult']], dtype=np.float64)
            data['a1c_high'] = np.array([1.0 if v == '>8' else 0.0 for v in data['A1Cresult']], dtype=np.float64)
        if 'max_glu_serum' in data:
            data['glucose_tested'] = np.array([0.0 if v in (None, 'None') else 1.0 for v in data['max_glu_serum']], dtype=np.float64)
            data['glucose_high'] = np.array([1.0 if v in ('>200', '>300') else 0.0 for v in data['max_glu_serum']], dtype=np.float64)

    # 9. Out-of-Fold Bayesian Smoothed Target Encoding
    def compute_oof_target_encoding(train_arr, y, eval_arr, n_splits=5, m=25.0, seed=42):
        n_tr = len(train_arr)
        n_ev = len(eval_arr)
        prior = float(np.mean(y))
        oof_train = np.full(n_tr, prior, dtype=np.float64)
        te_eval = np.full(n_ev, prior, dtype=np.float64)

        tr_clean = ["" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x) for x in train_arr]
        ev_clean = ["" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x) for x in eval_arr]

        rng = np.random.RandomState(seed)
        indices = np.arange(n_tr)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_splits)

        for val_idx in folds:
            mask = np.ones(n_tr, dtype=bool)
            mask[val_idx] = False

            counts = {}
            sums = {}
            for idx in np.where(mask)[0]:
                cat = tr_clean[idx]
                counts[cat] = counts.get(cat, 0) + 1
                sums[cat] = sums.get(cat, 0.0) + y[idx]

            for idx in val_idx:
                cat = tr_clean[idx]
                if cat in counts:
                    oof_train[idx] = (sums[cat] + m * prior) / (counts[cat] + m)
                else:
                    oof_train[idx] = prior

        full_counts = {}
        full_sums = {}
        for idx in range(n_tr):
            cat = tr_clean[idx]
            full_counts[cat] = full_counts.get(cat, 0) + 1
            full_sums[cat] = full_sums.get(cat, 0.0) + y[idx]

        for idx in range(n_ev):
            cat = ev_clean[idx]
            if cat in full_counts:
                te_eval[idx] = (full_sums[cat] + m * prior) / (full_counts[cat] + m)
            else:
                te_eval[idx] = prior

        return oof_train, te_eval

    te_targets = ['diag_1', 'diag_2', 'diag_3', 'discharge_disposition_id', 'medical_specialty', 'admission_source_id', 'diag_1_group']
    for col in te_targets:
        if col in train_ext and col in eval_ext:
            oof_tr, te_ev = compute_oof_target_encoding(train_ext[col], y_train, eval_ext[col], n_splits=5, m=25.0)
            train_ext[f'{col}_te'] = oof_tr
            eval_ext[f'{col}_te'] = te_ev

    # Collect all new numeric features
    extra_numeric_cols = [k for k in train_ext if k not in train and k not in extra_cat_cols]

    # 10. Encode Data
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_ext, extra_numeric=extra_numeric_cols, extra_categorical=extra_cat_cols)
    x_eval = encoder.transform(eval_ext)

    # 11. Multi-Architecture Regularized Gradient Boosting Ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=3.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    m2 = lgb.LGBMClassifier(
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=18,
        max_depth=5,
        min_child_samples=65,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=7.0,
        reg_alpha=1.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )

    m3 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.04,
        num_leaves=42,
        max_depth=7,
        min_child_samples=30,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.2,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )

    cat_indices = encoder.categorical_indices
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    return 0.40 * p1 + 0.35 * p2 + 0.25 * p3


# EVOLVE-BLOCK-END
# =====================================================================
