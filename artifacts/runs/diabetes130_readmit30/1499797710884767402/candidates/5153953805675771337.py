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
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    try:
        from clinical_toolkit import icd9_group, TabularEncoder
    except ImportError:
        pass
    from sklearn.model_selection import StratifiedKFold

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Clean continuous age
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for col_dict in [train_data, eval_data]:
        raw_age = col_dict.get('age', np.array([]))
        col_dict['num_age'] = np.array([age_map.get(a, np.nan) for a in raw_age], dtype=np.float64)
    extra_numeric.append('num_age')

    # 2. Non-redundant Utilisation & Hospital Stay intensity
    for col_dict in [train_data, eval_data]:
        inp = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        outp = np.nan_to_num(col_dict.get('number_outpatient', 0.0), nan=0.0)
        tot_visits = inp + emg + outp

        col_dict['util_total_visits'] = tot_visits
        col_dict['util_inpatient_ratio'] = inp / (tot_visits + 1.0)
        col_dict['util_emergency_ratio'] = emg / (tot_visits + 1.0)

        time_safe = np.maximum(np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0), 1.0)
        num_labs = np.nan_to_num(col_dict.get('num_lab_procedures', 0.0), nan=0.0)
        num_proc = np.nan_to_num(col_dict.get('num_procedures', 0.0), nan=0.0)
        num_meds = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)

        col_dict['hosp_labs_per_day'] = num_labs / time_safe
        col_dict['hosp_meds_per_day'] = num_meds / time_safe
        col_dict['hosp_service_intensity'] = (num_labs + num_proc + num_meds) / time_safe

    extra_numeric.extend([
        'util_total_visits', 'util_inpatient_ratio', 'util_emergency_ratio',
        'hosp_labs_per_day', 'hosp_meds_per_day', 'hosp_service_intensity'
    ])

    # 3. Medication summary features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col_dict in [train_data, eval_data]:
        n_rows = len(col_dict['time_in_hospital'])
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in col_dict:
                arr = col_dict[m]
                active += np.isin(arr, ['Steady', 'Up', 'Down']).astype(np.float64)
                up += (arr == 'Up').astype(np.float64)
                down += (arr == 'Down').astype(np.float64)

        insulin_arr = col_dict.get('insulin', np.full(n_rows, 'No', dtype=object))
        ins_active = np.isin(insulin_arr, ['Steady', 'Up', 'Down']).astype(np.float64)
        ins_up = (insulin_arr == 'Up').astype(np.float64)
        ins_down = (insulin_arr == 'Down').astype(np.float64)
        oral_meds = np.maximum(0.0, active - ins_active)

        col_dict['med_num_active'] = active
        col_dict['med_num_changed'] = up + down
        col_dict['med_insulin_active'] = ins_active
        col_dict['med_insulin_up'] = ins_up
        col_dict['med_insulin_down'] = ins_down
        col_dict['med_insulin_plus_oral'] = ((ins_active > 0.0) & (oral_meds > 0.0)).astype(np.float64)

    extra_numeric.extend([
        'med_num_active', 'med_num_changed', 'med_insulin_active',
        'med_insulin_up', 'med_insulin_down', 'med_insulin_plus_oral'
    ])

    # 4. Vectorized diagnosis grouping & comorbidity patterns
    def extract_prefix(code):
        if code is None or code == '?' or code == 'None':
            return 'missing'
        s = str(code).strip()
        return s.split('.')[0] if s else 'missing'

    for col_dict in [train_data, eval_data]:
        d1 = col_dict.get('diag_1', np.array([]))
        d2 = col_dict.get('diag_2', np.array([]))
        d3 = col_dict.get('diag_3', np.array([]))

        g1 = np.array([icd9_group(c) for c in d1], dtype=object)
        g2 = np.array([icd9_group(c) for c in d2], dtype=object)
        g3 = np.array([icd9_group(c) for c in d3], dtype=object)
        p1 = np.array([extract_prefix(c) for c in d1], dtype=object)
        p2 = np.array([extract_prefix(c) for c in d2], dtype=object)
        p3 = np.array([extract_prefix(c) for c in d3], dtype=object)

        col_dict['diag_1_group'] = g1
        col_dict['diag_2_group'] = g2
        col_dict['diag_3_group'] = g3
        col_dict['diag_1_prefix'] = p1

        circ = (g1 == 'circulatory') | (g2 == 'circulatory') | (g3 == 'circulatory')
        diab = (g1 == 'diabetes') | (g2 == 'diabetes') | (g3 == 'diabetes')
        resp = (g1 == 'respiratory') | (g2 == 'respiratory') | (g3 == 'respiratory')
        digest = (g1 == 'digestive') | (g2 == 'digestive') | (g3 == 'digestive')
        genit = (g1 == 'genitourinary') | (g2 == 'genitourinary') | (g3 == 'genitourinary')

        chf = (p1 == '428') | (p2 == '428') | (p3 == '428')
        ckd = (p1 == '585') | (p2 == '585') | (p3 == '585') | (p1 == '586') | (p2 == '586') | (p3 == '586')

        col_dict['comorb_has_circ'] = circ.astype(np.float64)
        col_dict['comorb_has_diab'] = diab.astype(np.float64)
        col_dict['comorb_has_resp'] = resp.astype(np.float64)
        col_dict['comorb_has_genit'] = genit.astype(np.float64)
        col_dict['comorb_has_chf'] = chf.astype(np.float64)
        col_dict['comorb_has_ckd'] = ckd.astype(np.float64)
        col_dict['comorb_num_systems'] = (
            circ.astype(int) + diab.astype(int) + resp.astype(int) + digest.astype(int) + genit.astype(int)
        ).astype(np.float64)

    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group', 'diag_1_prefix'])
    extra_numeric.extend([
        'comorb_has_circ', 'comorb_has_diab', 'comorb_has_resp', 'comorb_has_genit',
        'comorb_has_chf', 'comorb_has_ckd', 'comorb_num_systems'
    ])

    # 5. Out-of-fold Smoothed Target Encoding on key high-cardinality features
    def compute_oof_te(tr_arr, y_tr, ev_arr, smoothing=25.0, n_splits=5, seed=42):
        prior = float(np.mean(y_tr))
        tr_te = np.full(len(tr_arr), prior, dtype=np.float64)
        ev_te = np.full(len(ev_arr), prior, dtype=np.float64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_arr, y_tr):
            sub_c = tr_arr[train_idx]
            sub_y = y_tr[train_idx]
            v_c = tr_arr[val_idx]

            counts = {}
            sums = {}
            for c, y in zip(sub_c, sub_y):
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y

            vals = np.empty(len(val_idx), dtype=np.float64)
            for i, c in enumerate(v_c):
                cnt = counts.get(c, 0)
                vals[i] = (sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior
            tr_te[val_idx] = vals

        full_counts = {}
        full_sums = {}
        for c, y in zip(tr_arr, y_tr):
            full_counts[c] = full_counts.get(c, 0) + 1
            full_sums[c] = full_sums.get(c, 0.0) + y

        for i, c in enumerate(ev_arr):
            cnt = full_counts.get(c, 0)
            ev_te[i] = (full_sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior

        return tr_te, ev_te

    te_targets = ['discharge_disposition_id', 'admission_source_id', 'medical_specialty', 'diag_1_prefix']
    for c_name in te_targets:
        if c_name in train_data:
            tr_te, ev_te = compute_oof_te(train_data[c_name], y_train, eval_data[c_name], smoothing=25.0)
            te_col_name = f'te_{c_name}'
            train_data[te_col_name] = tr_te
            eval_data[te_col_name] = ev_te
            extra_numeric.append(te_col_name)

    # 6. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 7. Simplified multi-depth LightGBM ensemble
    models = [
        # Shallow, robust trees
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=16,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Medium trees
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=24,
            min_child_samples=70,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=8.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Deeper trees with stronger regularization
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=32,
            min_child_samples=90,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=10.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    preds = np.nan_to_num(preds, nan=0.114, posinf=1.0, neginf=0.0)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
