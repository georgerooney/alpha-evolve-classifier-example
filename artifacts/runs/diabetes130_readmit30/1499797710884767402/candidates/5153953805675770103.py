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


def target_encode(train_col, y_train, eval_col, alpha=10.0, n_splits=5):
    from sklearn.model_selection import KFold
    tr_col_clean = np.array([str(x) if x is not None else "None" for x in train_col])
    ev_col_clean = np.array([str(x) if x is not None else "None" for x in eval_col])
    
    train_encoded = np.zeros(len(tr_col_clean))
    eval_encoded = np.zeros(len(ev_col_clean))
    
    global_mean = np.mean(y_train)
    
    val_sum = {}
    val_count = {}
    for val, y in zip(tr_col_clean, y_train):
        val_sum[val] = val_sum.get(val, 0.0) + y
        val_count[val] = val_count.get(val, 0.0) + 1.0
        
    for i, val in enumerate(ev_col_clean):
        s = val_sum.get(val, 0.0)
        c = val_count.get(val, 0.0)
        eval_encoded[i] = (s + alpha * global_mean) / (c + alpha)
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(tr_col_clean):
        tr_c = tr_col_clean[train_idx]
        tr_y = y_train[train_idx]
        
        v_sum = {}
        v_count = {}
        for val, y in zip(tr_c, tr_y):
            v_sum[val] = v_sum.get(val, 0.0) + y
            v_count[val] = v_count.get(val, 0.0) + 1.0
            
        g_mean = np.mean(tr_y)
        for idx in val_idx:
            val = tr_col_clean[idx]
            s = v_sum.get(val, 0.0)
            c = v_count.get(val, 0.0)
            train_encoded[idx] = (s + alpha * g_mean) / (c + alpha)
            
    return train_encoded, eval_encoded


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    
    # Create shallow copies of dicts to add engineered features safely
    train_copy = dict(train)
    eval_copy = dict(eval_cols)
    
    extra_numeric = []
    extra_categorical = []
    
    # 1. Age midpoint
    def parse_age(age_str):
        if age_str is None:
            return np.nan
        s = age_str.replace('[', '').replace(')', '').replace(']', '').replace('(', '')
        parts = s.split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                return np.nan
        return np.nan

    train_copy['age_numeric'] = np.array([parse_age(x) for x in train_copy['age']])
    eval_copy['age_numeric'] = np.array([parse_age(x) for x in eval_copy['age']])
    extra_numeric.append('age_numeric')

    # 2. Utilization Metrics
    for d in (train_copy, eval_copy):
        d['total_utilization'] = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['has_hospitalization'] = (d['number_inpatient'] > 0).astype(float)
        d['has_emergency'] = (d['number_emergency'] > 0).astype(float)
        
    extra_numeric.extend(['total_utilization', 'has_hospitalization', 'has_emergency'])

    # 3. Ratio Features
    for d in (train_copy, eval_copy):
        hosp_days = d['time_in_hospital'] + 1e-5
        d['lab_per_day'] = d['num_lab_procedures'] / hosp_days
        d['proc_per_day'] = d['num_procedures'] / hosp_days
        d['med_per_day'] = d['num_medications'] / hosp_days
        d['diag_per_day'] = d['number_diagnoses'] / hosp_days
        
    extra_numeric.extend(['lab_per_day', 'proc_per_day', 'med_per_day', 'diag_per_day'])

    # 4. Medication stats
    non_med = {'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'}
    med_cols = [c for c in task.categorical_columns if c not in non_med]

    for d in (train_copy, eval_copy):
        n = len(d['time_in_hospital'])
        num_up = np.zeros(n)
        num_down = np.zeros(n)
        num_steady = np.zeros(n)
        num_taken = np.zeros(n)
        for col in med_cols:
            arr = d[col]
            num_up += (arr == 'Up').astype(float)
            num_down += (arr == 'Down').astype(float)
            num_steady += (arr == 'Steady').astype(float)
            num_taken += ((arr != 'No') & (arr != None)).astype(float)
        d['num_meds_up'] = num_up
        d['num_meds_down'] = num_down
        d['num_meds_steady'] = num_steady
        d['num_meds_taken'] = num_taken
        d['med_change_count'] = num_up + num_down
        
        if 'insulin' in d:
            d['insulin_change'] = ((d['insulin'] == 'Up') | (d['insulin'] == 'Down')).astype(float)
        else:
            d['insulin_change'] = np.zeros(n)

    extra_numeric.extend(['num_meds_up', 'num_meds_down', 'num_meds_steady', 'num_meds_taken', 'med_change_count', 'insulin_change'])

    # 5. ICD9 Groups
    train_diag1_grp = np.array([icd9_group(x) for x in train_copy['diag_1']])
    train_diag2_grp = np.array([icd9_group(x) for x in train_copy['diag_2']])
    train_diag3_grp = np.array([icd9_group(x) for x in train_copy['diag_3']])
    
    eval_diag1_grp = np.array([icd9_group(x) for x in eval_copy['diag_1']])
    eval_diag2_grp = np.array([icd9_group(x) for x in eval_copy['diag_2']])
    eval_diag3_grp = np.array([icd9_group(x) for x in eval_copy['diag_3']])
    
    train_copy['diag_1_group'] = train_diag1_grp
    train_copy['diag_2_group'] = train_diag2_grp
    train_copy['diag_3_group'] = train_diag3_grp
    
    eval_copy['diag_1_group'] = eval_diag1_grp
    eval_copy['diag_2_group'] = eval_diag2_grp
    eval_copy['diag_3_group'] = eval_diag3_grp
    
    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group'])

    # 6. Comorbidity counts
    all_groups = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms']
    for grp in all_groups:
        col_name = f'grp_cnt_{grp}'
        train_copy[col_name] = (train_diag1_grp == grp).astype(float) + (train_diag2_grp == grp).astype(float) + (train_diag3_grp == grp).astype(float)
        eval_copy[col_name] = (eval_diag1_grp == grp).astype(float) + (eval_diag2_grp == grp).astype(float) + (eval_diag3_grp == grp).astype(float)
        extra_numeric.append(col_name)

    # 7. Weight Measured
    train_copy['weight_measured'] = (train_copy['weight'] != None).astype(float)
    eval_copy['weight_measured'] = (eval_copy['weight'] != None).astype(float)
    extra_numeric.append('weight_measured')

    # 8. Clinical grouping of admission/discharge
    def group_discharge(disp):
        if disp is None:
            return 'missing'
        disp = str(disp)
        if disp == '1':
            return 'home'
        elif disp in ('2', '3', '4', '5', '22', '23', '24'):
            return 'transfer_facility'
        elif disp in ('6', '8'):
            return 'home_health'
        elif disp == '7':
            return 'left_ama'
        else:
            return 'other'
            
    train_copy['discharge_group'] = np.array([group_discharge(x) for x in train_copy['discharge_disposition_id']])
    eval_copy['discharge_group'] = np.array([group_discharge(x) for x in eval_copy['discharge_disposition_id']])
    extra_categorical.append('discharge_group')

    def group_admission_source(src):
        if src is None:
            return 'missing'
        src = str(src)
        if src in ('1', '2', '3'):
            return 'referral'
        elif src == '7':
            return 'emergency'
        elif src in ('4', '5', '6', '10', '22', '25'):
            return 'transfer'
        else:
            return 'other'
            
    train_copy['admission_source_group'] = np.array([group_admission_source(x) for x in train_copy['admission_source_id']])
    eval_copy['admission_source_group'] = np.array([group_admission_source(x) for x in eval_copy['admission_source_id']])
    extra_categorical.append('admission_source_group')

    def group_admission_type(typ):
        if typ is None:
            return 'missing'
        typ = str(typ)
        if typ == '1':
            return 'emergency'
        elif typ == '2':
            return 'urgent'
        elif typ == '3':
            return 'elective'
        else:
            return 'other'
            
    train_copy['admission_type_group'] = np.array([group_admission_type(x) for x in train_copy['admission_type_id']])
    eval_copy['admission_type_group'] = np.array([group_admission_type(x) for x in eval_copy['admission_type_id']])
    extra_categorical.append('admission_type_group')

    # 9. Lab result test performed indicators
    train_copy['glu_test_performed'] = (train_copy['max_glu_serum'] != 'None').astype(float)
    eval_copy['glu_test_performed'] = (eval_copy['max_glu_serum'] != 'None').astype(float)
    train_copy['a1c_test_performed'] = (train_copy['A1Cresult'] != 'None').astype(float)
    eval_copy['a1c_test_performed'] = (eval_copy['A1Cresult'] != 'None').astype(float)
    extra_numeric.extend(['glu_test_performed', 'a1c_test_performed'])

    # 10. Interaction features
    for d in (train_copy, eval_copy):
        is_high_risk_disp = np.isin(d['discharge_group'], ['home_health', 'transfer_facility']).astype(float)
        d['high_risk_disp_prior_inpatient'] = is_high_risk_disp * (d['number_inpatient'] > 0).astype(float)
    extra_numeric.append('high_risk_disp_prior_inpatient')

    # 11. Target Encoding
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id']
    for col in te_cols:
        te_train, te_eval = target_encode(train_copy[col], y_train, eval_copy[col])
        new_col_name = f'te_{col}'
        train_copy[new_col_name] = te_train
        eval_copy[new_col_name] = te_eval
        extra_numeric.append(new_col_name)

    # Now encode everything
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_copy,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical)
    )
    x_eval = encoder.transform(eval_copy)

    # Train an ensemble of multiple LightGBM models
    models = []
    
    # Model 1
    m1 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    models.append(m1)

    # Model 2
    m2 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    models.append(m2)

    # Model 3
    m3 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=45,
        min_child_samples=80,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=0.5,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    models.append(m3)

    # Predict average probabilities
    preds = np.zeros(len(x_eval))
    for model in models:
        preds += model.predict_proba(x_eval)[:, 1]
    preds /= len(models)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
