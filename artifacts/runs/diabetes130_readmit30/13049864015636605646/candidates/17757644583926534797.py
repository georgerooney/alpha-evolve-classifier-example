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
    from clinical_toolkit import TabularEncoder, icd9_group, icd9_comorbidities, oof_target_encode

    train_new = dict(train)
    eval_new = dict(eval_cols)
    
    extra_numeric = []
    extra_categorical = []
    
    # 1. Clinical Comorbidities
    comorb_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    comorb_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for key in comorb_train.keys():
        train_new[key] = comorb_train[key]
        eval_new[key] = comorb_eval[key]
        extra_numeric.append(key)
        
    # 2. ICD-9 Chapter Groups mapping
    train_new['diag_1_group'] = np.array([icd9_group(x) if x is not None else 'missing' for x in train['diag_1']], dtype=object)
    train_new['diag_2_group'] = np.array([icd9_group(x) if x is not None else 'missing' for x in train['diag_2']], dtype=object)
    train_new['diag_3_group'] = np.array([icd9_group(x) if x is not None else 'missing' for x in train['diag_3']], dtype=object)
    
    eval_new['diag_1_group'] = np.array([icd9_group(x) if x is not None else 'missing' for x in eval_cols['diag_1']], dtype=object)
    eval_new['diag_2_group'] = np.array([icd9_group(x) if x is not None else 'missing' for x in eval_cols['diag_2']], dtype=object)
    eval_new['diag_3_group'] = np.array([icd9_group(x) if x is not None else 'missing' for x in eval_cols['diag_3']], dtype=object)
    
    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group'])
    
    # 3. Healthcare Utilisation / Visit Intensity features
    for d in [train_new, eval_new]:
        outpatient = np.nan_to_num(d['number_outpatient'].astype(float), nan=0.0)
        emergency = np.nan_to_num(d['number_emergency'].astype(float), nan=0.0)
        inpatient = np.nan_to_num(d['number_inpatient'].astype(float), nan=0.0)
        total_visits = outpatient + emergency + inpatient
        d['total_visits'] = total_visits
        d['has_visits'] = (total_visits > 0).astype(float)
        d['inpatient_ratio'] = inpatient / (total_visits + 1e-5)
        d['outpatient_ratio'] = outpatient / (total_visits + 1e-5)
        d['emergency_ratio'] = emergency / (total_visits + 1e-5)
        
    extra_numeric.extend(['total_visits', 'has_visits', 'inpatient_ratio', 'outpatient_ratio', 'emergency_ratio'])

    # 3b. Advanced Demographics and Clinical Markers
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
        "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
    }
    train_new['age_numeric'] = np.array([age_map.get(x, 65.0) for x in train['age']], dtype=float)
    eval_new['age_numeric'] = np.array([age_map.get(x, 65.0) for x in eval_cols['age']], dtype=float)
    
    train_new['has_weight'] = np.array([0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else 1.0 for x in train['weight']], dtype=float)
    eval_new['has_weight'] = np.array([0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else 1.0 for x in eval_cols['weight']], dtype=float)
    
    for d in [train_new, eval_new]:
        d['med_intensity'] = d['num_medications'].astype(float) / (d['time_in_hospital'].astype(float) + 1e-5)
        d['lab_intensity'] = d['num_lab_procedures'].astype(float) / (d['time_in_hospital'].astype(float) + 1e-5)
        d['procedures_intensity'] = d['num_procedures'].astype(float) / (d['time_in_hospital'].astype(float) + 1e-5)
        d['is_emergency_admission'] = np.array([(1.0 if x == '1' else 0.0) for x in d['admission_type_id']], dtype=float)
        d['is_discharged_home'] = np.array([(1.0 if x == '1' else 0.0) for x in d['discharge_disposition_id']], dtype=float)
        
    a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    train_new['a1c_numeric'] = np.array([a1c_map.get(str(x), 0) for x in train['A1Cresult']], dtype=float)
    eval_new['a1c_numeric'] = np.array([a1c_map.get(str(x), 0) for x in eval_cols['A1Cresult']], dtype=float)
    
    train_new['glu_numeric'] = np.array([glu_map.get(str(x), 0) for x in train['max_glu_serum']], dtype=float)
    eval_new['glu_numeric'] = np.array([glu_map.get(str(x), 0) for x in eval_cols['max_glu_serum']], dtype=float)
    
    train_new['is_on_insulin'] = np.array([(1.0 if x != 'No' else 0.0) for x in train['insulin']], dtype=float)
    eval_new['is_on_insulin'] = np.array([(1.0 if x != 'No' else 0.0) for x in eval_cols['insulin']], dtype=float)
    
    train_new['insulin_change'] = np.array([(1.0 if x in ('Up', 'Down') else 0.0) for x in train['insulin']], dtype=float)
    eval_new['insulin_change'] = np.array([(1.0 if x in ('Up', 'Down') else 0.0) for x in eval_cols['insulin']], dtype=float)
    
    extra_numeric.extend([
        'age_numeric', 'has_weight', 'med_intensity', 'lab_intensity', 'procedures_intensity',
        'is_emergency_admission', 'is_discharged_home', 'a1c_numeric', 'glu_numeric',
        'is_on_insulin', 'insulin_change'
    ])
    
    # 4. Medication features (Active count, directional dosing changes)
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    
    for d in [train_new, eval_new]:
        active = np.zeros(len(d['insulin']), dtype=float)
        up = np.zeros_like(active)
        down = np.zeros_like(active)
        steady = np.zeros_like(active)
        for col in med_cols:
            if col in d:
                v = d[col]
                is_up = (v == 'Up')
                is_down = (v == 'Down')
                is_steady = (v == 'Steady')
                up += is_up
                down += is_down
                steady += is_steady
                active += is_up | is_down | is_steady
        d['num_active_meds'] = active
        d['num_up_meds'] = up
        d['num_down_meds'] = down
        d['num_steady_meds'] = steady
        d['change_numeric'] = (d['change'] == 'Ch').astype(float)
        d['diabetesMed_numeric'] = (d['diabetesMed'] == 'Yes').astype(float)
        d['multiple_meds_flag'] = (active > 1).astype(float)
        
    extra_numeric.extend(['num_active_meds', 'num_up_meds', 'num_down_meds', 'num_steady_meds', 'change_numeric', 'diabetesMed_numeric', 'multiple_meds_flag'])
    
    # 5. Out-Of-Fold Target Encoding for high-cardinality and important categorical features
    train_new['diag_disch_comb'] = np.array([f"{g}_{d}" for g, d in zip(train_new['diag_1_group'], train_new['discharge_disposition_id'])], dtype=object)
    eval_new['diag_disch_comb'] = np.array([f"{g}_{d}" for g, d in zip(eval_new['diag_1_group'], eval_new['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag_disch_comb')

    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'diag_1_group', 'diag_2_group', 'diag_3_group', 'payer_code', 'diag_disch_comb'
    ]
    for col in te_cols:
        tr_enc, ev_enc = oof_target_encode(train_new[col], y_train, eval_new[col], n_folds=5, smoothing=20.0, seed=42)
        train_new[f"{col}_te"] = tr_enc
        eval_new[f"{col}_te"] = ev_enc
        extra_numeric.append(f"{col}_te")

    # High-quality interaction features
    for d in [train_new, eval_new]:
        d['inpatient_disch_interaction'] = d['number_inpatient'].astype(float) * d['discharge_disposition_id_te']
        d['total_visits_disch_interaction'] = d['total_visits'].astype(float) * d['discharge_disposition_id_te']
        d['inpatient_ratio_disch_interaction'] = d['inpatient_ratio'] * d['discharge_disposition_id_te']
        d['emergency_disch_interaction'] = d['number_emergency'].astype(float) * d['discharge_disposition_id_te']
        d['inpatient_diag_interaction'] = d['number_inpatient'].astype(float) * d['diag_1_te']
        
    extra_numeric.extend([
        'inpatient_disch_interaction', 
        'total_visits_disch_interaction', 
        'inpatient_ratio_disch_interaction',
        'emergency_disch_interaction',
        'inpatient_diag_interaction'
    ])

    # 6. Fit TabularEncoder to obtain processed data matrices
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_new, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_new)
    x_eval = encoder.transform(eval_new)

    # 7. Model Ensemble: 3 diverse LGBM Models + 1 XGBoost Model
    import xgboost as xgb

    model1 = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    
    model2 = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.02,
        num_leaves=45,
        min_child_samples=250,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        reg_alpha=0.5,
        cat_smooth=100.0,
        random_state=100,
        n_jobs=1,
        verbose=-1,
    )
    
    model3 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.012,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.6,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=0.5,
        reg_alpha=0.05,
        cat_smooth=30.0,
        random_state=2023,
        n_jobs=1,
        verbose=-1,
    )

    model_xgb = xgb.XGBClassifier(
        n_estimators=160,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=4,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
    )

    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred1 = model1.predict_proba(x_eval)[:, 1]
    
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred2 = model2.predict_proba(x_eval)[:, 1]
    
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred3 = model3.predict_proba(x_eval)[:, 1]

    model_xgb.fit(x_train, y_train)
    pred_xgb = model_xgb.predict_proba(x_eval)[:, 1]
    
    return 0.3 * pred1 + 0.25 * pred2 + 0.25 * pred3 + 0.2 * pred_xgb


# EVOLVE-BLOCK-END
# =====================================================================
