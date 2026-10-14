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


def add_features(cols_dict: Dict[str, np.ndarray], med_cols: list) -> Dict[str, np.ndarray]:
    new_cols = dict(cols_dict)
    n = len(next(iter(cols_dict.values())))
    
    # 1. Age conversion
    age_arr = cols_dict.get('age')
    parsed_age = []
    if age_arr is not None:
        for val in age_arr:
            if val is None or not isinstance(val, str):
                parsed_age.append(np.nan)
            else:
                s = val.strip('[]()')
                parts = s.split('-')
                if len(parts) == 2:
                    try:
                        parsed_age.append((float(parts[0]) + float(parts[1])) / 2.0)
                    except ValueError:
                        parsed_age.append(np.nan)
                else:
                    parsed_age.append(np.nan)
    else:
        parsed_age = [np.nan] * n
    new_cols['age_num'] = np.array(parsed_age, dtype=np.float64)
    
    # 2. Utilization intensity
    outpatient = cols_dict.get('number_outpatient')
    emergency = cols_dict.get('number_emergency')
    inpatient = cols_dict.get('number_inpatient')
    if outpatient is not None and emergency is not None and inpatient is not None:
        total_visits = outpatient + emergency + inpatient
        new_cols['total_visits'] = total_visits.astype(np.float64)
        new_cols['has_prior_visits'] = (total_visits > 0).astype(np.float64)
        new_cols['inpatient_ratio'] = (inpatient / (total_visits + 1.0)).astype(np.float64)
        new_cols['emergency_ratio'] = (emergency / (total_visits + 1.0)).astype(np.float64)
    else:
        new_cols['total_visits'] = np.zeros(n, dtype=np.float64)
        new_cols['has_prior_visits'] = np.zeros(n, dtype=np.float64)
        new_cols['inpatient_ratio'] = np.zeros(n, dtype=np.float64)
        new_cols['emergency_ratio'] = np.zeros(n, dtype=np.float64)
        
    # 3. ICD-9 groupings
    diag_1 = cols_dict.get('diag_1')
    diag_2 = cols_dict.get('diag_2')
    diag_3 = cols_dict.get('diag_3')
    
    if diag_1 is not None:
        g1 = np.array([icd9_group(str(x)) if (x is not None and x != 'None') else 'missing' for x in diag_1], dtype=object)
    else:
        g1 = np.array(['missing'] * n, dtype=object)
    new_cols['diag_1_group'] = g1
    
    if diag_2 is not None:
        g2 = np.array([icd9_group(str(x)) if (x is not None and x != 'None') else 'missing' for x in diag_2], dtype=object)
    else:
        g2 = np.array(['missing'] * n, dtype=object)
    new_cols['diag_2_group'] = g2
        
    if diag_3 is not None:
        g3 = np.array([icd9_group(str(x)) if (x is not None and x != 'None') else 'missing' for x in diag_3], dtype=object)
    else:
        g3 = np.array(['missing'] * n, dtype=object)
    new_cols['diag_3_group'] = g3
        
    # 4. Comorbidities
    unique_groups = []
    has_diab = []
    has_circ = []
    for i in range(n):
        grp_set = {g1[i], g2[i], g3[i]}
        unique_groups.append(len(grp_set - {'missing'}))
        has_diab.append(1.0 if 'diabetes' in grp_set else 0.0)
        has_circ.append(1.0 if 'circulatory' in grp_set else 0.0)
        
    new_cols['num_uniq_diag_groups'] = np.array(unique_groups, dtype=np.float64)
    new_cols['has_diabetes_diag'] = np.array(has_diab, dtype=np.float64)
    new_cols['has_circulatory_diag'] = np.array(has_circ, dtype=np.float64)

    # 5. Med features
    active = np.zeros(n, dtype=np.float64)
    up = np.zeros(n, dtype=np.float64)
    down = np.zeros(n, dtype=np.float64)
    change = np.zeros(n, dtype=np.float64)
    
    for col in med_cols:
        if col in cols_dict:
            arr = cols_dict[col]
            arr_str = arr.astype(str)
            active += (arr_str != 'No') & (arr_str != 'None') & (arr_str != 'nan')
            up += (arr_str == 'Up')
            down += (arr_str == 'Down')
            change += (arr_str == 'Up') | (arr_str == 'Down')
            
    new_cols['num_active_meds'] = active
    new_cols['num_med_up'] = up
    new_cols['num_med_down'] = down
    new_cols['num_med_changes'] = change
    
    # 6. Lab result indicators
    max_glu = cols_dict.get('max_glu_serum')
    if max_glu is not None:
        new_cols['max_glu_measured'] = (max_glu.astype(str) != 'None').astype(np.float64)
    else:
        new_cols['max_glu_measured'] = np.zeros(n, dtype=np.float64)
       
    a1c = cols_dict.get('A1Cresult')
    if a1c is not None:
        new_cols['a1c_measured'] = (a1c.astype(str) != 'None').astype(np.float64)
    else:
        new_cols['a1c_measured'] = np.zeros(n, dtype=np.float64)
        
    # 7. Specialties missingness
    med_spec = cols_dict.get('medical_specialty')
    if med_spec is not None:
        spec_str = med_spec.astype(str)
        new_cols['has_medical_specialty'] = ((spec_str != 'None') & (spec_str != 'nan') & (spec_str != '?')).astype(np.float64)
    else:
        new_cols['has_medical_specialty'] = np.zeros(n, dtype=np.float64)
        
    # 8. Discharge disposition groupings
    discharge = cols_dict.get('discharge_disposition_id')
    if discharge is not None:
        d_str = discharge.astype(str)
        new_cols['discharged_to_home'] = (d_str == '1').astype(np.float64)
        new_cols['discharged_to_home_health'] = (d_str == '6').astype(np.float64)
        facility_set = {'2', '3', '4', '5', '9', '10', '12', '15', '16', '17', '22', '23', '24', '27', '28', '29', '30'}
        new_cols['discharged_to_facility'] = np.array([1.0 if x in facility_set else 0.0 for x in d_str], dtype=np.float64)
    else:
        new_cols['discharged_to_home'] = np.zeros(n, dtype=np.float64)
        new_cols['discharged_to_home_health'] = np.zeros(n, dtype=np.float64)
        new_cols['discharged_to_facility'] = np.zeros(n, dtype=np.float64)
        
    # 9. Specific medications
    insulin = cols_dict.get('insulin')
    if insulin is not None:
        ins_str = insulin.astype(str)
        new_cols['insulin_active'] = ((ins_str != 'No') & (ins_str != 'None') & (ins_str != 'nan')).astype(np.float64)
        new_cols['insulin_changed'] = ((ins_str == 'Up') | (ins_str == 'Down')).astype(np.float64)
    else:
        new_cols['insulin_active'] = np.zeros(n, dtype=np.float64)
        new_cols['insulin_changed'] = np.zeros(n, dtype=np.float64)
        
    metformin = cols_dict.get('metformin')
    if metformin is not None:
        met_str = metformin.astype(str)
        new_cols['metformin_active'] = ((met_str != 'No') & (met_str != 'None') & (met_str != 'nan')).astype(np.float64)
    else:
        new_cols['metformin_active'] = np.zeros(n, dtype=np.float64)
        
    return new_cols


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    # Identify medication columns
    med_cols = []
    for col, arr in train.items():
        if arr.dtype == object:
            unique_vals = set(np.unique(arr.astype(str)))
            if 'No' in unique_vals and 'Steady' in unique_vals:
                med_cols.append(col)

    # Add engineered features
    train_with_feats = add_features(train, med_cols)
    eval_with_feats = add_features(eval_cols, med_cols)

    extra_numeric = [
        'age_num', 'total_visits', 'has_prior_visits', 'inpatient_ratio', 
        'emergency_ratio', 'num_uniq_diag_groups', 'has_diabetes_diag', 
        'has_circulatory_diag', 'num_active_meds', 'num_med_up', 
        'num_med_down', 'num_med_changes', 'max_glu_measured', 
        'a1c_measured', 'has_medical_specialty', 'discharged_to_home', 
        'discharged_to_home_health', 'discharged_to_facility',
        'insulin_active', 'insulin_changed', 'metformin_active'
    ]
    extra_categorical = [
        'diag_1_group', 'diag_2_group', 'diag_3_group'
    ]

    encoder = TabularEncoder(min_count=10)
    encoder.fit(
        task, 
        train_with_feats, 
        extra_numeric=extra_numeric, 
        extra_categorical=extra_categorical
    )
    x_train = encoder.transform(train_with_feats)
    x_eval = encoder.transform(eval_with_feats)

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval))

    configs = [
        # Configuration 1: Standard tree depth
        {
            'n_estimators': 220,
            'learning_rate': 0.04,
            'num_leaves': 31,
            'min_child_samples': 45,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.7,
            'reg_lambda': 3.0,
            'reg_alpha': 0.5,
            'random_state': 100,
        },
        # Configuration 2: Shallower, more robust trees
        {
            'n_estimators': 250,
            'learning_rate': 0.04,
            'num_leaves': 15,
            'min_child_samples': 30,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.65,
            'reg_lambda': 5.0,
            'reg_alpha': 1.0,
            'random_state': 200,
        }
    ]

    total_models = 5 * len(configs)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        for config in configs:
            cfg = dict(config)
            cfg['random_state'] += fold
            cfg['n_jobs'] = 1
            cfg['verbose'] = -1
            
            model = lgb.LGBMClassifier(**cfg)
            model.fit(
                x_tr, 
                y_tr, 
                categorical_feature=encoder.categorical_indices
            )
            preds += model.predict_proba(x_eval)[:, 1] / total_models

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
