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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    from clinical_toolkit import (
        icd9_comorbidities,
        icd9_group,
        oof_target_encode,
        TabularEncoder
    )

    train_copy = dict(train)
    eval_cols_copy = dict(eval_cols)

    new_numeric_cols = []
    new_categorical_cols = []

    # 1. Comorbidities and Weighted Charlson Comorbidity Index
    train_cm = icd9_comorbidities(train_copy['diag_1'], train_copy['diag_2'], train_copy['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols_copy['diag_1'], eval_cols_copy['diag_2'], eval_cols_copy['diag_3'])
    for k in train_cm.keys():
        train_copy[k] = train_cm[k]
        eval_cols_copy[k] = eval_cm[k]
        new_numeric_cols.append(k)

    charlson_weights = {
        'cm_mi': 1.0, 'cm_chf': 1.0, 'cm_pvd': 1.0, 'cm_cerebrovascular': 1.0,
        'cm_dementia': 1.0, 'cm_pulmonary': 1.0, 'cm_rheumatic': 1.0, 'cm_peptic_ulcer': 1.0,
        'cm_liver': 1.0, 'cm_diabetes_complicated': 2.0, 'cm_hemiplegia': 2.0,
        'cm_renal': 2.0, 'cm_cancer': 2.0, 'cm_metastatic': 6.0, 'cm_hiv': 6.0,
    }
    train_charlson = np.zeros(len(y_train), dtype=float)
    eval_charlson = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    for k, w in charlson_weights.items():
        if k in train_cm:
            train_charlson += train_cm[k] * w
            eval_charlson += eval_cm[k] * w
    train_copy['charlson_weighted_score'] = train_charlson
    eval_cols_copy['charlson_weighted_score'] = eval_charlson
    new_numeric_cols.append('charlson_weighted_score')

    # Cardiorenal & complex comorbidity cross-terms
    train_copy['cm_cardiorenal'] = (train_cm['cm_chf'] * train_cm['cm_renal']).astype(float)
    eval_cols_copy['cm_cardiorenal'] = (eval_cm['cm_chf'] * eval_cm['cm_renal']).astype(float)
    new_numeric_cols.append('cm_cardiorenal')

    train_copy['cm_diabetic_renal'] = (train_cm['cm_diabetes_complicated'] * train_cm['cm_renal']).astype(float)
    eval_cols_copy['cm_diabetic_renal'] = (eval_cm['cm_diabetes_complicated'] * eval_cm['cm_renal']).astype(float)
    new_numeric_cols.append('cm_diabetic_renal')

    # 2. ICD-9 Chapter Groups and Canonical 3-Digit Category Prefixes
    def extract_diag_prefix(arr):
        res = []
        for x in arr:
            if x is None:
                res.append(None)
            else:
                s = str(x).strip()
                if not s or s == 'None' or s == '?':
                    res.append(None)
                else:
                    dot_idx = s.find('.')
                    res.append(s[:dot_idx] if dot_idx != -1 else (s[:3] if len(s) >= 3 else s))
        return np.array(res, dtype=object)

    for col in ['diag_1', 'diag_2', 'diag_3']:
        group_col_name = f"{col}_group"
        train_copy[group_col_name] = np.array([icd9_group(x) for x in train_copy[col]], dtype=object)
        eval_cols_copy[group_col_name] = np.array([icd9_group(x) for x in eval_cols_copy[col]], dtype=object)
        new_categorical_cols.append(group_col_name)

        pfx_col_name = f"{col}_prefix"
        train_copy[pfx_col_name] = extract_diag_prefix(train_copy[col])
        eval_cols_copy[pfx_col_name] = extract_diag_prefix(eval_cols_copy[col])
        new_categorical_cols.append(pfx_col_name)

        pfx_te_name = f"{pfx_col_name}_te"
        train_pfx_te, eval_pfx_te = oof_target_encode(
            train_copy[pfx_col_name], y_train, eval_cols_copy[pfx_col_name], smoothing=15.0, seed=42
        )
        train_copy[pfx_te_name] = train_pfx_te
        eval_cols_copy[pfx_te_name] = eval_pfx_te
        new_numeric_cols.append(pfx_te_name)

    # 3. High-cardinality & Pathway Target Encodings
    cols_to_target_encode = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code'
    ]
    for col in cols_to_target_encode:
        te_col_name = f"{col}_te"
        train_enc, eval_enc = oof_target_encode(
            train_copy[col], y_train, eval_cols_copy[col], smoothing=20.0, seed=42
        )
        train_copy[te_col_name] = train_enc
        eval_cols_copy[te_col_name] = eval_enc
        new_numeric_cols.append(te_col_name)

    # Joint admission source -> discharge disposition trajectory encoding
    def combine_cats(a1, a2):
        return np.array([f"{x}_{y}" for x, y in zip(a1, a2)], dtype=object)

    joint_traj_tr = combine_cats(train_copy['discharge_disposition_id'], train_copy['admission_source_id'])
    joint_traj_ev = combine_cats(eval_cols_copy['discharge_disposition_id'], eval_cols_copy['admission_source_id'])
    tr_joint_te, ev_joint_te = oof_target_encode(joint_traj_tr, y_train, joint_traj_ev, smoothing=25.0, seed=42)
    train_copy['joint_discharge_source_te'] = tr_joint_te
    eval_cols_copy['joint_discharge_source_te'] = ev_joint_te
    new_numeric_cols.append('joint_discharge_source_te')

    # 4. Utilization Intensity & Discharge Pathway Indicators
    train_copy['total_prior_visits'] = (
        train_copy['number_outpatient'] + train_copy['number_emergency'] + train_copy['number_inpatient']
    )
    eval_cols_copy['total_prior_visits'] = (
        eval_cols_copy['number_outpatient'] + eval_cols_copy['number_emergency'] + eval_cols_copy['number_inpatient']
    )
    new_numeric_cols.append('total_prior_visits')

    train_copy['has_prior_inpatient'] = (train_copy['number_inpatient'] > 0).astype(float)
    eval_cols_copy['has_prior_inpatient'] = (eval_cols_copy['number_inpatient'] > 0).astype(float)
    new_numeric_cols.append('has_prior_inpatient')

    train_copy['has_prior_emergency'] = (train_copy['number_emergency'] > 0).astype(float)
    eval_cols_copy['has_prior_emergency'] = (eval_cols_copy['number_emergency'] > 0).astype(float)
    new_numeric_cols.append('has_prior_emergency')

    train_copy['frequent_inpatient'] = (train_copy['number_inpatient'] >= 2).astype(float)
    eval_cols_copy['frequent_inpatient'] = (eval_cols_copy['number_inpatient'] >= 2).astype(float)
    new_numeric_cols.append('frequent_inpatient')

    train_copy['inpatient_ratio'] = train_copy['number_inpatient'] / (train_copy['total_prior_visits'] + 1.0)
    eval_cols_copy['inpatient_ratio'] = eval_cols_copy['number_inpatient'] / (eval_cols_copy['total_prior_visits'] + 1.0)
    new_numeric_cols.append('inpatient_ratio')

    train_copy['med_by_hospital_days'] = train_copy['num_medications'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['med_by_hospital_days'] = eval_cols_copy['num_medications'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('med_by_hospital_days')

    train_copy['lab_by_hospital_days'] = train_copy['num_lab_procedures'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['lab_by_hospital_days'] = eval_cols_copy['num_lab_procedures'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('lab_by_hospital_days')

    train_copy['procedures_by_hospital_days'] = train_copy['num_procedures'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['procedures_by_hospital_days'] = eval_cols_copy['num_procedures'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('procedures_by_hospital_days')

    # Discharge destination risk groups
    discharged_home_set = {'1'}
    discharged_facility_set = {'2', '3', '5', '6', '22', '28'}
    train_copy['discharged_to_home'] = np.array([1.0 if x in discharged_home_set else 0.0 for x in train_copy['discharge_disposition_id']], dtype=float)
    eval_cols_copy['discharged_to_home'] = np.array([1.0 if x in discharged_home_set else 0.0 for x in eval_cols_copy['discharge_disposition_id']], dtype=float)
    new_numeric_cols.append('discharged_to_home')

    train_copy['discharged_to_facility'] = np.array([1.0 if x in discharged_facility_set else 0.0 for x in train_copy['discharge_disposition_id']], dtype=float)
    eval_cols_copy['discharged_to_facility'] = np.array([1.0 if x in discharged_facility_set else 0.0 for x in eval_cols_copy['discharge_disposition_id']], dtype=float)
    new_numeric_cols.append('discharged_to_facility')

    train_copy['emergency_admission'] = np.array([1.0 if x == '1' else 0.0 for x in train_copy['admission_type_id']], dtype=float)
    eval_cols_copy['emergency_admission'] = np.array([1.0 if x == '1' else 0.0 for x in eval_cols_copy['admission_type_id']], dtype=float)
    new_numeric_cols.append('emergency_admission')

    train_copy['er_source'] = np.array([1.0 if x == '7' else 0.0 for x in train_copy['admission_source_id']], dtype=float)
    eval_cols_copy['er_source'] = np.array([1.0 if x == '7' else 0.0 for x in eval_cols_copy['admission_source_id']], dtype=float)
    new_numeric_cols.append('er_source')

    # 5. Medication and Dosage Change Counts
    med_cols = []
    for col in task.categorical_columns:
        if col in ['change', 'diabetesMed']:
            continue
        unique_vals = set(train_copy[col])
        if any(v in unique_vals for v in ['Steady', 'Up', 'Down']):
            med_cols.append(col)

    train_num_active = np.zeros(len(y_train), dtype=float)
    eval_num_active = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    train_num_up = np.zeros(len(y_train), dtype=float)
    eval_num_up = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    train_num_down = np.zeros(len(y_train), dtype=float)
    eval_num_down = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    train_num_changes = np.zeros(len(y_train), dtype=float)
    eval_num_changes = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)

    for col in med_cols:
        tr_active = ((train_copy[col] != 'No') & (train_copy[col] != None)).astype(float)
        ev_active = ((eval_cols_copy[col] != 'No') & (eval_cols_copy[col] != None)).astype(float)
        train_num_active += tr_active
        eval_num_active += ev_active

        tr_up = (train_copy[col] == 'Up').astype(float)
        ev_up = (eval_cols_copy[col] == 'Up').astype(float)
        train_num_up += tr_up
        eval_num_up += ev_up

        tr_down = (train_copy[col] == 'Down').astype(float)
        ev_down = (eval_cols_copy[col] == 'Down').astype(float)
        train_num_down += tr_down
        eval_num_down += ev_down

        train_num_changes += tr_up + tr_down
        eval_num_changes += ev_up + ev_down

    train_copy['num_active_meds'] = train_num_active
    eval_cols_copy['num_active_meds'] = eval_num_active
    new_numeric_cols.append('num_active_meds')

    train_copy['num_up_meds'] = train_num_up
    eval_cols_copy['num_up_meds'] = eval_num_up
    new_numeric_cols.append('num_up_meds')

    train_copy['num_down_meds'] = train_num_down
    eval_cols_copy['num_down_meds'] = eval_num_down
    new_numeric_cols.append('num_down_meds')

    train_copy['num_changes'] = train_num_changes
    eval_cols_copy['num_changes'] = eval_num_changes
    new_numeric_cols.append('num_changes')

    # Insulin specific treatment indicators
    train_copy['insulin_active'] = np.array([1.0 if x not in ('No', None) else 0.0 for x in train_copy['insulin']], dtype=float)
    eval_cols_copy['insulin_active'] = np.array([1.0 if x not in ('No', None) else 0.0 for x in eval_cols_copy['insulin']], dtype=float)
    new_numeric_cols.append('insulin_active')

    train_copy['insulin_change'] = np.array([1.0 if x in ('Up', 'Down') else 0.0 for x in train_copy['insulin']], dtype=float)
    eval_cols_copy['insulin_change'] = np.array([1.0 if x in ('Up', 'Down') else 0.0 for x in eval_cols_copy['insulin']], dtype=float)
    new_numeric_cols.append('insulin_change')

    # Glycemic test coverage and control
    train_copy['a1c_tested'] = (train_copy['A1Cresult'] != 'None').astype(float)
    eval_cols_copy['a1c_tested'] = (eval_cols_copy['A1Cresult'] != 'None').astype(float)
    new_numeric_cols.append('a1c_tested')

    train_copy['glu_tested'] = (train_copy['max_glu_serum'] != 'None').astype(float)
    eval_cols_copy['glu_tested'] = (eval_cols_copy['max_glu_serum'] != 'None').astype(float)
    new_numeric_cols.append('glu_tested')

    # Continuous representation of age groups
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0, '[40-50)': 45.0,
        '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0, '[80-90)': 85.0, '[90-100)': 95.0
    }
    def map_age(age_arr):
        return np.array([age_map.get(x, np.nan) for x in age_arr], dtype=float)

    train_copy['age_numeric'] = map_age(train_copy['age'])
    eval_cols_copy['age_numeric'] = map_age(eval_cols_copy['age'])
    new_numeric_cols.append('age_numeric')

    # 6. Tensor Decomposition: Low-Rank Latent Clinical Phenotype Factors
    # Unfold the patient x condition x medication interaction tensor and extract low-rank principal phenotypes
    pheno_features_tr = [
        train_cm[k] for k in sorted(train_cm.keys())
    ]
    pheno_features_ev = [
        eval_cm[k] for k in sorted(eval_cm.keys())
    ]
    for col in med_cols:
        pheno_features_tr.append(((train_copy[col] != 'No') & (train_copy[col] != None)).astype(float))
        pheno_features_ev.append(((eval_cols_copy[col] != 'No') & (eval_cols_copy[col] != None)).astype(float))
        pheno_features_tr.append(((train_copy[col] == 'Up') | (train_copy[col] == 'Down')).astype(float))
        pheno_features_ev.append(((eval_cols_copy[col] == 'Up') | (eval_cols_copy[col] == 'Down')).astype(float))

    base_num_indicators = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'total_prior_visits', 'charlson_weighted_score', 'a1c_tested', 'glu_tested'
    ]
    for col in base_num_indicators:
        pheno_features_tr.append(np.asarray(train_copy[col], dtype=float))
        pheno_features_ev.append(np.asarray(eval_cols_copy[col], dtype=float))

    train_pheno_mat = np.column_stack(pheno_features_tr)
    eval_pheno_mat = np.column_stack(pheno_features_ev)

    # Impute missing values with column means prior to SVD
    col_means = np.nanmean(train_pheno_mat, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)

    tr_nan_idx = np.where(np.isnan(train_pheno_mat))
    train_pheno_clean = train_pheno_mat.copy()
    train_pheno_clean[tr_nan_idx] = np.take(col_means, tr_nan_idx[1])

    ev_nan_idx = np.where(np.isnan(eval_pheno_mat))
    eval_pheno_clean = eval_pheno_mat.copy()
    eval_pheno_clean[ev_nan_idx] = np.take(col_means, ev_nan_idx[1])

    scaler = StandardScaler()
    train_pheno_scaled = scaler.fit_transform(train_pheno_clean)
    eval_pheno_scaled = scaler.transform(eval_pheno_clean)

    # Low-rank factor decomposition
    n_svd_components = 6
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
    train_svd_factors = svd.fit_transform(train_pheno_scaled)
    eval_svd_factors = svd.transform(eval_pheno_scaled)

    for comp_idx in range(n_svd_components):
        col_name = f'svd_phenotype_{comp_idx}'
        train_copy[col_name] = train_svd_factors[:, comp_idx]
        eval_cols_copy[col_name] = eval_svd_factors[:, comp_idx]
        new_numeric_cols.append(col_name)

    # Encode tabular features
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_copy,
        extra_numeric=tuple(new_numeric_cols),
        extra_categorical=tuple(new_categorical_cols)
    )
    x_train = encoder.transform(train_copy)
    x_eval = encoder.transform(eval_cols_copy)

    # 7. Cross-Validated Multi-Model Training with OOF Simplex Weight Optimization
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb1 = np.zeros(len(y_train))
    oof_lgb2 = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))

    eval_preds_lgb1 = np.zeros(len(x_eval))
    eval_preds_lgb2 = np.zeros(len(x_eval))
    eval_preds_xgb = np.zeros(len(x_eval))

    use_xgb = True
    try:
        import xgboost as xgb
    except Exception:
        use_xgb = False

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        x_va, y_va = x_train[val_idx], y_train[val_idx]

        # Model 1: Leaf-wise deep capacity LightGBM
        m1 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.5,
            reg_alpha=0.1,
            cat_smooth=40.0,
            random_state=42 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb1[val_idx] = m1.predict_proba(x_va)[:, 1]
        eval_preds_lgb1 += m1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: Conservative regularized shallow LightGBM
        m2 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.02,
            num_leaves=18,
            min_child_samples=160,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=60.0,
            random_state=123 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb2[val_idx] = m2.predict_proba(x_va)[:, 1]
        eval_preds_lgb2 += m2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: XGBoost (level-wise depth-bounded tree growth)
        if use_xgb:
            try:
                m_xgb = xgb.XGBClassifier(
                    n_estimators=250,
                    learning_rate=0.025,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.65,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    tree_method='hist',
                    random_state=42 + fold_idx,
                    n_jobs=1,
                    eval_metric='logloss',
                )
                m_xgb.fit(x_tr, y_tr)
                oof_xgb[val_idx] = m_xgb.predict_proba(x_va)[:, 1]
                eval_preds_xgb += m_xgb.predict_proba(x_eval)[:, 1] / 5.0
            except Exception:
                use_xgb = False

    # Optimize blending weights on OOF predictions directly maximizing ROC-AUC
    if use_xgb and np.any(oof_xgb > 0):
        best_auc = -1.0
        best_weights = (0.45, 0.35, 0.20)
        for w1 in np.linspace(0.15, 0.75, 13):
            for w2 in np.linspace(0.10, 0.70, 13):
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    blend = w1 * oof_lgb1 + w2 * oof_lgb2 + w3 * oof_xgb
                    score = roc_auc_score(y_train, blend)
                    if score > best_auc:
                        best_auc = score
                        best_weights = (w1, w2, w3)
        final_preds = best_weights[0] * eval_preds_lgb1 + best_weights[1] * eval_preds_lgb2 + best_weights[2] * eval_preds_xgb
    else:
        best_auc = -1.0
        best_w = 0.5
        for w in np.linspace(0.05, 0.95, 19):
            blend = w * oof_lgb1 + (1.0 - w) * oof_lgb2
            score = roc_auc_score(y_train, blend)
            if score > best_auc:
                best_auc = score
                best_w = w
        final_preds = best_w * eval_preds_lgb1 + (1.0 - best_w) * eval_preds_lgb2

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
