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
    import lightgbm as lgb
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedKFold
    
    extra_numeric = []
    extra_categorical = []
    
    # Make shallow copies to avoid modifying caller's dict keys
    train = dict(train)
    eval_cols = dict(eval_cols)
    n_train = len(y_train)
    n_eval = len(eval_cols['time_in_hospital'])
    
    # 1. Comorbidities extraction and clinically weighted Charlson Index
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # Validated epidemiological Charlson Comorbidity Index weighting
    def calc_weighted_cci(data_cm):
        return (
            1.0 * (data_cm['cm_mi'] + data_cm['cm_chf'] + data_cm['cm_pvd'] +
                   data_cm['cm_cerebrovascular'] + data_cm['cm_dementia'] +
                   data_cm['cm_pulmonary'] + data_cm['cm_rheumatic'] +
                   data_cm['cm_peptic_ulcer']) +
            2.0 * (data_cm['cm_hemiplegia'] + data_cm['cm_renal'] +
                   data_cm['cm_diabetes_complicated'] + data_cm['cm_cancer']) +
            3.0 * data_cm['cm_liver'] +
            6.0 * (data_cm['cm_metastatic'] + data_cm['cm_hiv'])
        )
    train['weighted_cci'] = calc_weighted_cci(train_cm)
    eval_cols['weighted_cci'] = calc_weighted_cci(eval_cm)
    extra_numeric.append('weighted_cci')
        
    # 2. Grouped ICD-9 chapters and canonical 3-digit ICD-9 disease abstraction
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    def icd9_3digit(code):
        if code is None:
            return "missing"
        c = str(code).strip()
        if not c or c == "?":
            return "missing"
        return c.split('.')[0]

    for col in ['diag_1', 'diag_2']:
        c_3digit = col + "_3digit"
        train[c_3digit] = np.array([icd9_3digit(x) for x in train[col]], dtype=object)
        eval_cols[c_3digit] = np.array([icd9_3digit(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(c_3digit)

    # Multi-way interaction pairs covering the 3-mode (Admission x Discharge x Diagnosis) tensor
    train['disch_adm_pair'] = np.array([f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    eval_cols['disch_adm_pair'] = np.array([f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('disch_adm_pair')
    
    train['diag1_disch_pair'] = np.array([f"{dg}_{d}" for dg, d in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch_pair'] = np.array([f"{dg}_{d}" for dg, d in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch_pair')

    train['adm_diag1_pair'] = np.array([f"{a}_{dg}" for a, dg in zip(train['admission_source_id'], train['diag_1_grp'])], dtype=object)
    eval_cols['adm_diag1_pair'] = np.array([f"{a}_{dg}" for a, dg in zip(eval_cols['admission_source_id'], eval_cols['diag_1_grp'])], dtype=object)
    extra_categorical.append('adm_diag1_pair')

    train['diag1_diag2_grp'] = np.array([f"{dg1}_{dg2}" for dg1, dg2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag1_diag2_grp'] = np.array([f"{dg1}_{dg2}" for dg1, dg2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag1_diag2_grp')
        
    # 3. Target encoding of high-cardinality & interactive features
    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_3digit', 'diag_2_3digit',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id',
        'payer_code', 'disch_adm_pair', 'diag1_disch_pair', 'adm_diag1_pair', 'diag1_diag2_grp'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 4. Utilisation intensity features and indicators
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')

    for prefix, data in [('train', train), ('eval', eval_cols)]:
        h_days = data['time_in_hospital'] + 1.0
        t_vis = data['total_visits'] + 1.0
        data['has_outpatient'] = (data['number_outpatient'] > 0).astype(np.float64)
        data['has_emergency'] = (data['number_emergency'] > 0).astype(np.float64)
        data['has_inpatient'] = (data['number_inpatient'] > 0).astype(np.float64)
        data['inpatient_gt1'] = (data['number_inpatient'] > 1).astype(np.float64)
        data['emergency_ratio'] = data['number_emergency'] / t_vis
        data['inpatient_per_day'] = data['number_inpatient'] / h_days
    extra_numeric.extend(['has_outpatient', 'has_emergency', 'has_inpatient', 'inpatient_gt1', 'emergency_ratio', 'inpatient_per_day'])
    
    # 5. Medication active, change counts, and mechanism-of-action drug classes
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    train_active_meds = np.zeros(n_train, dtype=np.float64)
    eval_active_meds = np.zeros(n_eval, dtype=np.float64)
    train_change_meds = np.zeros(n_train, dtype=np.float64)
    eval_change_meds = np.zeros(n_eval, dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')

    # Mechanism-of-action classes & dual-therapy indicators
    sulfonylureas = ['glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
    tzds = ['pioglitazone', 'rosiglitazone', 'troglitazone']
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        su_active = np.zeros(len(data['insulin']), dtype=bool)
        for su in sulfonylureas:
            if su in data:
                su_active |= (data[su] != 'No') & (data[su] != None)
        tzd_active = np.zeros(len(data['insulin']), dtype=bool)
        for tzd in tzds:
            if tzd in data:
                tzd_active |= (data[tzd] != 'No') & (data[tzd] != None)
        met_active = (data['metformin'] != 'No') & (data['metformin'] != None)
        ins_active = (data['insulin'] != 'No') & (data['insulin'] != None)
        
        data['has_sulfonylurea'] = su_active.astype(np.float64)
        data['has_tzd'] = tzd_active.astype(np.float64)
        data['has_metformin'] = met_active.astype(np.float64)
        data['has_insulin'] = ins_active.astype(np.float64)
        data['num_drug_classes'] = (su_active.astype(int) + tzd_active.astype(int) + met_active.astype(int) + ins_active.astype(int)).astype(np.float64)
        data['insulin_oral_combo'] = (ins_active & (su_active | met_active | tzd_active)).astype(np.float64)
    extra_numeric.extend(['has_sulfonylurea', 'has_tzd', 'has_metformin', 'has_insulin', 'num_drug_classes', 'insulin_oral_combo'])
    
    # 6. Insulin change interaction and dosage adjustments
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['insulin_up'] = (data['insulin'] == 'Up').astype(np.float64)
        data['insulin_down'] = (data['insulin'] == 'Down').astype(np.float64)
        data['insulin_steady'] = (data['insulin'] == 'Steady').astype(np.float64)
    extra_numeric.extend(['insulin_up', 'insulin_down', 'insulin_steady'])

    # 7. Admission and Discharge specific risk dispositions
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['discharged_home'] = (data['discharge_disposition_id'] == '1').astype(np.float64)
        data['discharged_snf'] = (data['discharge_disposition_id'] == '3').astype(np.float64)
        data['discharged_homehealth'] = (data['discharge_disposition_id'] == '6').astype(np.float64)
        data['is_emergency_adm'] = (data['admission_type_id'] == '1').astype(np.float64)
        data['is_elective_adm'] = (data['admission_type_id'] == '3').astype(np.float64)
        data['adm_from_er'] = (data['admission_source_id'] == '7').astype(np.float64)
        data['adm_from_transfer'] = np.isin(data['admission_source_id'], ['4', '5']).astype(np.float64)
    extra_numeric.extend([
        'discharged_home', 'discharged_snf', 'discharged_homehealth',
        'is_emergency_adm', 'is_elective_adm', 'adm_from_er', 'adm_from_transfer'
    ])
    
    # 8. Numeric age representation and clinical stay features
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = str(age_str).replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    for prefix, data in [('train', train), ('eval', eval_cols)]:
        h_days = data['time_in_hospital'] + 1.0
        data['meds_per_hospital_day'] = data['num_medications'] / h_days
        data['labs_per_hospital_day'] = data['num_lab_procedures'] / h_days
        data['procedures_per_hospital_day'] = data['num_procedures'] / h_days
    extra_numeric.extend(['meds_per_hospital_day', 'labs_per_hospital_day', 'procedures_per_hospital_day'])

    # Laboratory results flags
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['A1C_elevated'] = np.isin(data['A1Cresult'], ['>7', '>8']).astype(np.float64)
        data['glu_elevated'] = np.isin(data['max_glu_serum'], ['>200', '>300']).astype(np.float64)
        data['A1C_measured'] = (data['A1Cresult'] != 'None').astype(np.float64)
        data['glu_measured'] = (data['max_glu_serum'] != 'None').astype(np.float64)
    extra_numeric.extend(['A1C_elevated', 'glu_elevated', 'A1C_measured', 'glu_measured'])

    # Missingness indicators
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['payer_code_missing'] = (data['payer_code'] == None).astype(np.float64)
        data['medical_specialty_missing'] = (data['medical_specialty'] == None).astype(np.float64)
        data['weight_missing'] = (data['weight'] == None).astype(np.float64)
    extra_numeric.extend(['payer_code_missing', 'medical_specialty_missing', 'weight_missing'])

    # Medical complexity index
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['complexity_index'] = (
            data['num_lab_procedures']
            + data['num_procedures'] * 5.0
            + data['num_medications'] * 2.0
            + data['number_diagnoses'] * 3.0
        )
    extra_numeric.append('complexity_index')

    # Chapter distribution counts across all 3 diagnoses
    strack_chapters = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other']
    for ch in strack_chapters:
        ch_name = f"ch_cnt_{ch}"
        train[ch_name] = sum((train[col + '_grp'] == ch).astype(np.float64) for col in diag_cols)
        eval_cols[ch_name] = sum((eval_cols[col + '_grp'] == ch).astype(np.float64) for col in diag_cols)
        extra_numeric.append(ch_name)

    # 9. Low-Rank Tensor/Matrix Decomposition of Clinical Phenotype Manifold
    # Build clean dense clinical incidence matrix for Truncated SVD
    cm_keys = list(train_cm.keys())
    svd_base_cols = cm_keys + ['weighted_cci'] + [f"ch_cnt_{ch}" for ch in strack_chapters] + [
        'has_sulfonylurea', 'has_tzd', 'has_metformin', 'has_insulin', 'num_drug_classes', 'insulin_oral_combo',
        'discharged_home', 'discharged_snf', 'discharged_homehealth', 'is_emergency_adm', 'adm_from_er',
        'A1C_elevated', 'glu_elevated', 'A1C_measured', 'glu_measured'
    ]
    
    m_train_list = [np.nan_to_num(train[c], nan=0.0).reshape(-1, 1) for c in svd_base_cols]
    m_train_list.extend([
        np.nan_to_num(np.log1p(train['number_inpatient']), nan=0.0).reshape(-1, 1),
        np.nan_to_num(np.log1p(train['number_emergency']), nan=0.0).reshape(-1, 1),
        np.nan_to_num(np.log1p(train['number_outpatient']), nan=0.0).reshape(-1, 1),
        np.nan_to_num(np.log1p(train['time_in_hospital']), nan=0.0).reshape(-1, 1),
    ])
    m_train = np.hstack(m_train_list)

    m_eval_list = [np.nan_to_num(eval_cols[c], nan=0.0).reshape(-1, 1) for c in svd_base_cols]
    m_eval_list.extend([
        np.nan_to_num(np.log1p(eval_cols['number_inpatient']), nan=0.0).reshape(-1, 1),
        np.nan_to_num(np.log1p(eval_cols['number_emergency']), nan=0.0).reshape(-1, 1),
        np.nan_to_num(np.log1p(eval_cols['number_outpatient']), nan=0.0).reshape(-1, 1),
        np.nan_to_num(np.log1p(eval_cols['time_in_hospital']), nan=0.0).reshape(-1, 1),
    ])
    m_eval = np.hstack(m_eval_list)

    n_svd_components = 6
    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
    z_train = svd.fit_transform(m_train)
    z_eval = svd.transform(m_eval)

    for i in range(n_svd_components):
        k = f"svd_phenotype_{i}"
        train[k] = z_train[:, i]
        eval_cols[k] = z_eval[:, i]
        extra_numeric.append(k)

    # Low-rank reconstruction residual measuring clinical atypicality
    m_train_rec = z_train @ svd.components_
    m_eval_rec = z_eval @ svd.components_
    train['svd_recon_err'] = np.sum((m_train - m_train_rec) ** 2, axis=1)
    eval_cols['svd_recon_err'] = np.sum((m_eval - m_eval_rec) ** 2, axis=1)
    extra_numeric.append('svd_recon_err')

    # 10. Encode datasets
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 11. Fit Stratified 5-Fold Diverse LGBM Ensemble with Log-Odds (Logit) Blending
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(n_eval)
    eps = 1e-7
    
    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        # Model A: Balanced, Well-Converged Model
        model_a = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=130,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.5,
            reg_alpha=0.05,
            cat_smooth=45.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        model_a.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        pred_a = np.clip(model_a.predict_proba(x_eval)[:, 1], eps, 1.0 - eps)
        
        # Model B: Shallow, Regularized Model (Low-variance anchor)
        model_b = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.018,
            num_leaves=18,
            min_child_samples=90,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=2.0,
            cat_smooth=30.0,
            random_state=100,
            n_jobs=1,
            verbose=-1,
        )
        model_b.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        pred_b = np.clip(model_b.predict_proba(x_eval)[:, 1], eps, 1.0 - eps)
        
        # Model C: Deeper, High-Capacity Model with strong L1/L2 shrinkage
        model_c = lgb.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.016,
            num_leaves=55,
            min_child_samples=180,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=6.0,
            reg_alpha=0.2,
            cat_smooth=60.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        model_c.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        pred_c = np.clip(model_c.predict_proba(x_eval)[:, 1], eps, 1.0 - eps)
        
        # Logit (log-odds) transformation for principled Bayesian probability fusion
        logit_a = np.log(pred_a / (1.0 - pred_a))
        logit_b = np.log(pred_b / (1.0 - pred_b))
        logit_c = np.log(pred_c / (1.0 - pred_c))
        
        fold_logit = 0.40 * logit_a + 0.35 * logit_b + 0.25 * logit_c
        fold_pred = 1.0 / (1.0 + np.exp(-fold_logit))
        eval_preds += fold_pred / 5.0
        
    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
