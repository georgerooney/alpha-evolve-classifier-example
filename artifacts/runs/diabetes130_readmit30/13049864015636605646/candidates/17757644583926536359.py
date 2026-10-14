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
    # Shallow copy dicts so caller structures are not mutated
    train = dict(train)
    eval_cols = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities extraction via clinical mapping
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for cm_key in cm_train:
        train[cm_key] = cm_train[cm_key].astype(np.float64)
        eval_cols[cm_key] = cm_eval[cm_key].astype(np.float64)
        extra_numeric.append(cm_key)

    # Key cardiorenal syndrome and multi-organ comorbidity interactions
    train['cm_cardiorenal'] = (train['cm_chf'] * train['cm_renal']).astype(np.float64)
    eval_cols['cm_cardiorenal'] = (eval_cols['cm_chf'] * eval_cols['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_cardiorenal')

    train['cm_cardiopulmonary'] = (train['cm_chf'] * train['cm_pulmonary']).astype(np.float64)
    eval_cols['cm_cardiopulmonary'] = (eval_cols['cm_chf'] * eval_cols['cm_pulmonary']).astype(np.float64)
    extra_numeric.append('cm_cardiopulmonary')

    train['cm_renal_diabetes'] = (train['cm_renal'] * train['cm_diabetes_complicated']).astype(np.float64)
    eval_cols['cm_renal_diabetes'] = (eval_cols['cm_renal'] * eval_cols['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_renal_diabetes')

    # Cardiorenal-metabolic high risk clinical triad
    train['cm_cardiorenal_diab'] = (train['cm_chf'] * train['cm_renal'] * train['cm_diabetes_complicated']).astype(np.float64)
    eval_cols['cm_cardiorenal_diab'] = (eval_cols['cm_chf'] * eval_cols['cm_renal'] * eval_cols['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_cardiorenal_diab')

    train['cm_vascular'] = (train['cm_pvd'] + train['cm_cerebrovascular'] + train['cm_mi']).astype(np.float64)
    eval_cols['cm_vascular'] = (eval_cols['cm_pvd'] + eval_cols['cm_cerebrovascular'] + eval_cols['cm_mi']).astype(np.float64)
    extra_numeric.append('cm_vascular')

    train['cm_neurovascular'] = (train['cm_cerebrovascular'] + train['cm_dementia'] + train['cm_hemiplegia']).astype(np.float64)
    eval_cols['cm_neurovascular'] = (eval_cols['cm_cerebrovascular'] + eval_cols['cm_dementia'] + eval_cols['cm_hemiplegia']).astype(np.float64)
    extra_numeric.append('cm_neurovascular')

    train['cm_hepatorenal'] = (train['cm_liver'] * train['cm_renal']).astype(np.float64)
    eval_cols['cm_hepatorenal'] = (eval_cols['cm_liver'] * eval_cols['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_hepatorenal')

    # Weighted Charlson Comorbidity Prognostic Index
    train['cm_charlson_weighted'] = (
        train['cm_chf'] * 2.0
        + train['cm_renal'] * 2.0
        + train['cm_pulmonary'] * 1.5
        + train['cm_diabetes_complicated'] * 2.0
        + train['cm_cancer'] * 2.0
        + train['cm_metastatic'] * 4.0
        + train['cm_hiv'] * 4.0
        + train['cm_mi']
        + train['cm_cerebrovascular']
        + train['cm_pvd']
        + train['cm_dementia'] * 1.5
        + train['cm_liver'] * 2.0
        + train['cm_hemiplegia'] * 2.0
        + train['cm_peptic_ulcer']
        + train['cm_rheumatic']
    ).astype(np.float64)
    eval_cols['cm_charlson_weighted'] = (
        eval_cols['cm_chf'] * 2.0
        + eval_cols['cm_renal'] * 2.0
        + eval_cols['cm_pulmonary'] * 1.5
        + eval_cols['cm_diabetes_complicated'] * 2.0
        + eval_cols['cm_cancer'] * 2.0
        + eval_cols['cm_metastatic'] * 4.0
        + eval_cols['cm_hiv'] * 4.0
        + eval_cols['cm_mi']
        + eval_cols['cm_cerebrovascular']
        + eval_cols['cm_pvd']
        + eval_cols['cm_dementia'] * 1.5
        + eval_cols['cm_liver'] * 2.0
        + eval_cols['cm_hemiplegia'] * 2.0
        + eval_cols['cm_peptic_ulcer']
        + eval_cols['cm_rheumatic']
    ).astype(np.float64)
    extra_numeric.append('cm_charlson_weighted')

    train['cm_high_risk'] = (train['cm_chf'] + train['cm_renal'] + train['cm_pulmonary']).astype(np.float64)
    eval_cols['cm_high_risk'] = (eval_cols['cm_chf'] + eval_cols['cm_renal'] + eval_cols['cm_pulmonary']).astype(np.float64)
    extra_numeric.append('cm_high_risk')

    # 2. ICD-9 Chapter Grouping
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f'{diag_col}_group'
        train[grp_name] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_cols[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(grp_name)

    for grp in ['circulatory', 'diabetes', 'respiratory', 'genitourinary', 'digestive', 'neoplasms', 'injury', 'musculoskeletal']:
        flag_name = f'has_diag_grp_{grp}'
        train[flag_name] = (
            (train['diag_1_group'] == grp)
            | (train['diag_2_group'] == grp)
            | (train['diag_3_group'] == grp)
        ).astype(np.float64)
        eval_cols[flag_name] = (
            (eval_cols['diag_1_group'] == grp)
            | (eval_cols['diag_2_group'] == grp)
            | (eval_cols['diag_3_group'] == grp)
        ).astype(np.float64)
        extra_numeric.append(flag_name)

    # Cross-system multimorbidity: count of distinct diagnostic organ chapters
    train['distinct_diag_groups'] = np.array(
        [len({a, b, c} - {'missing', 'None', None}) for a, b, c in zip(train['diag_1_group'], train['diag_2_group'], train['diag_3_group'])],
        dtype=np.float64,
    )
    eval_cols['distinct_diag_groups'] = np.array(
        [len({a, b, c} - {'missing', 'None', None}) for a, b, c in zip(eval_cols['diag_1_group'], eval_cols['diag_2_group'], eval_cols['diag_3_group'])],
        dtype=np.float64,
    )
    extra_numeric.append('distinct_diag_groups')

    # Numeric age midpoint & informative documentation missingness flags
    age_map = {
        '[0-10)': 5.0,
        '[10-20)': 15.0,
        '[20-30)': 25.0,
        '[30-40)': 35.0,
        '[40-50)': 45.0,
        '[50-60)': 55.0,
        '[60-70)': 65.0,
        '[70-80)': 75.0,
        '[80-90)': 85.0,
        '[90-100)': 95.0,
    }
    train['age_num'] = np.array([age_map.get(x, 65.0) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_map.get(x, 65.0) for x in eval_cols['age']], dtype=np.float64)
    train['is_elderly'] = (train['age_num'] >= 65.0).astype(np.float64)
    eval_cols['is_elderly'] = (eval_cols['age_num'] >= 65.0).astype(np.float64)
    train['weight_known'] = np.array([0.0 if x is None else 1.0 for x in train['weight']], dtype=np.float64)
    eval_cols['weight_known'] = np.array([0.0 if x is None else 1.0 for x in eval_cols['weight']], dtype=np.float64)
    train['specialty_known'] = np.array([0.0 if x is None else 1.0 for x in train['medical_specialty']], dtype=np.float64)
    eval_cols['specialty_known'] = np.array([0.0 if x is None else 1.0 for x in eval_cols['medical_specialty']], dtype=np.float64)
    train['payer_known'] = np.array([0.0 if x is None else 1.0 for x in train['payer_code']], dtype=np.float64)
    eval_cols['payer_known'] = np.array([0.0 if x is None else 1.0 for x in eval_cols['payer_code']], dtype=np.float64)
    train['gender_is_female'] = (train['gender'] == 'Female').astype(np.float64)
    eval_cols['gender_is_female'] = (eval_cols['gender'] == 'Female').astype(np.float64)
    train['elderly_female'] = (train['is_elderly'] * train['gender_is_female']).astype(np.float64)
    eval_cols['elderly_female'] = (eval_cols['is_elderly'] * eval_cols['gender_is_female']).astype(np.float64)
    train['severe_multimorbidity'] = ((train['cm_count'] >= 3) & (train['is_elderly'] == 1.0)).astype(np.float64)
    eval_cols['severe_multimorbidity'] = ((eval_cols['cm_count'] >= 3) & (eval_cols['is_elderly'] == 1.0)).astype(np.float64)
    extra_numeric.extend(['age_num', 'is_elderly', 'weight_known', 'specialty_known', 'payer_known', 'gender_is_female', 'elderly_female', 'severe_multimorbidity'])

    # 3. Smoothed Out-Of-Fold Target Encodings of High-Cardinality Categoricals
    te_columns = [
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'medical_specialty',
        'diag_1',
        'diag_2',
        'diag_3',
        'payer_code',
        'diag_1_group',
    ]
    for col in te_columns:
        te_tr, te_ev = oof_target_encode(
            train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
        )
        te_col_name = f'{col}_te'
        train[te_col_name] = te_tr.astype(np.float64)
        eval_cols[te_col_name] = te_ev.astype(np.float64)
        extra_numeric.append(te_col_name)

    # Aggregate diagnosis risk severity
    train['diag_te_mean'] = (train['diag_1_te'] + train['diag_2_te'] + train['diag_3_te']) / 3.0
    eval_cols['diag_te_mean'] = (eval_cols['diag_1_te'] + eval_cols['diag_2_te'] + eval_cols['diag_3_te']) / 3.0
    extra_numeric.append('diag_te_mean')

    train['diag_te_max'] = np.maximum(np.maximum(train['diag_1_te'], train['diag_2_te']), train['diag_3_te'])
    eval_cols['diag_te_max'] = np.maximum(np.maximum(eval_cols['diag_1_te'], eval_cols['diag_2_te']), eval_cols['diag_3_te'])
    extra_numeric.append('diag_te_max')

    # Truncated 3-digit ICD-9 disease family target encodings
    d1_cat_tr = np.array([str(x).split('.')[0] if (x is not None and x != '?') else 'None' for x in train['diag_1']], dtype=object)
    d1_cat_ev = np.array([str(x).split('.')[0] if (x is not None and x != '?') else 'None' for x in eval_cols['diag_1']], dtype=object)
    d2_cat_tr = np.array([str(x).split('.')[0] if (x is not None and x != '?') else 'None' for x in train['diag_2']], dtype=object)
    d2_cat_ev = np.array([str(x).split('.')[0] if (x is not None and x != '?') else 'None' for x in eval_cols['diag_2']], dtype=object)
    d3_cat_tr = np.array([str(x).split('.')[0] if (x is not None and x != '?') else 'None' for x in train['diag_3']], dtype=object)
    d3_cat_ev = np.array([str(x).split('.')[0] if (x is not None and x != '?') else 'None' for x in eval_cols['diag_3']], dtype=object)

    te_d1c_tr, te_d1c_ev = oof_target_encode(d1_cat_tr, y_train, d1_cat_ev, n_folds=5, smoothing=20.0, seed=42)
    te_d2c_tr, te_d2c_ev = oof_target_encode(d2_cat_tr, y_train, d2_cat_ev, n_folds=5, smoothing=20.0, seed=42)
    te_d3c_tr, te_d3c_ev = oof_target_encode(d3_cat_tr, y_train, d3_cat_ev, n_folds=5, smoothing=20.0, seed=42)

    train['diag_1_trunc_te'] = te_d1c_tr.astype(np.float64)
    eval_cols['diag_1_trunc_te'] = te_d1c_ev.astype(np.float64)
    train['diag_2_trunc_te'] = te_d2c_tr.astype(np.float64)
    eval_cols['diag_2_trunc_te'] = te_d2c_ev.astype(np.float64)
    train['diag_3_trunc_te'] = te_d3c_tr.astype(np.float64)
    eval_cols['diag_3_trunc_te'] = te_d3c_ev.astype(np.float64)
    train['diag_trunc_te_mean'] = (te_d1c_tr + te_d2c_tr + te_d3c_tr) / 3.0
    eval_cols['diag_trunc_te_mean'] = (te_d1c_ev + te_d2c_ev + te_d3c_ev) / 3.0
    train['diag_trunc_te_max'] = np.maximum(np.maximum(te_d1c_tr, te_d2c_tr), te_d3c_tr)
    eval_cols['diag_trunc_te_max'] = np.maximum(np.maximum(te_d1c_ev, te_d2c_ev), te_d3c_ev)
    extra_numeric.extend(['diag_1_trunc_te', 'diag_2_trunc_te', 'diag_3_trunc_te', 'diag_trunc_te_mean', 'diag_trunc_te_max'])

    # Specific primary admitting diagnoses flags
    train['is_primary_chf'] = np.array([1.0 if (x is not None and str(x).startswith('428')) else 0.0 for x in train['diag_1']], dtype=np.float64)
    eval_cols['is_primary_chf'] = np.array([1.0 if (x is not None and str(x).startswith('428')) else 0.0 for x in eval_cols['diag_1']], dtype=np.float64)
    train['is_primary_copd_pna'] = np.array([1.0 if (x is not None and str(x)[:3] in ('486', '491', '492', '493', '494', '496')) else 0.0 for x in train['diag_1']], dtype=np.float64)
    eval_cols['is_primary_copd_pna'] = np.array([1.0 if (x is not None and str(x)[:3] in ('486', '491', '492', '493', '494', '496')) else 0.0 for x in eval_cols['diag_1']], dtype=np.float64)
    train['is_primary_diabetes'] = np.array([1.0 if (x is not None and str(x).startswith('250')) else 0.0 for x in train['diag_1']], dtype=np.float64)
    eval_cols['is_primary_diabetes'] = np.array([1.0 if (x is not None and str(x).startswith('250')) else 0.0 for x in eval_cols['diag_1']], dtype=np.float64)
    train['is_primary_renal'] = np.array([1.0 if (x is not None and str(x)[:3] in ('584', '585')) else 0.0 for x in train['diag_1']], dtype=np.float64)
    eval_cols['is_primary_renal'] = np.array([1.0 if (x is not None and str(x)[:3] in ('584', '585')) else 0.0 for x in eval_cols['diag_1']], dtype=np.float64)
    train['is_primary_sepsis'] = np.array([1.0 if (x is not None and (str(x).startswith('038') or str(x).startswith('995.9'))) else 0.0 for x in train['diag_1']], dtype=np.float64)
    eval_cols['is_primary_sepsis'] = np.array([1.0 if (x is not None and (str(x).startswith('038') or str(x).startswith('995.9'))) else 0.0 for x in eval_cols['diag_1']], dtype=np.float64)
    train['is_primary_dka'] = np.array([1.0 if (x is not None and (str(x).startswith('250.1') or str(x).startswith('250.2'))) else 0.0 for x in train['diag_1']], dtype=np.float64)
    eval_cols['is_primary_dka'] = np.array([1.0 if (x is not None and (str(x).startswith('250.1') or str(x).startswith('250.2'))) else 0.0 for x in eval_cols['diag_1']], dtype=np.float64)
    train['is_primary_circulatory'] = (train['diag_1_group'] == 'circulatory').astype(np.float64)
    eval_cols['is_primary_circulatory'] = (eval_cols['diag_1_group'] == 'circulatory').astype(np.float64)
    extra_numeric.extend(['is_primary_chf', 'is_primary_copd_pna', 'is_primary_diabetes', 'is_primary_renal', 'is_primary_sepsis', 'is_primary_dka', 'is_primary_circulatory'])

    # Interaction target encodings: clinical pathway and discharge disposition
    diag1_disc_tr = np.array([f'{g}_{d}' for g, d in zip(train['diag_1_group'], train['discharge_disposition_id'])], dtype=object)
    diag1_disc_ev = np.array([f'{g}_{d}' for g, d in zip(eval_cols['diag_1_group'], eval_cols['discharge_disposition_id'])], dtype=object)
    te_d1d_tr, te_d1d_ev = oof_target_encode(diag1_disc_tr, y_train, diag1_disc_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_diag1_disc'] = te_d1d_tr.astype(np.float64)
    eval_cols['te_diag1_disc'] = te_d1d_ev.astype(np.float64)
    extra_numeric.append('te_diag1_disc')

    adm_disc_tr = np.array([f'{a}_{d}' for a, d in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    adm_disc_ev = np.array([f'{a}_{d}' for a, d in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    te_ad_tr, te_ad_ev = oof_target_encode(adm_disc_tr, y_train, adm_disc_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_adm_disc'] = te_ad_tr.astype(np.float64)
    eval_cols['te_adm_disc'] = te_ad_ev.astype(np.float64)
    extra_numeric.append('te_adm_disc')

    diag1_2_tr = np.array([f'{a}_{b}' for a, b in zip(train['diag_1_group'], train['diag_2_group'])], dtype=object)
    diag1_2_ev = np.array([f'{a}_{b}' for a, b in zip(eval_cols['diag_1_group'], eval_cols['diag_2_group'])], dtype=object)
    te_d12_tr, te_d12_ev = oof_target_encode(diag1_2_tr, y_train, diag1_2_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_diag1_2'] = te_d12_tr.astype(np.float64)
    eval_cols['te_diag1_2'] = te_d12_ev.astype(np.float64)
    extra_numeric.append('te_diag1_2')

    # Age and clinical transition target encodings
    age_disc_tr = np.array([f'{a}_{d}' for a, d in zip(train['age'], train['discharge_disposition_id'])], dtype=object)
    age_disc_ev = np.array([f'{a}_{d}' for a, d in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])], dtype=object)
    te_aged_tr, te_aged_ev = oof_target_encode(age_disc_tr, y_train, age_disc_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_age_disc'] = te_aged_tr.astype(np.float64)
    eval_cols['te_age_disc'] = te_aged_ev.astype(np.float64)
    extra_numeric.append('te_age_disc')

    age_diag1_tr = np.array([f'{a}_{g}' for a, g in zip(train['age'], train['diag_1_group'])], dtype=object)
    age_diag1_ev = np.array([f'{a}_{g}' for a, g in zip(eval_cols['age'], eval_cols['diag_1_group'])], dtype=object)
    te_ad1_tr, te_ad1_ev = oof_target_encode(age_diag1_tr, y_train, age_diag1_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_age_diag1'] = te_ad1_tr.astype(np.float64)
    eval_cols['te_age_diag1'] = te_ad1_ev.astype(np.float64)
    extra_numeric.append('te_age_diag1')

    spec_diag1_tr = np.array([f'{s}_{g}' for s, g in zip(train['medical_specialty'], train['diag_1_group'])], dtype=object)
    spec_diag1_ev = np.array([f'{s}_{g}' for s, g in zip(eval_cols['medical_specialty'], eval_cols['diag_1_group'])], dtype=object)
    te_sd1_tr, te_sd1_ev = oof_target_encode(spec_diag1_tr, y_train, spec_diag1_ev, n_folds=5, smoothing=20.0, seed=42)
    train['te_spec_diag1'] = te_sd1_tr.astype(np.float64)
    eval_cols['te_spec_diag1'] = te_sd1_ev.astype(np.float64)
    extra_numeric.append('te_spec_diag1')

    # 4. Low-Rank Tensor Factor Decomposition on Clinical Condition & Medication Space
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]

    # Find frequent ICD-9 diagnosis codes across all 3 diagnosis positions
    diag_counts = {}
    for col in ['diag_1', 'diag_2', 'diag_3']:
        for val in train[col]:
            if val is not None and val != '?':
                diag_counts[val] = diag_counts.get(val, 0) + 1
    sorted_diags = sorted(diag_counts.items(), key=lambda x: x[1], reverse=True)
    top_diags = [code for code, count in sorted_diags[:50] if count >= 20]

    tensor_slice_tr = []
    tensor_slice_ev = []
    for code in top_diags:
        tensor_slice_tr.append(
            ((train['diag_1'] == code) | (train['diag_2'] == code) | (train['diag_3'] == code)).astype(np.float64)
        )
        tensor_slice_ev.append(
            ((eval_cols['diag_1'] == code) | (eval_cols['diag_2'] == code) | (eval_cols['diag_3'] == code)).astype(np.float64)
        )

    for med in med_cols:
        tensor_slice_tr.append(((train[med] != 'No') & (train[med] is not None)).astype(np.float64))
        tensor_slice_ev.append(((eval_cols[med] != 'No') & (eval_cols[med] is not None)).astype(np.float64))

    for cm_key in ['cm_mi', 'cm_chf', 'cm_pvd', 'cm_pulmonary', 'cm_renal', 'cm_diabetes_complicated']:
        tensor_slice_tr.append(train[cm_key])
        tensor_slice_ev.append(eval_cols[cm_key])

    # Core discharge and acute care transition indicators for latent embedding
    for d_code in ['1', '3', '6', '2', '7', '18', '22', '24']:
        tensor_slice_tr.append((train['discharge_disposition_id'] == d_code).astype(np.float64))
        tensor_slice_ev.append((eval_cols['discharge_disposition_id'] == d_code).astype(np.float64))

    for s_code in ['7', '1', '2', '4', '6']:
        tensor_slice_tr.append((train['admission_source_id'] == s_code).astype(np.float64))
        tensor_slice_ev.append((eval_cols['admission_source_id'] == s_code).astype(np.float64))

    tensor_slice_tr.append((train['change'] == 'Ch').astype(np.float64))
    tensor_slice_ev.append((eval_cols['change'] == 'Ch').astype(np.float64))
    tensor_slice_tr.append((train['A1Cresult'] == '>8').astype(np.float64))
    tensor_slice_ev.append((eval_cols['A1Cresult'] == '>8').astype(np.float64))

    M_tr = np.column_stack(tensor_slice_tr)
    M_ev = np.column_stack(tensor_slice_ev)
    M_mean = np.mean(M_tr, axis=0, keepdims=True)
    M_tr_c = M_tr - M_mean
    M_ev_c = M_ev - M_mean

    # Low-rank factor projection via sample covariance eigendecomposition
    cov_mat = (M_tr_c.T @ M_tr_c) / max(len(M_tr) - 1.0, 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx_sorted = np.argsort(eigvals)[::-1]
    n_latent = min(8, M_tr.shape[1])
    V_factors = eigvecs[:, idx_sorted[:n_latent]]

    latents_tr = M_tr_c @ V_factors
    latents_ev = M_ev_c @ V_factors
    for j in range(n_latent):
        latent_name = f'latent_phenotype_{j}'
        train[latent_name] = latents_tr[:, j].astype(np.float64)
        eval_cols[latent_name] = latents_ev[:, j].astype(np.float64)
        extra_numeric.append(latent_name)

    # Supervised Partial Least Squares (PLS) latent factors for readmission risk
    def _run_pls(x_tr, y_tr, x_te, n_comp=4):
        mu_x = np.mean(x_tr, axis=0, keepdims=True)
        mu_y = np.mean(y_tr)
        xc_tr = x_tr - mu_x
        yc_tr = y_tr - mu_y
        xc_te = x_te - mu_x

        f_tr = np.zeros((x_tr.shape[0], n_comp), dtype=np.float64)
        f_te = np.zeros((x_te.shape[0], n_comp), dtype=np.float64)

        for c in range(n_comp):
            w = xc_tr.T @ yc_tr
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                break
            w /= nw
            t_tr = xc_tr @ w
            t_te = xc_te @ w
            f_tr[:, c] = t_tr
            f_te[:, c] = t_te
            denom = np.dot(t_tr, t_tr)
            if denom < 1e-12:
                break
            p = (xc_tr.T @ t_tr) / denom
            xc_tr = xc_tr - np.outer(t_tr, p)
            xc_te = xc_te - np.outer(t_te, p)
            yc_tr = yc_tr - (np.dot(t_tr, yc_tr) / denom) * t_tr
        return f_tr, f_te

    n_samples_tr = len(y_train)
    pls_tr = np.zeros((n_samples_tr, 4), dtype=np.float64)
    rng_pls = np.random.RandomState(42)
    fold_indices = np.array_split(rng_pls.permutation(n_samples_tr), 5)
    for fold_idx in fold_indices:
        mask_tr = np.ones(n_samples_tr, dtype=bool)
        mask_tr[fold_idx] = False
        _, f_val = _run_pls(M_tr[mask_tr], y_train[mask_tr], M_tr[fold_idx], n_comp=4)
        pls_tr[fold_idx] = f_val

    _, pls_ev = _run_pls(M_tr, y_train, M_ev, n_comp=4)

    for j in range(4):
        pls_name = f'pls_risk_factor_{j}'
        train[pls_name] = pls_tr[:, j]
        eval_cols[pls_name] = pls_ev[:, j]
        extra_numeric.append(pls_name)

    # 5. Prior Healthcare Utilization Intensity Signals
    inp_tr = np.nan_to_num(train['number_inpatient'], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols['number_inpatient'], nan=0.0)
    emg_tr = np.nan_to_num(train['number_emergency'], nan=0.0)
    emg_ev = np.nan_to_num(eval_cols['number_emergency'], nan=0.0)
    otp_tr = np.nan_to_num(train['number_outpatient'], nan=0.0)
    otp_ev = np.nan_to_num(eval_cols['number_outpatient'], nan=0.0)

    tot_vis_tr = inp_tr + emg_tr + otp_tr
    tot_vis_ev = inp_ev + emg_ev + otp_ev

    train['has_inpatient'] = (inp_tr > 0).astype(np.float64)
    eval_cols['has_inpatient'] = (inp_ev > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train['has_emergency'] = (emg_tr > 0).astype(np.float64)
    eval_cols['has_emergency'] = (emg_ev > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    train['total_prior_visits'] = tot_vis_tr.astype(np.float64)
    eval_cols['total_prior_visits'] = tot_vis_ev.astype(np.float64)
    extra_numeric.append('total_prior_visits')

    train['inpatient_ratio'] = (inp_tr / (tot_vis_tr + 1.0)).astype(np.float64)
    eval_cols['inpatient_ratio'] = (inp_ev / (tot_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('inpatient_ratio')

    train['log_inpatient'] = np.log1p(np.maximum(inp_tr, 0.0)).astype(np.float64)
    eval_cols['log_inpatient'] = np.log1p(np.maximum(inp_ev, 0.0)).astype(np.float64)
    extra_numeric.append('log_inpatient')

    train['log_total_visits'] = np.log1p(np.maximum(tot_vis_tr, 0.0)).astype(np.float64)
    eval_cols['log_total_visits'] = np.log1p(np.maximum(tot_vis_ev, 0.0)).astype(np.float64)
    extra_numeric.append('log_total_visits')

    # Acute visit burden and frequent utilizer indicators
    train['acute_visits'] = (inp_tr + emg_tr).astype(np.float64)
    eval_cols['acute_visits'] = (inp_ev + emg_ev).astype(np.float64)
    train['acute_visit_ratio'] = ((inp_tr + emg_tr) / (tot_vis_tr + 1.0)).astype(np.float64)
    eval_cols['acute_visit_ratio'] = ((inp_ev + emg_ev) / (tot_vis_ev + 1.0)).astype(np.float64)
    train['frequent_inpatient'] = (inp_tr >= 2).astype(np.float64)
    eval_cols['frequent_inpatient'] = (inp_ev >= 2).astype(np.float64)
    train['prior_utilizer_score'] = (inp_tr * 3.0 + emg_tr * 2.0 + otp_tr * 0.5).astype(np.float64)
    eval_cols['prior_utilizer_score'] = (inp_ev * 3.0 + emg_ev * 2.0 + otp_ev * 0.5).astype(np.float64)
    train['high_utilizer'] = (tot_vis_tr >= 4).astype(np.float64)
    eval_cols['high_utilizer'] = (tot_vis_ev >= 4).astype(np.float64)
    train['has_outpatient'] = (otp_tr > 0).astype(np.float64)
    eval_cols['has_outpatient'] = (otp_ev > 0).astype(np.float64)
    train['emergency_ratio'] = (emg_tr / (tot_vis_tr + 1.0)).astype(np.float64)
    eval_cols['emergency_ratio'] = (emg_ev / (tot_vis_ev + 1.0)).astype(np.float64)
    train['outpatient_ratio'] = (otp_tr / (tot_vis_tr + 1.0)).astype(np.float64)
    eval_cols['outpatient_ratio'] = (otp_ev / (tot_vis_ev + 1.0)).astype(np.float64)
    train['acute_dominant'] = ((inp_tr + emg_tr) > otp_tr).astype(np.float64)
    eval_cols['acute_dominant'] = ((inp_ev + emg_ev) > otp_ev).astype(np.float64)
    train['is_first_encounter'] = (tot_vis_tr == 0).astype(np.float64)
    eval_cols['is_first_encounter'] = (tot_vis_ev == 0).astype(np.float64)
    train['inp_to_stay_ratio'] = (inp_tr / np.maximum(train['time_in_hospital'], 1.0)).astype(np.float64)
    eval_cols['inp_to_stay_ratio'] = (inp_ev / np.maximum(eval_cols['time_in_hospital'], 1.0)).astype(np.float64)
    extra_numeric.extend([
        'acute_visits', 'acute_visit_ratio', 'frequent_inpatient', 'prior_utilizer_score',
        'high_utilizer', 'has_outpatient', 'emergency_ratio', 'outpatient_ratio',
        'acute_dominant', 'is_first_encounter', 'inp_to_stay_ratio',
    ])

    # 6. Stay and Encounter Intensity Rates
    los_tr = np.maximum(np.nan_to_num(train['time_in_hospital'], nan=1.0), 1.0)
    los_ev = np.maximum(np.nan_to_num(eval_cols['time_in_hospital'], nan=1.0), 1.0)
    labs_tr = np.nan_to_num(train['num_lab_procedures'], nan=0.0)
    labs_ev = np.nan_to_num(eval_cols['num_lab_procedures'], nan=0.0)
    procs_tr = np.nan_to_num(train['num_procedures'], nan=0.0)
    procs_ev = np.nan_to_num(eval_cols['num_procedures'], nan=0.0)
    meds_tr = np.nan_to_num(train['num_medications'], nan=0.0)
    meds_ev = np.nan_to_num(eval_cols['num_medications'], nan=0.0)
    diags_tr = np.nan_to_num(train['number_diagnoses'], nan=0.0)
    diags_ev = np.nan_to_num(eval_cols['number_diagnoses'], nan=0.0)

    train['lab_per_day'] = (labs_tr / los_tr).astype(np.float64)
    eval_cols['lab_per_day'] = (labs_ev / los_ev).astype(np.float64)
    extra_numeric.append('lab_per_day')

    train['med_per_day'] = (meds_tr / los_tr).astype(np.float64)
    eval_cols['med_per_day'] = (meds_ev / los_ev).astype(np.float64)
    extra_numeric.append('med_per_day')

    train['diag_per_day'] = (diags_tr / los_tr).astype(np.float64)
    eval_cols['diag_per_day'] = (diags_ev / los_ev).astype(np.float64)
    extra_numeric.append('diag_per_day')

    train['proc_per_day'] = (procs_tr / los_tr).astype(np.float64)
    eval_cols['proc_per_day'] = (procs_ev / los_ev).astype(np.float64)
    extra_numeric.append('proc_per_day')

    train['meds_per_lab'] = (meds_tr / np.maximum(labs_tr, 1.0)).astype(np.float64)
    eval_cols['meds_per_lab'] = (meds_ev / np.maximum(labs_ev, 1.0)).astype(np.float64)
    extra_numeric.append('meds_per_lab')

    # Utilisation & clinical intensity composite interactions
    train['inpatient_x_stay'] = (inp_tr * los_tr).astype(np.float64)
    eval_cols['inpatient_x_stay'] = (inp_ev * los_ev).astype(np.float64)
    extra_numeric.append('inpatient_x_stay')

    train['total_procedures'] = (procs_tr + labs_tr).astype(np.float64)
    eval_cols['total_procedures'] = (procs_ev + labs_ev).astype(np.float64)
    extra_numeric.append('total_procedures')

    train['total_clinical_events'] = (procs_tr + labs_tr + meds_tr).astype(np.float64)
    eval_cols['total_clinical_events'] = (procs_ev + labs_ev + meds_ev).astype(np.float64)
    extra_numeric.append('total_clinical_events')

    train['meds_x_diags'] = (meds_tr * diags_tr).astype(np.float64)
    eval_cols['meds_x_diags'] = (meds_ev * diags_ev).astype(np.float64)
    extra_numeric.append('meds_x_diags')

    train['procs_per_diag'] = (procs_tr / np.maximum(diags_tr, 1.0)).astype(np.float64)
    eval_cols['procs_per_diag'] = (procs_ev / np.maximum(diags_ev, 1.0)).astype(np.float64)
    extra_numeric.append('procs_per_diag')

    train['labs_per_diag'] = (labs_tr / np.maximum(diags_tr, 1.0)).astype(np.float64)
    eval_cols['labs_per_diag'] = (labs_ev / np.maximum(diags_ev, 1.0)).astype(np.float64)
    extra_numeric.append('labs_per_diag')

    train['stay_complexity'] = ((meds_tr * diags_tr) / los_tr).astype(np.float64)
    eval_cols['stay_complexity'] = ((meds_ev * diags_ev) / los_ev).astype(np.float64)
    extra_numeric.append('stay_complexity')

    train['is_long_stay'] = (los_tr >= 7).astype(np.float64)
    eval_cols['is_long_stay'] = (los_ev >= 7).astype(np.float64)
    train['is_short_stay'] = (los_tr <= 2).astype(np.float64)
    eval_cols['is_short_stay'] = (los_ev <= 2).astype(np.float64)
    train['high_med_count'] = (meds_tr >= 20).astype(np.float64)
    eval_cols['high_med_count'] = (meds_ev >= 20).astype(np.float64)
    train['high_diag_count'] = (diags_tr >= 9).astype(np.float64)
    eval_cols['high_diag_count'] = (diags_ev >= 9).astype(np.float64)
    extra_numeric.extend(['is_long_stay', 'is_short_stay', 'high_med_count', 'high_diag_count'])

    train['age_x_inpatient'] = (train['age_num'] * train['log_inpatient']).astype(np.float64)
    eval_cols['age_x_inpatient'] = (eval_cols['age_num'] * eval_cols['log_inpatient']).astype(np.float64)
    extra_numeric.append('age_x_inpatient')

    # 7. Medication Modification Dynamics & Insulin Indicators
    n_tr = len(los_tr)
    n_ev = len(los_ev)
    changes_tr = np.zeros(n_tr, dtype=np.float64)
    changes_ev = np.zeros(n_ev, dtype=np.float64)
    med_up_tr = np.zeros(n_tr, dtype=np.float64)
    med_up_ev = np.zeros(n_ev, dtype=np.float64)
    med_down_tr = np.zeros(n_tr, dtype=np.float64)
    med_down_ev = np.zeros(n_ev, dtype=np.float64)
    med_active_tr = np.zeros(n_tr, dtype=np.float64)
    med_active_ev = np.zeros(n_ev, dtype=np.float64)

    for m in med_cols:
        changes_tr += np.isin(train[m], ['Up', 'Down']).astype(np.float64)
        changes_ev += np.isin(eval_cols[m], ['Up', 'Down']).astype(np.float64)
        med_up_tr += (train[m] == 'Up').astype(np.float64)
        med_up_ev += (eval_cols[m] == 'Up').astype(np.float64)
        med_down_tr += (train[m] == 'Down').astype(np.float64)
        med_down_ev += (eval_cols[m] == 'Down').astype(np.float64)
        med_active_tr += ((train[m] != 'No') & (train[m] is not None)).astype(np.float64)
        med_active_ev += ((eval_cols[m] != 'No') & (eval_cols[m] is not None)).astype(np.float64)

    train['num_med_changes'] = changes_tr
    eval_cols['num_med_changes'] = changes_ev
    extra_numeric.append('num_med_changes')

    train['num_med_up'] = med_up_tr
    eval_cols['num_med_up'] = med_up_ev
    extra_numeric.append('num_med_up')

    train['num_med_down'] = med_down_tr
    eval_cols['num_med_down'] = med_down_ev
    extra_numeric.append('num_med_down')

    train['num_active_meds'] = med_active_tr
    eval_cols['num_active_meds'] = med_active_ev
    extra_numeric.append('num_active_meds')

    train['insulin_is_up'] = (train['insulin'] == 'Up').astype(np.float64)
    eval_cols['insulin_is_up'] = (eval_cols['insulin'] == 'Up').astype(np.float64)
    extra_numeric.append('insulin_is_up')

    train['insulin_is_down'] = (train['insulin'] == 'Down').astype(np.float64)
    eval_cols['insulin_is_down'] = (eval_cols['insulin'] == 'Down').astype(np.float64)
    extra_numeric.append('insulin_is_down')

    train['insulin_is_active'] = (train['insulin'] != 'No').astype(np.float64)
    eval_cols['insulin_is_active'] = (eval_cols['insulin'] != 'No').astype(np.float64)
    extra_numeric.append('insulin_is_active')

    train['insulin_is_steady'] = (train['insulin'] == 'Steady').astype(np.float64)
    eval_cols['insulin_is_steady'] = (eval_cols['insulin'] == 'Steady').astype(np.float64)
    extra_numeric.append('insulin_is_steady')

    train['insulin_dose_direction'] = (train['insulin_is_up'] - train['insulin_is_down']).astype(np.float64)
    eval_cols['insulin_dose_direction'] = (eval_cols['insulin_is_up'] - eval_cols['insulin_is_down']).astype(np.float64)
    extra_numeric.append('insulin_dose_direction')

    train['metformin_active'] = ((train['metformin'] != 'No') & (train['metformin'] is not None)).astype(np.float64)
    eval_cols['metformin_active'] = ((eval_cols['metformin'] != 'No') & (eval_cols['metformin'] is not None)).astype(np.float64)
    extra_numeric.append('metformin_active')

    train['has_sulfonylurea'] = (
        (train['glipizide'] != 'No') | (train['glyburide'] != 'No') | (train['glimepiride'] != 'No')
    ).astype(np.float64)
    eval_cols['has_sulfonylurea'] = (
        (eval_cols['glipizide'] != 'No') | (eval_cols['glyburide'] != 'No') | (eval_cols['glimepiride'] != 'No')
    ).astype(np.float64)
    extra_numeric.append('has_sulfonylurea')

    train['med_change_ratio'] = (changes_tr / np.maximum(med_active_tr, 1.0)).astype(np.float64)
    eval_cols['med_change_ratio'] = (changes_ev / np.maximum(med_active_ev, 1.0)).astype(np.float64)
    extra_numeric.append('med_change_ratio')

    # Lab testing indicators
    train['a1c_tested'] = (train['A1Cresult'] != 'None').astype(np.float64)
    eval_cols['a1c_tested'] = (eval_cols['A1Cresult'] != 'None').astype(np.float64)
    extra_numeric.append('a1c_tested')

    train['glu_serum_tested'] = (train['max_glu_serum'] != 'None').astype(np.float64)
    eval_cols['glu_serum_tested'] = (eval_cols['max_glu_serum'] != 'None').astype(np.float64)
    extra_numeric.append('glu_serum_tested')

    train['discharged_home'] = (train['discharge_disposition_id'] == '1').astype(np.float64)
    eval_cols['discharged_home'] = (eval_cols['discharge_disposition_id'] == '1').astype(np.float64)
    extra_numeric.append('discharged_home')

    train['a1c_high'] = np.isin(train['A1Cresult'], ['>7', '>8']).astype(np.float64)
    eval_cols['a1c_high'] = np.isin(eval_cols['A1Cresult'], ['>7', '>8']).astype(np.float64)
    extra_numeric.append('a1c_high')

    train['a1c_normal'] = (train['A1Cresult'] == 'Norm').astype(np.float64)
    eval_cols['a1c_normal'] = (eval_cols['A1Cresult'] == 'Norm').astype(np.float64)
    extra_numeric.append('a1c_normal')

    train['a1c_high_no_change'] = ((train['A1Cresult'] == '>8') & (train['change'] == 'No')).astype(np.float64)
    eval_cols['a1c_high_no_change'] = ((eval_cols['A1Cresult'] == '>8') & (eval_cols['change'] == 'No')).astype(np.float64)
    extra_numeric.append('a1c_high_no_change')

    train['a1c_high_with_change'] = ((train['A1Cresult'] == '>8') & (train['change'] == 'Ch')).astype(np.float64)
    eval_cols['a1c_high_with_change'] = ((eval_cols['A1Cresult'] == '>8') & (eval_cols['change'] == 'Ch')).astype(np.float64)
    extra_numeric.append('a1c_high_with_change')

    train['a1c_untested_med_change'] = ((train['A1Cresult'] == 'None') & (train['change'] == 'Ch')).astype(np.float64)
    eval_cols['a1c_untested_med_change'] = ((eval_cols['A1Cresult'] == 'None') & (eval_cols['change'] == 'Ch')).astype(np.float64)
    extra_numeric.append('a1c_untested_med_change')

    train['glu_high'] = np.isin(train['max_glu_serum'], ['>200', '>300']).astype(np.float64)
    eval_cols['glu_high'] = np.isin(eval_cols['max_glu_serum'], ['>200', '>300']).astype(np.float64)
    extra_numeric.append('glu_high')

    train['adm_emergency'] = ((train['admission_type_id'] == '1') | (train['admission_source_id'] == '7')).astype(np.float64)
    eval_cols['adm_emergency'] = ((eval_cols['admission_type_id'] == '1') | (eval_cols['admission_source_id'] == '7')).astype(np.float64)
    extra_numeric.append('adm_emergency')

    # Specific clinical discharge pathways
    disp_tr = train['discharge_disposition_id']
    disp_ev = eval_cols['discharge_disposition_id']
    train['discharged_snf'] = np.isin(disp_tr, ['3', '4', '24']).astype(np.float64)
    eval_cols['discharged_snf'] = np.isin(disp_ev, ['3', '4', '24']).astype(np.float64)
    train['discharged_home_health'] = np.isin(disp_tr, ['6', '8']).astype(np.float64)
    eval_cols['discharged_home_health'] = np.isin(disp_ev, ['6', '8']).astype(np.float64)
    train['discharged_facility'] = np.isin(disp_tr, ['2', '3', '4', '5', '22', '23', '24', '28']).astype(np.float64)
    eval_cols['discharged_facility'] = np.isin(disp_ev, ['2', '3', '4', '5', '22', '23', '24', '28']).astype(np.float64)
    train['discharged_ama'] = np.isin(disp_tr, ['7', '18']).astype(np.float64)
    eval_cols['discharged_ama'] = np.isin(disp_ev, ['7', '18']).astype(np.float64)

    train['elderly_snf'] = (train['is_elderly'] * train['discharged_snf']).astype(np.float64)
    eval_cols['elderly_snf'] = (eval_cols['is_elderly'] * eval_cols['discharged_snf']).astype(np.float64)
    train['elderly_home_health'] = (train['is_elderly'] * train['discharged_home_health']).astype(np.float64)
    eval_cols['elderly_home_health'] = (eval_cols['is_elderly'] * eval_cols['discharged_home_health']).astype(np.float64)
    train['elderly_not_home'] = (train['is_elderly'] * (1.0 - train['discharged_home'])).astype(np.float64)
    eval_cols['elderly_not_home'] = (eval_cols['is_elderly'] * (1.0 - eval_cols['discharged_home'])).astype(np.float64)

    extra_numeric.extend([
        'discharged_snf', 'discharged_home_health', 'discharged_facility', 'discharged_ama',
        'elderly_snf', 'elderly_home_health', 'elderly_not_home',
    ])

    # Admission pathway groupings
    adm_t_tr = train['admission_type_id']
    adm_t_ev = eval_cols['admission_type_id']
    adm_s_tr = train['admission_source_id']
    adm_s_ev = eval_cols['admission_source_id']
    train['adm_elective'] = (adm_t_tr == '3').astype(np.float64)
    eval_cols['adm_elective'] = (adm_t_ev == '3').astype(np.float64)
    train['adm_urgent'] = (adm_t_tr == '2').astype(np.float64)
    eval_cols['adm_urgent'] = (adm_t_ev == '2').astype(np.float64)
    train['adm_from_transfer'] = np.isin(adm_s_tr, ['4', '5', '6', '10', '18', '22']).astype(np.float64)
    eval_cols['adm_from_transfer'] = np.isin(adm_s_ev, ['4', '5', '6', '10', '18', '22']).astype(np.float64)
    train['adm_from_physician'] = np.isin(adm_s_tr, ['1', '2', '3']).astype(np.float64)
    eval_cols['adm_from_physician'] = np.isin(adm_s_ev, ['1', '2', '3']).astype(np.float64)
    extra_numeric.extend(['adm_elective', 'adm_urgent', 'adm_from_transfer', 'adm_from_physician'])

    # Validated HOSPITAL Readmission Score proxy
    train['hospital_risk_score'] = (
        (inp_tr >= 2).astype(np.float64) * 2.0
        + (inp_tr == 1).astype(np.float64) * 1.0
        + (los_tr >= 5).astype(np.float64) * 2.0
        + (procs_tr > 0).astype(np.float64) * 1.0
        + np.isin(adm_t_tr, ['1', '2']).astype(np.float64) * 1.0
        + ((train['cm_cancer'] > 0) | (train['cm_metastatic'] > 0)).astype(np.float64) * 2.0
    )
    eval_cols['hospital_risk_score'] = (
        (inp_ev >= 2).astype(np.float64) * 2.0
        + (inp_ev == 1).astype(np.float64) * 1.0
        + (los_ev >= 5).astype(np.float64) * 2.0
        + (procs_ev > 0).astype(np.float64) * 1.0
        + np.isin(adm_t_ev, ['1', '2']).astype(np.float64) * 1.0
        + ((eval_cols['cm_cancer'] > 0) | (eval_cols['cm_metastatic'] > 0)).astype(np.float64) * 2.0
    )
    extra_numeric.append('hospital_risk_score')

    # Regimen classification & glycemic volatility
    oral_tr = np.zeros(n_tr, dtype=np.float64)
    oral_ev = np.zeros(n_ev, dtype=np.float64)
    for m in med_cols:
        if m != 'insulin':
            oral_tr += ((train[m] != 'No') & (train[m] is not None)).astype(np.float64)
            oral_ev += ((eval_cols[m] != 'No') & (eval_cols[m] is not None)).astype(np.float64)

    ins_tr = ((train['insulin'] != 'No') & (train['insulin'] is not None)).astype(np.float64)
    ins_ev = ((eval_cols['insulin'] != 'No') & (eval_cols['insulin'] is not None)).astype(np.float64)
    reg_tr = np.zeros(n_tr, dtype=np.float64)
    reg_tr[(oral_tr > 0) & (ins_tr == 0)] = 1.0
    reg_tr[(oral_tr == 0) & (ins_tr > 0)] = 2.0
    reg_tr[(oral_tr > 0) & (ins_tr > 0)] = 3.0
    train['diabetes_med_regimen'] = reg_tr
    reg_ev = np.zeros(n_ev, dtype=np.float64)
    reg_ev[(oral_ev > 0) & (ins_ev == 0)] = 1.0
    reg_ev[(oral_ev == 0) & (ins_ev > 0)] = 2.0
    reg_ev[(oral_ev > 0) & (ins_ev > 0)] = 3.0
    eval_cols['diabetes_med_regimen'] = reg_ev

    train['polypharmacy'] = (meds_tr >= 16).astype(np.float64)
    eval_cols['polypharmacy'] = (meds_ev >= 16).astype(np.float64)
    train['glycemic_volatility'] = (
        train['insulin_is_up']
        + train['insulin_is_down']
        + (train['change'] == 'Ch').astype(np.float64)
        + (train['A1Cresult'] == '>8').astype(np.float64)
    ).astype(np.float64)
    eval_cols['glycemic_volatility'] = (
        eval_cols['insulin_is_up']
        + eval_cols['insulin_is_down']
        + (eval_cols['change'] == 'Ch').astype(np.float64)
        + (eval_cols['A1Cresult'] == '>8').astype(np.float64)
    ).astype(np.float64)
    train['meds_per_diag'] = (meds_tr / np.maximum(diags_tr, 1.0)).astype(np.float64)
    eval_cols['meds_per_diag'] = (meds_ev / np.maximum(diags_ev, 1.0)).astype(np.float64)
    train['clinical_intensity_per_day'] = ((labs_tr + procs_tr + meds_tr) / los_tr).astype(np.float64)
    eval_cols['clinical_intensity_per_day'] = ((labs_ev + procs_ev + meds_ev) / los_ev).astype(np.float64)
    extra_numeric.extend(['diabetes_med_regimen', 'polypharmacy', 'glycemic_volatility', 'meds_per_diag', 'clinical_intensity_per_day'])

    # Multi-way tensor product coupling: Frailty x Therapeutic Volatility x Transition Vulnerability
    frailty_tr = train['cm_count'] + 2.0 * inp_tr + train['is_elderly']
    frailty_ev = eval_cols['cm_count'] + 2.0 * inp_ev + eval_cols['is_elderly']
    volatility_tr = train['insulin_is_up'] + train['insulin_is_down'] + train['num_med_changes']
    volatility_ev = eval_cols['insulin_is_up'] + eval_cols['insulin_is_down'] + eval_cols['num_med_changes']
    transition_risk_tr = 1.0 - train['discharged_home']
    transition_risk_ev = 1.0 - eval_cols['discharged_home']

    train['tensor_frailty_med'] = frailty_tr * volatility_tr
    eval_cols['tensor_frailty_med'] = frailty_ev * volatility_ev
    train['tensor_frailty_trans'] = frailty_tr * transition_risk_tr
    eval_cols['tensor_frailty_trans'] = frailty_ev * transition_risk_ev
    train['tensor_med_trans'] = volatility_tr * transition_risk_tr
    eval_cols['tensor_med_trans'] = volatility_ev * transition_risk_ev
    train['tensor_clinical_triad'] = frailty_tr * volatility_tr * transition_risk_tr
    eval_cols['tensor_clinical_triad'] = frailty_ev * volatility_ev * transition_risk_ev

    train['readmission_burden_index'] = (
        3.0 * inp_tr
        + 1.5 * emg_tr
        + 2.0 * (1.0 - train['discharged_home'])
        + train['cm_charlson_weighted']
        + 1.5 * (train['insulin_is_up'] + train['insulin_is_down'])
    ).astype(np.float64)
    eval_cols['readmission_burden_index'] = (
        3.0 * inp_ev
        + 1.5 * emg_ev
        + 2.0 * (1.0 - eval_cols['discharged_home'])
        + eval_cols['cm_charlson_weighted']
        + 1.5 * (eval_cols['insulin_is_up'] + eval_cols['insulin_is_down'])
    ).astype(np.float64)

    extra_numeric.extend([
        'tensor_frailty_med',
        'tensor_frailty_trans',
        'tensor_med_trans',
        'tensor_clinical_triad',
        'readmission_burden_index',
    ])

    # Deduplicate extra feature names
    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 8. Encode Tabular Data
    encoder = TabularEncoder(min_count=25)
    encoder.fit(
        task,
        train,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    cat_indices = encoder.categorical_indices

    # 9. Diverse Gradient-Boosted Model Ensemble
    # M1: Deep capacity model with tree path smoothing
    m1 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.024,
        num_leaves=42,
        max_depth=7,
        min_child_samples=105,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.62,
        reg_alpha=0.05,
        reg_lambda=1.2,
        cat_smooth=35.0,
        path_smooth=10.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    # M2: Shallow conservative regularized model
    m2 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.022,
        num_leaves=18,
        max_depth=5,
        min_child_samples=220,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.3,
        reg_lambda=5.0,
        cat_smooth=75.0,
        path_smooth=15.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    # M3: Extremely Randomized Trees (decorrelated randomized split candidate points)
    m3 = lgb.LGBMClassifier(
        extra_trees=True,
        n_estimators=240,
        learning_rate=0.025,
        num_leaves=38,
        min_child_samples=125,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.66,
        reg_alpha=0.03,
        reg_lambda=1.2,
        cat_smooth=40.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    # M4: Feature subspace specialist (narrow colsample)
    m4 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.021,
        num_leaves=28,
        min_child_samples=170,
        subsample=0.78,
        subsample_freq=1,
        colsample_bytree=0.40,
        reg_alpha=0.15,
        reg_lambda=2.5,
        cat_smooth=50.0,
        path_smooth=18.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    # M5: Positive boundary separation specialist with gradient reweighting
    m5 = lgb.LGBMClassifier(
        scale_pos_weight=1.35,
        n_estimators=240,
        learning_rate=0.023,
        num_leaves=30,
        max_depth=6,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.56,
        reg_alpha=0.10,
        reg_lambda=2.0,
        cat_smooth=45.0,
        path_smooth=8.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )

    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    m4.fit(x_train, y_train, categorical_feature=cat_indices)
    m5.fit(x_train, y_train, categorical_feature=cat_indices)

    p1 = m1.predict_proba(x_eval)[:, 1]
    p2 = m2.predict_proba(x_eval)[:, 1]
    p3 = m3.predict_proba(x_eval)[:, 1]
    p4 = m4.predict_proba(x_eval)[:, 1]
    p5 = m5.predict_proba(x_eval)[:, 1]

    # Fast pure NumPy rank normalization across heterogeneous tree structures
    def _fast_rankdata(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        return ranks

    n_eval_rows = float(len(p1))
    r1 = _fast_rankdata(p1) / n_eval_rows
    r2 = _fast_rankdata(p2) / n_eval_rows
    r3 = _fast_rankdata(p3) / n_eval_rows
    r4 = _fast_rankdata(p4) / n_eval_rows
    r5 = _fast_rankdata(p5) / n_eval_rows

    w = (0.23, 0.19, 0.21, 0.18, 0.19)
    p_raw = w[0] * p1 + w[1] * p2 + w[2] * p3 + w[3] * p4 + w[4] * p5
    r_blend = w[0] * r1 + w[1] * r2 + w[2] * r3 + w[3] * r4 + w[4] * r5

    # Quantile matching calibration: map rank-averaged consensus onto natural probability distribution
    r_min, r_max = np.min(r_blend), np.max(r_blend)
    r_norm = (r_blend - r_min) / max(r_max - r_min, 1e-12)
    p_sorted = np.sort(p_raw)
    q_grid = np.linspace(0.0, 1.0, len(p_sorted))
    p_calibrated = np.interp(r_norm, q_grid, p_sorted)

    return np.clip(p_calibrated, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
