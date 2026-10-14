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


from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

try:
    from clinical_toolkit import TabularEncoder, icd9_group, icd9_comorbidities, oof_target_encode
except ImportError:
    pass


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidity extraction (Charlson index features)
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_dict[k] = v.astype(float)
        eval_dict[k] = cm_eval[k].astype(float)
        extra_numeric.append(k)

    # 2. ICD-9 Chapter groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col}_group"
        train_dict[grp_col] = np.array([icd9_group(c) for c in train[col]], dtype=object)
        eval_dict[grp_col] = np.array([icd9_group(c) for c in eval_cols[col]], dtype=object)
        extra_categorical.append(grp_col)

    # Key diagnostic group indicators across any diagnosis slot
    for grp in ['circulatory', 'respiratory', 'diabetes', 'digestive', 'genitourinary']:
        col_name = f"has_group_{grp}"
        train_dict[col_name] = (
            (train_dict['diag_1_group'] == grp) |
            (train_dict['diag_2_group'] == grp) |
            (train_dict['diag_3_group'] == grp)
        ).astype(float)
        eval_dict[col_name] = (
            (eval_dict['diag_1_group'] == grp) |
            (eval_dict['diag_2_group'] == grp) |
            (eval_dict['diag_3_group'] == grp)
        ).astype(float)
        extra_numeric.append(col_name)

    # 3. High-cardinality out-of-fold target encoding
    te_candidates = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code'
    ]
    for col in te_candidates:
        if col in train:
            te_tr, te_ev = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            cname = f"{col}_te"
            train_dict[cname] = te_tr.astype(float)
            eval_dict[cname] = te_ev.astype(float)
            extra_numeric.append(cname)

    # 4. Clinical domain features: utilization & stay intensity
    def derive_clinical_numeric(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feats = {}
        stay = np.maximum(np.nan_to_num(d['time_in_hospital'].astype(float), nan=1.0), 1.0)
        n_inpatient = np.nan_to_num(d['number_inpatient'].astype(float), nan=0.0)
        n_emergency = np.nan_to_num(d['number_emergency'].astype(float), nan=0.0)
        n_outpatient = np.nan_to_num(d['number_outpatient'].astype(float), nan=0.0)

        total_prior = n_inpatient + n_emergency + n_outpatient
        feats['total_prior_visits'] = total_prior
        feats['inpatient_ratio'] = n_inpatient / (total_prior + 1.0)
        feats['emergency_ratio'] = n_emergency / (total_prior + 1.0)
        feats['has_prior_inpatient'] = (n_inpatient > 0.0).astype(float)
        feats['has_multiple_inpatient'] = (n_inpatient >= 2.0).astype(float)
        feats['has_prior_emergency'] = (n_emergency > 0.0).astype(float)
        feats['has_prior_visits'] = (total_prior > 0.0).astype(float)

        num_meds = np.nan_to_num(d['num_medications'].astype(float), nan=0.0)
        num_labs = np.nan_to_num(d['num_lab_procedures'].astype(float), nan=0.0)
        num_procs = np.nan_to_num(d['num_procedures'].astype(float), nan=0.0)
        num_diags = np.nan_to_num(d['number_diagnoses'].astype(float), nan=0.0)

        feats['labs_per_day'] = num_labs / stay
        feats['meds_per_day'] = num_meds / stay
        feats['procs_per_day'] = num_procs / stay
        feats['diags_per_day'] = num_diags / stay
        feats['med_proc_ratio'] = num_meds / (num_procs + 1.0)
        feats['lab_med_ratio'] = num_labs / (num_meds + 1.0)

        # Monotonic continuous age
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        feats['age_num'] = np.array([age_map.get(a, 60.0) for a in d['age']], dtype=float)

        # Admission & discharge flags
        feats['is_discharge_home'] = (d['discharge_disposition_id'] == '1').astype(float)
        feats['is_discharge_snf'] = (d['discharge_disposition_id'] == '3').astype(float)
        feats['is_discharge_home_health'] = (d['discharge_disposition_id'] == '6').astype(float)
        feats['is_emergency_admission'] = (d['admission_type_id'] == '1').astype(float)
        feats['is_elective'] = (d['admission_type_id'] == '3').astype(float)
        feats['is_emergency_source'] = (d['admission_source_id'] == '7').astype(float)

        # Lab test indicators
        feats['a1c_tested'] = (d['A1Cresult'] != 'None').astype(float)
        feats['a1c_abnormal'] = np.isin(d['A1Cresult'], ['>7', '>8']).astype(float)
        feats['glu_tested'] = (d['max_glu_serum'] != 'None').astype(float)
        feats['glu_abnormal'] = np.isin(d['max_glu_serum'], ['>200', '>300']).astype(float)

        return feats

    clin_tr = derive_clinical_numeric(train)
    clin_ev = derive_clinical_numeric(eval_cols)
    for k in clin_tr:
        train_dict[k] = clin_tr[k]
        eval_dict[k] = clin_ev[k]
        extra_numeric.append(k)

    # 5. Medication features
    med_list = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_list if m in train]

    def derive_med_features(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        if present_meds:
            m_stack = np.column_stack([d[m] for m in present_meds])
            is_active = (m_stack != 'No') & (m_stack != None)
            res['num_active_meds'] = np.sum(is_active, axis=1).astype(float)
            res['num_med_up'] = np.sum(m_stack == 'Up', axis=1).astype(float)
            res['num_med_down'] = np.sum(m_stack == 'Down', axis=1).astype(float)
            res['num_med_steady'] = np.sum(m_stack == 'Steady', axis=1).astype(float)
            res['num_med_changes'] = res['num_med_up'] + res['num_med_down']

        if 'insulin' in d:
            res['has_insulin'] = ((d['insulin'] != 'No') & (d['insulin'] != None)).astype(float)
            res['insulin_up'] = (d['insulin'] == 'Up').astype(float)
            res['insulin_down'] = (d['insulin'] == 'Down').astype(float)
            res['insulin_steady'] = (d['insulin'] == 'Steady').astype(float)

        if 'change' in d:
            res['has_med_change'] = (d['change'] == 'Ch').astype(float)
        if 'diabetesMed' in d:
            res['on_diabetes_med'] = (d['diabetesMed'] == 'Yes').astype(float)

        return res

    med_tr = derive_med_features(train)
    med_ev = derive_med_features(eval_cols)
    for k in med_tr:
        train_dict[k] = med_tr[k]
        eval_dict[k] = med_ev[k]
        extra_numeric.append(k)

    # Admission-Discharge interaction category
    train_dict['adm_disch_interaction'] = np.array(
        [f"{s}_{d}" for s, d in zip(train['admission_source_id'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_dict['adm_disch_interaction'] = np.array(
        [f"{s}_{d}" for s, d in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    extra_categorical.append('adm_disch_interaction')

    # 6. Tensor-inspired Low-Rank Phenotype Factorization (TruncatedSVD)
    # Decompose multi-way comorbidity-medication-utilization space into orthogonal latent factors
    svd_cols = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_diabetes_complicated',
        'has_group_circulatory', 'has_group_respiratory', 'has_group_diabetes',
        'has_group_digestive', 'has_group_genitourinary',
        'has_prior_inpatient', 'has_multiple_inpatient', 'has_prior_emergency',
        'is_discharge_home', 'is_discharge_snf', 'is_emergency_admission',
        'a1c_tested', 'a1c_abnormal', 'glu_tested'
    ]
    if present_meds:
        for m in present_meds:
            tr_m_active = ((train[m] != 'No') & (train[m] != None)).astype(float)
            ev_m_active = ((eval_cols[m] != 'No') & (eval_cols[m] != None)).astype(float)
            m_col = f"svd_in_{m}"
            train_dict[m_col] = tr_m_active
            eval_dict[m_col] = ev_m_active
            svd_cols.append(m_col)

    M_tr = np.column_stack([train_dict[c] for c in svd_cols])
    M_ev = np.column_stack([eval_dict[c] for c in svd_cols])

    n_svd_components = 8
    svd = TruncatedSVD(n_components=n_svd_components, algorithm='randomized', random_state=42)
    svd_tr = svd.fit_transform(M_tr)
    svd_ev = svd.transform(M_ev)

    for i in range(n_svd_components):
        sname = f"svd_phenotype_{i}"
        train_dict[sname] = svd_tr[:, i].astype(float)
        eval_dict[sname] = svd_ev[:, i].astype(float)
        extra_numeric.append(sname)

    # 7. Tabular encoding
    encoder = TabularEncoder(min_count=50)
    encoder.fit(task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # 8. Regularized ensemble of diverse gradient boosted trees
    models = [
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.5,
            reg_alpha=0.1,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.020,
            num_leaves=42,
            min_child_samples=80,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=3.0,
            reg_alpha=0.5,
            cat_smooth=30.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.022,
            num_leaves=22,
            min_child_samples=160,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=50.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=340,
            learning_rate=0.024,
            num_leaves=35,
            min_child_samples=100,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=1.0,
            reg_alpha=0.3,
            cat_smooth=35.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(x_eval), dtype=float)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
