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
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson-style comorbidities & Weighted Index
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_feat[k] = v.astype(np.float64)
        eval_feat[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # Standard Charlson Comorbidity Index (CCI) weighting
    charlson_weights = {
        'cm_mi': 1.0, 'cm_chf': 1.0, 'cm_pvd': 1.0, 'cm_cerebrovascular': 1.0,
        'cm_dementia': 1.0, 'cm_pulmonary': 1.0, 'cm_rheumatic': 1.0,
        'cm_peptic_ulcer': 1.0, 'cm_liver': 1.0,
        'cm_hemiplegia': 2.0, 'cm_renal': 2.0, 'cm_cancer': 2.0,
        'cm_diabetes_complicated': 2.0,
        'cm_metastatic': 6.0, 'cm_hiv': 6.0,
    }
    cci_train = np.zeros(len(y_train), dtype=np.float64)
    cci_eval = np.zeros(len(eval_cols['time_in_hospital']), dtype=np.float64)
    for cond, w in charlson_weights.items():
        if cond in cm_train:
            cci_train += w * cm_train[cond]
            cci_eval += w * cm_eval[cond]
    train_feat['charlson_score'] = cci_train
    eval_feat['charlson_score'] = cci_eval
    extra_numeric.append('charlson_score')

    train_feat['cm_high_multimorbidity'] = (cci_train >= 3.0).astype(np.float64)
    eval_feat['cm_high_multimorbidity'] = (cci_eval >= 3.0).astype(np.float64)
    extra_numeric.append('cm_high_multimorbidity')

    # Cardiorenal, metabolic, and pulmonary interactions
    train_feat['cm_chf_and_renal'] = (train_feat['cm_chf'] * train_feat['cm_renal']).astype(np.float64)
    eval_feat['cm_chf_and_renal'] = (eval_feat['cm_chf'] * eval_feat['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_chf_and_renal')

    train_feat['cm_chf_and_diabetes'] = (train_feat['cm_chf'] * train_feat['cm_diabetes_complicated']).astype(np.float64)
    eval_feat['cm_chf_and_diabetes'] = (eval_feat['cm_chf'] * eval_feat['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_chf_and_diabetes')

    train_feat['cm_renal_and_diabetes'] = (train_feat['cm_renal'] * train_feat['cm_diabetes_complicated']).astype(np.float64)
    eval_feat['cm_renal_and_diabetes'] = (eval_feat['cm_renal'] * eval_feat['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_renal_and_diabetes')

    # 2. ICD-9 Chapter groups and 3-digit categories (prefixes)
    def extract_prefix(code: Any) -> str:
        if code is None or code == '?' or code == '':
            return 'missing'
        s = str(code).strip()
        return s.split('.')[0] if '.' in s else s

    for d_col in ['diag_1', 'diag_2', 'diag_3']:
        g_col = f'{d_col}_group'
        train_feat[g_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_feat[g_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(g_col)

        p_col = f'{d_col}_cat'
        train_feat[p_col] = np.array([extract_prefix(c) for c in train[d_col]], dtype=object)
        eval_feat[p_col] = np.array([extract_prefix(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(p_col)

    # Combined encounter trajectory categories
    d_disp_tr = np.array([str(x) if x is not None else 'None' for x in train['discharge_disposition_id']], dtype=object)
    d_disp_ev = np.array([str(x) if x is not None else 'None' for x in eval_cols['discharge_disposition_id']], dtype=object)
    a_src_tr = np.array([str(x) if x is not None else 'None' for x in train['admission_source_id']], dtype=object)
    a_src_ev = np.array([str(x) if x is not None else 'None' for x in eval_cols['admission_source_id']], dtype=object)

    train_feat['disp_and_adm_src'] = np.array([f'{d}_{s}' for d, s in zip(d_disp_tr, a_src_tr)], dtype=object)
    eval_feat['disp_and_adm_src'] = np.array([f'{d}_{s}' for d, s in zip(d_disp_ev, a_src_ev)], dtype=object)
    extra_categorical.append('disp_and_adm_src')

    train_feat['disp_and_diag_grp'] = np.array([f'{d}_{g}' for d, g in zip(d_disp_tr, train_feat['diag_1_group'])], dtype=object)
    eval_feat['disp_and_diag_grp'] = np.array([f'{d}_{g}' for d, g in zip(d_disp_ev, eval_feat['diag_1_group'])], dtype=object)
    extra_categorical.append('disp_and_diag_grp')

    # 3. Out-of-fold target encoding for categorical features
    te_columns = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_cat', 'diag_2_cat', 'diag_3_cat',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code',
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'disp_and_adm_src',
        'disp_and_diag_grp',
    ]
    for col in te_columns:
        if col in train_feat and col in eval_feat:
            tr_enc, ev_enc = oof_target_encode(
                train_feat[col], y_train, eval_feat[col],
                n_folds=5, smoothing=25.0, seed=42
            )
            te_name = f'te_{col}'
            train_feat[te_name] = tr_enc.astype(np.float64)
            eval_feat[te_name] = ev_enc.astype(np.float64)
            extra_numeric.append(te_name)

    # Composite diagnosis risk indicators
    train_feat['te_diag_max'] = np.maximum(
        train_feat['te_diag_1'],
        np.maximum(train_feat['te_diag_2'], train_feat['te_diag_3'])
    )
    eval_feat['te_diag_max'] = np.maximum(
        eval_feat['te_diag_1'],
        np.maximum(eval_feat['te_diag_2'], eval_feat['te_diag_3'])
    )
    extra_numeric.append('te_diag_max')

    train_feat['te_diag_mean'] = (
        train_feat['te_diag_1'] + train_feat['te_diag_2'] + train_feat['te_diag_3']
    ) / 3.0
    eval_feat['te_diag_mean'] = (
        eval_feat['te_diag_1'] + eval_feat['te_diag_2'] + eval_feat['te_diag_3']
    ) / 3.0
    extra_numeric.append('te_diag_mean')

    train_feat['te_diag_cat_max'] = np.maximum(
        train_feat['te_diag_1_cat'],
        np.maximum(train_feat['te_diag_2_cat'], train_feat['te_diag_3_cat'])
    )
    eval_feat['te_diag_cat_max'] = np.maximum(
        eval_feat['te_diag_1_cat'],
        np.maximum(eval_feat['te_diag_2_cat'], eval_feat['te_diag_3_cat'])
    )
    extra_numeric.append('te_diag_cat_max')

    train_feat['te_diag_cat_mean'] = (
        train_feat['te_diag_1_cat'] + train_feat['te_diag_2_cat'] + train_feat['te_diag_3_cat']
    ) / 3.0
    eval_feat['te_diag_cat_mean'] = (
        eval_feat['te_diag_1_cat'] + eval_feat['te_diag_2_cat'] + eval_feat['te_diag_3_cat']
    ) / 3.0
    extra_numeric.append('te_diag_cat_mean')

    # Frequency encoding for high-cardinality features
    freq_columns = ['diag_1_cat', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id']
    for col in freq_columns:
        if col in train_feat and col in eval_feat:
            tr_c = np.array([str(x) if x is not None else '__MISSING__' for x in train_feat[col]])
            ev_c = np.array([str(x) if x is not None else '__MISSING__' for x in eval_feat[col]])
            u_vals, u_counts = np.unique(tr_c, return_counts=True)
            freq_dict = dict(zip(u_vals, u_counts / float(len(tr_c))))
            f_name = f'freq_{col}'
            train_feat[f_name] = np.array([freq_dict.get(x, 0.0) for x in tr_c], dtype=np.float64)
            eval_feat[f_name] = np.array([freq_dict.get(x, 0.0) for x in ev_c], dtype=np.float64)
            extra_numeric.append(f_name)

    # 4. Clinical domain features: utilization, acuity, medications, and physiology
    def compute_clinical_features(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        n = len(data['time_in_hospital'])

        # Prior visits & utilization intensity
        inp = np.nan_to_num(data.get('number_inpatient', np.zeros(n)), nan=0.0)
        emg = np.nan_to_num(data.get('number_emergency', np.zeros(n)), nan=0.0)
        outp = np.nan_to_num(data.get('number_outpatient', np.zeros(n)), nan=0.0)

        tot_visits = inp + emg + outp
        res['total_visits'] = tot_visits
        res['has_inpatient'] = (inp > 0).astype(np.float64)
        res['has_emergency'] = (emg > 0).astype(np.float64)
        res['multiple_inpatient'] = (inp >= 2).astype(np.float64)
        res['frequent_inpatient'] = (inp >= 3).astype(np.float64)
        res['high_utilizer'] = ((inp >= 2) | (tot_visits >= 4)).astype(np.float64)
        res['inpatient_ratio'] = inp / (tot_visits + 1.0)
        res['emergency_ratio'] = emg / (tot_visits + 1.0)
        res['util_intensity'] = inp * 3.0 + emg * 2.0 + outp
        res['emergency_no_inpatient'] = ((emg > 0) & (inp == 0)).astype(np.float64)

        # Log transforms for heavy-tailed counts
        res['log_inpatient'] = np.log1p(inp)
        res['log_emergency'] = np.log1p(emg)
        res['log_outpatient'] = np.log1p(outp)
        res['log_total_visits'] = np.log1p(tot_visits)

        # Length of stay & encounter intensity
        los = np.maximum(np.nan_to_num(data.get('time_in_hospital', np.ones(n)), nan=1.0), 1.0)
        n_labs = np.nan_to_num(data.get('num_lab_procedures', np.zeros(n)), nan=0.0)
        n_procs = np.nan_to_num(data.get('num_procedures', np.zeros(n)), nan=0.0)
        n_meds = np.nan_to_num(data.get('num_medications', np.zeros(n)), nan=0.0)
        n_diag = np.maximum(np.nan_to_num(data.get('number_diagnoses', np.ones(n)), nan=1.0), 1.0)

        res['log_time_in_hospital'] = np.log1p(los)
        res['log_num_medications'] = np.log1p(n_meds)
        res['log_num_lab_procedures'] = np.log1p(n_labs)
        res['labs_per_day'] = n_labs / los
        res['meds_per_day'] = n_meds / los
        res['procs_per_day'] = n_procs / los
        res['total_procedures'] = n_labs + n_procs
        res['meds_per_diag'] = n_meds / n_diag
        res['labs_per_med'] = n_labs / np.maximum(n_meds, 1.0)
        res['overall_intensity'] = (n_labs + 5.0 * n_procs + 2.0 * n_meds) / los
        res['long_stay'] = (los >= 7.0).astype(np.float64)
        res['inpatient_x_los'] = inp * los

        # Discharge disposition clinical indicators
        raw_disp = data.get('discharge_disposition_id', np.full(n, None))
        disp_str = np.array([str(x) if x is not None else 'None' for x in raw_disp])
        res['disp_is_home'] = (disp_str == '1').astype(np.float64)
        res['disp_is_home_health'] = (disp_str == '6').astype(np.float64)
        res['disp_is_facility'] = np.isin(disp_str, ['2', '3', '4', '5', '22', '23', '24']).astype(np.float64)
        res['disp_is_ama'] = (disp_str == '7').astype(np.float64)

        # Admission source & type indicators
        raw_src = data.get('admission_source_id', np.full(n, None))
        src_str = np.array([str(x) if x is not None else 'None' for x in raw_src])
        res['admitted_from_er'] = (src_str == '7').astype(np.float64)

        raw_typ = data.get('admission_type_id', np.full(n, None))
        typ_str = np.array([str(x) if x is not None else 'None' for x in raw_typ])
        res['admission_is_emergency'] = np.isin(typ_str, ['1', '2']).astype(np.float64)
        res['admission_is_elective'] = (typ_str == '3').astype(np.float64)

        # Interaction: Acute transfer / facility with prior inpatient
        res['facility_and_prior_inp'] = res['disp_is_facility'] * (inp > 0).astype(np.float64)
        res['home_health_and_prior_inp'] = res['disp_is_home_health'] * (inp > 0).astype(np.float64)

        # Demographics: Age mapping & interactions
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0,
        }
        raw_age = data.get('age', np.full(n, None))
        age_num = np.array([age_map.get(str(a), 65.0) for a in raw_age], dtype=np.float64)
        res['age_numeric'] = age_num
        res['age_elderly'] = (age_num >= 70.0).astype(np.float64)
        res['age_x_inpatient'] = age_num * inp
        res['age_x_los'] = age_num * los

        # Missingness indicators
        res['missing_specialty'] = (data.get('medical_specialty', np.full(n, None)) == None).astype(np.float64)
        res['missing_payer'] = (data.get('payer_code', np.full(n, None)) == None).astype(np.float64)

        # Lab tests: Glucose & A1C
        glu = data.get('max_glu_serum', np.full(n, None))
        res['glu_measured'] = np.isin(glu, ['Norm', '>200', '>300']).astype(np.float64)
        res['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)

        a1c = data.get('A1Cresult', np.full(n, None))
        res['a1c_measured'] = np.isin(a1c, ['Norm', '>7', '>8']).astype(np.float64)
        res['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        res['a1c_very_high'] = (a1c == '>8').astype(np.float64)

        # Medication changes & status
        ch = data.get('change', np.full(n, None))
        res['med_changed'] = (ch == 'Ch').astype(np.float64)
        dm = data.get('diabetesMed', np.full(n, None))
        res['on_diabetes_med'] = (dm == 'Yes').astype(np.float64)

        # Medication count analysis & Regimen classification
        oral_med_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
        ]
        active_count = np.zeros(n, dtype=np.float64)
        up_count = np.zeros(n, dtype=np.float64)
        down_count = np.zeros(n, dtype=np.float64)
        steady_count = np.zeros(n, dtype=np.float64)
        on_oral = np.zeros(n, dtype=bool)

        for m in oral_med_cols:
            if m in data:
                m_val = data[m]
                is_act = np.isin(m_val, ['Steady', 'Up', 'Down'])
                active_count += is_act
                on_oral |= is_act
                up_count += (m_val == 'Up')
                down_count += (m_val == 'Down')
                steady_count += (m_val == 'Steady')

        # Insulin details
        ins = data.get('insulin', np.full(n, None))
        ins_active = np.isin(ins, ['Steady', 'Up', 'Down'])
        active_count += ins_active
        up_count += (ins == 'Up')
        down_count += (ins == 'Down')
        steady_count += (ins == 'Steady')

        res['med_active_count'] = active_count
        res['med_up_count'] = up_count
        res['med_down_count'] = down_count
        res['med_change_count'] = up_count + down_count
        res['med_steady_count'] = steady_count

        res['insulin_active'] = ins_active.astype(np.float64)
        res['insulin_up'] = (ins == 'Up').astype(np.float64)
        res['insulin_down'] = (ins == 'Down').astype(np.float64)
        res['insulin_changed'] = np.isin(ins, ['Up', 'Down']).astype(np.float64)

        # Regimen patterns
        res['regimen_insulin_only'] = (ins_active & (~on_oral)).astype(np.float64)
        res['regimen_oral_only'] = ((~ins_active) & on_oral).astype(np.float64)
        res['regimen_dual_therapy'] = (ins_active & on_oral).astype(np.float64)
        res['regimen_no_meds'] = ((~ins_active) & (~on_oral)).astype(np.float64)

        met = data.get('metformin', np.full(n, None))
        res['on_metformin'] = np.isin(met, ['Steady', 'Up', 'Down']).astype(np.float64)

        return res

    tr_clin = compute_clinical_features(train)
    ev_clin = compute_clinical_features(eval_cols)
    for col_name, col_arr in tr_clin.items():
        train_feat[col_name] = col_arr
        eval_feat[col_name] = ev_clin[col_name]
        extra_numeric.append(col_name)

    # 5. Encode all features
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_feat,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)
    cat_idx = encoder.categorical_indices

    # 6. Diverse Ensemble: LightGBM Architectures + XGBoost
    # Model 1: Balanced LightGBM
    m1 = lgb.LGBMClassifier(
        n_estimators=380,
        learning_rate=0.022,
        num_leaves=31,
        max_depth=6,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=2.0,
        reg_alpha=0.2,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_idx)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative Regularized Shallow LightGBM
    m2 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.020,
        num_leaves=18,
        max_depth=5,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=5.0,
        reg_alpha=0.5,
        cat_smooth=70.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_idx)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Interaction-focused Deeper LightGBM
    m3 = lgb.LGBMClassifier(
        n_estimators=340,
        learning_rate=0.020,
        num_leaves=42,
        max_depth=7,
        min_child_samples=100,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=25.0,
        random_state=789,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_idx)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: Feature-subsampled Diverse LightGBM
    m4 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.024,
        num_leaves=25,
        max_depth=6,
        min_child_samples=180,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.45,
        reg_lambda=3.0,
        reg_alpha=0.3,
        cat_smooth=50.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=cat_idx)
    p4 = m4.predict_proba(x_eval)[:, 1]

    # Model 5: Histogram-based XGBoost
    xgb_preds = None
    try:
        xgb_mod = globals().get('xgb', globals().get('xgboost', None))
        if xgb_mod is not None:
            m_xgb = xgb_mod.XGBClassifier(
                n_estimators=280,
                learning_rate=0.022,
                max_depth=5,
                min_child_weight=15.0,
                subsample=0.75,
                colsample_bytree=0.60,
                reg_lambda=3.0,
                reg_alpha=0.2,
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                eval_metric='auc',
            )
            m_xgb.fit(x_train, y_train)
            xgb_preds = m_xgb.predict_proba(x_eval)[:, 1]
    except Exception:
        xgb_preds = None

    if xgb_preds is not None:
        preds = 0.28 * p1 + 0.25 * p2 + 0.18 * p3 + 0.14 * p4 + 0.15 * xgb_preds
    else:
        preds = 0.33 * p1 + 0.29 * p2 + 0.22 * p3 + 0.16 * p4

    # Safe clipping to guarantee valid probabilities
    prior = float(np.mean(y_train))
    preds = np.nan_to_num(preds, nan=prior)
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
