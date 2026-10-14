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


def safe_icd9_group(code):
    """Map ICD-9 code to Strack chapter group with guaranteed fallback."""
    if 'icd9_group' in globals():
        try:
            return icd9_group(code)
        except Exception:
            pass
    if code is None or code == '?' or code == '':
        return 'missing'
    s = str(code).strip()
    if s.startswith('V') or s.startswith('E'):
        return 'other'
    try:
        val = float(s)
        if (390 <= val <= 459) or abs(val - 785) < 0.5:
            return 'circulatory'
        if (460 <= val <= 519) or abs(val - 786) < 0.5:
            return 'respiratory'
        if (520 <= val <= 579) or abs(val - 787) < 0.5:
            return 'digestive'
        if 250 <= val < 251:
            return 'diabetes'
        if 800 <= val <= 999:
            return 'injury'
        if 710 <= val <= 739:
            return 'musculoskeletal'
        if (580 <= val <= 629) or abs(val - 788) < 0.5:
            return 'genitourinary'
        if 140 <= val <= 239:
            return 'neoplasms'
        return 'other'
    except ValueError:
        return 'other'


def _extract_domain_features(cols: Dict[str, np.ndarray]):
    """Extract clinical, utilization, and medication-regimen features."""
    from sklearn.decomposition import TruncatedSVD

    n_rows = len(cols['time_in_hospital'])

    # 1. Utilisation and Intensity
    time_hosp = np.nan_to_num(cols['time_in_hospital'].astype(np.float64), nan=1.0)
    time_hosp_safe = np.maximum(time_hosp, 1.0)
    num_meds = np.nan_to_num(cols['num_medications'].astype(np.float64), nan=0.0)
    num_labs = np.nan_to_num(cols['num_lab_procedures'].astype(np.float64), nan=0.0)
    num_procs = np.nan_to_num(cols['num_procedures'].astype(np.float64), nan=0.0)
    num_diags = np.nan_to_num(cols['number_diagnoses'].astype(np.float64), nan=0.0)

    n_out = np.nan_to_num(cols['number_outpatient'].astype(np.float64), nan=0.0)
    n_emerg = np.nan_to_num(cols['number_emergency'].astype(np.float64), nan=0.0)
    n_inpat = np.nan_to_num(cols['number_inpatient'].astype(np.float64), nan=0.0)

    total_prior = n_out + n_emerg + n_inpat
    inpat_ratio = n_inpat / (total_prior + 1.0)
    emerg_ratio = n_emerg / (total_prior + 1.0)
    has_prior_inpat = (n_inpat > 0).astype(np.float64)
    has_prior_emerg = (n_emerg > 0).astype(np.float64)

    med_per_day = num_meds / time_hosp_safe
    lab_per_day = num_labs / time_hosp_safe
    proc_per_day = num_procs / time_hosp_safe
    diag_per_day = num_diags / time_hosp_safe
    inpat_x_emerg = n_inpat * n_emerg
    inpat_x_diags = n_inpat * num_diags

    # 2. Age Midpoint
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    age_arr = cols.get('age', np.full(n_rows, None, dtype=object))
    age_midpoint = np.array([age_map.get(str(a) if a is not None else '', 65.0) for a in age_arr], dtype=np.float64)

    # 3. Admission and Discharge indicators
    disch = cols.get('discharge_disposition_id', np.full(n_rows, None, dtype=object))
    is_discharged_home = np.array([1.0 if str(x) == '1' else 0.0 for x in disch], dtype=np.float64)

    adm_type = cols.get('admission_type_id', np.full(n_rows, None, dtype=object))
    is_emergency_adm = np.array([1.0 if str(x) == '1' else 0.0 for x in adm_type], dtype=np.float64)

    adm_src = cols.get('admission_source_id', np.full(n_rows, None, dtype=object))
    adm_source_er = np.array([1.0 if str(x) == '7' else 0.0 for x in adm_src], dtype=np.float64)

    # 4. Lab results
    a1c = cols.get('A1Cresult', None)
    if a1c is not None:
        a1c_tested = np.array([0.0 if x is None or str(x) == 'None' else 1.0 for x in a1c], dtype=np.float64)
        a1c_high = np.array([1.0 if str(x) in ('>7', '>8') else 0.0 for x in a1c], dtype=np.float64)
    else:
        a1c_tested = np.zeros(n_rows, dtype=np.float64)
        a1c_high = np.zeros(n_rows, dtype=np.float64)

    glu = cols.get('max_glu_serum', None)
    if glu is not None:
        glu_tested = np.array([0.0 if x is None or str(x) == 'None' else 1.0 for x in glu], dtype=np.float64)
        glu_high = np.array([1.0 if str(x) in ('>200', '>300') else 0.0 for x in glu], dtype=np.float64)
    else:
        glu_tested = np.zeros(n_rows, dtype=np.float64)
        glu_high = np.zeros(n_rows, dtype=np.float64)

    # 5. Medications & Tensor Matrix Setup
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_matrix = np.zeros((n_rows, len(med_names)), dtype=np.float64)
    med_val_map = {'No': 0.0, 'Steady': 1.0, 'Up': 2.0, 'Down': -1.0}

    med_num_active = np.zeros(n_rows, dtype=np.float64)
    med_num_up = np.zeros(n_rows, dtype=np.float64)
    med_num_down = np.zeros(n_rows, dtype=np.float64)
    med_num_steady = np.zeros(n_rows, dtype=np.float64)

    for j, m_name in enumerate(med_names):
        if m_name in cols:
            arr = cols[m_name]
            for i, val in enumerate(arr):
                v_str = str(val) if val is not None else 'No'
                med_matrix[i, j] = med_val_map.get(v_str, 0.0)
                if v_str != 'No' and val is not None:
                    med_num_active[i] += 1.0
                if v_str == 'Up':
                    med_num_up[i] += 1.0
                elif v_str == 'Down':
                    med_num_down[i] += 1.0
                elif v_str == 'Steady':
                    med_num_steady[i] += 1.0

    med_num_changes = med_num_up + med_num_down

    insulin_arr = cols.get('insulin', None)
    insulin_code = np.zeros(n_rows, dtype=np.float64)
    if insulin_arr is not None:
        ins_map = {'No': 0.0, 'Steady': 1.0, 'Down': 2.0, 'Up': 3.0}
        for i, val in enumerate(insulin_arr):
            insulin_code[i] = ins_map.get(str(val) if val is not None else 'No', 0.0)

    chg = cols.get('change', None)
    has_med_change = np.array([1.0 if str(x) == 'Ch' else 0.0 for x in chg], dtype=np.float64) if chg is not None else np.zeros(n_rows)

    d_med = cols.get('diabetesMed', None)
    on_diabetes_med = np.array([1.0 if str(x) == 'Yes' else 0.0 for x in d_med], dtype=np.float64) if d_med is not None else np.zeros(n_rows)

    # 6. Diagnosis Chapter Groups & Comorbidities
    d1 = cols.get('diag_1', None)
    d2 = cols.get('diag_2', None)
    d3 = cols.get('diag_3', None)

    d1_group = np.empty(n_rows, dtype=object)
    d2_group = np.empty(n_rows, dtype=object)
    d3_group = np.empty(n_rows, dtype=object)

    has_diabetes = np.zeros(n_rows, dtype=np.float64)
    has_chf = np.zeros(n_rows, dtype=np.float64)
    has_cad = np.zeros(n_rows, dtype=np.float64)
    has_htn = np.zeros(n_rows, dtype=np.float64)
    has_ckd = np.zeros(n_rows, dtype=np.float64)
    has_copd = np.zeros(n_rows, dtype=np.float64)
    has_infection = np.zeros(n_rows, dtype=np.float64)
    has_liver = np.zeros(n_rows, dtype=np.float64)

    group_list = [
        'circulatory', 'respiratory', 'digestive', 'diabetes', 'injury',
        'musculoskeletal', 'genitourinary', 'neoplasms', 'other', 'missing'
    ]
    group_to_idx = {g: idx for idx, g in enumerate(group_list)}
    diag_group_matrix = np.zeros((n_rows, len(group_list)), dtype=np.float64)

    for i in range(n_rows):
        c1 = d1[i] if d1 is not None else None
        c2 = d2[i] if d2 is not None else None
        c3 = d3[i] if d3 is not None else None

        g1 = safe_icd9_group(c1)
        g2 = safe_icd9_group(c2)
        g3 = safe_icd9_group(c3)

        d1_group[i] = g1
        d2_group[i] = g2
        d3_group[i] = g3

        if g1 in group_to_idx:
            diag_group_matrix[i, group_to_idx[g1]] += 1.0
        if g2 in group_to_idx:
            diag_group_matrix[i, group_to_idx[g2]] += 1.0
        if g3 in group_to_idx:
            diag_group_matrix[i, group_to_idx[g3]] += 1.0

        for c in (c1, c2, c3):
            if c is None:
                continue
            s = str(c).strip()
            if not s:
                continue
            if s.startswith('250'):
                has_diabetes[i] = 1.0
            if s.startswith('428'):
                has_chf[i] = 1.0
            if s.startswith(('410', '411', '412', '413', '414')):
                has_cad[i] = 1.0
            if s.startswith(('401', '402', '403', '404', '405')):
                has_htn[i] = 1.0
            if s.startswith(('585', '586', '584')):
                has_ckd[i] = 1.0
            if s.startswith(('490', '491', '492', '493', '494', '495', '496')):
                has_copd[i] = 1.0
            if s.startswith(('038', '995.9', '003', '036')):
                has_infection[i] = 1.0
            if s.startswith(('571', '572', '573')):
                has_liver[i] = 1.0

    clinical_matrix = np.hstack([med_matrix, diag_group_matrix])

    features = {
        'age_midpoint': age_midpoint,
        'has_diabetes': has_diabetes,
        'has_chf': has_chf,
        'has_cad': has_cad,
        'has_htn': has_htn,
        'has_ckd': has_ckd,
        'has_copd': has_copd,
        'comorbidity_count': comorbidity_count,
        'med_num_active': med_num_active,
        'med_num_up': med_num_up,
        'med_num_down': med_num_down,
        'med_num_changes': med_num_changes,
        'med_num_steady': med_num_steady,
        'insulin_code': insulin_code,
        'has_med_change': has_med_change,
        'on_diabetes_med': on_diabetes_med,
        'total_prior_visits': total_prior,
        'inpatient_ratio': inpat_ratio,
        'emergency_ratio': emerg_ratio,
        'has_prior_inpatient': has_prior_inpat,
        'has_prior_emergency': has_prior_emerg,
        'med_per_day': med_per_day,
        'lab_per_day': lab_per_day,
        'proc_per_day': proc_per_day,
        'diag_per_day': diag_per_day,
        'inpatient_x_emergency': inpat_x_emerg,
        'inpatient_x_diagnoses': inpat_x_diags,
        'is_discharged_home': is_discharged_home,
        'is_emergency_admission': is_emergency_adm,
        'admission_source_er': adm_source_er,
        'a1c_tested': a1c_tested,
        'a1c_high': a1c_high,
        'glu_tested': glu_tested,
        'glu_high': glu_high,
        'diag_1_group': d1_group,
        'diag_2_group': d2_group,
        'diag_3_group': d3_group,
    }
    return features, clinical_matrix


def _compute_target_encodings(train_arr, y_train, eval_arr, m=20.0, n_splits=5):
    """Compute out-of-fold Bayesian smoothed target encodings."""
    from sklearn.model_selection import StratifiedKFold

    global_mean = float(np.mean(y_train))
    n_train = len(y_train)
    oof_train = np.full(n_train, global_mean, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(train_arr, y_train):
        cat_stats = {}
        for c, y in zip(train_arr[tr_idx], y_train[tr_idx]):
            k = str(c) if c is not None else '__MISSING__'
            stat = cat_stats.setdefault(k, [0.0, 0])
            stat[0] += y
            stat[1] += 1

        val_enc = np.empty(len(val_idx), dtype=np.float64)
        for i, c in enumerate(train_arr[val_idx]):
            k = str(c) if c is not None else '__MISSING__'
            if k in cat_stats:
                s, cnt = cat_stats[k]
                val_enc[i] = (s + m * global_mean) / (cnt + m)
            else:
                val_enc[i] = global_mean
        oof_train[val_idx] = val_enc

    full_stats = {}
    for c, y in zip(train_arr, y_train):
        k = str(c) if c is not None else '__MISSING__'
        stat = full_stats.setdefault(k, [0.0, 0])
        stat[0] += y
        stat[1] += 1

    eval_enc = np.empty(len(eval_arr), dtype=np.float64)
    for i, c in enumerate(eval_arr):
        k = str(c) if c is not None else '__MISSING__'
        if k in full_stats:
            s, cnt = full_stats[k]
            eval_enc[i] = (s + m * global_mean) / (cnt + m)
        else:
            eval_enc[i] = global_mean

    return oof_train, eval_enc


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit optimized low-rank decomposition + tree ensemble to predict 30-day readmission."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    import xgboost as xgb

    train_dict = dict(train)
    eval_dict = dict(eval_cols)

    # 1. Feature Engineering
    tr_feats, tr_clinical = _extract_domain_features(train_dict)
    ev_feats, ev_clinical = _extract_domain_features(eval_dict)

    for k, v in tr_feats.items():
        train_dict[k] = v
    for k, v in ev_feats.items():
        eval_dict[k] = v

    # 2. Low-Rank Factorization (SVD Tensor Decomposition)
    n_components = 5
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tr_clinical)

    tr_svd = svd.transform(tr_clinical)
    ev_svd = svd.transform(ev_clinical)

    tr_rec = svd.inverse_transform(tr_svd)
    ev_rec = svd.inverse_transform(ev_svd)

    tr_residual = np.linalg.norm(tr_clinical - tr_rec, axis=1)
    ev_residual = np.linalg.norm(ev_clinical - ev_rec, axis=1)

    extra_numeric = []
    for comp in range(n_components):
        col_name = f'clinical_svd_{comp}'
        train_dict[col_name] = tr_svd[:, comp]
        eval_dict[col_name] = ev_svd[:, comp]
        extra_numeric.append(col_name)

    train_dict['clinical_svd_residual'] = tr_residual
    eval_dict['clinical_svd_residual'] = ev_residual
    extra_numeric.append('clinical_svd_residual')

    # Domain numeric features
    domain_num_keys = [
        'age_midpoint', 'is_elderly', 'has_diabetes', 'has_chf', 'has_cad', 'has_htn', 'has_ckd', 'has_copd',
        'has_infection', 'has_liver',
        'comorbidity_count', 'med_num_active', 'med_num_up', 'med_num_down', 'med_num_changes',
        'med_num_steady', 'insulin_code', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_active',
        'has_med_change', 'on_diabetes_med',
        'total_prior_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'has_prior_emergency', 'med_per_day', 'lab_per_day',
        'proc_per_day', 'diag_per_day', 'inpatient_x_emergency', 'inpatient_x_diagnoses',
        'is_discharged_home', 'is_discharged_to_facility', 'is_emergency_admission', 'admission_source_er',
        'is_admission_referral',
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high',
        'log_num_labs', 'log_num_meds', 'log_num_procs', 'log_num_diags', 'log_time_hosp',
        'log_n_out', 'log_n_emerg', 'log_n_inpat', 'log_total_prior'
    ]
    extra_numeric.extend(domain_num_keys)

    # 3. Target Encodings on Key Categoricals
    te_columns = [
        ('discharge_disposition_id', 'te_discharge_disp'),
        ('medical_specialty', 'te_medical_specialty'),
        ('admission_source_id', 'te_admission_source'),
        ('admission_type_id', 'te_admission_type'),
        ('payer_code', 'te_payer_code'),
        ('diag_1', 'te_diag_1'),
        ('diag_2', 'te_diag_2'),
        ('diag_3', 'te_diag_3'),
        ('diag_1_group', 'te_diag_1_group'),
    ]
    for orig_col, te_name in te_columns:
        tr_te, ev_te = _compute_target_encodings(train_dict[orig_col], y_train, eval_dict[orig_col])
        train_dict[te_name] = tr_te
        eval_dict[te_name] = ev_te
        extra_numeric.append(te_name)

    extra_categorical = ['diag_1_group', 'diag_2_group', 'diag_3_group']

    # 4. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)
    cat_indices = encoder.categorical_indices

    # 5. Dual Model Configurations + XGBoost
    cfg_a = {
        'n_estimators': 350,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'min_child_samples': 45,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'random_state': 42,
        'n_jobs': 1,
        'verbose': -1,
    }

    cfg_b = {
        'n_estimators': 380,
        'learning_rate': 0.025,
        'num_leaves': 20,
        'min_child_samples': 60,
        'subsample': 0.85,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 5.0,
        'random_state': 123,
        'n_jobs': 1,
        'verbose': -1,
    }

    cfg_xgb = {
        'n_estimators': 250,
        'learning_rate': 0.03,
        'max_depth': 5,
        'min_child_weight': 4.0,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'random_state': 99,
        'n_jobs': 1,
        'verbosity': 0,
    }

    # 6. Stratified K-Fold CV Bagging
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_eval_preds = np.zeros(len(x_eval), dtype=np.float64)

    for tr_idx, _ in skf.split(x_train, y_train):
        xt_fold = x_train[tr_idx]
        yt_fold = y_train[tr_idx]

        ma = lgb.LGBMClassifier(**cfg_a)
        ma.fit(xt_fold, yt_fold, categorical_feature=cat_indices)
        p_a = ma.predict_proba(x_eval)[:, 1]

        mb = lgb.LGBMClassifier(**cfg_b)
        mb.fit(xt_fold, yt_fold, categorical_feature=cat_indices)
        p_b = mb.predict_proba(x_eval)[:, 1]

        mx = xgb.XGBClassifier(**cfg_xgb)
        mx.fit(xt_fold, yt_fold)
        p_x = mx.predict_proba(x_eval)[:, 1]

        cv_eval_preds += (0.40 * p_a + 0.35 * p_b + 0.25 * p_x) / n_splits

    # Full data models
    ma_full = lgb.LGBMClassifier(**cfg_a)
    ma_full.fit(x_train, y_train, categorical_feature=cat_indices)
    p_a_full = ma_full.predict_proba(x_eval)[:, 1]

    mb_full = lgb.LGBMClassifier(**cfg_b)
    mb_full.fit(x_train, y_train, categorical_feature=cat_indices)
    p_b_full = mb_full.predict_proba(x_eval)[:, 1]

    mx_full = xgb.XGBClassifier(**cfg_xgb)
    mx_full.fit(x_train, y_train)
    p_x_full = mx_full.predict_proba(x_eval)[:, 1]

    full_eval_preds = 0.40 * p_a_full + 0.35 * p_b_full + 0.25 * p_x_full

    # Blended ensemble
    final_preds = 0.65 * cv_eval_preds + 0.35 * full_eval_preds
    return final_preds


# EVOLVE-BLOCK-END
# =====================================================================
