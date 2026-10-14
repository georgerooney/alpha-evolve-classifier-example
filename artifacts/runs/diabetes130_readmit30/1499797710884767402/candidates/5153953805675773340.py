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
    from sklearn.model_selection import StratifiedKFold, KFold

    def map_icd9(code):
        if 'icd9_group' in globals():
            try:
                return globals()['icd9_group'](code)
            except Exception:
                pass
        if code is None or code == '?' or code == '' or code == 'None':
            return 'missing'
        s = str(code).strip()
        if not s or s == 'None':
            return 'missing'
        if s.startswith('V') or s.startswith('E'):
            return 'other'
        try:
            val = float(s)
        except ValueError:
            return 'other'
        if (390 <= val <= 459) or (int(val) == 785):
            return 'circulatory'
        if (460 <= val <= 519) or (int(val) == 786):
            return 'respiratory'
        if (520 <= val <= 579) or (int(val) == 787):
            return 'digestive'
        if 250 <= val < 251:
            return 'diabetes'
        if 800 <= val <= 999:
            return 'injury'
        if 710 <= val <= 739:
            return 'musculoskeletal'
        if (580 <= val <= 629) or (int(val) == 788):
            return 'genitourinary'
        if 140 <= val <= 239:
            return 'neoplasms'
        return 'other'

    def check_prefix_3(cols_tuple, prefixes):
        n = len(cols_tuple[0])
        out = np.zeros(n, dtype=np.float64)
        for c in cols_tuple:
            for i, val in enumerate(c):
                if val is not None and str(val).startswith(prefixes):
                    out[i] = 1.0
        return out

    def check_dm_comp(cols_tuple):
        n = len(cols_tuple[0])
        out = np.zeros(n, dtype=np.float64)
        for c in cols_tuple:
            for i, val in enumerate(c):
                if val is not None:
                    s = str(val).strip()
                    if s.startswith('250.') and not s.startswith('250.0'):
                        out[i] = 1.0
        return out

    def extract_features(d):
        n_rows = len(d['time_in_hospital'])
        num_features = {}
        cat_features = {}

        # 1. Demographics
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        num_features['age_num'] = np.array([age_map.get(str(a), 65.0) for a in d['age']], dtype=np.float64)

        # 2. Utilisation
        num_inp = np.nan_to_num(d['number_inpatient'], nan=0.0)
        num_emg = np.nan_to_num(d['number_emergency'], nan=0.0)
        num_out = np.nan_to_num(d['number_outpatient'], nan=0.0)

        total_visits = num_inp + num_emg + num_out
        num_features['total_visits'] = total_visits
        num_features['has_prior_inpatient'] = (num_inp > 0).astype(np.float64)
        num_features['has_prior_emergency'] = (num_emg > 0).astype(np.float64)
        num_features['has_prior_outpatient'] = (num_out > 0).astype(np.float64)
        num_features['inpatient_ratio'] = num_inp / (total_visits + 1.0)
        num_features['emergency_ratio'] = num_emg / (total_visits + 1.0)
        num_features['log_number_inpatient'] = np.log1p(np.maximum(num_inp, 0.0))
        num_features['log_total_visits'] = np.log1p(np.maximum(total_visits, 0.0))

        # 3. Hospital stay intensity
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 1.0)
        num_lab = np.nan_to_num(d['num_lab_procedures'], nan=0.0)
        num_proc = np.nan_to_num(d['num_procedures'], nan=0.0)
        num_med = np.nan_to_num(d['num_medications'], nan=0.0)
        num_diag = np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0)

        num_features['med_per_day'] = num_med / los
        num_features['lab_per_day'] = num_lab / los
        num_features['proc_per_day'] = num_proc / los
        num_features['diag_per_day'] = num_diag / los
        num_features['med_per_diag'] = num_med / num_diag
        num_features['total_procedures'] = num_lab + num_proc
        num_features['service_intensity'] = (num_lab + 5.0 * num_proc + 2.0 * num_med) / los
        num_features['inp_x_los'] = num_inp * los
        num_features['inp_x_diag'] = num_inp * num_diag

        # 4. Diagnoses and chronic conditions
        diag1_grp = np.array([map_icd9(c) for c in d['diag_1']], dtype=object)
        diag2_grp = np.array([map_icd9(c) for c in d['diag_2']], dtype=object)
        diag3_grp = np.array([map_icd9(c) for c in d['diag_3']], dtype=object)
        cat_features['diag_1_group'] = diag1_grp
        cat_features['diag_2_group'] = diag2_grp
        cat_features['diag_3_group'] = diag3_grp

        num_features['num_circ_diags'] = ((diag1_grp == 'circulatory').astype(np.float64) +
                                          (diag2_grp == 'circulatory').astype(np.float64) +
                                          (diag3_grp == 'circulatory').astype(np.float64))
        num_features['num_dm_diags'] = ((diag1_grp == 'diabetes').astype(np.float64) +
                                        (diag2_grp == 'diabetes').astype(np.float64) +
                                        (diag3_grp == 'diabetes').astype(np.float64))
        num_features['num_resp_diags'] = ((diag1_grp == 'respiratory').astype(np.float64) +
                                          (diag2_grp == 'respiratory').astype(np.float64) +
                                          (diag3_grp == 'respiratory').astype(np.float64))
        num_features['num_digest_diags'] = ((diag1_grp == 'digestive').astype(np.float64) +
                                            (diag2_grp == 'digestive').astype(np.float64) +
                                            (diag3_grp == 'digestive').astype(np.float64))
        num_features['num_gu_diags'] = ((diag1_grp == 'genitourinary').astype(np.float64) +
                                        (diag2_grp == 'genitourinary').astype(np.float64) +
                                        (diag3_grp == 'genitourinary').astype(np.float64))

        diag_tuple = (d['diag_1'], d['diag_2'], d['diag_3'])
        has_hf = check_prefix_3(diag_tuple, ('428',))
        has_ckd = check_prefix_3(diag_tuple, ('585', '586'))
        has_cad = check_prefix_3(diag_tuple, ('410', '411', '412', '413', '414'))
        has_htn = check_prefix_3(diag_tuple, ('401', '402', '403', '404', '405'))
        has_dm_comp = check_dm_comp(diag_tuple)

        num_features['has_hf'] = has_hf
        num_features['has_ckd'] = has_ckd
        num_features['has_cad'] = has_cad
        num_features['has_htn'] = has_htn
        num_features['has_dm_comp'] = has_dm_comp
        num_features['comorbid_score'] = has_hf + has_ckd + has_cad + has_htn + has_dm_comp

        # 5. Medications
        med_list = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'examide', 'citoglipton', 'insulin',
            'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        avail_meds = [m for m in med_list if m in d]

        meds_active = np.zeros(n_rows, dtype=np.float64)
        meds_up = np.zeros(n_rows, dtype=np.float64)
        meds_down = np.zeros(n_rows, dtype=np.float64)
        meds_changed = np.zeros(n_rows, dtype=np.float64)

        for m in avail_meds:
            arr = d[m]
            for i, v in enumerate(arr):
                if v is not None and v != 'No':
                    meds_active[i] += 1.0
                    if v == 'Up':
                        meds_up[i] += 1.0
                        meds_changed[i] += 1.0
                    elif v == 'Down':
                        meds_down[i] += 1.0
                        meds_changed[i] += 1.0

        num_features['meds_active'] = meds_active
        num_features['meds_up'] = meds_up
        num_features['meds_down'] = meds_down
        num_features['meds_changed'] = meds_changed

        insulin_arr = d['insulin'] if 'insulin' in d else np.array(['No'] * n_rows)
        num_features['insulin_steady'] = np.array([1.0 if v == 'Steady' else 0.0 for v in insulin_arr], dtype=np.float64)
        num_features['insulin_up'] = np.array([1.0 if v == 'Up' else 0.0 for v in insulin_arr], dtype=np.float64)
        num_features['insulin_down'] = np.array([1.0 if v == 'Down' else 0.0 for v in insulin_arr], dtype=np.float64)
        num_features['insulin_active'] = np.array([1.0 if v in ('Steady', 'Up', 'Down') else 0.0 for v in insulin_arr], dtype=np.float64)

        # 6. Admission / Discharge indicators
        disp = d.get('discharge_disposition_id', np.array([''] * n_rows))
        adm_type = d.get('admission_type_id', np.array([''] * n_rows))
        adm_src = d.get('admission_source_id', np.array([''] * n_rows))

        num_features['disch_home'] = np.array([1.0 if str(v) == '1' else 0.0 for v in disp], dtype=np.float64)
        num_features['disch_snf'] = np.array([1.0 if str(v) in ('3', '4') else 0.0 for v in disp], dtype=np.float64)
        num_features['disch_home_health'] = np.array([1.0 if str(v) == '6' else 0.0 for v in disp], dtype=np.float64)
        num_features['disch_facility'] = np.array([1.0 if str(v) in ('2', '3', '4', '5', '22', '28') else 0.0 for v in disp], dtype=np.float64)

        num_features['adm_emergency'] = np.array([1.0 if str(v) == '1' else 0.0 for v in adm_type], dtype=np.float64)
        num_features['adm_elective'] = np.array([1.0 if str(v) == '3' else 0.0 for v in adm_type], dtype=np.float64)
        num_features['adm_from_er'] = np.array([1.0 if str(v) == '7' else 0.0 for v in adm_src], dtype=np.float64)

        # 7. Lab results and general flags
        a1c = d.get('A1Cresult', np.array(['None'] * n_rows))
        num_features['a1c_tested'] = np.array([0.0 if v in ('None', None) else 1.0 for v in a1c], dtype=np.float64)
        num_features['a1c_high'] = np.array([1.0 if v in ('>7', '>8') else 0.0 for v in a1c], dtype=np.float64)

        glu = d.get('max_glu_serum', np.array(['None'] * n_rows))
        num_features['glu_tested'] = np.array([0.0 if v in ('None', None) else 1.0 for v in glu], dtype=np.float64)
        num_features['glu_high'] = np.array([1.0 if v in ('>200', '>300') else 0.0 for v in glu], dtype=np.float64)

        chg = d.get('change', np.array(['No'] * n_rows))
        num_features['has_change'] = np.array([1.0 if v == 'Ch' else 0.0 for v in chg], dtype=np.float64)

        dmed = d.get('diabetesMed', np.array(['No'] * n_rows))
        num_features['has_dmed'] = np.array([1.0 if v == 'Yes' else 0.0 for v in dmed], dtype=np.float64)

        num_features['payer_missing'] = np.array([1.0 if (v is None or v == '?') else 0.0 for v in d.get('payer_code', [])], dtype=np.float64)
        num_features['spec_missing'] = np.array([1.0 if (v is None or v == '?') else 0.0 for v in d.get('medical_specialty', [])], dtype=np.float64)

        return num_features, cat_features

    def compute_oof_target_encoding(train_arr, y_tr, eval_arr, smoothing=20.0, n_splits=5, seed=42):
        n_tr = len(train_arr)
        n_ev = len(eval_arr)
        global_mean = float(np.mean(y_tr))
        oof_te = np.full(n_tr, global_mean, dtype=np.float64)
        eval_te = np.full(n_ev, global_mean, dtype=np.float64)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fit_idx, val_idx in kf.split(train_arr):
            fit_data = train_arr[fit_idx]
            fit_y = y_tr[fit_idx]

            sums = {}
            counts = {}
            for c, y in zip(fit_data, fit_y):
                k = str(c)
                sums[k] = sums.get(k, 0.0) + y
                counts[k] = counts.get(k, 0) + 1

            for idx in val_idx:
                k = str(train_arr[idx])
                cnt = counts.get(k, 0)
                if cnt > 0:
                    oof_te[idx] = (sums[k] + smoothing * global_mean) / (cnt + smoothing)
                else:
                    oof_te[idx] = global_mean

        sums = {}
        counts = {}
        for c, y in zip(train_arr, y_tr):
            k = str(c)
            sums[k] = sums.get(k, 0.0) + y
            counts[k] = counts.get(k, 0) + 1

        for i in range(n_ev):
            k = str(eval_arr[i])
            cnt = counts.get(k, 0)
            if cnt > 0:
                eval_te[i] = (sums[k] + smoothing * global_mean) / (cnt + smoothing)
            else:
                eval_te[i] = global_mean

        return oof_te, eval_te

    # Extract engineered features
    train_num, train_cat = extract_features(train)
    eval_num, eval_cat = extract_features(eval_cols)

    # Compute out-of-fold target encodings for high-cardinality columns
    te_cols = [
        ('te_diag_1', 'diag_1'),
        ('te_diag_2', 'diag_2'),
        ('te_diag_3', 'diag_3'),
        ('te_spec', 'medical_specialty'),
        ('te_disch', 'discharge_disposition_id'),
        ('te_adm_src', 'admission_source_id'),
    ]
    for te_name, src_col in te_cols:
        tr_te, ev_te = compute_oof_target_encoding(train[src_col], y_train, eval_cols[src_col], smoothing=20.0)
        train_num[te_name] = tr_te
        eval_num[te_name] = ev_te

    # Build copies of train and eval dicts with added columns
    train_copy = dict(train)
    eval_copy = dict(eval_cols)

    for k, v in train_num.items():
        train_copy[k] = v
        eval_copy[k] = eval_num[k]

    for k, v in train_cat.items():
        train_copy[k] = v
        eval_copy[k] = eval_cat[k]

    extra_numeric = list(train_num.keys())
    extra_categorical = list(train_cat.keys())

    # Encode with TabularEncoder
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task, train_copy,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical
    )
    x_eval = encoder.transform(eval_copy)

    cat_indices = encoder.categorical_indices

    # 1. Model 1: 5-Fold Stratified LightGBM (shallow, well-regularised)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    p_cv = np.zeros(len(x_eval), dtype=np.float64)

    for train_idx, _ in skf.split(x_train, y_train):
        m1 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(x_train[train_idx], y_train[train_idx], categorical_feature=cat_indices)
        p_cv += m1.predict_proba(x_eval)[:, 1] / n_splits

    # 2. Model 2: Full-data LightGBM (medium depth, high feature subsampling)
    m2 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=25,
        max_depth=6,
        min_child_samples=40,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p_m2 = m2.predict_proba(x_eval)[:, 1]

    # 3. Model 3: Full-data LightGBM (deeper, lower learning rate)
    m3 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=20,
        max_depth=-1,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=456,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p_m3 = m3.predict_proba(x_eval)[:, 1]

    # 4. Model 4: Full-data LightGBM (stump-like shallow trees)
    m4 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.04,
        num_leaves=12,
        max_depth=4,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=4.0,
        random_state=789,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=cat_indices)
    p_m4 = m4.predict_proba(x_eval)[:, 1]

    # Weighted blend
    p_blend = 0.40 * p_cv + 0.25 * p_m2 + 0.20 * p_m3 + 0.15 * p_m4
    return np.clip(p_blend, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
