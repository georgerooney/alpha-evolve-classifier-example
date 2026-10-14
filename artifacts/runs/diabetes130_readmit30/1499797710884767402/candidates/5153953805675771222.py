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
    """Fit on `train`/`y_train` and return readmission probabilities using Clinical Holographic Topics & Tri-GBDTs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import xgboost as xgb
    import pandas as pd

    non_med_cols = {
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    }
    med_cols = [col for col in task.categorical_columns if col not in non_med_cols]

    age_midpoints = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }

    def safe_str(val):
        if val is None or val == '?' or val == '':
            return '__missing__'
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        return str(code).strip().split('.')[0]

    def get_icd9_subchapter(code):
        if code is None or code == '?' or code == '' or code == 'missing':
            return 'missing'
        code_str = str(code).strip()
        if not code_str:
            return 'missing'
        first_char = code_str[0].upper()
        if first_char == 'V':
            return 'v_codes'
        if first_char == 'E':
            return 'e_codes'
        digits = []
        for char in code_str:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if not digits:
            return 'unknown'
        try:
            val = int("".join(digits[:3]))
            if 1 <= val <= 139: return 'infectious'
            elif 140 <= val <= 239: return 'neoplasms'
            elif 240 <= val <= 279: return 'endocrine'
            elif 280 <= val <= 289: return 'blood'
            elif 290 <= val <= 319: return 'mental'
            elif 320 <= val <= 389: return 'nervous'
            elif 390 <= val <= 459: return 'circulatory'
            elif 460 <= val <= 519: return 'respiratory'
            elif 520 <= val <= 579: return 'digestive'
            elif 580 <= val <= 629: return 'genitourinary'
            elif 630 <= val <= 679: return 'pregnancy'
            elif 680 <= val <= 709: return 'skin'
            elif 710 <= val <= 739: return 'musculoskeletal'
            elif 740 <= val <= 759: return 'congenital'
            elif 760 <= val <= 779: return 'perinatal'
            elif 780 <= val <= 799: return 'symptoms'
            elif 800 <= val <= 999: return 'injury'
            else: return 'other'
        except ValueError:
            return 'unknown'

    # 1. Feature Engineering on both sets
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        # Utilization & clinical intensity metrics
        total_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['total_visits'] = total_v
        d['inpatient_ratio'] = d['number_inpatient'] / (total_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (total_v + 1.0)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)
        stay = d['time_in_hospital'] + 1.0
        d['med_per_day'] = d['num_medications'] / stay
        d['lab_per_day'] = d['num_lab_procedures'] / stay
        d['proc_per_day'] = d['num_procedures'] / stay
        d['diag_per_day'] = d['number_diagnoses'] / stay
        d['service_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / stay

        # Polypharmacy & Medication instability
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
        d['active_meds'] = active
        d['num_med_changes'] = changes

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['has_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Oral medication markers
        active_oral = np.zeros(n_rows, dtype=float)
        changes_oral = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col != 'insulin' and col in d:
                val = d[col]
                active_oral += (val != 'No').astype(float)
                changes_oral += np.isin(val, ['Up', 'Down']).astype(float)
        d['active_oral'] = active_oral
        d['changes_oral'] = changes_oral
        d['insulin_only'] = ((d['has_insulin'] > 0) & (active_oral == 0)).astype(float)
        d['oral_only'] = ((d['has_insulin'] == 0) & (active_oral > 0)).astype(float)
        d['comb_therapy'] = ((d['has_insulin'] > 0) & (active_oral > 0)).astype(float)

        # Discharge risk dispositions
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf_hh'] = np.isin(disp, ['3', '6', '22', '28']).astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)

        # Joint combo
        d['disp_src'] = np.array([f"{x}_{y}" for x, y in zip(d['discharge_disposition_id'], d['admission_source_id'])])

        # Diagnoses: groups, 3-digit categories, subchapters & clinical comorbidity profile
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)
            d[f'{col}_sub'] = np.array([get_icd9_subchapter(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_ckd = np.isin(c1, ['585', '586']) | np.isin(c2, ['585', '586']) | np.isin(c3, ['585', '586'])
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_diab_comp = np.zeros(n_rows, dtype=bool)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            has_diab_comp |= np.array([str(x).startswith('250') and str(x) not in ['250', '250.0', '250.00', '250.02'] for x in d[col]])

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_diab_comp'] = has_diab_comp.astype(float)
        comorb = d['has_chf'] + d['has_ckd'] + d['has_cad'] + d['has_htn'] + d['has_copd'] + d['has_diab_comp']
        d['comorbidity_count'] = comorb
        d['frailty_index'] = comorb * (d['age_num'] / 50.0) * (d['number_inpatient'] + 1.0)

        # Interaction features
        d['inpatient_by_disch_snf'] = d['number_inpatient'] * d['disch_is_snf_hh']
        d['inpatient_by_disch_home'] = d['number_inpatient'] * d['disch_is_home']
        d['inpatient_by_disch_ama'] = d['number_inpatient'] * d['disch_is_ama']
        d['inpatient_by_total_visits'] = d['number_inpatient'] * d['total_visits']
        d['emergency_by_total_visits'] = d['number_emergency'] * d['total_visits']
        d['comorbidity_by_total_visits'] = d['comorbidity_count'] * d['total_visits']
        d['age_by_comorbidity'] = d['age_num'] * d['comorbidity_count']
        d['age_by_inpatient'] = d['age_num'] * d['number_inpatient']
        d['med_by_comorbidity'] = d['num_medications'] * d['comorbidity_count']

    # 2. Bayesian Smoothed Target Encoding (Out-of-Fold within train)
    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tr_diag_max = np.zeros(n_tr, dtype=float)
    tr_diag_mean = np.zeros(n_tr, dtype=float)
    tr_diag_sum = np.zeros(n_tr, dtype=float)
    tr_diag_1 = np.zeros(n_tr, dtype=float)
    tr_diag_2 = np.zeros(n_tr, dtype=float)
    tr_diag_3 = np.zeros(n_tr, dtype=float)
    tr_disp = np.zeros(n_tr, dtype=float)
    tr_spec = np.zeros(n_tr, dtype=float)
    tr_src = np.zeros(n_tr, dtype=float)
    tr_disp_src = np.zeros(n_tr, dtype=float)
    tr_payer = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf.split(train['gender'], y_train):
        # Pool all 3 diagnoses into one unified clinical registry
        pooled_diag = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_y = np.concatenate([y_train[tr_idx], y_train[tr_idx], y_train[tr_idx]])
        d_map = compute_smoothed_map(pooled_diag, pooled_y, smooth=25.0, prior=prior_glob)

        d1 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        d2 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        d3 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        tr_diag_max[val_idx] = np.maximum(np.maximum(d1, d2), d3)
        tr_diag_mean[val_idx] = (d1 + d2 + d3) / 3.0
        tr_diag_sum[val_idx] = d1 + d2 + d3

        # Specific diagnosis maps
        map_d1 = compute_smoothed_map(train['diag_1_3d'][tr_idx], y_train[tr_idx], smooth=25.0, prior=prior_glob)
        map_d2 = compute_smoothed_map(train['diag_2_3d'][tr_idx], y_train[tr_idx], smooth=25.0, prior=prior_glob)
        map_d3 = compute_smoothed_map(train['diag_3_3d'][tr_idx], y_train[tr_idx], smooth=25.0, prior=prior_glob)

        tr_diag_1[val_idx] = np.array([map_d1.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        tr_diag_2[val_idx] = np.array([map_d2.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        tr_diag_3[val_idx] = np.array([map_d3.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])

        disp_map = compute_smoothed_map(train['discharge_disposition_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp[val_idx] = np.array([disp_map.get(safe_str(c), prior_glob) for c in train['discharge_disposition_id'][val_idx]])

        spec_map = compute_smoothed_map(train['medical_specialty'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_spec[val_idx] = np.array([spec_map.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

        src_map = compute_smoothed_map(train['admission_source_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_src[val_idx] = np.array([src_map.get(safe_str(c), prior_glob) for c in train['admission_source_id'][val_idx]])

        disp_src_map = compute_smoothed_map(train['disp_src'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp_src[val_idx] = np.array([disp_src_map.get(safe_str(c), prior_glob) for c in train['disp_src'][val_idx]])

        payer_map = compute_smoothed_map(train['payer_code'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_payer[val_idx] = np.array([payer_map.get(safe_str(c), prior_glob) for c in train['payer_code'][val_idx]])

    train['te_diag_max'] = tr_diag_max
    train['te_diag_mean'] = tr_diag_mean
    train['te_diag_sum'] = tr_diag_sum
    train['te_diag_1'] = tr_diag_1
    train['te_diag_2'] = tr_diag_2
    train['te_diag_3'] = tr_diag_3
    train['te_discharge'] = tr_disp
    train['te_specialty'] = tr_spec
    train['te_adm_source'] = tr_src
    train['te_disp_src'] = tr_disp_src
    train['te_payer'] = tr_payer

    # Full train-to-eval target mappings
    pooled_diag_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    d_map_all = compute_smoothed_map(pooled_diag_all, pooled_y_all, smooth=25.0, prior=prior_glob)
    ed1 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    ed2 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    ed3 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_max'] = np.maximum(np.maximum(ed1, ed2), ed3)
    eval_cols['te_diag_mean'] = (ed1 + ed2 + ed3) / 3.0
    eval_cols['te_diag_sum'] = ed1 + ed2 + ed3

    map_d1_all = compute_smoothed_map(train['diag_1_3d'], y_train, smooth=25.0, prior=prior_glob)
    map_d2_all = compute_smoothed_map(train['diag_2_3d'], y_train, smooth=25.0, prior=prior_glob)
    map_d3_all = compute_smoothed_map(train['diag_3_3d'], y_train, smooth=25.0, prior=prior_glob)

    eval_cols['te_diag_1'] = np.array([map_d1_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    eval_cols['te_diag_2'] = np.array([map_d2_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    eval_cols['te_diag_3'] = np.array([map_d3_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])

    disp_map_all = compute_smoothed_map(train['discharge_disposition_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_discharge'] = np.array([disp_map_all.get(safe_str(c), prior_glob) for c in eval_cols['discharge_disposition_id']])

    spec_map_all = compute_smoothed_map(train['medical_specialty'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([spec_map_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']])

    src_map_all = compute_smoothed_map(train['admission_source_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_source'] = np.array([src_map_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_source_id']])

    disp_src_map_all = compute_smoothed_map(train['disp_src'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_disp_src'] = np.array([disp_src_map_all.get(safe_str(c), prior_glob) for c in eval_cols['disp_src']])

    payer_map_all = compute_smoothed_map(train['payer_code'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_payer'] = np.array([payer_map_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_code']])

    # 3. Clinical Holographic Topic SVD
    def build_narrative_corpus(d):
        n = len(d['gender'])
        med_pairs = [(m, d[m]) for m in med_cols if m in d]
        c = []
        for i in range(n):
            w = [
                f"d1_{d['diag_1_3d'][i]}", f"d2_{d['diag_2_3d'][i]}", f"d3_{d['diag_3_3d'][i]}",
                f"g1_{d['diag_1_grp'][i]}", f"g2_{d['diag_2_grp'][i]}", f"g3_{d['diag_3_grp'][i]}",
                f"sub1_{d['diag_1_sub'][i]}", f"sub2_{d['diag_2_sub'][i]}", f"sub3_{d['diag_3_sub'][i]}",
                f"disp_{safe_str(d['discharge_disposition_id'][i])}",
                f"src_{safe_str(d['admission_source_id'][i])}",
                f"spec_{safe_str(d['medical_specialty'][i])}",
                f"age_{safe_str(d['age'][i])}",
            ]
            if d['number_inpatient'][i] > 0:
                w.append("prior_inp")
            if d['number_inpatient'][i] >= 2:
                w.append("freq_inp")
            if d['number_emergency'][i] > 0:
                w.append("prior_er")
            for m_name, m_vals in med_pairs:
                val = m_vals[i]
                if val is not None and val != 'No':
                    w.append(f"{m_name}_{val}")
            c.append(" ".join(w))
        return c

    tfidf = TfidfVectorizer(max_features=200, min_df=5, token_pattern=r'(?u)\b\S+\b')
    tr_tfidf = tfidf.fit_transform(build_narrative_corpus(train))
    ev_tfidf = tfidf.transform(build_narrative_corpus(eval_cols))

    svd = TruncatedSVD(n_components=10, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    topic_names = [f'topic_{k}' for k in range(10)]
    for k in range(10):
        train[f'topic_{k}'] = tr_svd[:, k].astype(float)
        eval_cols[f'topic_{k}'] = ev_svd[:, k].astype(float)

    interaction_cols = [
        'inpatient_by_disch_snf', 'inpatient_by_disch_home', 'inpatient_by_disch_ama',
        'inpatient_by_total_visits', 'emergency_by_total_visits', 'comorbidity_by_total_visits',
        'age_by_comorbidity', 'age_by_inpatient', 'med_by_comorbidity',
        'active_oral', 'changes_oral', 'insulin_only', 'oral_only', 'comb_therapy',
    ]

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'frequent_inpatient', 'med_per_day', 'lab_per_day',
        'proc_per_day', 'diag_per_day', 'service_intensity',
        'active_meds', 'num_med_changes', 'has_insulin', 'insulin_up', 'insulin_down',
        'change_flag', 'diabetes_med_flag',
        'disch_is_home', 'disch_is_snf_hh', 'disch_is_ama',
        'has_chf', 'has_ckd', 'has_cad', 'has_htn', 'has_copd', 'has_diab_comp',
        'comorbidity_count', 'frailty_index',
        'te_diag_max', 'te_diag_mean', 'te_diag_sum',
        'te_diag_1', 'te_diag_2', 'te_diag_3',
        'te_discharge', 'te_specialty', 'te_adm_source',
        'te_disp_src', 'te_payer',
    ] + interaction_cols + topic_names

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag_1_sub', 'diag_2_sub', 'diag_3_sub',
        'disp_src',
    ]

    # 4. Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # Convert encoded features to high-fidelity DataFrames for native pandas-category support in XGBoost
    df_train = pd.DataFrame(x_train)
    df_eval = pd.DataFrame(x_eval)
    for idx in encoder.categorical_indices:
        df_train[idx] = df_train[idx].astype('category')
        df_eval[idx] = df_eval[idx].astype('category')

    # 5. Tri-Engine GBDT Models
    # Model 1: LightGBM (Leaf-wise expansion)
    model_lgb = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=35,
        min_child_samples=55,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.8,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb = model_lgb.predict_proba(x_eval)[:, 1]

    # Model 2: HistGradientBoosting (Depth/leaf expansion with sklearn binning)
    model_hgb = HistGradientBoostingClassifier(
        max_iter=240,
        learning_rate=0.03,
        max_leaf_nodes=35,
        min_samples_leaf=55,
        l2_regularization=4.0,
        categorical_features=encoder.categorical_indices,
        random_state=42,
    )
    model_hgb.fit(x_train, y_train)
    p_hgb = model_hgb.predict_proba(x_eval)[:, 1]

    # Model 3: XGBoost (Depth-wise expansion with histogram splitting and native categorical splits)
    model_xgb = xgb.XGBClassifier(
        n_estimators=260,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.8,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=1,
        tree_method='hist',
        enable_categorical=True,
        eval_metric='logloss',
    )
    model_xgb.fit(df_train, y_train)
    p_xgb = model_xgb.predict_proba(df_eval)[:, 1]

    # Optimal calibrated blend
    pred = 0.40 * p_lgb + 0.35 * p_xgb + 0.25 * p_hgb
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
