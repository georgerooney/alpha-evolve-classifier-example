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


MED_COLS = (
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
)

AGE_MAP = {
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


def _extract_domain_features(cols: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract clinical, utilisation, intensity and demographic features."""
    n = len(cols['time_in_hospital'])
    num_feats: Dict[str, np.ndarray] = {}
    cat_feats: Dict[str, np.ndarray] = {}

    # 1. ICD-9 Chapter groups and 3-digit prefixes
    d1, d2, d3 = cols['diag_1'], cols['diag_2'], cols['diag_3']
    d1_grp = np.array([icd9_group(c) for c in d1], dtype=object)
    d2_grp = np.array([icd9_group(c) for c in d2], dtype=object)
    d3_grp = np.array([icd9_group(c) for c in d3], dtype=object)

    cat_feats['diag_1_group'] = d1_grp
    cat_feats['diag_2_group'] = d2_grp
    cat_feats['diag_3_group'] = d3_grp

    def get_prefix(codes):
        out = np.empty(n, dtype=object)
        for i, c in enumerate(codes):
            if c is not None and isinstance(c, str) and len(c) > 0:
                out[i] = c.split('.')[0]
            else:
                out[i] = None
        return out

    cat_feats['diag_1_prefix'] = get_prefix(d1)
    cat_feats['diag_2_prefix'] = get_prefix(d2)
    cat_feats['diag_3_prefix'] = get_prefix(d3)

    # Comorbidity chapter indicators & multivariable diagnosis complexity
    distinct_grps = np.zeros(n, dtype=np.float64)
    for grp in ('circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms'):
        flag = ((d1_grp == grp) | (d2_grp == grp) | (d3_grp == grp)).astype(np.float64)
        num_feats[f'has_{grp}'] = flag
        distinct_grps += flag
    num_feats['num_distinct_diag_groups'] = distinct_grps

    # Continuous age mapping
    age_raw = cols.get('age')
    num_feats['age_num'] = np.array([AGE_MAP.get(a, np.nan) for a in age_raw], dtype=np.float64) if age_raw is not None else np.full(n, np.nan)

    # Vectorized medication intensity & regimen adjustments
    n_up = np.zeros(n, dtype=np.float64)
    n_down = np.zeros(n, dtype=np.float64)
    n_steady = np.zeros(n, dtype=np.float64)
    for m in MED_COLS:
        if m in cols:
            arr = cols[m]
            n_steady += (arr == 'Steady').astype(np.float64)
            n_up += (arr == 'Up').astype(np.float64)
            n_down += (arr == 'Down').astype(np.float64)

    num_feats['n_active_meds'] = n_steady + n_up + n_down
    num_feats['n_med_up'] = n_up
    num_feats['n_med_down'] = n_down
    num_feats['n_med_steady'] = n_steady
    num_feats['n_med_changes'] = n_up + n_down

    insulin = cols.get('insulin')
    if insulin is not None:
        num_feats['ins_prescribed'] = ((insulin == 'Steady') | (insulin == 'Up') | (insulin == 'Down')).astype(np.float64)
        num_feats['ins_changed'] = ((insulin == 'Up') | (insulin == 'Down')).astype(np.float64)
    else:
        num_feats['ins_prescribed'] = np.zeros(n, dtype=np.float64)
        num_feats['ins_changed'] = np.zeros(n, dtype=np.float64)

    chg = cols.get('change')
    num_feats['change_flag'] = (chg == 'Ch').astype(np.float64) if chg is not None else np.zeros(n, dtype=np.float64)
    dm = cols.get('diabetesMed')
    num_feats['dm_flag'] = (dm == 'Yes').astype(np.float64) if dm is not None else np.zeros(n, dtype=np.float64)

    # Hospital utilisation and intensity continuous metrics
    t_hosp = np.maximum(cols['time_in_hospital'].astype(np.float64), 1.0)
    n_lab = cols['num_lab_procedures'].astype(np.float64)
    n_proc = cols['num_procedures'].astype(np.float64)
    n_med = cols['num_medications'].astype(np.float64)
    n_diag = cols['number_diagnoses'].astype(np.float64)
    n_out = cols['number_outpatient'].astype(np.float64)
    n_emg = cols['number_emergency'].astype(np.float64)
    n_inp = cols['number_inpatient'].astype(np.float64)

    total_prior = n_out + n_emg + n_inp
    num_feats['total_prior_visits'] = total_prior
    num_feats['inp_ratio'] = n_inp / (total_prior + 1.0)
    num_feats['emg_ratio'] = n_emg / (total_prior + 1.0)
    num_feats['prior_score'] = 3.0 * n_inp + 2.0 * n_emg + n_out

    num_feats['med_rate'] = n_med / t_hosp
    num_feats['lab_rate'] = n_lab / t_hosp
    num_feats['proc_rate'] = n_proc / t_hosp
    num_feats['diag_rate'] = n_diag / t_hosp
    num_feats['total_intensity'] = (n_med + n_lab + n_proc) / t_hosp
    num_feats['diag_per_med'] = n_diag / (n_med + 1.0)
    num_feats['inp_to_time'] = n_inp / t_hosp
    num_feats['lab_to_med'] = n_lab / (n_med + 1.0)

    # Lab indicators
    a1c = cols.get('A1Cresult')
    if a1c is not None:
        num_feats['a1c_tested'] = ((a1c != 'None') & (a1c != None)).astype(np.float64)
        num_feats['a1c_high'] = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
    else:
        num_feats['a1c_tested'] = np.zeros(n, dtype=np.float64)
        num_feats['a1c_high'] = np.zeros(n, dtype=np.float64)

    glu = cols.get('max_glu_serum')
    if glu is not None:
        num_feats['glu_tested'] = ((glu != 'None') & (glu != None)).astype(np.float64)
        num_feats['glu_high'] = ((glu == '>200') | (glu == '>300')).astype(np.float64)
    else:
        num_feats['glu_tested'] = np.zeros(n, dtype=np.float64)
        num_feats['glu_high'] = np.zeros(n, dtype=np.float64)

    payer = cols.get('payer_code')
    num_feats['has_payer'] = (payer != None).astype(np.float64) if payer is not None else np.zeros(n, dtype=np.float64)
    spec = cols.get('medical_specialty')
    num_feats['has_specialty'] = (spec != None).astype(np.float64) if spec is not None else np.zeros(n, dtype=np.float64)

    return num_feats, cat_feats


def _compute_pooled_diag_te(
    train_p1: np.ndarray, train_p2: np.ndarray, train_p3: np.ndarray,
    eval_p1: np.ndarray, eval_p2: np.ndarray, eval_p3: np.ndarray,
    y_train: np.ndarray, folds: List[np.ndarray], smooth: float = 25.0
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pooled low-rank target encoding across symmetric diagnosis positions."""
    n_train = len(train_p1)
    oof_1 = np.empty(n_train, dtype=np.float64)
    oof_2 = np.empty(n_train, dtype=np.float64)
    oof_3 = np.empty(n_train, dtype=np.float64)
    global_prior = float(np.mean(y_train))

    for val_idx in folds:
        trn_mask = np.ones(n_train, dtype=bool)
        trn_mask[val_idx] = False
        trn_idx = np.flatnonzero(trn_mask)
        prior = float(np.mean(y_train[trn_idx]))

        counts: Dict[str, List[float]] = {}
        for idx in trn_idx:
            y = float(y_train[idx])
            for c in {train_p1[idx], train_p2[idx], train_p3[idx]} - {None}:
                if c not in counts:
                    counts[c] = [0.0, 0.0]
                counts[c][0] += 1.0
                counts[c][1] += y

        def _lookup(c, pr=prior, d=counts):
            if c is not None and c in d:
                cnt, sy = d[c]
                return (sy + smooth * pr) / (cnt + smooth)
            return pr

        for idx in val_idx:
            oof_1[idx] = _lookup(train_p1[idx])
            oof_2[idx] = _lookup(train_p2[idx])
            oof_3[idx] = _lookup(train_p3[idx])

    full_counts: Dict[str, List[float]] = {}
    for idx in range(n_train):
        y = float(y_train[idx])
        for c in {train_p1[idx], train_p2[idx], train_p3[idx]} - {None}:
            if c not in full_counts:
                full_counts[c] = [0.0, 0.0]
            full_counts[c][0] += 1.0
            full_counts[c][1] += y

    def _lookup_full(c):
        if c is not None and c in full_counts:
            cnt, sy = full_counts[c]
            return (sy + smooth * global_prior) / (cnt + smooth)
        return global_prior

    ev_1 = np.array([_lookup_full(c) for c in eval_p1], dtype=np.float64)
    ev_2 = np.array([_lookup_full(c) for c in eval_p2], dtype=np.float64)
    ev_3 = np.array([_lookup_full(c) for c in eval_p3], dtype=np.float64)

    return (oof_1, oof_2, oof_3), (ev_1, ev_2, ev_3)


def _compute_col_te(
    train_col: np.ndarray, eval_col: np.ndarray,
    y_train: np.ndarray, folds: List[np.ndarray], smooth: float = 25.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Synchronized single-column out-of-fold target encoding."""
    n_train = len(train_col)
    oof_train = np.empty(n_train, dtype=np.float64)
    global_prior = float(np.mean(y_train))

    for val_idx in folds:
        trn_mask = np.ones(n_train, dtype=bool)
        trn_mask[val_idx] = False
        trn_idx = np.flatnonzero(trn_mask)
        prior = float(np.mean(y_train[trn_idx]))

        counts: Dict[object, List[float]] = {}
        for idx in trn_idx:
            c = train_col[idx]
            if c is not None:
                if c not in counts:
                    counts[c] = [0.0, 0.0]
                counts[c][0] += 1.0
                counts[c][1] += float(y_train[idx])

        for idx in val_idx:
            c = train_col[idx]
            if c is not None and c in counts:
                cnt, sy = counts[c]
                oof_train[idx] = (sy + smooth * prior) / (cnt + smooth)
            else:
                oof_train[idx] = prior

    full_counts: Dict[object, List[float]] = {}
    for idx in range(n_train):
        c = train_col[idx]
        if c is not None:
            if c not in full_counts:
                full_counts[c] = [0.0, 0.0]
            full_counts[c][0] += 1.0
            full_counts[c][1] += float(y_train[idx])

    oof_eval = np.empty(len(eval_col), dtype=np.float64)
    for idx in range(len(eval_col)):
        c = eval_col[idx]
        if c is not None and c in full_counts:
            cnt, sy = full_counts[c]
            oof_eval[idx] = (sy + smooth * global_prior) / (cnt + smooth)
        else:
            oof_eval[idx] = global_prior

    return oof_train, oof_eval


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit engineered gradient-boosted ensemble and return readmission probabilities."""
    train_c = dict(train)
    eval_c = dict(eval_cols)

    # 1. Extract domain features
    tr_num, tr_cat = _extract_domain_features(train_c)
    ev_num, ev_cat = _extract_domain_features(eval_c)

    for k, v in tr_num.items():
        train_c[k] = v
        eval_c[k] = ev_num[k]

    for k, v in tr_cat.items():
        train_c[k] = v
        eval_c[k] = ev_cat[k]

    # 2. Synchronized K-fold target encodings
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(y_train))
    folds = np.array_split(indices, 5)

    # Pooled diagnosis risk estimation
    (tr_te1, tr_te2, tr_te3), (ev_te1, ev_te2, ev_te3) = _compute_pooled_diag_te(
        train_c['diag_1_prefix'], train_c['diag_2_prefix'], train_c['diag_3_prefix'],
        eval_c['diag_1_prefix'], eval_c['diag_2_prefix'], eval_c['diag_3_prefix'],
        y_train, folds, smooth=25.0
    )
    for name, tr_arr, ev_arr in [
        ('te_diag1', tr_te1, ev_te1),
        ('te_diag2', tr_te2, ev_te2),
        ('te_diag3', tr_te3, ev_te3),
        ('te_diag_max', np.maximum(tr_te1, np.maximum(tr_te2, tr_te3)), np.maximum(ev_te1, np.maximum(ev_te2, ev_te3))),
        ('te_diag_mean', (tr_te1 + tr_te2 + tr_te3) / 3.0, (ev_te1 + ev_te2 + ev_te3) / 3.0),
    ]:
        train_c[name] = tr_arr
        eval_c[name] = ev_arr
        tr_num[name] = tr_arr

    # Key categorical target encodings
    single_te_cols = [
        ('discharge_disposition_id', 'te_disp', 20.0),
        ('admission_source_id', 'te_adm_src', 25.0),
        ('admission_type_id', 'te_adm_type', 25.0),
        ('medical_specialty', 'te_spec', 35.0),
    ]
    for col_name, te_name, smooth in single_te_cols:
        tr_te, ev_te = _compute_col_te(train_c[col_name], eval_c[col_name], y_train, folds, smooth=smooth)
        train_c[te_name] = tr_te
        eval_c[te_name] = ev_te
        tr_num[te_name] = tr_te

    extra_numeric = list(tr_num.keys())
    extra_categorical = list(tr_cat.keys())

    # 3. Encode tabular features
    encoder = TabularEncoder(min_count=15)
    x_train = encoder.fit_transform(
        task,
        train_c,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    x_eval = encoder.transform(eval_c)

    # 4. Multi-scale regularized LightGBM ensemble (shallow, balanced, deep)
    model_configs = [
        {
            'n_estimators': 360,
            'learning_rate': 0.025,
            'num_leaves': 22,
            'max_depth': 5,
            'min_child_samples': 40,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.70,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 42,
        },
        {
            'n_estimators': 400,
            'learning_rate': 0.02,
            'num_leaves': 16,
            'max_depth': 4,
            'min_child_samples': 50,
            'subsample': 0.75,
            'subsample_freq': 1,
            'colsample_bytree': 0.65,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'cat_smooth': 30.0,
            'cat_l2': 15.0,
            'random_state': 123,
        },
        {
            'n_estimators': 320,
            'learning_rate': 0.03,
            'num_leaves': 28,
            'max_depth': 6,
            'min_child_samples': 60,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.3,
            'reg_lambda': 3.0,
            'cat_smooth': 20.0,
            'cat_l2': 10.0,
            'random_state': 456,
        },
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for cfg in model_configs:
        clf = lgb.LGBMClassifier(**cfg, n_jobs=1, verbose=-1)
        clf.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += clf.predict_proba(x_eval)[:, 1]

    preds /= len(model_configs)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
