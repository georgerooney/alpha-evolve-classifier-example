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

    # Comorbidity flags
    num_feats['has_circulatory'] = ((d1_grp == 'circulatory') | (d2_grp == 'circulatory') | (d3_grp == 'circulatory')).astype(np.float64)
    num_feats['has_respiratory'] = ((d1_grp == 'respiratory') | (d2_grp == 'respiratory') | (d3_grp == 'respiratory')).astype(np.float64)
    num_feats['has_digestive'] = ((d1_grp == 'digestive') | (d2_grp == 'digestive') | (d3_grp == 'digestive')).astype(np.float64)
    num_feats['has_diabetes_diag'] = ((d1_grp == 'diabetes') | (d2_grp == 'diabetes') | (d3_grp == 'diabetes')).astype(np.float64)
    num_feats['has_injury'] = ((d1_grp == 'injury') | (d2_grp == 'injury') | (d3_grp == 'injury')).astype(np.float64)
    num_feats['has_genitourinary'] = ((d1_grp == 'genitourinary') | (d2_grp == 'genitourinary') | (d3_grp == 'genitourinary')).astype(np.float64)
    num_feats['has_neoplasms'] = ((d1_grp == 'neoplasms') | (d2_grp == 'neoplasms') | (d3_grp == 'neoplasms')).astype(np.float64)

    distinct_grps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        valid = {d1_grp[i], d2_grp[i], d3_grp[i]} - {'missing', 'other', None}
        distinct_grps[i] = float(len(valid))
    num_feats['num_distinct_diag_groups'] = distinct_grps

    # 2. Age as continuous midpoint & senior indicators
    age_raw = cols.get('age')
    age_num = np.empty(n, dtype=np.float64)
    if age_raw is not None:
        for i, a in enumerate(age_raw):
            age_num[i] = AGE_MAP.get(a, np.nan)
    else:
        age_num[:] = np.nan
    num_feats['age_num'] = age_num
    num_feats['is_elderly'] = (age_num >= 70.0).astype(np.float64)
    num_feats['is_very_old'] = (age_num >= 80.0).astype(np.float64)

    # 3. Medications intensity & dosing changes
    med_cols = [m for m in MED_COLS if m in cols]
    n_act = np.zeros(n, dtype=np.float64)
    n_up = np.zeros(n, dtype=np.float64)
    n_down = np.zeros(n, dtype=np.float64)
    n_steady = np.zeros(n, dtype=np.float64)
    n_chg = np.zeros(n, dtype=np.float64)

    for m in med_cols:
        arr = cols[m]
        for i, v in enumerate(arr):
            if v == 'Steady':
                n_act[i] += 1.0
                n_steady[i] += 1.0
            elif v == 'Up':
                n_act[i] += 1.0
                n_up[i] += 1.0
                n_chg[i] += 1.0
            elif v == 'Down':
                n_act[i] += 1.0
                n_down[i] += 1.0
                n_chg[i] += 1.0

    num_feats['n_active_meds'] = n_act
    num_feats['n_med_up'] = n_up
    num_feats['n_med_down'] = n_down
    num_feats['n_med_steady'] = n_steady
    num_feats['n_med_changes'] = n_chg

    insulin = cols.get('insulin')
    ins_presc = np.zeros(n, dtype=np.float64)
    ins_chg = np.zeros(n, dtype=np.float64)
    if insulin is not None:
        for i, v in enumerate(insulin):
            if v in ('Steady', 'Up', 'Down'):
                ins_presc[i] = 1.0
            if v in ('Up', 'Down'):
                ins_chg[i] = 1.0
    num_feats['ins_prescribed'] = ins_presc
    num_feats['ins_changed'] = ins_chg

    change_raw = cols.get('change')
    num_feats['change_flag'] = np.array([1.0 if v == 'Ch' else 0.0 for v in change_raw], dtype=np.float64) if change_raw is not None else np.zeros(n)
    dm_raw = cols.get('diabetesMed')
    num_feats['dm_flag'] = np.array([1.0 if v == 'Yes' else 0.0 for v in dm_raw], dtype=np.float64) if dm_raw is not None else np.zeros(n)

    # 4. Utilisation & Hospital Stay Intensity
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
    num_feats['has_prior_inp'] = (n_inp > 0.0).astype(np.float64)
    num_feats['has_prior_emg'] = (n_emg > 0.0).astype(np.float64)
    num_feats['has_prior_out'] = (n_out > 0.0).astype(np.float64)
    num_feats['has_any_prior'] = (total_prior > 0.0).astype(np.float64)
    num_feats['inp_severe'] = (n_inp >= 2.0).astype(np.float64)
    num_feats['inp_ratio'] = n_inp / (total_prior + 1.0)
    num_feats['emg_ratio'] = n_emg / (total_prior + 1.0)
    num_feats['prior_score'] = 3.0 * n_inp + 2.0 * n_emg + n_out

    num_feats['med_rate'] = n_med / t_hosp
    num_feats['lab_rate'] = n_lab / t_hosp
    num_feats['proc_rate'] = n_proc / t_hosp
    num_feats['diag_rate'] = n_diag / t_hosp
    total_proc = n_lab + n_proc
    num_feats['total_proc'] = total_proc
    num_feats['total_intensity'] = (n_med + total_proc) / t_hosp
    num_feats['diag_per_med'] = n_diag / (n_med + 1.0)
    num_feats['inp_to_time'] = n_inp / t_hosp
    num_feats['lab_to_med'] = n_lab / (n_med + 1.0)

    # 5. Lab results
    a1c = cols.get('A1Cresult')
    a1c_tested = np.zeros(n, dtype=np.float64)
    a1c_high = np.zeros(n, dtype=np.float64)
    if a1c is not None:
        for i, v in enumerate(a1c):
            if v is not None and v != 'None':
                a1c_tested[i] = 1.0
                if v in ('>7', '>8'):
                    a1c_high[i] = 1.0
    num_feats['a1c_tested'] = a1c_tested
    num_feats['a1c_high'] = a1c_high

    glu = cols.get('max_glu_serum')
    glu_tested = np.zeros(n, dtype=np.float64)
    glu_high = np.zeros(n, dtype=np.float64)
    if glu is not None:
        for i, v in enumerate(glu):
            if v is not None and v != 'None':
                glu_tested[i] = 1.0
                if v in ('>200', '>300'):
                    glu_high[i] = 1.0
    num_feats['glu_tested'] = glu_tested
    num_feats['glu_high'] = glu_high

    # 6. Admission / Discharge / Provider indicators
    disp = cols.get('discharge_disposition_id')
    dis_home = np.zeros(n, dtype=np.float64)
    dis_snf = np.zeros(n, dtype=np.float64)
    dis_hh = np.zeros(n, dtype=np.float64)
    if disp is not None:
        for i, v in enumerate(disp):
            if v == '1': dis_home[i] = 1.0
            elif v == '3': dis_snf[i] = 1.0
            elif v == '6': dis_hh[i] = 1.0
    num_feats['discharged_home'] = dis_home
    num_feats['discharged_snf'] = dis_snf
    num_feats['discharged_home_health'] = dis_hh

    adm_type = cols.get('admission_type_id')
    adm_emg = np.zeros(n, dtype=np.float64)
    adm_urg = np.zeros(n, dtype=np.float64)
    adm_elec = np.zeros(n, dtype=np.float64)
    if adm_type is not None:
        for i, v in enumerate(adm_type):
            if v == '1': adm_emg[i] = 1.0
            elif v == '2': adm_urg[i] = 1.0
            elif v == '3': adm_elec[i] = 1.0
    num_feats['adm_emergency'] = adm_emg
    num_feats['adm_urgent'] = adm_urg
    num_feats['adm_elective'] = adm_elec

    adm_src = cols.get('admission_source_id')
    adm_er = np.zeros(n, dtype=np.float64)
    adm_tr = np.zeros(n, dtype=np.float64)
    if adm_src is not None:
        for i, v in enumerate(adm_src):
            if v == '7': adm_er[i] = 1.0
            elif v in ('4', '5', '6', '10', '18', '22'): adm_tr[i] = 1.0
    num_feats['adm_from_er'] = adm_er
    num_feats['adm_transfer'] = adm_tr

    payer = cols.get('payer_code')
    num_feats['has_payer'] = np.array([1.0 if v is not None else 0.0 for v in payer], dtype=np.float64) if payer is not None else np.zeros(n)
    spec = cols.get('medical_specialty')
    num_feats['has_specialty'] = np.array([1.0 if v is not None else 0.0 for v in spec], dtype=np.float64) if spec is not None else np.zeros(n)

    return num_feats, cat_feats


def _compute_oof_te(
    train_cats: np.ndarray,
    y_train: np.ndarray,
    eval_cats: np.ndarray,
    smooth: float = 30.0,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding with shrinkage."""
    n_train = len(train_cats)
    n_eval = len(eval_cats)
    oof_train = np.empty(n_train, dtype=np.float64)
    oof_eval = np.empty(n_eval, dtype=np.float64)
    prior = float(np.mean(y_train))

    rng = np.random.RandomState(seed)
    indices = np.arange(n_train)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        trn_idx = np.setdiff1d(indices, val_idx)
        stats: Dict[object, List[int]] = {}
        for idx in trn_idx:
            c = train_cats[idx]
            if c is not None:
                if c not in stats:
                    stats[c] = [0, 0]
                stats[c][0] += 1
                stats[c][1] += y_train[idx]

        for idx in val_idx:
            c = train_cats[idx]
            if c is not None and c in stats:
                cnt, sy = stats[c]
                oof_train[idx] = (sy + smooth * prior) / (cnt + smooth)
            else:
                oof_train[idx] = prior

    full_stats: Dict[object, List[int]] = {}
    for idx in range(n_train):
        c = train_cats[idx]
        if c is not None:
            if c not in full_stats:
                full_stats[c] = [0, 0]
            full_stats[c][0] += 1
            full_stats[c][1] += y_train[idx]

    for idx in range(n_eval):
        c = eval_cats[idx]
        if c is not None and c in full_stats:
            cnt, sy = full_stats[c]
            oof_eval[idx] = (sy + smooth * prior) / (cnt + smooth)
        else:
            oof_eval[idx] = prior

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

    # 2. Target encodings & frequency features
    te_cols = [
        ('diag_1_prefix', 'te_diag1', 30.0),
        ('diag_2_prefix', 'te_diag2', 30.0),
        ('diag_3_prefix', 'te_diag3', 30.0),
        ('discharge_disposition_id', 'te_disp', 25.0),
        ('admission_source_id', 'te_adm_src', 25.0),
        ('medical_specialty', 'te_spec', 40.0),
    ]

    for col_name, te_name, smooth in te_cols:
        tr_te, ev_te = _compute_oof_te(train_c[col_name], y_train, eval_c[col_name], smooth=smooth)
        train_c[te_name] = tr_te
        eval_c[te_name] = ev_te
        tr_num[te_name] = tr_te

    # Diagnosis risk aggregates
    mean_te_tr = (train_c['te_diag1'] + train_c['te_diag2'] + train_c['te_diag3']) / 3.0
    mean_te_ev = (eval_c['te_diag1'] + eval_c['te_diag2'] + eval_c['te_diag3']) / 3.0
    max_te_tr = np.maximum(train_c['te_diag1'], np.maximum(train_c['te_diag2'], train_c['te_diag3']))
    max_te_ev = np.maximum(eval_c['te_diag1'], np.maximum(eval_c['te_diag2'], eval_c['te_diag3']))

    train_c['te_diag_mean'] = mean_te_tr
    eval_c['te_diag_mean'] = mean_te_ev
    train_c['te_diag_max'] = max_te_tr
    eval_c['te_diag_max'] = max_te_ev
    tr_num['te_diag_mean'] = mean_te_tr
    tr_num['te_diag_max'] = max_te_tr

    # Frequency encodings
    freq_cols = ['medical_specialty', 'diag_1_prefix', 'discharge_disposition_id', 'admission_source_id']
    n_tr = len(y_train)
    for col_name in freq_cols:
        f_name = f'freq_{col_name}'
        freq_map: Dict[object, float] = {}
        for val in train_c[col_name]:
            if val is not None:
                freq_map[val] = freq_map.get(val, 0.0) + 1.0 / n_tr
        train_c[f_name] = np.array([freq_map.get(v, 0.0) for v in train_c[col_name]], dtype=np.float64)
        eval_c[f_name] = np.array([freq_map.get(v, 0.0) for v in eval_c[col_name]], dtype=np.float64)
        tr_num[f_name] = train_c[f_name]

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

    # 4. Multi-model Bagged LightGBM Ensemble
    model_configs = [
        {
            'n_estimators': 320,
            'learning_rate': 0.03,
            'num_leaves': 20,
            'max_depth': 5,
            'min_child_samples': 40,
            'subsample': 0.75,
            'subsample_freq': 1,
            'colsample_bytree': 0.70,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 42,
        },
        {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'num_leaves': 28,
            'max_depth': 6,
            'min_child_samples': 50,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.60,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'cat_smooth': 20.0,
            'cat_l2': 10.0,
            'random_state': 123,
        },
        {
            'n_estimators': 340,
            'learning_rate': 0.025,
            'num_leaves': 16,
            'max_depth': 5,
            'min_child_samples': 60,
            'subsample': 0.70,
            'subsample_freq': 1,
            'colsample_bytree': 0.80,
            'reg_alpha': 0.2,
            'reg_lambda': 2.0,
            'cat_smooth': 30.0,
            'cat_l2': 15.0,
            'random_state': 456,
        },
        {
            'n_estimators': 280,
            'learning_rate': 0.035,
            'num_leaves': 24,
            'max_depth': 5,
            'min_child_samples': 45,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.65,
            'reg_alpha': 0.8,
            'reg_lambda': 4.0,
            'cat_smooth': 25.0,
            'cat_l2': 10.0,
            'random_state': 789,
        },
        {
            'n_estimators': 260,
            'learning_rate': 0.04,
            'num_leaves': 22,
            'max_depth': 6,
            'min_child_samples': 70,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.75,
            'reg_alpha': 1.5,
            'reg_lambda': 6.0,
            'cat_smooth': 20.0,
            'cat_l2': 10.0,
            'random_state': 999,
        },
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for cfg in model_configs:
        clf = lgb.LGBMClassifier(
            **cfg,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += clf.predict_proba(x_eval)[:, 1]

    preds /= len(model_configs)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
