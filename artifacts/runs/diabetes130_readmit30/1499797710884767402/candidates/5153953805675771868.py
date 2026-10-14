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
    from sklearn.model_selection import StratifiedKFold

    # 1. Prepare shallow copies of dictionary to register derived features
    tr = dict(train)
    ev = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    def register_num(name: str, arr_tr: np.ndarray, arr_ev: np.ndarray):
        tr[name] = arr_tr.astype(np.float64)
        ev[name] = arr_ev.astype(np.float64)
        extra_numeric.append(name)

    def register_cat(name: str, arr_tr: np.ndarray, arr_ev: np.ndarray):
        tr[name] = arr_tr.astype(object)
        ev[name] = arr_ev.astype(object)
        extra_categorical.append(name)

    n_train = len(y_train)
    n_eval = len(eval_cols['time_in_hospital'])

    # 2. Age parsing into numeric midpoint
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0
    }
    def parse_age(age_arr):
        return np.array([age_map.get(str(x), 65.0) for x in age_arr], dtype=np.float64)

    age_tr = parse_age(train['age'])
    age_ev = parse_age(eval_cols['age'])
    register_num('age_num', age_tr, age_ev)
    register_num('is_elderly', (age_tr >= 65.0).astype(np.float64), (age_ev >= 65.0).astype(np.float64))

    # 3. ICD-9 Chapter Groupings and Clinical Comorbidities
    def get_diag_groups(codes):
        res = []
        for c in codes:
            if c is None or (isinstance(c, float) and np.isnan(c)):
                res.append('missing')
            else:
                res.append(icd9_group(str(c)))
        return np.array(res, dtype=object)

    d1_tr, d1_ev = get_diag_groups(train['diag_1']), get_diag_groups(eval_cols['diag_1'])
    d2_tr, d2_ev = get_diag_groups(train['diag_2']), get_diag_groups(eval_cols['diag_2'])
    d3_tr, d3_ev = get_diag_groups(train['diag_3']), get_diag_groups(eval_cols['diag_3'])

    register_cat('diag_1_group', d1_tr, d1_ev)
    register_cat('diag_2_group', d2_tr, d2_ev)
    register_cat('diag_3_group', d3_tr, d3_ev)

    def extract_diag_flags(d1, d2, d3):
        circ = ((d1 == 'circulatory') | (d2 == 'circulatory') | (d3 == 'circulatory')).astype(np.float64)
        diab = ((d1 == 'diabetes') | (d2 == 'diabetes') | (d3 == 'diabetes')).astype(np.float64)
        resp = ((d1 == 'respiratory') | (d2 == 'respiratory') | (d3 == 'respiratory')).astype(np.float64)
        genit = ((d1 == 'genitourinary') | (d2 == 'genitourinary') | (d3 == 'genitourinary')).astype(np.float64)
        digest = ((d1 == 'digestive') | (d2 == 'digestive') | (d3 == 'digestive')).astype(np.float64)
        injury = ((d1 == 'injury') | (d2 == 'injury') | (d3 == 'injury')).astype(np.float64)
        neoplasm = ((d1 == 'neoplasms') | (d2 == 'neoplasms') | (d3 == 'neoplasms')).astype(np.float64)
        primary_diab = (d1 == 'diabetes').astype(np.float64)
        primary_circ = (d1 == 'circulatory').astype(np.float64)
        comorb = np.array([len({g1, g2, g3} - {'missing'}) for g1, g2, g3 in zip(d1, d2, d3)], dtype=np.float64)
        return circ, diab, resp, genit, digest, injury, neoplasm, primary_diab, primary_circ, comorb

    c_tr, db_tr, r_tr, g_tr, dg_tr, inj_tr, neo_tr, pd_tr, pc_tr, cm_tr = extract_diag_flags(d1_tr, d2_tr, d3_tr)
    c_ev, db_ev, r_ev, g_ev, dg_ev, inj_ev, neo_ev, pd_ev, pc_ev, cm_ev = extract_diag_flags(d1_ev, d2_ev, d3_ev)

    register_num('has_diag_circulatory', c_tr, c_ev)
    register_num('has_diag_diabetes', db_tr, db_ev)
    register_num('has_diag_respiratory', r_tr, r_ev)
    register_num('has_diag_genitourinary', g_tr, g_ev)
    register_num('has_diag_digestive', dg_tr, dg_ev)
    register_num('has_diag_injury', inj_tr, inj_ev)
    register_num('has_diag_neoplasms', neo_tr, neo_ev)
    register_num('primary_is_diabetes', pd_tr, pd_ev)
    register_num('primary_is_circulatory', pc_tr, pc_ev)
    register_num('diag_comorbidity_count', cm_tr, cm_ev)

    # 4. Prior Utilization and Resource Intensity
    def extract_utilization(cols):
        inp = np.nan_to_num(cols['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(cols['number_emergency'], nan=0.0)
        out = np.nan_to_num(cols['number_outpatient'], nan=0.0)
        los = np.maximum(np.nan_to_num(cols['time_in_hospital'], nan=1.0), 1.0)
        meds = np.nan_to_num(cols['num_medications'], nan=0.0)
        labs = np.nan_to_num(cols['num_lab_procedures'], nan=0.0)
        procs = np.nan_to_num(cols['num_procedures'], nan=0.0)
        diags = np.maximum(np.nan_to_num(cols['number_diagnoses'], nan=1.0), 1.0)

        tot_prior = inp + emg + out
        has_inp = (inp > 0).astype(np.float64)
        has_emg = (emg > 0).astype(np.float64)
        has_out = (out > 0).astype(np.float64)
        high_inp = (inp >= 2.0).astype(np.float64)
        inp_ratio = inp / (tot_prior + 1.0)
        emg_ratio = emg / (tot_prior + 1.0)
        acute_prior = inp + emg

        meds_per_day = meds / los
        labs_per_day = labs / los
        procs_per_day = procs / los
        meds_per_diag = meds / diags
        labs_per_diag = labs / diags
        service_density = (meds + labs + procs) / los

        inp_x_los = inp * los
        inp_x_diag = inp * diags
        inp_x_meds = inp * meds
        emg_x_inp = emg * inp
        labs_per_med = labs / (meds + 1.0)
        procs_per_med = procs / (meds + 1.0)

        return {
            'total_prior_visits': tot_prior,
            'has_prior_inpatient': has_inp,
            'has_prior_emergency': has_emg,
            'has_prior_outpatient': has_out,
            'high_inpatient': high_inp,
            'inpatient_ratio': inp_ratio,
            'emergency_ratio': emg_ratio,
            'acute_prior': acute_prior,
            'meds_per_day': meds_per_day,
            'labs_per_day': labs_per_day,
            'procs_per_day': procs_per_day,
            'meds_per_diag': meds_per_diag,
            'labs_per_diag': labs_per_diag,
            'service_density': service_density,
            'inpatient_x_los': inp_x_los,
            'inpatient_x_diag': inp_x_diag,
            'inpatient_x_meds': inp_x_meds,
            'emergency_x_inpatient': emg_x_inp,
            'labs_per_med': labs_per_med,
            'procs_per_med': procs_per_med,
        }

    util_tr = extract_utilization(train)
    util_ev = extract_utilization(eval_cols)
    for k in util_tr:
        register_num(k, util_tr[k], util_ev[k])

    register_num('inpatient_x_age', util_tr['has_prior_inpatient'] * age_tr, util_ev['has_prior_inpatient'] * age_ev)

    meds_tr = np.nan_to_num(train.get('num_medications', 0.0), nan=0.0)
    meds_ev = np.nan_to_num(eval_cols.get('num_medications', 0.0), nan=0.0)
    register_num('polypharmacy_risk', meds_tr * (age_tr >= 65.0), meds_ev * (age_ev >= 65.0))
    register_num('very_elderly_x_comorb', (age_tr >= 85.0).astype(np.float64) * cm_tr, (age_ev >= 85.0).astype(np.float64) * cm_ev)
    register_num('very_elderly_x_meds', (age_tr >= 85.0).astype(np.float64) * meds_tr, (age_ev >= 85.0).astype(np.float64) * meds_ev)

    # 5. Medication and Lab Dynamics
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]

    def extract_meds_and_labs(cols, n_rows):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        change_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in cols:
                arr = cols[m]
                is_up = (arr == 'Up')
                is_down = (arr == 'Down')
                is_steady = (arr == 'Steady')
                active_cnt += (is_up | is_down | is_steady).astype(np.float64)
                change_cnt += (is_up | is_down).astype(np.float64)
                up_cnt += is_up.astype(np.float64)
                down_cnt += is_down.astype(np.float64)

        ins = cols.get('insulin')
        ins_status = np.zeros(n_rows, dtype=np.float64)
        if ins is not None:
            ins_status[ins == 'Steady'] = 1.0
            ins_status[ins == 'Down'] = 2.0
            ins_status[ins == 'Up'] = 3.0
            ins_active = (ins_status > 0).astype(np.float64)
            ins_changed = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        else:
            ins_active = np.zeros(n_rows, dtype=np.float64)
            ins_changed = np.zeros(n_rows, dtype=np.float64)

        ch = cols.get('change')
        ch_flag = (ch == 'Ch').astype(np.float64) if ch is not None else np.zeros(n_rows, dtype=np.float64)

        dm = cols.get('diabetesMed')
        dm_flag = (dm == 'Yes').astype(np.float64) if dm is not None else np.zeros(n_rows, dtype=np.float64)

        a1c = cols.get('A1Cresult')
        if a1c is not None:
            a1c_tested = np.array([1.0 if x not in ('None', None) and not (isinstance(x, float) and np.isnan(x)) else 0.0 for x in a1c], dtype=np.float64)
            a1c_high = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        else:
            a1c_tested = np.zeros(n_rows, dtype=np.float64)
            a1c_high = np.zeros(n_rows, dtype=np.float64)

        glu = cols.get('max_glu_serum')
        if glu is not None:
            glu_tested = np.array([1.0 if x not in ('None', None) and not (isinstance(x, float) and np.isnan(x)) else 0.0 for x in glu], dtype=np.float64)
            glu_high = np.isin(glu, ['>200', '>300']).astype(np.float64)
        else:
            glu_tested = np.zeros(n_rows, dtype=np.float64)
            glu_high = np.zeros(n_rows, dtype=np.float64)

        return {
            'num_active_meds': active_cnt,
            'num_med_changes': change_cnt,
            'num_med_up': up_cnt,
            'num_med_down': down_cnt,
            'insulin_status': ins_status,
            'insulin_active': ins_active,
            'insulin_changed': ins_changed,
            'change_flag': ch_flag,
            'diabetes_med_flag': dm_flag,
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'glu_tested': glu_tested,
            'glu_high': glu_high,
        }

    med_tr = extract_meds_and_labs(train, n_train)
    med_ev = extract_meds_and_labs(eval_cols, n_eval)
    for k in med_tr:
        register_num(k, med_tr[k], med_ev[k])

    # 6. Admission, Discharge, and Missingness Indicators
    def extract_context(cols, n_rows):
        disch = cols.get('discharge_disposition_id')
        if disch is not None:
            disch_home = (disch == '1').astype(np.float64)
            disch_snf = (disch == '3').astype(np.float64)
            disch_hh = (disch == '6').astype(np.float64)
            disch_fac = np.isin(disch, ['2', '4', '5', '22', '28']).astype(np.float64)
            disch_ama = (disch == '7').astype(np.float64)
        else:
            disch_home = np.zeros(n_rows, dtype=np.float64)
            disch_snf = np.zeros(n_rows, dtype=np.float64)
            disch_hh = np.zeros(n_rows, dtype=np.float64)
            disch_fac = np.zeros(n_rows, dtype=np.float64)
            disch_ama = np.zeros(n_rows, dtype=np.float64)

        src = cols.get('admission_source_id')
        adm_er = (src == '7').astype(np.float64) if src is not None else np.zeros(n_rows, dtype=np.float64)

        adm_type = cols.get('admission_type_id')
        adm_emerg = (adm_type == '1').astype(np.float64) if adm_type is not None else np.zeros(n_rows, dtype=np.float64)

        spec = cols.get('medical_specialty')
        missing_spec = np.array([1.0 if x is None or str(x).strip() in ('?', '', 'None') else 0.0 for x in (spec if spec is not None else [None]*n_rows)], dtype=np.float64)

        payer = cols.get('payer_code')
        missing_payer = np.array([1.0 if x is None or str(x).strip() in ('?', '', 'None') else 0.0 for x in (payer if payer is not None else [None]*n_rows)], dtype=np.float64)

        return {
            'disch_home': disch_home,
            'disch_snf': disch_snf,
            'disch_home_health': disch_hh,
            'disch_other_facility': disch_fac,
            'disch_ama': disch_ama,
            'adm_er': adm_er,
            'adm_emergency_type': adm_emerg,
            'missing_specialty': missing_spec,
            'missing_payer': missing_payer,
        }

    ctx_tr = extract_context(train, n_train)
    ctx_ev = extract_context(eval_cols, n_eval)
    for k in ctx_tr:
        register_num(k, ctx_tr[k], ctx_ev[k])

    register_num('very_elderly_x_disch_home', (age_tr >= 85.0).astype(np.float64) * ctx_tr['disch_home'], (age_ev >= 85.0).astype(np.float64) * ctx_ev['disch_home'])
    register_num('disch_home_x_inpatient', ctx_tr['disch_home'] * util_tr['inpatient_ratio'], ctx_ev['disch_home'] * util_ev['inpatient_ratio'])
    register_num('er_admission_combo', ctx_tr['adm_er'] * ctx_tr['adm_emergency_type'], ctx_ev['adm_er'] * ctx_ev['adm_emergency_type'])

    # 6.5 Frequency Encoding
    def freq_encode(train_arr, eval_arr):
        def clean_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "__MISSING__"
            s = str(v).strip()
            return s if s and s != "?" else "__MISSING__"

        tr_clean = [clean_val(x) for x in train_arr]
        ev_clean = [clean_val(x) for x in eval_arr]

        from collections import Counter
        counts = Counter(tr_clean)
        n = len(tr_clean)
        
        tr_freq = np.array([counts[x] / n for x in tr_clean], dtype=np.float64)
        ev_freq = np.array([counts[x] / n for x in ev_clean], dtype=np.float64)
        return tr_freq, ev_freq

    for col_name in ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id']:
        f_tr, f_ev = freq_encode(train[col_name], eval_cols[col_name])
        register_num(f'freq_{col_name}', f_tr, f_ev)

    # 7. Out-of-fold Target Encoding with Bayesian Smoothing
    def target_encode(train_arr, y_arr, eval_arr, n_splits=5, smoothing=20.0, random_state=42):
        global_mean = float(np.mean(y_arr))
        def clean_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "__MISSING__"
            s = str(v).strip()
            return s if s and s != "?" else "__MISSING__"

        tr_clean = [clean_val(x) for x in train_arr]
        ev_clean = [clean_val(x) for x in eval_arr]

        train_encoded = np.full(len(train_arr), global_mean, dtype=np.float64)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for tr_idx, val_idx in skf.split(np.zeros(len(y_arr)), y_arr):
            counts = {}
            sums = {}
            for idx in tr_idx:
                v = tr_clean[idx]
                counts[v] = counts.get(v, 0) + 1
                sums[v] = sums.get(v, 0.0) + y_arr[idx]

            for idx in val_idx:
                v = tr_clean[idx]
                cnt = counts.get(v, 0)
                s = sums.get(v, 0.0)
                train_encoded[idx] = (s + smoothing * global_mean) / (cnt + smoothing)

        counts = {}
        sums = {}
        for v, y in zip(tr_clean, y_arr):
            counts[v] = counts.get(v, 0) + 1
            sums[v] = sums.get(v, 0.0) + y

        eval_encoded = np.array([
            (sums.get(v, 0.0) + smoothing * global_mean) / (counts.get(v, 0) + smoothing)
            for v in ev_clean
        ], dtype=np.float64)

        return train_encoded, eval_encoded

    te_cols = [
        ('diag_1', train['diag_1'], eval_cols['diag_1'], 25.0),
        ('diag_2', train['diag_2'], eval_cols['diag_2'], 25.0),
        ('diag_3', train['diag_3'], eval_cols['diag_3'], 25.0),
        ('discharge_disposition_id', train['discharge_disposition_id'], eval_cols['discharge_disposition_id'], 15.0),
        ('admission_source_id', train['admission_source_id'], eval_cols['admission_source_id'], 15.0),
        ('medical_specialty', train['medical_specialty'], eval_cols['medical_specialty'], 20.0),
        ('payer_code', train['payer_code'], eval_cols['payer_code'], 15.0),
        ('diag_1_group', d1_tr, d1_ev, 20.0),
        ('admission_type_id', train['admission_type_id'], eval_cols['admission_type_id'], 15.0),
    ]

    for col_name, tr_arr, ev_arr, sm in te_cols:
        te_tr, te_ev = target_encode(tr_arr, y_train, ev_arr, n_splits=5, smoothing=sm)
        register_num(f'te_{col_name}', te_tr, te_ev)

    disch_prior_inp_tr = np.array([f"{d}_{p}" for d, p in zip(train['discharge_disposition_id'], util_tr['has_prior_inpatient'])], dtype=object)
    disch_prior_inp_ev = np.array([f"{d}_{p}" for d, p in zip(eval_cols['discharge_disposition_id'], util_ev['has_prior_inpatient'])], dtype=object)
    te_disch_prior_inp_tr, te_disch_prior_inp_ev = target_encode(disch_prior_inp_tr, y_train, disch_prior_inp_ev, n_splits=5, smoothing=20.0)
    register_num('te_disch_prior_inp', te_disch_prior_inp_tr, te_disch_prior_inp_ev)

    diag_disch_tr = np.array([f"{diag}_{disch}" for diag, disch in zip(d1_tr, train['discharge_disposition_id'])], dtype=object)
    diag_disch_ev = np.array([f"{diag}_{disch}" for diag, disch in zip(d1_ev, eval_cols['discharge_disposition_id'])], dtype=object)
    te_diag_disch_tr, te_diag_disch_ev = target_encode(diag_disch_tr, y_train, diag_disch_ev, n_splits=5, smoothing=20.0)
    register_num('te_diag_disch', te_diag_disch_tr, te_diag_disch_ev)

    # 8. Encode full feature matrices using TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        tr,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_indices = encoder.categorical_indices

    # 9. Diverse Multi-Depth LightGBM Ensemble & XGBoost
    import lightgbm as lgb
    import xgboost as xgb

    m1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=24,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    m2 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.035,
        max_depth=6,
        num_leaves=30,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )

    m3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.045,
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.05,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    m4 = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=3.0,
        reg_alpha=0.2,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
    )

    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    m4.fit(x_train, y_train)

    p1 = m1.predict_proba(x_eval)[:, 1]
    p2 = m2.predict_proba(x_eval)[:, 1]
    p3 = m3.predict_proba(x_eval)[:, 1]
    p4 = m4.predict_proba(x_eval)[:, 1]

    preds = 0.30 * p1 + 0.30 * p2 + 0.20 * p3 + 0.20 * p4
    preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))
    preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
