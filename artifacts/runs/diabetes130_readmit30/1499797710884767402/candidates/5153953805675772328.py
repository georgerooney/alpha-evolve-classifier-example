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
    try:
        from clinical_toolkit import icd9_group, TabularEncoder
    except ImportError:
        pass
    from sklearn.model_selection import StratifiedKFold

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Age numeric mapping
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for col_dict in [train_data, eval_data]:
        raw_age = col_dict.get('age', np.array([]))
        parsed = np.array([age_map.get(a, np.nan) for a in raw_age], dtype=np.float64)
        col_dict['num_age'] = parsed
        col_dict['num_age_over_65'] = (parsed >= 65.0).astype(np.float64)
        col_dict['num_age_over_75'] = (parsed >= 75.0).astype(np.float64)
        col_dict['num_age_sq'] = (parsed / 10.0) ** 2.0
    extra_numeric.extend(['num_age', 'num_age_over_65', 'num_age_over_75', 'num_age_sq'])

    # 2. Utilisation & Hospital Stay features
    for col_dict in [train_data, eval_data]:
        inp = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        outp = np.nan_to_num(col_dict.get('number_outpatient', 0.0), nan=0.0)
        tot_visits = inp + emg + outp

        col_dict['util_total_visits'] = tot_visits
        col_dict['util_has_inpatient'] = (inp > 0.0).astype(np.float64)
        col_dict['util_has_emergency'] = (emg > 0.0).astype(np.float64)
        col_dict['util_has_outpatient'] = (outp > 0.0).astype(np.float64)
        col_dict['util_inpatient_ratio'] = inp / (tot_visits + 1.0)
        col_dict['util_emergency_ratio'] = emg / (tot_visits + 1.0)
        col_dict['util_outpatient_ratio'] = outp / (tot_visits + 1.0)
        col_dict['util_severity'] = (inp * 3.0 + emg * 2.0 + outp) / 3.0
        col_dict['util_frequent_flyer'] = ((inp >= 2.0) | (tot_visits >= 5.0)).astype(np.float64)

        time_hosp = np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0)
        time_safe = np.maximum(time_hosp, 1.0)
        num_labs = np.nan_to_num(col_dict.get('num_lab_procedures', 0.0), nan=0.0)
        num_proc = np.nan_to_num(col_dict.get('num_procedures', 0.0), nan=0.0)
        num_meds = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)
        num_diag = np.nan_to_num(col_dict.get('number_diagnoses', 0.0), nan=0.0)

        col_dict['hosp_labs_per_day'] = num_labs / time_safe
        col_dict['hosp_procs_per_day'] = num_proc / time_safe
        col_dict['hosp_meds_per_day'] = num_meds / time_safe
        col_dict['hosp_diags_per_day'] = num_diag / time_safe
        col_dict['hosp_service_intensity'] = (num_labs + num_proc + num_meds) / time_safe
        col_dict['hosp_total_procedures'] = num_labs + num_proc
        col_dict['hosp_short_stay'] = (time_hosp <= 2.0).astype(np.float64)
        col_dict['hosp_long_stay'] = (time_hosp >= 7.0).astype(np.float64)
        col_dict['hosp_high_diagnoses'] = (num_diag >= 9.0).astype(np.float64)

    extra_numeric.extend([
        'util_total_visits', 'util_has_inpatient', 'util_has_emergency', 'util_has_outpatient',
        'util_inpatient_ratio', 'util_emergency_ratio', 'util_outpatient_ratio',
        'util_severity', 'util_frequent_flyer',
        'hosp_labs_per_day', 'hosp_procs_per_day', 'hosp_meds_per_day', 'hosp_diags_per_day',
        'hosp_service_intensity', 'hosp_total_procedures',
        'hosp_short_stay', 'hosp_long_stay', 'hosp_high_diagnoses'
    ])

    # 3. Discharge disposition & admission source/type flags
    for col_dict in [train_data, eval_data]:
        disp = col_dict.get('discharge_disposition_id', np.array([]))
        adm_t = col_dict.get('admission_type_id', np.array([]))
        adm_s = col_dict.get('admission_source_id', np.array([]))

        col_dict['disp_home'] = (disp == '1').astype(np.float64)
        col_dict['disp_facility'] = np.isin(disp, ['2', '3', '4', '5', '22']).astype(np.float64)
        col_dict['disp_home_health'] = (disp == '6').astype(np.float64)
        col_dict['disp_ama'] = (disp == '7').astype(np.float64)

        col_dict['adm_type_urgent_er'] = np.isin(adm_t, ['1', '2', '7']).astype(np.float64)
        col_dict['adm_type_elective'] = (adm_t == '3').astype(np.float64)

        col_dict['adm_source_er'] = (adm_s == '7').astype(np.float64)
        col_dict['adm_source_ref'] = np.isin(adm_s, ['1', '2']).astype(np.float64)

    extra_numeric.extend([
        'disp_home', 'disp_facility', 'disp_home_health', 'disp_ama',
        'adm_type_urgent_er', 'adm_type_elective', 'adm_source_er', 'adm_source_ref'
    ])

    # 4. Medication summary features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col_dict in [train_data, eval_data]:
        n_rows = len(col_dict['time_in_hospital'])
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)

        available_meds = [m for m in med_cols if m in col_dict]
        for m in available_meds:
            arr = col_dict[m]
            active += np.isin(arr, ['Steady', 'Up', 'Down']).astype(np.float64)
            up += (arr == 'Up').astype(np.float64)
            down += (arr == 'Down').astype(np.float64)

        changed = up + down
        insulin_arr = col_dict.get('insulin', np.full(n_rows, 'No', dtype=object))
        ins_active = np.isin(insulin_arr, ['Steady', 'Up', 'Down']).astype(np.float64)
        ins_up = (insulin_arr == 'Up').astype(np.float64)
        ins_down = (insulin_arr == 'Down').astype(np.float64)
        oral_meds = np.maximum(0.0, active - ins_active)
        ins_plus_oral = ((ins_active > 0.0) & (oral_meds > 0.0)).astype(np.float64)

        dmed = col_dict.get('diabetesMed', np.full(n_rows, 'No', dtype=object))
        ch = col_dict.get('change', np.full(n_rows, 'No', dtype=object))

        col_dict['med_num_active'] = active
        col_dict['med_num_up'] = up
        col_dict['med_num_down'] = down
        col_dict['med_num_changed'] = changed
        col_dict['med_has_change'] = (changed > 0.0).astype(np.float64)
        col_dict['med_insulin_active'] = ins_active
        col_dict['med_insulin_up'] = ins_up
        col_dict['med_insulin_down'] = ins_down
        col_dict['med_num_oral'] = oral_meds
        col_dict['med_insulin_plus_oral'] = ins_plus_oral
        col_dict['med_diabetes_med_flag'] = (dmed == 'Yes').astype(np.float64)
        col_dict['med_change_flag'] = (ch == 'Ch').astype(np.float64)

    extra_numeric.extend([
        'med_num_active', 'med_num_up', 'med_num_down', 'med_num_changed', 'med_has_change',
        'med_insulin_active', 'med_insulin_up', 'med_insulin_down', 'med_num_oral',
        'med_insulin_plus_oral', 'med_diabetes_med_flag', 'med_change_flag'
    ])

    # 5. Diagnosis grouping, prefixes & comorbidity patterns
    def extract_prefix(code):
        if code is None or code == '?' or code == 'None':
            return 'missing'
        s = str(code).strip()
        return s.split('.')[0] if s else 'missing'

    for col_dict in [train_data, eval_data]:
        d1 = col_dict.get('diag_1', np.array([]))
        d2 = col_dict.get('diag_2', np.array([]))
        d3 = col_dict.get('diag_3', np.array([]))

        g1 = np.array([icd9_group(c) for c in d1], dtype=object)
        g2 = np.array([icd9_group(c) for c in d2], dtype=object)
        g3 = np.array([icd9_group(c) for c in d3], dtype=object)

        p1 = np.array([extract_prefix(c) for c in d1], dtype=object)
        p2 = np.array([extract_prefix(c) for c in d2], dtype=object)
        p3 = np.array([extract_prefix(c) for c in d3], dtype=object)

        col_dict['diag_1_group'] = g1
        col_dict['diag_2_group'] = g2
        col_dict['diag_3_group'] = g3
        col_dict['diag_1_prefix'] = p1
        col_dict['diag_2_prefix'] = p2
        col_dict['diag_3_prefix'] = p3

        n = len(d1)
        has_circ = np.zeros(n, dtype=np.float64)
        has_diab = np.zeros(n, dtype=np.float64)
        has_resp = np.zeros(n, dtype=np.float64)
        has_digest = np.zeros(n, dtype=np.float64)
        has_genit = np.zeros(n, dtype=np.float64)
        has_chf = np.zeros(n, dtype=np.float64)
        has_ckd = np.zeros(n, dtype=np.float64)
        num_systems = np.zeros(n, dtype=np.float64)

        for i in range(n):
            systems = {g1[i], g2[i], g3[i]} - {'missing', 'other'}
            num_systems[i] = len(systems)
            if 'circulatory' in systems:
                has_circ[i] = 1.0
            if 'diabetes' in systems:
                has_diab[i] = 1.0
            if 'respiratory' in systems:
                has_resp[i] = 1.0
            if 'digestive' in systems:
                has_digest[i] = 1.0
            if 'genitourinary' in systems:
                has_genit[i] = 1.0

            prefs = (p1[i], p2[i], p3[i])
            if '428' in prefs:
                has_chf[i] = 1.0
            if '585' in prefs or '586' in prefs:
                has_ckd[i] = 1.0

        col_dict['comorb_has_circ'] = has_circ
        col_dict['comorb_has_diab'] = has_diab
        col_dict['comorb_has_resp'] = has_resp
        col_dict['comorb_has_digest'] = has_digest
        col_dict['comorb_has_genit'] = has_genit
        col_dict['comorb_has_chf'] = has_chf
        col_dict['comorb_has_ckd'] = has_ckd
        col_dict['comorb_num_systems'] = num_systems

    extra_categorical.extend([
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix'
    ])
    extra_numeric.extend([
        'comorb_has_circ', 'comorb_has_diab', 'comorb_has_resp', 'comorb_has_digest',
        'comorb_has_genit', 'comorb_has_chf', 'comorb_has_ckd', 'comorb_num_systems'
    ])

    # 6. Labs and Demographics missingness
    for col_dict in [train_data, eval_data]:
        a1c = col_dict.get('A1Cresult', np.array([]))
        glu = col_dict.get('max_glu_serum', np.array([]))
        weight = col_dict.get('weight', np.array([]))
        payer = col_dict.get('payer_code', np.array([]))
        spec = col_dict.get('medical_specialty', np.array([]))

        col_dict['lab_a1c_tested'] = ((a1c != 'None') & (a1c is not None)).astype(np.float64)
        col_dict['lab_a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        col_dict['lab_a1c_norm'] = (a1c == 'Norm').astype(np.float64)
        col_dict['lab_glu_tested'] = ((glu != 'None') & (glu is not None)).astype(np.float64)
        col_dict['lab_glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)
        col_dict['demo_weight_known'] = ((weight != '?') & (weight is not None)).astype(np.float64)
        col_dict['demo_payer_missing'] = ((payer == '?') | (payer is None)).astype(np.float64)
        col_dict['demo_payer_medicare'] = (payer == 'MC').astype(np.float64)
        col_dict['demo_payer_medicaid'] = (payer == 'MD').astype(np.float64)
        col_dict['demo_spec_missing'] = ((spec == '?') | (spec is None)).astype(np.float64)

    extra_numeric.extend([
        'lab_a1c_tested', 'lab_a1c_high', 'lab_a1c_norm', 'lab_glu_tested', 'lab_glu_high',
        'demo_weight_known', 'demo_payer_missing', 'demo_payer_medicare', 'demo_payer_medicaid',
        'demo_spec_missing'
    ])

    # 7. Out-of-fold Smoothed Target Encoding
    def compute_oof_te(tr_arr, y_tr, ev_arr, smoothing=20.0, n_splits=5, seed=42):
        prior = float(np.mean(y_tr))
        n_tr = len(tr_arr)
        n_ev = len(ev_arr)
        tr_te = np.full(n_tr, prior, dtype=np.float64)
        ev_te = np.full(n_ev, prior, dtype=np.float64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_arr, y_tr):
            sub_c = tr_arr[train_idx]
            sub_y = y_tr[train_idx]
            v_c = tr_arr[val_idx]

            counts = {}
            sums = {}
            for c, y in zip(sub_c, sub_y):
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y

            vals = np.empty(len(val_idx), dtype=np.float64)
            for i, c in enumerate(v_c):
                cnt = counts.get(c, 0)
                if cnt > 0:
                    vals[i] = (sums[c] + smoothing * prior) / (cnt + smoothing)
                else:
                    vals[i] = prior
            tr_te[val_idx] = vals

        full_counts = {}
        full_sums = {}
        for c, y in zip(tr_arr, y_tr):
            full_counts[c] = full_counts.get(c, 0) + 1
            full_sums[c] = full_sums.get(c, 0.0) + y

        for i, c in enumerate(ev_arr):
            cnt = full_counts.get(c, 0)
            if cnt > 0:
                ev_te[i] = (full_sums[c] + smoothing * prior) / (cnt + smoothing)
            else:
                ev_te[i] = prior

        return tr_te, ev_te

    te_targets = [
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'medical_specialty', 'payer_code', 'diag_1_prefix', 'diag_2_prefix',
        'diag_3_prefix', 'diag_1_group', 'age'
    ]
    for c_name in te_targets:
        if c_name in train_data:
            tr_te, ev_te = compute_oof_te(train_data[c_name], y_train, eval_data[c_name], smoothing=25.0)
            te_col_name = f'te_{c_name}'
            train_data[te_col_name] = tr_te
            eval_data[te_col_name] = ev_te
            extra_numeric.append(te_col_name)

    # 8. Frequency encoding for high cardinality columns
    def compute_freq_enc(tr_arr, ev_arr):
        counts = {}
        for c in tr_arr:
            counts[c] = counts.get(c, 0) + 1
        n_tr = float(len(tr_arr))
        tr_fe = np.array([counts.get(c, 0) / n_tr for c in tr_arr], dtype=np.float64)
        ev_fe = np.array([counts.get(c, 0) / n_tr for c in ev_arr], dtype=np.float64)
        return tr_fe, ev_fe

    fe_targets = ['medical_specialty', 'diag_1_prefix', 'discharge_disposition_id', 'admission_source_id']
    for c_name in fe_targets:
        if c_name in train_data:
            tr_fe, ev_fe = compute_freq_enc(train_data[c_name], eval_data[c_name])
            fe_col_name = f'fe_{c_name}'
            train_data[fe_col_name] = tr_fe
            eval_data[fe_col_name] = ev_fe
            extra_numeric.append(fe_col_name)

    # 9. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 10. Diverse regularized LightGBM ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=18,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=28,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=10.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=14,
            min_child_samples=50,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=3.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.035,
            num_leaves=22,
            min_child_samples=70,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=8.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    preds = np.nan_to_num(preds, nan=0.114, posinf=1.0, neginf=0.0)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
