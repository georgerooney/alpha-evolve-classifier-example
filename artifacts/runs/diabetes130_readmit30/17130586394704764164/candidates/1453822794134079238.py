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
    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson Comorbidities & Cardiorenal Syndromes
    cm_train = icd9_comorbidities(train_data['diag_1'], train_data['diag_2'], train_data['diag_3'])
    cm_eval = icd9_comorbidities(eval_data['diag_1'], eval_data['diag_2'], eval_data['diag_3'])
    for k, v in cm_train.items():
        train_data[k] = v
        eval_data[k] = cm_eval[k]
        extra_numeric.append(k)

    train_data['cm_cardiorenal'] = ((cm_train['cm_chf'] > 0) & (cm_train['cm_renal'] > 0)).astype(np.float64)
    eval_data['cm_cardiorenal'] = ((cm_eval['cm_chf'] > 0) & (cm_eval['cm_renal'] > 0)).astype(np.float64)
    extra_numeric.append('cm_cardiorenal')

    # 2. ICD-9 Chapter Groupings & 3-Digit Category Extraction
    def extract_icd9_3digit(diag_col):
        out = np.empty(len(diag_col), dtype=object)
        for i, c in enumerate(diag_col):
            if c is None:
                out[i] = 'missing'
            else:
                s = str(c).strip()
                out[i] = s.split('.')[0] if (s and s != '?') else 'missing'
        return out

    train_data['diag_1_group'] = np.array([icd9_group(c) for c in train_data['diag_1']], dtype=object)
    eval_data['diag_1_group'] = np.array([icd9_group(c) for c in eval_data['diag_1']], dtype=object)
    train_data['diag_2_group'] = np.array([icd9_group(c) for c in train_data['diag_2']], dtype=object)
    eval_data['diag_2_group'] = np.array([icd9_group(c) for c in eval_data['diag_2']], dtype=object)
    train_data['diag_3_group'] = np.array([icd9_group(c) for c in train_data['diag_3']], dtype=object)
    eval_data['diag_3_group'] = np.array([icd9_group(c) for c in eval_data['diag_3']], dtype=object)

    train_data['diag_1_3digit'] = extract_icd9_3digit(train_data['diag_1'])
    eval_data['diag_1_3digit'] = extract_icd9_3digit(eval_data['diag_1'])
    train_data['diag_2_3digit'] = extract_icd9_3digit(train_data['diag_2'])
    eval_data['diag_2_3digit'] = extract_icd9_3digit(eval_data['diag_2'])
    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group', 'diag_1_3digit', 'diag_2_3digit'])

    # Acute Readmission Precipitants (Electrolyte Imbalance, Pneumonia, Sepsis)
    def extract_prefix_flag(d1, d2, d3, prefixes):
        pref_tuple = tuple(prefixes)
        n = len(d1)
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            c1 = str(d1[i]).strip() if d1[i] is not None else ''
            c2 = str(d2[i]).strip() if d2[i] is not None else ''
            c3 = str(d3[i]).strip() if d3[i] is not None else ''
            if c1.startswith(pref_tuple) or c2.startswith(pref_tuple) or c3.startswith(pref_tuple):
                res[i] = 1.0
        return res

    for d, d_ref in ((train_data, train_data), (eval_data, eval_data)):
        d['cm_electrolyte'] = extract_prefix_flag(d_ref['diag_1'], d_ref['diag_2'], d_ref['diag_3'], ['276'])
        d['cm_pneumonia'] = extract_prefix_flag(d_ref['diag_1'], d_ref['diag_2'], d_ref['diag_3'], ['480', '481', '482', '483', '484', '485', '486'])
        d['cm_sepsis'] = extract_prefix_flag(d_ref['diag_1'], d_ref['diag_2'], d_ref['diag_3'], ['038', '995.9'])
    extra_numeric.extend(['cm_electrolyte', 'cm_pneumonia', 'cm_sepsis'])

    # 3. Validated Clinical LACE Readmission Risk Index
    for d in (train_data, eval_data):
        los = np.nan_to_num(d['time_in_hospital'], nan=0.0)
        l_score = np.select(
            [los < 1, los == 1, los == 2, los == 3,
             (los >= 4) & (los <= 6), (los >= 7) & (los <= 13), los >= 14],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
            default=0.0
        )
        a_score = np.where(d['admission_type_id'] == '1', 3.0, 0.0)
        c_score = np.minimum(d['cm_count'], 5.0)
        e_score = np.minimum(np.nan_to_num(d['number_emergency'], nan=0.0), 4.0)
        d['lace_index'] = l_score + a_score + c_score + e_score
    extra_numeric.append('lace_index')

    # 4. Out-of-Fold Target Encodings (Individual & Composite Care Transitions)
    te_features = ['diag_1', 'diag_2', 'diag_3', 'diag_1_3digit', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_features:
        te_tr, te_ev = oof_target_encode(train_data[col], y_train, eval_data[col], n_folds=5, smoothing=25.0, seed=42)
        te_name = f'te_{col}'
        train_data[te_name] = te_tr
        eval_data[te_name] = te_ev
        extra_numeric.append(te_name)

    # Composite: Discharge Disposition x Primary Diagnosis Group
    disp_diag_tr = np.array([f"{d}_{g}" for d, g in zip(train_data['discharge_disposition_id'], train_data['diag_1_group'])], dtype=object)
    disp_diag_ev = np.array([f"{d}_{g}" for d, g in zip(eval_data['discharge_disposition_id'], eval_data['diag_1_group'])], dtype=object)
    te_dd_tr, te_dd_ev = oof_target_encode(disp_diag_tr, y_train, disp_diag_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_disp_diag'] = te_dd_tr
    eval_data['te_disp_diag'] = te_dd_ev
    extra_numeric.append('te_disp_diag')

    # Composite: Admission Type x Admission Source
    adm_combo_tr = np.array([f"{t}_{s}" for t, s in zip(train_data['admission_type_id'], train_data['admission_source_id'])], dtype=object)
    adm_combo_ev = np.array([f"{t}_{s}" for t, s in zip(eval_data['admission_type_id'], eval_data['admission_source_id'])], dtype=object)
    te_adm_tr, te_adm_ev = oof_target_encode(adm_combo_tr, y_train, adm_combo_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_adm_combo'] = te_adm_tr
    eval_data['te_adm_combo'] = te_adm_ev
    extra_numeric.append('te_adm_combo')

    # Composite: Care Transition Pathway (Admission Source x Discharge Disposition)
    adm_disp_tr = np.array([f"{a}_{d}" for a, d in zip(train_data['admission_source_id'], train_data['discharge_disposition_id'])], dtype=object)
    adm_disp_ev = np.array([f"{a}_{d}" for a, d in zip(eval_data['admission_source_id'], eval_data['discharge_disposition_id'])], dtype=object)
    te_ad_tr, te_ad_ev = oof_target_encode(adm_disp_tr, y_train, adm_disp_ev, n_folds=5, smoothing=25.0, seed=42)
    train_data['te_adm_disp'] = te_ad_tr
    eval_data['te_adm_disp'] = te_ad_ev
    extra_numeric.append('te_adm_disp')

    # 5. Prior Healthcare Utilization Intensity & Encounter Velocity
    for d in (train_data, eval_data):
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(d['number_emergency'], nan=0.0)
        out = np.nan_to_num(d['number_outpatient'], nan=0.0)
        tot = inp + emg + out
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 0.5)

        d['total_prior_visits'] = tot
        d['log_total_visits'] = np.log1p(tot)
        d['log_inpatient'] = np.log1p(inp)
        d['inpatient_ratio'] = (inp + 0.1) / (tot + 1.0)
        d['emergency_ratio'] = (emg + 0.1) / (tot + 1.0)
        d['acute_utilization_score'] = 3.0 * inp + 2.0 * emg + out
        d['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d['freq_inpatient'] = (inp >= 2).astype(np.float64)
        d['high_inpatient'] = (inp >= 3).astype(np.float64)
        d['no_prior_visits'] = (tot == 0).astype(np.float64)

        d['labs_per_day'] = np.nan_to_num(d['num_lab_procedures'], nan=0.0) / los
        d['meds_per_day'] = np.nan_to_num(d['num_medications'], nan=0.0) / los
        d['procs_per_day'] = np.nan_to_num(d['num_procedures'], nan=0.0) / los
        d['meds_per_diag'] = np.nan_to_num(d['num_medications'], nan=0.0) / np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0)
        d['polypharmacy'] = (np.nan_to_num(d['num_medications'], nan=0.0) >= 15).astype(np.float64)
        d['multi_diagnoses'] = (np.nan_to_num(d['number_diagnoses'], nan=0.0) >= 9).astype(np.float64)

        d['inp_x_los'] = inp * los
        d['inp_x_cm'] = inp * d['cm_count']
        d['inp_x_lace'] = inp * d['lace_index']

    extra_numeric.extend([
        'total_prior_visits', 'log_total_visits', 'log_inpatient', 'inpatient_ratio', 'emergency_ratio',
        'acute_utilization_score', 'has_prior_inpatient', 'freq_inpatient', 'high_inpatient', 'no_prior_visits',
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'meds_per_diag', 'polypharmacy', 'multi_diagnoses',
        'inp_x_los', 'inp_x_cm', 'inp_x_lace'
    ])

    # 6. Age Continuous Mapping, Dispositions, & Clinical Lab Tests
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for d in (train_data, eval_data):
        d['age_numeric'] = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in d['age']], dtype=np.float64)
        d['age_x_inpatient'] = d['age_numeric'] * np.log1p(np.nan_to_num(d['number_inpatient'], nan=0.0))

        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = np.isin(disp, ['3', '4', '5', '22', '23', '24']).astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)

        # Glycemic Testing & Physician Responsiveness
        a1c = d['A1Cresult']
        d['a1c_tested'] = (a1c != 'None').astype(np.float64)
        d['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        d['a1c_very_high'] = (a1c == '>8').astype(np.float64)

        glu = d['max_glu_serum']
        d['glu_tested'] = (glu != 'None').astype(np.float64)
        d['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)

        ch = d.get('change', np.array(['No'] * len(disp)))
        d['a1c_high_no_change'] = ((d['a1c_high'] == 1.0) & (ch == 'No')).astype(np.float64)
        d['a1c_high_changed'] = ((d['a1c_high'] == 1.0) & (ch == 'Ch')).astype(np.float64)

        d['spec_is_missing'] = np.array([1.0 if s is None or s == '?' else 0.0 for s in d['medical_specialty']], dtype=np.float64)
        d['payer_is_missing'] = np.array([1.0 if p is None or p == '?' else 0.0 for p in d['payer_code']], dtype=np.float64)

    extra_numeric.extend([
        'age_numeric', 'age_x_inpatient', 'disch_home', 'disch_snf', 'disch_ama', 'disch_home_health',
        'a1c_tested', 'a1c_high', 'a1c_very_high', 'glu_tested', 'glu_high', 'a1c_high_no_change', 'a1c_high_changed',
        'spec_is_missing', 'payer_is_missing'
    ])

    # 7. Medication Regimen Complexity, Escalation vs De-escalation
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
        'insulin'
    ]
    avail_meds = [m for m in med_cols if m in task.categorical_columns]
    for d in (train_data, eval_data):
        n_rows = len(d['time_in_hospital'])
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        for m in avail_meds:
            col = d[m]
            active_cnt += np.isin(col, ['Steady', 'Up', 'Down']).astype(np.float64)
            up_cnt += (col == 'Up').astype(np.float64)
            down_cnt += (col == 'Down').astype(np.float64)
        d['num_active_meds'] = active_cnt
        d['num_meds_up'] = up_cnt
        d['num_meds_down'] = down_cnt
        d['num_changed_meds'] = up_cnt + down_cnt

        if 'insulin' in d:
            ins = d['insulin']
            d['has_insulin'] = (ins != 'No').astype(np.float64)
            d['insulin_status'] = np.select([ins == 'Steady', ins == 'Down', ins == 'Up'], [1.0, 2.0, 3.0], default=0.0)
            d['insulin_unstable'] = np.isin(ins, ['Up', 'Down']).astype(np.float64)

        if 'metformin' in d and 'insulin' in d:
            met = d['metformin']
            ins = d['insulin']
            d['insulin_only'] = ((ins != 'No') & (met == 'No')).astype(np.float64)
            d['insulin_and_met'] = ((ins != 'No') & (met != 'No')).astype(np.float64)

        if 'change' in d:
            d['med_change_flag'] = (d['change'] == 'Ch').astype(np.float64)
        if 'diabetesMed' in d:
            d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(np.float64)

    extra_numeric.extend(['num_active_meds', 'num_meds_up', 'num_meds_down', 'num_changed_meds', 'med_change_flag', 'diabetes_med_flag'])
    if 'insulin' in train_data:
        extra_numeric.extend(['has_insulin', 'insulin_status', 'insulin_unstable'])
    if 'metformin' in train_data and 'insulin' in train_data:
        extra_numeric.extend(['insulin_only', 'insulin_and_met'])

    # 8. Encode Features with Moderate min_count
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_data, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 9. Diverse Multi-Scale Acuity Triad Ensemble
    model_balanced = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_balanced.fit(x_train, y_train, categorical_feature=cat_indices)

    model_interaction = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.015,
        num_leaves=48,
        max_depth=7,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.52,
        reg_lambda=3.5,
        reg_alpha=0.4,
        cat_smooth=35.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    model_interaction.fit(x_train, y_train, categorical_feature=cat_indices)

    model_conservative = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.022,
        num_leaves=18,
        max_depth=4,
        min_child_samples=250,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.68,
        reg_lambda=5.0,
        cat_smooth=75.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model_conservative.fit(x_train, y_train, categorical_feature=cat_indices)

    pred1 = model_balanced.predict_proba(x_eval)[:, 1]
    pred2 = model_interaction.predict_proba(x_eval)[:, 1]
    pred3 = model_conservative.predict_proba(x_eval)[:, 1]

    # 10. Diverse Depth-Wise Model Integration (XGBoost)
    pred_xgb = None
    xgb_pkg = globals().get('xgb', globals().get('xgboost', None))
    if xgb_pkg is not None and hasattr(xgb_pkg, 'XGBClassifier'):
        try:
            model_xgb = xgb_pkg.XGBClassifier(
                n_estimators=200,
                learning_rate=0.022,
                max_depth=5,
                min_child_weight=8,
                subsample=0.80,
                colsample_bytree=0.55,
                reg_lambda=3.0,
                reg_alpha=0.2,
                random_state=314,
                n_jobs=1,
                eval_metric='logloss',
            )
            model_xgb.fit(x_train, y_train)
            pred_xgb = model_xgb.predict_proba(x_eval)[:, 1]
        except Exception:
            pred_xgb = None

    if pred_xgb is not None:
        return 0.35 * pred1 + 0.30 * pred2 + 0.18 * pred3 + 0.17 * pred_xgb
    return 0.40 * pred1 + 0.35 * pred2 + 0.25 * pred3


# EVOLVE-BLOCK-END
# =====================================================================
