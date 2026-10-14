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
    # 1. Comorbidities via clinical toolkit
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])

    def compute_comorbidity_syndromes(cm):
        chf = cm['cm_chf']
        renal = cm['cm_renal']
        mi = cm['cm_mi']
        pvd = cm['cm_pvd']
        cevd = cm['cm_cerebrovascular']
        cancer = cm['cm_cancer']
        met = cm['cm_metastatic']
        liver = cm['cm_liver']
        diab_comp = cm['cm_diabetes_complicated']

        cardiorenal = ((chf > 0.5) & (renal > 0.5)).astype(np.float64)
        vascular = ((mi > 0.5) | (pvd > 0.5) | (cevd > 0.5)).astype(np.float64)
        high_burden = ((cancer > 0.5) | (met > 0.5) | (liver > 0.5)).astype(np.float64)

        charlson_weighted = (
            1.0 * (mi + chf + pvd + cevd + cm['cm_dementia'] + cm['cm_pulmonary'] +
                   cm['cm_rheumatic'] + cm['cm_peptic_ulcer'] + diab_comp) +
            2.0 * (cm['cm_hemiplegia'] + renal + cancer) +
            3.0 * liver +
            6.0 * (met + cm['cm_hiv'])
        ).astype(np.float64)

        return {
            'cm_cardiorenal': cardiorenal,
            'cm_vascular': vascular,
            'cm_high_burden': high_burden,
            'cm_charlson_weighted': charlson_weighted,
        }

    cm_synd_tr = compute_comorbidity_syndromes(cm_tr)
    cm_synd_ev = compute_comorbidity_syndromes(cm_ev)

    # 2. Strack ICD-9 chapter groups
    grp1_tr = np.array([icd9_group(c) for c in train['diag_1']], dtype=object)
    grp1_ev = np.array([icd9_group(c) for c in eval_cols['diag_1']], dtype=object)
    grp2_tr = np.array([icd9_group(c) for c in train['diag_2']], dtype=object)
    grp2_ev = np.array([icd9_group(c) for c in eval_cols['diag_2']], dtype=object)
    grp3_tr = np.array([icd9_group(c) for c in train['diag_3']], dtype=object)
    grp3_ev = np.array([icd9_group(c) for c in eval_cols['diag_3']], dtype=object)

    # 3. Out-of-fold target encodings on high-cardinality codes & interactions
    te_diag1_tr, te_diag1_ev = oof_target_encode(train['diag_1'], y_train, eval_cols['diag_1'], smoothing=20.0, seed=42)
    te_diag2_tr, te_diag2_ev = oof_target_encode(train['diag_2'], y_train, eval_cols['diag_2'], smoothing=25.0, seed=42)
    te_diag3_tr, te_diag3_ev = oof_target_encode(train['diag_3'], y_train, eval_cols['diag_3'], smoothing=25.0, seed=42)
    te_disch_tr, te_disch_ev = oof_target_encode(train['discharge_disposition_id'], y_train, eval_cols['discharge_disposition_id'], smoothing=15.0, seed=42)
    te_admit_tr, te_admit_ev = oof_target_encode(train['admission_source_id'], y_train, eval_cols['admission_source_id'], smoothing=15.0, seed=42)
    te_spec_tr, te_spec_ev = oof_target_encode(train['medical_specialty'], y_train, eval_cols['medical_specialty'], smoothing=25.0, seed=42)
    te_payer_tr, te_payer_ev = oof_target_encode(train['payer_code'], y_train, eval_cols['payer_code'], smoothing=20.0, seed=42)

    # Combinatorial interactions (Discharge x Admission) & (ICD9 Group 1 x Discharge)
    da_tr = np.array([f"{d}_{s}" for d, s in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    da_ev = np.array([f"{d}_{s}" for d, s in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    te_da_tr, te_da_ev = oof_target_encode(da_tr, y_train, da_ev, smoothing=20.0, seed=42)

    gd_tr = np.array([f"{g}_{d}" for g, d in zip(grp1_tr, train['discharge_disposition_id'])], dtype=object)
    gd_ev = np.array([f"{g}_{d}" for g, d in zip(grp1_ev, eval_cols['discharge_disposition_id'])], dtype=object)
    te_gd_tr, te_gd_ev = oof_target_encode(gd_tr, y_train, gd_ev, smoothing=20.0, seed=42)

    # 4. Clinical domain velocities, utilization, and discordance features
    def extract_domain_features(data, cm):
        los = np.maximum(np.nan_to_num(data['time_in_hospital'], nan=1.0), 0.5)
        n_labs = np.nan_to_num(data['num_lab_procedures'], nan=0.0)
        n_procs = np.nan_to_num(data['num_procedures'], nan=0.0)
        n_meds = np.nan_to_num(data['num_medications'], nan=0.0)
        n_diags = np.nan_to_num(data['number_diagnoses'], nan=0.0)

        n_inp = np.nan_to_num(data['number_inpatient'], nan=0.0)
        n_emg = np.nan_to_num(data['number_emergency'], nan=0.0)
        n_out = np.nan_to_num(data['number_outpatient'], nan=0.0)

        labs_per_day = n_labs / los
        procs_per_day = n_procs / los
        meds_per_day = n_meds / los
        diags_per_day = n_diags / los
        total_events = n_labs + n_procs + n_meds
        events_per_day = total_events / los
        med_per_diag = n_meds / np.maximum(n_diags, 1.0)

        total_prior = n_inp + n_emg + n_out
        weighted_util = 3.0 * n_inp + 1.5 * n_emg + 0.5 * n_out
        inpatient_ratio = n_inp / (total_prior + 1.0)
        acute_util_ratio = (n_inp + n_emg) / (total_prior + 1.0)
        has_prior_inp = (n_inp > 0).astype(np.float64)
        frequent_inpatient = (n_inp >= 2).astype(np.float64)

        cm_count = cm['cm_count']
        discharge_acuity_gap = (n_diags + 2.0 * cm_count) / los
        premature_discharge = ((los <= 2.0) & ((n_diags >= 8.0) | (cm_count >= 2.0))).astype(np.float64)

        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        age_col = data['age']
        age_num = np.array([age_map.get(a, 65.0) for a in age_col], dtype=np.float64)
        age_x_inpatient = age_num * n_inp
        age_x_cm = age_num * cm_count
        is_elderly = (age_num >= 75.0).astype(np.float64)

        med_names = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        n_rows = len(los)
        num_active_meds = np.zeros(n_rows, dtype=np.float64)
        num_up_meds = np.zeros(n_rows, dtype=np.float64)
        num_down_meds = np.zeros(n_rows, dtype=np.float64)

        for m in med_names:
            if m in data:
                arr = data[m]
                num_active_meds += (arr != 'No').astype(np.float64)
                num_up_meds += (arr == 'Up').astype(np.float64)
                num_down_meds += (arr == 'Down').astype(np.float64)

        num_changed_meds = num_up_meds + num_down_meds
        polypharmacy = (num_active_meds >= 3).astype(np.float64)

        ins = data.get('insulin')
        if ins is not None:
            insulin_is_up = (ins == 'Up').astype(np.float64)
            insulin_is_down = (ins == 'Down').astype(np.float64)
            insulin_is_steady = (ins == 'Steady').astype(np.float64)
            insulin_is_active = (ins != 'No').astype(np.float64)
            insulin_changed = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        else:
            insulin_is_up = np.zeros(n_rows, dtype=np.float64)
            insulin_is_down = np.zeros(n_rows, dtype=np.float64)
            insulin_is_steady = np.zeros(n_rows, dtype=np.float64)
            insulin_is_active = np.zeros(n_rows, dtype=np.float64)
            insulin_changed = np.zeros(n_rows, dtype=np.float64)

        change_flag = (data.get('change') == 'Ch').astype(np.float64)
        dm_med_flag = (data.get('diabetesMed') == 'Yes').astype(np.float64)

        a1c = data.get('A1Cresult')
        if a1c is not None:
            a1c_tested = (a1c != 'None').astype(np.float64)
            a1c_high = ((a1c == '>8') | (a1c == '>7')).astype(np.float64)
        else:
            a1c_tested = np.zeros(n_rows, dtype=np.float64)
            a1c_high = np.zeros(n_rows, dtype=np.float64)

        glu = data.get('max_glu_serum')
        if glu is not None:
            glu_tested = (glu != 'None').astype(np.float64)
            glu_high = ((glu == '>300') | (glu == '>200')).astype(np.float64)
        else:
            glu_tested = np.zeros(n_rows, dtype=np.float64)
            glu_high = np.zeros(n_rows, dtype=np.float64)

        return {
            'labs_per_day': labs_per_day,
            'procs_per_day': procs_per_day,
            'meds_per_day': meds_per_day,
            'diags_per_day': diags_per_day,
            'total_events': total_events,
            'events_per_day': events_per_day,
            'med_per_diag': med_per_diag,
            'total_prior': total_prior,
            'weighted_util': weighted_util,
            'inpatient_ratio': inpatient_ratio,
            'acute_util_ratio': acute_util_ratio,
            'has_prior_inp': has_prior_inp,
            'frequent_inpatient': frequent_inpatient,
            'discharge_acuity_gap': discharge_acuity_gap,
            'premature_discharge': premature_discharge,
            'age_num': age_num,
            'age_x_inpatient': age_x_inpatient,
            'age_x_cm': age_x_cm,
            'is_elderly': is_elderly,
            'num_active_meds': num_active_meds,
            'num_up_meds': num_up_meds,
            'num_down_meds': num_down_meds,
            'num_changed_meds': num_changed_meds,
            'polypharmacy': polypharmacy,
            'insulin_is_up': insulin_is_up,
            'insulin_is_down': insulin_is_down,
            'insulin_is_steady': insulin_is_steady,
            'insulin_is_active': insulin_is_active,
            'insulin_changed': insulin_changed,
            'change_flag': change_flag,
            'dm_med_flag': dm_med_flag,
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'glu_tested': glu_tested,
            'glu_high': glu_high,
        }

    dom_tr = extract_domain_features(train, cm_tr)
    dom_ev = extract_domain_features(eval_cols, cm_ev)

    # 5. Populate feature dictionaries
    train_dict = dict(train)
    eval_dict = dict(eval_cols)

    for k, v in cm_tr.items():
        train_dict[k] = v
        eval_dict[k] = cm_ev[k]

    for k, v in cm_synd_tr.items():
        train_dict[k] = v
        eval_dict[k] = cm_synd_ev[k]

    for k, v in dom_tr.items():
        train_dict[k] = v
        eval_dict[k] = dom_ev[k]

    te_features = {
        'te_diag1': (te_diag1_tr, te_diag1_ev),
        'te_diag2': (te_diag2_tr, te_diag2_ev),
        'te_diag3': (te_diag3_tr, te_diag3_ev),
        'te_discharge': (te_discharge_tr, te_discharge_ev),
        'te_admit': (te_admit_tr, te_admit_ev),
        'te_spec': (te_spec_tr, te_spec_ev),
        'te_payer': (te_payer_tr, te_payer_ev),
        'te_da': (te_da_tr, te_da_ev),
        'te_gd': (te_gd_tr, te_gd_ev),
    }
    for k, (v_tr, v_ev) in te_features.items():
        train_dict[k] = v_tr
        eval_dict[k] = v_ev

    train_dict['diag_1_group'] = grp1_tr
    eval_dict['diag_1_group'] = grp1_ev
    train_dict['diag_2_group'] = grp2_tr
    eval_dict['diag_2_group'] = grp2_ev
    train_dict['diag_3_group'] = grp3_tr
    eval_dict['diag_3_group'] = grp3_ev

    extra_numeric = tuple(
        list(cm_tr.keys()) +
        list(cm_synd_tr.keys()) +
        list(dom_tr.keys()) +
        list(te_features.keys())
    )
    extra_categorical = ('diag_1_group', 'diag_2_group', 'diag_3_group')

    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    cat_idx = encoder.categorical_indices

    # 6. Tri-Engine Heterogeneous Ensemble
    # Engine A: Deep interaction specialist
    model_deep = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.016,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=0.05,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_deep.fit(x_train, y_train, categorical_feature=cat_idx)
    p_deep = model_deep.predict_proba(x_eval)[:, 1]

    # Engine B: Broad robust generalizer
    model_shallow = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.020,
        num_leaves=16,
        min_child_samples=200,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.8,
        cat_smooth=80.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model_shallow.fit(x_train, y_train, categorical_feature=cat_idx)
    p_shallow = model_shallow.predict_proba(x_eval)[:, 1]

    # Engine C: Readmission boundary margin-weighted learner
    model_margin = lgb.LGBMClassifier(
        n_estimators=170,
        learning_rate=0.018,
        num_leaves=24,
        min_child_samples=130,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.55,
        scale_pos_weight=1.35,
        reg_lambda=0.2,
        cat_smooth=60.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    model_margin.fit(x_train, y_train, categorical_feature=cat_idx)
    p_margin = model_margin.predict_proba(x_eval)[:, 1]

    # Engine D: XGBoost histogram level-wise booster
    p_xgb = None
    xgb_pkg = globals().get('xgb') or globals().get('xgboost')
    if xgb_pkg is None:
        try:
            import xgboost as xgb_pkg
        except Exception:
            xgb_pkg = None

    if xgb_pkg is not None:
        try:
            xgb_model = xgb_pkg.XGBClassifier(
                n_estimators=140,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.60,
                reg_lambda=1.5,
                min_child_weight=5.0,
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                eval_metric='logloss',
            )
            xgb_model.fit(x_train, y_train)
            p_xgb = xgb_model.predict_proba(x_eval)[:, 1]
        except Exception:
            p_xgb = None

    # 7. Logit-space Bayesian blending
    def logit(p):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p_c / (1.0 - p_c))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    if p_xgb is not None:
        weights = [0.35, 0.25, 0.20, 0.20]
        preds = [p_deep, p_shallow, p_margin, p_xgb]
    else:
        weights = [0.45, 0.30, 0.25]
        preds = [p_deep, p_shallow, p_margin]

    blend_logit = np.zeros_like(p_deep)
    for w, p in zip(weights, preds):
        blend_logit += w * logit(p)

    return sigmoid(blend_logit)


# EVOLVE-BLOCK-END
# =====================================================================
