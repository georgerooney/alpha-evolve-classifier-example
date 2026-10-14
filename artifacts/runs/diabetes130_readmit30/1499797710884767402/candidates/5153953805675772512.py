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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

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
        if val is None or val == '' or val == '?' or (isinstance(val, float) and np.isnan(val)):
            return '__missing__'
        return str(val)

    def clean_icd(code):
        if code is None or code == '' or code == '?' or (isinstance(code, float) and np.isnan(code)):
            return '__missing__'
        return str(code).strip().split('.')[0]

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

        # Clinical Guidelines & Biomarkers (Strack et al. 2014)
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['a1c_high_no_change'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)

        # Discharge and Admission risk factors
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf_hh'] = np.isin(disp, ['3', '6', '22', '28']).astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5']).astype(float)

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)

        # Diagnoses: groups, 3-digit categories & clinical comorbidity profile
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_ckd = np.isin(c1, ['585', '586']) | np.isin(c2, ['585', '586']) | np.isin(c3, ['585', '586'])
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_pvd = np.isin(c1, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c2, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c3, ['440', '441', '442', '443', '444', '445', '446', '447'])
        has_stroke = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')
        has_pneumonia = np.isin(c1, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c2, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c3, ['480', '481', '482', '483', '484', '485', '486'])

        has_diab_comp = np.zeros(n_rows, dtype=bool)
        has_diab_acute = np.zeros(n_rows, dtype=bool)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            has_diab_comp |= np.array([str(x).startswith('250') and str(x) not in ['250', '250.0', '250.00', '250.02', '250.1', '250.2', '250.3'] for x in d[col]])
            has_diab_acute |= np.array([str(x).startswith(('250.1', '250.2', '250.3')) for x in d[col]])

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_pvd'] = has_pvd.astype(float)
        d['has_stroke'] = has_stroke.astype(float)
        d['has_fluid_elect'] = has_fluid_elect.astype(float)
        d['has_pneumonia'] = has_pneumonia.astype(float)
        d['has_diab_comp'] = has_diab_comp.astype(float)
        d['has_diab_acute'] = has_diab_acute.astype(float)
        d['cardiorenal_risk'] = (has_chf & has_ckd).astype(float)

        comorb = (
            d['has_chf'] + d['has_ckd'] + d['has_cad'] + d['has_htn'] + d['has_copd'] +
            d['has_pvd'] + d['has_stroke'] + d['has_fluid_elect'] + d['has_pneumonia'] + d['has_diab_comp']
        )
        d['comorbidity_count'] = comorb
        d['frailty_index'] = comorb * (d['age_num'] / 50.0) * np.log1p(d['number_inpatient'] + 1.0)

    # 2. Bayesian Smoothed Target Encoding (5-Fold Stratified OOF within train)
    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def calc_map(cats, targets, smooth=20.0):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            cs = safe_str(c)
            counts[cs] = counts.get(cs, 0) + 1
            sums[cs] = sums.get(cs, 0.0) + y
        return {cs: (sums[cs] + smooth * prior_glob) / (counts[cs] + smooth) for cs in counts}

    te_cat_cols = [
        'diag_1_3d', 'diag_1_grp', 'discharge_disposition_id',
        'medical_specialty', 'admission_source_id', 'admission_type_id'
    ]

    for col in te_cat_cols:
        te_name = f'te_{col}'
        oof_te = np.full(n_tr, prior_glob, dtype=float)
        for tr_idx, val_idx in skf_te.split(train['gender'], y_train):
            m = calc_map(train[col][tr_idx], y_train[tr_idx])
            oof_te[val_idx] = np.array([m.get(safe_str(c), prior_glob) for c in train[col][val_idx]], dtype=float)
        train[te_name] = oof_te

        full_m = calc_map(train[col], y_train)
        eval_cols[te_name] = np.array([full_m.get(safe_str(c), prior_glob) for c in eval_cols[col]], dtype=float)

    # Pooled diagnoses target encoding
    diag_cols = ['diag_1_3d', 'diag_2_3d', 'diag_3_3d']
    te_diag_max_tr = np.full(n_tr, prior_glob, dtype=float)
    te_diag_mean_tr = np.full(n_tr, prior_glob, dtype=float)

    for tr_idx, val_idx in skf_te.split(train['gender'], y_train):
        pooled_diag = np.concatenate([train[c][tr_idx] for c in diag_cols])
        pooled_y = np.concatenate([y_train[tr_idx], y_train[tr_idx], y_train[tr_idx]])
        p_map = calc_map(pooled_diag, pooled_y, smooth=25.0)

        d1 = np.array([p_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]], dtype=float)
        d2 = np.array([p_map.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]], dtype=float)
        d3 = np.array([p_map.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]], dtype=float)
        te_diag_max_tr[val_idx] = np.maximum(np.maximum(d1, d2), d3)
        te_diag_mean_tr[val_idx] = (d1 + d2 + d3) / 3.0

    train['te_diag_max'] = te_diag_max_tr
    train['te_diag_mean'] = te_diag_mean_tr

    full_pooled_diag = np.concatenate([train[c] for c in diag_cols])
    full_pooled_y = np.concatenate([y_train, y_train, y_train])
    full_p_map = calc_map(full_pooled_diag, full_pooled_y, smooth=25.0)

    ed1 = np.array([full_p_map.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']], dtype=float)
    ed2 = np.array([full_p_map.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']], dtype=float)
    ed3 = np.array([full_p_map.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']], dtype=float)
    eval_cols['te_diag_max'] = np.maximum(np.maximum(ed1, ed2), ed3)
    eval_cols['te_diag_mean'] = (ed1 + ed2 + ed3) / 3.0

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'frequent_inpatient', 'med_per_day', 'lab_per_day',
        'proc_per_day', 'diag_per_day', 'service_intensity',
        'active_meds', 'num_med_changes', 'has_insulin', 'insulin_up', 'insulin_down',
        'change_flag', 'diabetes_med_flag',
        'a1c_tested', 'a1c_high', 'a1c_high_no_change', 'glu_high',
        'disch_is_home', 'disch_is_snf_hh', 'disch_is_ama', 'disch_is_transfer',
        'is_emergency_adm', 'is_elective_adm',
        'has_chf', 'has_ckd', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_diab_acute', 'has_diab_comp', 'cardiorenal_risk',
        'comorbidity_count', 'frailty_index',
        'te_diag_1_3d', 'te_diag_1_grp', 'te_discharge_disposition_id',
        'te_medical_specialty', 'te_admission_source_id', 'te_admission_type_id',
        'te_diag_max', 'te_diag_mean',
    ]

    extra_cat = ['diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d']

    # 3. Tabular Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 4. Streamlined Dual-Engine Ensemble (Leaf-Wise LightGBM + Depth-Wise XGBoost) with 5-Fold CV
    skf_models = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_lgb = np.zeros(n_eval_rows, dtype=float)
    eval_xgb = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # LightGBM: Leaf-wise tree growth with native categorical handling
        model_lgb = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb[val_idx] = model_lgb.predict_proba(x_va)[:, 1]
        eval_lgb += model_lgb.predict_proba(x_eval)[:, 1] / 5.0

        # XGBoost: Depth-wise histogram splitting for complementary regularized inductive bias
        model_xgb = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=25,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        model_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = model_xgb.predict_proba(x_va)[:, 1]
        eval_xgb += model_xgb.predict_proba(x_eval)[:, 1] / 5.0

    # 5. Robust 1D Line-Search on OOF ROC-AUC
    best_w = 0.55
    best_auc = -1.0
    for w in np.linspace(0.1, 0.9, 17):
        blend_oof = w * oof_lgb + (1.0 - w) * oof_xgb
        auc = roc_auc_score(y_train, blend_oof)
        if auc > best_auc:
            best_auc = auc
            best_w = w

    eval_pred = best_w * eval_lgb + (1.0 - best_w) * eval_xgb
    return np.clip(np.nan_to_num(eval_pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
