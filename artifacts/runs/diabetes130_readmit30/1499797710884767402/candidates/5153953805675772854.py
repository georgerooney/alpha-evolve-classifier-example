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
    """Radically Parsimonious Clinical GBDT: Pruned Domain Signal + Dual LightGBM Ensemble."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

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
        try:
            if isinstance(val, float) and np.isnan(val):
                return '__missing__'
        except Exception:
            pass
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        try:
            if isinstance(code, float) and np.isnan(code):
                return 'missing'
        except Exception:
            pass
        return str(code).strip().split('.')[0]

    # 1. Parsimonious Clinical Feature Engineering (Non-linear Ratios, Scores & Regimens Only)
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']

        # Key High-Impact Chronic Comorbidities
        has_mi = np.isin(c1, ['410', '412']) | np.isin(c2, ['410', '412']) | np.isin(c3, ['410', '412'])
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_ckd = np.isin(c1, ['582', '583', '585', '586', '588']) | np.isin(c2, ['582', '583', '585', '586', '588']) | np.isin(c3, ['582', '583', '585', '586', '588'])
        has_renal = has_ckd | (c1 == '584') | (c2 == '584') | (c3 == '584')
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')
        has_sepsis = np.isin(c1, ['038', '995']) | np.isin(c2, ['038', '995']) | np.isin(c3, ['038', '995'])
        has_anemia = np.isin(c1, ['280', '281', '282', '283', '284', '285']) | np.isin(c2, ['280', '281', '282', '283', '284', '285']) | np.isin(c3, ['280', '281', '282', '283', '284', '285'])

        # Detailed Diabetic Manifestation Flags
        has_diab_acute = np.zeros(n_rows, dtype=float)
        has_diab_comp = np.zeros(n_rows, dtype=float)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            for idx, raw in enumerate(d[col]):
                if raw is None or raw == '?' or raw == '':
                    continue
                s = str(raw).strip()
                if s.startswith('250'):
                    if s.startswith(('250.1', '250.2', '250.3')):
                        has_diab_acute[idx] = 1.0
                    elif s.startswith(('250.4', '250.5', '250.6', '250.7', '250.8', '250.9')):
                        has_diab_comp[idx] = 1.0

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_fluid_elect'] = has_fluid_elect.astype(float)
        d['has_sepsis'] = has_sepsis.astype(float)
        d['has_anemia'] = has_anemia.astype(float)
        d['has_diab_comp'] = has_diab_comp
        d['has_diab_acute'] = has_diab_acute

        # Clinical Summary Indices (Charlson & LACE)
        age_pts = np.clip(np.floor((d['age_num'] - 40.0) / 10.0), 0.0, 4.0)
        charlson = (
            1.0 * (has_mi | has_chf | has_copd | (c1 == '571') | (c2 == '571') | (c3 == '571')).astype(float)
            + 1.0 * has_diab_comp
            + 2.0 * has_renal.astype(float)
            + age_pts
        )
        d['charlson_score'] = charlson

        stay = np.maximum(0.0, d['time_in_hospital'])
        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        is_acute_adm = np.isin(adm_type, ['1', '2', '7']).astype(float)

        l_score = np.zeros(n_rows, dtype=float)
        l_score += (stay == 1).astype(float) * 1.0
        l_score += (stay == 2).astype(float) * 2.0
        l_score += (stay == 3).astype(float) * 3.0
        l_score += ((stay >= 4) & (stay <= 6)).astype(float) * 4.0
        l_score += ((stay >= 7) & (stay <= 13)).astype(float) * 5.0
        l_score += (stay >= 14).astype(float) * 7.0

        lace = l_score + is_acute_adm * 3.0 + np.clip(charlson, 0.0, 5.0) + np.clip(d['number_emergency'], 0.0, 4.0)
        d['lace_score'] = lace

        # Utilization & Intensity Dynamics (Non-linear Ratios & Log-transforms)
        tot_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['total_visits'] = tot_v.astype(float)
        d['acute_visits'] = (d['number_emergency'] + d['number_inpatient']).astype(float)
        d['inpatient_ratio'] = d['number_inpatient'] / (tot_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (tot_v + 1.0)
        d['log_inpatient'] = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_outpatient'] = np.log1p(np.maximum(0.0, d['number_outpatient']))

        d['med_per_day'] = d['num_medications'] / (stay + 1.0)
        d['lab_per_day'] = d['num_lab_procedures'] / (stay + 1.0)
        d['proc_per_day'] = d['num_procedures'] / (stay + 1.0)
        d['clinical_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / (stay + 1.0)
        d['lab_per_med'] = d['num_lab_procedures'] / (d['num_medications'] + 1.0)

        # Medication Profile & Glycemic Regimen Complexity
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        ups = np.zeros(n_rows, dtype=float)
        downs = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
                ups += (val == 'Up').astype(float)
                downs += (val == 'Down').astype(float)

        d['active_med_count'] = active
        d['med_change_count'] = changes
        d['med_up_count'] = ups
        d['med_down_count'] = downs

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['on_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['insulin_steady'] = (insulin == 'Steady').astype(float)

        met = d.get('metformin', np.array(['No'] * n_rows))
        d['on_metformin'] = (met != 'No').astype(float)

        sulf = np.zeros(n_rows, dtype=bool)
        for s_col in ['glipizide', 'glyburide', 'glimepiride']:
            if s_col in d:
                sulf |= (d[s_col] != 'No')
        d['on_sulfonylurea'] = sulf.astype(float)
        d['num_drug_classes'] = d['on_insulin'] + d['on_metformin'] + d['on_sulfonylurea']

        # Missingness & Care Quality Signals
        payer_str = np.array([safe_str(x) for x in d['payer_code']])
        spec_str = np.array([safe_str(x) for x in d['medical_specialty']])
        d['payer_missing'] = np.isin(payer_str, ['__missing__', '?']).astype(float)
        d['spec_missing'] = np.isin(spec_str, ['__missing__', '?']).astype(float)

        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['therapeutic_inertia'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)

    # 2. Cross-Position Pooled Diagnosis Risk Encoding (Target & Frequency Mappings)
    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)

    def compute_diag_map(d1, d2, d3, targets, smooth=30.0, prior=0.114):
        counts, sums = {}, {}
        for d_arr in [d1, d2, d3]:
            for c, y in zip(d_arr, targets):
                s = clean_icd(c)
                if s != 'missing':
                    counts[s] = counts.get(s, 0) + 1
                    sums[s] = sums.get(s, 0.0) + y
        return {k: (sums[k] + smooth * prior) / (counts[k] + smooth) for k in counts}

    def compute_cat_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            s = safe_str(c)
            counts[s] = counts.get(s, 0) + 1
            sums[s] = sums.get(s, 0.0) + y
        return {k: (sums[k] + smooth * prior) / (counts[k] + smooth) for k in counts}

    # Frequency mappings
    d1_counts = {}
    for x in train['diag_1_3d']:
        s = clean_icd(x)
        d1_counts[s] = d1_counts.get(s, 0) + 1
    train['freq_diag_1'] = np.array([d1_counts.get(clean_icd(x), 0) / n_tr for x in train['diag_1_3d']], dtype=float)
    eval_cols['freq_diag_1'] = np.array([d1_counts.get(clean_icd(x), 0) / n_tr for x in eval_cols['diag_1_3d']], dtype=float)

    spec_counts = {}
    for x in train['medical_specialty']:
        s = safe_str(x)
        spec_counts[s] = spec_counts.get(s, 0) + 1
    train['freq_specialty'] = np.array([spec_counts.get(safe_str(x), 0) / n_tr for x in train['medical_specialty']], dtype=float)
    eval_cols['freq_specialty'] = np.array([spec_counts.get(safe_str(x), 0) / n_tr for x in eval_cols['medical_specialty']], dtype=float)

    # 5-Fold Stratified Target Encoding
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tf in ['te_diag_1', 'te_diag_max', 'te_diag_mean', 'te_specialty']:
        train[tf] = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf_te.split(train['gender'], y_train):
        y_tr_f = y_train[tr_idx]
        diag_map = compute_diag_map(train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)

        p1 = np.array([diag_map.get(clean_icd(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        p2 = np.array([diag_map.get(clean_icd(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        p3 = np.array([diag_map.get(clean_icd(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        train['te_diag_1'][val_idx] = p1
        train['te_diag_max'][val_idx] = np.maximum(np.maximum(p1, p2), p3)
        train['te_diag_mean'][val_idx] = (p1 + p2 + p3) / 3.0

        spec_map = compute_cat_map(train['medical_specialty'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_specialty'][val_idx] = np.array([spec_map.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

    # Full train-to-eval mappings
    diag_map_all = compute_diag_map(train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d'], y_train, smooth=30.0, prior=prior_glob)
    ep1 = np.array([diag_map_all.get(clean_icd(c), prior_glob) for c in eval_cols['diag_1_3d']])
    ep2 = np.array([diag_map_all.get(clean_icd(c), prior_glob) for c in eval_cols['diag_2_3d']])
    ep3 = np.array([diag_map_all.get(clean_icd(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_1'] = ep1
    eval_cols['te_diag_max'] = np.maximum(np.maximum(ep1, ep2), ep3)
    eval_cols['te_diag_mean'] = (ep1 + ep2 + ep3) / 3.0

    spec_map_all = compute_cat_map(train['medical_specialty'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([spec_map_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']], dtype=float)

    extra_num = [
        'age_num', 'total_visits', 'acute_visits', 'inpatient_ratio', 'emergency_ratio',
        'log_inpatient', 'log_emergency', 'log_outpatient',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'clinical_intensity', 'lab_per_med',
        'active_med_count', 'med_change_count', 'med_up_count', 'med_down_count',
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'on_metformin', 'on_sulfonylurea', 'num_drug_classes',
        'payer_missing', 'spec_missing', 'a1c_tested', 'a1c_high', 'therapeutic_inertia',
        'has_chf', 'has_ckd', 'has_cad', 'has_copd', 'has_fluid_elect', 'has_sepsis', 'has_anemia', 'has_diab_comp', 'has_diab_acute',
        'charlson_score', 'lace_score',
        'freq_diag_1', 'freq_specialty',
        'te_diag_1', 'te_diag_max', 'te_diag_mean', 'te_specialty'
    ]

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d'
    ]

    # 3. Clean Matrix Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 4. Dual-Architecture LightGBM Ensemble across 5 Stratified Folds
    skf_models = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_deep = np.zeros(n_tr, dtype=float)
    oof_robust = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_preds_deep = np.zeros(n_eval_rows, dtype=float)
    eval_preds_robust = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # Model A: Deep Leaf-Wise Explorer (Higher capacity for multi-feature clinical interaction)
        m_deep = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_split_gain=0.01,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_deep.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_deep[val_idx] = m_deep.predict_proba(x_va)[:, 1]
        eval_preds_deep += m_deep.predict_proba(x_eval)[:, 1] / 5.0

        # Model B: Shallow Depth-Constrained Regularizer (Protects demographic subgroups against overfitting)
        m_robust = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.032,
            max_depth=5,
            num_leaves=20,
            min_child_samples=75,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=2.5,
            reg_lambda=10.0,
            min_split_gain=0.02,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_robust.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_robust[val_idx] = m_robust.predict_proba(x_va)[:, 1]
        eval_preds_robust += m_robust.predict_proba(x_eval)[:, 1] / 5.0

    # 5. Deterministic 1D ROC-AUC Simplex Blend on Out-of-Fold Evidence
    best_w = 0.5
    best_auc = -1.0
    for w in np.linspace(0.1, 0.9, 17):
        blend = w * oof_deep + (1.0 - w) * oof_robust
        score = roc_auc_score(y_train, blend)
        if score > best_auc:
            best_auc = score
            best_w = w

    final_pred = best_w * eval_preds_deep + (1.0 - best_w) * eval_preds_robust
    return np.clip(np.nan_to_num(final_pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
