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
    """Streamlined Clinical Scoring & Tri-LightGBM Native Fisher Ensemble."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

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

    def check_neoplasm(codes):
        res = np.zeros(len(codes), dtype=bool)
        for idx, c in enumerate(codes):
            try:
                v = int(c)
                if 140 <= v <= 208:
                    res[idx] = True
            except Exception:
                pass
        return res

    # 1. High-Density Domain Feature Engineering: Indices & Dynamics
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']

        # Core clinical condition flags
        has_mi = np.isin(c1, ['410', '412']) | np.isin(c2, ['410', '412']) | np.isin(c3, ['410', '412'])
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_pvd = np.isin(c1, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c2, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c3, ['440', '441', '442', '443', '444', '445', '446', '447'])
        has_stroke = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])
        has_dementia = np.isin(c1, ['290', '294', '331']) | np.isin(c2, ['290', '294', '331']) | np.isin(c3, ['290', '294', '331'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_rheum = np.isin(c1, ['710', '714', '725']) | np.isin(c2, ['710', '714', '725']) | np.isin(c3, ['710', '714', '725'])
        has_pud = np.isin(c1, ['531', '532', '533', '534']) | np.isin(c2, ['531', '532', '533', '534']) | np.isin(c3, ['531', '532', '533', '534'])
        has_mild_liver = (c1 == '571') | (c2 == '571') | (c3 == '571')
        has_sev_liver = np.isin(c1, ['572', '456']) | np.isin(c2, ['572', '456']) | np.isin(c3, ['572', '456'])
        has_hemi = np.isin(c1, ['342', '344']) | np.isin(c2, ['342', '344']) | np.isin(c3, ['342', '344'])
        has_ckd = np.isin(c1, ['582', '583', '585', '586', '588']) | np.isin(c2, ['582', '583', '585', '586', '588']) | np.isin(c3, ['582', '583', '585', '586', '588'])
        has_aki = (c1 == '584') | (c2 == '584') | (c3 == '584')
        has_renal = has_ckd | has_aki
        has_met_cancer = np.isin(c1, ['196', '197', '198', '199']) | np.isin(c2, ['196', '197', '198', '199']) | np.isin(c3, ['196', '197', '198', '199'])
        has_hiv = (c1 == '042') | (c2 == '042') | (c3 == '042')
        has_neoplasm = check_neoplasm(c1) | check_neoplasm(c2) | check_neoplasm(c3)
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')

        # Diabetes condition mapping
        is_diab_uncontrolled = np.zeros(n_rows, dtype=float)
        has_diab_uncomp = np.zeros(n_rows, dtype=float)
        has_diab_acute = np.zeros(n_rows, dtype=float)
        has_diab_comp = np.zeros(n_rows, dtype=float)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            for idx, raw in enumerate(d[col]):
                if raw is None or raw == '?' or raw == '':
                    continue
                s = str(raw).strip()
                if s.startswith('250'):
                    if s.endswith(('2', '3')):
                        is_diab_uncontrolled[idx] = 1.0
                    if s.startswith(('250.1', '250.2', '250.3')):
                        has_diab_acute[idx] = 1.0
                    elif s.startswith(('250.4', '250.5', '250.6', '250.7', '250.8', '250.9')):
                        has_diab_comp[idx] = 1.0
                    else:
                        has_diab_uncomp[idx] = 1.0

        d['is_diab_uncontrolled'] = is_diab_uncontrolled
        d['has_diab_acute'] = has_diab_acute
        d['has_diab_comp'] = has_diab_comp
        d['has_chf'] = has_chf.astype(float)
        d['has_renal'] = has_renal.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['cardiorenal_risk'] = (has_chf & has_renal).astype(float)

        # Charlson Comorbidity Index
        age_pts = np.clip(np.floor((d['age_num'] - 40.0) / 10.0), 0.0, 4.0)
        charlson = (
            1.0 * (has_mi | has_chf | has_pvd | has_stroke | has_dementia | has_copd | has_rheum | has_pud | has_mild_liver | (has_diab_uncomp > 0.5)).astype(float)
            + 1.0 * has_diab_comp
            + 2.0 * (has_hemi | has_renal | has_neoplasm).astype(float)
            + 3.0 * has_sev_liver.astype(float)
            + 6.0 * (has_met_cancer | has_hiv).astype(float)
            + age_pts
        )
        d['charlson_score'] = charlson

        # Elixhauser Comorbidity Count
        d['elixhauser_count'] = (
            has_chf.astype(float) + has_cad.astype(float) + has_pvd.astype(float) +
            has_stroke.astype(float) + has_copd.astype(float) + has_renal.astype(float) +
            has_sev_liver.astype(float) + has_neoplasm.astype(float) + has_met_cancer.astype(float) +
            has_fluid_elect.astype(float) + has_diab_comp
        )

        # Admission & Discharge Flags
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf'] = (disp == '3').astype(float)
        d['disch_is_home_health'] = np.isin(disp, ['6', '8']).astype(float)
        d['disch_is_ama'] = (disp == '7').astype(float)
        disch_is_facility = np.isin(disp, ['2', '3', '4', '5', '6', '8', '22', '23', '24', '27', '28', '29', '30']).astype(float)
        d['disch_is_facility'] = disch_is_facility

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        is_acute_adm = np.isin(adm_type, ['1', '2', '7']).astype(float)

        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])
        d['adm_from_er'] = (adm_src == '7').astype(float)
        d['disch_adm_pair'] = np.array([f"{p}_{a}" for p, a in zip(disp, adm_src)], dtype=object)

        # LACE and LACE+ Scoring
        stay = np.maximum(0.0, d['time_in_hospital'])
        l_score = np.zeros(n_rows, dtype=float)
        l_score += (stay == 1).astype(float) * 1.0
        l_score += (stay == 2).astype(float) * 2.0
        l_score += (stay == 3).astype(float) * 3.0
        l_score += ((stay >= 4) & (stay <= 6)).astype(float) * 4.0
        l_score += ((stay >= 7) & (stay <= 13)).astype(float) * 5.0
        l_score += (stay >= 14).astype(float) * 7.0

        a_score = is_acute_adm * 3.0
        c_score = np.where(charlson >= 4.0, 5.0, np.clip(charlson, 0.0, 3.0))
        e_score = np.clip(d['number_emergency'], 0.0, 4.0)

        lace = l_score + a_score + c_score + e_score
        d['lace_score'] = lace
        d['lace_plus'] = lace + 2.0 * np.log1p(d['number_inpatient']) + 0.5 * (d['age_num'] / 10.0) + 1.5 * disch_is_facility

        # Utilization & Intensity Dynamics
        tot_v = (d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']).astype(float)
        d['total_visits'] = tot_v
        d['acute_visits'] = (d['number_emergency'] + d['number_inpatient']).astype(float)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)
        d['inpatient_ratio'] = d['number_inpatient'] / (tot_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (tot_v + 1.0)
        log_inp = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_inpatient'] = log_inp
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_total_visits'] = np.log1p(np.maximum(0.0, tot_v))

        d['med_per_day'] = d['num_medications'] / (stay + 1.0)
        d['lab_per_day'] = d['num_lab_procedures'] / (stay + 1.0)
        d['clinical_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / (stay + 1.0)
        d['high_lab_no_proc'] = ((d['num_lab_procedures'] >= 50) & (d['num_procedures'] == 0)).astype(float)
        d['diag_maxed'] = (d['number_diagnoses'] >= 9).astype(float)
        d['utilization_x_stay'] = log_inp * np.log1p(stay)
        d['utilization_x_facility'] = log_inp * disch_is_facility

        # Medication Profile
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
        d['active_med_count'] = active
        d['med_change_count'] = changes

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['on_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Biomarkers
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)
        d['therapeutic_inertia'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)

    extra_num = [
        'age_num', 'total_visits', 'acute_visits', 'has_prior_inpatient', 'frequent_inpatient',
        'inpatient_ratio', 'emergency_ratio', 'log_inpatient', 'log_emergency', 'log_total_visits',
        'med_per_day', 'lab_per_day', 'clinical_intensity', 'high_lab_no_proc', 'diag_maxed',
        'utilization_x_stay', 'utilization_x_facility',
        'charlson_score', 'elixhauser_count', 'lace_score', 'lace_plus',
        'disch_is_home', 'disch_is_snf', 'disch_is_home_health', 'disch_is_facility', 'disch_is_ama',
        'is_emergency_adm', 'adm_from_er',
        'has_chf', 'has_renal', 'has_copd', 'cardiorenal_risk',
        'has_diab_comp', 'has_diab_acute', 'is_diab_uncontrolled',
        'active_med_count', 'med_change_count', 'on_insulin', 'insulin_up', 'insulin_down',
        'change_flag', 'diabetes_med_flag', 'a1c_tested', 'a1c_high', 'glu_high', 'therapeutic_inertia',
    ]

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d', 'disch_adm_pair'
    ]

    # 2. Native Categorical Encoding
    encoder = TabularEncoder(min_count=15)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)
    cat_indices = encoder.categorical_indices

    # 3. 5-Fold Stratified Tri-LightGBM Architecture
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_tr = len(y_train)
    n_eval_rows = len(eval_cols['gender'])
    prior_glob = float(np.mean(y_train))

    oof_1 = np.zeros(n_tr, dtype=float)
    oof_2 = np.zeros(n_tr, dtype=float)
    oof_3 = np.zeros(n_tr, dtype=float)

    eval_pred_1 = np.zeros(n_eval_rows, dtype=float)
    eval_pred_2 = np.zeros(n_eval_rows, dtype=float)
    eval_pred_3 = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # Model 1: Leaf-Wise Deep Explorer
        m1 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=35,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.8,
            reg_lambda=4.0,
            min_data_per_group=15,
            cat_smooth=15.0,
            cat_l2=10.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(x_tr, y_tr, categorical_feature=cat_indices)
        oof_1[val_idx] = m1.predict_proba(x_va)[:, 1]
        eval_pred_1 += m1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: Depth-Constrained Conservative Tree
        m2 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=20,
            min_child_samples=55,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=2.0,
            reg_lambda=8.0,
            min_data_per_group=20,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(x_tr, y_tr, categorical_feature=cat_indices)
        oof_2[val_idx] = m2.predict_proba(x_va)[:, 1]
        eval_pred_2 += m2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: Heavily Regularized Feature Subspacer
        m3 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.035,
            max_depth=4,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=3.0,
            reg_lambda=12.0,
            min_data_per_group=20,
            cat_smooth=25.0,
            cat_l2=15.0,
            random_state=242 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m3.fit(x_tr, y_tr, categorical_feature=cat_indices)
        oof_3[val_idx] = m3.predict_proba(x_va)[:, 1]
        eval_pred_3 += m3.predict_proba(x_eval)[:, 1] / 5.0

    # 4. Exact Out-of-Fold Simplex Grid Search for Global ROC-AUC Maximization
    best_auc = -1.0
    best_w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    for i in range(21):
        w1 = i * 0.05
        for j in range(21 - i):
            w2 = j * 0.05
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < -1e-6:
                continue
            w3 = max(0.0, w3)
            oof_blend = w1 * oof_1 + w2 * oof_2 + w3 * oof_3
            auc = roc_auc_score(y_train, oof_blend)
            if auc > best_auc:
                best_auc = auc
                best_w = (w1, w2, w3)

    final_eval = best_w[0] * eval_pred_1 + best_w[1] * eval_pred_2 + best_w[2] * eval_pred_3
    return np.clip(np.nan_to_num(final_eval, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
