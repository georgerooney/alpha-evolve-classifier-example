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
) -> Sequence[float]:
    """Fit an ensemble on `train`/`y_train` and return readmission probabilities for `eval_cols`."""
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    def get_chapter(code):
        if 'icd9_group' in globals():
            return icd9_group(code)
        s = str(code).strip() if code is not None else ''
        if not s or s in ('None', '?'):
            return 'missing'
        if s.startswith(('V', 'E')):
            return 'other'
        try:
            val = float(s)
            if (390 <= val <= 459) or abs(val - 785) < 0.5:
                return 'circulatory'
            if (460 <= val <= 519) or abs(val - 786) < 0.5:
                return 'respiratory'
            if (520 <= val <= 579) or abs(val - 787) < 0.5:
                return 'digestive'
            if 250 <= val < 251:
                return 'diabetes'
            if 800 <= val <= 999:
                return 'injury'
            if 710 <= val <= 739:
                return 'musculoskeletal'
            if (580 <= val <= 629) or abs(val - 788) < 0.5:
                return 'genitourinary'
            if 140 <= val <= 239:
                return 'neoplasms'
        except ValueError:
            pass
        return 'other'

    def extract_core_features(d):
        n_rows = len(d['time_in_hospital'])
        num_f = {}
        cat_f = {}

        # 1. Continuous Demographics
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        age_raw = d.get('age', np.array(['[60-70)'] * n_rows))
        num_f['age_num'] = np.array([age_map.get(str(a).strip(), 65.0) if a is not None else 65.0 for a in age_raw], dtype=np.float64)

        # 2. Utilisation Intensity (Primary Driver of Readmission)
        num_inp = np.nan_to_num(d['number_inpatient'].astype(np.float64), nan=0.0)
        num_emg = np.nan_to_num(d['number_emergency'].astype(np.float64), nan=0.0)
        num_out = np.nan_to_num(d['number_outpatient'].astype(np.float64), nan=0.0)

        total_prior = num_inp + num_emg + num_out
        num_f['total_prior'] = total_prior
        num_f['inpatient_ratio'] = num_inp / (total_prior + 1.0)
        num_f['emergency_ratio'] = num_emg / (total_prior + 1.0)
        num_f['has_prior_inpatient'] = (num_inp > 0).astype(np.float64)
        num_f['has_prior_emergency'] = (num_emg > 0).astype(np.float64)

        # 3. Hospital Care & Stay Intensity
        los = np.maximum(np.nan_to_num(d['time_in_hospital'].astype(np.float64), nan=1.0), 1.0)
        num_lab = np.nan_to_num(d['num_lab_procedures'].astype(np.float64), nan=0.0)
        num_proc = np.nan_to_num(d['num_procedures'].astype(np.float64), nan=0.0)
        num_med = np.nan_to_num(d['num_medications'].astype(np.float64), nan=0.0)
        num_diag = np.maximum(np.nan_to_num(d['number_diagnoses'].astype(np.float64), nan=1.0), 1.0)

        num_f['med_per_day'] = num_med / los
        num_f['lab_per_day'] = num_lab / los
        num_f['proc_per_day'] = num_proc / los
        num_f['diag_per_day'] = num_diag / los
        num_f['inp_x_los'] = num_inp * los
        num_f['inp_x_diag'] = num_inp * num_diag

        # 4. Clinical Diagnosis Chapters and Chronic Comorbidities
        d1 = d['diag_1']
        d2 = d['diag_2']
        d3 = d['diag_3']

        g1 = np.array([get_chapter(c) for c in d1], dtype=object)
        g2 = np.array([get_chapter(c) for c in d2], dtype=object)
        g3 = np.array([get_chapter(c) for c in d3], dtype=object)

        cat_f['diag_1_group'] = g1
        cat_f['diag_2_group'] = g2
        cat_f['diag_3_group'] = g3

        cat_f['diag_1_pref'] = np.array([str(c).split('.')[0].strip() if (c is not None and str(c) != '?') else 'None' for c in d1], dtype=object)
        cat_f['diag_2_pref'] = np.array([str(c).split('.')[0].strip() if (c is not None and str(c) != '?') else 'None' for c in d2], dtype=object)
        cat_f['diag_3_pref'] = np.array([str(c).split('.')[0].strip() if (c is not None and str(c) != '?') else 'None' for c in d3], dtype=object)

        num_f['has_circulatory_diag'] = ((g1 == 'circulatory') | (g2 == 'circulatory') | (g3 == 'circulatory')).astype(np.float64)
        num_f['has_diabetes_diag'] = ((g1 == 'diabetes') | (g2 == 'diabetes') | (g3 == 'diabetes')).astype(np.float64)
        num_f['has_respiratory_diag'] = ((g1 == 'respiratory') | (g2 == 'respiratory') | (g3 == 'respiratory')).astype(np.float64)
        num_f['has_digestive_diag'] = ((g1 == 'digestive') | (g2 == 'digestive') | (g3 == 'digestive')).astype(np.float64)
        num_f['has_genitourinary_diag'] = ((g1 == 'genitourinary') | (g2 == 'genitourinary') | (g3 == 'genitourinary')).astype(np.float64)

        has_hf = np.zeros(n_rows, dtype=np.float64)
        has_ckd = np.zeros(n_rows, dtype=np.float64)
        has_cad = np.zeros(n_rows, dtype=np.float64)
        has_htn = np.zeros(n_rows, dtype=np.float64)
        has_copd = np.zeros(n_rows, dtype=np.float64)

        for col in (d1, d2, d3):
            for i, val in enumerate(col):
                if val is not None:
                    s = str(val).strip()
                    if s.startswith('428'):
                        has_hf[i] = 1.0
                    elif s.startswith(('585', '586')):
                        has_ckd[i] = 1.0
                    elif s.startswith(('410', '411', '412', '413', '414')):
                        has_cad[i] = 1.0
                    elif s.startswith(('401', '402', '403', '404', '405')):
                        has_htn[i] = 1.0
                    elif s.startswith(('490', '491', '492', '493', '494', '496')):
                        has_copd[i] = 1.0

        num_f['has_hf'] = has_hf
        num_f['has_ckd'] = has_ckd
        num_f['has_cad'] = has_cad
        num_f['has_htn'] = has_htn
        num_f['has_copd'] = has_copd
        num_f['comorbid_score'] = has_hf + has_ckd + has_cad + has_htn + has_copd

        # 5. Medication Regimen Intensity (Vectorised)
        med_names = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        meds_active = np.zeros(n_rows, dtype=np.float64)
        meds_changed = np.zeros(n_rows, dtype=np.float64)

        for m in med_names:
            if m in d:
                arr = d[m]
                meds_active += ((arr != 'No') & (arr != None)).astype(np.float64)
                meds_changed += ((arr == 'Up') | (arr == 'Down')).astype(np.float64)

        num_f['meds_active'] = meds_active
        num_f['meds_changed'] = meds_changed

        ins = d.get('insulin', np.array(['No'] * n_rows))
        num_f['insulin_active'] = np.array([1.0 if v in ('Steady', 'Up', 'Down') else 0.0 for v in ins], dtype=np.float64)
        num_f['insulin_changed'] = np.array([1.0 if v in ('Up', 'Down') else 0.0 for v in ins], dtype=np.float64)

        # 6. Admission and Discharge Risk Indicators
        disp = d.get('discharge_disposition_id', np.array([''] * n_rows))
        num_f['disch_home'] = np.array([1.0 if str(v) == '1' else 0.0 for v in disp], dtype=np.float64)
        num_f['disch_snf'] = np.array([1.0 if str(v) in ('3', '4') else 0.0 for v in disp], dtype=np.float64)
        num_f['disch_home_health'] = np.array([1.0 if str(v) == '6' else 0.0 for v in disp], dtype=np.float64)

        adm_type = d.get('admission_type_id', np.array([''] * n_rows))
        num_f['adm_emergency'] = np.array([1.0 if str(v) == '1' else 0.0 for v in adm_type], dtype=np.float64)

        adm_src = d.get('admission_source_id', np.array([''] * n_rows))
        num_f['adm_from_er'] = np.array([1.0 if str(v) == '7' else 0.0 for v in adm_src], dtype=np.float64)

        # 7. Lab Results and Diabetes Regimen
        a1c = d.get('A1Cresult', np.array(['None'] * n_rows))
        num_f['a1c_tested'] = np.array([0.0 if v in ('None', None) else 1.0 for v in a1c], dtype=np.float64)
        num_f['a1c_high'] = np.array([1.0 if v in ('>7', '>8') else 0.0 for v in a1c], dtype=np.float64)

        glu = d.get('max_glu_serum', np.array(['None'] * n_rows))
        num_f['glu_tested'] = np.array([0.0 if v in ('None', None) else 1.0 for v in glu], dtype=np.float64)
        num_f['glu_high'] = np.array([1.0 if v in ('>200', '>300') else 0.0 for v in glu], dtype=np.float64)

        chg = d.get('change', np.array(['No'] * n_rows))
        num_f['has_change'] = np.array([1.0 if v == 'Ch' else 0.0 for v in chg], dtype=np.float64)

        dmed = d.get('diabetesMed', np.array(['No'] * n_rows))
        num_f['has_dmed'] = np.array([1.0 if v == 'Yes' else 0.0 for v in dmed], dtype=np.float64)

        return num_f, cat_f

    # 1. Feature Extraction
    train_num, train_cat = extract_core_features(train)
    eval_num, eval_cat = extract_core_features(eval_cols)

    train_data = dict(train)
    eval_data = dict(eval_cols)

    for k, v in train_num.items():
        train_data[k] = v
        eval_data[k] = eval_num[k]

    for k, v in train_cat.items():
        train_data[k] = v
        eval_data[k] = eval_cat[k]

    extra_numeric = tuple(train_num.keys())
    extra_categorical = tuple(train_cat.keys())

    # 2. Native Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task, train_data,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical
    )
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 3. Two Complementary LightGBM Architectures (Leaf-Wise & Depth-Constrained)
    cfg_leaf = {
        'n_estimators': 320,
        'learning_rate': 0.03,
        'num_leaves': 22,
        'max_depth': -1,
        'min_child_samples': 45,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.2,
        'reg_lambda': 3.0,
        'n_jobs': 1,
        'verbose': -1,
    }

    cfg_depth = {
        'n_estimators': 280,
        'learning_rate': 0.035,
        'num_leaves': 16,
        'max_depth': 5,
        'min_child_samples': 55,
        'subsample': 0.85,
        'subsample_freq': 1,
        'colsample_bytree': 0.65,
        'reg_alpha': 0.5,
        'reg_lambda': 5.0,
        'n_jobs': 1,
        'verbose': -1,
    }

    # 4. 8-Fold Stratified Cross-Validation Bagging Ensemble
    n_splits = 8
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    eval_preds = np.zeros(len(x_eval), dtype=np.float64)

    for fold_idx, (tr_idx, _) in enumerate(skf.split(x_train, y_train)):
        xt_fold = x_train[tr_idx]
        yt_fold = y_train[tr_idx]

        m_leaf = lgb.LGBMClassifier(**cfg_leaf, random_state=100 + fold_idx)
        m_leaf.fit(xt_fold, yt_fold, categorical_feature=cat_indices)
        p_leaf = m_leaf.predict_proba(x_eval)[:, 1]

        m_depth = lgb.LGBMClassifier(**cfg_depth, random_state=200 + fold_idx)
        m_depth.fit(xt_fold, yt_fold, categorical_feature=cat_indices)
        p_depth = m_depth.predict_proba(x_eval)[:, 1]

        eval_preds += (0.55 * p_leaf + 0.45 * p_depth) / n_splits

    return np.clip(np.nan_to_num(eval_preds, nan=0.114), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
