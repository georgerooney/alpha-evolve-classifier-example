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

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    n_tr = len(y_train)
    n_ev = len(eval_cols['time_in_hospital'])

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities via icd9_comorbidities
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in cm_tr:
        train_data[k] = cm_tr[k].astype(np.float64)
        eval_data[k] = cm_ev[k].astype(np.float64)
        extra_numeric.append(k)

    # Weighted Charlson comorbidity index & critical interactions
    def calc_weighted_cci(cm):
        return (
            cm['cm_mi'] + cm['cm_chf'] + cm['cm_pvd'] + cm['cm_cerebrovascular']
            + cm['cm_dementia'] + cm['cm_pulmonary'] + cm['cm_rheumatic']
            + cm['cm_peptic_ulcer'] + cm['cm_liver']
            + 2.0 * cm['cm_hemiplegia'] + 2.0 * cm['cm_renal']
            + 2.0 * cm['cm_cancer'] + 2.0 * cm['cm_diabetes_complicated']
            + 6.0 * cm['cm_metastatic'] + 6.0 * cm['cm_hiv']
        ).astype(np.float64)

    train_data['weighted_cci'] = calc_weighted_cci(cm_tr)
    eval_data['weighted_cci'] = calc_weighted_cci(cm_ev)
    extra_numeric.append('weighted_cci')

    train_data['cardiorenal'] = (cm_tr['cm_chf'] * cm_tr['cm_renal']).astype(np.float64)
    eval_data['cardiorenal'] = (cm_ev['cm_chf'] * cm_ev['cm_renal']).astype(np.float64)
    extra_numeric.append('cardiorenal')

    train_data['pulmonary_chf'] = (cm_tr['cm_pulmonary'] * cm_tr['cm_chf']).astype(np.float64)
    eval_data['pulmonary_chf'] = (cm_ev['cm_pulmonary'] * cm_ev['cm_chf']).astype(np.float64)
    extra_numeric.append('pulmonary_chf')

    train_data['diabetic_renal'] = (cm_tr['cm_diabetes_complicated'] * cm_tr['cm_renal']).astype(np.float64)
    eval_data['diabetic_renal'] = (cm_ev['cm_diabetes_complicated'] * cm_ev['cm_renal']).astype(np.float64)
    extra_numeric.append('diabetic_renal')

    train_data['cardio_metabolic'] = (cm_tr['cm_chf'] * cm_tr['cm_diabetes_complicated']).astype(np.float64)
    eval_data['cardio_metabolic'] = (cm_ev['cm_chf'] * cm_ev['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cardio_metabolic')

    train_data['severe_multimorbidity'] = (cm_tr['cm_count'] >= 3.0).astype(np.float64)
    eval_data['severe_multimorbidity'] = (cm_ev['cm_count'] >= 3.0).astype(np.float64)
    extra_numeric.append('severe_multimorbidity')

    # 2. ICD-9 Chapter groups
    for diag_field in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f"{diag_field}_group"
        train_data[grp_name] = np.array([icd9_group(c) for c in train[diag_field]], dtype=object)
        eval_data[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_field]], dtype=object)
        extra_categorical.append(grp_name)

    # 3. Composites & Interaction Categories
    train_data['disch_admit_combo'] = np.array(
        [f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_type_id'])],
        dtype=object
    )
    eval_data['disch_admit_combo'] = np.array(
        [f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_type_id'])],
        dtype=object
    )
    extra_categorical.append('disch_admit_combo')

    train_data['diag1_disch_combo'] = np.array(
        [f"{g}_{d}" for g, d in zip(train_data['diag_1_group'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_data['diag1_disch_combo'] = np.array(
        [f"{g}_{d}" for g, d in zip(eval_data['diag_1_group'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    extra_categorical.append('diag1_disch_combo')

    train_data['age_disch_combo'] = np.array(
        [f"{a}_{d}" for a, d in zip(train['age'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_data['age_disch_combo'] = np.array(
        [f"{a}_{d}" for a, d in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    extra_categorical.append('age_disch_combo')

    # 4. Out-of-fold target encoding of high-cardinality columns & interactions
    te_columns = [
        'diag_1', 'diag_2', 'diag_3',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code',
        'diag_1_group',
        'diag_2_group',
        'diag_3_group',
        'age',
        'disch_admit_combo',
        'diag1_disch_combo',
        'age_disch_combo',
    ]
    for col in te_columns:
        te_tr, te_ev = oof_target_encode(
            train_data[col], y_train, eval_data[col], n_folds=5, smoothing=20.0, seed=42
        )
        col_te = f"{col}_te"
        train_data[col_te] = te_tr.astype(np.float64)
        eval_data[col_te] = te_ev.astype(np.float64)
        extra_numeric.append(col_te)

    # Summary of diagnosis risk
    d1_te_tr = train_data['diag_1_te']
    d2_te_tr = train_data['diag_2_te']
    d3_te_tr = train_data['diag_3_te']
    train_data['diag_te_max'] = np.maximum(d1_te_tr, np.maximum(d2_te_tr, d3_te_tr))
    train_data['diag_te_min'] = np.minimum(d1_te_tr, np.minimum(d2_te_tr, d3_te_tr))
    train_data['diag_te_mean'] = (d1_te_tr + d2_te_tr + d3_te_tr) / 3.0
    train_data['diag_te_sum'] = d1_te_tr + d2_te_tr + d3_te_tr
    train_data['diag_te_range'] = train_data['diag_te_max'] - train_data['diag_te_min']

    d1_te_ev = eval_data['diag_1_te']
    d2_te_ev = eval_data['diag_2_te']
    d3_te_ev = eval_data['diag_3_te']
    eval_data['diag_te_max'] = np.maximum(d1_te_ev, np.maximum(d2_te_ev, d3_te_ev))
    eval_data['diag_te_min'] = np.minimum(d1_te_ev, np.minimum(d2_te_ev, d3_te_ev))
    eval_data['diag_te_mean'] = (d1_te_ev + d2_te_ev + d3_te_ev) / 3.0
    eval_data['diag_te_sum'] = d1_te_ev + d2_te_ev + d3_te_ev
    eval_data['diag_te_range'] = eval_data['diag_te_max'] - eval_data['diag_te_min']

    for te_sum_name in ['diag_te_max', 'diag_te_min', 'diag_te_mean', 'diag_te_sum', 'diag_te_range']:
        extra_numeric.append(te_sum_name)

    # 5. Tensor Decomposition: Multi-way Clinical Phenotype Embedding via TruncatedSVD
    def build_clinical_corpus(d, g1_arr, g2_arr, g3_arr):
        n_rows = len(d['time_in_hospital'])
        diag1 = d['diag_1']
        diag2 = d['diag_2']
        diag3 = d['diag_3']
        insulin = d['insulin']
        change = d['change']
        disposition = d['discharge_disposition_id']
        corpus = []
        for i in range(n_rows):
            toks = []
            c1 = str(diag1[i]) if diag1[i] is not None and diag1[i] != '?' else 'none'
            c2 = str(diag2[i]) if diag2[i] is not None and diag2[i] != '?' else 'none'
            c3 = str(diag3[i]) if diag3[i] is not None and diag3[i] != '?' else 'none'
            g1 = str(g1_arr[i])
            g2 = str(g2_arr[i])
            g3 = str(g3_arr[i])

            toks.append(f"d1_{c1}")
            toks.append(f"d1_{c1}")
            toks.append(f"d2_{c2}")
            toks.append(f"d3_{c3}")
            toks.append(f"any_{c1}")
            toks.append(f"any_{c2}")
            toks.append(f"any_{c3}")

            toks.append(f"g1_{g1}")
            toks.append(f"g2_{g2}")
            toks.append(f"g3_{g3}")
            toks.append(f"pair_{'_'.join(sorted([g1, g2]))}")
            toks.append(f"pair_{'_'.join(sorted([g1, g3]))}")
            toks.append(f"pair_{'_'.join(sorted([g2, g3]))}")

            toks.append(f"ins_{insulin[i]}_g1_{g1}")
            toks.append(f"chg_{change[i]}_ins_{insulin[i]}")
            toks.append(f"disp_{disposition[i]}_g1_{g1}")

            corpus.append(" ".join(toks))
        return corpus

    corpus_tr = build_clinical_corpus(train, train_data['diag_1_group'], train_data['diag_2_group'], train_data['diag_3_group'])
    corpus_ev = build_clinical_corpus(eval_cols, eval_data['diag_1_group'], eval_data['diag_2_group'], eval_data['diag_3_group'])

    tfidf = TfidfVectorizer(max_features=1000, min_df=5, sublinear_tf=True, token_pattern=r'\S+')
    x_tfidf_tr = tfidf.fit_transform(corpus_tr)
    x_tfidf_ev = tfidf.transform(corpus_ev)

    n_svd_comp = min(10, max(2, x_tfidf_tr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_svd_comp, algorithm='randomized', n_iter=5, random_state=42)
    z_tr = svd.fit_transform(x_tfidf_tr)
    z_ev = svd.transform(x_tfidf_ev)

    for comp_idx in range(n_svd_comp):
        comp_name = f"svd_phenotype_{comp_idx}"
        train_data[comp_name] = z_tr[:, comp_idx].astype(np.float64)
        eval_data[comp_name] = z_ev[:, comp_idx].astype(np.float64)
        extra_numeric.append(comp_name)

    train_data['svd_phenotype_norm'] = np.linalg.norm(z_tr, axis=1).astype(np.float64)
    eval_data['svd_phenotype_norm'] = np.linalg.norm(z_ev, axis=1).astype(np.float64)
    extra_numeric.append('svd_phenotype_norm')

    train_data['svd_phenotype_max'] = np.max(z_tr, axis=1).astype(np.float64)
    eval_data['svd_phenotype_max'] = np.max(z_ev, axis=1).astype(np.float64)
    extra_numeric.append('svd_phenotype_max')

    # 6. Prior utilisation features
    def make_utilization_features(d):
        inpatient = np.nan_to_num(d['number_inpatient'], nan=0.0).astype(np.float64)
        emergency = np.nan_to_num(d['number_emergency'], nan=0.0).astype(np.float64)
        outpatient = np.nan_to_num(d['number_outpatient'], nan=0.0).astype(np.float64)
        total_vis = inpatient + emergency + outpatient
        return {
            'has_prior_inpatient': (inpatient > 0).astype(np.float64),
            'frequent_inpatient': (inpatient >= 2).astype(np.float64),
            'has_emergency': (emergency > 0).astype(np.float64),
            'has_outpatient': (outpatient > 0).astype(np.float64),
            'total_prior_visits': total_vis,
            'inpatient_visit_fraction': (inpatient + 0.1) / (total_vis + 1.0),
            'emergency_visit_fraction': (emergency + 0.1) / (total_vis + 1.0),
        }

    util_tr = make_utilization_features(train)
    util_ev = make_utilization_features(eval_cols)
    for k in util_tr:
        train_data[k] = util_tr[k]
        eval_data[k] = util_ev[k]
        extra_numeric.append(k)

    # 6. Stay intensity & complexity ratios
    def make_intensity_features(d):
        los = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 1.0).astype(np.float64)
        meds = np.nan_to_num(d['num_medications'], nan=0.0).astype(np.float64)
        labs = np.nan_to_num(d['num_lab_procedures'], nan=0.0).astype(np.float64)
        procs = np.nan_to_num(d['num_procedures'], nan=0.0).astype(np.float64)
        diags = np.maximum(np.nan_to_num(d['number_diagnoses'], nan=1.0), 1.0).astype(np.float64)
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0).astype(np.float64)
        return {
            'meds_per_day': meds / los,
            'labs_per_day': labs / los,
            'procs_per_day': procs / los,
            'meds_per_diag': meds / diags,
            'labs_per_med': labs / np.maximum(meds, 1.0),
            'los_times_inpatient': los * inp,
            'diags_times_inpatient': diags * inp,
            'clinical_complexity': np.log1p(los) * np.log1p(meds) * np.log1p(diags),
        }

    int_tr = make_intensity_features(train)
    int_ev = make_intensity_features(eval_cols)
    for k in int_tr:
        train_data[k] = int_tr[k]
        eval_data[k] = int_ev[k]
        extra_numeric.append(k)

    # 7. Demographics (numeric age midpoint)
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    age_tr = np.array([age_map.get(a, 60.0) for a in train['age']], dtype=np.float64)
    age_ev = np.array([age_map.get(a, 60.0) for a in eval_cols['age']], dtype=np.float64)
    train_data['age_num'] = age_tr
    eval_data['age_num'] = age_ev
    extra_numeric.append('age_num')

    train_data['is_elderly'] = (age_tr >= 75.0).astype(np.float64)
    eval_data['is_elderly'] = (age_ev >= 75.0).astype(np.float64)
    extra_numeric.append('is_elderly')

    train_data['gender_is_female'] = (train['gender'] == 'Female').astype(np.float64)
    eval_data['gender_is_female'] = (eval_cols['gender'] == 'Female').astype(np.float64)
    extra_numeric.append('gender_is_female')

    train_data['elderly_snf'] = train_data['is_elderly'] * train_data['disch_snf']
    eval_data['elderly_snf'] = eval_data['is_elderly'] * eval_data['disch_snf']
    extra_numeric.append('elderly_snf')

    # 8. Medications patterns
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in med_names if m in train]

    def extract_med_features(d, n_rows):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        for m in avail_meds:
            col = d[m]
            is_up = (col == 'Up')
            is_down = (col == 'Down')
            is_steady = (col == 'Steady')
            up_cnt += is_up.astype(np.float64)
            down_cnt += is_down.astype(np.float64)
            active_cnt += (is_up | is_down | is_steady).astype(np.float64)

        ins = d['insulin']
        ins_up = (ins == 'Up').astype(np.float64)
        ins_down = (ins == 'Down').astype(np.float64)
        ins_steady = (ins == 'Steady').astype(np.float64)
        ins_active = ((ins == 'Up') | (ins == 'Down') | (ins == 'Steady')).astype(np.float64)
        ins_changed = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        ins_num = np.zeros(n_rows, dtype=np.float64)
        ins_num[ins == 'Down'] = -1.0
        ins_num[ins == 'Steady'] = 1.0
        ins_num[ins == 'Up'] = 2.0

        met = d['metformin']
        met_active = ((met == 'Up') | (met == 'Down') | (met == 'Steady')).astype(np.float64)

        chg = (d['change'] == 'Ch').astype(np.float64)
        dm_med = (d['diabetesMed'] == 'Yes').astype(np.float64)
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0).astype(np.float64)

        return {
            'num_active_meds': active_cnt,
            'num_meds_up': up_cnt,
            'num_meds_down': down_cnt,
            'num_meds_changed': up_cnt + down_cnt,
            'insulin_is_up': ins_up,
            'insulin_is_down': ins_down,
            'insulin_is_steady': ins_steady,
            'insulin_is_active': ins_active,
            'insulin_is_changed': ins_changed,
            'insulin_numeric': ins_num,
            'insulin_and_oral': (ins_active * (active_cnt > 1.0)).astype(np.float64),
            'metformin_is_active': met_active,
            'med_change_flag': chg,
            'has_diabetes_med': dm_med,
            'change_with_prior_inpatient': (chg * (inp > 0.0)).astype(np.float64),
            'med_switch_risk': ((up_cnt > 0.0) & (down_cnt > 0.0)).astype(np.float64),
        }

    med_tr = extract_med_features(train, n_tr)
    med_ev = extract_med_features(eval_cols, n_ev)
    for k in med_tr:
        train_data[k] = med_tr[k]
        eval_data[k] = med_ev[k]
        extra_numeric.append(k)

    # 9. Lab results and disposition flags
    def extract_lab_and_disp_features(d):
        a1c = d['A1Cresult']
        a1c_tested = ((a1c != 'None') & (a1c != None)).astype(np.float64)
        a1c_high = ((a1c == '>8') | (a1c == '>7')).astype(np.float64)
        a1c_gt8 = (a1c == '>8').astype(np.float64)

        glu = d['max_glu_serum']
        glu_tested = ((glu != 'None') & (glu != None)).astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)

        disch = d['discharge_disposition_id']
        disch_home = (disch == '1').astype(np.float64)
        disch_snf = (disch == '3').astype(np.float64)
        disch_homehealth = (disch == '6').astype(np.float64)
        disch_ama = (disch == '7').astype(np.float64)
        disch_high_risk = np.isin(disch, ['3', '6', '7', '22', '28']).astype(np.float64)

        admit = d['admission_type_id']
        admit_er = (admit == '1').astype(np.float64)
        admit_elective = (admit == '3').astype(np.float64)

        source = d['admission_source_id']
        source_er = (source == '7').astype(np.float64)
        source_transfer = np.isin(source, ['4', '5', '6']).astype(np.float64)

        payer_miss = np.isin(d['payer_code'], [None, '?']).astype(np.float64)
        spec_miss = np.isin(d['medical_specialty'], [None, '?']).astype(np.float64)

        return {
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'a1c_gt8': a1c_gt8,
            'glu_tested': glu_tested,
            'glu_high': glu_high,
            'disch_home': disch_home,
            'disch_snf': disch_snf,
            'disch_homehealth': disch_homehealth,
            'disch_ama': disch_ama,
            'disch_high_risk': disch_high_risk,
            'admit_er': admit_er,
            'admit_elective': admit_elective,
            'source_er': source_er,
            'source_transfer': source_transfer,
            'payer_miss': payer_miss,
            'spec_miss': spec_miss,
        }

    lab_tr = extract_lab_and_disp_features(train)
    lab_ev = extract_lab_and_disp_features(eval_cols)
    for k in lab_tr:
        train_data[k] = lab_tr[k]
        eval_data[k] = lab_ev[k]
        extra_numeric.append(k)

    # De-duplicate names
    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 10. Encoding: min_count=20 preserves informative granular categories
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 11. Multi-Family Diverse Ensemble (LightGBM + XGBoost)
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=130,
            subsample=0.72,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_lambda=2.0,
            reg_alpha=0.1,
            cat_smooth=45.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=340,
            learning_rate=0.025,
            num_leaves=46,
            min_child_samples=85,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=5.0,
            reg_alpha=0.5,
            cat_smooth=65.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=220,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.68,
            reg_lambda=0.5,
            reg_alpha=0.05,
            cat_smooth=30.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.032,
            num_leaves=28,
            min_child_samples=110,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.48,
            reg_lambda=2.5,
            reg_alpha=0.25,
            cat_smooth=55.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    lgb_preds = np.zeros(n_ev, dtype=np.float64)
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_models)

    # Heterogeneous model: Histogram-based XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=240,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=5.0,
        subsample=0.80,
        colsample_bytree=0.60,
        reg_alpha=0.5,
        reg_lambda=3.0,
        tree_method='hist',
        random_state=999,
        n_jobs=1,
        verbosity=0,
        eval_metric='logloss',
    )

    try:
        xgb_model.fit(x_train, y_train)
        xgb_preds = xgb_model.predict_proba(x_eval)[:, 1]
        final_preds = 0.80 * lgb_preds + 0.20 * xgb_preds
    except Exception:
        final_preds = lgb_preds

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
