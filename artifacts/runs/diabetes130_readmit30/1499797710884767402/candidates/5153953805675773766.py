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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import xgboost as xgb
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
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
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
        d['disch_is_snf'] = (disp == '3').astype(float)
        d['disch_is_hh'] = (disp == '6').astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5']).astype(float)

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)

        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])
        d['src_is_er'] = (adm_src == '7').astype(float)
        d['src_is_referral'] = np.isin(adm_src, ['1', '2']).astype(float)

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

        # Clinically Validated Indexes (LACE and HOSPITAL Scores)
        los = np.nan_to_num(d['time_in_hospital'], nan=3.0)
        l_score = np.zeros_like(los)
        l_score[los == 1] = 1.0
        l_score[los == 2] = 2.0
        l_score[los == 3] = 3.0
        l_score[(los >= 4) & (los <= 6)] = 4.0
        l_score[(los >= 7) & (los <= 13)] = 5.0
        l_score[los >= 14] = 7.0

        a_score = np.where(d['is_emergency_adm'] == 1.0, 3.0, 0.0)
        c_score = np.minimum(d['comorbidity_count'], 6.0)

        ev_cnt = np.nan_to_num(d['number_emergency'], nan=0.0)
        e_score = np.zeros_like(ev_cnt)
        e_score[ev_cnt == 1] = 1.0
        e_score[ev_cnt == 2] = 2.0
        e_score[ev_cnt == 3] = 3.0
        e_score[ev_cnt >= 4] = 4.0

        d['lace_score'] = l_score + a_score + c_score + e_score

        is_oncology = np.array([str(x).lower() == 'oncology' for x in d['medical_specialty']]).astype(float)
        has_proc = (d['num_procedures'] > 0).astype(float)
        los_gte_5 = (d['time_in_hospital'] >= 5).astype(float)
        
        inp_cnt = np.nan_to_num(d['number_inpatient'], nan=0.0)
        inp_score = np.zeros_like(inp_cnt)
        inp_score[inp_cnt == 1] = 1.0
        inp_score[inp_cnt == 2] = 2.0
        inp_score[inp_cnt >= 3] = 5.0
        
        d['hospital_score'] = (
            2.0 * is_oncology +
            1.0 * has_proc +
            1.0 * d['is_emergency_adm'] +
            inp_score +
            2.0 * los_gte_5
        )

        d['glu_high_no_change'] = ((d['glu_high'] == 1.0) & (d['change'] == 'No')).astype(float)
        d['elderly_high_risk'] = ((d['age_num'] >= 75.0) & (d['comorbidity_count'] >= 3.0)).astype(float)
        d['age_inp_interaction'] = d['age_num'] * np.log1p(inp_cnt)
        d['metabolic_triad'] = (has_diab_comp & has_ckd & has_htn).astype(float)
        d['cardiopulmonary_risk'] = (has_chf & has_copd).astype(float)
        d['neurovascular_risk'] = (has_stroke | has_pvd | has_cad).astype(float)

    # 2. Bayesian Smoothed Target Encoding (Out-of-Fold within train)
    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    individual_te_cols = [
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp'
    ] + med_cols

    te_features_tr = {col: np.zeros(n_tr, dtype=float) for col in individual_te_cols}
    tr_diag_max = np.zeros(n_tr, dtype=float)
    tr_diag_mean = np.zeros(n_tr, dtype=float)
    tr_diag_sum = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf.split(train['gender'], y_train):
        for col in individual_te_cols:
            smooth_val = 20.0
            if col in ['medical_specialty', 'payer_code']:
                smooth_val = 30.0
            elif col in ['diag_1', 'diag_2', 'diag_3', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
                smooth_val = 25.0
            
            col_map = compute_smoothed_map(train[col][tr_idx], y_train[tr_idx], smooth=smooth_val, prior=prior_glob)
            te_features_tr[col][val_idx] = np.array([col_map.get(safe_str(c), prior_glob) for c in train[col][val_idx]])

        # Pooled secondary diagnoses mapping
        pooled_diag = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_y = np.concatenate([y_train[tr_idx], y_train[tr_idx], y_train[tr_idx]])
        d_map = compute_smoothed_map(pooled_diag, pooled_y, smooth=25.0, prior=prior_glob)

        d1 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        d2 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        d3 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        tr_diag_max[val_idx] = np.maximum(np.maximum(d1, d2), d3)
        tr_diag_mean[val_idx] = (d1 + d2 + d3) / 3.0
        tr_diag_sum[val_idx] = d1 + d2 + d3

    te_features_ev = {}
    for col in individual_te_cols:
        smooth_val = 20.0
        if col in ['medical_specialty', 'payer_code']:
            smooth_val = 30.0
        elif col in ['diag_1', 'diag_2', 'diag_3', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
            smooth_val = 25.0
        col_map_all = compute_smoothed_map(train[col], y_train, smooth=smooth_val, prior=prior_glob)
        te_features_ev[col] = np.array([col_map_all.get(safe_str(c), prior_glob) for c in eval_cols[col]])

    pooled_diag_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    d_map_all = compute_smoothed_map(pooled_diag_all, pooled_y_all, smooth=25.0, prior=prior_glob)
    ed1 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    ed2 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    ed3 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_max'] = np.maximum(np.maximum(ed1, ed2), ed3)
    eval_cols['te_diag_mean'] = (ed1 + ed2 + ed3) / 3.0
    eval_cols['te_diag_sum'] = ed1 + ed2 + ed3

    for col in individual_te_cols:
        train[f'te_{col}'] = te_features_tr[col]
        eval_cols[f'te_{col}'] = te_features_ev[col]

    train['te_diag_max'] = tr_diag_max
    train['te_diag_mean'] = tr_diag_mean
    train['te_diag_sum'] = tr_diag_sum

    # 3. Non-Negative Clinical Phenotyping Matrix Factorization (NMPF) & Narrative Topics
    from sklearn.decomposition import NMF

    def build_phenotype_matrix(d):
        n = len(d['gender'])
        med_intensity_map = {'No': 0.0, 'Steady': 1.0, 'Down': 1.5, 'Up': 2.0}
        ins_val = np.array([med_intensity_map.get(str(x), 0.0) for x in d.get('insulin', ['No'] * n)])
        met_val = np.array([med_intensity_map.get(str(x), 0.0) for x in d.get('metformin', ['No'] * n)])

        # Sulfonylurea class max
        sulf_val = np.zeros(n, dtype=float)
        for s_col in ['glipizide', 'glyburide', 'glimepiride']:
            if s_col in d:
                s_arr = np.array([med_intensity_map.get(str(x), 0.0) for x in d[s_col]])
                sulf_val = np.maximum(sulf_val, s_arr)

        log_inp = np.log1p(np.maximum(0.0, d['number_inpatient']))
        log_er = np.log1p(np.maximum(0.0, d['number_emergency']))
        stay_norm = d['time_in_hospital'] / 14.0

        # Multi-modal non-negative feature slices: Comorbidities, Meds, Trajectory, and Bilinear Interactions
        cols = [
            d['has_chf'], d['has_ckd'], d['has_cad'], d['has_htn'], d['has_copd'],
            d['has_pvd'], d['has_stroke'], d['has_fluid_elect'], d['has_pneumonia'],
            d['has_diab_acute'], d['has_diab_comp'], d['cardiorenal_risk'],
            ins_val, met_val, sulf_val,
            d['active_meds'] / 10.0, d['num_med_changes'] / 4.0,
            d['disch_is_snf_hh'], d['disch_is_ama'], d['disch_is_transfer'],
            d['is_emergency_adm'], log_inp, log_er, stay_norm,
            d['has_chf'] * ins_val,
            d['has_ckd'] * ins_val,
            d['has_cad'] * ins_val,
            d['has_diab_acute'] * (d['num_med_changes'] > 0).astype(float),
            d['has_chf'] * log_inp,
            d['has_ckd'] * log_inp,
            d['lace_score'] / 19.0, d['hospital_score'] / 11.0,
            d['disch_is_snf'], d['disch_is_hh'],
            d['src_is_er'], d['src_is_referral'],
            d['metabolic_triad'], d['cardiopulmonary_risk'], d['neurovascular_risk'],
        ]
        mat = np.column_stack(cols).astype(float)
        return np.maximum(0.0, np.nan_to_num(mat, nan=0.0))

    X_pheno_tr = build_phenotype_matrix(train)
    X_pheno_ev = build_phenotype_matrix(eval_cols)

    nmf = NMF(n_components=8, init='nndsvda', max_iter=200, random_state=42)
    tr_nmf = nmf.fit_transform(X_pheno_tr)
    ev_nmf = nmf.transform(X_pheno_ev)

    pheno_names = [f'pheno_{k}' for k in range(8)]
    for k in range(8):
        train[f'pheno_{k}'] = tr_nmf[:, k].astype(float)
        eval_cols[f'pheno_{k}'] = ev_nmf[:, k].astype(float)

    # Topic SVD on narrative categorical corpus
    def build_narrative_corpus(d):
        n = len(d['gender'])
        med_pairs = [(m, d[m]) for m in med_cols if m in d]
        c = []
        for i in range(n):
            w = [
                f"d1_{d['diag_1_3d'][i]}", f"d2_{d['diag_2_3d'][i]}", f"d3_{d['diag_3_3d'][i]}",
                f"g1_{d['diag_1_grp'][i]}", f"g2_{d['diag_2_grp'][i]}", f"g3_{d['diag_3_grp'][i]}",
                f"disp_{safe_str(d['discharge_disposition_id'][i])}",
                f"src_{safe_str(d['admission_source_id'][i])}",
                f"spec_{safe_str(d['medical_specialty'][i])}",
                f"adm_{safe_str(d['admission_type_id'][i])}",
                f"age_{safe_str(d['age'][i])}",
            ]
            if d['number_inpatient'][i] > 0:
                w.append("prior_inp")
            if d['number_inpatient'][i] >= 2:
                w.append("freq_inp")
            if d['number_emergency'][i] > 0:
                w.append("prior_er")
            for m_name, m_vals in med_pairs:
                val = m_vals[i]
                if val is not None and val != 'No':
                    w.append(f"{m_name}_{val}")
            c.append(" ".join(w))
        return c

    tfidf = TfidfVectorizer(max_features=500, min_df=3, token_pattern=r'(?u)\b\S+\b')
    tr_tfidf = tfidf.fit_transform(build_narrative_corpus(train))
    ev_tfidf = tfidf.transform(build_narrative_corpus(eval_cols))

    svd = TruncatedSVD(n_components=12, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    topic_names = [f'topic_{k}' for k in range(12)]
    for k in range(12):
        train[f'topic_{k}'] = tr_svd[:, k].astype(float)
        eval_cols[f'topic_{k}'] = ev_svd[:, k].astype(float)

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'has_prior_inpatient', 'frequent_inpatient', 'med_per_day', 'lab_per_day',
        'proc_per_day', 'diag_per_day', 'service_intensity',
        'active_meds', 'num_med_changes', 'has_insulin', 'insulin_up', 'insulin_down',
        'change_flag', 'diabetes_med_flag',
        'a1c_tested', 'a1c_high', 'a1c_high_no_change', 'glu_high', 'glu_high_no_change',
        'disch_is_home', 'disch_is_snf_hh', 'disch_is_snf', 'disch_is_hh', 'disch_is_ama', 'disch_is_transfer',
        'is_emergency_adm', 'is_elective_adm', 'src_is_er', 'src_is_referral',
        'has_chf', 'has_ckd', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_diab_acute', 'has_diab_comp', 'cardiorenal_risk',
        'metabolic_triad', 'cardiopulmonary_risk', 'neurovascular_risk',
        'comorbidity_count', 'frailty_index', 'elderly_high_risk', 'age_inp_interaction',
        'lace_score', 'hospital_score'
    ]
    for col in individual_te_cols:
        extra_num.append(f'te_{col}')
    extra_num.extend(['te_diag_max', 'te_diag_mean', 'te_diag_sum'])
    extra_num += pheno_names + topic_names

    extra_cat = ['diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d']

    # 4. Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 5. XGBoost Specific Feature Preparation (Align target-encoded arrays)
    x_train_xgb = x_train.copy()
    x_eval_xgb = x_eval.copy()
    
    feat_to_idx = {name: i for i, name in enumerate(encoder.feature_names)}
    for idx in encoder.categorical_indices:
        feat_name = encoder.feature_names[idx]
        te_name = f'te_{feat_name}'
        if te_name in feat_to_idx:
            te_idx = feat_to_idx[te_name]
            x_train_xgb[:, idx] = x_train[:, te_idx]
            x_eval_xgb[:, idx] = x_eval[:, te_idx]

    # 6. Tri-Engine GBDT Models with 4-Fold Cross-Validation and Simplex ROC-AUC Optimization
    from sklearn.metrics import roc_auc_score

    skf_models = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)
    oof_hgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_lgb = np.zeros(n_eval_rows, dtype=float)
    eval_xgb = np.zeros(n_eval_rows, dtype=float)
    eval_hgb = np.zeros(n_eval_rows, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        x_tr_xgb, y_tr_xgb = x_train_xgb[tr_idx], y_train[tr_idx]
        x_va_xgb = x_train_xgb[val_idx]

        # LightGBM (Leaf-wise expansion)
        model_lgb = lgb.LGBMClassifier(
            n_estimators=250,
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
        eval_lgb += model_lgb.predict_proba(x_eval)[:, 1] / 4.0

        # HistGradientBoosting (Histogram binning)
        model_hgb = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=5.0,
            categorical_features=encoder.categorical_indices,
            random_state=42 + fold,
        )
        model_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = model_hgb.predict_proba(x_va)[:, 1]
        eval_hgb += model_hgb.predict_proba(x_eval)[:, 1] / 4.0

        # XGBoost (Depth-wise expansion on target encoded data)
        model_xgb = xgb.XGBClassifier(
            n_estimators=220,
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
        model_xgb.fit(x_tr_xgb, y_tr_xgb)
        oof_xgb[val_idx] = model_xgb.predict_proba(x_va_xgb)[:, 1]
        eval_xgb += model_xgb.predict_proba(x_eval_xgb)[:, 1] / 4.0

    # 7. Bayesian Logit Space Simplex Optimization for Maximum ROC-AUC
    def to_logit(p, eps=1e-6):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    z_oof_lgb = to_logit(oof_lgb)
    z_oof_xgb = to_logit(oof_xgb)
    z_oof_hgb = to_logit(oof_hgb)

    best_auc = -1.0
    best_w = (0.45, 0.30, 0.25)

    # Exhaustive fine-simplex grid search over blend weights
    for w1 in np.linspace(0.1, 0.8, 15):
        for w2 in np.linspace(0.1, 0.8, 15):
            if w1 + w2 < 0.95:
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    blend_oof = w1 * z_oof_lgb + w2 * z_oof_xgb + w3 * z_oof_hgb
                    auc = roc_auc_score(y_train, blend_oof)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = (w1, w2, w3)

    w1, w2, w3 = best_w
    z_eval = w1 * to_logit(eval_lgb) + w2 * to_logit(eval_xgb) + w3 * to_logit(eval_hgb)
    pred = 1.0 / (1.0 + np.exp(-z_eval))
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
