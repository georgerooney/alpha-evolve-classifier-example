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
    """Fit on `train`/`y_train` and return readmission probabilities using Bio-Gravitational Syndromic Phenotyping & Rank Consensus."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge
    from scipy.stats import rankdata

    non_med_cols = {
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    }
    med_cols = [col for col in task.categorical_columns if col not in non_med_cols]

    sulfonylureas = {'glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide'}
    tzds = {'pioglitazone', 'rosiglitazone', 'troglitazone'}
    metformins = {'metformin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'}

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

    def make_dyad(c1, c2):
        s1, s2 = clean_icd(c1), clean_icd(c2)
        return f"{s1}_{s2}" if s1 < s2 else f"{s2}_{s1}"

    def make_grp_dyad(g1, g2):
        s1, s2 = str(g1), str(g2)
        return f"{s1}_{s2}" if s1 < s2 else f"{s2}_{s1}"

    # 1. Feature Engineering on both sets
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        # Utilization & clinical intensity metrics
        total_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['total_visits'] = total_v
        d['inpatient_ratio'] = d['number_inpatient'] / (total_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (total_v + 1.0)
        d['acute_util_ratio'] = (d['number_inpatient'] + d['number_emergency']) / (total_v + 1.0)
        d['prior_util_severity'] = (3.0 * d['number_inpatient'] + 1.5 * d['number_emergency'] + 0.5 * d['number_outpatient'])
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)

        stay = d['time_in_hospital'] + 1.0
        d['med_per_day'] = d['num_medications'] / stay
        d['lab_per_day'] = d['num_lab_procedures'] / stay
        d['proc_per_day'] = d['num_procedures'] / stay
        d['diag_per_day'] = d['number_diagnoses'] / stay
        d['service_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / stay
        d['lab_per_diag'] = d['num_lab_procedures'] / (d['number_diagnoses'] + 1.0)
        d['proc_per_diag'] = d['num_procedures'] / (d['number_diagnoses'] + 1.0)
        d['iatrogenic_index'] = (d['num_medications'] * d['time_in_hospital']) / 50.0

        # Medication regimen, class mechanisms & titration dynamics
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        med_ups = np.zeros(n_rows, dtype=float)
        med_downs = np.zeros(n_rows, dtype=float)
        med_steady = np.zeros(n_rows, dtype=float)
        has_su = np.zeros(n_rows, dtype=float)
        has_tzd = np.zeros(n_rows, dtype=float)
        has_met = np.zeros(n_rows, dtype=float)

        for col in med_cols:
            if col in d:
                val = d[col]
                is_on = (val != 'No').astype(float)
                active += is_on
                changes += np.isin(val, ['Up', 'Down']).astype(float)
                med_ups += (val == 'Up').astype(float)
                med_downs += (val == 'Down').astype(float)
                med_steady += (val == 'Steady').astype(float)

                if col in sulfonylureas:
                    has_su = np.maximum(has_su, is_on)
                if col in tzds:
                    has_tzd = np.maximum(has_tzd, is_on)
                if col in metformins:
                    has_met = np.maximum(has_met, is_on)

        d['active_meds'] = active
        d['num_med_changes'] = changes
        d['med_ups'] = med_ups
        d['med_downs'] = med_downs
        d['med_steady'] = med_steady
        d['med_instability_ratio'] = (med_ups + med_downs) / (active + 1.0)
        d['has_su'] = has_su
        d['has_tzd'] = has_tzd
        d['has_met'] = has_met

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['has_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Admission, discharge and lab biomarkers
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf_hh'] = np.isin(disp, ['3', '6', '22', '28']).astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5']).astype(float)

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = np.isin(adm_type, ['1', '2', '7']).astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)

        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])
        d['adm_from_er'] = (adm_src == '7').astype(float)
        d['adm_from_transfer'] = np.isin(adm_src, ['4', '5', '6']).astype(float)

        d['a1c_tested'] = (d['A1Cresult'] != 'None').astype(float)
        d['a1c_high'] = np.isin(d['A1Cresult'], ['>7', '>8']).astype(float)
        d['a1c_very_high'] = (d['A1Cresult'] == '>8').astype(float)
        d['glu_tested'] = (d['max_glu_serum'] != 'None').astype(float)
        d['glu_high'] = np.isin(d['max_glu_serum'], ['>200', '>300']).astype(float)

        # Diagnoses: groups, 3-digit categories & clinical syndromic dyads
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        d['diag_dyad_12'] = np.array([make_dyad(a, b) for a, b in zip(d['diag_1'], d['diag_2'])], dtype=object)
        d['diag_dyad_13'] = np.array([make_dyad(a, b) for a, b in zip(d['diag_1'], d['diag_3'])], dtype=object)
        d['diag_dyad_23'] = np.array([make_dyad(a, b) for a, b in zip(d['diag_2'], d['diag_3'])], dtype=object)

        d['grp_dyad_12'] = np.array([make_grp_dyad(a, b) for a, b in zip(d['diag_1_grp'], d['diag_2_grp'])], dtype=object)
        d['grp_dyad_13'] = np.array([make_grp_dyad(a, b) for a, b in zip(d['diag_1_grp'], d['diag_3_grp'])], dtype=object)
        d['grp_dyad_23'] = np.array([make_grp_dyad(a, b) for a, b in zip(d['diag_2_grp'], d['diag_3_grp'])], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_ckd = np.isin(c1, ['585', '586']) | np.isin(c2, ['585', '586']) | np.isin(c3, ['585', '586'])
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_diab_comp = np.zeros(n_rows, dtype=bool)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            has_diab_comp |= np.array([str(x).startswith('250') and str(x) not in ['250', '250.0', '250.00', '250.02'] for x in d[col]])

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_diab_comp'] = has_diab_comp.astype(float)
        comorb = d['has_chf'] + d['has_ckd'] + d['has_cad'] + d['has_htn'] + d['has_copd'] + d['has_diab_comp']
        d['comorbidity_count'] = comorb
        d['frailty_index'] = comorb * (d['age_num'] / 50.0) * (d['number_inpatient'] + 1.0)

        # Pharmacological contraindications & clinical hazards (Beers Criteria)
        d['tzd_chf_hazard'] = d['has_tzd'] * d['has_chf']
        d['met_ckd_hazard'] = d['has_met'] * d['has_ckd']
        d['su_elderly_hazard'] = d['has_su'] * (d['age_num'] >= 75.0).astype(float)
        d['complex_multidrug'] = ((d['has_su'] + d['has_tzd'] + d['has_met'] + d['has_insulin']) >= 3).astype(float)

    # 2. Biological Age Acceleration Residual (Gompertz-Klemera Frailty Estimator)
    bio_features = [
        'total_visits', 'number_inpatient', 'number_emergency',
        'time_in_hospital', 'num_medications', 'num_lab_procedures',
        'num_procedures', 'number_diagnoses', 'comorbidity_count', 'active_meds'
    ]
    bio_mat_train = np.column_stack([np.nan_to_num(train[f], nan=0.0) for f in bio_features])
    bio_mat_eval = np.column_stack([np.nan_to_num(eval_cols[f], nan=0.0) for f in bio_features])

    # 3. Clinical Holographic Topic SVD
    def build_narrative_corpus(d):
        n = len(d['gender'])
        med_pairs = [(m, d[m]) for m in med_cols if m in d]
        c = []
        for i in range(n):
            w = [
                f"d1_{d['diag_1_3d'][i]}", f"d2_{d['diag_2_3d'][i]}", f"d3_{d['diag_3_3d'][i]}",
                f"g1_{d['diag_1_grp'][i]}", f"g2_{d['diag_2_grp'][i]}", f"g3_{d['diag_3_grp'][i]}",
                f"dy12_{d['diag_dyad_12'][i]}", f"dy13_{d['diag_dyad_13'][i]}",
                f"disp_{safe_str(d['discharge_disposition_id'][i])}",
                f"src_{safe_str(d['admission_source_id'][i])}",
                f"adm_{safe_str(d['admission_type_id'][i])}",
                f"spec_{safe_str(d['medical_specialty'][i])}",
                f"age_{safe_str(d['age'][i])}",
            ]
            if d['number_inpatient'][i] > 0:
                w.append("prior_inp")
            if d['number_inpatient'][i] >= 2:
                w.append("freq_inp")
            if d['number_emergency'][i] > 0:
                w.append("prior_er")
            if d['has_chf'][i] > 0:
                w.append("has_chf")
            if d['has_ckd'][i] > 0:
                w.append("has_ckd")
            for m_name, m_vals in med_pairs:
                val = m_vals[i]
                if val is not None and val != 'No':
                    w.append(f"{m_name}_{val}")
            c.append(" ".join(w))
        return c

    tfidf = TfidfVectorizer(max_features=160, min_df=5, token_pattern=r'(?u)\b\S+\b')
    tr_tfidf = tfidf.fit_transform(build_narrative_corpus(train))
    ev_tfidf = tfidf.transform(build_narrative_corpus(eval_cols))

    svd = TruncatedSVD(n_components=8, random_state=42)
    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    topic_names = [f'topic_{k}' for k in range(8)]
    for k in range(8):
        train[f'topic_{k}'] = tr_svd[:, k].astype(float)
        eval_cols[f'topic_{k}'] = ev_svd[:, k].astype(float)

    # 4. Bayesian Smoothed Target Encoding & Out-of-Fold Prototype Gravitational Dynamics
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

    tr_diag_max = np.zeros(n_tr, dtype=float)
    tr_diag_mean = np.zeros(n_tr, dtype=float)
    tr_diag_sum = np.zeros(n_tr, dtype=float)
    tr_dyad_max = np.zeros(n_tr, dtype=float)
    tr_dyad_mean = np.zeros(n_tr, dtype=float)
    tr_disp = np.zeros(n_tr, dtype=float)
    tr_spec = np.zeros(n_tr, dtype=float)
    tr_src = np.zeros(n_tr, dtype=float)
    tr_bio_age = np.zeros(n_tr, dtype=float)
    tr_grav_pull = np.zeros(n_tr, dtype=float)
    tr_dist_pos = np.zeros(n_tr, dtype=float)
    tr_dist_neg = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf.split(train['gender'], y_train):
        # Target Encoding: Diagnoses
        pooled_diag = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_y = np.concatenate([y_train[tr_idx], y_train[tr_idx], y_train[tr_idx]])
        d_map = compute_smoothed_map(pooled_diag, pooled_y, smooth=25.0, prior=prior_glob)

        d1 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_1_3d'][val_idx]])
        d2 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_2_3d'][val_idx]])
        d3 = np.array([d_map.get(safe_str(c), prior_glob) for c in train['diag_3_3d'][val_idx]])
        tr_diag_max[val_idx] = np.maximum(np.maximum(d1, d2), d3)
        tr_diag_mean[val_idx] = (d1 + d2 + d3) / 3.0
        tr_diag_sum[val_idx] = d1 + d2 + d3

        # Target Encoding: Comorbidity Dyads
        pooled_dyads = np.concatenate([train['diag_dyad_12'][tr_idx], train['diag_dyad_13'][tr_idx], train['diag_dyad_23'][tr_idx]])
        dy_map = compute_smoothed_map(pooled_dyads, pooled_y, smooth=35.0, prior=prior_glob)
        dy12 = np.array([dy_map.get(safe_str(c), prior_glob) for c in train['diag_dyad_12'][val_idx]])
        dy13 = np.array([dy_map.get(safe_str(c), prior_glob) for c in train['diag_dyad_13'][val_idx]])
        dy23 = np.array([dy_map.get(safe_str(c), prior_glob) for c in train['diag_dyad_23'][val_idx]])
        tr_dyad_max[val_idx] = np.maximum(np.maximum(dy12, dy13), dy23)
        tr_dyad_mean[val_idx] = (dy12 + dy13 + dy23) / 3.0

        # Target Encoding: Contextual fields
        disp_map = compute_smoothed_map(train['discharge_disposition_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_disp[val_idx] = np.array([disp_map.get(safe_str(c), prior_glob) for c in train['discharge_disposition_id'][val_idx]])

        spec_map = compute_smoothed_map(train['medical_specialty'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_spec[val_idx] = np.array([spec_map.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

        src_map = compute_smoothed_map(train['admission_source_id'][tr_idx], y_train[tr_idx], smooth=20.0, prior=prior_glob)
        tr_src[val_idx] = np.array([src_map.get(safe_str(c), prior_glob) for c in train['admission_source_id'][val_idx]])

        # Biological Age Estimator
        ridge = Ridge(alpha=20.0)
        ridge.fit(bio_mat_train[tr_idx], train['age_num'][tr_idx])
        tr_bio_age[val_idx] = ridge.predict(bio_mat_train[val_idx])

        # Gravitational Prototype Attractors
        mask_pos = (y_train[tr_idx] == 1)
        c_pos = np.mean(tr_svd[tr_idx][mask_pos], axis=0)
        c_neg = np.mean(tr_svd[tr_idx][~mask_pos], axis=0)
        val_svd = tr_svd[val_idx]
        norm_val = np.linalg.norm(val_svd, axis=1, keepdims=True) + 1e-9
        norm_pos = np.linalg.norm(c_pos) + 1e-9
        norm_neg = np.linalg.norm(c_neg) + 1e-9
        sim_pos = np.squeeze((val_svd @ c_pos[:, None]) / (norm_val * norm_pos))
        sim_neg = np.squeeze((val_svd @ c_neg[:, None]) / (norm_val * norm_neg))
        tr_grav_pull[val_idx] = sim_pos - sim_neg
        tr_dist_pos[val_idx] = np.linalg.norm(val_svd - c_pos, axis=1)
        tr_dist_neg[val_idx] = np.linalg.norm(val_svd - c_neg, axis=1)

    train['te_diag_max'] = tr_diag_max
    train['te_diag_mean'] = tr_diag_mean
    train['te_diag_sum'] = tr_diag_sum
    train['te_dyad_max'] = tr_dyad_max
    train['te_dyad_mean'] = tr_dyad_mean
    train['te_discharge'] = tr_disp
    train['te_specialty'] = tr_spec
    train['te_adm_source'] = tr_src
    train['bio_age_accel'] = tr_bio_age - train['age_num']
    train['frailty_ratio'] = (tr_bio_age + 1.0) / (train['age_num'] + 1.0)
    train['grav_pull'] = tr_grav_pull
    train['dist_pos'] = tr_dist_pos
    train['dist_neg'] = tr_dist_neg

    # Full train-to-eval mappings
    pooled_diag_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    d_map_all = compute_smoothed_map(pooled_diag_all, pooled_y_all, smooth=25.0, prior=prior_glob)
    ed1 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_3d']])
    ed2 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_2_3d']])
    ed3 = np.array([d_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_3_3d']])
    eval_cols['te_diag_max'] = np.maximum(np.maximum(ed1, ed2), ed3)
    eval_cols['te_diag_mean'] = (ed1 + ed2 + ed3) / 3.0
    eval_cols['te_diag_sum'] = ed1 + ed2 + ed3

    pooled_dyads_all = np.concatenate([train['diag_dyad_12'], train['diag_dyad_13'], train['diag_dyad_23']])
    dy_map_all = compute_smoothed_map(pooled_dyads_all, pooled_y_all, smooth=35.0, prior=prior_glob)
    edy12 = np.array([dy_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_dyad_12']])
    edy13 = np.array([dy_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_dyad_13']])
    edy23 = np.array([dy_map_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_dyad_23']])
    eval_cols['te_dyad_max'] = np.maximum(np.maximum(edy12, edy13), edy23)
    eval_cols['te_dyad_mean'] = (edy12 + edy13 + edy23) / 3.0

    disp_map_all = compute_smoothed_map(train['discharge_disposition_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_discharge'] = np.array([disp_map_all.get(safe_str(c), prior_glob) for c in eval_cols['discharge_disposition_id']])

    spec_map_all = compute_smoothed_map(train['medical_specialty'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([spec_map_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']])

    src_map_all = compute_smoothed_map(train['admission_source_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_source'] = np.array([src_map_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_source_id']])

    ridge_full = Ridge(alpha=20.0)
    ridge_full.fit(bio_mat_train, train['age_num'])
    ev_bio_age = ridge_full.predict(bio_mat_eval)
    eval_cols['bio_age_accel'] = ev_bio_age - eval_cols['age_num']
    eval_cols['frailty_ratio'] = (ev_bio_age + 1.0) / (eval_cols['age_num'] + 1.0)

    mask_pos_all = (y_train == 1)
    c_pos_all = np.mean(tr_svd[mask_pos_all], axis=0)
    c_neg_all = np.mean(tr_svd[~mask_pos_all], axis=0)
    norm_ev = np.linalg.norm(ev_svd, axis=1, keepdims=True) + 1e-9
    norm_pos_all = np.linalg.norm(c_pos_all) + 1e-9
    norm_neg_all = np.linalg.norm(c_neg_all) + 1e-9
    ev_sim_pos = np.squeeze((ev_svd @ c_pos_all[:, None]) / (norm_ev * norm_pos_all))
    ev_sim_neg = np.squeeze((ev_svd @ c_neg_all[:, None]) / (norm_ev * norm_neg_all))
    eval_cols['grav_pull'] = ev_sim_pos - ev_sim_neg
    eval_cols['dist_pos'] = np.linalg.norm(ev_svd - c_pos_all, axis=1)
    eval_cols['dist_neg'] = np.linalg.norm(ev_svd - c_neg_all, axis=1)

    extra_num = [
        'age_num', 'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'acute_util_ratio', 'prior_util_severity',
        'has_prior_inpatient', 'frequent_inpatient', 'med_per_day', 'lab_per_day',
        'proc_per_day', 'diag_per_day', 'service_intensity',
        'lab_per_diag', 'proc_per_diag', 'iatrogenic_index',
        'active_meds', 'num_med_changes', 'has_insulin', 'insulin_up', 'insulin_down',
        'change_flag', 'diabetes_med_flag',
        'has_su', 'has_tzd', 'has_met',
        'tzd_chf_hazard', 'met_ckd_hazard', 'su_elderly_hazard', 'complex_multidrug',
        'med_ups', 'med_downs', 'med_steady', 'med_instability_ratio',
        'disch_is_home', 'disch_is_snf_hh', 'disch_is_ama', 'disch_is_transfer',
        'is_emergency_adm', 'is_elective_adm', 'adm_from_er', 'adm_from_transfer',
        'a1c_tested', 'a1c_high', 'a1c_very_high', 'glu_tested', 'glu_high',
        'has_chf', 'has_ckd', 'has_cad', 'has_htn', 'has_copd', 'has_diab_comp',
        'comorbidity_count', 'frailty_index',
        'bio_age_accel', 'frailty_ratio',
        'te_diag_max', 'te_diag_mean', 'te_diag_sum',
        'te_dyad_max', 'te_dyad_mean',
        'te_discharge', 'te_specialty', 'te_adm_source',
        'grav_pull', 'dist_pos', 'dist_neg',
    ] + topic_names

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'grp_dyad_12', 'grp_dyad_13', 'grp_dyad_23'
    ]

    # 5. Encoding
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 6. Quad-Engine GBDT Ensemble
    # Engine 1: LightGBM (Deep leaf-wise expansion)
    model_lgb1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.032,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb1 = model_lgb1.predict_proba(x_eval)[:, 1]

    # Engine 2: LightGBM (Depth-constrained conservative regularized)
    model_lgb2 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.038,
        max_depth=4,
        num_leaves=15,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb2 = model_lgb2.predict_proba(x_eval)[:, 1]

    # Engine 3: HistGradientBoosting (Scikit-learn integer-binned categorical tree)
    model_hgb = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.032,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=3.0,
        categorical_features=encoder.categorical_indices,
        random_state=42,
    )
    model_hgb.fit(x_train, y_train)
    p_hgb = model_hgb.predict_proba(x_eval)[:, 1]

    # Engine 4: XGBoost (Histogram depth-wise tree)
    model_xgb = xgb.XGBClassifier(
        n_estimators=240,
        learning_rate=0.032,
        max_depth=5,
        min_child_weight=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=1,
        tree_method='hist',
        eval_metric='logloss',
    )
    model_xgb.fit(x_train, y_train)
    p_xgb = model_xgb.predict_proba(x_eval)[:, 1]

    # 7. Wilcoxon Rank-Consensus Ensembling
    n_ev = len(p_lgb1)
    r_lgb1 = rankdata(np.nan_to_num(p_lgb1, nan=prior_glob)) / n_ev
    r_lgb2 = rankdata(np.nan_to_num(p_lgb2, nan=prior_glob)) / n_ev
    r_hgb  = rankdata(np.nan_to_num(p_hgb, nan=prior_glob)) / n_ev
    r_xgb  = rankdata(np.nan_to_num(p_xgb, nan=prior_glob)) / n_ev

    rank_blend = 0.35 * r_lgb1 + 0.20 * r_lgb2 + 0.25 * r_xgb + 0.20 * r_hgb
    return np.clip(rank_blend, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
