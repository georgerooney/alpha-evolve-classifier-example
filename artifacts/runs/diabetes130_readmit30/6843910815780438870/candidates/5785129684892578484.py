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
    """Multi-Scale Clinical Recidivism Engine (MSCRE) with Pharmacological Flux and Heterogeneous GBDT Stacking."""
    train_data = dict(train)
    eval_data = dict(eval_cols)

    n_train = len(y_train)
    n_eval = len(eval_data[task.numeric_columns[0]]) if task.numeric_columns else len(list(eval_data.values())[0])

    extra_num_cols = []
    extra_cat_cols = []

    # 1. Resolving Frozen Toolkit Functions (with bulletproof local fallbacks)
    _comorb_fn = globals().get('icd9_comorbidities')
    _group_fn = globals().get('icd9_group')
    _oof_te_fn = globals().get('oof_target_encode')

    if _comorb_fn is None or _group_fn is None or _oof_te_fn is None:
        try:
            import clinical_toolkit
            if _comorb_fn is None:
                _comorb_fn = getattr(clinical_toolkit, 'icd9_comorbidities', None)
            if _group_fn is None:
                _group_fn = getattr(clinical_toolkit, 'icd9_group', None)
            if _oof_te_fn is None:
                _oof_te_fn = getattr(clinical_toolkit, 'oof_target_encode', None)
        except Exception:
            pass

    def _fallback_icd9_group(code):
        if code is None or code == "" or code == "?":
            return "missing"
        c_str = str(code).strip()
        if c_str.startswith("V") or c_str.startswith("E"):
            return "other"
        try:
            val = float(c_str.split(".")[0])
            if 390 <= val <= 459 or val == 785:
                return "circulatory"
            elif 460 <= val <= 519 or val == 786:
                return "respiratory"
            elif 520 <= val <= 579 or val == 787:
                return "digestive"
            elif int(val) == 250:
                return "diabetes"
            elif 800 <= val <= 999:
                return "injury"
            elif 710 <= val <= 739:
                return "musculoskeletal"
            elif 580 <= val <= 629 or val == 788:
                return "genitourinary"
            elif 140 <= val <= 239:
                return "neoplasms"
            else:
                return "other"
        except Exception:
            return "other"

    group_resolver = _group_fn if _group_fn is not None else _fallback_icd9_group

    def _fallback_oof_te(tr_arr, y_tr, ev_arr, n_folds=5, smoothing=20.0, seed=42):
        n = len(y_tr)
        prior = float(np.mean(y_tr))
        oof_tr = np.full(n, prior, dtype=np.float64)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        folds = np.array_split(perm, n_folds)

        tr_clean = [str(x) if x is not None else "__NONE__" for x in tr_arr]
        ev_clean = [str(x) if x is not None else "__NONE__" for x in ev_arr]

        for f_idx in range(n_folds):
            val_idx = folds[f_idx]
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False

            sub_counts = {}
            sub_sums = {}
            for idx in np.where(train_mask)[0]:
                k = tr_clean[idx]
                sub_counts[k] = sub_counts.get(k, 0) + 1
                sub_sums[k] = sub_sums.get(k, 0.0) + y_tr[idx]

            for idx in val_idx:
                k = tr_clean[idx]
                c = sub_counts.get(k, 0)
                s = sub_sums.get(k, 0.0)
                oof_tr[idx] = (s + smoothing * prior) / (c + smoothing)

        all_counts = {}
        all_sums = {}
        for k, y in zip(tr_clean, y_tr):
            all_counts[k] = all_counts.get(k, 0) + 1
            all_sums[k] = all_sums.get(k, 0.0) + y

        ev_enc = np.array([
            (all_sums.get(k, 0.0) + smoothing * prior) / (all_counts.get(k, 0) + smoothing)
            for k in ev_clean
        ], dtype=np.float64)
        return oof_tr, ev_enc

    te_resolver = _oof_te_fn if _oof_te_fn is not None else _fallback_oof_te

    # 2. Extract Charlson Comorbidities & 3-Digit ICD-9 Truncations
    cm_count_tr = np.zeros(n_train, dtype=np.float64)
    cm_count_ev = np.zeros(n_eval, dtype=np.float64)
    cm_renal_tr = np.zeros(n_train, dtype=np.float64)
    cm_renal_ev = np.zeros(n_eval, dtype=np.float64)

    if _comorb_fn is not None and all(c in train_data for c in ('diag_1', 'diag_2', 'diag_3')):
        try:
            cm_tr = _comorb_fn(train_data['diag_1'], train_data['diag_2'], train_data['diag_3'])
            cm_ev = _comorb_fn(eval_data['diag_1'], eval_data['diag_2'], eval_data['diag_3'])
            for cm_name, cm_vals in cm_tr.items():
                train_data[cm_name] = cm_vals
                eval_data[cm_name] = cm_ev[cm_name]
                extra_num_cols.append(cm_name)
            if 'cm_count' in cm_tr:
                cm_count_tr = cm_tr['cm_count']
                cm_count_ev = cm_ev['cm_count']
            if 'cm_renal' in cm_tr:
                cm_renal_tr = cm_tr['cm_renal']
                cm_renal_ev = cm_ev['cm_renal']
            if 'cm_chf' in cm_tr and 'cm_renal' in cm_tr:
                train_data['cm_cardiorenal'] = ((cm_tr['cm_chf'] > 0) & (cm_tr['cm_renal'] > 0)).astype(np.float64)
                eval_data['cm_cardiorenal'] = ((cm_ev['cm_chf'] > 0) & (cm_ev['cm_renal'] > 0)).astype(np.float64)
                extra_num_cols.append('cm_cardiorenal')
            charlson_weights = {
                'cm_mi': 1.0, 'cm_chf': 2.0, 'cm_pvd': 1.0, 'cm_cerebrovascular': 1.0,
                'cm_dementia': 1.0, 'cm_pulmonary': 1.0, 'cm_rheumatic': 1.0, 'cm_peptic_ulcer': 1.0,
                'cm_liver': 2.0, 'cm_hemiplegia': 2.0, 'cm_renal': 2.0, 'cm_cancer': 2.0,
                'cm_metastatic': 3.0, 'cm_diabetes_complicated': 2.0, 'cm_hiv': 3.0
            }
            c_score_tr = np.zeros(n_train, dtype=np.float64)
            c_score_ev = np.zeros(n_eval, dtype=np.float64)
            for cm_k, w in charlson_weights.items():
                if cm_k in cm_tr:
                    c_score_tr += w * cm_tr[cm_k]
                    c_score_ev += w * cm_ev[cm_k]
            train_data['charlson_weighted_score'] = c_score_tr
            eval_data['charlson_weighted_score'] = c_score_ev
            extra_num_cols.append('charlson_weighted_score')
        except Exception:
            pass

    def _diag_3digit(code):
        if code is None or code == "" or code == "?":
            return "missing"
        return str(code).strip().split(".")[0]

    for d_col in ('diag_1', 'diag_2', 'diag_3'):
        if d_col in train_data:
            g_col = f'{d_col}_group'
            train_data[g_col] = np.array([group_resolver(c) for c in train_data[d_col]], dtype=object)
            eval_data[g_col] = np.array([group_resolver(c) for c in eval_data[d_col]], dtype=object)
            extra_cat_cols.append(g_col)

            c_3d = f'{d_col}_3d'
            train_data[c_3d] = np.array([_diag_3digit(c) for c in train_data[d_col]], dtype=object)
            eval_data[c_3d] = np.array([_diag_3digit(c) for c in eval_data[d_col]], dtype=object)
            extra_cat_cols.append(c_3d)

    # 3. Pharmacological Flux, Regimen Discordance & Glycemic Inertia
    med_candidates = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    active_meds = [m for m in med_candidates if m in task.categorical_columns]

    def _med_features(cols_dict, n_rows):
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)
        steady = np.zeros(n_rows, dtype=np.float64)
        for m in active_meds:
            arr = cols_dict[m]
            up += (arr == 'Up').astype(np.float64)
            down += (arr == 'Down').astype(np.float64)
            steady += (arr == 'Steady').astype(np.float64)
        churn = up + down
        active = up + down + steady

        ins_code = np.zeros(n_rows, dtype=np.float64)
        if 'insulin' in cols_dict:
            ins = cols_dict['insulin']
            ins_code = np.where(ins == 'Up', 3.0,
                       np.where(ins == 'Down', 2.0,
                       np.where(ins == 'Steady', 1.0, 0.0)))

        # Sulfonylureas (hypoglycemia risk when combined with insulin)
        sulfas = ['chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'tolazamide']
        sulfa_active = np.zeros(n_rows, dtype=np.float64)
        for s in sulfas:
            if s in cols_dict:
                sulfa_active += (cols_dict[s] != 'No').astype(np.float64)

        ins_arr = cols_dict.get('insulin', np.array(['No'] * n_rows))
        ins_active = (ins_arr != 'No').astype(np.float64)
        ins_sulfa_synergy = ins_active * (sulfa_active > 0).astype(np.float64)

        # Glycemic volatility and therapeutic inertia
        a1c_arr = cols_dict.get('A1Cresult', np.array(['None'] * n_rows))
        a1c_high = ((a1c_arr == '>8') | (a1c_arr == '>7')).astype(np.float64)
        ch_arr = cols_dict.get('change', np.array(['No'] * n_rows))
        therapeutic_inertia = a1c_high * (ch_arr == 'No').astype(np.float64)

        glu_arr = cols_dict.get('max_glu_serum', np.array(['None'] * n_rows))
        glu_high = ((glu_arr == '>300') | (glu_arr == '>200')).astype(np.float64)

        return {
            'med_up_count': up,
            'med_down_count': down,
            'med_steady_count': steady,
            'med_churn_count': churn,
            'med_active_count': active,
            'insulin_flux_code': ins_code,
            'sulfa_active_count': sulfa_active,
            'insulin_sulfa_synergy': ins_sulfa_synergy,
            'therapeutic_inertia': therapeutic_inertia,
            'glucose_hyperglycemia_flag': glu_high,
        }

    tr_med = _med_features(train_data, n_train)
    ev_med = _med_features(eval_data, n_eval)
    for k in tr_med:
        train_data[k] = tr_med[k]
        eval_data[k] = ev_med[k]
        extra_num_cols.append(k)

    # 4. Bedside LACE Index Synthesis & Utilization Inertia
    def _util_features(cols_dict, cm_count, n_rows):
        t_hosp = np.nan_to_num(cols_dict['time_in_hospital'], nan=1.0)
        n_inpat = np.nan_to_num(cols_dict['number_inpatient'], nan=0.0)
        n_emerg = np.nan_to_num(cols_dict['number_emergency'], nan=0.0)
        n_outpat = np.nan_to_num(cols_dict['number_outpatient'], nan=0.0)
        n_labs = np.nan_to_num(cols_dict['num_lab_procedures'], nan=0.0)
        n_procs = np.nan_to_num(cols_dict['num_procedures'], nan=0.0)
        n_meds = np.nan_to_num(cols_dict['num_medications'], nan=0.0)
        n_diag = np.nan_to_num(cols_dict['number_diagnoses'], nan=0.0)

        # Bedside LACE Index Components (van Walraven et al.)
        l_score = np.where(t_hosp >= 14, 7.0,
                  np.where(t_hosp >= 7, 5.0,
                  np.where(t_hosp >= 4, 4.0,
                  np.where(t_hosp == 3, 3.0,
                  np.where(t_hosp == 2, 2.0,
                  np.where(t_hosp == 1, 1.0, 0.0))))))

        adm_type = cols_dict.get('admission_type_id', np.array(['0'] * n_rows))
        adm_src = cols_dict.get('admission_source_id', np.array(['0'] * n_rows))
        a_score = np.where((adm_type == '1') | (adm_src == '7'), 3.0, 0.0)
        c_score = np.where(cm_count >= 4, 5.0, cm_count)
        e_score = np.where(n_emerg >= 4, 4.0, n_emerg)
        lace_score = l_score + a_score + c_score + e_score
        lace_high_risk = (lace_score >= 10.0).astype(np.float64)

        total_prior = n_inpat + n_emerg + n_outpat
        inpat_ratio = n_inpat / (total_prior + 1.0)
        emerg_ratio = n_emerg / (total_prior + 1.0)
        acute_score = n_inpat * 3.0 + n_emerg * 1.5 + n_outpat * 0.5
        high_inpat = (n_inpat >= 1.0).astype(np.float64)
        super_inpat = (n_inpat >= 2.0).astype(np.float64)

        safe_t = np.maximum(t_hosp, 1.0)
        labs_per_day = n_labs / safe_t
        meds_per_day = n_meds / safe_t
        procs_per_day = n_procs / safe_t
        diag_per_day = n_diag / safe_t
        treatment_load = np.log1p(n_meds) * np.log1p(n_labs)
        inpat_diag_ratio = n_inpat / (n_diag + 1.0)
        cm_inpat_interaction = cm_count * (n_inpat + 0.5)
        lace_inpat_synergy = lace_score * (n_inpat + 1.0)

        return {
            'lace_score': lace_score,
            'lace_high_risk_flag': lace_high_risk,
            'lace_inpat_synergy': lace_inpat_synergy,
            'total_prior_visits': total_prior,
            'inpatient_prior_ratio': inpat_ratio,
            'emergency_prior_ratio': emerg_ratio,
            'acute_util_score': acute_score,
            'high_inpatient_flag': high_inpat,
            'super_inpatient_flag': super_inpat,
            'labs_per_day': labs_per_day,
            'meds_per_day': meds_per_day,
            'procs_per_day': procs_per_day,
            'diag_per_day': diag_per_day,
            'treatment_load': treatment_load,
            'inpat_diag_ratio': inpat_diag_ratio,
            'cm_inpat_interaction': cm_inpat_interaction,
        }

    tr_util = _util_features(train_data, cm_count_tr, n_train)
    ev_util = _util_features(eval_data, cm_count_ev, n_eval)
    for k in tr_util:
        train_data[k] = tr_util[k]
        eval_data[k] = ev_util[k]
        extra_num_cols.append(k)

    # 5. Age Band Conversion & Interactions
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    if 'age' in train_data:
        tr_age_num = np.array([age_map.get(str(a), 65.0) for a in train_data['age']], dtype=np.float64)
        ev_age_num = np.array([age_map.get(str(a), 65.0) for a in eval_data['age']], dtype=np.float64)
        train_data['age_numeric'] = tr_age_num
        eval_data['age_numeric'] = ev_age_num
        extra_num_cols.append('age_numeric')

        tr_age_inpat = tr_age_num * np.nan_to_num(train_data['number_inpatient'], nan=0.0)
        ev_age_inpat = ev_age_num * np.nan_to_num(eval_data['number_inpatient'], nan=0.0)
        train_data['age_inpatient_interaction'] = tr_age_inpat
        eval_data['age_inpatient_interaction'] = ev_age_inpat
        extra_num_cols.append('age_inpatient_interaction')

    # 6. Out-of-Fold Bayesian Target Encodings
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'medical_specialty'
    ]
    for c in te_cols:
        if c in train_data:
            tr_enc, ev_enc = te_resolver(train_data[c], y_train, eval_data[c], n_folds=5, smoothing=25.0, seed=42)
            col_name = f'{c}_target_enc'
            train_data[col_name] = tr_enc
            eval_data[col_name] = ev_enc
            extra_num_cols.append(col_name)

    # Joint clinical interactions: Diagnosis x Discharge disposition
    if 'diag_1_group' in train_data and 'discharge_disposition_id' in train_data:
        tr_joint = np.array([f"{g}_{d}" for g, d in zip(train_data['diag_1_group'], train_data['discharge_disposition_id'])], dtype=object)
        ev_joint = np.array([f"{g}_{d}" for g, d in zip(eval_data['diag_1_group'], eval_data['discharge_disposition_id'])], dtype=object)
        tr_j_enc, ev_j_enc = te_resolver(tr_joint, y_train, ev_joint, n_folds=5, smoothing=30.0, seed=101)
        train_data['diag_group_disposition_enc'] = tr_j_enc
        eval_data['diag_group_disposition_enc'] = ev_j_enc
        extra_num_cols.append('diag_group_disposition_enc')

    if 'diag_1_3d' in train_data and 'discharge_disposition_id' in train_data:
        tr_j3 = np.array([f"{g}_{d}" for g, d in zip(train_data['diag_1_3d'], train_data['discharge_disposition_id'])], dtype=object)
        ev_j3 = np.array([f"{g}_{d}" for g, d in zip(eval_data['diag_1_3d'], eval_data['discharge_disposition_id'])], dtype=object)
        tr_j3_enc, ev_j3_enc = te_resolver(tr_j3, y_train, ev_j3, n_folds=5, smoothing=35.0, seed=202)
        train_data['diag_3d_disposition_enc'] = tr_j3_enc
        eval_data['diag_3d_disposition_enc'] = ev_j3_enc
        extra_num_cols.append('diag_3d_disposition_enc')

    # 6b. Latent Clinical Phenotyping via TruncatedSVD
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import TruncatedSVD

        def _build_clinical_tokens(data_dict, n_rows):
            d1 = data_dict.get('diag_1_group', ['other'] * n_rows)
            d2 = data_dict.get('diag_2_group', ['other'] * n_rows)
            d3 = data_dict.get('diag_3_group', ['other'] * n_rows)
            d1_3 = data_dict.get('diag_1_3d', ['missing'] * n_rows)
            disp = data_dict.get('discharge_disposition_id', ['0'] * n_rows)
            src = data_dict.get('admission_source_id', ['0'] * n_rows)
            a1c = data_dict.get('A1Cresult', ['None'] * n_rows)
            glu = data_dict.get('max_glu_serum', ['None'] * n_rows)
            ins = data_dict.get('insulin', ['No'] * n_rows)
            ch = data_dict.get('change', ['No'] * n_rows)

            toks = []
            for i in range(n_rows):
                toks.append(f"g1_{d1[i]} g2_{d2[i]} g3_{d3[i]} c1_{d1_3[i]} disp_{disp[i]} src_{src[i]} a1c_{a1c[i]} glu_{glu[i]} ins_{ins[i]} ch_{ch[i]}")
            return toks

        tr_toks = _build_clinical_tokens(train_data, n_train)
        ev_toks = _build_clinical_tokens(eval_data, n_eval)

        cv = CountVectorizer(min_df=5, token_pattern=r'\S+', binary=True)
        X_bow_tr = cv.fit_transform(tr_toks)
        X_bow_ev = cv.transform(ev_toks)

        svd = TruncatedSVD(n_components=6, random_state=42)
        svd_tr = svd.fit_transform(X_bow_tr)
        svd_ev = svd.transform(X_bow_ev)

        for c_i in range(6):
            c_col = f'svd_clinical_dim_{c_i+1}'
            train_data[c_col] = svd_tr[:, c_i].astype(np.float64)
            eval_data[c_col] = svd_ev[:, c_i].astype(np.float64)
            extra_num_cols.append(c_col)
    except Exception:
        pass

    # 6c. Categorical Frequency Signals & High-Risk Dispositions
    disp_tr = train_data.get('discharge_disposition_id', np.array(['0'] * n_train))
    disp_ev = eval_data.get('discharge_disposition_id', np.array(['0'] * n_eval))
    train_data['disp_is_home'] = (disp_tr == '1').astype(np.float64)
    eval_data['disp_is_home'] = (disp_ev == '1').astype(np.float64)
    extra_num_cols.append('disp_is_home')
    train_data['disp_is_snf'] = (disp_tr == '3').astype(np.float64)
    eval_data['disp_is_snf'] = (disp_ev == '3').astype(np.float64)
    extra_num_cols.append('disp_is_snf')
    train_data['disp_is_home_health'] = (disp_tr == '6').astype(np.float64)
    eval_data['disp_is_home_health'] = (disp_ev == '6').astype(np.float64)
    extra_num_cols.append('disp_is_home_health')

    for fc in ('diag_1', 'discharge_disposition_id', 'medical_specialty'):
        if fc in train_data:
            c_dict = {}
            for item in train_data[fc]:
                k = str(item) if item is not None else '__NONE__'
                c_dict[k] = c_dict.get(k, 0) + 1
            f_col = f'{fc}_freq'
            train_data[f_col] = np.array([c_dict.get(str(x) if x is not None else '__NONE__', 0) for x in train_data[fc]], dtype=np.float64)
            eval_data[f_col] = np.array([c_dict.get(str(x) if x is not None else '__NONE__', 0) for x in eval_data[fc]], dtype=np.float64)
            extra_num_cols.append(f_col)

    # 7. Tabular Encoding
    _enc_cls = globals().get('TabularEncoder')
    if _enc_cls is None:
        try:
            import clinical_toolkit
            _enc_cls = getattr(clinical_toolkit, 'TabularEncoder', None)
        except Exception:
            pass

    encoder = _enc_cls(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_num_cols, extra_categorical=extra_cat_cols)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_idx = encoder.categorical_indices

    # 8. Heterogeneous GBDT Architecture Ensemble
    lgb_mod = globals().get('lgb') or globals().get('lightgbm')
    if lgb_mod is None:
        try:
            import lightgbm as lgb_mod
        except Exception:
            pass

    # Model 1: Deep Feature LightGBM
    m1 = lgb_mod.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.024,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.10,
        reg_lambda=1.2,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_idx)
    p_m1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Shallow Conservative LightGBM
    m2 = lgb_mod.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.020,
        num_leaves=18,
        min_child_samples=250,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.30,
        reg_lambda=3.0,
        cat_smooth=80.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_idx)
    p_m2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Wide Asymmetric LightGBM
    m3 = lgb_mod.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.028,
        num_leaves=45,
        min_child_samples=80,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.05,
        reg_lambda=1.5,
        cat_smooth=35.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_idx)
    p_m3 = m3.predict_proba(x_eval)[:, 1]

    p_lgb = 0.38 * p_m1 + 0.34 * p_m2 + 0.28 * p_m3

    # XGBoost Ensembling
    p_xgb = None
    xgb_mod = globals().get('xgb') or globals().get('xgboost')
    if xgb_mod is None:
        try:
            import xgboost as xgb_mod
        except Exception:
            pass

    if xgb_mod is not None:
        try:
            x1 = xgb_mod.XGBClassifier(
                n_estimators=250,
                learning_rate=0.030,
                max_depth=5,
                min_child_weight=25.0,
                subsample=0.75,
                colsample_bytree=0.55,
                reg_alpha=0.15,
                reg_lambda=2.0,
                tree_method='hist',
                random_state=101,
                n_jobs=1,
                eval_metric='logloss',
            )
            x1.fit(x_train, y_train)
            p_x1 = x1.predict_proba(x_eval)[:, 1]

            x2 = xgb_mod.XGBClassifier(
                n_estimators=230,
                learning_rate=0.028,
                max_depth=6,
                min_child_weight=35.0,
                subsample=0.70,
                colsample_bytree=0.48,
                reg_alpha=0.25,
                reg_lambda=3.0,
                tree_method='hist',
                random_state=999,
                n_jobs=1,
                eval_metric='logloss',
            )
            x2.fit(x_train, y_train)
            p_x2 = x2.predict_proba(x_eval)[:, 1]

            p_xgb = 0.55 * p_x1 + 0.45 * p_x2
        except Exception:
            p_xgb = None

    # 9. Synergistic Ensemble Blending
    if p_xgb is not None:
        p_eval = 0.68 * p_lgb + 0.32 * p_xgb
    else:
        p_eval = p_lgb

    p_eval = np.clip(p_eval, 0.0, 1.0)
    p_eval = np.nan_to_num(p_eval, nan=float(np.mean(y_train)))
    return p_eval


# EVOLVE-BLOCK-END
# =====================================================================
