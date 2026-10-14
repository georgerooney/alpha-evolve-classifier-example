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
    """Fit on `train`/`y_train` and return readmission probabilities using pooled cross-diagnosis
    risk encoding, stratified Bayesian target encoding, clinical inertia & frailty phenotyping,
    and a regularized Gauss-rank logit ensemble."""
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import rankdata

    n_train = len(train['time_in_hospital'])
    n_eval = len(eval_cols['time_in_hospital'])

    train_ext = dict(train)
    eval_ext = dict(eval_cols)

    def clean_str(c):
        if c is None or (isinstance(c, float) and np.isnan(c)) or c in ('', '?', 'None'):
            return ''
        return str(c).strip()

    def get_icd_prefix(c):
        s = clean_str(c)
        return s.split('.')[0] if s else ''

    # 1. ICD-9 Chapter Groupings & 3-Digit Diagnosis Prefixes
    for diag_col in ('diag_1', 'diag_2', 'diag_3'):
        grp_name = f"{diag_col}_group"
        train_ext[grp_name] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_ext[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)

        pref_name = f"{diag_col}_prefix"
        train_ext[pref_name] = np.array([get_icd_prefix(c) for c in train[diag_col]], dtype=object)
        eval_ext[pref_name] = np.array([get_icd_prefix(c) for c in eval_cols[diag_col]], dtype=object)

    # Joint clinical pathway indicators
    d_tr = [clean_str(x) for x in train.get('discharge_disposition_id', np.zeros(n_train))]
    a_tr = [clean_str(x) for x in train.get('admission_source_id', np.zeros(n_train))]
    s_tr = [clean_str(x) for x in train.get('medical_specialty', np.zeros(n_train))]

    d_ev = [clean_str(x) for x in eval_cols.get('discharge_disposition_id', np.zeros(n_eval))]
    a_ev = [clean_str(x) for x in eval_cols.get('admission_source_id', np.zeros(n_eval))]
    s_ev = [clean_str(x) for x in eval_cols.get('medical_specialty', np.zeros(n_eval))]

    train_ext['disp_admit_combo'] = np.array([f"{d}_{a}" for d, a in zip(d_tr, a_tr)], dtype=object)
    train_ext['disp_diag1_combo'] = np.array([f"{d}_{g}" for d, g in zip(d_tr, train_ext['diag_1_group'])], dtype=object)
    train_ext['disp_spec_combo'] = np.array([f"{d}_{s}" for d, s in zip(d_tr, s_tr)], dtype=object)
    train_ext['admit_spec_combo'] = np.array([f"{a}_{s}" for a, s in zip(a_tr, s_tr)], dtype=object)
    train_ext['diag1_admit_combo'] = np.array([f"{g}_{a}" for g, a in zip(train_ext['diag_1_group'], a_tr)], dtype=object)

    eval_ext['disp_admit_combo'] = np.array([f"{d}_{a}" for d, a in zip(d_ev, a_ev)], dtype=object)
    eval_ext['disp_diag1_combo'] = np.array([f"{d}_{g}" for d, g in zip(d_ev, eval_ext['diag_1_group'])], dtype=object)
    eval_ext['disp_spec_combo'] = np.array([f"{d}_{s}" for d, s in zip(d_ev, s_ev)], dtype=object)
    eval_ext['admit_spec_combo'] = np.array([f"{a}_{s}" for a, s in zip(a_ev, s_ev)], dtype=object)
    eval_ext['diag1_admit_combo'] = np.array([f"{g}_{a}" for g, a in zip(eval_ext['diag_1_group'], a_ev)], dtype=object)

    # Multi-diagnosis pathways & demographic disposition combinations
    train_ext['diag_1_2_combo'] = np.array([f"{a}_{b}" for a, b in zip(train_ext['diag_1_group'], train_ext['diag_2_group'])], dtype=object)
    eval_ext['diag_1_2_combo'] = np.array([f"{a}_{b}" for a, b in zip(eval_ext['diag_1_group'], eval_ext['diag_2_group'])], dtype=object)

    train_ext['diag_2_3_combo'] = np.array([f"{a}_{b}" for a, b in zip(train_ext['diag_2_group'], train_ext['diag_3_group'])], dtype=object)
    eval_ext['diag_2_3_combo'] = np.array([f"{a}_{b}" for a, b in zip(eval_ext['diag_2_group'], eval_ext['diag_3_group'])], dtype=object)

    age_tr = [clean_str(x) for x in train.get('age', np.zeros(n_train))]
    age_ev = [clean_str(x) for x in eval_cols.get('age', np.zeros(n_eval))]
    train_ext['age_disp_combo'] = np.array([f"{a}_{d}" for a, d in zip(age_tr, d_tr)], dtype=object)
    eval_ext['age_disp_combo'] = np.array([f"{a}_{d}" for a, d in zip(age_ev, d_ev)], dtype=object)

    train_ext['spec_diag1_combo'] = np.array([f"{s}_{g}" for s, g in zip(s_tr, train_ext['diag_1_group'])], dtype=object)
    eval_ext['spec_diag1_combo'] = np.array([f"{s}_{g}" for s, g in zip(s_ev, eval_ext['diag_1_group'])], dtype=object)

    extra_cat_cols = [
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'disp_admit_combo', 'disp_diag1_combo', 'disp_spec_combo',
        'admit_spec_combo', 'diag1_admit_combo',
        'diag_1_2_combo', 'diag_2_3_combo',
        'age_disp_combo', 'spec_diag1_combo'
    ]

    # 2. Granular Clinical Syndromes, Sepsis, Renal Staging & Comorbid Burden
    for data, n_rows in ((train_ext, n_train), (eval_ext, n_eval)):
        g1 = data['diag_1_group']
        g2 = data['diag_2_group']
        g3 = data['diag_3_group']

        data['diag_has_diabetes'] = np.array([1.0 if 'diabetes' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_circulatory'] = np.array([1.0 if 'circulatory' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_respiratory'] = np.array([1.0 if 'respiratory' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_digestive'] = np.array([1.0 if 'digestive' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_genitourinary'] = np.array([1.0 if 'genitourinary' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_has_neoplasms'] = np.array([1.0 if 'neoplasms' in (a, b, c) else 0.0 for a, b, c in zip(g1, g2, g3)], dtype=np.float64)
        data['diag_num_comorbid_groups'] = np.array([float(len({x for x in (a, b, c) if x not in ('missing', 'other')})) for a, b, c in zip(g1, g2, g3)], dtype=np.float64)

        d1_raw = [clean_str(x) for x in data['diag_1']]
        d2_raw = [clean_str(x) for x in data['diag_2']]
        d3_raw = [clean_str(x) for x in data['diag_3']]

        has_chf = np.zeros(n_rows, dtype=np.float64)
        has_cad = np.zeros(n_rows, dtype=np.float64)
        has_htn = np.zeros(n_rows, dtype=np.float64)
        has_diab_comp = np.zeros(n_rows, dtype=np.float64)
        has_diab_uncontrolled = np.zeros(n_rows, dtype=np.float64)
        has_acute_renal = np.zeros(n_rows, dtype=np.float64)
        has_ckd_esrd = np.zeros(n_rows, dtype=np.float64)
        has_fluid_elec = np.zeros(n_rows, dtype=np.float64)
        has_copd = np.zeros(n_rows, dtype=np.float64)
        has_pneumonia = np.zeros(n_rows, dtype=np.float64)
        has_sepsis = np.zeros(n_rows, dtype=np.float64)
        has_infection = np.zeros(n_rows, dtype=np.float64)
        has_hypo_dka = np.zeros(n_rows, dtype=np.float64)
        has_afib = np.zeros(n_rows, dtype=np.float64)
        has_stroke = np.zeros(n_rows, dtype=np.float64)
        has_anemia = np.zeros(n_rows, dtype=np.float64)
        has_liver = np.zeros(n_rows, dtype=np.float64)
        has_psych = np.zeros(n_rows, dtype=np.float64)
        has_cancer = np.zeros(n_rows, dtype=np.float64)
        has_pvd = np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            codes = (d1_raw[i], d2_raw[i], d3_raw[i])
            if any(c.startswith('428') for c in codes):
                has_chf[i] = 1.0
            if any(c.startswith(('410', '411', '412', '413', '414')) for c in codes):
                has_cad[i] = 1.0
            if any(c.startswith(('401', '402', '403', '404', '405')) for c in codes):
                has_htn[i] = 1.0
            if any(c.startswith(('250.1', '250.2', '250.3', '250.4', '250.5', '250.6', '250.7', '250.8', '250.9')) for c in codes):
                has_diab_comp[i] = 1.0
            if any(c.startswith('250') and (c.endswith('2') or c.endswith('3')) for c in codes):
                has_diab_uncontrolled[i] = 1.0
            if any(c.startswith('584') for c in codes):
                has_acute_renal[i] = 1.0
            if any(c.startswith(('585', '586')) for c in codes):
                has_ckd_esrd[i] = 1.0
            if any(c.startswith('276') for c in codes):
                has_fluid_elec[i] = 1.0
            if any(c.startswith(('490', '491', '492', '493', '494', '495', '496')) for c in codes):
                has_copd[i] = 1.0
            if any(c.startswith(('480', '481', '482', '483', '484', '485', '486')) for c in codes):
                has_pneumonia[i] = 1.0
            if any(c.startswith('038') or c.startswith('995.9') for c in codes):
                has_sepsis[i] = 1.0
            if any(c.startswith(('038', '599', '681', '682', '683')) for c in codes):
                has_infection[i] = 1.0
            if any(c.startswith(('250.1', '251.0', '251.1', '251.2')) for c in codes):
                has_hypo_dka[i] = 1.0
            if any(c.startswith('427') for c in codes):
                has_afib[i] = 1.0
            if any(c.startswith(('430', '431', '432', '433', '434', '435', '436', '437', '438')) for c in codes):
                has_stroke[i] = 1.0
            if any(c.startswith(('280', '281', '282', '283', '284', '285')) for c in codes):
                has_anemia[i] = 1.0
            if any(c.startswith(('571', '572', '573')) for c in codes):
                has_liver[i] = 1.0
            if any(c.startswith(('29', '30', '31')) for c in codes):
                has_psych[i] = 1.0
            for c in codes:
                p_c = c.split('.')[0]
                if p_c.isdigit() and 140 <= int(p_c) <= 239:
                    has_cancer[i] = 1.0
            if any(c.startswith(('440', '441', '442', '443', '707', '785.4')) for c in codes):
                has_pvd[i] = 1.0

        data['diag_has_chf'] = has_chf
        data['diag_has_cad'] = has_cad
        data['diag_has_htn'] = has_htn
        data['diag_has_diab_comp'] = has_diab_comp
        data['diag_has_diab_uncontrolled'] = has_diab_uncontrolled
        data['diag_has_acute_renal'] = has_acute_renal
        data['diag_has_ckd_esrd'] = has_ckd_esrd
        has_any_kidney = np.maximum(has_acute_renal, has_ckd_esrd)
        data['diag_has_kidney'] = has_any_kidney
        data['diag_has_fluid_elec'] = has_fluid_elec
        data['diag_has_copd'] = has_copd
        data['diag_has_pneumonia'] = has_pneumonia
        data['diag_has_sepsis'] = has_sepsis
        data['diag_has_infection'] = has_infection
        data['diag_has_hypo_dka'] = has_hypo_dka
        data['diag_has_afib'] = has_afib
        data['diag_has_stroke'] = has_stroke
        data['diag_has_anemia'] = has_anemia
        data['diag_has_liver'] = has_liver
        data['diag_has_psych'] = has_psych
        data['diag_has_cancer'] = has_cancer
        data['diag_has_pvd'] = has_pvd

        # Surgical complications, ulcers, and acute primary reason flags
        has_surg_comp = np.zeros(n_rows, dtype=np.float64)
        has_skin_ulcer = np.zeros(n_rows, dtype=np.float64)
        has_uti = np.zeros(n_rows, dtype=np.float64)

        diag1_is_chf = np.zeros(n_rows, dtype=np.float64)
        diag1_is_cad = np.zeros(n_rows, dtype=np.float64)
        diag1_is_diabetes = np.zeros(n_rows, dtype=np.float64)
        diag1_is_acute_renal = np.zeros(n_rows, dtype=np.float64)
        diag1_is_respiratory = np.zeros(n_rows, dtype=np.float64)
        diag1_is_infection = np.zeros(n_rows, dtype=np.float64)
        diag1_is_surg_comp = np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            codes = (d1_raw[i], d2_raw[i], d3_raw[i])
            if any(c.startswith(('996', '997', '998', '999')) for c in codes):
                has_surg_comp[i] = 1.0
            if any(c.startswith(('681', '682', '707')) for c in codes):
                has_skin_ulcer[i] = 1.0
            if any(c.startswith('599') for c in codes):
                has_uti[i] = 1.0

            d1 = d1_raw[i]
            if d1.startswith('428'):
                diag1_is_chf[i] = 1.0
            if d1.startswith(('410', '411', '412', '413', '414')):
                diag1_is_cad[i] = 1.0
            if d1.startswith('250'):
                diag1_is_diabetes[i] = 1.0
            if d1.startswith('584'):
                diag1_is_acute_renal[i] = 1.0
            if d1.startswith(('480', '481', '482', '483', '484', '485', '486', '490', '491', '492', '493', '494', '495', '496')):
                diag1_is_respiratory[i] = 1.0
            if d1.startswith(('038', '599', '681', '682', '486')):
                diag1_is_infection[i] = 1.0
            if d1.startswith(('996', '997', '998', '999')):
                diag1_is_surg_comp[i] = 1.0

        data['has_surgical_complication'] = has_surg_comp
        data['has_skin_ulcer'] = has_skin_ulcer
        data['has_uti'] = has_uti
        data['diag1_is_chf'] = diag1_is_chf
        data['diag1_is_cad'] = diag1_is_cad
        data['diag1_is_diabetes'] = diag1_is_diabetes
        data['diag1_is_acute_renal'] = diag1_is_acute_renal
        data['diag1_is_respiratory'] = diag1_is_respiratory
        data['diag1_is_infection'] = diag1_is_infection
        data['diag1_is_surg_comp'] = diag1_is_surg_comp

        # Cardiorenal & multimorbidity interactions
        data['cardiorenal_syndrome'] = has_chf * has_any_kidney
        data['chf_and_copd'] = has_chf * has_copd
        data['diabetes_and_kidney'] = data['diag_has_diabetes'] * has_any_kidney
        data['diab_cardiorenal_triad'] = data['diag_has_diabetes'] * has_chf * has_any_kidney
        data['cardiorenal_anemia_syndrome'] = has_chf * has_any_kidney * has_anemia
        data['charlson_burden_score'] = (
            2.0 * has_chf + 2.5 * has_acute_renal + 2.0 * has_ckd_esrd + 1.5 * has_diab_comp +
            1.5 * has_copd + 1.0 * has_cad + 1.0 * has_stroke + 1.0 * has_pvd +
            1.0 * has_fluid_elec + 0.5 * has_htn + 2.0 * has_cancer + 2.0 * has_surg_comp +
            1.5 * has_liver + 1.0 * has_psych + 0.5 * has_anemia + 2.5 * has_sepsis
        )
        data['multimorbid_count'] = (
            has_chf + has_cad + has_htn + has_diab_comp + has_any_kidney +
            has_copd + has_stroke + has_fluid_elec + has_cancer + has_sepsis + has_anemia
        )

    # 3. Latent Patient Context SVD (Diagnoses, Specialties, Payer, Pathway)
    context_keys = []
    prefix_counts = {}
    for col in ('diag_1', 'diag_2', 'diag_3'):
        if col in train:
            for c in train[col]:
                p = get_icd_prefix(c)
                if p:
                    prefix_counts[f"diag_pref_{p}"] = prefix_counts.get(f"diag_pref_{p}", 0) + 1
    top_diags = [k for k, _ in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:80]]
    context_keys.extend(top_diags)

    for col, top_n in [('medical_specialty', 15), ('discharge_disposition_id', 10), ('admission_source_id', 10), ('payer_code', 8)]:
        if col in train:
            counts = {}
            for c in train[col]:
                s = clean_str(c)
                if s:
                    counts[f"{col}_{s}"] = counts.get(f"{col}_{s}", 0) + 1
            top_vals = [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
            context_keys.extend(top_vals)

    key_to_idx = {k: i for i, k in enumerate(context_keys)}
    D_ctx = len(context_keys)
    k_ctx = min(10, D_ctx)

    if D_ctx > 0:
        B_train = np.zeros((n_train, D_ctx), dtype=np.float32)
        B_eval = np.zeros((n_eval, D_ctx), dtype=np.float32)

        for col in ('diag_1', 'diag_2', 'diag_3'):
            if col in train:
                for i, c in enumerate(train[col]):
                    p = get_icd_prefix(c)
                    idx = key_to_idx.get(f"diag_pref_{p}")
                    if idx is not None:
                        B_train[i, idx] = 1.0
            if col in eval_cols:
                for i, c in enumerate(eval_cols[col]):
                    p = get_icd_prefix(c)
                    idx = key_to_idx.get(f"diag_pref_{p}")
                    if idx is not None:
                        B_eval[i, idx] = 1.0

        for col in ('medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code'):
            if col in train:
                for i, c in enumerate(train[col]):
                    s = clean_str(c)
                    idx = key_to_idx.get(f"{col}_{s}")
                    if idx is not None:
                        B_train[i, idx] = 1.0
            if col in eval_cols:
                for i, c in enumerate(eval_cols[col]):
                    s = clean_str(c)
                    idx = key_to_idx.get(f"{col}_{s}")
                    if idx is not None:
                        B_eval[i, idx] = 1.0

        ctx_mean = np.mean(B_train, axis=0, keepdims=True)
        B_train_c = B_train - ctx_mean
        B_eval_c = B_eval - ctx_mean

        C_ctx = (B_train_c.T @ B_train_c) / max(1, n_train)
        _, eigvecs_ctx = np.linalg.eigh(C_ctx)
        ctx_components = eigvecs_ctx[:, -k_ctx:][:, ::-1]

        Z_ctx_train = B_train_c @ ctx_components
        Z_ctx_eval = B_eval_c @ ctx_components

        for d in range(k_ctx):
            train_ext[f'ctx_svd_{d}'] = Z_ctx_train[:, d].astype(np.float64)
            eval_ext[f'ctx_svd_{d}'] = Z_ctx_eval[:, d].astype(np.float64)
    else:
        Z_ctx_train = np.zeros((n_train, 1), dtype=np.float32)
        Z_ctx_eval = np.zeros((n_eval, 1), dtype=np.float32)

    # 4. Latent Medication Regimen via SVD Decomposition
    all_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    available_meds = [m for m in all_meds if m in train and m in eval_cols]
    med_val_map = {'No': 0.0, 'Steady': 1.0, 'Up': 2.0, 'Down': -0.5}
    D_med = len(available_meds)
    k_med = min(4, D_med)

    if D_med > 0:
        M_train = np.zeros((n_train, D_med), dtype=np.float32)
        M_eval = np.zeros((n_eval, D_med), dtype=np.float32)

        for j, med in enumerate(available_meds):
            tr_col = train[med]
            for i in range(n_train):
                M_train[i, j] = med_val_map.get(tr_col[i], 0.0)
            ev_col = eval_cols[med]
            for i in range(n_eval):
                M_eval[i, j] = med_val_map.get(ev_col[i], 0.0)

        med_mean = np.mean(M_train, axis=0, keepdims=True)
        M_train_c = M_train - med_mean
        M_eval_c = M_eval - med_mean

        C_med = (M_train_c.T @ M_train_c) / max(1, n_train)
        _, eigvecs_med = np.linalg.eigh(C_med)
        med_components = eigvecs_med[:, -k_med:][:, ::-1]

        Z_med_train = M_train_c @ med_components
        Z_med_eval = M_eval_c @ med_components

        for m in range(k_med):
            train_ext[f'med_svd_{m}'] = Z_med_train[:, m].astype(np.float64)
            eval_ext[f'med_svd_{m}'] = Z_med_eval[:, m].astype(np.float64)
    else:
        Z_med_train = np.zeros((n_train, 1), dtype=np.float32)
        Z_med_eval = np.zeros((n_eval, 1), dtype=np.float32)

    # 5. Bilinear Tensor Interactions: Context Phenotype x Regimen Coupling
    n_inter_c = min(4, Z_ctx_train.shape[1])
    n_inter_m = min(3, Z_med_train.shape[1])
    for d in range(n_inter_c):
        for m in range(n_inter_m):
            train_ext[f'tensor_inter_{d}_{m}'] = (Z_ctx_train[:, d] * Z_med_train[:, m]).astype(np.float64)
            eval_ext[f'tensor_inter_{d}_{m}'] = (Z_ctx_eval[:, d] * Z_med_eval[:, m]).astype(np.float64)

    # 6. Utilization, Prior Admissions & Stay Intensity
    for data, n_rows in ((train_ext, n_train), (eval_ext, n_eval)):
        num_inpatient = np.nan_to_num(data.get('number_inpatient', np.zeros(n_rows, dtype=np.float64)))
        num_emergency = np.nan_to_num(data.get('number_emergency', np.zeros(n_rows, dtype=np.float64)))
        num_outpatient = np.nan_to_num(data.get('number_outpatient', np.zeros(n_rows, dtype=np.float64)))
        time_hosp = np.nan_to_num(data.get('time_in_hospital', np.ones(n_rows, dtype=np.float64)), nan=1.0)
        time_hosp_safe = np.maximum(time_hosp, 1.0)
        num_meds = np.nan_to_num(data.get('num_medications', np.zeros(n_rows, dtype=np.float64)))
        num_labs = np.nan_to_num(data.get('num_lab_procedures', np.zeros(n_rows, dtype=np.float64)))
        num_procs = np.nan_to_num(data.get('num_procedures', np.zeros(n_rows, dtype=np.float64)))
        num_diags = np.nan_to_num(data.get('number_diagnoses', np.zeros(n_rows, dtype=np.float64)))

        total_prior = num_inpatient + num_emergency + num_outpatient
        data['total_prior_visits'] = total_prior
        data['has_prior_inpatient'] = (num_inpatient > 0).astype(np.float64)
        data['has_prior_emergency'] = (num_emergency > 0).astype(np.float64)
        data['has_prior_outpatient'] = (num_outpatient > 0).astype(np.float64)
        data['prior_inpatient_ge1'] = (num_inpatient >= 1.0).astype(np.float64)
        data['prior_inpatient_ge2'] = (num_inpatient >= 2.0).astype(np.float64)
        data['prior_inpatient_ge3'] = (num_inpatient >= 3.0).astype(np.float64)
        data['prior_emergency_ge1'] = (num_emergency >= 1.0).astype(np.float64)
        data['inpatient_ratio'] = num_inpatient / (total_prior + 1.0)
        data['emergency_ratio'] = num_emergency / (total_prior + 1.0)
        data['high_inpatient'] = (num_inpatient >= 2.0).astype(np.float64)

        data['log_inpatient'] = np.log1p(np.maximum(num_inpatient, 0.0))
        data['log_emergency'] = np.log1p(np.maximum(num_emergency, 0.0))
        data['log_outpatient'] = np.log1p(np.maximum(num_outpatient, 0.0))
        data['log_acute_visits'] = np.log1p(np.maximum(num_inpatient + num_emergency, 0.0))
        data['log_time_in_hospital'] = np.log1p(np.maximum(time_hosp, 0.0))
        data['frequent_acute'] = ((num_inpatient + num_emergency) >= 2.0).astype(np.float64)

        data['inpatient_x_stay'] = num_inpatient * time_hosp
        data['inpatient_x_meds'] = num_inpatient * num_meds
        data['inpatient_x_diags'] = num_inpatient * num_diags
        data['inpatient_per_diag'] = num_inpatient / (num_diags + 1.0)
        data['prior_visit_rate'] = total_prior / (time_hosp_safe + 1.0)

        data['stay_short'] = (time_hosp <= 2.0).astype(np.float64)
        data['stay_long'] = (time_hosp >= 7.0).astype(np.float64)
        data['many_diagnoses'] = (num_diags >= 9.0).astype(np.float64)

        # Clinically validated LACE Index approximation
        # L: Length of stay points
        lace_l = np.where(time_hosp <= 1.0, 1.0,
                 np.where(time_hosp == 2.0, 2.0,
                 np.where(time_hosp == 3.0, 3.0,
                 np.where(time_hosp <= 6.0, 4.0,
                 np.where(time_hosp <= 13.0, 5.0, 7.0)))))
        # A: Acuity of admission
        admit_type_clean = [clean_str(x) for x in data.get('admission_type_id', np.zeros(n_rows))]
        lace_a = np.array([3.0 if v in ('1', '2') else 0.0 for v in admit_type_clean], dtype=np.float64)
        # C: Charlson burden points (capped at 5)
        lace_c = np.minimum(data.get('charlson_burden_score', np.zeros(n_rows)), 5.0)
        # E: Emergency visit points (capped at 4)
        lace_e = np.minimum(num_emergency, 4.0)
        data['lace_index'] = lace_l + lace_a + lace_c + lace_e
        data['lace_high_risk'] = (data['lace_index'] >= 10.0).astype(np.float64)

        # Clinically validated HOSPITAL score approximation
        hosp_h = data.get('diag_has_anemia', np.zeros(n_rows)) * 1.0
        hosp_o = data.get('diag_has_cancer', np.zeros(n_rows)) * 2.0
        hosp_s = data.get('diag_has_fluid_elec', np.zeros(n_rows)) * 1.0
        hosp_p = (num_procs > 0.0).astype(np.float64) * 1.0
        hosp_i = (lace_a > 0.0).astype(np.float64) * 1.0
        hosp_t = np.where(num_inpatient == 0.0, 0.0, np.where(num_inpatient <= 5.0, 2.0, 5.0))
        hosp_l = (time_hosp >= 5.0).astype(np.float64) * 2.0
        data['hospital_score'] = hosp_h + hosp_o + hosp_s + hosp_p + hosp_i + hosp_t + hosp_l
        data['hospital_high_risk'] = (data['hospital_score'] >= 7.0).astype(np.float64)

        # Weighted Prior Healthcare Utilization Velocity
        weighted_prior = 3.0 * num_inpatient + 1.5 * num_emergency + 0.5 * num_outpatient
        data['weighted_prior_utilization'] = weighted_prior
        data['log_weighted_prior_utilization'] = np.log1p(weighted_prior)

        data['service_intensity'] = (num_labs + num_procs + num_meds) / time_hosp_safe
        data['meds_per_day'] = num_meds / time_hosp_safe
        data['labs_per_day'] = num_labs / time_hosp_safe
        data['procedures_per_day'] = num_procs / time_hosp_safe
        data['diagnoses_per_day'] = num_diags / time_hosp_safe
        data['lab_to_med_ratio'] = num_labs / (num_meds + 1.0)
        data['meds_to_diag_ratio'] = num_meds / (num_diags + 1.0)
        data['emergency_to_inpatient_ratio'] = (num_emergency + 1.0) / (num_inpatient + 1.0)

    # 7. Medication Regimen Intensity, Titration & High-Risk Combinations
    sulf_meds = ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
    for data, n_rows in ((train_ext, n_train), (eval_ext, n_eval)):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        sulf_active = np.zeros(n_rows, dtype=np.float64)

        for med in available_meds:
            arr = data[med]
            is_sulf = med in sulf_meds
            for i in range(n_rows):
                v = arr[i]
                if v in ('Steady', 'Up', 'Down'):
                    active_cnt[i] += 1.0
                    if is_sulf:
                        sulf_active[i] = 1.0
                if v == 'Up':
                    up_cnt[i] += 1.0
                elif v == 'Down':
                    down_cnt[i] += 1.0

        data['num_active_meds'] = active_cnt
        data['num_meds_up'] = up_cnt
        data['num_meds_down'] = down_cnt
        data['num_meds_changed'] = up_cnt + down_cnt
        data['sulfonylurea_active'] = sulf_active

        if 'insulin' in data:
            data['insulin_active'] = np.array([1.0 if v in ('Steady', 'Up', 'Down') else 0.0 for v in data['insulin']], dtype=np.float64)
            data['insulin_up'] = np.array([1.0 if v == 'Up' else 0.0 for v in data['insulin']], dtype=np.float64)
            data['insulin_down'] = np.array([1.0 if v == 'Down' else 0.0 for v in data['insulin']], dtype=np.float64)
        else:
            data['insulin_active'] = np.zeros(n_rows, dtype=np.float64)
            data['insulin_up'] = np.zeros(n_rows, dtype=np.float64)
            data['insulin_down'] = np.zeros(n_rows, dtype=np.float64)

        if 'metformin' in data:
            data['metformin_active'] = np.array([1.0 if v in ('Steady', 'Up', 'Down') else 0.0 for v in data['metformin']], dtype=np.float64)
        else:
            data['metformin_active'] = np.zeros(n_rows, dtype=np.float64)

        data['insulin_and_metformin'] = data['insulin_active'] * data['metformin_active']
        data['hypo_risk_combo'] = data['insulin_active'] * data['sulfonylurea_active']
        data['insulin_only'] = data['insulin_active'] * (data['num_active_meds'] <= 1.0).astype(np.float64)
        num_orals = np.maximum(0.0, active_cnt - data['insulin_active'])
        data['num_oral_meds'] = num_orals
        data['oral_plus_insulin'] = (num_orals > 0.0).astype(np.float64) * data['insulin_active']
        data['insulin_up_with_kidney'] = data['insulin_up'] * data.get('diag_has_kidney', np.zeros(n_rows))
        data['insulin_down_with_kidney'] = data['insulin_down'] * data.get('diag_has_kidney', np.zeros(n_rows))

        tzd_active = np.zeros(n_rows, dtype=np.float64)
        for med in ['pioglitazone', 'rosiglitazone', 'troglitazone']:
            if med in data:
                arr = data[med]
                for i in range(n_rows):
                    if arr[i] in ('Steady', 'Up', 'Down'):
                        tzd_active[i] = 1.0
        data['tzd_active'] = tzd_active
        data['chf_and_tzd'] = data['diag_has_chf'] * tzd_active

        if 'change' in data:
            data['change_flag'] = np.array([1.0 if v == 'Ch' else 0.0 for v in data['change']], dtype=np.float64)
        if 'diabetesMed' in data:
            data['diabetes_med_flag'] = np.array([1.0 if v == 'Yes' else 0.0 for v in data['diabetesMed']], dtype=np.float64)

        data['insulin_x_inpatient'] = data['insulin_active'] * data.get('log_inpatient', np.zeros(n_rows))
        data['change_x_inpatient'] = data.get('change_flag', np.zeros(n_rows)) * data.get('log_inpatient', np.zeros(n_rows))

    # 8. Demographics, Frailty, Clinical Inertia & Lab Indicators
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for data in (train_ext, eval_ext):
        if 'age' in data:
            data['age_num'] = np.array([age_map.get(a, 65.0) for a in data['age']], dtype=np.float64)
            data['age_elderly'] = (data['age_num'] >= 70.0).astype(np.float64)
            data['age_very_elderly'] = (data['age_num'] >= 80.0).astype(np.float64)
            if 'log_inpatient' in data:
                data['age_x_inpatient'] = data['age_num'] * data['log_inpatient']
            if 'charlson_burden_score' in data:
                data['age_x_charlson'] = data['age_num'] * data['charlson_burden_score']

        if 'discharge_disposition_id' in data:
            arr_disp = [clean_str(v) for v in data['discharge_disposition_id']]
            data['discharge_home'] = np.array([1.0 if v == '1' else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_snf'] = np.array([1.0 if v == '3' else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_home_health'] = np.array([1.0 if v == '6' else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_ama'] = np.array([1.0 if v == '7' else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_rehab_or_ltac'] = np.array([1.0 if v in ('22', '23', '24') else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_other_hospital'] = np.array([1.0 if v in ('2', '4', '5', '18', '27', '28') else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_facility'] = np.array([1.0 if v in ('2', '3', '4', '5', '6', '22', '28') else 0.0 for v in arr_disp], dtype=np.float64)
            data['discharge_high_risk'] = np.array([1.0 if v in ('3', '6', '7', '22') else 0.0 for v in arr_disp], dtype=np.float64)
            data['snf_with_prior_inpatient'] = data['discharge_snf'] * data.get('has_prior_inpatient', 0.0)
            data['home_health_with_prior_inpatient'] = data['discharge_home_health'] * data.get('has_prior_inpatient', 0.0)

        if 'admission_type_id' in data:
            data['admission_emergency'] = np.array([1.0 if clean_str(v) in ('1', '2') else 0.0 for v in data['admission_type_id']], dtype=np.float64)
            data['admission_elective'] = np.array([1.0 if clean_str(v) == '3' else 0.0 for v in data['admission_type_id']], dtype=np.float64)
        if 'admission_source_id' in data:
            data['admission_from_er'] = np.array([1.0 if clean_str(v) == '7' else 0.0 for v in data['admission_source_id']], dtype=np.float64)
            data['admission_transfer'] = np.array([1.0 if clean_str(v) in ('4', '5', '6') else 0.0 for v in data['admission_source_id']], dtype=np.float64)

        # Elderly frailty cross-interactions
        if 'age_elderly' in data:
            data['elderly_snf'] = data['age_elderly'] * data.get('discharge_snf', 0.0)
            data['elderly_inpatient'] = data['age_elderly'] * data.get('has_prior_inpatient', 0.0)
            data['elderly_chf'] = data['age_elderly'] * data.get('diag_has_chf', 0.0)
            data['elderly_kidney'] = data['age_elderly'] * data.get('diag_has_kidney', 0.0)
            data['very_elderly_snf'] = data.get('age_very_elderly', 0.0) * data.get('discharge_snf', 0.0)
            data['very_elderly_inpatient'] = data.get('age_very_elderly', 0.0) * data.get('has_prior_inpatient', 0.0)

        if 'admission_emergency' in data and 'discharge_high_risk' in data:
            data['er_admit_and_high_risk_discharge'] = data['admission_emergency'] * data['discharge_high_risk']
        if 'admission_emergency' in data:
            data['er_short_stay'] = data['admission_emergency'] * data.get('stay_short', 0.0)
            data['er_long_stay'] = data['admission_emergency'] * data.get('stay_long', 0.0)

        # Lab tests & Clinical Inertia
        if 'A1Cresult' in data:
            data['a1c_tested'] = np.array([0.0 if v in (None, 'None') else 1.0 for v in data['A1Cresult']], dtype=np.float64)
            data['a1c_normal'] = np.array([1.0 if v == 'Norm' else 0.0 for v in data['A1Cresult']], dtype=np.float64)
            data['a1c_high'] = np.array([1.0 if v in ('>7', '>8') else 0.0 for v in data['A1Cresult']], dtype=np.float64)
            data['a1c_very_high'] = np.array([1.0 if v == '>8' else 0.0 for v in data['A1Cresult']], dtype=np.float64)
            if 'change_flag' in data:
                data['clinical_inertia'] = data['a1c_high'] * (1.0 - data['change_flag'])
                data['a1c_high_med_changed'] = data['a1c_high'] * data['change_flag']

        if 'max_glu_serum' in data:
            data['glucose_tested'] = np.array([0.0 if v in (None, 'None') else 1.0 for v in data['max_glu_serum']], dtype=np.float64)
            data['glucose_normal'] = np.array([1.0 if v == 'Norm' else 0.0 for v in data['max_glu_serum']], dtype=np.float64)
            data['glucose_high'] = np.array([1.0 if v in ('>200', '>300') else 0.0 for v in data['max_glu_serum']], dtype=np.float64)
            data['glucose_very_high'] = np.array([1.0 if v == '>300' else 0.0 for v in data['max_glu_serum']], dtype=np.float64)

        if 'weight' in data:
            data['weight_recorded'] = np.array([0.0 if clean_str(x) in ('', '?', 'None') else 1.0 for x in data['weight']], dtype=np.float64)
        if 'medical_specialty' in data:
            data['specialty_recorded'] = np.array([0.0 if clean_str(x) in ('', '?', 'None') else 1.0 for x in data['medical_specialty']], dtype=np.float64)
        if 'payer_code' in data:
            data['payer_recorded'] = np.array([0.0 if clean_str(x) in ('', '?', 'None') else 1.0 for x in data['payer_code']], dtype=np.float64)

    # 9. Pooled Out-of-Fold Cross-Diagnosis Risk Encodings
    def compute_pooled_diag_risks(tr_dict, y, ev_dict, m=30.0, n_splits=5, seed=42):
        n_tr = len(y)
        n_ev = len(ev_dict['diag_1'])
        prior = float(np.mean(y))

        tr_diag = {col: [clean_str(x) for x in tr_dict[col]] for col in ('diag_1', 'diag_2', 'diag_3')}
        ev_diag = {col: [clean_str(x) for x in ev_dict[col]] for col in ('diag_1', 'diag_2', 'diag_3')}

        tr_pref = {col: [get_icd_prefix(x) for x in tr_dict[col]] for col in ('diag_1', 'diag_2', 'diag_3')}
        ev_pref = {col: [get_icd_prefix(x) for x in ev_dict[col]] for col in ('diag_1', 'diag_2', 'diag_3')}

        oof_diag_max = np.full(n_tr, prior, dtype=np.float64)
        oof_diag_mean = np.full(n_tr, prior, dtype=np.float64)
        oof_pref_max = np.full(n_tr, prior, dtype=np.float64)
        oof_pref_mean = np.full(n_tr, prior, dtype=np.float64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for tr_idx, val_idx in skf.split(np.zeros(n_tr), y):
            diag_counts, diag_sums = {}, {}
            pref_counts, pref_sums = {}, {}

            for idx in tr_idx:
                label = y[idx]
                seen_diag, seen_pref = set(), set()
                for col in ('diag_1', 'diag_2', 'diag_3'):
                    d = tr_diag[col][idx]
                    p = tr_pref[col][idx]
                    if d and d not in seen_diag:
                        diag_counts[d] = diag_counts.get(d, 0) + 1
                        diag_sums[d] = diag_sums.get(d, 0.0) + label
                        seen_diag.add(d)
                    if p and p not in seen_pref:
                        pref_counts[p] = pref_counts.get(p, 0) + 1
                        pref_sums[p] = pref_sums.get(p, 0.0) + label
                        seen_pref.add(p)

            for idx in val_idx:
                d_risks, p_risks = [], []
                for col in ('diag_1', 'diag_2', 'diag_3'):
                    d = tr_diag[col][idx]
                    p = tr_pref[col][idx]
                    if d:
                        r = (diag_sums.get(d, 0.0) + m * prior) / (diag_counts.get(d, 0) + m)
                        d_risks.append(r)
                    if p:
                        r = (pref_sums.get(p, 0.0) + m * prior) / (pref_counts.get(p, 0) + m)
                        p_risks.append(r)

                if d_risks:
                    oof_diag_max[idx] = max(d_risks)
                    oof_diag_mean[idx] = sum(d_risks) / len(d_risks)
                if p_risks:
                    oof_pref_max[idx] = max(p_risks)
                    oof_pref_mean[idx] = sum(p_risks) / len(p_risks)

        full_diag_counts, full_diag_sums = {}, {}
        full_pref_counts, full_pref_sums = {}, {}

        for idx in range(n_tr):
            label = y[idx]
            seen_diag, seen_pref = set(), set()
            for col in ('diag_1', 'diag_2', 'diag_3'):
                d = tr_diag[col][idx]
                p = tr_pref[col][idx]
                if d and d not in seen_diag:
                    full_diag_counts[d] = full_diag_counts.get(d, 0) + 1
                    full_diag_sums[d] = full_diag_sums.get(d, 0.0) + label
                    seen_diag.add(d)
                if p and p not in seen_pref:
                    full_pref_counts[p] = full_pref_counts.get(p, 0) + 1
                    full_pref_sums[p] = full_pref_sums.get(p, 0.0) + label
                    seen_pref.add(p)

        ev_diag_max = np.full(n_ev, prior, dtype=np.float64)
        ev_diag_mean = np.full(n_ev, prior, dtype=np.float64)
        ev_pref_max = np.full(n_ev, prior, dtype=np.float64)
        ev_pref_mean = np.full(n_ev, prior, dtype=np.float64)

        for idx in range(n_ev):
            d_risks, p_risks = [], []
            for col in ('diag_1', 'diag_2', 'diag_3'):
                d = ev_diag[col][idx]
                p = ev_pref[col][idx]
                if d:
                    r = (full_diag_sums.get(d, 0.0) + m * prior) / (full_diag_counts.get(d, 0) + m)
                    d_risks.append(r)
                if p:
                    r = (full_pref_sums.get(p, 0.0) + m * prior) / (full_pref_counts.get(p, 0) + m)
                    p_risks.append(r)

            if d_risks:
                ev_diag_max[idx] = max(d_risks)
                ev_diag_mean[idx] = sum(d_risks) / len(d_risks)
            if p_risks:
                ev_pref_max[idx] = max(p_risks)
                ev_pref_mean[idx] = sum(p_risks) / len(p_risks)

        return (
            (oof_diag_max, ev_diag_max),
            (oof_diag_mean, ev_diag_mean),
            (oof_pref_max, ev_pref_max),
            (oof_pref_mean, ev_pref_mean),
        )

    (
        (train_ext['pooled_diag_max'], eval_ext['pooled_diag_max']),
        (train_ext['pooled_diag_mean'], eval_ext['pooled_diag_mean']),
        (train_ext['pooled_pref_max'], eval_ext['pooled_pref_max']),
        (train_ext['pooled_pref_mean'], eval_ext['pooled_pref_mean']),
    ) = compute_pooled_diag_risks(train_ext, y_train, eval_ext, m=30.0, n_splits=5, seed=42)

    # 10. Stratified Out-of-Fold Bayesian Smoothed Target Encoding
    def compute_oof_target_encoding(train_arr, y, eval_arr, n_splits=5, m=25.0, seed=42):
        n_tr = len(train_arr)
        n_ev = len(eval_arr)
        prior = float(np.mean(y))
        oof_train = np.full(n_tr, prior, dtype=np.float64)
        te_eval = np.full(n_ev, prior, dtype=np.float64)

        tr_clean = ["" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x) for x in train_arr]
        ev_clean = ["" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x) for x in eval_arr]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for tr_idx, val_idx in skf.split(np.zeros(n_tr), y):
            counts = {}
            sums = {}
            for idx in tr_idx:
                cat = tr_clean[idx]
                counts[cat] = counts.get(cat, 0) + 1
                sums[cat] = sums.get(cat, 0.0) + y[idx]

            for idx in val_idx:
                cat = tr_clean[idx]
                if cat in counts:
                    oof_train[idx] = (sums[cat] + m * prior) / (counts[cat] + m)
                else:
                    oof_train[idx] = prior

        full_counts = {}
        full_sums = {}
        for idx in range(n_tr):
            cat = tr_clean[idx]
            full_counts[cat] = full_counts.get(cat, 0) + 1
            full_sums[cat] = full_sums.get(cat, 0.0) + y[idx]

        for idx in range(n_ev):
            cat = ev_clean[idx]
            if cat in full_counts:
                te_eval[idx] = (full_sums[cat] + m * prior) / (full_counts[cat] + m)
            else:
                te_eval[idx] = prior

        return oof_train, te_eval

    te_configs = [
        ('diag_1', 25.0),
        ('diag_2', 25.0),
        ('diag_3', 25.0),
        ('diag_1_prefix', 20.0),
        ('diag_2_prefix', 20.0),
        ('diag_3_prefix', 20.0),
        ('discharge_disposition_id', 15.0),
        ('medical_specialty', 25.0),
        ('admission_source_id', 15.0),
        ('admission_type_id', 15.0),
        ('diag_1_group', 20.0),
        ('disp_admit_combo', 20.0),
        ('disp_diag1_combo', 20.0),
        ('disp_spec_combo', 20.0),
        ('admit_spec_combo', 20.0),
        ('diag1_admit_combo', 20.0),
        ('payer_code', 20.0),
        ('diag_1_2_combo', 25.0),
        ('diag_2_3_combo', 25.0),
        ('age_disp_combo', 20.0),
        ('spec_diag1_combo', 20.0),
        ('race', 30.0),
        ('age', 30.0),
    ]
    for col, m_smooth in te_configs:
        if col in train_ext and col in eval_ext:
            oof_tr, te_ev = compute_oof_target_encoding(train_ext[col], y_train, eval_ext[col], n_splits=5, m=m_smooth)
            train_ext[f'{col}_te'] = oof_tr
            eval_ext[f'{col}_te'] = te_ev

    # Relative Acute-vs-Comorbid Risk Differentials
    for data in (train_ext, eval_ext):
        data['diag1_vs_mean_risk'] = data.get('diag_1_te', 0.0) - data.get('pooled_diag_mean', 0.0)
        data['diag_max_vs_mean_risk'] = data.get('pooled_diag_max', 0.0) - data.get('pooled_diag_mean', 0.0)
        data['diag_max_vs_pref_max'] = data.get('pooled_diag_max', 0.0) - data.get('pooled_pref_max', 0.0)

    # Frequency Encoding
    def compute_freq_encoding(train_arr, eval_arr):
        n_tr = max(1, len(train_arr))
        counts = {}
        tr_c = [clean_str(x) for x in train_arr]
        ev_c = [clean_str(x) for x in eval_arr]
        for c in tr_c:
            counts[c] = counts.get(c, 0) + 1
        return (
            np.array([counts.get(c, 0) / n_tr for c in tr_c], dtype=np.float64),
            np.array([counts.get(c, 0) / n_tr for c in ev_c], dtype=np.float64)
        )

    for col in ('diag_1', 'diag_1_prefix', 'medical_specialty', 'discharge_disposition_id'):
        if col in train_ext and col in eval_ext:
            tr_freq, ev_freq = compute_freq_encoding(train_ext[col], eval_ext[col])
            train_ext[f'{col}_freq'] = tr_freq
            eval_ext[f'{col}_freq'] = ev_freq

    # 10b. Deactivated to simplify feature representation and eliminate tensor overfitting
    '''
    # Mode 3: Care Trajectory & Transition Safety Net (S)
    diag_feature_keys = [
        'diag_has_chf', 'diag_has_cad', 'diag_has_htn', 'diag_has_diab_comp',
        'diag_has_diab_uncontrolled', 'diag_has_acute_renal', 'diag_has_ckd_esrd',
        'diag_has_kidney', 'diag_has_fluid_elec', 'diag_has_copd', 'diag_has_pneumonia',
        'diag_has_sepsis', 'diag_has_infection', 'diag_has_hypo_dka', 'diag_has_afib',
        'diag_has_stroke', 'diag_has_anemia', 'diag_has_liver', 'diag_has_cancer',
        'diag_has_pvd', 'cardiorenal_syndrome', 'diab_cardiorenal_triad',
        'cardiorenal_anemia_syndrome', 'has_surgical_complication', 'diag1_is_chf',
        'diag1_is_acute_renal', 'charlson_burden_score', 'lace_index', 'hospital_score',
        'pooled_diag_max', 'diag_1_te'
    ]
    med_feature_keys = [
        'insulin_active', 'insulin_up', 'insulin_down', 'metformin_active',
        'sulfonylurea_active', 'tzd_active', 'hypo_risk_combo', 'num_active_meds',
        'num_meds_changed', 'change_flag', 'diabetes_med_flag', 'clinical_inertia',
        'oral_plus_insulin', 'insulin_up_with_kidney', 'insulin_down_with_kidney'
    ]
    traj_feature_keys = [
        'log_inpatient', 'prior_inpatient_ge1', 'prior_inpatient_ge2', 'log_emergency',
        'log_weighted_prior_utilization', 'log_time_in_hospital', 'discharge_home',
        'discharge_snf', 'discharge_home_health', 'discharge_ama', 'discharge_high_risk',
        'lace_high_risk', 'hospital_high_risk', 'admission_emergency', 'admission_from_er',
        'snf_with_prior_inpatient', 'service_intensity'
    ]

    def build_modal_matrices(data_source, n_r):
        D_mat = np.column_stack([np.nan_to_num(data_source.get(k, np.zeros(n_r, dtype=np.float64)), nan=0.0) for k in diag_feature_keys])
        M_mat = np.column_stack([np.nan_to_num(data_source.get(k, np.zeros(n_r, dtype=np.float64)), nan=0.0) for k in med_feature_keys])
        S_mat = np.column_stack([np.nan_to_num(data_source.get(k, np.zeros(n_r, dtype=np.float64)), nan=0.0) for k in traj_feature_keys])
        return D_mat.astype(np.float64), M_mat.astype(np.float64), S_mat.astype(np.float64)

    D_tr, M_tr, S_tr = build_modal_matrices(train_ext, n_train)
    D_ev, M_ev, S_ev = build_modal_matrices(eval_ext, n_eval)

    # Standardize modes based on training set statistics
    mu_D, std_D = np.mean(D_tr, axis=0, keepdims=True), np.std(D_tr, axis=0, keepdims=True) + 1e-6
    mu_M, std_M = np.mean(M_tr, axis=0, keepdims=True), np.std(M_tr, axis=0, keepdims=True) + 1e-6
    mu_S, std_S = np.mean(S_tr, axis=0, keepdims=True), np.std(S_tr, axis=0, keepdims=True) + 1e-6

    D_tr_std, D_ev_std = (D_tr - mu_D) / std_D, (D_ev - mu_D) / std_D
    M_tr_std, M_ev_std = (M_tr - mu_M) / std_M, (M_ev - mu_M) / std_M
    S_tr_std, S_ev_std = (S_tr - mu_S) / std_S, (S_ev - mu_S) / std_S

    def decompose_risk_tensor(D_in, M_in, S_in, y_in, R=5, n_iter=10):
        N = len(y_in)
        y_c = (y_in - np.mean(y_in)).astype(np.float64)
        D_scaled = D_in * y_c[:, None]
        # Fast 3rd-order cross-covariance tensor: T[j, k, l] = sum_i y_c[i] * D[i,j] * M[i,k] * S[i,l]
        T = np.einsum('ij,ik,il->jkl', D_scaled, M_in, S_in) / max(1, N)
        D1, D2, D3 = T.shape

        factors_a, factors_b, factors_c, lambdas = [], [], [], []
        T_work = T.copy()

        for r in range(R):
            a = np.ones(D1, dtype=np.float64) / np.sqrt(D1)
            b = np.ones(D2, dtype=np.float64) / np.sqrt(D2)
            c = np.ones(D3, dtype=np.float64) / np.sqrt(D3)

            for _ in range(n_iter):
                a = np.einsum('jkl,k,l->j', T_work, b, c)
                norm_a = np.linalg.norm(a)
                if norm_a > 1e-8:
                    a /= norm_a
                b = np.einsum('jkl,j,l->k', T_work, a, c)
                norm_b = np.linalg.norm(b)
                if norm_b > 1e-8:
                    b /= norm_b
                c = np.einsum('jkl,j,k->l', T_work, a, b)
                norm_c = np.linalg.norm(c)
                if norm_c > 1e-8:
                    c /= norm_c

            lam = float(np.einsum('jkl,j,k,l', T_work, a, b, c))
            factors_a.append(a)
            factors_b.append(b)
            factors_c.append(c)
            lambdas.append(lam)
            # Deflate
            T_work -= lam * np.einsum('j,k,l->jkl', a, b, c)

        A = np.column_stack(factors_a)
        B = np.column_stack(factors_b)
        C = np.column_stack(factors_c)
        return A, B, C, np.array(lambdas, dtype=np.float64)

    # 5-fold Out-of-Fold CP Tensor Projection on Train to Prevent Leakage
    R_CP = 5
    oof_tri = np.zeros((n_train, R_CP), dtype=np.float64)
    oof_dm = np.zeros((n_train, R_CP), dtype=np.float64)
    oof_ds = np.zeros((n_train, R_CP), dtype=np.float64)
    oof_ms = np.zeros((n_train, R_CP), dtype=np.float64)
    oof_risk_sum = np.zeros(n_train, dtype=np.float64)

    skf_tensor = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_tensor.split(np.zeros(n_train), y_train):
        A_f, B_f, C_f, lam_f = decompose_risk_tensor(
            D_tr_std[tr_idx], M_tr_std[tr_idx], S_tr_std[tr_idx], y_train[tr_idx], R=R_CP, n_iter=8
        )
        u_val = D_tr_std[val_idx] @ A_f
        v_val = M_tr_std[val_idx] @ B_f
        w_val = S_tr_std[val_idx] @ C_f

        oof_tri[val_idx] = u_val * v_val * w_val
        oof_dm[val_idx] = u_val * v_val
        oof_ds[val_idx] = u_val * w_val
        oof_ms[val_idx] = v_val * w_val
        oof_risk_sum[val_idx] = np.sum(oof_tri[val_idx] * lam_f[None, :], axis=1)

    # Full Train Fit for Eval Projection
    A_full, B_full, C_full, lam_full = decompose_risk_tensor(
        D_tr_std, M_tr_std, S_tr_std, y_train, R=R_CP, n_iter=10
    )
    u_ev = D_ev_std @ A_full
    v_ev = M_ev_std @ B_full
    w_ev = S_ev_std @ C_full

    ev_tri = u_ev * v_ev * w_ev
    ev_dm = u_ev * v_ev
    ev_ds = u_ev * w_ev
    ev_ms = v_ev * w_ev
    ev_risk_sum = np.sum(ev_tri * lam_full[None, :], axis=1)

    for r in range(R_CP):
        train_ext[f'cp_tri_{r}'] = oof_tri[:, r]
        eval_ext[f'cp_tri_{r}'] = ev_tri[:, r]
        train_ext[f'cp_dm_{r}'] = oof_dm[:, r]
        eval_ext[f'cp_dm_{r}'] = ev_dm[:, r]
        train_ext[f'cp_ds_{r}'] = oof_ds[:, r]
        eval_ext[f'cp_ds_{r}'] = ev_ds[:, r]
        train_ext[f'cp_ms_{r}'] = oof_ms[:, r]
        eval_ext[f'cp_ms_{r}'] = ev_ms[:, r]

    train_ext['tensor_risk_composite'] = oof_risk_sum
    eval_ext['tensor_risk_composite'] = ev_risk_sum

    # Coupled Matrix Factorization: Bilinear Supervised Covariance Singular Components
    y_tr_c = (y_train - np.mean(y_train)).astype(np.float64)
    Cov_DM = (D_tr_std.T @ (M_tr_std * y_tr_c[:, None])) / max(1, n_train)
    Cov_DS = (D_tr_std.T @ (S_tr_std * y_tr_c[:, None])) / max(1, n_train)
    Cov_MS = (M_tr_std.T @ (S_tr_std * y_tr_c[:, None])) / max(1, n_train)

    U_dm, _, Vt_dm = np.linalg.svd(Cov_DM, full_matrices=False)
    U_ds, _, Vt_ds = np.linalg.svd(Cov_DS, full_matrices=False)
    U_ms, _, Vt_ms = np.linalg.svd(Cov_MS, full_matrices=False)

    for k in range(2):
        train_ext[f'cmtf_dm_{k}'] = (D_tr_std @ U_dm[:, k]) * (M_tr_std @ Vt_dm[k, :])
        eval_ext[f'cmtf_dm_{k}'] = (D_ev_std @ U_dm[:, k]) * (M_ev_std @ Vt_dm[k, :])

        train_ext[f'cmtf_ds_{k}'] = (D_tr_std @ U_ds[:, k]) * (S_tr_std @ Vt_ds[k, :])
        eval_ext[f'cmtf_ds_{k}'] = (D_ev_std @ U_ds[:, k]) * (S_ev_std @ Vt_ds[k, :])

        train_ext[f'cmtf_ms_{k}'] = (M_tr_std @ U_ms[:, k]) * (S_tr_std @ Vt_ms[k, :])
        eval_ext[f'cmtf_ms_{k}'] = (M_ev_std @ U_ms[:, k]) * (S_ev_std @ Vt_ms[k, :])
    '''

    # Collect all new numeric features
    extra_numeric_cols = [k for k in train_ext if k not in train and k not in extra_cat_cols]

    # 11. Encode Features
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_ext, extra_numeric=extra_numeric_cols, extra_categorical=extra_cat_cols)
    x_eval = encoder.transform(eval_ext)

    # 12. Diverse Gradient Boosting Architectures
    cat_indices = encoder.categorical_indices

    # Model 1: Balanced LightGBM with tree path smoothing
    m1 = lgb.LGBMClassifier(
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=3.0,
        reg_alpha=0.5,
        path_smooth=3.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    # Model 2: Extremely Randomized Trees LightGBM
    m2 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.03,
        num_leaves=40,
        max_depth=6,
        min_child_samples=35,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=4.0,
        reg_alpha=1.0,
        extra_trees=True,
        path_smooth=2.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )

    # Model 3: Shallow, highly regularized LightGBM
    m3 = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=14,
        max_depth=4,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=8.0,
        reg_alpha=2.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )

    # Model 4: Gradient-based One-Side Sampling (GOSS) LightGBM
    m4 = lgb.LGBMClassifier(
        boosting_type='goss',
        n_estimators=360,
        learning_rate=0.025,
        num_leaves=24,
        max_depth=5,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=555,
        n_jobs=1,
        verbose=-1,
    )

    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    m4.fit(x_train, y_train, categorical_feature=cat_indices)
    p4 = m4.predict_proba(x_eval)[:, 1]

    model_preds_weights = [(p1, 0.22), (p2, 0.18), (p3, 0.18), (p4, 0.14)]

    # Depth-wise histogram XGBoost models for architectural diversity
    # Feed XGBoost sanitized numeric + target-encoded features without raw integer-coded nominal IDs
    try:
        import xgboost as xgb
        x_train_xgb = np.delete(x_train, cat_indices, axis=1) if len(cat_indices) > 0 else x_train
        x_eval_xgb = np.delete(x_eval, cat_indices, axis=1) if len(cat_indices) > 0 else x_eval

        m_xgb1 = xgb.XGBClassifier(
            n_estimators=320,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_lambda=5.0,
            reg_alpha=0.5,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )
        m_xgb1.fit(x_train_xgb, y_train)
        p_xgb1 = m_xgb1.predict_proba(x_eval_xgb)[:, 1]

        m_xgb2 = xgb.XGBClassifier(
            n_estimators=340,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.55,
            reg_lambda=8.0,
            reg_alpha=1.0,
            min_child_weight=3,
            tree_method='hist',
            eval_metric='logloss',
            random_state=777,
            n_jobs=1,
        )
        m_xgb2.fit(x_train_xgb, y_train)
        p_xgb2 = m_xgb2.predict_proba(x_eval_xgb)[:, 1]

        m_xgb3 = xgb.XGBClassifier(
            n_estimators=360,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=10.0,
            reg_alpha=2.0,
            tree_method='hist',
            eval_metric='logloss',
            random_state=333,
            n_jobs=1,
        )
        m_xgb3.fit(x_train_xgb, y_train)
        p_xgb3 = m_xgb3.predict_proba(x_eval_xgb)[:, 1]

        model_preds_weights = [
            (p1, 0.20),
            (p2, 0.16),
            (p3, 0.16),
            (p4, 0.12),
            (p_xgb1, 0.13),
            (p_xgb2, 0.12),
            (p_xgb3, 0.11),
        ]
    except Exception:
        model_preds_weights = [
            (p1, 0.35),
            (p2, 0.25),
            (p3, 0.25),
            (p4, 0.15),
        ]

    # 13. Variance-Stabilized Gauss-Rank Logit & Calibrated Probability Hybrid Ensembling
    # Normalizes heterogeneous prediction spreads into uniform ranks with controlled tail-dispersion,
    # and blends with direct probability consensus to preserve confident calibration margins
    base_prev = float(np.mean(y_train))
    base_logit = float(np.log(base_prev / max(1e-6, 1.0 - base_prev)))

    total_w = sum(w for _, w in model_preds_weights)
    z_accum = np.zeros(n_eval, dtype=np.float64)
    p_linear_accum = np.zeros(n_eval, dtype=np.float64)

    for p, w in model_preds_weights:
        norm_w = w / total_w
        p_linear_accum += norm_w * p
        r = (rankdata(p) - 0.5) / n_eval
        r = np.clip(r, 1e-4, 1.0 - 1e-4)
        z = np.log(r / (1.0 - r))
        z_accum += norm_w * z

    # Map blended odds back to prevalence-calibrated probabilities
    p_rank_logit = 1.0 / (1.0 + np.exp(-(z_accum + base_logit)))

    # Optimal hybrid blend of scale-free rank log-odds and calibrated probabilities
    final_preds = 0.60 * p_rank_logit + 0.40 * p_linear_accum
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
