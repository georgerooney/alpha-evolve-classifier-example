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

MED_NAMES = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)


def compute_spectral_components(
    train_mat: np.ndarray, eval_mat: np.ndarray, n_components: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """Low-rank spectral decomposition of joint clinical multi-mode interaction matrix."""
    mean = np.mean(train_mat, axis=0, keepdims=True)
    std = np.std(train_mat, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    tr_norm = (train_mat - mean) / std
    ev_norm = (eval_mat - mean) / std

    cov = (tr_norm.T @ tr_norm) / float(train_mat.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    top_v = eigvecs[:, idx[:n_components]]

    return tr_norm @ top_v, ev_norm @ top_v


def extract_med_features(cols_dict: Dict[str, np.ndarray]):
    """Extract clean medication dosage intensity, titration direction, and regimen structure."""
    n = len(cols_dict["time_in_hospital"])
    mat = np.zeros((n, len(MED_NAMES)), dtype=np.float64)
    for j, med in enumerate(MED_NAMES):
        col = cols_dict.get(med)
        if col is not None:
            mat[col == "Steady", j] = 1.0
            mat[col == "Up", j] = 2.0
            mat[col == "Down", j] = 1.5

    active_count = np.sum(mat > 0, axis=1, keepdims=True)
    changed_count = np.sum(mat > 1.0, axis=1, keepdims=True)
    num_up = np.sum(mat == 2.0, axis=1, keepdims=True)
    num_down = np.sum(mat == 1.5, axis=1, keepdims=True)
    num_steady = np.sum(mat == 1.0, axis=1, keepdims=True)

    insulin_idx = MED_NAMES.index("insulin")
    insulin_dose = mat[:, insulin_idx : insulin_idx + 1]
    on_insulin = (insulin_dose > 0).astype(np.float64)
    insulin_up = (insulin_dose == 2.0).astype(np.float64)
    insulin_down = (insulin_dose == 1.5).astype(np.float64)
    insulin_changed = ((insulin_up > 0) | (insulin_down > 0)).astype(np.float64)

    metformin_idx = MED_NAMES.index("metformin")
    on_metformin = (mat[:, metformin_idx : metformin_idx + 1] > 0).astype(np.float64)

    sulfo_names = (
        "glimepiride",
        "glipizide",
        "glyburide",
        "chlorpropamide",
        "tolbutamide",
        "acetohexamide",
        "tolazamide",
    )
    sulfo_indices = [MED_NAMES.index(m) for m in sulfo_names if m in MED_NAMES]
    sulfo_mat = mat[:, sulfo_indices]
    on_sulfo = (np.sum(sulfo_mat > 0, axis=1, keepdims=True) > 0).astype(np.float64)

    oral_count = active_count - on_insulin
    insulin_oral_combo = ((on_insulin > 0) & (oral_count > 0)).astype(np.float64)
    polypharmacy = (active_count >= 3).astype(np.float64)
    regimen_change_ratio = changed_count / (active_count + 1.0)
    dose_reduced = (num_down > 0).astype(np.float64)
    dose_escalated = (num_up > 0).astype(np.float64)

    summary_feats = np.hstack(
        [
            active_count,
            changed_count,
            num_up,
            num_down,
            num_steady,
            insulin_dose,
            on_insulin,
            insulin_up,
            insulin_down,
            insulin_changed,
            on_metformin,
            on_sulfo,
            oral_count,
            insulin_oral_combo,
            polypharmacy,
            regimen_change_ratio,
            dose_reduced,
            dose_escalated,
        ]
    )
    return mat, summary_feats


def extract_util_features(cols_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Clinical utilisation, admission/discharge pathway flags, and lab testing dynamics."""
    time_hosp = np.asarray(cols_dict["time_in_hospital"], dtype=np.float64)
    n_lab = np.asarray(cols_dict["num_lab_procedures"], dtype=np.float64)
    n_proc = np.asarray(cols_dict["num_procedures"], dtype=np.float64)
    n_med = np.asarray(cols_dict["num_medications"], dtype=np.float64)
    n_out = np.asarray(cols_dict["number_outpatient"], dtype=np.float64)
    n_emg = np.asarray(cols_dict["number_emergency"], dtype=np.float64)
    n_inp = np.asarray(cols_dict["number_inpatient"], dtype=np.float64)
    n_diag = np.asarray(cols_dict["number_diagnoses"], dtype=np.float64)

    total_vis = n_out + n_emg + n_inp
    acute_vis = n_emg + n_inp
    hosp_safe = np.maximum(time_hosp, 1.0)

    has_inp = (n_inp > 0).astype(np.float64)
    has_emg = (n_emg > 0).astype(np.float64)
    has_out = (n_out > 0).astype(np.float64)
    has_any = (total_vis > 0).astype(np.float64)
    has_multiple_inp = (n_inp >= 2).astype(np.float64)
    has_multiple_emg = (n_emg >= 2).astype(np.float64)
    has_prior_inp_and_emg = ((n_inp > 0) & (n_emg > 0)).astype(np.float64)
    outpatient_only = ((n_out > 0) & (n_inp == 0) & (n_emg == 0)).astype(np.float64)
    super_utilizer = ((n_inp >= 3) | (total_vis >= 5)).astype(np.float64)

    inp_ratio = n_inp / (total_vis + 1.0)
    emg_ratio = n_emg / (total_vis + 1.0)
    out_ratio = n_out / (total_vis + 1.0)
    acute_ratio = acute_vis / (total_vis + 1.0)
    inp_to_acute_ratio = n_inp / (acute_vis + 1.0)

    log_inp = np.log1p(n_inp)
    log_emg = np.log1p(n_emg)
    log_out = np.log1p(n_out)
    log_total = np.log1p(total_vis)
    weighted_util = 3.0 * n_inp + 2.0 * n_emg + 1.0 * n_out

    inp_per_day = n_inp / hosp_safe
    med_per_day = n_med / hosp_safe
    lab_per_day = n_lab / hosp_safe
    proc_per_day = n_proc / hosp_safe
    diag_per_day = n_diag / hosp_safe

    intensity = (n_lab + 5.0 * n_proc + 2.0 * n_med) / hosp_safe
    inp_x_diag = n_inp * n_diag
    inp_x_emg = n_inp * n_emg
    inp_x_time = n_inp * time_hosp
    med_x_time = n_med * time_hosp
    proc_per_med = n_proc / (n_med + 1.0)
    inp_to_time_ratio = n_inp / (time_hosp + n_inp + 1.0)

    med_per_diag = n_med / (n_diag + 1.0)
    lab_per_diag = n_lab / (n_diag + 1.0)
    lab_to_med = n_lab / (n_med + 1.0)
    total_events = n_lab + n_proc + n_med
    events_per_day = total_events / hosp_safe

    long_stay = (time_hosp >= 7).astype(np.float64)
    short_stay = (time_hosp <= 2).astype(np.float64)

    age_col = cols_dict["age"]
    age_map = {
        "[0-10)": 5.0,
        "[10-20)": 15.0,
        "[20-30)": 25.0,
        "[30-40)": 35.0,
        "[40-50)": 45.0,
        "[50-60)": 55.0,
        "[60-70)": 65.0,
        "[70-80)": 75.0,
        "[80-90)": 85.0,
        "[90-100)": 95.0,
    }
    age_num = np.array(
        [age_map.get(a, 65.0) if a is not None else 65.0 for a in age_col],
        dtype=np.float64,
    )
    age_x_inp = age_num * n_inp
    elderly = (age_num >= 75.0).astype(np.float64)
    elderly_x_inp = elderly * n_inp

    # Admission & discharge risk pathways
    disch = cols_dict["discharge_disposition_id"]
    adm_src = cols_dict["admission_source_id"]
    adm_typ = cols_dict["admission_type_id"]

    disch_is_home = np.array([1.0 if x == "1" else 0.0 for x in disch], dtype=np.float64)
    disch_is_snf = np.array([1.0 if x == "3" else 0.0 for x in disch], dtype=np.float64)
    disch_is_home_health = np.array([1.0 if x == "6" else 0.0 for x in disch], dtype=np.float64)
    disch_is_transfer = np.array([1.0 if x in ("2", "4", "5", "22") else 0.0 for x in disch], dtype=np.float64)
    disch_is_ama = np.array([1.0 if x == "7" else 0.0 for x in disch], dtype=np.float64)
    disch_not_home = 1.0 - disch_is_home
    disch_high_risk = np.array([1.0 if x in ("3", "6", "7", "22") else 0.0 for x in disch], dtype=np.float64)

    adm_is_emergency_src = np.array([1.0 if x == "7" else 0.0 for x in adm_src], dtype=np.float64)
    adm_is_emergency_typ = np.array([1.0 if x == "1" else 0.0 for x in adm_typ], dtype=np.float64)
    adm_is_elective = np.array([1.0 if x == "3" else 0.0 for x in adm_typ], dtype=np.float64)
    adm_is_transfer = np.array([1.0 if x in ("4", "5", "6") else 0.0 for x in adm_src], dtype=np.float64)
    adm_is_snf_transfer = np.array([1.0 if x == "5" else 0.0 for x in adm_src], dtype=np.float64)

    disch_is_rehab_ltc = np.array([1.0 if x in ("22", "23", "24") else 0.0 for x in disch], dtype=np.float64)
    disch_post_acute = np.array([1.0 if x in ("3", "4", "5", "6", "22", "23", "24") else 0.0 for x in disch], dtype=np.float64)
    adm_is_physician = np.array([1.0 if x == "1" else 0.0 for x in adm_src], dtype=np.float64)

    # Payer and clinical specialty risk markers
    payer = cols_dict.get("payer_code")
    payer_is_mc = np.array([1.0 if p == "MC" else 0.0 for p in payer], dtype=np.float64) if payer is not None else np.zeros(len(disch))
    payer_is_md = np.array([1.0 if p == "MD" else 0.0 for p in payer], dtype=np.float64) if payer is not None else np.zeros(len(disch))
    payer_is_missing = np.array([1.0 if p is None or p == "?" else 0.0 for p in payer], dtype=np.float64) if payer is not None else np.zeros(len(disch))

    spec = cols_dict.get("medical_specialty")
    spec_is_cardio = np.array([1.0 if s is not None and "Cardiology" in s else 0.0 for s in spec], dtype=np.float64) if spec is not None else np.zeros(len(disch))
    spec_is_internal = np.array([1.0 if s is not None and "InternalMedicine" in s else 0.0 for s in spec], dtype=np.float64) if spec is not None else np.zeros(len(disch))
    spec_is_emergency = np.array([1.0 if s is not None and "Emergency" in s else 0.0 for s in spec], dtype=np.float64) if spec is not None else np.zeros(len(disch))

    # Glycemic & lab results
    a1c = cols_dict["A1Cresult"]
    a1c_tested = np.array(
        [0.0 if x == "None" or x is None else 1.0 for x in a1c],
        dtype=np.float64,
    )
    a1c_high = np.array(
        [1.0 if x in (">7", ">8") else 0.0 for x in a1c], dtype=np.float64
    )
    a1c_normal = np.array(
        [1.0 if x == "Norm" else 0.0 for x in a1c], dtype=np.float64
    )

    glu = cols_dict["max_glu_serum"]
    glu_tested = np.array(
        [0.0 if x == "None" or x is None else 1.0 for x in glu],
        dtype=np.float64,
    )
    glu_high = np.array(
        [1.0 if x in (">200", ">300") else 0.0 for x in glu], dtype=np.float64
    )
    glu_normal = np.array(
        [1.0 if x == "Norm" else 0.0 for x in glu], dtype=np.float64
    )

    chg = cols_dict["change"]
    change_flag = np.array(
        [1.0 if x == "Ch" else 0.0 for x in chg], dtype=np.float64
    )
    dm_med = cols_dict["diabetesMed"]
    dm_flag = np.array(
        [1.0 if x == "Yes" else 0.0 for x in dm_med], dtype=np.float64
    )

    a1c_high_med_changed = a1c_high * change_flag
    a1c_high_no_change = a1c_high * (1.0 - change_flag)
    a1c_tested_no_change = a1c_tested * (1.0 - change_flag)
    no_a1c_and_no_glu = (1.0 - a1c_tested) * (1.0 - glu_tested)

    return np.column_stack(
        [
            total_vis,
            acute_vis,
            has_inp,
            has_emg,
            has_out,
            has_any,
            has_multiple_inp,
            has_multiple_emg,
            has_prior_inp_and_emg,
            outpatient_only,
            super_utilizer,
            inp_ratio,
            emg_ratio,
            out_ratio,
            acute_ratio,
            inp_to_acute_ratio,
            log_inp,
            log_emg,
            log_out,
            log_total,
            weighted_util,
            inp_per_day,
            med_per_day,
            lab_per_day,
            proc_per_day,
            diag_per_day,
            intensity,
            inp_x_diag,
            inp_x_emg,
            inp_x_time,
            med_x_time,
            proc_per_med,
            inp_to_time_ratio,
            med_per_diag,
            lab_per_diag,
            lab_to_med,
            events_per_day,
            long_stay,
            short_stay,
            age_num,
            age_x_inp,
            elderly,
            elderly_x_inp,
            disch_is_home,
            disch_is_snf,
            disch_is_home_health,
            disch_is_transfer,
            disch_is_ama,
            disch_not_home,
            disch_high_risk,
            adm_is_emergency_src,
            adm_is_emergency_typ,
            adm_is_elective,
            adm_is_transfer,
            adm_is_snf_transfer,
            disch_is_rehab_ltc,
            disch_post_acute,
            adm_is_physician,
            payer_is_mc,
            payer_is_md,
            payer_is_missing,
            spec_is_cardio,
            spec_is_internal,
            spec_is_emergency,
            a1c_tested,
            a1c_high,
            a1c_normal,
            glu_tested,
            glu_high,
            glu_normal,
            change_flag,
            dm_flag,
            a1c_high_med_changed,
            a1c_high_no_change,
            a1c_tested_no_change,
            no_a1c_and_no_glu,
        ]
    )


def extract_cm_derived(cm_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Charlson comorbidity index and core clinical vulnerability indicators."""
    mi = cm_dict["cm_mi"]
    chf = cm_dict["cm_chf"]
    pvd = cm_dict["cm_pvd"]
    cevd = cm_dict["cm_cerebrovascular"]
    dem = cm_dict["cm_dementia"]
    pulm = cm_dict["cm_pulmonary"]
    rheum = cm_dict["cm_rheumatic"]
    pep = cm_dict["cm_peptic_ulcer"]
    liv = cm_dict["cm_liver"]
    hemi = cm_dict["cm_hemiplegia"]
    ren = cm_dict["cm_renal"]
    canc = cm_dict["cm_cancer"]
    dm_comp = cm_dict["cm_diabetes_complicated"]
    met = cm_dict["cm_metastatic"]
    hiv = cm_dict["cm_hiv"]
    cm_cnt = cm_dict["cm_count"]

    charlson_idx = (
        mi + chf + pvd + cevd + dem + pulm + rheum + pep + liv
        + 2.0 * hemi + 2.0 * ren + 2.0 * canc + 2.0 * dm_comp
        + 4.0 * met + 4.0 * hiv
    )

    high_charlson = (charlson_idx >= 3.0).astype(np.float64)
    multi_cm = (cm_cnt >= 2.0).astype(np.float64)
    severe_systemic = np.clip(met + hiv + ren + liv, 0.0, 1.0)
    cardiorenal = chf * ren

    return np.column_stack([charlson_idx, high_charlson, multi_cm, severe_systemic, cardiorenal])


def extract_diag_group_features(
    d1_grp: np.ndarray, d2_grp: np.ndarray, d3_grp: np.ndarray
) -> np.ndarray:
    """Organ-system diagnosis chapter counts and primary etiology indicators."""
    circ_cnt = (
        (d1_grp == "circulatory").astype(np.float64)
        + (d2_grp == "circulatory").astype(np.float64)
        + (d3_grp == "circulatory").astype(np.float64)
    )
    resp_cnt = (
        (d1_grp == "respiratory").astype(np.float64)
        + (d2_grp == "respiratory").astype(np.float64)
        + (d3_grp == "respiratory").astype(np.float64)
    )
    dm_cnt = (
        (d1_grp == "diabetes").astype(np.float64)
        + (d2_grp == "diabetes").astype(np.float64)
        + (d3_grp == "diabetes").astype(np.float64)
    )
    digest_cnt = (
        (d1_grp == "digestive").astype(np.float64)
        + (d2_grp == "digestive").astype(np.float64)
        + (d3_grp == "digestive").astype(np.float64)
    )
    genit_cnt = (
        (d1_grp == "genitourinary").astype(np.float64)
        + (d2_grp == "genitourinary").astype(np.float64)
        + (d3_grp == "genitourinary").astype(np.float64)
    )
    neoplasm_cnt = (
        (d1_grp == "neoplasms").astype(np.float64)
        + (d2_grp == "neoplasms").astype(np.float64)
        + (d3_grp == "neoplasms").astype(np.float64)
    )
    injury_cnt = (
        (d1_grp == "injury").astype(np.float64)
        + (d2_grp == "injury").astype(np.float64)
        + (d3_grp == "injury").astype(np.float64)
    )

    is_prim_circ = (d1_grp == "circulatory").astype(np.float64)
    is_prim_resp = (d1_grp == "respiratory").astype(np.float64)
    is_prim_dm = (d1_grp == "diabetes").astype(np.float64)

    return np.column_stack(
        [
            circ_cnt,
            resp_cnt,
            dm_cnt,
            digest_cnt,
            genit_cnt,
            neoplasm_cnt,
            injury_cnt,
            is_prim_circ,
            is_prim_resp,
            is_prim_dm,
        ]
    )


def rank_average_probabilities(
    probs_list: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """Copula Quantile-Matched Rank Aggregation.
    
    Computes consensus ranking across heterogeneous model families via normalized rank
    percentiles, then maps the empirical quantile distribution of the ensemble probabilities
    onto the consensus ranks. Yields optimal bipartite ranking while preserving calibration.
    """
    from scipy.stats import rankdata

    n = len(probs_list[0])
    w = np.array(weights, dtype=np.float64)
    w = w / np.sum(w)

    rank_matrix = np.zeros((n, len(probs_list)), dtype=np.float64)
    for j, p in enumerate(probs_list):
        rank_matrix[:, j] = (rankdata(p, method="average") - 0.5) / float(n)

    r_consensus = rank_matrix @ w
    prob_matrix = np.column_stack(probs_list)
    p_ensemble = prob_matrix @ w

    sort_order = np.argsort(r_consensus)
    sorted_probs = np.sort(p_ensemble)

    p_final = np.empty_like(sorted_probs)
    p_final[sort_order] = sorted_probs
    return np.clip(p_final, 0.0, 1.0)


def compute_freq_arrays(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Prevalence/frequency encoding for categorical and diagnosis attributes."""
    tr_cols, ev_cols = [], []
    for tr_arr, ev_arr in pairs:
        n_tr = float(len(tr_arr))
        counts: Dict[Any, int] = {}
        for val in tr_arr:
            counts[val] = counts.get(val, 0) + 1
        freq_map = {k: v / n_tr for k, v in counts.items()}
        tr_cols.append(np.array([freq_map.get(x, 0.0) for x in tr_arr], dtype=np.float64))
        ev_cols.append(np.array([freq_map.get(x, 0.0) for x in ev_arr], dtype=np.float64))
    return np.column_stack(tr_cols), np.column_stack(ev_cols)


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    encoder = TabularEncoder(min_count=25)
    x_tr_enc = encoder.fit_transform(task, train)
    x_ev_enc = encoder.transform(eval_cols)
    cat_indices = list(encoder.categorical_indices)

    # 1. Low-rank medication regime factors
    med_tr, med_sum_tr = extract_med_features(train)
    med_ev, med_sum_ev = extract_med_features(eval_cols)
    med_pca_tr, med_pca_ev = compute_pca_features(med_tr, med_ev, n_components=4)

    # 2. Low-rank diagnosis co-occurrence factors
    diag_pca_tr, diag_pca_ev = extract_diag_pca(train, eval_cols, n_components=6)

    # 3. Charlson comorbidities and derived severity indices
    cm_tr_dict = icd9_comorbidities(
        train["diag_1"], train["diag_2"], train["diag_3"]
    )
    cm_ev_dict = icd9_comorbidities(
        eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"]
    )
    cm_keys = sorted(cm_tr_dict.keys())
    cm_tr_mat = np.column_stack([cm_tr_dict[k] for k in cm_keys])
    cm_ev_mat = np.column_stack([cm_ev_dict[k] for k in cm_keys])

    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    age_tr = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in train["age"]], dtype=np.float64)
    age_ev = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in eval_cols["age"]], dtype=np.float64)
    inp_tr = np.asarray(train["number_inpatient"], dtype=np.float64)
    inp_ev = np.asarray(eval_cols["number_inpatient"], dtype=np.float64)

    cm_der_tr = extract_cm_derived(cm_tr_dict, inp_tr, age_tr)
    cm_der_ev = extract_cm_derived(cm_ev_dict, inp_ev, age_ev)

    # 4. Diagnosis organ-system chapter aggregations
    d1_grp_tr = np.array([icd9_group(c) for c in train["diag_1"]], dtype=object)
    d1_grp_ev = np.array([icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
    d2_grp_tr = np.array([icd9_group(c) for c in train["diag_2"]], dtype=object)
    d2_grp_ev = np.array([icd9_group(c) for c in eval_cols["diag_2"]], dtype=object)
    d3_grp_tr = np.array([icd9_group(c) for c in train["diag_3"]], dtype=object)
    d3_grp_ev = np.array([icd9_group(c) for c in eval_cols["diag_3"]], dtype=object)

    diag_grp_tr = extract_diag_group_features(d1_grp_tr, d2_grp_tr, d3_grp_tr)
    diag_grp_ev = extract_diag_group_features(d1_grp_ev, d2_grp_ev, d3_grp_ev)

    # 5. Specific high-risk primary and comorbidity conditions
    spec_diag_tr = extract_diag_patterns(train["diag_1"], train["diag_2"], train["diag_3"])
    spec_diag_ev = extract_diag_patterns(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

    # 6. Diagnosis prefix generation
    d1_pfx_tr = np.array([c[:3] if c is not None and isinstance(c, str) else "None" for c in train["diag_1"]], dtype=object)
    d1_pfx_ev = np.array([c[:3] if c is not None and isinstance(c, str) else "None" for c in eval_cols["diag_1"]], dtype=object)
    d2_pfx_tr = np.array([c[:3] if c is not None and isinstance(c, str) else "None" for c in train["diag_2"]], dtype=object)
    d2_pfx_ev = np.array([c[:3] if c is not None and isinstance(c, str) else "None" for c in eval_cols["diag_2"]], dtype=object)
    d3_pfx_tr = np.array([c[:3] if c is not None and isinstance(c, str) else "None" for c in train["diag_3"]], dtype=object)
    d3_pfx_ev = np.array([c[:3] if c is not None and isinstance(c, str) else "None" for c in eval_cols["diag_3"]], dtype=object)

    # 7. Non-leaking prevalence/frequency encodings
    freq_pairs = [
        (train["diag_1"], eval_cols["diag_1"]),
        (train["diag_2"], eval_cols["diag_2"]),
        (train["diag_3"], eval_cols["diag_3"]),
        (d1_pfx_tr, d1_pfx_ev),
        (d2_pfx_tr, d2_pfx_ev),
        (d3_pfx_tr, d3_pfx_ev),
        (train["medical_specialty"], eval_cols["medical_specialty"]),
        (train["payer_code"], eval_cols["payer_code"]),
        (train["discharge_disposition_id"], eval_cols["discharge_disposition_id"]),
        (train["admission_source_id"], eval_cols["admission_source_id"]),
        (train["admission_type_id"], eval_cols["admission_type_id"]),
    ]
    freq_tr, freq_ev = compute_freq_arrays(freq_pairs)

    # 8. Out-of-fold target encodings for high-cardinality & key clinical interactions
    d1_disch_tr = np.array([f"{g}_{d}" for g, d in zip(d1_grp_tr, train["discharge_disposition_id"])], dtype=object)
    d1_disch_ev = np.array([f"{g}_{d}" for g, d in zip(d1_grp_ev, eval_cols["discharge_disposition_id"])], dtype=object)
    adm_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["admission_source_id"], train["discharge_disposition_id"])], dtype=object)
    adm_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["admission_source_id"], eval_cols["discharge_disposition_id"])], dtype=object)

    inp_bkt_tr = np.clip(inp_tr, 0, 3).astype(int).astype(str)
    inp_bkt_ev = np.clip(inp_ev, 0, 3).astype(int).astype(str)
    disch_inp_tr = np.array([f"{d}_{b}" for d, b in zip(train["discharge_disposition_id"], inp_bkt_tr)], dtype=object)
    disch_inp_ev = np.array([f"{d}_{b}" for d, b in zip(eval_cols["discharge_disposition_id"], inp_bkt_ev)], dtype=object)

    adm_inp_tr = np.array([f"{a}_{b}" for a, b in zip(train["admission_source_id"], inp_bkt_tr)], dtype=object)
    adm_inp_ev = np.array([f"{a}_{b}" for a, b in zip(eval_cols["admission_source_id"], inp_bkt_ev)], dtype=object)

    d1_disch_pfx_tr = np.array([f"{p}_{d}" for p, d in zip(d1_pfx_tr, train["discharge_disposition_id"])], dtype=object)
    d1_disch_pfx_ev = np.array([f"{p}_{d}" for p, d in zip(d1_pfx_ev, eval_cols["discharge_disposition_id"])], dtype=object)

    payer_disch_tr = np.array([f"{p}_{d}" for p, d in zip(train["payer_code"], train["discharge_disposition_id"])], dtype=object)
    payer_disch_ev = np.array([f"{p}_{d}" for p, d in zip(eval_cols["payer_code"], eval_cols["discharge_disposition_id"])], dtype=object)
    age_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["age"], train["discharge_disposition_id"])], dtype=object)
    age_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["age"], eval_cols["discharge_disposition_id"])], dtype=object)
    age_inp_tr = np.array([f"{a}_{b}" for a, b in zip(train["age"], inp_bkt_tr)], dtype=object)
    age_inp_ev = np.array([f"{a}_{b}" for a, b in zip(eval_cols["age"], inp_bkt_ev)], dtype=object)
    d1_pfx_inp_tr = np.array([f"{p}_{b}" for p, b in zip(d1_pfx_tr, inp_bkt_tr)], dtype=object)
    d1_pfx_inp_ev = np.array([f"{p}_{b}" for p, b in zip(d1_pfx_ev, inp_bkt_ev)], dtype=object)

    te_candidates = [
        ("diag_1", train["diag_1"], eval_cols["diag_1"]),
        ("diag_2", train["diag_2"], eval_cols["diag_2"]),
        ("diag_3", train["diag_3"], eval_cols["diag_3"]),
        ("d1_grp", d1_grp_tr, d1_grp_ev),
        ("d2_grp", d2_grp_tr, d2_grp_ev),
        ("d3_grp", d3_grp_tr, d3_grp_ev),
        ("d1_pfx", d1_pfx_tr, d1_pfx_ev),
        ("d2_pfx", d2_pfx_tr, d2_pfx_ev),
        ("d3_pfx", d3_pfx_tr, d3_pfx_ev),
        ("d1_disch", d1_disch_tr, d1_disch_ev),
        ("adm_disch", adm_disch_tr, adm_disch_ev),
        ("disch_inp", disch_inp_tr, disch_inp_ev),
        ("adm_inp", adm_inp_tr, adm_inp_ev),
        ("d1_disch_pfx", d1_disch_pfx_tr, d1_disch_pfx_ev),
        ("payer_disch", payer_disch_tr, payer_disch_ev),
        ("age_disch", age_disch_tr, age_disch_ev),
        ("age_inp", age_inp_tr, age_inp_ev),
        ("d1_pfx_inp", d1_pfx_inp_tr, d1_pfx_inp_ev),
        ("disch", train["discharge_disposition_id"], eval_cols["discharge_disposition_id"]),
        ("adm_src", train["admission_source_id"], eval_cols["admission_source_id"]),
        ("adm_typ", train["admission_type_id"], eval_cols["admission_type_id"]),
        ("spec", train["medical_specialty"], eval_cols["medical_specialty"]),
        ("payer", train["payer_code"], eval_cols["payer_code"]),
    ]

    te_tr_list, te_ev_list = [], []
    for _, tr_c, ev_c in te_candidates:
        tr_e, ev_e = oof_target_encode(
            tr_c, y_train, ev_c, n_folds=5, smoothing=25.0, seed=42
        )
        te_tr_list.append(tr_e)
        te_ev_list.append(ev_e)

    te_tr_mat = np.column_stack(te_tr_list)
    te_ev_mat = np.column_stack(te_ev_list)

    # 9. Utilisation & intensity features
    util_tr = extract_util_features(train)
    util_ev = extract_util_features(eval_cols)

    # Stack full feature matrix for LightGBM
    x_train_full = np.hstack(
        [
            x_tr_enc,
            util_tr,
            cm_tr_mat,
            cm_der_tr,
            diag_grp_tr,
            spec_diag_tr,
            freq_tr,
            te_tr_mat,
            med_sum_tr,
            med_pca_tr,
            diag_pca_tr,
        ]
    )
    x_eval_full = np.hstack(
        [
            x_ev_enc,
            util_ev,
            cm_ev_mat,
            cm_der_ev,
            diag_grp_ev,
            spec_diag_ev,
            freq_ev,
            te_ev_mat,
            med_sum_ev,
            med_pca_ev,
            diag_pca_ev,
        ]
    )

    # Clean numeric-only feature matrix for XGBoost (excludes arbitrary integer codes)
    num_indices = [i for i in range(x_tr_enc.shape[1]) if i not in cat_indices]
    x_tr_xgb = np.hstack(
        [
            x_tr_enc[:, num_indices],
            util_tr,
            cm_tr_mat,
            cm_der_tr,
            diag_grp_tr,
            spec_diag_tr,
            freq_tr,
            te_tr_mat,
            med_sum_tr,
            med_pca_tr,
            diag_pca_tr,
        ]
    )
    x_ev_xgb = np.hstack(
        [
            x_ev_enc[:, num_indices],
            util_ev,
            cm_ev_mat,
            cm_der_ev,
            diag_grp_ev,
            spec_diag_ev,
            freq_ev,
            te_ev_mat,
            med_sum_ev,
            med_pca_ev,
            diag_pca_ev,
        ]
    )

    # 10. Diverse ensemble: 4 complementary LightGBM configurations + Hist XGBoost
    lgb_mod = globals().get("lgb") or globals().get("lightgbm")

    # M1: Standard balanced depth-6
    m1 = lgb_mod.LGBMClassifier(
        n_estimators=210,
        learning_rate=0.022,
        num_leaves=31,
        max_depth=6,
        min_child_samples=140,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval_full)[:, 1]

    # M2: Shallow, high regularization depth-4
    m2 = lgb_mod.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=220,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=5.0,
        reg_alpha=1.0,
        cat_smooth=70.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval_full)[:, 1]

    # M3: Medium depth-5, wide feature subsampling
    m3 = lgb_mod.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.025,
        num_leaves=24,
        max_depth=5,
        min_child_samples=170,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.8,
        reg_alpha=0.05,
        cat_smooth=55.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval_full)[:, 1]

    # M4: Deep, conservative depth-7, smaller learning rate
    m4 = lgb_mod.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.018,
        num_leaves=36,
        max_depth=7,
        min_child_samples=130,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=2.0,
        reg_alpha=0.2,
        cat_smooth=40.0,
        random_state=888,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p4 = m4.predict_proba(x_eval_full)[:, 1]

    # M5: Extra-Trees mode LightGBM (randomized split thresholds for diversity)
    p5 = None
    try:
        m5 = lgb_mod.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.022,
            num_leaves=31,
            max_depth=6,
            extra_trees=True,
            min_child_samples=130,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.0,
            cat_smooth=45.0,
            random_state=333,
            n_jobs=1,
            verbose=-1,
        )
        m5.fit(x_train_full, y_train, categorical_feature=cat_indices)
        p5 = m5.predict_proba(x_eval_full)[:, 1]
    except Exception:
        p5 = None

    # XGBoost trained on cleaned feature space (dual depth: 4 and 6)
    xgb_mod = globals().get("xgb") or globals().get("xgboost")
    p_xgb1, p_xgb2 = None, None
    if xgb_mod is not None:
        try:
            m_xgb1 = xgb_mod.XGBClassifier(
                n_estimators=200,
                learning_rate=0.026,
                max_depth=4,
                min_child_weight=25,
                subsample=0.75,
                colsample_bytree=0.60,
                reg_alpha=0.5,
                reg_lambda=3.0,
                tree_method="hist",
                random_state=777,
                n_jobs=1,
                eval_metric="logloss",
            )
            m_xgb1.fit(x_tr_xgb, y_train)
            p_xgb1 = m_xgb1.predict_proba(x_ev_xgb)[:, 1]

            m_xgb2 = xgb_mod.XGBClassifier(
                n_estimators=180,
                learning_rate=0.022,
                max_depth=6,
                min_child_weight=40,
                subsample=0.80,
                colsample_bytree=0.50,
                reg_alpha=1.0,
                reg_lambda=5.0,
                tree_method="hist",
                random_state=999,
                n_jobs=1,
                eval_metric="logloss",
            )
            m_xgb2.fit(x_tr_xgb, y_train)
            p_xgb2 = m_xgb2.predict_proba(x_ev_xgb)[:, 1]
        except Exception:
            p_xgb1, p_xgb2 = None, None

    # Sklearn HistGradientBoostingClassifier for third distinct implementation
    p_hgb = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        m_hgb = HistGradientBoostingClassifier(
            max_iter=170,
            learning_rate=0.028,
            max_leaf_nodes=25,
            min_samples_leaf=60,
            l2_regularization=2.0,
            random_state=555,
        )
        m_hgb.fit(x_tr_xgb, y_train)
        p_hgb = m_hgb.predict_proba(x_ev_xgb)[:, 1]
    except Exception:
        p_hgb = None

    # Collect models and assigned weights
    models_preds = [p1, p2, p3, p4]
    model_weights = [0.18, 0.16, 0.13, 0.13]

    if p5 is not None:
        models_preds.append(p5)
        model_weights.append(0.12)
    if p_xgb1 is not None:
        models_preds.append(p_xgb1)
        model_weights.append(0.11)
    if p_xgb2 is not None:
        models_preds.append(p_xgb2)
        model_weights.append(0.09)
    if p_hgb is not None:
        models_preds.append(p_hgb)
        model_weights.append(0.08)

    # Copula Quantile-Matched Rank Aggregation
    preds = rank_average_probabilities(models_preds, model_weights)
    return preds



# EVOLVE-BLOCK-END
# =====================================================================
