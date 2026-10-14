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


def compute_pca_features(
    train_mat: np.ndarray, eval_mat: np.ndarray, n_components: int
):
    """Low-rank spectral projection via exact covariance eigendecomposition."""
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
    """Extract medication dosage intensity and summary regimen features."""
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

    insulin_idx = MED_NAMES.index("insulin")
    insulin_dose = mat[:, insulin_idx : insulin_idx + 1]
    on_insulin = (insulin_dose > 0).astype(np.float64)

    metformin_idx = MED_NAMES.index("metformin")
    on_metformin = (mat[:, metformin_idx : metformin_idx + 1] > 0).astype(
        np.float64
    )
    insulin_oral_combo = ((on_insulin > 0) & (active_count > 1)).astype(
        np.float64
    )

    summary_feats = np.hstack(
        [
            active_count,
            changed_count,
            insulin_dose,
            on_insulin,
            on_metformin,
            insulin_oral_combo,
        ]
    )
    return mat, summary_feats


def extract_diag_pca(
    train: Dict[str, np.ndarray],
    eval_cols: Dict[str, np.ndarray],
    n_components: int = 5,
):
    """Decompose high-order diagnosis co-occurrence into latent comorbidity axes."""
    prefix_counts: Dict[str, int] = {}
    for col_name in ("diag_1", "diag_2", "diag_3"):
        for code in train[col_name]:
            if code is not None and isinstance(code, str) and len(code) >= 3:
                p = code[:3]
                prefix_counts[p] = prefix_counts.get(p, 0) + 1

    top_prefixes = [
        p for p, _ in sorted(prefix_counts.items(), key=lambda x: -x[1])[:30]
    ]
    pfx_to_idx = {p: i for i, p in enumerate(top_prefixes)}

    def build_diag_mat(d):
        n = len(d["diag_1"])
        mat = np.zeros((n, len(top_prefixes)), dtype=np.float64)
        d1, d2, d3 = d["diag_1"], d["diag_2"], d["diag_3"]
        for i in range(n):
            for c in (d1[i], d2[i], d3[i]):
                if c is not None and isinstance(c, str) and len(c) >= 3:
                    idx = pfx_to_idx.get(c[:3])
                    if idx is not None:
                        mat[i, idx] = 1.0
        return mat

    diag_tr = build_diag_mat(train)
    diag_ev = build_diag_mat(eval_cols)
    return compute_pca_features(diag_tr, diag_ev, n_components=n_components)


def extract_util_features(cols_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Clinical utilisation, procedure/medication intensity, and lab indicators."""
    time_hosp = np.asarray(cols_dict["time_in_hospital"], dtype=np.float64)
    n_lab = np.asarray(cols_dict["num_lab_procedures"], dtype=np.float64)
    n_proc = np.asarray(cols_dict["num_procedures"], dtype=np.float64)
    n_med = np.asarray(cols_dict["num_medications"], dtype=np.float64)
    n_out = np.asarray(cols_dict["number_outpatient"], dtype=np.float64)
    n_emg = np.asarray(cols_dict["number_emergency"], dtype=np.float64)
    n_inp = np.asarray(cols_dict["number_inpatient"], dtype=np.float64)
    n_diag = np.asarray(cols_dict["number_diagnoses"], dtype=np.float64)

    total_vis = n_out + n_emg + n_inp
    hosp_safe = np.maximum(time_hosp, 1.0)

    has_inp = (n_inp > 0).astype(np.float64)
    has_emg = (n_emg > 0).astype(np.float64)
    has_out = (n_out > 0).astype(np.float64)
    has_any = (total_vis > 0).astype(np.float64)

    inp_ratio = n_inp / (total_vis + 1.0)
    emg_ratio = n_emg / (total_vis + 1.0)

    inp_per_day = n_inp / hosp_safe
    med_per_day = n_med / hosp_safe
    lab_per_day = n_lab / hosp_safe
    proc_per_day = n_proc / hosp_safe
    diag_per_day = n_diag / hosp_safe

    intensity = (n_lab + 5.0 * n_proc + 2.0 * n_med) / hosp_safe
    inp_x_diag = n_inp * n_diag
    inp_x_emg = n_inp * n_emg
    inp_x_time = n_inp * time_hosp

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

    a1c = cols_dict["A1Cresult"]
    a1c_tested = np.array(
        [0.0 if x == "None" or x is None else 1.0 for x in a1c],
        dtype=np.float64,
    )
    a1c_high = np.array(
        [1.0 if x in (">7", ">8") else 0.0 for x in a1c], dtype=np.float64
    )

    glu = cols_dict["max_glu_serum"]
    glu_tested = np.array(
        [0.0 if x == "None" or x is None else 1.0 for x in glu],
        dtype=np.float64,
    )
    glu_high = np.array(
        [1.0 if x in (">200", ">300") else 0.0 for x in glu], dtype=np.float64
    )

    chg = cols_dict["change"]
    change_flag = np.array(
        [1.0 if x == "Ch" else 0.0 for x in chg], dtype=np.float64
    )
    dm_med = cols_dict["diabetesMed"]
    dm_flag = np.array(
        [1.0 if x == "Yes" else 0.0 for x in dm_med], dtype=np.float64
    )

    return np.column_stack(
        [
            total_vis,
            has_inp,
            has_emg,
            has_out,
            has_any,
            inp_ratio,
            emg_ratio,
            inp_per_day,
            med_per_day,
            lab_per_day,
            proc_per_day,
            diag_per_day,
            intensity,
            inp_x_diag,
            inp_x_emg,
            inp_x_time,
            age_num,
            age_x_inp,
            a1c_tested,
            a1c_high,
            glu_tested,
            glu_high,
            change_flag,
            dm_flag,
        ]
    )


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    encoder = TabularEncoder(min_count=30)
    x_tr_enc = encoder.fit_transform(task, train)
    x_ev_enc = encoder.transform(eval_cols)
    cat_indices = list(encoder.categorical_indices)

    # 1. Low-rank medication regime factors
    med_tr, med_sum_tr = extract_med_features(train)
    med_ev, med_sum_ev = extract_med_features(eval_cols)
    med_pca_tr, med_pca_ev = compute_pca_features(med_tr, med_ev, n_components=4)

    # 2. Low-rank diagnosis co-occurrence factors
    diag_pca_tr, diag_pca_ev = extract_diag_pca(train, eval_cols, n_components=5)

    # 3. Charlson comorbidities
    cm_tr_dict = icd9_comorbidities(
        train["diag_1"], train["diag_2"], train["diag_3"]
    )
    cm_ev_dict = icd9_comorbidities(
        eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"]
    )
    cm_keys = sorted(cm_tr_dict.keys())
    cm_tr_mat = np.column_stack([cm_tr_dict[k] for k in cm_keys])
    cm_ev_mat = np.column_stack([cm_ev_dict[k] for k in cm_keys])

    # 4. Out-of-fold target encodings for high-cardinality features
    d1_grp_tr = np.array([icd9_group(c) for c in train["diag_1"]], dtype=object)
    d1_grp_ev = np.array([icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
    d2_grp_tr = np.array([icd9_group(c) for c in train["diag_2"]], dtype=object)
    d2_grp_ev = np.array([icd9_group(c) for c in eval_cols["diag_2"]], dtype=object)
    d3_grp_tr = np.array([icd9_group(c) for c in train["diag_3"]], dtype=object)
    d3_grp_ev = np.array([icd9_group(c) for c in eval_cols["diag_3"]], dtype=object)

    d1_pfx_tr = np.array(
        [c[:3] if c is not None and isinstance(c, str) else "None" for c in train["diag_1"]],
        dtype=object,
    )
    d1_pfx_ev = np.array(
        [c[:3] if c is not None and isinstance(c, str) else "None" for c in eval_cols["diag_1"]],
        dtype=object,
    )

    te_candidates = [
        ("diag_1", train["diag_1"], eval_cols["diag_1"]),
        ("diag_2", train["diag_2"], eval_cols["diag_2"]),
        ("diag_3", train["diag_3"], eval_cols["diag_3"]),
        ("d1_grp", d1_grp_tr, d1_grp_ev),
        ("d2_grp", d2_grp_tr, d2_grp_ev),
        ("d3_grp", d3_grp_tr, d3_grp_ev),
        ("d1_pfx", d1_pfx_tr, d1_pfx_ev),
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

    # 5. Utilisation & intensity features
    util_tr = extract_util_features(train)
    util_ev = extract_util_features(eval_cols)

    # Horizontally stack all continuous and encoded representations
    x_train_full = np.hstack(
        [x_tr_enc, util_tr, cm_tr_mat, te_tr_mat, med_sum_tr, med_pca_tr, diag_pca_tr]
    )
    x_eval_full = np.hstack(
        [x_ev_enc, util_ev, cm_ev_mat, te_ev_mat, med_sum_ev, med_pca_ev, diag_pca_ev]
    )

    # 6. Multi-scale regularized gradient boosting ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=150,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval_full)[:, 1]

    m2 = lgb.LGBMClassifier(
        n_estimators=210,
        learning_rate=0.025,
        num_leaves=16,
        max_depth=4,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=5.0,
        reg_alpha=1.0,
        cat_smooth=75.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval_full)[:, 1]

    m3 = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.025,
        num_leaves=24,
        max_depth=5,
        min_child_samples=180,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.5,
        reg_alpha=0.05,
        cat_smooth=60.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train_full, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval_full)[:, 1]

    preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
    return np.clip(preds, 0.0, 1.0)



# EVOLVE-BLOCK-END
# =====================================================================
