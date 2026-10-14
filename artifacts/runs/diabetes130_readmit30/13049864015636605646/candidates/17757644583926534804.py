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
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    from collections import Counter
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    try:
        from clinical_toolkit import (
            TabularEncoder,
            icd9_comorbidities,
            icd9_group,
            oof_target_encode,
        )
    except ImportError:
        pass

    n_train = len(y_train)
    n_eval = len(eval_cols["time_in_hospital"])

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson comorbidities derived from diagnosis codes
    cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k in sorted(cm_train.keys()):
        train_feats[k] = cm_train[k].astype(np.float64)
        eval_feats[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # High-risk comorbidity clusters (Cardiorenal & Cardiopulmonary synergy)
    cardiorenal_tr = (cm_train["cm_chf"] * cm_train["cm_renal"]).astype(np.float64)
    cardiorenal_ev = (cm_eval["cm_chf"] * cm_eval["cm_renal"]).astype(np.float64)
    train_feats["cm_cardiorenal"] = cardiorenal_tr
    eval_feats["cm_cardiorenal"] = cardiorenal_ev
    extra_numeric.append("cm_cardiorenal")

    chf_copd_tr = (cm_train["cm_chf"] * cm_train["cm_pulmonary"]).astype(np.float64)
    chf_copd_ev = (cm_eval["cm_chf"] * cm_eval["cm_pulmonary"]).astype(np.float64)
    train_feats["cm_chf_copd"] = chf_copd_tr
    eval_feats["cm_chf_copd"] = chf_copd_ev
    extra_numeric.append("cm_chf_copd")

    vasc_tr = ((cm_train["cm_mi"] + cm_train["cm_pvd"] + cm_train["cm_cerebrovascular"] > 0).astype(np.float64) * cm_train["cm_chf"])
    vasc_ev = ((cm_eval["cm_mi"] + cm_eval["cm_pvd"] + cm_eval["cm_cerebrovascular"] > 0).astype(np.float64) * cm_eval["cm_chf"])
    train_feats["cm_complex_vascular"] = vasc_tr.astype(np.float64)
    eval_feats["cm_complex_vascular"] = vasc_ev.astype(np.float64)
    extra_numeric.append("cm_complex_vascular")

    multimorbid_tr = (cm_train["cm_count"] >= 3.0).astype(np.float64)
    multimorbid_ev = (cm_eval["cm_count"] >= 3.0).astype(np.float64)
    train_feats["cm_multimorbid"] = multimorbid_tr
    eval_feats["cm_multimorbid"] = multimorbid_ev
    extra_numeric.append("cm_multimorbid")

    # 2. ICD-9 Chapter groupings (organ systems) & Multi-system Homeostatic Entropy
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        gcol = f"{diag_col}_group"
        train_feats[gcol] = np.array([icd9_group(c) for c in train[diag_col]], dtype=object)
        eval_feats[gcol] = np.array([icd9_group(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(gcol)

    train_feats["distinct_organ_systems"] = np.array([
        len({a, b, c} - {"missing", "other", None})
        for a, b, c in zip(train_feats["diag_1_group"], train_feats["diag_2_group"], train_feats["diag_3_group"])
    ], dtype=np.float64)
    eval_feats["distinct_organ_systems"] = np.array([
        len({a, b, c} - {"missing", "other", None})
        for a, b, c in zip(eval_feats["diag_1_group"], eval_feats["diag_2_group"], eval_feats["diag_3_group"])
    ], dtype=np.float64)
    extra_numeric.append("distinct_organ_systems")

    # 3-digit ICD-9 disease family groupings
    def to_3digit(code):
        if code is None or code == "" or code == "?":
            return "missing"
        s = str(code).strip()
        if "." in s:
            s = s.split(".")[0]
        return s[:3] if len(s) >= 3 else s

    diag_1_3d_tr = np.array([to_3digit(c) for c in train["diag_1"]], dtype=object)
    diag_1_3d_ev = np.array([to_3digit(c) for c in eval_cols["diag_1"]], dtype=object)
    train_feats["diag_1_3d"] = diag_1_3d_tr
    eval_feats["diag_1_3d"] = diag_1_3d_ev
    extra_categorical.append("diag_1_3d")

    te_d1_3d_tr, te_d1_3d_ev = oof_target_encode(diag_1_3d_tr, y_train, diag_1_3d_ev, n_folds=5, smoothing=25.0, seed=42)
    train_feats["te_diag_1_3d"] = te_d1_3d_tr.astype(np.float64)
    eval_feats["te_diag_1_3d"] = te_d1_3d_ev.astype(np.float64)
    extra_numeric.append("te_diag_1_3d")

    # 3. Out-of-fold target encoding for high-cardinality nominals & Clinical Pathways
    te_columns = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "admission_type_id",
    ]
    for col in te_columns:
        if col in train and col in eval_cols:
            te_tr, te_ev = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
            )
            col_name = f"te_{col}"
            train_feats[col_name] = te_tr.astype(np.float64)
            eval_feats[col_name] = te_ev.astype(np.float64)
            extra_numeric.append(col_name)

    # Clinical Trajectory Interactions: Diagnosis Chapter x Discharge Disposition & Admission Source
    inter1_tr = np.array([f"{g}_{d}" for g, d in zip(train_feats["diag_1_group"], train["discharge_disposition_id"])], dtype=object)
    inter1_ev = np.array([f"{g}_{d}" for g, d in zip(eval_feats["diag_1_group"], eval_cols["discharge_disposition_id"])], dtype=object)
    te_i1_tr, te_i1_ev = oof_target_encode(inter1_tr, y_train, inter1_ev, n_folds=5, smoothing=20.0, seed=42)
    train_feats["te_diag1_disposition"] = te_i1_tr.astype(np.float64)
    eval_feats["te_diag1_disposition"] = te_i1_ev.astype(np.float64)
    extra_numeric.append("te_diag1_disposition")

    inter2_tr = np.array([f"{g}_{a}" for g, a in zip(train_feats["diag_1_group"], train["admission_source_id"])], dtype=object)
    inter2_ev = np.array([f"{g}_{a}" for g, a in zip(eval_feats["diag_1_group"], eval_cols["admission_source_id"])], dtype=object)
    te_i2_tr, te_i2_ev = oof_target_encode(inter2_tr, y_train, inter2_ev, n_folds=5, smoothing=20.0, seed=42)
    train_feats["te_diag1_admsource"] = te_i2_tr.astype(np.float64)
    eval_feats["te_diag1_admsource"] = te_i2_ev.astype(np.float64)
    extra_numeric.append("te_diag1_admsource")

    # Admission Source x Discharge Disposition Care Pathway Journey
    inter3_tr = np.array([f"{a}_{d}" for a, d in zip(train["admission_source_id"], train["discharge_disposition_id"])], dtype=object)
    inter3_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["admission_source_id"], eval_cols["discharge_disposition_id"])], dtype=object)
    te_i3_tr, te_i3_ev = oof_target_encode(inter3_tr, y_train, inter3_ev, n_folds=5, smoothing=20.0, seed=42)
    train_feats["te_admsource_disposition"] = te_i3_tr.astype(np.float64)
    eval_feats["te_admsource_disposition"] = te_i3_ev.astype(np.float64)
    extra_numeric.append("te_admsource_disposition")

    # 4. Ordinal continuous age parsing
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    train_feats["age_num"] = np.array([age_map.get(a, np.nan) for a in train["age"]], dtype=np.float64)
    eval_feats["age_num"] = np.array([age_map.get(a, np.nan) for a in eval_cols["age"]], dtype=np.float64)
    extra_numeric.append("age_num")

    # 5. Prior healthcare utilization & care intensity metrics
    t_hosp_tr = np.maximum(np.nan_to_num(train["time_in_hospital"], nan=1.0), 1.0)
    t_hosp_ev = np.maximum(np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0), 1.0)

    inpat_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inpat_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emerg_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emerg_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    outpat_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    outpat_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    prior_tot_tr = inpat_tr + emerg_tr + outpat_tr
    prior_tot_ev = inpat_ev + emerg_ev + outpat_ev

    util_feats = {
        "prior_visits_total": (prior_tot_tr, prior_tot_ev),
        "log_inpatient": (np.log1p(inpat_tr), np.log1p(inpat_ev)),
        "log_emergency": (np.log1p(emerg_tr), np.log1p(emerg_ev)),
        "log_prior_tot": (np.log1p(prior_tot_tr), np.log1p(prior_tot_ev)),
        "has_prior_inpatient": ((inpat_tr > 0).astype(np.float64), (inpat_ev > 0).astype(np.float64)),
        "has_prior_emergency": ((emerg_tr > 0).astype(np.float64), (emerg_ev > 0).astype(np.float64)),
        "has_prior_outpatient": ((outpat_tr > 0).astype(np.float64), (outpat_ev > 0).astype(np.float64)),
        "inpatient_ratio": (inpat_tr / (prior_tot_tr + 1.0), inpat_ev / (prior_tot_ev + 1.0)),
        "emergency_ratio": (emerg_tr / (prior_tot_tr + 1.0), emerg_ev / (prior_tot_ev + 1.0)),
        "inpatient_emergency_interaction": (inpat_tr * emerg_tr, inpat_ev * emerg_ev),
        "inpatient_sq": (inpat_tr ** 2, inpat_ev ** 2),
        "med_per_day": (np.nan_to_num(train["num_medications"], nan=0.0) / t_hosp_tr,
                        np.nan_to_num(eval_cols["num_medications"], nan=0.0) / t_hosp_ev),
        "lab_per_day": (np.nan_to_num(train["num_lab_procedures"], nan=0.0) / t_hosp_tr,
                        np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0) / t_hosp_ev),
        "proc_per_day": (np.nan_to_num(train["num_procedures"], nan=0.0) / t_hosp_tr,
                         np.nan_to_num(eval_cols["num_procedures"], nan=0.0) / t_hosp_ev),
        "diag_per_day": (np.nan_to_num(train["number_diagnoses"], nan=0.0) / t_hosp_tr,
                         np.nan_to_num(eval_cols["number_diagnoses"], nan=0.0) / t_hosp_ev),
        "med_x_time": (np.nan_to_num(train["num_medications"], nan=0.0) * t_hosp_tr,
                       np.nan_to_num(eval_cols["num_medications"], nan=0.0) * t_hosp_ev),
        "total_procedures": (np.nan_to_num(train["num_lab_procedures"], nan=0.0) + np.nan_to_num(train["num_procedures"], nan=0.0),
                            np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0) + np.nan_to_num(eval_cols["num_procedures"], nan=0.0)),
        "clinical_acuity_index": (inpat_tr * 3.5 + emerg_tr * 1.8 + (t_hosp_tr >= 5).astype(np.float64) * 1.5 + (np.nan_to_num(train["number_diagnoses"], nan=0.0) >= 9).astype(np.float64),
                                  inpat_ev * 3.5 + emerg_ev * 1.8 + (t_hosp_ev >= 5).astype(np.float64) * 1.5 + (np.nan_to_num(eval_cols["number_diagnoses"], nan=0.0) >= 9).astype(np.float64)),
        "frequent_inpatient": ((inpat_tr >= 2.0).astype(np.float64), (inpat_ev >= 2.0).astype(np.float64)),
        "high_inpatient": ((inpat_tr >= 3.0).astype(np.float64), (inpat_ev >= 3.0).astype(np.float64)),
        "utilization_velocity": (emerg_tr / (inpat_tr + 1.0), emerg_ev / (inpat_ev + 1.0)),
        "inpatient_per_day": (inpat_tr / t_hosp_tr, inpat_ev / t_hosp_ev),
    }
    for fname, (ftr, fev) in util_feats.items():
        train_feats[fname] = ftr.astype(np.float64)
        eval_feats[fname] = fev.astype(np.float64)
        extra_numeric.append(fname)

    # Specific Discharge & Admission clinical indicators
    train_feats["disch_home"] = (train["discharge_disposition_id"] == "1").astype(np.float64)
    eval_feats["disch_home"] = (eval_cols["discharge_disposition_id"] == "1").astype(np.float64)
    extra_numeric.append("disch_home")

    train_feats["disch_snf"] = np.isin(train["discharge_disposition_id"], ["3", "4", "5", "18", "22"]).astype(np.float64)
    eval_feats["disch_snf"] = np.isin(eval_cols["discharge_disposition_id"], ["3", "4", "5", "18", "22"]).astype(np.float64)
    extra_numeric.append("disch_snf")

    train_feats["disch_ama"] = (train["discharge_disposition_id"] == "7").astype(np.float64)
    eval_feats["disch_ama"] = (eval_cols["discharge_disposition_id"] == "7").astype(np.float64)
    extra_numeric.append("disch_ama")

    train_feats["admit_er"] = (train["admission_source_id"] == "7").astype(np.float64)
    eval_feats["admit_er"] = (eval_cols["admission_source_id"] == "7").astype(np.float64)
    extra_numeric.append("admit_er")

    # 6. Pharmacotherapy profile extraction & medication indicators
    med_cols = [
        col for col in task.categorical_columns
        if set(np.unique(train[col][train[col] != None])).issubset({"No", "Steady", "Up", "Down"})
    ]

    med_mat_tr = np.zeros((n_train, len(med_cols)), dtype=np.float64)
    med_mat_ev = np.zeros((n_eval, len(med_cols)), dtype=np.float64)
    for j, mcol in enumerate(med_cols):
        tr_c = train[mcol]
        ev_c = eval_cols[mcol]
        med_mat_tr[tr_c == "Steady", j] = 1.0
        med_mat_tr[tr_c == "Down", j] = -1.0
        med_mat_tr[tr_c == "Up", j] = 2.0
        med_mat_ev[ev_c == "Steady", j] = 1.0
        med_mat_ev[ev_c == "Down", j] = -1.0
        med_mat_ev[ev_c == "Up", j] = 2.0

    train_feats["med_active_count"] = np.sum(med_mat_tr != 0.0, axis=1, dtype=np.float64)
    eval_feats["med_active_count"] = np.sum(med_mat_ev != 0.0, axis=1, dtype=np.float64)
    extra_numeric.append("med_active_count")

    train_feats["med_up_count"] = np.sum(med_mat_tr == 2.0, axis=1, dtype=np.float64)
    eval_feats["med_up_count"] = np.sum(med_mat_ev == 2.0, axis=1, dtype=np.float64)
    extra_numeric.append("med_up_count")

    train_feats["med_down_count"] = np.sum(med_mat_tr == -1.0, axis=1, dtype=np.float64)
    eval_feats["med_down_count"] = np.sum(med_mat_ev == -1.0, axis=1, dtype=np.float64)
    extra_numeric.append("med_down_count")

    train_feats["med_change_count"] = np.sum((med_mat_tr == 2.0) | (med_mat_tr == -1.0), axis=1, dtype=np.float64)
    eval_feats["med_change_count"] = np.sum((med_mat_ev == 2.0) | (med_mat_ev == -1.0), axis=1, dtype=np.float64)
    extra_numeric.append("med_change_count")

    if "change" in train:
        train_feats["has_change"] = (train["change"] == "Ch").astype(np.float64)
        eval_feats["has_change"] = (eval_cols["change"] == "Ch").astype(np.float64)
        extra_numeric.append("has_change")

    if "diabetesMed" in train:
        train_feats["has_diabetes_med"] = (train["diabetesMed"] == "Yes").astype(np.float64)
        eval_feats["has_diabetes_med"] = (eval_cols["diabetesMed"] == "Yes").astype(np.float64)
        extra_numeric.append("has_diabetes_med")

    if "insulin" in train:
        ins_map = {"No": 0.0, "Steady": 1.0, "Down": 2.0, "Up": 3.0}
        train_feats["insulin_status"] = np.array([ins_map.get(v, 0.0) for v in train["insulin"]], dtype=np.float64)
        eval_feats["insulin_status"] = np.array([ins_map.get(v, 0.0) for v in eval_cols["insulin"]], dtype=np.float64)
        extra_numeric.append("insulin_status")

        train_feats["insulin_titrated"] = np.isin(train["insulin"], ["Up", "Down"]).astype(np.float64)
        eval_feats["insulin_titrated"] = np.isin(eval_cols["insulin"], ["Up", "Down"]).astype(np.float64)
        extra_numeric.append("insulin_titrated")

    # Glycemic Biomarkers & Clinical Inertia (uncontrolled glucose with unchanged regimen)
    a1c_tested_tr = np.array([c is not None and c not in ("None", "") for c in train["A1Cresult"]], dtype=np.float64)
    a1c_tested_ev = np.array([c is not None and c not in ("None", "") for c in eval_cols["A1Cresult"]], dtype=np.float64)
    a1c_high_tr = np.array([c == ">8" for c in train["A1Cresult"]], dtype=np.float64)
    a1c_high_ev = np.array([c == ">8" for c in eval_cols["A1Cresult"]], dtype=np.float64)

    glu_tested_tr = np.array([c is not None and c not in ("None", "") for c in train["max_glu_serum"]], dtype=np.float64)
    glu_tested_ev = np.array([c is not None and c not in ("None", "") for c in eval_cols["max_glu_serum"]], dtype=np.float64)
    glu_high_tr = np.array([c in (">300", ">200") for c in train["max_glu_serum"]], dtype=np.float64)
    glu_high_ev = np.array([c in (">300", ">200") for c in eval_cols["max_glu_serum"]], dtype=np.float64)

    chg_no_tr = (train["change"] == "No").astype(np.float64)
    chg_no_ev = (eval_cols["change"] == "No").astype(np.float64)

    train_feats["a1c_tested"] = a1c_tested_tr
    eval_feats["a1c_tested"] = a1c_tested_ev
    extra_numeric.append("a1c_tested")

    train_feats["a1c_high"] = a1c_high_tr
    eval_feats["a1c_high"] = a1c_high_ev
    extra_numeric.append("a1c_high")

    train_feats["glu_high"] = glu_high_tr
    eval_feats["glu_high"] = glu_high_ev
    extra_numeric.append("glu_high")

    train_feats["clinical_inertia"] = (((a1c_high_tr > 0) | (glu_high_tr > 0)) & (chg_no_tr > 0)).astype(np.float64)
    eval_feats["clinical_inertia"] = (((a1c_high_ev > 0) | (glu_high_ev > 0)) & (chg_no_ev > 0)).astype(np.float64)
    extra_numeric.append("clinical_inertia")

    # 6b. ICD-9 Disease Co-occurrence Graph Spectral Embedding & Syndemic Divergence
    all_diag_train = []
    for col in ("diag_1", "diag_2", "diag_3"):
        if col in train:
            all_diag_train.extend([c for c in train[col] if c is not None and c != "?"])

    diag_counts = Counter(all_diag_train)
    # Corrected indexing: assign contiguous indices 0..V-1 to frequent codes
    frequent_codes = [code for code, cnt in diag_counts.items() if cnt >= 15]
    diag_vocab = {code: idx for idx, code in enumerate(frequent_codes)}
    V = len(diag_vocab)

    if V >= 15:
        C_diag = np.zeros((V, V), dtype=np.float64)
        for i in range(n_train):
            codes = []
            for c in (train["diag_1"][i], train["diag_2"][i], train["diag_3"][i]):
                if c in diag_vocab:
                    codes.append(diag_vocab[c])
            codes = list(set(codes))
            for idx_a in codes:
                C_diag[idx_a, idx_a] += 1.0
                for idx_b in codes:
                    if idx_a != idx_b:
                        C_diag[idx_a, idx_b] += 1.0

        total_cooc = np.sum(C_diag)
        if total_cooc > 0:
            p_i = np.sum(C_diag, axis=1) / total_cooc
            outer_p = np.outer(p_i, p_i) + 1e-9
            ppmi = np.maximum(0.0, np.log(np.maximum(C_diag / total_cooc, 1e-9) / outer_p))

            n_diag_dim = 6
            svd_diag = TruncatedSVD(n_components=n_diag_dim, random_state=42)
            diag_embed_mat = svd_diag.fit_transform(ppmi)

            embed_table = np.vstack([diag_embed_mat, np.zeros((1, n_diag_dim), dtype=np.float64)])

            for split_name, d_dict in [("train", train), ("eval", eval_cols)]:
                d1_idx = np.array([diag_vocab.get(c, -1) for c in d_dict["diag_1"]], dtype=np.int32)
                d2_idx = np.array([diag_vocab.get(c, -1) for c in d_dict["diag_2"]], dtype=np.int32)
                d3_idx = np.array([diag_vocab.get(c, -1) for c in d_dict["diag_3"]], dtype=np.int32)
                d1_idx[d1_idx < 0] = V
                d2_idx[d2_idx < 0] = V
                d3_idx[d3_idx < 0] = V

                v1 = embed_table[d1_idx]
                v2 = embed_table[d2_idx]
                v3 = embed_table[d3_idx]

                m1 = (d1_idx != V).astype(np.float64)[:, None]
                m2 = (d2_idx != V).astype(np.float64)[:, None]
                m3 = (d3_idx != V).astype(np.float64)[:, None]
                denom = np.maximum(m1 + m2 + m3, 1.0)
                emb_means = (v1 * m1 + v2 * m2 + v3 * m3) / denom

                dist12 = np.sqrt(np.sum((v1 - v2) ** 2, axis=1)) * (m1[:, 0] * m2[:, 0])
                dist13 = np.sqrt(np.sum((v1 - v3) ** 2, axis=1)) * (m1[:, 0] * m3[:, 0])
                d_count = np.maximum(m1[:, 0] * m2[:, 0] + m1[:, 0] * m3[:, 0], 1.0)
                divergence = (dist12 + dist13) / d_count

                target_dict = train_feats if split_name == "train" else eval_feats
                for dim_k in range(n_diag_dim):
                    dname = f"diag_spectral_{dim_k}"
                    target_dict[dname] = emb_means[:, dim_k].astype(np.float64)
                    if split_name == "train":
                        extra_numeric.append(dname)
                target_dict["syndemic_divergence"] = divergence.astype(np.float64)
                if split_name == "train":
                    extra_numeric.append("syndemic_divergence")

    # 7. Low-Rank Tensor/Matrix Decomposition of Multimodal Clinical Phenotypes
    cm_flag_keys = [k for k in sorted(cm_train.keys()) if k.startswith("cm_") and k != "cm_count"]
    cm_mat_tr = np.column_stack([cm_train[k] for k in cm_flag_keys])
    cm_mat_ev = np.column_stack([cm_eval[k] for k in cm_flag_keys])

    # Unfolded joint clinical state: Patient x (Pharmacotherapy + Comorbidities)
    clin_tr = np.column_stack([med_mat_tr, cm_mat_tr])
    clin_ev = np.column_stack([med_mat_ev, cm_mat_ev])

    n_svd = 6
    svd_clin = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_tr = svd_clin.fit_transform(clin_tr)
    svd_ev = svd_clin.transform(clin_ev)

    for k in range(n_svd):
        cname = f"svd_clin_{k}"
        train_feats[cname] = svd_tr[:, k].astype(np.float64)
        eval_feats[cname] = svd_ev[:, k].astype(np.float64)
        extra_numeric.append(cname)

    # Low-rank subspace reconstruction error: measures patient atypicality / clinical anomaly
    recon_tr = clin_tr - svd_clin.inverse_transform(svd_tr)
    recon_ev = clin_ev - svd_clin.inverse_transform(svd_ev)
    train_feats["clin_recon_err"] = np.sum(recon_tr ** 2, axis=1, dtype=np.float64)
    eval_feats["clin_recon_err"] = np.sum(recon_ev ** 2, axis=1, dtype=np.float64)
    extra_numeric.append("clin_recon_err")

    # Latent pharmacotherapy axes
    if med_mat_tr.shape[1] >= 4:
        svd_med = TruncatedSVD(n_components=3, random_state=42)
        svd_m_tr = svd_med.fit_transform(med_mat_tr)
        svd_m_ev = svd_med.transform(med_mat_ev)
        for k in range(3):
            mname = f"svd_med_{k}"
            train_feats[mname] = svd_m_tr[:, k].astype(np.float64)
            eval_feats[mname] = svd_m_ev[:, k].astype(np.float64)
            extra_numeric.append(mname)

    # 7b. Phenotypic Neighborhood Risk Gravitation (Out-of-Fold Latent Manifold Density)
    knn_feats_tr = np.column_stack([
        svd_tr,
        np.log1p(inpat_tr),
        np.log1p(emerg_tr),
        np.log1p(prior_tot_tr),
    ])
    knn_feats_ev = np.column_stack([
        svd_ev,
        np.log1p(inpat_ev),
        np.log1p(emerg_ev),
        np.log1p(prior_tot_ev),
    ])

    scaler = StandardScaler()
    knn_tr_scaled = scaler.fit_transform(knn_feats_tr)
    knn_ev_scaled = scaler.transform(knn_feats_ev)

    k_neighbors = 35
    oof_knn_risk_tr = np.zeros(n_train, dtype=np.float64)
    oof_knn_dist_tr = np.zeros(n_train, dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(knn_tr_scaled):
        nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree", n_jobs=1)
        nn.fit(knn_tr_scaled[tr_idx])
        distances, indices = nn.kneighbors(knn_tr_scaled[val_idx])
        weights = 1.0 / (distances + 1e-3)
        y_sub = y_train[tr_idx]
        val_y_neighbors = y_sub[indices]
        oof_knn_risk_tr[val_idx] = np.sum(val_y_neighbors * weights, axis=1) / np.sum(weights, axis=1)
        oof_knn_dist_tr[val_idx] = np.mean(distances, axis=1)

    nn_full = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree", n_jobs=1)
    nn_full.fit(knn_tr_scaled)
    distances_ev, indices_ev = nn_full.kneighbors(knn_ev_scaled)
    weights_ev = 1.0 / (distances_ev + 1e-3)
    ev_y_neighbors = y_train[indices_ev]
    knn_risk_ev = np.sum(ev_y_neighbors * weights_ev, axis=1) / np.sum(weights_ev, axis=1)
    knn_dist_ev = np.mean(distances_ev, axis=1)

    train_feats["phenotype_neighbor_risk"] = oof_knn_risk_tr
    eval_feats["phenotype_neighbor_risk"] = knn_risk_ev
    extra_numeric.append("phenotype_neighbor_risk")

    train_feats["phenotype_neighbor_dist"] = oof_knn_dist_tr
    eval_feats["phenotype_neighbor_dist"] = knn_dist_ev
    extra_numeric.append("phenotype_neighbor_dist")

    # 8. Encode features
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_feats, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 9. Diverse Gradient-Boosted Decision Tree Ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=130,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.1,
        reg_lambda=1.5,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    model2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.018,
        num_leaves=20,
        max_depth=5,
        min_child_samples=220,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=5.0,
        cat_smooth=75.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )

    model3 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.02,
        num_leaves=25,
        max_depth=6,
        min_child_samples=170,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.25,
        reg_lambda=3.0,
        cat_smooth=60.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)

    preds1 = model1.predict_proba(x_eval)[:, 1]
    preds2 = model2.predict_proba(x_eval)[:, 1]
    preds3 = model3.predict_proba(x_eval)[:, 1]

    # Incorporate histogram XGBoost for model family diversity
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.80,
            colsample_bytree=0.60,
            reg_alpha=0.5,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=777,
            n_jobs=1,
        )
        xgb_model.fit(x_train, y_train)
        preds_xgb = xgb_model.predict_proba(x_eval)[:, 1]
        ensemble_preds = 0.30 * preds1 + 0.30 * preds2 + 0.25 * preds3 + 0.15 * preds_xgb
    except Exception:
        ensemble_preds = 0.35 * preds1 + 0.35 * preds2 + 0.30 * preds3

    return np.clip(ensemble_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
