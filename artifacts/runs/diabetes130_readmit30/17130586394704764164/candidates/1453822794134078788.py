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
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])

    # 1. Comorbidities via Charlson mapping & severe clinical triads
    cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    for k, v in cm_train.items():
        train_dict[k] = v.astype(np.float64)
        eval_dict[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    if "cm_chf" in cm_train and "cm_renal" in cm_train:
        train_dict["cm_cardiorenal"] = (cm_train["cm_chf"] * cm_train["cm_renal"]).astype(np.float64)
        eval_dict["cm_cardiorenal"] = (cm_eval["cm_chf"] * cm_eval["cm_renal"]).astype(np.float64)
        extra_numeric.append("cm_cardiorenal")

    if "cm_pulmonary" in cm_train and "cm_chf" in cm_train:
        train_dict["cm_cardiopulmonary"] = (cm_train["cm_pulmonary"] * cm_train["cm_chf"]).astype(np.float64)
        eval_dict["cm_cardiopulmonary"] = (cm_eval["cm_pulmonary"] * cm_eval["cm_chf"]).astype(np.float64)
        extra_numeric.append("cm_cardiopulmonary")

    if "cm_chf" in cm_train and "cm_renal" in cm_train and "cm_diabetes_complicated" in cm_train:
        train_dict["cm_severe_triad"] = (cm_train["cm_chf"] * cm_train["cm_renal"] * cm_train["cm_diabetes_complicated"]).astype(np.float64)
        eval_dict["cm_severe_triad"] = (cm_eval["cm_chf"] * cm_eval["cm_renal"] * cm_eval["cm_diabetes_complicated"]).astype(np.float64)
        extra_numeric.append("cm_severe_triad")

    # 2. ICD-9 Chapter groupings & 3-Digit Category Mining
    d1_tr = np.array([icd9_group(x) for x in train["diag_1"]], dtype=object)
    d1_ev = np.array([icd9_group(x) for x in eval_cols["diag_1"]], dtype=object)
    d2_tr = np.array([icd9_group(x) for x in train["diag_2"]], dtype=object)
    d2_ev = np.array([icd9_group(x) for x in eval_cols["diag_2"]], dtype=object)
    d3_tr = np.array([icd9_group(x) for x in train["diag_3"]], dtype=object)
    d3_ev = np.array([icd9_group(x) for x in eval_cols["diag_3"]], dtype=object)

    train_dict["diag_1_group"] = d1_tr
    eval_dict["diag_1_group"] = d1_ev
    train_dict["diag_2_group"] = d2_tr
    eval_dict["diag_2_group"] = d2_ev
    train_dict["diag_3_group"] = d3_tr
    eval_dict["diag_3_group"] = d3_ev
    extra_categorical.extend(["diag_1_group", "diag_2_group", "diag_3_group"])

    d12_tr = np.array([f"{g1}_{g2}" for g1, g2 in zip(d1_tr, d2_tr)], dtype=object)
    d12_ev = np.array([f"{g1}_{g2}" for g1, g2 in zip(d1_ev, d2_ev)], dtype=object)
    train_dict["diag_1_2_group"] = d12_tr
    eval_dict["diag_1_2_group"] = d12_ev
    extra_categorical.append("diag_1_2_group")

    d1_3tr = np.array([str(x)[:3] if x is not None and str(x) != "None" else "missing" for x in train["diag_1"]], dtype=object)
    d1_3ev = np.array([str(x)[:3] if x is not None and str(x) != "None" else "missing" for x in eval_cols["diag_1"]], dtype=object)
    d2_3tr = np.array([str(x)[:3] if x is not None and str(x) != "None" else "missing" for x in train["diag_2"]], dtype=object)
    d2_3ev = np.array([str(x)[:3] if x is not None and str(x) != "None" else "missing" for x in eval_cols["diag_2"]], dtype=object)
    d3_3tr = np.array([str(x)[:3] if x is not None and str(x) != "None" else "missing" for x in train["diag_3"]], dtype=object)
    d3_3ev = np.array([str(x)[:3] if x is not None and str(x) != "None" else "missing" for x in eval_cols["diag_3"]], dtype=object)

    train_dict["diag_1_3digit"] = d1_3tr
    eval_dict["diag_1_3digit"] = d1_3ev
    train_dict["diag_2_3digit"] = d2_3tr
    eval_dict["diag_2_3digit"] = d2_3ev
    train_dict["diag_3_3digit"] = d3_3tr
    eval_dict["diag_3_3digit"] = d3_3ev
    extra_categorical.extend(["diag_1_3digit", "diag_2_3digit", "diag_3_3digit"])

    def compute_diag_diversity(d1, d2, d3):
        return np.array([
            float(len({c1, c2, c3} - {"missing", ""}))
            for c1, c2, c3 in zip(d1, d2, d3)
        ], dtype=np.float64)

    train_dict["diag_diversity"] = compute_diag_diversity(d1_3tr, d2_3tr, d3_3tr)
    eval_dict["diag_diversity"] = compute_diag_diversity(d1_3ev, d2_3ev, d3_3ev)
    extra_numeric.append("diag_diversity")

    # 2b. Levy-Goldberg Spectral Disease Embeddings & Semantic Clinical Dispersion
    # Construct a clinical disease co-occurrence graph and compute low-rank PPMI spectral embeddings
    code_counts = {}
    for c1, c2, c3 in zip(d1_3tr, d2_3tr, d3_3tr):
        for c in (c1, c2, c3):
            if c != "missing" and c != "":
                code_counts[c] = code_counts.get(c, 0) + 1

    vocab = [c for c, cnt in code_counts.items() if cnt >= 12]
    vocab_map = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    if V >= 20:
        C_mat = np.zeros((V, V), dtype=np.float64)
        for c1, c2, c3 in zip(d1_3tr, d2_3tr, d3_3tr):
            idxs = [vocab_map[c] for c in (c1, c2, c3) if c in vocab_map]
            idxs = list(set(idxs))
            for i_a in idxs:
                C_mat[i_a, i_a] += 1.0
                for i_b in idxs:
                    if i_a != i_b:
                        C_mat[i_a, i_b] += 1.0

        total_pairs = float(np.sum(C_mat))
        p_diag = np.sum(C_mat, axis=1) / (total_pairs + 1e-9)
        expected = np.outer(p_diag, p_diag) * total_pairs
        ppmi = np.maximum(0.0, np.log((C_mat + 1.0) / (expected + 1.0)))

        U, s, _ = np.linalg.svd(ppmi, full_matrices=False)
        k_embed = min(6, V)
        code_embed = U[:, :k_embed] * np.sqrt(s[:k_embed])

        def get_encounter_spectral_feats(d1_arr, d2_arr, d3_arr, n_rows):
            feats = np.zeros((n_rows, k_embed + 2), dtype=np.float64)
            for r in range(n_rows):
                v0 = code_embed[vocab_map[d1_arr[r]]] if d1_arr[r] in vocab_map else None
                v1 = code_embed[vocab_map[d2_arr[r]]] if d2_arr[r] in vocab_map else None
                v2 = code_embed[vocab_map[d3_arr[r]]] if d3_arr[r] in vocab_map else None
                valid = [v for v in (v0, v1, v2) if v is not None]
                n_v = len(valid)
                if n_v == 1:
                    feats[r, :k_embed] = valid[0]
                elif n_v == 2:
                    feats[r, :k_embed] = 0.5 * (valid[0] + valid[1])
                    d = float(np.linalg.norm(valid[0] - valid[1]))
                    feats[r, k_embed] = d
                    feats[r, k_embed + 1] = d
                elif n_v == 3:
                    feats[r, :k_embed] = (valid[0] + valid[1] + valid[2]) / 3.0
                    d01 = float(np.linalg.norm(valid[0] - valid[1]))
                    d02 = float(np.linalg.norm(valid[0] - valid[2]))
                    d12 = float(np.linalg.norm(valid[1] - valid[2]))
                    feats[r, k_embed] = (d01 + d02 + d12) / 3.0
                    feats[r, k_embed + 1] = max(d01, d02, d12)
            return feats

        sp_tr = get_encounter_spectral_feats(d1_3tr, d2_3tr, d3_3tr, n_tr)
        sp_ev = get_encounter_spectral_feats(d1_3ev, d2_3ev, d3_3ev, n_ev)

        for dim in range(k_embed):
            col_name = f"spectral_diag_dim_{dim+1}"
            train_dict[col_name] = sp_tr[:, dim]
            eval_dict[col_name] = sp_ev[:, dim]
            extra_numeric.append(col_name)

        train_dict["spectral_clinical_dispersion"] = sp_tr[:, k_embed]
        eval_dict["spectral_clinical_dispersion"] = sp_ev[:, k_embed]
        train_dict["spectral_max_divergence"] = sp_tr[:, k_embed + 1]
        eval_dict["spectral_max_divergence"] = sp_ev[:, k_embed + 1]
        extra_numeric.extend(["spectral_clinical_dispersion", "spectral_max_divergence"])

    # Composite Care Trajectory transitions
    adm_s_tr = np.array([str(x) if x is not None else "missing" for x in train.get("admission_source_id", ["missing"] * n_tr)], dtype=object)
    adm_s_ev = np.array([str(x) if x is not None else "missing" for x in eval_cols.get("admission_source_id", ["missing"] * n_ev)], dtype=object)
    disch_tr = np.array([str(x) if x is not None else "missing" for x in train.get("discharge_disposition_id", ["missing"] * n_tr)], dtype=object)
    disch_ev = np.array([str(x) if x is not None else "missing" for x in eval_cols.get("discharge_disposition_id", ["missing"] * n_ev)], dtype=object)
    adm_t_tr = np.array([str(x) if x is not None else "missing" for x in train.get("admission_type_id", ["missing"] * n_tr)], dtype=object)
    adm_t_ev = np.array([str(x) if x is not None else "missing" for x in eval_cols.get("admission_type_id", ["missing"] * n_ev)], dtype=object)

    train_dict["traj_adm_to_disch"] = np.array([f"{s}_{di}" for s, di in zip(adm_s_tr, disch_tr)], dtype=object)
    eval_dict["traj_adm_to_disch"] = np.array([f"{s}_{di}" for s, di in zip(adm_s_ev, disch_ev)], dtype=object)
    train_dict["traj_admtype_to_disch"] = np.array([f"{t}_{di}" for t, di in zip(adm_t_tr, disch_tr)], dtype=object)
    eval_dict["traj_admtype_to_disch"] = np.array([f"{t}_{di}" for t, di in zip(adm_t_ev, disch_ev)], dtype=object)
    extra_categorical.extend(["traj_adm_to_disch", "traj_admtype_to_disch"])

    # diag_1_disch
    d1_disch_tr = np.array([f"{d}_{g}" for d, g in zip(d1_3tr, disch_tr)], dtype=object)
    d1_disch_ev = np.array([f"{d}_{g}" for d, g in zip(d1_3ev, disch_ev)], dtype=object)
    train_dict["diag_1_disch"] = d1_disch_tr
    eval_dict["diag_1_disch"] = d1_disch_ev
    extra_categorical.append("diag_1_disch")

    # diag_1_age
    age_tr_str = np.array([str(x) if x is not None else "missing" for x in train.get("age", ["missing"]*n_tr)], dtype=object)
    age_ev_str = np.array([str(x) if x is not None else "missing" for x in eval_cols.get("age", ["missing"]*n_ev)], dtype=object)
    d1_age_tr = np.array([f"{d}_{a}" for d, a in zip(d1_3tr, age_tr_str)], dtype=object)
    d1_age_ev = np.array([f"{d}_{a}" for d, a in zip(d1_3ev, age_ev_str)], dtype=object)
    train_dict["diag_1_age"] = d1_age_tr
    eval_dict["diag_1_age"] = d1_age_ev
    extra_categorical.append("diag_1_age")

    # disch_age
    disch_age_tr = np.array([f"{d}_{a}" for d, a in zip(disch_tr, age_tr_str)], dtype=object)
    disch_age_ev = np.array([f"{d}_{a}" for d, a in zip(disch_ev, age_ev_str)], dtype=object)
    train_dict["disch_age"] = disch_age_tr
    eval_dict["disch_age"] = disch_age_ev
    extra_categorical.append("disch_age")

    # diag_1_2_3_group
    d123_tr = np.array([f"{g1}_{g2}_{g3}" for g1, g2, g3 in zip(d1_tr, d2_tr, d3_tr)], dtype=object)
    d123_ev = np.array([f"{g1}_{g2}_{g3}" for g1, g2, g3 in zip(d1_ev, d2_ev, d3_ev)], dtype=object)
    train_dict["diag_1_2_3_group"] = d123_tr
    eval_dict["diag_1_2_3_group"] = d123_ev
    extra_categorical.append("diag_1_2_3_group")

    # 3. Latent Clinical Phenotyping via Spectral Covariance Decomposition (Eigen-Phenotypes)
    def extract_phenotype_indicators(d, n_rows, cm_dict):
        indicators = []
        d1 = np.array([str(x)[:3] if x is not None and str(x) != "None" else "" for x in d["diag_1"]])
        d2 = np.array([str(x)[:3] if x is not None and str(x) != "None" else "" for x in d["diag_2"]])
        d3 = np.array([str(x)[:3] if x is not None and str(x) != "None" else "" for x in d["diag_3"]])

        top_icd_prefixes = [
            "250", "428", "414", "585", "496", "410", "486", "276",
            "599", "682", "401", "403", "707", "571", "434", "038", "518"
        ]
        for code in top_icd_prefixes:
            indicators.append(((d1 == code) | (d2 == code) | (d3 == code)).astype(np.float64))

        for med in ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone"]:
            if med in d:
                m_col = d[med]
                indicators.append((m_col != "No").astype(np.float64))
                indicators.append((m_col == "Up").astype(np.float64))
                indicators.append((m_col == "Down").astype(np.float64))

        disch = d.get("discharge_disposition_id")
        if disch is not None:
            indicators.append((disch == "1").astype(np.float64))
            indicators.append(np.isin(disch, ["3", "6", "22", "28"]).astype(np.float64))
            indicators.append((disch == "7").astype(np.float64))

        adm_s = d.get("admission_source_id")
        if adm_s is not None:
            indicators.append((adm_s == "7").astype(np.float64))
            indicators.append(np.isin(adm_s, ["1", "4", "6"]).astype(np.float64))

        inp = np.nan_to_num(d["number_inpatient"].astype(np.float64), nan=0.0)
        emg = np.nan_to_num(d["number_emergency"].astype(np.float64), nan=0.0)
        indicators.append((inp > 0.0).astype(np.float64))
        indicators.append((inp >= 2.0).astype(np.float64))
        indicators.append((emg > 0.0).astype(np.float64))

        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            indicators.append(((a1c == ">8") | (a1c == ">7")).astype(np.float64))
            indicators.append((a1c == "Norm").astype(np.float64))

        for k in sorted(cm_dict.keys()):
            indicators.append(cm_dict[k].astype(np.float64))

        return np.column_stack(indicators)

    B_tr = extract_phenotype_indicators(train, n_tr, cm_train)
    B_ev = extract_phenotype_indicators(eval_cols, n_ev, cm_eval)
    mu_B = np.mean(B_tr, axis=0)
    B_tr_c = B_tr - mu_B
    B_ev_c = B_ev - mu_B

    C_cov = (B_tr_c.T @ B_tr_c) / float(n_tr)
    eigvals, eigvecs = np.linalg.eigh(C_cov)
    top_idx = np.argsort(eigvals)[::-1][:8]
    V_top = eigvecs[:, top_idx]

    pheno_tr = B_tr_c @ V_top
    pheno_ev = B_ev_c @ V_top

    for i in range(8):
        col_name = f"latent_phenotype_{i+1}"
        train_dict[col_name] = pheno_tr[:, i].astype(np.float64)
        eval_dict[col_name] = pheno_ev[:, i].astype(np.float64)
        extra_numeric.append(col_name)

    # 5. Utilization & care intensity metrics
    def compute_utilization(d):
        los = np.maximum(np.nan_to_num(d["time_in_hospital"].astype(np.float64), nan=1.0), 1.0)
        inpatient = np.nan_to_num(d["number_inpatient"].astype(np.float64), nan=0.0)
        emergency = np.nan_to_num(d["number_emergency"].astype(np.float64), nan=0.0)
        outpatient = np.nan_to_num(d["number_outpatient"].astype(np.float64), nan=0.0)
        total_visits = inpatient + emergency + outpatient

        labs = np.nan_to_num(d["num_lab_procedures"].astype(np.float64), nan=0.0)
        procs = np.nan_to_num(d["num_procedures"].astype(np.float64), nan=0.0)
        meds = np.nan_to_num(d["num_medications"].astype(np.float64), nan=0.0)
        diags = np.nan_to_num(d["number_diagnoses"].astype(np.float64), nan=0.0)

        return {
            "total_prior_visits": total_visits,
            "has_inpatient": (inpatient > 0.0).astype(np.float64),
            "has_emergency": (emergency > 0.0).astype(np.float64),
            "frequent_inpatient": (inpatient >= 2.0).astype(np.float64),
            "inpatient_ratio": inpatient / (total_visits + 1.0),
            "emergency_ratio": emergency / (total_visits + 1.0),
            "acute_utilization_ratio": (inpatient + emergency) / (total_visits + 1.0),
            "prior_utilization_severity": (inpatient * 3.0 + emergency * 2.0 + outpatient * 1.0),
            "log_inpatient": np.log1p(inpatient),
            "log_emergency": np.log1p(emergency),
            "log_total_visits": np.log1p(total_visits),
            "labs_per_day": labs / los,
            "meds_per_day": meds / los,
            "procs_per_day": procs / los,
            "diags_per_day": diags / los,
            "clinical_intensity": (labs + procs + meds) / los,
            "med_proc_interaction": np.log1p(meds) * np.log1p(procs),
            "diag_los_interaction": diags * los,
            "diag_to_med_ratio": diags / (meds + 1.0),
            "inpatient_los_interaction": inpatient * los,
            "revolving_door_acute": ((inpatient > 0.0) & (emergency > 0.0)).astype(np.float64),
            "ed_to_outpatient_ratio": (emergency + 0.1) / (outpatient + 0.1),
            "prior_visit_velocity": (inpatient * 2.5 + emergency * 1.5 - outpatient * 0.5),
            "frequent_utilizer_flag": (total_visits >= 3.0).astype(np.float64),
            "inpatient_per_day": inpatient / los,
            "super_utilizer": ((inpatient >= 2.0) | (emergency >= 3.0)).astype(np.float64),
        }

    util_tr = compute_utilization(train)
    util_ev = compute_utilization(eval_cols)
    for k, v in util_tr.items():
        train_dict[k] = v
        eval_dict[k] = util_ev[k]
        extra_numeric.append(k)

    # 6. Admission and discharge disposition clinical flags
    def compute_adm_disch(d):
        disch = d.get("discharge_disposition_id")
        adm_s = d.get("admission_source_id")
        adm_t = d.get("admission_type_id")
        res = {}
        if disch is not None:
            res["is_disch_home"] = (disch == "1").astype(np.float64)
            res["is_disch_facility"] = np.isin(disch, ["3", "6", "22", "28"]).astype(np.float64)
            res["is_disch_ama"] = (disch == "7").astype(np.float64)
            res["is_disch_special"] = np.isin(disch, ["2", "4", "5", "8", "15", "23", "24"]).astype(np.float64)
        if adm_s is not None:
            res["is_adm_er"] = (adm_s == "7").astype(np.float64)
            res["is_adm_transfer"] = np.isin(adm_s, ["1", "4", "6"]).astype(np.float64)
        if adm_t is not None:
            res["is_adm_elective"] = (adm_t == "3").astype(np.float64)
            res["is_adm_emergency"] = (adm_t == "1").astype(np.float64)
            res["is_adm_urgent"] = (adm_t == "2").astype(np.float64)
        return res

    adm_tr = compute_adm_disch(train)
    adm_ev = compute_adm_disch(eval_cols)
    for k, v in adm_tr.items():
        train_dict[k] = v
        eval_dict[k] = adm_ev[k]
        extra_numeric.append(k)

    # 6. Age midpoint numeric feature
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    if "age" in train:
        age_tr_num = np.array([age_map.get(str(x), np.nan) for x in train["age"]], dtype=np.float64)
        age_ev_num = np.array([age_map.get(str(x), np.nan) for x in eval_cols["age"]], dtype=np.float64)
        train_dict["age_numeric"] = age_tr_num
        eval_dict["age_numeric"] = age_ev_num
        extra_numeric.append("age_numeric")

        train_dict["frail_elderly"] = (age_tr_num >= 75.0).astype(np.float64)
        eval_dict["frail_elderly"] = (age_ev_num >= 75.0).astype(np.float64)
        extra_numeric.append("frail_elderly")

        if "cm_count" in train_dict:
            train_dict["frail_multimorbidity"] = ((age_tr_num >= 75.0) & (train_dict["cm_count"] >= 2.0)).astype(np.float64)
            eval_dict["frail_multimorbidity"] = ((age_ev_num >= 75.0) & (eval_dict["cm_count"] >= 2.0)).astype(np.float64)
            extra_numeric.append("frail_multimorbidity")

    # 7. Medication regimen complexity, polypharmacy flux & contraindications
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    avail_meds = [m for m in med_names if m in train]

    # Medication Regimen Signature (Categorical)
    med_cols_to_signature = ["metformin", "glipizide", "glyburide", "pioglitazone", "insulin"]
    avail_sig_meds = [m for m in med_cols_to_signature if m in train]
    if avail_sig_meds:
        vals_tr = [train[m] for m in avail_sig_meds]
        vals_ev = [eval_cols[m] for m in avail_sig_meds]
        
        sig_tr = np.array([
            "|".join(f"{m}_{v}" for m, v in zip(avail_sig_meds, row) if v != "No" and v is not None)
            for row in zip(*vals_tr)
        ], dtype=object)
        sig_tr[sig_tr == ""] = "none"
        
        sig_ev = np.array([
            "|".join(f"{m}_{v}" for m, v in zip(avail_sig_meds, row) if v != "No" and v is not None)
            for row in zip(*vals_ev)
        ], dtype=object)
        sig_ev[sig_ev == ""] = "none"
        
        train_dict["med_regimen_signature"] = sig_tr
        eval_dict["med_regimen_signature"] = sig_ev
        extra_categorical.append("med_regimen_signature")

    def compute_meds(d, n_rows):
        act = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        dn_cnt = np.zeros(n_rows, dtype=np.float64)
        std_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in avail_meds:
            col = d[m]
            u = (col == "Up").astype(np.float64)
            dn = (col == "Down").astype(np.float64)
            s = (col == "Steady").astype(np.float64)
            up_cnt += u
            dn_cnt += dn
            std_cnt += s
            act += (u + dn + s)

        total_num_meds = np.nan_to_num(d.get("num_medications", act).astype(np.float64), nan=1.0)
        res = {
            "num_meds_active": act,
            "num_meds_up": up_cnt,
            "num_meds_down": dn_cnt,
            "num_meds_steady": std_cnt,
            "num_meds_changed": up_cnt + dn_cnt,
            "polypharmacy_active": (act >= 3.0).astype(np.float64),
            "diabetes_med_burden": act / (total_num_meds + 1.0),
        }

        if "insulin" in d:
            ins = d["insulin"]
            res["insulin_active"] = ((ins == "Up") | (ins == "Down") | (ins == "Steady")).astype(np.float64)
            res["insulin_up"] = (ins == "Up").astype(np.float64)
            res["insulin_down"] = (ins == "Down").astype(np.float64)
            res["insulin_steady"] = (ins == "Steady").astype(np.float64)
            res["insulin_changed"] = ((ins == "Up") | (ins == "Down")).astype(np.float64)
            res["insulin_oral_combination"] = (res["insulin_active"] * (act > 1.0)).astype(np.float64)
        return res

    meds_tr = compute_meds(train, n_tr)
    meds_ev = compute_meds(eval_cols, n_ev)
    for k, v in meds_tr.items():
        train_dict[k] = v
        eval_dict[k] = meds_ev[k]
        extra_numeric.append(k)

    if "metformin" in train and "cm_renal" in train_dict:
        train_dict["metformin_renal_hazard"] = ((train["metformin"] != "No") & (train_dict["cm_renal"] > 0)).astype(np.float64)
        eval_dict["metformin_renal_hazard"] = ((eval_cols["metformin"] != "No") & (eval_dict["cm_renal"] > 0)).astype(np.float64)
        extra_numeric.append("metformin_renal_hazard")

    # 8. Laboratory testing flags
    def compute_labs(d):
        res = {}
        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            res["a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
            res["a1c_high"] = ((a1c == ">8") | (a1c == ">7")).astype(np.float64)
        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            res["glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
            res["glu_high"] = ((glu == ">300") | (glu == ">200")).astype(np.float64)
        return res

    labs_tr = compute_labs(train)
    labs_ev = compute_labs(eval_cols)
    for k, v in labs_tr.items():
        train_dict[k] = v
        eval_dict[k] = labs_ev[k]
        extra_numeric.append(k)

    # Therapeutic friction and glycemic inertia
    a1c_tr = train.get("A1Cresult", np.array(["None"] * n_tr, dtype=object))
    a1c_ev = eval_cols.get("A1Cresult", np.array(["None"] * n_ev, dtype=object))
    glu_tr = train.get("max_glu_serum", np.array(["None"] * n_tr, dtype=object))
    glu_ev = eval_cols.get("max_glu_serum", np.array(["None"] * n_ev, dtype=object))
    chg_tr = train.get("change", np.array(["No"] * n_tr, dtype=object))
    chg_ev = eval_cols.get("change", np.array(["No"] * n_ev, dtype=object))

    hg_tr = ((a1c_tr == ">8") | (a1c_tr == ">7") | (glu_tr == ">300") | (glu_tr == ">200")).astype(np.float64)
    hg_ev = ((a1c_ev == ">8") | (a1c_ev == ">7") | (glu_ev == ">300") | (glu_ev == ">200")).astype(np.float64)
    train_dict["high_glycemia_marker"] = hg_tr
    eval_dict["high_glycemia_marker"] = hg_ev
    extra_numeric.append("high_glycemia_marker")

    train_dict["therapeutic_inertia_risk"] = (hg_tr * (chg_tr == "No")).astype(np.float64)
    eval_dict["therapeutic_inertia_risk"] = (hg_ev * (chg_ev == "No")).astype(np.float64)
    extra_numeric.append("therapeutic_inertia_risk")

    # Metformin / Insulin glycemic undertreatment indicators
    insulin_tr = train.get("insulin", np.array(["No"] * n_tr, dtype=object))
    insulin_ev = eval_cols.get("insulin", np.array(["No"] * n_ev, dtype=object))
    train_dict["insulin_undertreatment"] = (hg_tr * (insulin_tr == "No")).astype(np.float64)
    eval_dict["insulin_undertreatment"] = (hg_ev * (insulin_ev == "No")).astype(np.float64)
    extra_numeric.append("insulin_undertreatment")

    # 7b. Advanced Clinical Risk Indices (CCI, ACCI, LACE, HOSPITAL, MRCI)
    def compute_advanced_indices(d_dict, d_raw, cm_dict):
        cci_score = np.zeros_like(d_raw["time_in_hospital"], dtype=np.float64)
        weights = {
            "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
            "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
            "cm_liver": 1.0, "cm_hemiplegia": 2.0, "cm_renal": 2.0, "cm_cancer": 2.0,
            "cm_diabetes_complicated": 2.0, "cm_metastatic": 6.0, "cm_hiv": 6.0,
        }
        for k, w in weights.items():
            if k in cm_dict:
                cci_score += cm_dict[k].astype(np.float64) * w

        age_num = d_dict.get("age_numeric", np.zeros_like(cci_score))
        age_pts = np.zeros_like(age_num)
        age_pts[(age_num >= 50.0) & (age_num < 60.0)] = 1.0
        age_pts[(age_num >= 60.0) & (age_num < 70.0)] = 2.0
        age_pts[(age_num >= 70.0) & (age_num < 80.0)] = 3.0
        age_pts[age_num >= 80.0] = 4.0
        acci_score = cci_score + age_pts

        los = np.maximum(np.nan_to_num(d_raw["time_in_hospital"].astype(np.float64), nan=1.0), 1.0)
        is_acute = d_dict.get("is_adm_emergency", np.zeros_like(cci_score)) + d_dict.get("is_adm_urgent", np.zeros_like(cci_score))
        num_emg = np.nan_to_num(d_raw["number_emergency"].astype(np.float64), nan=0.0)

        l_pts = np.zeros_like(los)
        l_pts[los == 1.0] = 1.0
        l_pts[los == 2.0] = 2.0
        l_pts[los == 3.0] = 3.0
        l_pts[(los >= 4.0) & (los <= 6.0)] = 4.0
        l_pts[(los >= 7.0) & (los <= 13.0)] = 5.0
        l_pts[los >= 14.0] = 7.0

        a_pts = np.where(is_acute > 0.0, 3.0, 0.0)

        c_pts = np.zeros_like(cci_score)
        c_pts[cci_score == 1.0] = 1.0
        c_pts[cci_score == 2.0] = 2.0
        c_pts[cci_score == 3.0] = 3.0
        c_pts[cci_score >= 4.0] = 5.0

        e_pts = np.zeros_like(num_emg)
        e_pts[num_emg == 1.0] = 1.0
        e_pts[num_emg == 2.0] = 2.0
        e_pts[num_emg == 3.0] = 3.0
        e_pts[num_emg >= 4.0] = 4.0

        lace_score = l_pts + a_pts + c_pts + e_pts

        cm_cancer = cm_dict.get("cm_cancer", np.zeros_like(cci_score))
        cm_metastatic = cm_dict.get("cm_metastatic", np.zeros_like(cci_score))
        num_inp = np.nan_to_num(d_raw["number_inpatient"].astype(np.float64), nan=0.0)

        hospital_score = np.zeros_like(los)
        has_onco = ((cm_cancer > 0.0) | (cm_metastatic > 0.0)).astype(np.float64)
        hospital_score += has_onco * 2.0
        hospital_score += (is_acute > 0.0).astype(np.float64) * 1.0
        hospital_score += (los >= 5.0).astype(np.float64) * 1.0
        hospital_score += np.where((num_inp >= 2.0) & (num_inp <= 4.0), 2.0, 0.0)
        hospital_score += np.where(num_inp >= 5.0, 5.0, 0.0)

        act = d_dict.get("num_meds_active", np.zeros_like(cci_score))
        insulin_active = d_dict.get("insulin_active", np.zeros_like(cci_score))
        num_meds_changed = d_dict.get("num_meds_changed", np.zeros_like(cci_score))
        num_meds_up = d_dict.get("num_meds_up", np.zeros_like(cci_score))
        num_meds_down = d_dict.get("num_meds_down", np.zeros_like(cci_score))

        mrci_score = act + insulin_active * 2.0 + num_meds_changed * 1.5 + (num_meds_up + num_meds_down) * 1.0

        return {
            "charlson_comorbidity_index": cci_score,
            "age_adjusted_charlson_index": acci_score,
            "lace_readmission_index": lace_score,
            "hospital_readmission_score": hospital_score,
            "medication_complexity_index": mrci_score,
        }

    indices_tr = compute_advanced_indices(train_dict, train, cm_train)
    indices_ev = compute_advanced_indices(eval_dict, eval_cols, cm_eval)
    for k, v in indices_tr.items():
        train_dict[k] = v
        eval_dict[k] = indices_ev[k]
        extra_numeric.append(k)

    # 4. Smoothed Out-Of-Fold Target Encoding (moved here so all created features are accessible)
    te_target_cols = [
        "diag_1", "diag_2", "diag_3",
        "diag_1_3digit", "diag_2_3digit",
        "discharge_disposition_id", "admission_source_id", "admission_type_id",
        "medical_specialty", "payer_code", "diag_1_group", "diag_1_2_group",
        "age",
        "traj_adm_to_disch", "traj_admtype_to_disch",
        "diag_1_disch", "diag_1_age", "disch_age", "diag_1_2_3_group",
        "med_regimen_signature",
    ]
    for col in te_target_cols:
        if col in train_dict and col in eval_dict:
            tr_enc, ev_enc = oof_target_encode(
                train_dict[col], y_train, eval_dict[col], n_folds=5, smoothing=20.0, seed=42
            )
            te_name = f"te_{col}"
            train_dict[te_name] = tr_enc.astype(np.float64)
            eval_dict[te_name] = ev_enc.astype(np.float64)
            extra_numeric.append(te_name)

    # 3b. 3-Way Multimorbidity Co-occurrence Tensor Decomposition via CP-ALS Optimization
    code_counts_3d = {}
    for c1, c2, c3 in zip(d1_3tr, d2_3tr, d3_3tr):
        for c in (c1, c2, c3):
            if c != "missing" and c != "":
                code_counts_3d[c] = code_counts_3d.get(c, 0) + 1

    top_codes_3d = [c for c, cnt in sorted(code_counts_3d.items(), key=lambda x: x[1], reverse=True)[:26] if cnt >= 15]
    V3 = len(top_codes_3d)
    if V3 >= 12:
        vocab3_map = {c: i for i, c in enumerate(top_codes_3d)}
        T_mat = np.zeros((V3, V3, V3), dtype=np.float64)
        for c1, c2, c3 in zip(d1_3tr, d2_3tr, d3_3tr):
            idxs = [vocab3_map[c] for c in (c1, c2, c3) if c in vocab3_map]
            idxs = list(set(idxs))
            k_idx = len(idxs)
            if k_idx == 3:
                i0, i1, i2 = idxs
                for a, b, c in [(i0, i1, i2), (i0, i2, i1), (i1, i0, i2), (i1, i2, i0), (i2, i0, i1), (i2, i1, i0)]:
                    T_mat[a, b, c] += 1.0
            elif k_idx == 2:
                i0, i1 = idxs
                for a, b in [(i0, i1), (i1, i0)]:
                    T_mat[a, b, a] += 0.5
                    T_mat[a, b, b] += 0.5
            elif k_idx == 1:
                i0 = idxs[0]
                T_mat[i0, i0, i0] += 1.0

        total_triples = float(np.sum(T_mat)) + 1e-9
        P_triad = (T_mat + 1.0 / (V3 ** 3)) / (total_triples + 1.0)
        p_marg = np.sum(P_triad, axis=(1, 2))
        expected_3d = np.einsum('i,j,k->ijk', p_marg, p_marg, p_marg)
        ppmi_3d = np.maximum(0.0, np.log(P_triad / (expected_3d + 1e-12)))

        # Solve symmetric CP-ALS rank-R tensor decomposition
        R_cp = 6
        unfold_1 = ppmi_3d.reshape(V3, V3 * V3)
        u_init, _, _ = np.linalg.svd(unfold_1, full_matrices=False)
        A_cp = u_init[:, :R_cp].copy()
        reg_cp = 1e-3

        for _ in range(10):
            # Khatri-Rao product of A_cp and A_cp: shape (V3*V3, R_cp)
            K_aa = np.zeros((V3 * V3, R_cp), dtype=np.float64)
            for r in range(R_cp):
                K_aa[:, r] = np.outer(A_cp[:, r], A_cp[:, r]).ravel()
            gamma = (A_cp.T @ A_cp) * (A_cp.T @ A_cp)
            A_next = unfold_1 @ K_aa @ np.linalg.inv(gamma + reg_cp * np.eye(R_cp))
            col_norms = np.linalg.norm(A_next, axis=0) + 1e-9
            A_cp = A_next / col_norms

        lambda_cp = col_norms

        def extract_cp_tensor_feats(d1_arr, d2_arr, d3_arr, n_rows):
            feats = np.zeros((n_rows, R_cp + 2), dtype=np.float64)
            for r in range(n_rows):
                idxs = [vocab3_map[c] for c in (d1_arr[r], d2_arr[r], d3_arr[r]) if c in vocab3_map]
                if not idxs:
                    continue
                emb_stack = A_cp[idxs, :]
                feats[r, :R_cp] = np.mean(emb_stack, axis=0)
                if len(idxs) == 3:
                    feats[r, R_cp] = float(np.sum(lambda_cp * A_cp[idxs[0], :] * A_cp[idxs[1], :] * A_cp[idxs[2], :]))
                    feats[r, R_cp + 1] = float(np.mean([
                        np.dot(A_cp[idxs[0], :], A_cp[idxs[1], :]),
                        np.dot(A_cp[idxs[0], :], A_cp[idxs[2], :]),
                        np.dot(A_cp[idxs[1], :], A_cp[idxs[2], :]),
                    ]))
                elif len(idxs) == 2:
                    feats[r, R_cp] = float(np.sum(lambda_cp * A_cp[idxs[0], :] * A_cp[idxs[1], :] * feats[r, :R_cp]))
                    feats[r, R_cp + 1] = float(np.dot(A_cp[idxs[0], :], A_cp[idxs[1], :]))
            return feats

        cp_tr = extract_cp_tensor_feats(d1_3tr, d2_3tr, d3_3tr, n_tr)
        cp_ev = extract_cp_tensor_feats(d1_3ev, d2_3ev, d3_3ev, n_ev)

        for r in range(R_cp):
            col_name = f"cp_tensor_latent_{r+1}"
            train_dict[col_name] = cp_tr[:, r]
            eval_dict[col_name] = cp_ev[:, r]
            extra_numeric.append(col_name)

        train_dict["cp_tensor_synergy"] = cp_tr[:, R_cp]
        eval_dict["cp_tensor_synergy"] = cp_ev[:, R_cp]
        train_dict["cp_tensor_coherence"] = cp_tr[:, R_cp + 1]
        eval_dict["cp_tensor_coherence"] = cp_ev[:, R_cp + 1]
        extra_numeric.extend(["cp_tensor_synergy", "cp_tensor_coherence"])

    # 3c. Supervised Tucker / HOSVD Clinical Risk Tensor (Pathology x Regimen x Pathway)
    def map_pathology_mode(diag_arr):
        res = np.zeros(len(diag_arr), dtype=np.int32)
        for idx, d in enumerate(diag_arr):
            s = str(d).strip() if d is not None and str(d) != "None" else ""
            if not s or s == "missing":
                res[idx] = 15
                continue
            p3 = s[:3]
            if p3 == "250":
                res[idx] = 0
            elif p3 == "428":
                res[idx] = 1
            elif p3 in ("410", "411", "412", "413", "414"):
                res[idx] = 2
            elif p3.isdigit() and 390 <= int(p3) <= 459:
                res[idx] = 3
            elif p3 in ("585", "586", "580", "581", "582", "583", "584", "587", "588", "589"):
                res[idx] = 4
            elif p3.isdigit() and 460 <= int(p3) <= 519:
                res[idx] = 5
            elif p3.isdigit() and 520 <= int(p3) <= 579:
                res[idx] = 6
            elif p3.isdigit() and 240 <= int(p3) <= 279:
                res[idx] = 7
            elif p3.isdigit() and 590 <= int(p3) <= 629:
                res[idx] = 8
            elif p3.isdigit() and 710 <= int(p3) <= 739:
                res[idx] = 9
            elif p3.isdigit() and 680 <= int(p3) <= 709:
                res[idx] = 10
            elif (p3.isdigit() and 140 <= int(p3) <= 239) or s.startswith("V58"):
                res[idx] = 11
            elif p3.isdigit() and 430 <= int(p3) <= 438:
                res[idx] = 12
            elif p3.isdigit() and 800 <= int(p3) <= 999:
                res[idx] = 13
            elif (p3.isdigit() and 1 <= int(p3) <= 139) or p3 == "038":
                res[idx] = 14
            else:
                res[idx] = 15
        return res

    def map_regimen_mode(data_dict, n_rows):
        res = np.zeros(n_rows, dtype=np.int32)
        ins = data_dict.get("insulin", np.array(["No"] * n_rows, dtype=object))
        met = data_dict.get("metformin", np.array(["No"] * n_rows, dtype=object))
        dm = data_dict.get("diabetesMed", np.array(["No"] * n_rows, dtype=object))
        for idx in range(n_rows):
            if dm[idx] == "No":
                res[idx] = 0
                continue
            i_val = str(ins[idx])
            m_val = str(met[idx])
            if i_val == "Down":
                res[idx] = 1
            elif i_val == "Up":
                res[idx] = 2
            elif i_val == "Steady":
                res[idx] = 3
            elif i_val != "No" and m_val != "No":
                res[idx] = 4
            elif m_val != "No":
                res[idx] = 5
            elif i_val != "No":
                res[idx] = 6
            else:
                res[idx] = 7
        return res

    def map_pathway_mode(data_dict, n_rows):
        res = np.zeros(n_rows, dtype=np.int32)
        adm_s = data_dict.get("admission_source_id", np.array(["1"] * n_rows, dtype=object))
        disch = data_dict.get("discharge_disposition_id", np.array(["1"] * n_rows, dtype=object))
        for idx in range(n_rows):
            d_val = str(disch[idx])
            s_val = str(adm_s[idx])
            is_er = (s_val == "7")
            if d_val == "1":
                res[idx] = 0 if is_er else 1
            elif d_val in ("3", "22", "28"):
                res[idx] = 2 if is_er else 3
            elif d_val == "6":
                res[idx] = 4
            elif d_val == "7":
                res[idx] = 5
            elif d_val in ("2", "4", "5"):
                res[idx] = 6
            elif d_val in ("8", "23", "24"):
                res[idx] = 7
            else:
                res[idx] = 8 if is_er else 9
        return res

    mode1_tr = map_pathology_mode(train["diag_1"])
    mode1_ev = map_pathology_mode(eval_cols["diag_1"])
    mode2_tr = map_regimen_mode(train, n_tr)
    mode2_ev = map_regimen_mode(eval_cols, n_ev)
    mode3_tr = map_pathway_mode(train, n_tr)
    mode3_ev = map_pathway_mode(eval_cols, n_ev)

    I_dim, J_dim, K_dim = 16, 8, 10
    r1, r2, r3 = 4, 3, 3

    def decompose_risk_tensor(m1, m2, m3, y, smooth=15.0):
        N_ijk = np.zeros((I_dim, J_dim, K_dim), dtype=np.float64)
        S_ijk = np.zeros((I_dim, J_dim, K_dim), dtype=np.float64)
        for i, j, k, val in zip(m1, m2, m3, y):
            N_ijk[i, j, k] += 1.0
            if val > 0:
                S_ijk[i, j, k] += 1.0

        p_global = float(np.sum(S_ijk)) / (float(np.sum(N_ijk)) + 1e-9)
        p_smooth = (S_ijk + smooth * p_global) / (N_ijk + smooth)
        p_smooth = np.clip(p_smooth, 1e-4, 1.0 - 1e-4)
        R_tensor = np.log(p_smooth / (1.0 - p_smooth)) - np.log(p_global / (1.0 - p_global))

        # Modal SVD unfoldings (HOSVD)
        R_1 = R_tensor.reshape(I_dim, J_dim * K_dim)
        R_2 = np.moveaxis(R_tensor, 1, 0).reshape(J_dim, I_dim * K_dim)
        R_3 = np.moveaxis(R_tensor, 2, 0).reshape(K_dim, I_dim * J_dim)

        u1, _, _ = np.linalg.svd(R_1, full_matrices=False)
        u2, _, _ = np.linalg.svd(R_2, full_matrices=False)
        u3, _, _ = np.linalg.svd(R_3, full_matrices=False)

        U1 = u1[:, :r1]
        U2 = u2[:, :r2]
        U3 = u3[:, :r3]

        # Compute Core Tensor G and Low-Rank Reconstructed Tensor R_hat
        G = np.einsum('ijk,ia,jb,kc->abc', R_tensor, U1, U2, U3)
        R_hat = np.einsum('abc,ia,jb,kc->ijk', G, U1, U2, U3)
        return U1, U2, U3, R_tensor, R_hat

    # 5-Fold Cross-Validation for Out-Of-Fold Tensor Representations
    from sklearn.model_selection import KFold
    oof_tensor_risk = np.zeros(n_tr, dtype=np.float64)
    oof_tensor_u1 = np.zeros((n_tr, r1), dtype=np.float64)
    oof_tensor_u2 = np.zeros((n_tr, r2), dtype=np.float64)
    oof_tensor_u3 = np.zeros((n_tr, r3), dtype=np.float64)
    oof_tensor_energy = np.zeros(n_tr, dtype=np.float64)

    kf_ten = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf_ten.split(y_train):
        U1_f, U2_f, U3_f, _, R_hat_f = decompose_risk_tensor(
            mode1_tr[tr_idx], mode2_tr[tr_idx], mode3_tr[tr_idx], y_train[tr_idx]
        )
        for v_i in val_idx:
            i, j, k = mode1_tr[v_i], mode2_tr[v_i], mode3_tr[v_i]
            oof_tensor_risk[v_i] = R_hat_f[i, j, k]
            oof_tensor_u1[v_i, :] = U1_f[i, :]
            oof_tensor_u2[v_i, :] = U2_f[j, :]
            oof_tensor_u3[v_i, :] = U3_f[k, :]
            oof_tensor_energy[v_i] = float(np.sum(U1_f[i, :3] * U2_f[j, :3] * U3_f[k, :3]))

    # Decompose on full train for evaluation set
    U1_full, U2_full, U3_full, _, R_hat_full = decompose_risk_tensor(
        mode1_tr, mode2_tr, mode3_tr, y_train
    )
    ev_tensor_risk = np.zeros(n_ev, dtype=np.float64)
    ev_tensor_u1 = np.zeros((n_ev, r1), dtype=np.float64)
    ev_tensor_u2 = np.zeros((n_ev, r2), dtype=np.float64)
    ev_tensor_u3 = np.zeros((n_ev, r3), dtype=np.float64)
    ev_tensor_energy = np.zeros(n_ev, dtype=np.float64)

    for r in range(n_ev):
        i, j, k = mode1_ev[r], mode2_ev[r], mode3_ev[r]
        ev_tensor_risk[r] = R_hat_full[i, j, k]
        ev_tensor_u1[r, :] = U1_full[i, :]
        ev_tensor_u2[r, :] = U2_full[j, :]
        ev_tensor_u3[r, :] = U3_full[k, :]
        ev_tensor_energy[r] = float(np.sum(U1_full[i, :3] * U2_full[j, :3] * U3_full[k, :3]))

    train_dict["tensor_hosvd_risk"] = oof_tensor_risk
    eval_dict["tensor_hosvd_risk"] = ev_tensor_risk
    train_dict["tensor_interaction_energy"] = oof_tensor_energy
    eval_dict["tensor_interaction_energy"] = ev_tensor_energy
    extra_numeric.extend(["tensor_hosvd_risk", "tensor_interaction_energy"])

    for a in range(r1):
        col_name = f"tensor_pathology_u1_{a+1}"
        train_dict[col_name] = oof_tensor_u1[:, a]
        eval_dict[col_name] = ev_tensor_u1[:, a]
        extra_numeric.append(col_name)

    for b in range(r2):
        col_name = f"tensor_regimen_u2_{b+1}"
        train_dict[col_name] = oof_tensor_u2[:, b]
        eval_dict[col_name] = ev_tensor_u2[:, b]
        extra_numeric.append(col_name)

    for c in range(r3):
        col_name = f"tensor_pathway_u3_{c+1}"
        train_dict[col_name] = oof_tensor_u3[:, c]
        eval_dict[col_name] = ev_tensor_u3[:, c]
        extra_numeric.append(col_name)

    # Encode tabular data
    encoder = TabularEncoder(min_count=25)
    encoder.fit(
        task,
        train_dict,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # 9. Cross-Architecture Heterogeneous Ensemble (LightGBM + XGBoost) with Logit Blending
    import xgboost as xgb

    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=140,
            subsample=0.72,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.2,
            cat_smooth=60.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.018,
            num_leaves=45,
            max_depth=7,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=1.5,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.024,
            num_leaves=18,
            max_depth=5,
            min_child_samples=200,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=0.1,
            reg_alpha=0.05,
            cat_smooth=65.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.015,
            num_leaves=63,
            max_depth=8,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=40.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.015,
            num_leaves=8,
            max_depth=3,
            min_child_samples=150,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=5.0,
            reg_alpha=1.0,
            cat_smooth=80.0,
            random_state=888,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=210,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.60,
            min_child_weight=12,
            reg_alpha=0.2,
            reg_lambda=2.0,
            eval_metric="logloss",
            random_state=789,
            n_jobs=1,
            verbosity=0,
        ),
        xgb.XGBClassifier(
            n_estimators=190,
            learning_rate=0.030,
            max_depth=4,
            subsample=0.80,
            colsample_bytree=0.50,
            min_child_weight=15,
            reg_alpha=0.5,
            reg_lambda=3.0,
            eval_metric="logloss",
            random_state=101112,
            n_jobs=1,
            verbosity=0,
        ),
        xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.70,
            colsample_bytree=0.60,
            min_child_weight=20,
            reg_alpha=1.0,
            reg_lambda=5.0,
            eval_metric="logloss",
            random_state=111,
            n_jobs=1,
            verbosity=0,
        ),
    ]

    models_with_weights = [
        (lgb_models[0], 1.2, True),
        (lgb_models[1], 1.2, True),
        (lgb_models[2], 1.0, True),
        (lgb_models[3], 1.0, True),
        (lgb_models[4], 0.9, True),
        (xgb_models[0], 0.9, False),
        (xgb_models[1], 0.9, False),
        (xgb_models[2], 0.8, False),
    ]

    final_logits = np.zeros(n_ev, dtype=np.float64)
    total_weight = 0.0

    for model, weight, is_lgb in models_with_weights:
        if is_lgb:
            model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        else:
            model.fit(x_train, y_train)
        p = model.predict_proba(x_eval)[:, 1]
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        z = np.log(p / (1.0 - p))
        final_logits += weight * z
        total_weight += weight

    combined_z = final_logits / total_weight
    final_preds = 1.0 / (1.0 + np.exp(-combined_z))
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
