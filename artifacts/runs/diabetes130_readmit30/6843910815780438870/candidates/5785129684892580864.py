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
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` using Thermodynamic Fragility, Attractor Dynamics & 4-Fold Meta-Ensembling."""
    icd9_comb = globals().get("icd9_comorbidities")
    icd9_grp = globals().get("icd9_group")
    oof_te = globals().get("oof_target_encode")

    if icd9_comb is None or icd9_grp is None or oof_te is None:
        try:
            from clinical_toolkit import (
                icd9_comorbidities as _icd9_comb,
                icd9_group as _icd9_grp,
                oof_target_encode as _oof_te,
            )
            icd9_comb = icd9_comb or _icd9_comb
            icd9_grp = icd9_grp or _icd9_grp
            oof_te = oof_te or _oof_te
        except Exception:
            pass

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities & Weighted Charlson Severity Index
    if icd9_comb is not None and "diag_1" in train and "diag_2" in train and "diag_3" in train:
        cm_train = icd9_comb(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_eval = icd9_comb(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

        for k, arr in cm_train.items():
            train_data[k] = arr.astype(np.float64)
            eval_data[k] = cm_eval[k].astype(np.float64)
            extra_numeric.append(k)

        weights = {
            "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
            "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0,
            "cm_peptic_ulcer": 1.0, "cm_liver": 1.0, "cm_hemiplegia": 2.0,
            "cm_renal": 2.0, "cm_cancer": 2.0, "cm_diabetes_complicated": 2.0,
            "cm_metastatic": 4.0, "cm_hiv": 4.0,
        }
        charlson_tr = np.zeros(len(y_train), dtype=np.float64)
        charlson_ev = np.zeros(len(eval_cols["time_in_hospital"]), dtype=np.float64)
        for k, w in weights.items():
            if k in cm_train:
                charlson_tr += w * cm_train[k]
                charlson_ev += w * cm_eval[k]
        train_data["charlson_score"] = charlson_tr
        eval_data["charlson_score"] = charlson_ev
        extra_numeric.append("charlson_score")

    # 2. ICD-9 Chapter Mapping & Multi-Organ Diversity
    if icd9_grp is not None:
        for diag in ["diag_1", "diag_2", "diag_3"]:
            if diag in train and diag in eval_cols:
                grp_col = f"{diag}_group"
                train_data[grp_col] = np.array([icd9_grp(c) for c in train[diag]], dtype=object)
                eval_data[grp_col] = np.array([icd9_grp(c) for c in eval_cols[diag]], dtype=object)
                extra_categorical.append(grp_col)

        if "diag_1_group" in train_data and "diag_2_group" in train_data and "diag_3_group" in train_data:
            g1_tr, g2_tr, g3_tr = train_data["diag_1_group"], train_data["diag_2_group"], train_data["diag_3_group"]
            g1_ev, g2_ev, g3_ev = eval_data["diag_1_group"], eval_data["diag_2_group"], eval_data["diag_3_group"]

            organ_div_tr = np.array([len({str(a), str(b), str(c)} - {"missing", "other", "None"}) for a, b, c in zip(g1_tr, g2_tr, g3_tr)], dtype=np.float64)
            organ_div_ev = np.array([len({str(a), str(b), str(c)} - {"missing", "other", "None"}) for a, b, c in zip(g1_ev, g2_ev, g3_ev)], dtype=np.float64)
            train_data["organ_diversity"] = organ_div_tr
            eval_data["organ_diversity"] = organ_div_ev
            extra_numeric.append("organ_diversity")

            if "charlson_score" in train_data:
                train_data["systemic_fragility"] = (organ_div_tr * (train_data["charlson_score"] + 1.0)).astype(np.float64)
                eval_data["systemic_fragility"] = (organ_div_ev * (eval_data["charlson_score"] + 1.0)).astype(np.float64)
                extra_numeric.append("systemic_fragility")

    # 3. Relational Dyads (Transition-of-Care & Clinical Context Interactions)
    if "admission_source_id" in train and "discharge_disposition_id" in train:
        ct_tr = np.array([f"{s}_{d}" for s, d in zip(train["admission_source_id"], train["discharge_disposition_id"])], dtype=object)
        ct_ev = np.array([f"{s}_{d}" for s, d in zip(eval_cols["admission_source_id"], eval_cols["discharge_disposition_id"])], dtype=object)
        train_data["care_transition"] = ct_tr
        eval_data["care_transition"] = ct_ev
        extra_categorical.append("care_transition")
        if oof_te is not None:
            ct_tr_te, ct_ev_te = oof_te(ct_tr, y_train, ct_ev, n_folds=5, smoothing=25.0, seed=77)
            train_data["care_transition_te"] = ct_tr_te.astype(np.float64)
            eval_data["care_transition_te"] = ct_ev_te.astype(np.float64)
            extra_numeric.append("care_transition_te")

    if "admission_type_id" in train and "discharge_disposition_id" in train:
        ad_tr = np.array([f"{a}_{d}" for a, d in zip(train["admission_type_id"], train["discharge_disposition_id"])], dtype=object)
        ad_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["admission_type_id"], eval_cols["discharge_disposition_id"])], dtype=object)
        train_data["adm_disp_dyad"] = ad_tr
        eval_data["adm_disp_dyad"] = ad_ev
        extra_categorical.append("adm_disp_dyad")
        if oof_te is not None:
            ad_tr_te, ad_ev_te = oof_te(ad_tr, y_train, ad_ev, n_folds=5, smoothing=25.0, seed=88)
            train_data["adm_disp_dyad_te"] = ad_tr_te.astype(np.float64)
            eval_data["adm_disp_dyad_te"] = ad_ev_te.astype(np.float64)
            extra_numeric.append("adm_disp_dyad_te")

    if "diag_1_group" in train_data and "discharge_disposition_id" in train:
        d1d_tr = np.array([f"{g}_{d}" for g, d in zip(train_data["diag_1_group"], train["discharge_disposition_id"])], dtype=object)
        d1d_ev = np.array([f"{g}_{d}" for g, d in zip(eval_data["diag_1_group"], eval_cols["discharge_disposition_id"])], dtype=object)
        train_data["d1_disp_dyad"] = d1d_tr
        eval_data["d1_disp_dyad"] = d1d_ev
        extra_categorical.append("d1_disp_dyad")
        if oof_te is not None:
            d1d_tr_te, d1d_ev_te = oof_te(d1d_tr, y_train, d1d_ev, n_folds=5, smoothing=25.0, seed=99)
            train_data["d1_disp_dyad_te"] = d1d_tr_te.astype(np.float64)
            eval_data["d1_disp_dyad_te"] = d1d_ev_te.astype(np.float64)
            extra_numeric.append("d1_disp_dyad_te")

    # 4. Out-Of-Fold Bayesian Target Encoding for Primary High-Cardinality Fields
    if oof_te is not None:
        te_targets = [
            "diag_1", "diag_2", "diag_3",
            "discharge_disposition_id", "admission_source_id",
            "medical_specialty", "admission_type_id"
        ]
        te_train_map = {}
        te_eval_map = {}
        for col in te_targets:
            if col in train and col in eval_cols:
                tr_enc, ev_enc = oof_te(train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42)
                col_name = f"{col}_te"
                train_data[col_name] = tr_enc.astype(np.float64)
                eval_data[col_name] = ev_enc.astype(np.float64)
                extra_numeric.append(col_name)
                te_train_map[col] = tr_enc
                te_eval_map[col] = ev_enc

        if "diag_1" in te_train_map and "diag_2" in te_train_map and "diag_3" in te_train_map:
            train_data["diag_te_max"] = np.maximum.reduce([te_train_map["diag_1"], te_train_map["diag_2"], te_train_map["diag_3"]]).astype(np.float64)
            eval_data["diag_te_max"] = np.maximum.reduce([te_eval_map["diag_1"], te_eval_map["diag_2"], te_eval_map["diag_3"]]).astype(np.float64)
            extra_numeric.append("diag_te_max")

            train_data["diag_te_mean"] = ((te_train_map["diag_1"] + te_train_map["diag_2"] + te_train_map["diag_3"]) / 3.0).astype(np.float64)
            eval_data["diag_te_mean"] = ((te_eval_map["diag_1"] + te_eval_map["diag_2"] + te_eval_map["diag_3"]) / 3.0).astype(np.float64)
            extra_numeric.append("diag_te_mean")

    # 5. Thermodynamic Fragility & Clinical Phase-Space Dynamics
    def extract_features(src):
        num_inp = np.nan_to_num(src["number_inpatient"].astype(np.float64), nan=0.0)
        num_emg = np.nan_to_num(src["number_emergency"].astype(np.float64), nan=0.0)
        num_out = np.nan_to_num(src["number_outpatient"].astype(np.float64), nan=0.0)
        time_hosp = np.nan_to_num(src["time_in_hospital"].astype(np.float64), nan=1.0)
        time_safe = np.maximum(time_hosp, 0.5)
        num_meds = np.nan_to_num(src["num_medications"].astype(np.float64), nan=0.0)
        num_labs = np.nan_to_num(src["num_lab_procedures"].astype(np.float64), nan=0.0)
        num_procs = np.nan_to_num(src["num_procedures"].astype(np.float64), nan=0.0)
        num_diag = np.nan_to_num(src["number_diagnoses"].astype(np.float64), nan=0.0)

        tot_visits = num_inp + num_emg + num_out
        feats = {
            "total_prior_visits": tot_visits,
            "has_prior_inpatient": (num_inp > 0).astype(np.float64),
            "has_prior_emergency": (num_emg > 0).astype(np.float64),
            "has_prior_visits": (tot_visits > 0).astype(np.float64),
            "frequent_inpatient": (num_inp >= 2).astype(np.float64),
            "inpatient_ratio": num_inp / (tot_visits + 1.0),
            "emergency_ratio": num_emg / (tot_visits + 1.0),
            "inpatient_per_day": num_inp / time_safe,
            "labs_per_day": num_labs / time_safe,
            "meds_per_day": num_meds / time_safe,
            "procs_per_day": num_procs / time_safe,
            "diag_per_day": num_diag / time_safe,
            "service_intensity": (num_labs + num_procs + num_meds) / time_safe,
            "meds_x_time": num_meds * time_hosp,
            "diag_x_meds": num_diag * num_meds,
            "clinical_velocity": (1.5 * num_emg + 2.0 * num_inp) / time_safe,
            "clinical_momentum": ((1.5 * num_emg + 2.0 * num_inp) / time_safe) * num_diag,
            "gravity_premature_ejection": (num_inp * num_diag) / (time_safe ** 2),
            "acute_to_chronic_ratio": (num_emg + num_inp) / (num_out + 1.0),
            "bed_intensity_strain": (time_hosp * (num_labs + num_meds)) / (num_procs + 1.0),
        }

        if "age" in src:
            age_map = {
                "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
                "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
                "[80-90)": 85.0, "[90-100)": 95.0,
            }
            feats["age_numeric"] = np.array([age_map.get(str(a), np.nan) for a in src["age"]], dtype=np.float64)

        med_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        ]
        n_rows = len(time_hosp)
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)

        for mc in med_cols:
            if mc in src:
                val = src[mc]
                active_cnt += ((val != "No") & (val != None)).astype(np.float64)
                up_cnt += (val == "Up").astype(np.float64)
                down_cnt += (val == "Down").astype(np.float64)

        feats["num_active_diabetes_meds"] = active_cnt
        feats["num_meds_up"] = up_cnt
        feats["num_meds_down"] = down_cnt
        feats["num_meds_changed"] = up_cnt + down_cnt

        # Regimen Shannon entropy (dissipation & medication chaos)
        p_no = np.maximum(23.0 - active_cnt, 0.0) / 23.0
        p_std = np.maximum(active_cnt - (up_cnt + down_cnt), 0.0) / 23.0
        p_up = up_cnt / 23.0
        p_dn = down_cnt / 23.0
        med_entropy = np.zeros(n_rows, dtype=np.float64)
        for p in [p_no, p_std, p_up, p_dn]:
            mask = p > 1e-6
            med_entropy[mask] -= p[mask] * np.log2(p[mask])
        feats["medication_entropy"] = med_entropy

        if "insulin" in src:
            ins = src["insulin"]
            feats["insulin_active"] = ((ins != "No") & (ins != None)).astype(np.float64)
            feats["insulin_up"] = (ins == "Up").astype(np.float64)
            feats["insulin_down"] = (ins == "Down").astype(np.float64)
            feats["insulin_steady"] = (ins == "Steady").astype(np.float64)

        if "A1Cresult" in src:
            a1c = src["A1Cresult"]
            feats["a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
            feats["a1c_high"] = ((a1c == ">7") | (a1c == ">8")).astype(np.float64)

        if "max_glu_serum" in src:
            glu = src["max_glu_serum"]
            feats["glucose_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
            feats["glucose_high"] = ((glu == ">200") | (glu == ">300")).astype(np.float64)

        if "discharge_disposition_id" in src:
            feats["high_risk_discharge"] = np.array(
                [1.0 if str(v) in {"3", "6", "22", "28"} else 0.0 for v in src["discharge_disposition_id"]],
                dtype=np.float64
            )

        feats["glycemic_unmonitored"] = (
            (feats.get("a1c_tested", 0.0) == 0.0) &
            (feats.get("glucose_tested", 0.0) == 0.0) &
            (time_hosp >= 3.0)
        ).astype(np.float64)

        return feats

    feats_train = extract_features(train)
    feats_eval = extract_features(eval_cols)
    for fname, farr in feats_train.items():
        train_data[fname] = farr
        eval_data[fname] = feats_eval[fname]
        extra_numeric.append(fname)

    # Metformin / Renal conflict quality indicator
    if "cm_renal" in train_data and "metformin" in train:
        tr_met = ((train["metformin"] != "No") & (train["metformin"] != None)).astype(np.float64)
        ev_met = ((eval_cols["metformin"] != "No") & (eval_cols["metformin"] != None)).astype(np.float64)
        train_data["metformin_renal_conflict"] = (tr_met * train_data["cm_renal"]).astype(np.float64)
        eval_data["metformin_renal_conflict"] = (ev_met * eval_data["cm_renal"]).astype(np.float64)
        extra_numeric.append("metformin_renal_conflict")

    # 6. ICD-9 3-Digit Prefix Target Encodings
    def get_icd9_prefix(code):
        if code is None:
            return "missing"
        c_str = str(code).strip()
        if not c_str or c_str == "?":
            return "missing"
        return c_str.split(".")[0]

    pfx_targets = ["diag_1", "diag_2", "diag_3"]
    pfx_train_map = {}
    pfx_eval_map = {}
    for diag in pfx_targets:
        if diag in train and diag in eval_cols:
            pfx_tr = np.array([get_icd9_prefix(c) for c in train[diag]], dtype=object)
            pfx_ev = np.array([get_icd9_prefix(c) for c in eval_cols[diag]], dtype=object)
            pfx_col = f"{diag}_pfx"
            train_data[pfx_col] = pfx_tr
            eval_data[pfx_col] = pfx_ev
            extra_categorical.append(pfx_col)

            if oof_te is not None:
                tr_te, ev_te = oof_te(pfx_tr, y_train, pfx_ev, n_folds=5, smoothing=30.0, seed=101)
                te_pfx_name = f"{pfx_col}_te"
                train_data[te_pfx_name] = tr_te.astype(np.float64)
                eval_data[te_pfx_name] = ev_te.astype(np.float64)
                extra_numeric.append(te_pfx_name)
                pfx_train_map[diag] = tr_te
                pfx_eval_map[diag] = ev_te

    if "diag_1" in pfx_train_map and "diag_2" in pfx_train_map and "diag_3" in pfx_train_map:
        train_data["pfx_te_max"] = np.maximum.reduce([pfx_train_map["diag_1"], pfx_train_map["diag_2"], pfx_train_map["diag_3"]]).astype(np.float64)
        eval_data["pfx_te_max"] = np.maximum.reduce([pfx_eval_map["diag_1"], pfx_eval_map["diag_2"], pfx_eval_map["diag_3"]]).astype(np.float64)
        extra_numeric.append("pfx_te_max")

    # 7. Low-Rank Phenotype & Medication Decomposition
    def compute_low_rank_factors(tr_mat, ev_mat, n_components=4, prefix="svd"):
        mu = np.nanmean(tr_mat, axis=0)
        sigma = np.nanstd(tr_mat, axis=0)
        sigma[sigma < 1e-6] = 1.0

        z_tr = np.nan_to_num((tr_mat - mu) / sigma, nan=0.0)
        z_ev = np.nan_to_num((ev_mat - mu) / sigma, nan=0.0)

        p = tr_mat.shape[1]
        k = min(n_components, p - 1)
        if k < 1:
            return {}, {}

        cov = np.dot(z_tr.T, z_tr) / max(len(z_tr) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        top_vecs = eigvecs[:, idx[:k]]

        proj_tr = np.dot(z_tr, top_vecs)
        proj_ev = np.dot(z_ev, top_vecs)
        rec_tr = z_tr - np.dot(proj_tr, top_vecs.T)
        rec_ev = z_ev - np.dot(proj_ev, top_vecs.T)
        res_tr = np.sum(rec_tr ** 2, axis=1)
        res_ev = np.sum(rec_ev ** 2, axis=1)

        tr_feats = {f"{prefix}_comp_{i}": proj_tr[:, i].astype(np.float64) for i in range(k)}
        ev_feats = {f"{prefix}_comp_{i}": proj_ev[:, i].astype(np.float64) for i in range(k)}
        tr_feats[f"{prefix}_residual"] = res_tr.astype(np.float64)
        ev_feats[f"{prefix}_residual"] = res_ev.astype(np.float64)
        return tr_feats, ev_feats

    cm_keys = [
        "cm_mi", "cm_chf", "cm_pvd", "cm_cerebrovascular", "cm_dementia", "cm_pulmonary",
        "cm_rheumatic", "cm_peptic_ulcer", "cm_liver", "cm_hemiplegia", "cm_renal",
        "cm_cancer", "cm_metastatic", "cm_diabetes_complicated"
    ]
    present_cm = [k for k in cm_keys if k in train_data]
    if len(present_cm) >= 4:
        cm_mat_tr = np.column_stack([train_data[k] for k in present_cm])
        cm_mat_ev = np.column_stack([eval_data[k] for k in present_cm])
        if "cm_chf" in train_data and "cm_renal" in train_data:
            cm_mat_tr = np.hstack([cm_mat_tr, (train_data["cm_chf"] * train_data["cm_renal"])[:, None]])
            cm_mat_ev = np.hstack([cm_mat_ev, (eval_data["cm_chf"] * eval_data["cm_renal"])[:, None]])
        tr_pheno, ev_pheno = compute_low_rank_factors(cm_mat_tr, cm_mat_ev, n_components=5, prefix="pheno_svd")
        for fn, fa in tr_pheno.items():
            train_data[fn] = fa
            eval_data[fn] = ev_pheno[fn]
            extra_numeric.append(fn)

    # 8. Clinical Archetype Metric Gravity (Latent Attractor Field)
    arch_cols = [
        "number_inpatient", "number_emergency", "number_outpatient",
        "time_in_hospital", "num_medications", "num_lab_procedures",
        "num_procedures", "number_diagnoses", "charlson_score",
        "num_active_diabetes_meds", "num_meds_changed"
    ]
    arch_present = [c for c in arch_cols if c in train_data and c in eval_data]
    if len(arch_present) >= 5:
        a_tr = np.column_stack([np.nan_to_num(train_data[c].astype(np.float64), nan=0.0) for c in arch_present])
        a_ev = np.column_stack([np.nan_to_num(eval_data[c].astype(np.float64), nan=0.0) for c in arch_present])

        mu_a = np.mean(a_tr, axis=0)
        std_a = np.std(a_tr, axis=0)
        std_a[std_a < 1e-6] = 1.0

        z_a_tr = (a_tr - mu_a) / std_a
        z_a_ev = (a_ev - mu_a) / std_a

        mu_pos = np.mean(z_a_tr[y_train == 1], axis=0)
        mu_neg = np.mean(z_a_tr[y_train == 0], axis=0)
        u_pos = mu_pos / (np.linalg.norm(mu_pos) + 1e-12)
        u_neg = mu_neg / (np.linalg.norm(mu_neg) + 1e-12)

        u_tr_norms = np.linalg.norm(z_a_tr, axis=1, keepdims=True) + 1e-12
        u_ev_norms = np.linalg.norm(z_a_ev, axis=1, keepdims=True) + 1e-12
        z_unit_tr = z_a_tr / u_tr_norms
        z_unit_ev = z_a_ev / u_ev_norms

        cos_pos_tr = np.dot(z_unit_tr, u_pos)
        cos_pos_ev = np.dot(z_unit_ev, u_pos)
        cos_neg_tr = np.dot(z_unit_tr, u_neg)
        cos_neg_ev = np.dot(z_unit_ev, u_neg)

        dist_pos_tr = np.linalg.norm(z_a_tr - mu_pos, axis=1)
        dist_pos_ev = np.linalg.norm(z_a_ev - mu_pos, axis=1)
        dist_neg_tr = np.linalg.norm(z_a_tr - mu_neg, axis=1)
        dist_neg_ev = np.linalg.norm(z_a_ev - mu_neg, axis=1)

        train_data["archetype_pull"] = (cos_pos_tr - cos_neg_tr).astype(np.float64)
        eval_data["archetype_pull"] = (cos_pos_ev - cos_neg_ev).astype(np.float64)
        extra_numeric.append("archetype_pull")

        train_data["archetype_dist_ratio"] = (dist_neg_tr / (dist_pos_tr + 1e-6)).astype(np.float64)
        eval_data["archetype_dist_ratio"] = (dist_neg_ev / (dist_pos_ev + 1e-6)).astype(np.float64)
        extra_numeric.append("archetype_dist_ratio")

    # 9. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 10. Stratified 4-Fold Cross-Validated Multi-Architecture Ensemble
    lgb_mod = globals().get("lgb") or globals().get("lightgbm")
    if lgb_mod is None:
        import lightgbm as lgb_mod

    xgb_mod = globals().get("xgb") or globals().get("xgboost")
    if xgb_mod is None:
        try:
            import xgboost as xgb_mod
        except Exception:
            xgb_mod = None

    use_xgb = xgb_mod is not None
    n_models = 4 if use_xgb else 3
    n_splits = 4
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_eval_rows = len(eval_cols["time_in_hospital"])
    oof_preds = [np.zeros(len(y_train), dtype=np.float64) for _ in range(n_models)]
    eval_preds = [np.zeros(n_eval_rows, dtype=np.float64) for _ in range(n_models)]

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        xtr_f, ytr_f = x_train[tr_idx], y_train[tr_idx]
        xval_f = x_train[val_idx]

        # Model 0: Balanced Leaf-Wise LightGBM
        m0 = lgb_mod.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=0.10,
            cat_smooth=30.0,
            random_state=42 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        m0.fit(xtr_f, ytr_f, categorical_feature=cat_indices)
        oof_preds[0][val_idx] = m0.predict_proba(xval_f)[:, 1]
        eval_preds[0] += m0.predict_proba(x_eval)[:, 1] / float(n_splits)

        # Model 1: Shallow Regularized LightGBM
        m1 = lgb_mod.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.03,
            num_leaves=16,
            min_child_samples=180,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_alpha=0.20,
            reg_lambda=1.5,
            cat_smooth=60.0,
            random_state=123 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(xtr_f, ytr_f, categorical_feature=cat_indices)
        oof_preds[1][val_idx] = m1.predict_proba(xval_f)[:, 1]
        eval_preds[1] += m1.predict_proba(x_eval)[:, 1] / float(n_splits)

        # Model 2: Interaction-Focused Class-Weighted LightGBM
        m2 = lgb_mod.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.025,
            num_leaves=42,
            min_child_samples=60,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.20,
            scale_pos_weight=1.12,
            cat_smooth=20.0,
            random_state=777 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(xtr_f, ytr_f, categorical_feature=cat_indices)
        oof_preds[2][val_idx] = m2.predict_proba(xval_f)[:, 1]
        eval_preds[2] += m2.predict_proba(x_eval)[:, 1] / float(n_splits)

        # Model 3: Histogram XGBoost (Depth-Wise)
        if use_xgb:
            try:
                m3 = xgb_mod.XGBClassifier(
                    n_estimators=190,
                    learning_rate=0.035,
                    max_depth=5,
                    min_child_weight=20,
                    subsample=0.80,
                    colsample_bytree=0.55,
                    reg_alpha=0.15,
                    reg_lambda=2.0,
                    tree_method="hist",
                    random_state=42 + fold_idx,
                    n_jobs=1,
                    eval_metric="logloss",
                )
                m3.fit(xtr_f, ytr_f)
                oof_preds[3][val_idx] = m3.predict_proba(xval_f)[:, 1]
                eval_preds[3] += m3.predict_proba(x_eval)[:, 1] / float(n_splits)
            except Exception:
                pass

    # 11. Nelder-Mead Out-Of-Fold Rank-AUC Optimization
    def to_percentiles(p_arr: np.ndarray) -> np.ndarray:
        n = len(p_arr)
        pct = np.empty(n, dtype=np.float64)
        order = np.argsort(p_arr)
        pct[order] = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
        return pct

    r_oof = [to_percentiles(p) for p in oof_preds]
    r_eval = [to_percentiles(p) for p in eval_preds]

    def auc_objective(w):
        w_pos = np.maximum(w, 0.0)
        s = np.sum(w_pos)
        if s < 1e-8:
            return 0.0
        w_norm = w_pos / s
        blend = sum(w_norm[i] * r_oof[i] for i in range(n_models))
        return -roc_auc_score(y_train, blend)

    w_init = np.ones(n_models) / float(n_models)
    opt_res = minimize(auc_objective, w_init, method="Nelder-Mead", options={"maxiter": 70, "disp": False})
    w_opt = np.maximum(opt_res.x, 0.0)
    if np.sum(w_opt) < 1e-8:
        w_opt = np.ones(n_models) / float(n_models)
    else:
        w_opt /= np.sum(w_opt)

    eval_blend_rank = sum(w_opt[i] * r_eval[i] for i in range(n_models))
    eval_blend_prob = sum(w_opt[i] * eval_preds[i] for i in range(n_models))

    # 12. Monotonic Quantile Matching Calibration
    sorted_prob = np.sort(eval_blend_prob)
    pct_grid = np.linspace(0.5 / n_eval_rows, 1.0 - 0.5 / n_eval_rows, n_eval_rows)
    preds = np.interp(eval_blend_rank, pct_grid, sorted_prob)

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
