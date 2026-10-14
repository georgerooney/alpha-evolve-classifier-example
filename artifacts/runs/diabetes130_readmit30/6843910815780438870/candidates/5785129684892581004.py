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
    # Safe retrieval of toolkit helpers
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

    # 1. Comorbidities & Weighted Charlson Score
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

    # 2. ICD-9 Chapter Categories
    if icd9_grp is not None:
        for diag in ["diag_1", "diag_2", "diag_3"]:
            if diag in train and diag in eval_cols:
                grp_col = f"{diag}_group"
                train_data[grp_col] = np.array([icd9_grp(c) for c in train[diag]], dtype=object)
                eval_data[grp_col] = np.array([icd9_grp(c) for c in eval_cols[diag]], dtype=object)
                extra_categorical.append(grp_col)

    # 3. Out-Of-Fold Bayesian Target Encoding for High-Cardinality Fields
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
                tr_enc, ev_enc = oof_te(
                    train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
                )
                col_name = f"{col}_te"
                train_data[col_name] = tr_enc.astype(np.float64)
                eval_data[col_name] = ev_enc.astype(np.float64)
                extra_numeric.append(col_name)
                te_train_map[col] = tr_enc
                te_eval_map[col] = ev_enc

        # Diagnosis composite risk features
        if "diag_1" in te_train_map and "diag_2" in te_train_map and "diag_3" in te_train_map:
            train_data["diag_te_max"] = np.maximum.reduce(
                [te_train_map["diag_1"], te_train_map["diag_2"], te_train_map["diag_3"]]
            ).astype(np.float64)
            eval_data["diag_te_max"] = np.maximum.reduce(
                [te_eval_map["diag_1"], te_eval_map["diag_2"], te_eval_map["diag_3"]]
            ).astype(np.float64)
            extra_numeric.append("diag_te_max")

            train_data["diag_te_mean"] = (
                (te_train_map["diag_1"] + te_train_map["diag_2"] + te_train_map["diag_3"]) / 3.0
            ).astype(np.float64)
            eval_data["diag_te_mean"] = (
                (te_eval_map["diag_1"] + te_eval_map["diag_2"] + te_eval_map["diag_3"]) / 3.0
            ).astype(np.float64)
            extra_numeric.append("diag_te_mean")

    # 4. Domain Feature Engineering (Utilization, Intensity, Medication Regimen)
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
        }

        # Age mapping
        if "age" in src:
            age_map = {
                "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
                "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
                "[80-90)": 85.0, "[90-100)": 95.0,
            }
            feats["age_numeric"] = np.array([age_map.get(str(a), np.nan) for a in src["age"]], dtype=np.float64)

        # Medication regime analysis
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

        return feats

    feats_train = extract_features(train)
    feats_eval = extract_features(eval_cols)
    for fname, farr in feats_train.items():
        train_data[fname] = farr
        eval_data[fname] = feats_eval[fname]
        extra_numeric.append(fname)

    # ICD-9 3-Digit Category-Level Encodings
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
        train_data["pfx_te_max"] = np.maximum.reduce(
            [pfx_train_map["diag_1"], pfx_train_map["diag_2"], pfx_train_map["diag_3"]]
        ).astype(np.float64)
        eval_data["pfx_te_max"] = np.maximum.reduce(
            [pfx_eval_map["diag_1"], pfx_eval_map["diag_2"], pfx_eval_map["diag_3"]]
        ).astype(np.float64)
        extra_numeric.append("pfx_te_max")

    # Low-Rank Matrix / Tensor Decomposition Helper
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

        # Reconstruction residual norm (quantifies clinical atypia & complexity)
        rec_tr = z_tr - np.dot(proj_tr, top_vecs.T)
        rec_ev = z_ev - np.dot(proj_ev, top_vecs.T)
        res_tr = np.sum(rec_tr ** 2, axis=1)
        res_ev = np.sum(rec_ev ** 2, axis=1)

        tr_feats = {f"{prefix}_comp_{i}": proj_tr[:, i].astype(np.float64) for i in range(k)}
        ev_feats = {f"{prefix}_comp_{i}": proj_ev[:, i].astype(np.float64) for i in range(k)}
        tr_feats[f"{prefix}_residual"] = res_tr.astype(np.float64)
        ev_feats[f"{prefix}_residual"] = res_ev.astype(np.float64)
        return tr_feats, ev_feats

    # 1. Comorbidity Phenotype Decomposition
    cm_keys = [
        "cm_mi", "cm_chf", "cm_pvd", "cm_cerebrovascular", "cm_dementia", "cm_pulmonary",
        "cm_rheumatic", "cm_peptic_ulcer", "cm_liver", "cm_hemiplegia", "cm_renal",
        "cm_cancer", "cm_metastatic", "cm_diabetes_complicated"
    ]
    present_cm = [k for k in cm_keys if k in train_data]
    if len(present_cm) >= 4:
        cm_mat_tr = np.column_stack([train_data[k] for k in present_cm])
        cm_mat_ev = np.column_stack([eval_data[k] for k in present_cm])
        
        # Clinical condition interactions
        if "cm_chf" in train_data and "cm_renal" in train_data:
            cr_tr = (train_data["cm_chf"] * train_data["cm_renal"])[:, None]
            cr_ev = (eval_data["cm_chf"] * eval_data["cm_renal"])[:, None]
            cm_mat_tr = np.hstack([cm_mat_tr, cr_tr])
            cm_mat_ev = np.hstack([cm_mat_ev, cr_ev])

        tr_pheno, ev_pheno = compute_low_rank_factors(cm_mat_tr, cm_mat_ev, n_components=5, prefix="pheno_svd")
        for fn, fa in tr_pheno.items():
            train_data[fn] = fa
            eval_data[fn] = ev_pheno[fn]
            extra_numeric.append(fn)

    # 2. Polypharmacy & Medication Regimen Decomposition
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "insulin", "glyburide-metformin", "glipizide-metformin"
    ]
    med_mat_tr_list = []
    med_mat_ev_list = []
    for mc in med_cols:
        if mc in train and mc in eval_cols:
            tr_act = ((train[mc] != "No") & (train[mc] != None)).astype(np.float64)
            ev_act = ((eval_cols[mc] != "No") & (eval_cols[mc] != None)).astype(np.float64)
            tr_chg = ((train[mc] == "Up") | (train[mc] == "Down")).astype(np.float64)
            ev_chg = ((eval_cols[mc] == "Up") | (eval_cols[mc] == "Down")).astype(np.float64)
            med_mat_tr_list.extend([tr_act, tr_chg])
            med_mat_ev_list.extend([ev_act, ev_chg])

    if len(med_mat_tr_list) >= 4:
        med_mat_tr = np.column_stack(med_mat_tr_list)
        med_mat_ev = np.column_stack(med_mat_ev_list)
        tr_poly, ev_poly = compute_low_rank_factors(med_mat_tr, med_mat_ev, n_components=4, prefix="poly_svd")
        for fn, fa in tr_poly.items():
            train_data[fn] = fa
            eval_data[fn] = ev_poly[fn]
            extra_numeric.append(fn)

    # 3. Utilization & Consumption Intensity Decomposition
    util_cols = [
        "number_inpatient", "number_emergency", "number_outpatient",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_diagnoses"
    ]
    u_tr_list = [np.log1p(np.nan_to_num(train[c].astype(np.float64), nan=0.0)) for c in util_cols if c in train]
    u_ev_list = [np.log1p(np.nan_to_num(eval_cols[c].astype(np.float64), nan=0.0)) for c in util_cols if c in eval_cols]
    if len(u_tr_list) >= 4:
        u_mat_tr = np.column_stack(u_tr_list)
        u_mat_ev = np.column_stack(u_ev_list)
        tr_util, ev_util = compute_low_rank_factors(u_mat_tr, u_mat_ev, n_components=3, prefix="util_svd")
        for fn, fa in tr_util.items():
            train_data[fn] = fa
            eval_data[fn] = ev_util[fn]
            extra_numeric.append(fn)

    # 5. Fit TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 6. Diverse Model Ensemble
    # LightGBM Model 1: Balanced leaf-wise
    lgb_1 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=0.08,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb_1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb1 = lgb_1.predict_proba(x_eval)[:, 1]

    # LightGBM Model 2: Shallow regularized (robust against noise)
    lgb_2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.025,
        num_leaves=16,
        min_child_samples=220,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.15,
        reg_lambda=1.2,
        cat_smooth=60.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    lgb_2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb2 = lgb_2.predict_proba(x_eval)[:, 1]

    # LightGBM Model 3: Interaction-focused with minority class emphasis
    lgb_3 = lgb.LGBMClassifier(
        n_estimators=210,
        learning_rate=0.02,
        num_leaves=40,
        min_child_samples=75,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.15,
        scale_pos_weight=1.15,
        cat_smooth=20.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    lgb_3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb3 = lgb_3.predict_proba(x_eval)[:, 1]

    # XGBoost Model: Histogram depth-wise trees
    p_xgb = None
    try:
        xgb_mod = globals().get("xgb") or globals().get("xgboost")
        if xgb_mod is None:
            import xgboost as xgb_mod
        xgb_model = xgb_mod.XGBClassifier(
            n_estimators=210,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=22,
            subsample=0.80,
            colsample_bytree=0.55,
            reg_alpha=0.15,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        xgb_model.fit(x_train, y_train)
        p_xgb = xgb_model.predict_proba(x_eval)[:, 1]
    except Exception:
        p_xgb = None

    # 7. Concordance-Maximizing Rank Ensemble Optimization
    def to_percentiles(p_arr: np.ndarray) -> np.ndarray:
        n = len(p_arr)
        pct = np.empty(n, dtype=np.float64)
        order = np.argsort(p_arr)
        pct[order] = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
        return pct

    if p_xgb is not None:
        r1, r2, r3, rx = to_percentiles(p_lgb1), to_percentiles(p_lgb2), to_percentiles(p_lgb3), to_percentiles(p_xgb)
        rank_score = 0.35 * r1 + 0.25 * r2 + 0.15 * r3 + 0.25 * rx
        raw_blend = 0.35 * p_lgb1 + 0.25 * p_lgb2 + 0.15 * p_lgb3 + 0.25 * p_xgb
    else:
        r1, r2, r3 = to_percentiles(p_lgb1), to_percentiles(p_lgb2), to_percentiles(p_lgb3)
        rank_score = 0.45 * r1 + 0.35 * r2 + 0.20 * r3
        raw_blend = 0.45 * p_lgb1 + 0.35 * p_lgb2 + 0.20 * p_lgb3

    # Monotonic quantile matching: preserves optimal rank ordering while matching probability distribution
    n_ev = len(eval_cols["time_in_hospital"])
    sorted_raw = np.sort(raw_blend)
    pct_grid = np.linspace(0.5 / n_ev, 1.0 - 0.5 / n_ev, n_ev)
    preds = np.interp(rank_score, pct_grid, sorted_raw)

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
