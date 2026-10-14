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
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols`."""
    n_train = len(y_train)
    n_eval = len(eval_cols[task.numeric_columns[0]])
    train_dict = dict(train)
    eval_dict = dict(eval_cols)

    # 1. Resilient ICD-9 chapter grouping
    grouper = globals().get("icd9_group", None)

    def safe_icd9_group(code):
        if grouper is not None:
            try:
                return str(grouper(code))
            except Exception:
                pass
        if code is None or code == "" or code == "?":
            return "missing"
        s = str(code).strip()
        if s.startswith("250"):
            return "diabetes"
        if s.startswith(("V", "E")):
            return "other"
        try:
            val = float(s.split(".")[0])
            if 390 <= val <= 459 or val == 785:
                return "circulatory"
            elif 460 <= val <= 519 or val == 786:
                return "respiratory"
            elif 520 <= val <= 579 or val == 787:
                return "digestive"
            elif 800 <= val <= 999:
                return "injury"
            elif 710 <= val <= 739:
                return "musculoskeletal"
            elif 580 <= val <= 629 or val == 788:
                return "genitourinary"
            elif 140 <= val <= 239:
                return "neoplasms"
        except Exception:
            pass
        return "other"

    extra_cat = ["diag_1_group", "diag_2_group", "diag_3_group"]
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        grp_name = f"{d_col}_group"
        train_dict[grp_name] = np.array(
            [safe_icd9_group(c) for c in train_dict.get(d_col, [])],
            dtype=object,
        )
        eval_dict[grp_name] = np.array(
            [safe_icd9_group(c) for c in eval_dict.get(d_col, [])], dtype=object
        )

    # 2. Domain & Utilization Features
    extra_num = []

    def add_num(name, tr_arr, ev_arr):
        train_dict[name] = np.asarray(tr_arr, dtype=np.float64)
        eval_dict[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_num.append(name)

    # Utilization intensity (primary predictor in 30-day readmission)
    for d, n in [(train_dict, n_train), (eval_dict, n_eval)]:
        inp = np.nan_to_num(d.get("number_inpatient", np.zeros(n)), nan=0.0)
        emg = np.nan_to_num(d.get("number_emergency", np.zeros(n)), nan=0.0)
        out = np.nan_to_num(d.get("number_outpatient", np.zeros(n)), nan=0.0)
        tot = inp + emg + out
        stay = np.maximum(
            np.nan_to_num(d.get("time_in_hospital", np.ones(n)), nan=1.0), 1.0
        )
        meds = np.nan_to_num(d.get("num_medications", np.zeros(n)), nan=0.0)
        labs = np.nan_to_num(d.get("num_lab_procedures", np.zeros(n)), nan=0.0)
        procs = np.nan_to_num(d.get("num_procedures", np.zeros(n)), nan=0.0)
        diags = np.nan_to_num(d.get("number_diagnoses", np.zeros(n)), nan=0.0)

        d["_tot_visits"] = tot
        d["_inp_ratio"] = inp / (tot + 1.0)
        d["_emg_ratio"] = emg / (tot + 1.0)
        d["_has_inp"] = (inp > 0.0).astype(np.float64)
        d["_has_emg"] = (emg > 0.0).astype(np.float64)
        d["_freq_inp"] = (inp >= 2.0).astype(np.float64)

        d["_med_per_day"] = meds / stay
        d["_lab_per_day"] = labs / stay
        d["_proc_per_day"] = procs / stay
        d["_diag_per_day"] = diags / stay
        d["_inp_x_stay"] = inp * stay
        d["_inp_x_diag"] = inp * diags

        # Discharge and admission indicators
        disch = d.get("discharge_disposition_id", np.empty(n, dtype=object))
        admit = d.get("admission_type_id", np.empty(n, dtype=object))
        src = d.get("admission_source_id", np.empty(n, dtype=object))
        d["_disch_home"] = (disch == "1").astype(np.float64)
        d["_disch_facility"] = np.isin(disch, ["3", "6", "22", "23"]).astype(
            np.float64
        )
        d["_admit_emerg"] = (admit == "1").astype(np.float64)
        d["_src_emerg"] = (src == "7").astype(np.float64)

        # Lab & medication change indicators
        a1c = d.get("A1Cresult", np.empty(n, dtype=object))
        glu = d.get("max_glu_serum", np.empty(n, dtype=object))
        chg = d.get("change", np.empty(n, dtype=object))
        dmed = d.get("diabetesMed", np.empty(n, dtype=object))
        ins = d.get("insulin", np.empty(n, dtype=object))

        d["_a1c_tested"] = (~np.isin(a1c, ["None", None])).astype(np.float64)
        d["_a1c_high"] = np.isin(a1c, [">7", ">8"]).astype(np.float64)
        d["_glu_tested"] = (~np.isin(glu, ["None", None])).astype(np.float64)
        d["_glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)
        d["_has_med_chg"] = (chg == "Ch").astype(np.float64)
        d["_has_diab_med"] = (dmed == "Yes").astype(np.float64)
        d["_insulin_prescribed"] = (ins != "No").astype(np.float64)
        d["_insulin_chg"] = np.isin(ins, ["Up", "Down"]).astype(np.float64)

        # High-risk diagnosis chapter flags
        g1, g2, g3 = (
            d["diag_1_group"],
            d["diag_2_group"],
            d["diag_3_group"],
        )
        d["_diag_circ"] = (
            (g1 == "circulatory")
            | (g2 == "circulatory")
            | (g3 == "circulatory")
        ).astype(np.float64)
        d["_diag_diab"] = (
            (g1 == "diabetes") | (g2 == "diabetes") | (g3 == "diabetes")
        ).astype(np.float64)
        d["_diag_resp"] = (
            (g1 == "respiratory")
            | (g2 == "respiratory")
            | (g3 == "respiratory")
        ).astype(np.float64)
        d["_diag_digest"] = (
            (g1 == "digestive") | (g2 == "digestive") | (g3 == "digestive")
        ).astype(np.float64)
        d["_diag_genit"] = (
            (g1 == "genitourinary")
            | (g2 == "genitourinary")
            | (g3 == "genitourinary")
        ).astype(np.float64)

    for feat_key in [
        "_tot_visits",
        "_inp_ratio",
        "_emg_ratio",
        "_has_inp",
        "_has_emg",
        "_freq_inp",
        "_med_per_day",
        "_lab_per_day",
        "_proc_per_day",
        "_diag_per_day",
        "_inp_x_stay",
        "_inp_x_diag",
        "_disch_home",
        "_disch_facility",
        "_admit_emerg",
        "_src_emerg",
        "_a1c_tested",
        "_a1c_high",
        "_glu_tested",
        "_glu_high",
        "_has_med_chg",
        "_has_diab_med",
        "_insulin_prescribed",
        "_insulin_chg",
        "_diag_circ",
        "_diag_diab",
        "_diag_resp",
        "_diag_digest",
        "_diag_genit",
    ]:
        add_num(feat_key, train_dict.pop(feat_key), eval_dict.pop(feat_key))

    # 3. Out-Of-Fold Smoothed Target Encoding for High-Cardinality Categoricals
    def oof_target_encode(tr_cat, y_tr, ev_cat, n_splits=5, smoothing=15.0):
        mean_y = float(np.mean(y_tr))
        n_tr, n_ev = len(y_tr), len(ev_cat)
        tr_out = np.full(n_tr, mean_y, dtype=np.float64)
        ev_out = np.full(n_ev, mean_y, dtype=np.float64)

        tr_s = [str(x) if x is not None else "__NONE__" for x in tr_cat]
        ev_s = [str(x) if x is not None else "__NONE__" for x in ev_cat]

        fold_sz = (n_tr + n_splits - 1) // n_splits
        folds = [
            np.arange(i * fold_sz, min((i + 1) * fold_sz, n_tr))
            for i in range(n_splits)
        ]

        for i in range(n_splits):
            val_idx = folds[i]
            tr_idx = np.concatenate(
                [folds[j] for j in range(n_splits) if j != i]
            )

            cnts, sums = {}, {}
            for idx in tr_idx:
                c = tr_s[idx]
                cnts[c] = cnts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y_tr[idx]

            s_map = {
                c: (sums[c] + smoothing * mean_y) / (cnts[c] + smoothing)
                for c in cnts
            }
            for idx in val_idx:
                tr_out[idx] = s_map.get(tr_s[idx], mean_y)

        # Whole train encoding for eval
        full_cnts, full_sums = {}, {}
        for idx in range(n_tr):
            c = tr_s[idx]
            full_cnts[c] = full_cnts.get(c, 0) + 1
            full_sums[c] = full_sums.get(c, 0.0) + y_tr[idx]
        full_map = {
            c: (full_sums[c] + smoothing * mean_y) / (full_cnts[c] + smoothing)
            for c in full_cnts
        }
        for idx in range(n_ev):
            ev_out[idx] = full_map.get(ev_s[idx], mean_y)

        return tr_out, ev_out

    te_features = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "medical_specialty",
        "admission_source_id",
    ]
    te_tr_diags, te_ev_diags = [], []
    for col in te_features:
        if col in train_dict:
            tr_te, ev_te = oof_target_encode(
                train_dict[col], y_train, eval_dict[col]
            )
            add_num(f"te_{col}", tr_te, ev_te)
            if col.startswith("diag_"):
                te_tr_diags.append(tr_te)
                te_ev_diags.append(ev_te)

    if len(te_tr_diags) == 3:
        add_num(
            "te_diag_mean",
            (te_tr_diags[0] + te_tr_diags[1] + te_tr_diags[2]) / 3.0,
            (te_ev_diags[0] + te_ev_diags[1] + te_ev_diags[2]) / 3.0,
        )
        add_num(
            "te_diag_max",
            np.maximum(
                te_tr_diags[0], np.maximum(te_tr_diags[1], te_tr_diags[2])
            ),
            np.maximum(
                te_ev_diags[0], np.maximum(te_ev_diags[1], te_ev_diags[2])
            ),
        )

    # 4. Low-Rank Phenotype Decomposition (Tensor SVD on Meds x Diagnoses)
    med_cols = [
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
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
    ]
    diag_chapters = [
        "circulatory",
        "respiratory",
        "digestive",
        "diabetes",
        "injury",
        "musculoskeletal",
        "genitourinary",
        "neoplasms",
        "other",
    ]

    def build_phenotype_matrix(d, n):
        cols = []
        for m in med_cols:
            arr = d.get(m, np.empty(n, dtype=object))
            val = np.zeros(n, dtype=np.float64)
            val[arr == "Steady"] = 1.0
            val[arr == "Down"] = -1.0
            val[arr == "Up"] = 2.0
            cols.append(val)
        g1, g2, g3 = (
            d["diag_1_group"],
            d["diag_2_group"],
            d["diag_3_group"],
        )
        for chap in diag_chapters:
            cols.append(
                ((g1 == chap).astype(np.float64))
                + ((g2 == chap).astype(np.float64))
                + ((g3 == chap).astype(np.float64))
            )
        return np.column_stack(cols)

    P_tr = build_phenotype_matrix(train_dict, n_train)
    P_ev = build_phenotype_matrix(eval_dict, n_eval)
    mu_p = np.mean(P_tr, axis=0)
    X_tr = P_tr - mu_p
    cov_p = (X_tr.T @ X_tr) / max(n_train, 1)
    eigvals, eigvecs = np.linalg.eigh(cov_p)
    top_components = eigvecs[:, np.argsort(eigvals)[::-1][:6]]

    svd_tr = X_tr @ top_components
    svd_ev = (P_ev - mu_p) @ top_components
    for k in range(6):
        add_num(f"svd_pheno_{k}", svd_tr[:, k], svd_ev[:, k])

    # 5. Encoding & LightGBM Multi-Order Interaction Ensemble
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_dict,
        extra_numeric=tuple(extra_num),
        extra_categorical=tuple(extra_cat),
    )
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)
    cat_indices = encoder.categorical_indices

    # Model 1: Depth 5 higher-order interactions
    m1 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=24,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)

    # Model 2: Depth 4 low-rank interaction regularizer
    m2 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=6.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)

    pred1 = m1.predict_proba(x_eval)[:, 1]
    pred2 = m2.predict_proba(x_eval)[:, 1]
    pred = 0.55 * pred1 + 0.45 * pred2

    return np.clip(np.nan_to_num(pred, nan=float(np.mean(y_train))), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
