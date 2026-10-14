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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # Access frozen toolkit tools if available
    _icd9_comorbidities = globals().get("icd9_comorbidities", None)
    _icd9_group = globals().get("icd9_group", None)
    _oof_target_encode = globals().get("oof_target_encode", None)

    n_train = len(y_train)
    n_eval = len(eval_cols["time_in_hospital"])

    # 1. Base Tabular Encoding
    encoder = TabularEncoder(min_count=50)
    x_train_base = encoder.fit_transform(task, train)
    x_eval_base = encoder.transform(eval_cols)

    # 2. Pharmacotherapy & Medication Instability Features
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def extract_med_features(data_dict, n_rows):
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        change_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in data_dict:
                col = data_dict[m]
                is_up = (col == "Up")
                is_down = (col == "Down")
                is_steady = (col == "Steady")
                active_cnt += (is_up | is_down | is_steady).astype(np.float64)
                change_cnt += (is_up | is_down).astype(np.float64)
                up_cnt += is_up.astype(np.float64)
                down_cnt += is_down.astype(np.float64)

        insulin_col = data_dict.get("insulin", np.array(["No"] * n_rows))
        insulin_num = np.zeros(n_rows, dtype=np.float64)
        insulin_num[insulin_col == "Steady"] = 1.0
        insulin_num[insulin_col == "Down"] = 2.0
        insulin_num[insulin_col == "Up"] = 3.0

        metf_col = data_dict.get("metformin", np.array(["No"] * n_rows))
        metf_active = ((metf_col == "Steady") | (metf_col == "Up") | (metf_col == "Down")).astype(np.float64)
        volatility = change_cnt / (active_cnt + 1.0)
        return np.column_stack([active_cnt, change_cnt, up_cnt, down_cnt, insulin_num, metf_active, volatility])

    med_train = extract_med_features(train, n_train)
    med_eval = extract_med_features(eval_cols, n_eval)

    # 3. Clinical Healthcare Velocity & Intensity Metrics
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }

    def extract_clinical_metrics(data_dict, n_rows):
        num_inp = np.nan_to_num(data_dict.get("number_inpatient", np.zeros(n_rows)), nan=0.0)
        num_emg = np.nan_to_num(data_dict.get("number_emergency", np.zeros(n_rows)), nan=0.0)
        num_out = np.nan_to_num(data_dict.get("number_outpatient", np.zeros(n_rows)), nan=0.0)
        time_hosp = np.nan_to_num(data_dict.get("time_in_hospital", np.ones(n_rows)), nan=1.0)
        n_meds = np.nan_to_num(data_dict.get("num_medications", np.zeros(n_rows)), nan=0.0)
        n_labs = np.nan_to_num(data_dict.get("num_lab_procedures", np.zeros(n_rows)), nan=0.0)
        n_procs = np.nan_to_num(data_dict.get("num_procedures", np.zeros(n_rows)), nan=0.0)
        n_diags = np.nan_to_num(data_dict.get("number_diagnoses", np.zeros(n_rows)), nan=0.0)

        total_prior = num_inp + num_emg + num_out
        acute_prior = num_inp + num_emg
        has_prior_inp = (num_inp > 0).astype(np.float64)
        has_prior_emg = (num_emg > 0).astype(np.float64)
        frequent_inp = (num_inp >= 2).astype(np.float64)

        inp_ratio = num_inp / (total_prior + 1.0)
        emg_ratio = num_emg / (total_prior + 1.0)
        acute_ratio = acute_prior / (total_prior + 1.0)

        meds_per_day = n_meds / (time_hosp + 0.1)
        labs_per_day = n_labs / (time_hosp + 0.1)
        procs_per_day = n_procs / (time_hosp + 0.1)
        diags_per_day = n_diags / (time_hosp + 0.1)
        proc_med_ratio = n_procs / (n_meds + 1.0)

        raw_age = data_dict.get("age", np.array(["[60-70)"] * n_rows))
        age_num = np.array([age_map.get(str(a), 60.0) for a in raw_age], dtype=np.float64)
        elderly_fragile = ((age_num >= 75.0) & (num_inp > 0)).astype(np.float64)
        age_inp_interaction = age_num * num_inp

        disch = data_dict.get("discharge_disposition_id", np.array(["1"] * n_rows))
        disch_home = (disch == "1").astype(np.float64)
        disch_snf = np.isin(disch, ["2", "3", "4", "5", "22", "23", "24"]).astype(np.float64)
        disch_homehealth = (disch == "6").astype(np.float64)
        disch_ama = (disch == "7").astype(np.float64)

        adm_src = data_dict.get("admission_source_id", np.array(["1"] * n_rows))
        adm_emergency = (adm_src == "7").astype(np.float64)
        adm_transfer = np.isin(adm_src, ["4", "5", "6"]).astype(np.float64)

        adm_typ = data_dict.get("admission_type_id", np.array(["1"] * n_rows))
        admtype_emerg = (adm_typ == "1").astype(np.float64)
        admtype_elect = (adm_typ == "3").astype(np.float64)

        glu_col = data_dict.get("max_glu_serum", np.array(["None"] * n_rows))
        glu_tested = (glu_col != "None").astype(np.float64)
        glu_high = np.isin(glu_col, [">200", ">300"]).astype(np.float64)

        a1c_col = data_dict.get("A1Cresult", np.array(["None"] * n_rows))
        a1c_tested = (a1c_col != "None").astype(np.float64)
        a1c_high = np.isin(a1c_col, [">7", ">8"]).astype(np.float64)

        chg_col = data_dict.get("change", np.array(["No"] * n_rows))
        chg_flag = (chg_col == "Ch").astype(np.float64)

        dm_col = data_dict.get("diabetesMed", np.array(["No"] * n_rows))
        dm_flag = (dm_col == "Yes").astype(np.float64)

        return np.column_stack([
            total_prior, acute_prior, has_prior_inp, has_prior_emg, frequent_inp,
            inp_ratio, emg_ratio, acute_ratio, meds_per_day, labs_per_day,
            procs_per_day, diags_per_day, proc_med_ratio, age_num, elderly_fragile,
            age_inp_interaction, disch_home, disch_snf, disch_homehealth, disch_ama,
            adm_emergency, adm_transfer, admtype_emerg, admtype_elect,
            glu_tested, glu_high, a1c_tested, a1c_high, chg_flag, dm_flag,
        ])

    metrics_train = extract_clinical_metrics(train, n_train)
    metrics_eval = extract_clinical_metrics(eval_cols, n_eval)

    # 4. Comorbidities extraction
    if _icd9_comorbidities is not None:
        cm_tr = _icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_ev = _icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        cm_keys = sorted(cm_tr.keys())
        cm_mat_tr = np.column_stack([cm_tr[k].astype(np.float64) for k in cm_keys])
        cm_mat_ev = np.column_stack([cm_ev[k].astype(np.float64) for k in cm_keys])
    else:
        cm_mat_tr = np.zeros((n_train, 0), dtype=np.float64)
        cm_mat_ev = np.zeros((n_eval, 0), dtype=np.float64)

    # 5. Out-of-Fold Target Encodings
    te_cols = [
        "discharge_disposition_id", "admission_source_id", "diag_1",
        "diag_2", "diag_3", "medical_specialty", "admission_type_id"
    ]
    te_tr_list, te_ev_list = [], []

    def safe_te(tr_col, y_tr, ev_col):
        if _oof_target_encode is not None:
            return _oof_target_encode(tr_col, y_tr, ev_col, smoothing=25.0, seed=42)
        prior = float(np.mean(y_tr))
        tr_s = np.array([str(x) for x in tr_col], dtype=object)
        ev_s = np.array([str(x) for x in ev_col], dtype=object)
        tr_out = np.full(len(tr_s), prior, dtype=np.float64)
        ev_out = np.full(len(ev_s), prior, dtype=np.float64)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, va_idx in kf.split(tr_s):
            f_tr, f_y = tr_s[tr_idx], y_tr[tr_idx]
            for val in np.unique(f_tr):
                m = (f_tr == val)
                smoothed = (np.sum(f_y[m]) + 25.0 * prior) / (np.sum(m) + 25.0)
                tr_out[va_idx[tr_s[va_idx] == val]] = smoothed
        for val in np.unique(tr_s):
            m = (tr_s == val)
            ev_out[ev_s == val] = (np.sum(y_tr[m]) + 25.0 * prior) / (np.sum(m) + 25.0)
        return tr_out, ev_out

    for col in te_cols:
        tr_enc, ev_enc = safe_te(train[col], y_train, eval_cols[col])
        te_tr_list.append(tr_enc.astype(np.float64))
        te_ev_list.append(ev_enc.astype(np.float64))

    te_mat_tr = np.column_stack(te_tr_list)
    te_mat_ev = np.column_stack(te_ev_list)

    # 6. Bag-of-Diagnoses Latent Semantic Analysis (BoD-LSA)
    def build_encounter_docs(data_dict, n_rows):
        d1 = data_dict.get("diag_1", np.array([None] * n_rows))
        d2 = data_dict.get("diag_2", np.array([None] * n_rows))
        d3 = data_dict.get("diag_3", np.array([None] * n_rows))
        spec = data_dict.get("medical_specialty", np.array([None] * n_rows))
        disch = data_dict.get("discharge_disposition_id", np.array([None] * n_rows))
        adm_src = data_dict.get("admission_source_id", np.array([None] * n_rows))

        docs = []
        for i in range(n_rows):
            toks = ["encounter"]
            c1, c2, c3 = d1[i], d2[i], d3[i]
            if c1 is not None and str(c1) != "None":
                s1 = str(c1).strip()
                toks.append(f"d1_{s1}")
                toks.append(f"icd_{s1}")
                if _icd9_group is not None:
                    toks.append(f"grp1_{_icd9_group(s1)}")
            if c2 is not None and str(c2) != "None":
                s2 = str(c2).strip()
                toks.append(f"d2_{s2}")
                toks.append(f"icd_{s2}")
                if _icd9_group is not None:
                    toks.append(f"grp2_{_icd9_group(s2)}")
            if c3 is not None and str(c3) != "None":
                s3 = str(c3).strip()
                toks.append(f"d3_{s3}")
                toks.append(f"icd_{s3}")
                if _icd9_group is not None:
                    toks.append(f"grp3_{_icd9_group(s3)}")
            if spec[i] is not None and str(spec[i]) != "None":
                toks.append(f"sp_{str(spec[i]).strip().replace(' ', '_')}")
            if disch[i] is not None and str(disch[i]) != "None":
                toks.append(f"ds_{str(disch[i]).strip()}")
            if adm_src[i] is not None and str(adm_src[i]) != "None":
                toks.append(f"as_{str(adm_src[i]).strip()}")
            docs.append(" ".join(toks))
        return docs

    tr_docs = build_encounter_docs(train, n_train)
    ev_docs = build_encounter_docs(eval_cols, n_eval)

    tfidf = TfidfVectorizer(token_pattern=r"\S+", min_df=2, max_features=500, sublinear_tf=True)
    X_tfidf_tr = tfidf.fit_transform(tr_docs)
    X_tfidf_ev = tfidf.transform(ev_docs)

    n_components = min(8, max(2, X_tfidf_tr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_tr = svd.fit_transform(X_tfidf_tr)
    svd_ev = svd.transform(X_tfidf_ev)

    # Combine all feature representations
    extra_train = np.column_stack([med_train, metrics_train, cm_mat_tr, te_mat_tr, svd_tr])
    extra_eval = np.column_stack([med_eval, metrics_eval, cm_mat_ev, te_mat_ev, svd_ev])

    x_train_full = np.hstack([x_train_base, extra_train])
    x_eval_full = np.hstack([x_eval_base, extra_eval])

    # 7. Asymmetric Dual-Engine Ensemble
    # LightGBM Model 1: Broad, leaf-wise
    lgb1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=60.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb1.fit(x_train_full, y_train, categorical_feature=encoder.categorical_indices)
    p1 = lgb1.predict_proba(x_eval_full)[:, 1]

    # LightGBM Model 2: Deeper, regularized with conservative sampling
    lgb2 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=45,
        min_child_samples=180,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=3.0,
        reg_alpha=0.5,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    lgb2.fit(x_train_full, y_train, categorical_feature=encoder.categorical_indices)
    p2 = lgb2.predict_proba(x_eval_full)[:, 1]

    # XGBoost Histogram Model for Structural Diversity
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.80,
            colsample_bytree=0.55,
            reg_lambda=2.0,
            reg_alpha=0.2,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        xgb_model.fit(x_train_full, y_train)
        p_xgb = xgb_model.predict_proba(x_eval_full)[:, 1]
        final_preds = 0.45 * p1 + 0.35 * p2 + 0.20 * p_xgb
    except Exception:
        final_preds = 0.55 * p1 + 0.45 * p2

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
