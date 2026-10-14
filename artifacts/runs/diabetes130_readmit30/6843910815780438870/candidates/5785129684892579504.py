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
        insulin_up = (insulin_col == "Up").astype(np.float64)
        insulin_down = (insulin_col == "Down").astype(np.float64)
        insulin_steady = (insulin_col == "Steady").astype(np.float64)
        insulin_active = (insulin_col != "No").astype(np.float64)
        insulin_num = np.zeros(n_rows, dtype=np.float64)
        insulin_num[insulin_col == "Steady"] = 1.0
        insulin_num[insulin_col == "Down"] = 2.0
        insulin_num[insulin_col == "Up"] = 3.0

        metf_col = data_dict.get("metformin", np.array(["No"] * n_rows))
        metf_active = ((metf_col == "Steady") | (metf_col == "Up") | (metf_col == "Down")).astype(np.float64)

        sulfonylureas = [
            "glimepiride", "glipizide", "glyburide", "tolbutamide",
            "acetohexamide", "chlorpropamide", "tolazamide",
        ]
        has_su = np.zeros(n_rows, dtype=np.float64)
        for s in sulfonylureas:
            if s in data_dict:
                c = data_dict[s]
                has_su = np.maximum(has_su, np.isin(c, ["Steady", "Up", "Down"]).astype(np.float64))

        tzds = ["pioglitazone", "rosiglitazone"]
        has_tzd = np.zeros(n_rows, dtype=np.float64)
        for t in tzds:
            if t in data_dict:
                c = data_dict[t]
                has_tzd = np.maximum(has_tzd, np.isin(c, ["Steady", "Up", "Down"]).astype(np.float64))

        volatility = change_cnt / (active_cnt + 1.0)
        complex_regimen = ((insulin_active > 0) & (active_cnt >= 2)).astype(np.float64)

        return np.column_stack([
            active_cnt, change_cnt, up_cnt, down_cnt, insulin_num,
            insulin_up, insulin_down, insulin_steady, insulin_active,
            metf_active, has_su, has_tzd, complex_regimen, volatility
        ])

    med_train = extract_med_features(train, n_train)
    med_eval = extract_med_features(eval_cols, n_eval)

    # 3. Clinical Healthcare Velocity, Inertia & Intensity Metrics
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

        log_inp = np.log1p(num_inp)
        log_emg = np.log1p(num_emg)
        log_out = np.log1p(num_out)

        total_prior = num_inp + num_emg + num_out
        log_total_prior = np.log1p(total_prior)
        acute_prior = num_inp + num_emg
        acute_weighted = 2.0 * num_inp + num_emg
        has_prior_inp = (num_inp > 0).astype(np.float64)
        has_prior_emg = (num_emg > 0).astype(np.float64)
        frequent_inp = (num_inp >= 2).astype(np.float64)
        super_utilizer = (num_inp >= 3).astype(np.float64)

        inp_ratio = num_inp / (total_prior + 1.0)
        emg_ratio = num_emg / (total_prior + 1.0)
        acute_ratio = acute_prior / (total_prior + 1.0)

        meds_per_day = n_meds / (time_hosp + 0.1)
        labs_per_day = n_labs / (time_hosp + 0.1)
        procs_per_day = n_procs / (time_hosp + 0.1)
        diags_per_day = n_diags / (time_hosp + 0.1)
        proc_med_ratio = n_procs / (n_meds + 1.0)
        complexity_index = np.log1p(n_meds) * np.log1p(n_labs) * np.log1p(time_hosp)
        high_lab_intensity = (labs_per_day > 15.0).astype(np.float64)
        no_procedures = (n_procs == 0).astype(np.float64)

        raw_age = data_dict.get("age", np.array(["[60-70)"] * n_rows))
        age_num = np.array([age_map.get(str(a), 60.0) for a in raw_age], dtype=np.float64)
        elderly = (age_num >= 70.0).astype(np.float64)
        elderly_fragile = ((age_num >= 70.0) & (num_inp > 0)).astype(np.float64)
        age_inp_interaction = age_num * num_inp
        age_stay_interaction = age_num * time_hosp

        disch = data_dict.get("discharge_disposition_id", np.array(["1"] * n_rows))
        disch_home = (disch == "1").astype(np.float64)
        disch_snf = np.isin(disch, ["2", "3", "4", "5", "22", "23", "24"]).astype(np.float64)
        disch_homehealth = (disch == "6").astype(np.float64)
        disch_ama = (disch == "7").astype(np.float64)
        disch_other_high_risk = np.isin(disch, ["8", "15", "28"]).astype(np.float64)

        adm_src = data_dict.get("admission_source_id", np.array(["1"] * n_rows))
        adm_emergency = (adm_src == "7").astype(np.float64)
        adm_transfer = np.isin(adm_src, ["4", "5", "6"]).astype(np.float64)

        adm_typ = data_dict.get("admission_type_id", np.array(["1"] * n_rows))
        admtype_emerg = (adm_typ == "1").astype(np.float64)
        admtype_urgent = (adm_typ == "2").astype(np.float64)
        admtype_elect = (adm_typ == "3").astype(np.float64)
        adm_non_elective = np.isin(adm_typ, ["1", "2"]).astype(np.float64)

        glu_col = data_dict.get("max_glu_serum", np.array(["None"] * n_rows))
        glu_tested = (glu_col != "None").astype(np.float64)
        glu_high = np.isin(glu_col, [">200", ">300"]).astype(np.float64)

        a1c_col = data_dict.get("A1Cresult", np.array(["None"] * n_rows))
        a1c_tested = (a1c_col != "None").astype(np.float64)
        a1c_high = np.isin(a1c_col, [">7", ">8"]).astype(np.float64)
        a1c_uncontrolled = (a1c_col == ">8").astype(np.float64)

        chg_col = data_dict.get("change", np.array(["No"] * n_rows))
        chg_flag = (chg_col == "Ch").astype(np.float64)

        dm_col = data_dict.get("diabetesMed", np.array(["No"] * n_rows))
        dm_flag = (dm_col == "Yes").astype(np.float64)

        # Clinical Inertia: High hyperglycemia without therapy escalation
        a1c_high_no_change = a1c_uncontrolled * (1.0 - chg_flag)
        glu_high_no_change = glu_high * (1.0 - chg_flag)

        weight_col = data_dict.get("weight", np.array([None] * n_rows))
        weight_present = (~np.isin(weight_col, [None, "?", "None"])).astype(np.float64)

        payer_col = data_dict.get("payer_code", np.array([None] * n_rows))
        payer_missing = np.isin(payer_col, [None, "?", "None"]).astype(np.float64)

        spec_col = data_dict.get("medical_specialty", np.array([None] * n_rows))
        spec_missing = np.isin(spec_col, [None, "?", "None"]).astype(np.float64)

        return np.column_stack([
            total_prior, log_total_prior, acute_prior, acute_weighted,
            log_inp, log_emg, log_out,
            has_prior_inp, has_prior_emg, frequent_inp, super_utilizer,
            inp_ratio, emg_ratio, acute_ratio, meds_per_day, labs_per_day,
            procs_per_day, diags_per_day, proc_med_ratio, complexity_index,
            high_lab_intensity, no_procedures,
            age_num, elderly, elderly_fragile, age_inp_interaction, age_stay_interaction,
            disch_home, disch_snf, disch_homehealth, disch_ama, disch_other_high_risk,
            adm_emergency, adm_transfer, admtype_emerg, admtype_urgent, admtype_elect,
            adm_non_elective,
            glu_tested, glu_high, a1c_tested, a1c_high, a1c_uncontrolled,
            chg_flag, dm_flag, a1c_high_no_change, glu_high_no_change,
            weight_present, payer_missing, spec_missing,
        ])

    metrics_train = extract_clinical_metrics(train, n_train)
    metrics_eval = extract_clinical_metrics(eval_cols, n_eval)

    # 4. Comorbidities & Interaction Extraction
    if _icd9_comorbidities is not None:
        cm_tr = _icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_ev = _icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        cm_keys = sorted(cm_tr.keys())
        cm_mat_tr = np.column_stack([cm_tr[k].astype(np.float64) for k in cm_keys])
        cm_mat_ev = np.column_stack([cm_ev[k].astype(np.float64) for k in cm_keys])

        def get_cm_interactions(cm_dict, num_inp, raw_age):
            age_n = np.array([age_map.get(str(a), 60.0) for a in raw_age], dtype=np.float64)
            chf = cm_dict.get("cm_chf", np.zeros_like(num_inp))
            renal = cm_dict.get("cm_renal", np.zeros_like(num_inp))
            pulm = cm_dict.get("cm_pulmonary", np.zeros_like(num_inp))
            cnt = cm_dict.get("cm_count", np.zeros_like(num_inp))
            cardiorenal = ((chf > 0) & (renal > 0)).astype(np.float64)
            chf_inp = chf * num_inp
            renal_inp = renal * num_inp
            chf_elderly = chf * (age_n >= 70.0).astype(np.float64)
            pulm_elderly = pulm * (age_n >= 70.0).astype(np.float64)
            multimorbid = (cnt >= 3.0).astype(np.float64)
            return np.column_stack([cardiorenal, chf_inp, renal_inp, chf_elderly, pulm_elderly, multimorbid])

        inp_tr = np.nan_to_num(train.get("number_inpatient", np.zeros(n_train)), nan=0.0)
        inp_ev = np.nan_to_num(eval_cols.get("number_inpatient", np.zeros(n_eval)), nan=0.0)
        cm_int_tr = get_cm_interactions(cm_tr, inp_tr, train.get("age", np.array(["[60-70)"] * n_train)))
        cm_int_ev = get_cm_interactions(cm_ev, inp_ev, eval_cols.get("age", np.array(["[60-70)"] * n_eval)))
        cm_mat_tr = np.column_stack([cm_mat_tr, cm_int_tr])
        cm_mat_ev = np.column_stack([cm_mat_ev, cm_int_ev])
    else:
        cm_mat_tr = np.zeros((n_train, 0), dtype=np.float64)
        cm_mat_ev = np.zeros((n_eval, 0), dtype=np.float64)

    # 5. Multi-Resolution Hierarchical Out-of-Fold Target Encodings
    def safe_te(tr_col, y_tr, ev_col, smoothing=25.0, seed=42):
        if _oof_target_encode is not None:
            return _oof_target_encode(tr_col, y_tr, ev_col, smoothing=smoothing, seed=seed)
        prior = float(np.mean(y_tr))
        tr_s = np.array([str(x) for x in tr_col], dtype=object)
        ev_s = np.array([str(x) for x in ev_col], dtype=object)
        tr_out = np.full(len(tr_s), prior, dtype=np.float64)
        ev_out = np.full(len(ev_s), prior, dtype=np.float64)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in kf.split(tr_s):
            f_tr, f_y = tr_s[tr_idx], y_tr[tr_idx]
            for val in np.unique(f_tr):
                m = (f_tr == val)
                smoothed = (np.sum(f_y[m]) + smoothing * prior) / (np.sum(m) + smoothing)
                tr_out[va_idx[tr_s[va_idx] == val]] = smoothed
        for val in np.unique(tr_s):
            m = (tr_s == val)
            ev_out[ev_s == val] = (np.sum(y_tr[m]) + smoothing * prior) / (np.sum(m) + smoothing)
        return tr_out, ev_out

    # 3-digit diagnosis prefixes and chapter groups
    def get_diag_3d(col):
        return np.array([str(c).strip()[:3] if c is not None and str(c) != "None" else "missing" for c in col], dtype=object)

    def get_diag_grp(col):
        if _icd9_group is not None:
            return np.array([_icd9_group(str(c).strip()) if c is not None and str(c) != "None" else "missing" for c in col], dtype=object)
        return get_diag_3d(col)

    te_features_tr, te_features_ev = [], []

    # Standard nominal columns
    te_std_cols = [
        "discharge_disposition_id", "admission_source_id", "admission_type_id",
        "medical_specialty", "payer_code", "race", "age",
        "diag_1", "diag_2", "diag_3"
    ]
    for c in te_std_cols:
        tr_val = train[c]
        ev_val = eval_cols[c]
        tr_enc, ev_enc = safe_te(tr_val, y_train, ev_val, smoothing=30.0, seed=42)
        te_features_tr.append(tr_enc.astype(np.float64))
        te_features_ev.append(ev_enc.astype(np.float64))

    # Hierarchical 3-digit categories and groups
    for c in ["diag_1", "diag_2", "diag_3"]:
        tr_3d = get_diag_3d(train[c])
        ev_3d = get_diag_3d(eval_cols[c])
        tr_enc, ev_enc = safe_te(tr_3d, y_train, ev_3d, smoothing=20.0, seed=101)
        te_features_tr.append(tr_enc.astype(np.float64))
        te_features_ev.append(ev_enc.astype(np.float64))

        tr_grp = get_diag_grp(train[c])
        ev_grp = get_diag_grp(eval_cols[c])
        tr_enc_g, ev_enc_g = safe_te(tr_grp, y_train, ev_grp, smoothing=15.0, seed=202)
        te_features_tr.append(tr_enc_g.astype(np.float64))
        te_features_ev.append(ev_enc_g.astype(np.float64))

    # Cross interaction target encodings
    disch_adm_tr = np.array([f"{d}_{a}" for d, a in zip(train["discharge_disposition_id"], train["admission_source_id"])], dtype=object)
    disch_adm_ev = np.array([f"{d}_{a}" for d, a in zip(eval_cols["discharge_disposition_id"], eval_cols["admission_source_id"])], dtype=object)
    tr_enc, ev_enc = safe_te(disch_adm_tr, y_train, disch_adm_ev, smoothing=20.0, seed=303)
    te_features_tr.append(tr_enc.astype(np.float64))
    te_features_ev.append(ev_enc.astype(np.float64))

    disch_age_tr = np.array([f"{d}_{a}" for d, a in zip(train["discharge_disposition_id"], train["age"])], dtype=object)
    disch_age_ev = np.array([f"{d}_{a}" for d, a in zip(eval_cols["discharge_disposition_id"], eval_cols["age"])], dtype=object)
    tr_enc, ev_enc = safe_te(disch_age_tr, y_train, disch_age_ev, smoothing=20.0, seed=404)
    te_features_tr.append(tr_enc.astype(np.float64))
    te_features_ev.append(ev_enc.astype(np.float64))

    te_mat_tr = np.column_stack(te_features_tr)
    te_mat_ev = np.column_stack(te_features_ev)

    # 6. Bag-of-Diagnoses LSA & Out-of-Fold Linear Stacking
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
                toks.append(f"icd_{s1[:3]}")
                if _icd9_group is not None:
                    toks.append(f"grp1_{_icd9_group(s1)}")
            if c2 is not None and str(c2) != "None":
                s2 = str(c2).strip()
                toks.append(f"d2_{s2}")
                toks.append(f"icd_{s2[:3]}")
                if _icd9_group is not None:
                    toks.append(f"grp2_{_icd9_group(s2)}")
            if c3 is not None and str(c3) != "None":
                s3 = str(c3).strip()
                toks.append(f"d3_{s3}")
                toks.append(f"icd_{s3[:3]}")
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

    tfidf = TfidfVectorizer(token_pattern=r"\S+", min_df=2, max_features=600, sublinear_tf=True)
    X_tfidf_tr = tfidf.fit_transform(tr_docs)
    X_tfidf_ev = tfidf.transform(ev_docs)

    n_components = min(10, max(2, X_tfidf_tr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_tr = svd.fit_transform(X_tfidf_tr)
    svd_ev = svd.transform(X_tfidf_ev)

    # Out-of-fold regularized linear classifier on TF-IDF token space
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    text_risk_tr = np.zeros(n_train, dtype=np.float64)
    text_risk_ev = np.zeros(n_eval, dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_tfidf_tr, y_train):
        clf = RidgeClassifier(alpha=8.0, random_state=42)
        clf.fit(X_tfidf_tr[tr_idx], y_train[tr_idx])
        text_risk_tr[va_idx] = clf.decision_function(X_tfidf_tr[va_idx])
        text_risk_ev += clf.decision_function(X_tfidf_ev) / 5.0

    text_prob_tr = 1.0 / (1.0 + np.exp(-np.clip(text_risk_tr, -10.0, 10.0)))
    text_prob_ev = 1.0 / (1.0 + np.exp(-np.clip(text_risk_ev, -10.0, 10.0)))

    # Extra features matrix
    extra_train = np.column_stack([
        med_train, metrics_train, cm_mat_tr, te_mat_tr, svd_tr, text_prob_tr
    ])
    extra_eval = np.column_stack([
        med_eval, metrics_eval, cm_mat_ev, te_mat_ev, svd_ev, text_prob_ev
    ])

    # Feature sets for models
    # LightGBM uses base tabular encoding + extra features with native categoricals
    x_train_lgb = np.hstack([x_train_base, extra_train])
    x_eval_lgb = np.hstack([x_eval_base, extra_eval])

    # XGBoost uses true numerics + extra features (avoiding arbitrary ordinal splits on category integers)
    cat_set = set(encoder.categorical_indices)
    num_indices = [i for i in range(x_train_base.shape[1]) if i not in cat_set]
    x_train_xgb = np.column_stack([x_train_base[:, num_indices], extra_train])
    x_eval_xgb = np.column_stack([x_eval_base[:, num_indices], extra_eval])

    # 7. Heterogeneous GBDT Ensemble
    # LightGBM Model 1: Broad, leaf-wise
    lgb1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.022,
        num_leaves=35,
        min_child_samples=110,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.62,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb1.fit(x_train_lgb, y_train, categorical_feature=encoder.categorical_indices)
    p1 = lgb1.predict_proba(x_eval_lgb)[:, 1]

    # LightGBM Model 2: Deeper, heavily regularized
    lgb2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.019,
        num_leaves=52,
        min_child_samples=160,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=4.0,
        reg_alpha=0.5,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    lgb2.fit(x_train_lgb, y_train, categorical_feature=encoder.categorical_indices)
    p2 = lgb2.predict_proba(x_eval_lgb)[:, 1]

    # XGBoost Histogram Model on clean continuous + target-encoded features
    from scipy.stats import rankdata

    def to_percentile(arr):
        n = len(arr)
        return (rankdata(arr) - 0.5) / n

    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=240,
            learning_rate=0.028,
            max_depth=5,
            subsample=0.80,
            colsample_bytree=0.55,
            reg_lambda=3.0,
            reg_alpha=0.3,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        xgb_model.fit(x_train_xgb, y_train)
        p_xgb = xgb_model.predict_proba(x_eval_xgb)[:, 1]

        r1 = to_percentile(p1)
        r2 = to_percentile(p2)
        r_xgb = to_percentile(p_xgb)

        r_ens = 0.42 * r1 + 0.38 * r2 + 0.20 * r_xgb
        p_raw = 0.42 * p1 + 0.38 * p2 + 0.20 * p_xgb
    except Exception:
        r1 = to_percentile(p1)
        r2 = to_percentile(p2)
        r_ens = 0.55 * r1 + 0.45 * r2
        p_raw = 0.55 * p1 + 0.45 * p2

    # Calibrate ensemble ranks back to exact probability distribution
    final_ranks = to_percentile(r_ens)
    final_preds = np.quantile(p_raw, final_ranks)

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
