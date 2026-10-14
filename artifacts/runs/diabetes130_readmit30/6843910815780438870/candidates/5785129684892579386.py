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
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols`."""
    import sys
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    # Retrieve toolkit helpers from globals or loaded module
    toolkit = sys.modules.get("clinical_toolkit")
    icd9_grp = globals().get("icd9_group") or (getattr(toolkit, "icd9_group", None) if toolkit else None)
    icd9_cm = globals().get("icd9_comorbidities") or (getattr(toolkit, "icd9_comorbidities", None) if toolkit else None)
    oof_te = globals().get("oof_target_encode") or (getattr(toolkit, "oof_target_encode", None) if toolkit else None)
    EncoderClass = globals().get("TabularEncoder") or (getattr(toolkit, "TabularEncoder", None) if toolkit else None)

    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])

    tr = dict(train)
    ev = dict(eval_cols)
    extra_num = []
    extra_cat = []

    def add_num(name: str, arr_tr, arr_ev):
        t = np.nan_to_num(np.asarray(arr_tr, dtype=np.float64), nan=np.nan, posinf=0.0, neginf=0.0)
        e = np.nan_to_num(np.asarray(arr_ev, dtype=np.float64), nan=np.nan, posinf=0.0, neginf=0.0)
        tr[name] = t
        ev[name] = e
        extra_num.append(name)

    def add_cat(name: str, arr_tr, arr_ev):
        tr[name] = np.asarray(arr_tr, dtype=object)
        ev[name] = np.asarray(arr_ev, dtype=object)
        extra_cat.append(name)

    # 1. Charlson Comorbidities & Multi-System Interactions
    if icd9_cm is not None:
        cm_tr = icd9_cm(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_ev = icd9_cm(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        for k, v in cm_tr.items():
            add_num(k, v, cm_ev[k])

        # Cardio-Renal-Metabolic (CRM) syndrome dynamics
        crm_tr = cm_tr["cm_chf"] * cm_tr["cm_renal"] * cm_tr["cm_diabetes_complicated"]
        crm_ev = cm_ev["cm_chf"] * cm_ev["cm_renal"] * cm_ev["cm_diabetes_complicated"]
        add_num("crm_triad", crm_tr, crm_ev)

        cardiorenal_tr = cm_tr["cm_chf"] * cm_tr["cm_renal"]
        cardiorenal_ev = cm_ev["cm_chf"] * cm_ev["cm_renal"]
        add_num("cardiorenal", cardiorenal_tr, cardiorenal_ev)

        vasc_tr = cm_tr["cm_mi"] + cm_tr["cm_pvd"] + cm_tr["cm_cerebrovascular"]
        vasc_ev = cm_ev["cm_mi"] + cm_ev["cm_pvd"] + cm_ev["cm_cerebrovascular"]
        add_num("vascular_burden", vasc_tr, vasc_ev)

        pulmo_renal_tr = cm_tr["cm_pulmonary"] * cm_tr["cm_renal"]
        pulmo_renal_ev = cm_ev["cm_pulmonary"] * cm_ev["cm_renal"]
        add_num("pulmo_renal", pulmo_renal_tr, pulmo_renal_ev)

    # 2. ICD-9 Chapter Mapping & Systemic Diversity
    if icd9_grp is not None:
        g1_tr = np.array([icd9_grp(c) for c in train["diag_1"]], dtype=object)
        g1_ev = np.array([icd9_grp(c) for c in eval_cols["diag_1"]], dtype=object)
        add_cat("diag_1_group", g1_tr, g1_ev)

        g2_tr = np.array([icd9_grp(c) for c in train["diag_2"]], dtype=object)
        g2_ev = np.array([icd9_grp(c) for c in eval_cols["diag_2"]], dtype=object)
        add_cat("diag_2_group", g2_tr, g2_ev)

        g3_tr = np.array([icd9_grp(c) for c in train["diag_3"]], dtype=object)
        g3_ev = np.array([icd9_grp(c) for c in eval_cols["diag_3"]], dtype=object)
        add_cat("diag_3_group", g3_tr, g3_ev)

        pair_tr = np.array([f"{a}_{b}" for a, b in zip(g1_tr, g2_tr)], dtype=object)
        pair_ev = np.array([f"{a}_{b}" for a, b in zip(g1_ev, g2_ev)], dtype=object)
        add_cat("diag_pair_group", pair_tr, pair_ev)

        # Multi-system diagnosis entropy / dispersion
        def system_diversity(a1, a2, a3):
            res = np.zeros(len(a1), dtype=np.float64)
            for i in range(len(a1)):
                active = {x for x in (a1[i], a2[i], a3[i]) if x not in ("missing", None, "")}
                res[i] = len(active)
            return res

        add_num("diag_system_diversity", system_diversity(g1_tr, g2_tr, g3_tr), system_diversity(g1_ev, g2_ev, g3_ev))
    else:
        pair_tr = np.array(["missing"] * n_tr, dtype=object)
        pair_ev = np.array(["missing"] * n_ev, dtype=object)

    # 3. Healthcare Utilization Intensity & Velocity
    def extract_utilization(d):
        inp = np.nan_to_num(d["number_inpatient"].astype(np.float64), nan=0.0)
        emg = np.nan_to_num(d["number_emergency"].astype(np.float64), nan=0.0)
        out = np.nan_to_num(d["number_outpatient"].astype(np.float64), nan=0.0)
        time_hosp = np.maximum(np.nan_to_num(d["time_in_hospital"].astype(np.float64), nan=1.0), 1.0)
        labs = np.nan_to_num(d["num_lab_procedures"].astype(np.float64), nan=0.0)
        procs = np.nan_to_num(d["num_procedures"].astype(np.float64), nan=0.0)
        meds = np.nan_to_num(d["num_medications"].astype(np.float64), nan=0.0)
        diags = np.maximum(np.nan_to_num(d["number_diagnoses"].astype(np.float64), nan=1.0), 1.0)

        total_prior = inp + emg + out
        has_prior_inp = (inp > 0).astype(np.float64)
        has_multi_inp = (inp >= 2).astype(np.float64)
        inp_ratio = inp / (total_prior + 1.0)
        emg_ratio = emg / (total_prior + 1.0)
        acute_score = 3.0 * inp + 1.5 * emg + 0.5 * out

        return {
            "total_prior_visits": total_prior,
            "has_prior_inp": has_prior_inp,
            "has_multi_inp": has_multi_inp,
            "inp_ratio": inp_ratio,
            "emg_ratio": emg_ratio,
            "acute_severity_score": acute_score,
            "log_acute_severity": np.log1p(acute_score),
            "labs_per_day": labs / time_hosp,
            "meds_per_day": meds / time_hosp,
            "procs_per_day": procs / time_hosp,
            "diags_per_day": diags / time_hosp,
            "meds_per_diag": meds / diags,
            "service_density": (labs + 4.0 * procs + 2.0 * meds) / time_hosp,
        }

    u_tr = extract_utilization(train)
    u_ev = extract_utilization(eval_cols)
    for k in u_tr:
        add_num(k, u_tr[k], u_ev[k])

    # 4. Pharmacological Regimen Complexity & Titration Volatility
    med_names = [
        c for c in task.categorical_columns if c in {
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        }
    ]

    def extract_meds(d):
        n = len(d["time_in_hospital"])
        act = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        dn = np.zeros(n, dtype=np.float64)
        st = np.zeros(n, dtype=np.float64)
        for m in med_names:
            col = d[m]
            act += ((col != "No") & (col is not None)).astype(np.float64)
            up += (col == "Up").astype(np.float64)
            dn += (col == "Down").astype(np.float64)
            st += (col == "Steady").astype(np.float64)

        ins = d.get("insulin")
        if ins is not None:
            ins_act = (ins != "No").astype(np.float64)
            ins_up = (ins == "Up").astype(np.float64)
            ins_dn = (ins == "Down").astype(np.float64)
            ins_st = (ins == "Steady").astype(np.float64)
        else:
            ins_act = np.zeros(n, dtype=np.float64)
            ins_up = np.zeros(n, dtype=np.float64)
            ins_dn = np.zeros(n, dtype=np.float64)
            ins_st = np.zeros(n, dtype=np.float64)

        oral_act = np.maximum(act - ins_act, 0.0)
        return {
            "active_med_count": act,
            "med_up_count": up,
            "med_down_count": dn,
            "med_steady_count": st,
            "dosage_volatility": up + dn,
            "insulin_active": ins_act,
            "insulin_up": ins_up,
            "insulin_down": ins_dn,
            "insulin_steady": ins_st,
            "oral_active_count": oral_act,
            "insulin_oral_combo": ins_act * (oral_act > 0).astype(np.float64),
        }

    m_tr = extract_meds(train)
    m_ev = extract_meds(eval_cols)
    for k in m_tr:
        add_num(k, m_tr[k], m_ev[k])

    # 5. Out-of-Fold Target Encodings
    def encode_target(tr_col, ev_col, smoothing=25.0, seed=42):
        if oof_te is not None:
            return oof_te(tr_col, y_train, ev_col, smoothing=smoothing, seed=seed)
        prior = float(np.mean(y_train))
        return np.full(len(tr_col), prior), np.full(len(ev_col), prior)

    for col_name, smooth in [
        ("diag_1", 25.0),
        ("diag_2", 25.0),
        ("diag_3", 25.0),
        ("discharge_disposition_id", 15.0),
        ("admission_source_id", 20.0),
        ("medical_specialty", 25.0),
    ]:
        te_tr, te_ev = encode_target(train[col_name], eval_cols[col_name], smoothing=smooth)
        add_num(f"te_{col_name}", te_tr, te_ev)

    te_pair_tr, te_pair_ev = encode_target(pair_tr, pair_ev, smoothing=25.0)
    add_num("te_diag_pair", te_pair_tr, te_pair_ev)

    age_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["age"], train["discharge_disposition_id"])], dtype=object)
    age_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["age"], eval_cols["discharge_disposition_id"])], dtype=object)
    te_ad_tr, te_ad_ev = encode_target(age_disch_tr, age_disch_ev, smoothing=20.0)
    add_num("te_age_discharge", te_ad_tr, te_ad_ev)

    # Demographic baseline risk and excess individual divergence
    demo_tr = np.array([f"{a}_{g}_{r}" for a, g, r in zip(train["age"], train["gender"], train["race"])], dtype=object)
    demo_ev = np.array([f"{a}_{g}_{r}" for a, g, r in zip(eval_cols["age"], eval_cols["gender"], eval_cols["race"])], dtype=object)
    te_demo_tr, te_demo_ev = encode_target(demo_tr, demo_ev, smoothing=35.0)
    add_num("te_demographic_risk", te_demo_tr, te_demo_ev)
    add_num("excess_individual_risk", u_tr["acute_severity_score"] * te_demo_tr, u_ev["acute_severity_score"] * te_demo_ev)

    # 6. Unsupervised Clinical Phenotype Archetype Projection (PARF)
    archetype_keys = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_inpatient", "number_emergency", "number_diagnoses",
        "active_med_count", "acute_severity_score"
    ]
    if "cm_count" in tr:
        archetype_keys.append("cm_count")

    X_arch_tr = np.column_stack([np.nan_to_num(tr[k].astype(np.float64), nan=0.0) for k in archetype_keys])
    X_arch_ev = np.column_stack([np.nan_to_num(ev[k].astype(np.float64), nan=0.0) for k in archetype_keys])

    scaler = StandardScaler()
    X_scaled_tr = scaler.fit_transform(X_arch_tr)
    X_scaled_ev = scaler.transform(X_arch_ev)

    n_archetypes = 6
    kmeans = MiniBatchKMeans(n_clusters=n_archetypes, random_state=42, batch_size=1024, n_init=3)
    kmeans.fit(X_scaled_tr)

    dists_tr = kmeans.transform(X_scaled_tr)
    dists_ev = kmeans.transform(X_scaled_ev)
    for c_i in range(n_archetypes):
        add_num(f"archetype_dist_{c_i}", dists_tr[:, c_i], dists_ev[:, c_i])

    clust_tr = kmeans.predict(X_scaled_tr).astype(str)
    clust_ev = kmeans.predict(X_scaled_ev).astype(str)
    te_cl_tr, te_cl_ev = encode_target(clust_tr, clust_ev, smoothing=20.0)
    add_num("te_phenotype_archetype", te_cl_tr, te_cl_ev)

    # 7. Tabular Encoding
    encoder = EncoderClass(min_count=20)
    x_train = encoder.fit_transform(task, tr, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(ev)
    cat_indices = encoder.categorical_indices

    # 8. Diverse Gradient Boosting Ensemble
    # LightGBM Model 1: Balanced capacity
    m1 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.026,
        num_leaves=31,
        max_depth=6,
        min_child_samples=120,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.15,
        reg_lambda=1.5,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # LightGBM Model 2: Deeper, fine-grained interactions
    m2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.028,
        num_leaves=46,
        max_depth=7,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.4,
        reg_lambda=3.0,
        cat_smooth=45.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # LightGBM Model 3: Shallow, conservative regularizer
    m3 = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.022,
        num_leaves=18,
        max_depth=5,
        min_child_samples=220,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=1.0,
        cat_smooth=25.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # XGBoost Model: Histogram-based tree splits
    xgb_preds = None
    xgb_mod = globals().get("xgb") or globals().get("xgboost")
    if xgb_mod is not None and hasattr(xgb_mod, "XGBClassifier"):
        try:
            m_xgb = xgb_mod.XGBClassifier(
                n_estimators=240,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.6,
                reg_alpha=0.2,
                reg_lambda=2.0,
                tree_method="hist",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            m_xgb.fit(x_train, y_train)
            xgb_preds = m_xgb.predict_proba(x_eval)[:, 1]
        except Exception:
            xgb_preds = None

    if xgb_preds is not None:
        final_preds = 0.32 * p1 + 0.28 * p2 + 0.25 * p3 + 0.15 * xgb_preds
    else:
        final_preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3

    final_preds = np.nan_to_num(final_preds, nan=0.114)
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
