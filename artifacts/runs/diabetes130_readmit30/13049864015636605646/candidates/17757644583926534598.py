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
    icd9_comorb_fn = globals().get("icd9_comorbidities")
    icd9_group_fn = globals().get("icd9_group")
    target_enc_fn = globals().get("oof_target_encode")

    if icd9_comorb_fn is None or icd9_group_fn is None or target_enc_fn is None:
        try:
            import clinical_toolkit

            if icd9_comorb_fn is None:
                icd9_comorb_fn = getattr(clinical_toolkit, "icd9_comorbidities", None)
            if icd9_group_fn is None:
                icd9_group_fn = getattr(clinical_toolkit, "icd9_group", None)
            if target_enc_fn is None:
                target_enc_fn = getattr(clinical_toolkit, "oof_target_encode", None)
        except Exception:
            pass

    # Fallback target encoder if needed
    if target_enc_fn is None:
        def fallback_te(tr_c, y_tr, ev_c, n_folds=5, smoothing=20.0, seed=0):
            from sklearn.model_selection import KFold
            prior = float(np.mean(y_tr))
            tr_res = np.full(len(tr_c), prior, dtype=np.float64)
            ev_res = np.full(len(ev_c), prior, dtype=np.float64)
            vc, vs = {}, {}
            for v, y in zip(tr_c, y_tr):
                k = str(v)
                vc[k] = vc.get(k, 0) + 1
                vs[k] = vs.get(k, 0.0) + y
            for i, v in enumerate(ev_c):
                k = str(v)
                c = vc.get(k, 0)
                ev_res[i] = (vs.get(k, 0.0) + smoothing * prior) / (c + smoothing)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for tr_idx, va_idx in kf.split(tr_c):
                f_vc, f_vs = {}, {}
                for idx in tr_idx:
                    k = str(tr_c[idx])
                    f_vc[k] = f_vc.get(k, 0) + 1
                    f_vs[k] = f_vs.get(k, 0.0) + y_tr[idx]
                for idx in va_idx:
                    k = str(tr_c[idx])
                    c = f_vc.get(k, 0)
                    tr_res[idx] = (f_vs.get(k, 0.0) + smoothing * prior) / (c + smoothing)
            return tr_res, ev_res
        target_enc_fn = fallback_te

    # Fallback icd9_group if needed
    if icd9_group_fn is None:
        def fallback_grp(code):
            if code is None or code in ("?", "None", ""):
                return "missing"
            c = str(code).strip()
            if c.startswith(("V", "E")):
                return "other"
            try:
                num = float(c)
            except Exception:
                return "other"
            if 390 <= num <= 459 or num == 785:
                return "circulatory"
            elif 460 <= num <= 519 or num == 786:
                return "respiratory"
            elif 520 <= num <= 579 or num == 787:
                return "digestive"
            elif 250 <= num < 251:
                return "diabetes"
            elif 800 <= num <= 999:
                return "injury"
            elif 710 <= num <= 739:
                return "musculoskeletal"
            elif 580 <= num <= 629 or num == 788:
                return "genitourinary"
            elif 140 <= num <= 239:
                return "neoplasms"
            return "other"
        icd9_group_fn = fallback_grp

    def extract_engineered_features(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        n = len(data_dict["time_in_hospital"])

        # Prior visits
        n_inpatient = np.nan_to_num(data_dict.get("number_inpatient", np.zeros(n)), nan=0.0)
        n_emergency = np.nan_to_num(data_dict.get("number_emergency", np.zeros(n)), nan=0.0)
        n_outpatient = np.nan_to_num(data_dict.get("number_outpatient", np.zeros(n)), nan=0.0)

        tot_visits = n_inpatient + n_emergency + n_outpatient
        res["total_prior_visits"] = tot_visits
        res["acute_prior_visits"] = n_inpatient + n_emergency
        res["has_prior_inpatient"] = (n_inpatient > 0).astype(np.float64)
        res["has_prior_emergency"] = (n_emergency > 0).astype(np.float64)
        res["has_prior_outpatient"] = (n_outpatient > 0).astype(np.float64)
        res["multiple_prior_inpatient"] = (n_inpatient >= 2).astype(np.float64)
        res["inpatient_ratio"] = (n_inpatient + 0.1) / (tot_visits + 1.0)
        res["emergency_ratio"] = (n_emergency + 0.1) / (tot_visits + 1.0)

        # Inpatient stay intensity
        los = np.nan_to_num(data_dict.get("time_in_hospital", np.ones(n)), nan=1.0)
        n_meds = np.nan_to_num(data_dict.get("num_medications", np.zeros(n)), nan=0.0)
        n_labs = np.nan_to_num(data_dict.get("num_lab_procedures", np.zeros(n)), nan=0.0)
        n_procs = np.nan_to_num(data_dict.get("num_procedures", np.zeros(n)), nan=0.0)
        n_diags = np.nan_to_num(data_dict.get("number_diagnoses", np.ones(n)), nan=1.0)

        res["med_per_day"] = n_meds / (los + 0.5)
        res["lab_per_day"] = n_labs / (los + 0.5)
        res["proc_per_day"] = n_procs / (los + 0.5)
        res["total_procedures"] = n_labs + n_procs
        res["meds_per_diag"] = n_meds / (n_diags + 0.5)
        res["labs_per_diag"] = n_labs / (n_diags + 0.5)
        res["los_times_inpatient"] = los * (n_inpatient + 1.0)
        res["los_times_diags"] = los * (n_diags + 1.0)

        # Age conversion
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
        if "age" in data_dict:
            age_arr = data_dict["age"]
            age_num = np.array([age_map.get(a, np.nan) for a in age_arr], dtype=np.float64)
            res["age_num"] = age_num
            res["is_elderly"] = (age_num >= 65.0).astype(np.float64)
            res["is_very_elderly"] = (age_num >= 75.0).astype(np.float64)
            res["age_times_inpatient"] = np.nan_to_num(age_num, nan=65.0) * (n_inpatient + 1.0)

        # Discharge disposition flags
        disp = data_dict.get("discharge_disposition_id")
        if disp is not None:
            res["disp_home"] = (disp == "1").astype(np.float64)
            res["disp_snf"] = (disp == "3").astype(np.float64)
            res["disp_home_health"] = (disp == "6").astype(np.float64)
            res["disp_rehab"] = (disp == "22").astype(np.float64)

        # Admission flags
        adm_t = data_dict.get("admission_type_id")
        if adm_t is not None:
            res["adm_emergency"] = (adm_t == "1").astype(np.float64)
            res["adm_urgent"] = (adm_t == "2").astype(np.float64)
            res["adm_elective"] = (adm_t == "3").astype(np.float64)

        adm_s = data_dict.get("admission_source_id")
        if adm_s is not None:
            res["adm_from_er"] = (adm_s == "7").astype(np.float64)

        # Medications
        med_list = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]
        num_act = np.zeros(n, dtype=np.float64)
        num_up = np.zeros(n, dtype=np.float64)
        num_down = np.zeros(n, dtype=np.float64)
        num_steady = np.zeros(n, dtype=np.float64)

        for m in med_list:
            if m in data_dict:
                arr = data_dict[m]
                num_act += ((arr != "No") & (arr != None)).astype(np.float64)
                num_up += (arr == "Up").astype(np.float64)
                num_down += (arr == "Down").astype(np.float64)
                num_steady += (arr == "Steady").astype(np.float64)

        res["med_num_active"] = num_act
        res["med_num_up"] = num_up
        res["med_num_down"] = num_down
        res["med_num_steady"] = num_steady
        res["med_num_changed"] = num_up + num_down

        if "insulin" in data_dict:
            ins = data_dict["insulin"]
            ins_act = ((ins != "No") & (ins != None)).astype(np.float64)
            res["insulin_active"] = ins_act
            res["insulin_up"] = (ins == "Up").astype(np.float64)
            res["insulin_down"] = (ins == "Down").astype(np.float64)
            res["insulin_steady"] = (ins == "Steady").astype(np.float64)
            res["insulin_plus_oral"] = ((ins_act > 0) & ((num_act - ins_act) > 0)).astype(np.float64)

        if "change" in data_dict:
            res["has_change"] = (data_dict["change"] == "Ch").astype(np.float64)
        if "diabetesMed" in data_dict:
            res["has_diabetes_med"] = (data_dict["diabetesMed"] == "Yes").astype(np.float64)

        # Labs
        if "A1Cresult" in data_dict:
            a1c = data_dict["A1Cresult"]
            res["a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
            res["a1c_high"] = np.isin(a1c, [">8", ">7"]).astype(np.float64)
        if "max_glu_serum" in data_dict:
            glu = data_dict["max_glu_serum"]
            res["glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
            res["glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

        # Missingness
        if "weight" in data_dict:
            res["weight_missing"] = np.isin(data_dict["weight"], [None, "?", "None"]).astype(np.float64)
        if "medical_specialty" in data_dict:
            res["spec_missing"] = np.isin(data_dict["medical_specialty"], [None, "?", "None"]).astype(np.float64)
        if "payer_code" in data_dict:
            res["payer_missing"] = np.isin(data_dict["payer_code"], [None, "?", "None"]).astype(np.float64)

        return res

    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities
    if icd9_comorb_fn is not None and "diag_1" in train and "diag_2" in train and "diag_3" in train:
        cm_tr = icd9_comorb_fn(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_ev = icd9_comorb_fn(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        for k in cm_tr:
            train_dict[k] = cm_tr[k]
            eval_dict[k] = cm_ev[k]
            extra_numeric.append(k)

        if "cm_chf" in cm_tr and "cm_renal" in cm_tr and "cm_diabetes_complicated" in cm_tr:
            train_dict["cardiorenal_risk"] = cm_tr["cm_chf"] + cm_tr["cm_renal"] + cm_tr["cm_diabetes_complicated"]
            eval_dict["cardiorenal_risk"] = cm_ev["cm_chf"] + cm_ev["cm_renal"] + cm_ev["cm_diabetes_complicated"]
            extra_numeric.append("cardiorenal_risk")

        if "cm_mi" in cm_tr and "cm_pvd" in cm_tr and "cm_cerebrovascular" in cm_tr:
            train_dict["vascular_risk"] = cm_tr["cm_mi"] + cm_tr["cm_pvd"] + cm_tr["cm_cerebrovascular"]
            eval_dict["vascular_risk"] = cm_ev["cm_mi"] + cm_ev["cm_pvd"] + cm_ev["cm_cerebrovascular"]
            extra_numeric.append("vascular_risk")

    # 2. ICD-9 Chapter groups
    if "diag_1" in train and "diag_2" in train and "diag_3" in train:
        g1_tr = np.array([icd9_group_fn(c) for c in train["diag_1"]], dtype=object)
        g2_tr = np.array([icd9_group_fn(c) for c in train["diag_2"]], dtype=object)
        g3_tr = np.array([icd9_group_fn(c) for c in train["diag_3"]], dtype=object)

        g1_ev = np.array([icd9_group_fn(c) for c in eval_cols["diag_1"]], dtype=object)
        g2_ev = np.array([icd9_group_fn(c) for c in eval_cols["diag_2"]], dtype=object)
        g3_ev = np.array([icd9_group_fn(c) for c in eval_cols["diag_3"]], dtype=object)

        train_dict["diag1_group"] = g1_tr
        train_dict["diag2_group"] = g2_tr
        train_dict["diag3_group"] = g3_tr
        eval_dict["diag1_group"] = g1_ev
        eval_dict["diag2_group"] = g2_ev
        eval_dict["diag3_group"] = g3_ev
        extra_categorical.extend(["diag1_group", "diag2_group", "diag3_group"])

        train_dict["circulatory_count"] = ((g1_tr == "circulatory") | (g2_tr == "circulatory") | (g3_tr == "circulatory")).astype(np.float64)
        eval_dict["circulatory_count"] = ((g1_ev == "circulatory") | (g2_ev == "circulatory") | (g3_ev == "circulatory")).astype(np.float64)
        extra_numeric.append("circulatory_count")

        train_dict["respiratory_count"] = ((g1_tr == "respiratory") | (g2_tr == "respiratory") | (g3_tr == "respiratory")).astype(np.float64)
        eval_dict["respiratory_count"] = ((g1_ev == "respiratory") | (g2_ev == "respiratory") | (g3_ev == "respiratory")).astype(np.float64)
        extra_numeric.append("respiratory_count")

    # 3. Domain features
    dom_tr = extract_engineered_features(train)
    dom_ev = extract_engineered_features(eval_cols)
    for k in dom_tr:
        train_dict[k] = dom_tr[k]
        eval_dict[k] = dom_ev[k]
        extra_numeric.append(k)

    # 4. Out-of-fold target encoding
    te_specs = [
        ("diag_1", 25.0),
        ("diag_2", 25.0),
        ("diag_3", 25.0),
        ("discharge_disposition_id", 20.0),
        ("admission_source_id", 20.0),
        ("admission_type_id", 20.0),
        ("medical_specialty", 25.0),
        ("payer_code", 25.0),
    ]
    for col_n, sm in te_specs:
        if col_n in train and col_n in eval_cols:
            tr_te, ev_te = target_enc_fn(
                train[col_n], y_train, eval_cols[col_n], n_folds=5, smoothing=sm, seed=42
            )
            te_k = f"te_{col_n}"
            train_dict[te_k] = tr_te
            eval_dict[te_k] = ev_te
            extra_numeric.append(te_k)

    if "te_diag_1" in train_dict and "te_diag_2" in train_dict and "te_diag_3" in train_dict:
        train_dict["te_diag_max"] = np.maximum(train_dict["te_diag_1"], np.maximum(train_dict["te_diag_2"], train_dict["te_diag_3"]))
        eval_dict["te_diag_max"] = np.maximum(eval_dict["te_diag_1"], np.maximum(eval_dict["te_diag_2"], eval_dict["te_diag_3"]))
        extra_numeric.append("te_diag_max")

        train_dict["te_diag_mean"] = (train_dict["te_diag_1"] + train_dict["te_diag_2"] + train_dict["te_diag_3"]) / 3.0
        eval_dict["te_diag_mean"] = (eval_dict["te_diag_1"] + eval_dict["te_diag_2"] + eval_dict["te_diag_3"]) / 3.0
        extra_numeric.append("te_diag_mean")

    # Fit tabular encoder
    encoder = TabularEncoder(min_count=40)
    encoder.fit(task, train_dict, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)
    cat_idx = encoder.categorical_indices

    # Diverse 4-model LightGBM ensemble
    models = [
        # Model 1: Balanced leaf-wise model
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=180,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_lambda=0.1,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Depth-constrained model for robust main & pairwise interactions
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.025,
            max_depth=5,
            num_leaves=20,
            min_child_samples=120,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_lambda=0.5,
            cat_smooth=30.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Higher-capacity model with L1/L2 regularization
        lgb.LGBMClassifier(
            n_estimators=190,
            learning_rate=0.018,
            num_leaves=42,
            min_child_samples=220,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=65.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: Conservative model with high min_child_samples
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.02,
            num_leaves=26,
            min_child_samples=260,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=0.05,
            cat_smooth=70.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(eval_cols["time_in_hospital"]), dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=cat_idx)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
