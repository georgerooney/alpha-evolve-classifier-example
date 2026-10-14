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
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    train_feats = dict(train)
    eval_feats = dict(eval_cols)

    # 1. Extract clinical comorbidities flags
    comorb_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    comorb_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    extra_numeric = list(comorb_train.keys())
    for k, v in comorb_train.items():
        train_feats[k] = v
        eval_feats[k] = comorb_eval[k]

    # 2. Add grouped ICD-9 chapters
    extra_categorical = []
    for col in ["diag_1", "diag_2", "diag_3"]:
        grouped_col = f"{col}_grouped"
        train_feats[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feats[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

    # 3. Domain feature engineering (Utilization, Medication, Lab, Intensity)
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    known_meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    med_cols = [c for c in known_meds if c in train]

    for d, feats, com in [(train, train_feats, comorb_train), (eval_cols, eval_feats, comorb_eval)]:
        num_inp = d["number_inpatient"]
        num_emg = d["number_emergency"]
        num_out = d["number_outpatient"]
        time_hosp = np.maximum(d["time_in_hospital"], 1.0)
        num_meds = d["num_medications"]
        num_labs = d["num_lab_procedures"]
        num_procs = d["num_procedures"]
        num_diags = d["number_diagnoses"]

        # Utilization patterns
        tot_prior = num_inp + num_emg + num_out
        feats["total_prior_visits"] = tot_prior
        feats["has_prior_inpatient"] = (num_inp > 0).astype(np.float64)
        feats["has_prior_emergency"] = (num_emg > 0).astype(np.float64)
        feats["has_prior_outpatient"] = (num_out > 0).astype(np.float64)
        feats["has_any_prior"] = (tot_prior > 0).astype(np.float64)
        feats["inpatient_ratio"] = num_inp / (tot_prior + 1e-4)
        feats["emergency_ratio"] = num_emg / (tot_prior + 1e-4)
        feats["log_inpatient"] = np.log1p(num_inp)
        feats["log_emergency"] = np.log1p(num_emg)
        feats["log_outpatient"] = np.log1p(num_out)
        feats["log_total_visits"] = np.log1p(tot_prior)

        # Inpatient care intensity rates
        feats["labs_per_day"] = num_labs / time_hosp
        feats["meds_per_day"] = num_meds / time_hosp
        feats["procs_per_day"] = num_procs / time_hosp
        feats["diags_per_day"] = num_diags / time_hosp
        feats["meds_per_diag"] = num_meds / (num_diags + 1e-4)
        feats["total_procedures"] = num_labs + num_procs
        feats["inp_per_day"] = num_inp / time_hosp

        # Numeric age and weighted Charlson Comorbidity Index
        age_num = np.array([age_map.get(str(x), np.nan) for x in d["age"]], dtype=np.float64)
        feats["age_numeric"] = age_num
        cci = (
            com["cm_mi"] + com["cm_chf"] + com["cm_pvd"] + com["cm_cerebrovascular"] +
            com["cm_dementia"] + com["cm_pulmonary"] + com["cm_rheumatic"] + com["cm_peptic_ulcer"] +
            com["cm_liver"] + 2.0 * com["cm_hemiplegia"] + 2.0 * com["cm_renal"] +
            2.0 * com["cm_cancer"] + 2.0 * com["cm_diabetes_complicated"] +
            4.0 * com["cm_metastatic"] + 4.0 * com["cm_hiv"]
        )
        feats["charlson_score"] = cci
        feats["charlson_per_age"] = cci / (age_num + 1e-4)
        feats["util_severity"] = np.log1p(num_inp) * (com["cm_count"] + 1.0)

        # Discharge and admission high-risk subgroups
        dd = d["discharge_disposition_id"]
        feats["is_discharged_home"] = (dd == "1").astype(np.float64)
        feats["is_discharged_snf"] = (dd == "3").astype(np.float64)
        feats["is_discharged_home_health"] = (dd == "6").astype(np.float64)
        feats["is_discharged_ama"] = (dd == "7").astype(np.float64)

        adm_t = d["admission_type_id"]
        feats["is_emergency_adm"] = (adm_t == "1").astype(np.float64)
        feats["is_urgent_adm"] = (adm_t == "2").astype(np.float64)
        feats["is_elective_adm"] = (adm_t == "3").astype(np.float64)

        # Medication count, changes, and insulin dosage
        n_rows = len(num_inp)
        num_act = np.zeros(n_rows, dtype=np.float64)
        num_up = np.zeros(n_rows, dtype=np.float64)
        num_down = np.zeros(n_rows, dtype=np.float64)
        num_std = np.zeros(n_rows, dtype=np.float64)
        num_chg = np.zeros(n_rows, dtype=np.float64)
        for mc in med_cols:
            mvals = d[mc]
            u = (mvals == "Up")
            dw = (mvals == "Down")
            s = (mvals == "Steady")
            num_up += u.astype(np.float64)
            num_down += dw.astype(np.float64)
            num_std += s.astype(np.float64)
            num_chg += (u | dw).astype(np.float64)
            num_act += (u | dw | s).astype(np.float64)

        feats["num_meds_active"] = num_act
        feats["num_meds_up"] = num_up
        feats["num_meds_down"] = num_down
        feats["num_meds_steady"] = num_std
        feats["num_meds_changed"] = num_chg

        if "insulin" in d:
            ins = d["insulin"]
            feats["insulin_up"] = (ins == "Up").astype(np.float64)
            feats["insulin_down"] = (ins == "Down").astype(np.float64)
            feats["insulin_steady"] = (ins == "Steady").astype(np.float64)
            feats["insulin_active"] = (ins != "No").astype(np.float64)

        if "change" in d:
            feats["change_flag"] = (d["change"] == "Ch").astype(np.float64)
        if "diabetesMed" in d:
            feats["diabetes_med_flag"] = (d["diabetesMed"] == "Yes").astype(np.float64)

        # Lab test indicators and ordinals
        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            feats["a1c_tested"] = (a1c != "None").astype(np.float64)
            feats["a1c_high"] = ((a1c == ">7") | (a1c == ">8")).astype(np.float64)
            feats["a1c_very_high"] = (a1c == ">8").astype(np.float64)
            a1c_ord = np.zeros(n_rows, dtype=np.float64)
            a1c_ord[a1c == "Norm"] = 1.0
            a1c_ord[a1c == ">7"] = 2.0
            a1c_ord[a1c == ">8"] = 3.0
            feats["a1c_ordinal"] = a1c_ord

        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            feats["glu_tested"] = (glu != "None").astype(np.float64)
            feats["glu_high"] = ((glu == ">200") | (glu == ">300")).astype(np.float64)
            feats["glu_very_high"] = (glu == ">300").astype(np.float64)
            glu_ord = np.zeros(n_rows, dtype=np.float64)
            glu_ord[glu == "Norm"] = 1.0
            glu_ord[glu == ">200"] = 2.0
            glu_ord[glu == ">300"] = 3.0
            feats["glu_ordinal"] = glu_ord

        # Primary and multi-diagnosis clinical condition flags
        g1 = feats["diag_1_grouped"]
        g2 = feats["diag_2_grouped"]
        g3 = feats["diag_3_grouped"]
        feats["diag1_is_diabetes"] = (g1 == "diabetes").astype(np.float64)
        feats["diag1_is_circulatory"] = (g1 == "circulatory").astype(np.float64)
        feats["any_diabetes_diag"] = ((g1 == "diabetes") | (g2 == "diabetes") | (g3 == "diabetes")).astype(np.float64)
        feats["any_circulatory_diag"] = ((g1 == "circulatory") | (g2 == "circulatory") | (g3 == "circulatory")).astype(np.float64)

    derived_numeric_keys = [
        k for k in train_feats.keys()
        if k not in train and k not in extra_categorical and k not in comorb_train
    ]
    extra_numeric.extend(derived_numeric_keys)

    # 4. Out-of-fold target encodings for high-cardinality administrative and clinical features
    te_cols = [
        "diag_1", "diag_2", "diag_3",
        "discharge_disposition_id", "admission_source_id", "admission_type_id",
        "medical_specialty", "payer_code"
    ]
    for col in te_cols:
        if col in train:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
            te_name = f"{col}_te"
            train_feats[te_name] = tr_enc.astype(np.float64)
            eval_feats[te_name] = ev_enc.astype(np.float64)
            extra_numeric.append(te_name)

    # 5. Tabular encoding
    encoder = TabularEncoder(min_count=30)
    encoder.fit(
        task,
        train_feats,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 6. Stratified 5-Fold cross-validated ensembling of diverse LightGBM models
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]

        # Model 1: Moderate depth, higher capacity
        m1 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=80,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.0,
            cat_smooth=30.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        p1 = m1.predict_proba(x_eval)[:, 1]

        # Model 2: Shallow, heavily regularized model for variance reduction
        m2 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.02,
            num_leaves=16,
            max_depth=4,
            min_child_samples=130,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_alpha=0.5,
            reg_lambda=3.0,
            cat_smooth=50.0,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        p2 = m2.predict_proba(x_eval)[:, 1]

        preds += (0.55 * p1 + 0.45 * p2) / 5.0

    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
