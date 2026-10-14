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
    from scipy.stats import rankdata

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    def add_num(name: str, tr_arr, ev_arr):
        train_feats[name] = np.asarray(tr_arr, dtype=np.float64)
        eval_feats[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name: str, tr_arr, ev_arr):
        train_feats[name] = np.asarray(tr_arr, dtype=object)
        eval_feats[name] = np.asarray(ev_arr, dtype=object)
        extra_categorical.append(name)

    # 1. Core Clinical Comorbidities (Charlson indicators & total condition count)
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in cm_tr:
        add_num(k, cm_tr[k], cm_ev[k])

    # 2. ICD-9 Chapter groups (consolidating 700+ codes into 10 clinical bodies)
    for diag in ('diag_1', 'diag_2', 'diag_3'):
        if diag in train and diag in eval_cols:
            add_cat(f"{diag}_group",
                    [icd9_group(c) for c in train[diag]],
                    [icd9_group(c) for c in eval_cols[diag]])

    # 3. Clean Out-Of-Fold Target Encoding for Primary High-Cardinality Codes (Dual-Smoothed)
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            # Sensitive fine-grained target encoding
            tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
            add_num(f"{col}_te", tr_te, ev_te)
            # Regularized coarse-grained target encoding for robustness
            tr_te2, ev_te2 = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=100.0, seed=123)
            add_num(f"{col}_te_smooth", tr_te2, ev_te2)

    # 4. Inpatient & Healthcare Prior Utilization and Interactions
    inpat_tr = np.nan_to_num(train.get('number_inpatient', 0.0), nan=0.0)
    emerg_tr = np.nan_to_num(train.get('number_emergency', 0.0), nan=0.0)
    outpat_tr = np.nan_to_num(train.get('number_outpatient', 0.0), nan=0.0)
    inpat_ev = np.nan_to_num(eval_cols.get('number_inpatient', 0.0), nan=0.0)
    emerg_ev = np.nan_to_num(eval_cols.get('number_emergency', 0.0), nan=0.0)
    outpat_ev = np.nan_to_num(eval_cols.get('number_outpatient', 0.0), nan=0.0)
    
    total_tr = inpat_tr + emerg_tr + outpat_tr
    total_ev = inpat_ev + emerg_ev + outpat_ev
    add_num('total_prior_visits', total_tr, total_ev)
    
    add_num('log_inpatient', np.log1p(inpat_tr), np.log1p(inpat_ev))
    add_num('log_emergency', np.log1p(emerg_tr), np.log1p(emerg_ev))
    add_num('log_outpatient', np.log1p(outpat_tr), np.log1p(outpat_ev))
    add_num('log_total_prior', np.log1p(total_tr), np.log1p(total_ev))
    add_num('inpatient_ratio', inpat_tr / (total_tr + 1e-5), inpat_ev / (total_ev + 1e-5))

    # 5. Demographics, Stay Intensity and Comorbidity Interactions
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    age_tr = np.array([age_map.get(a, 50.0) for a in train.get('age', [])], dtype=np.float64)
    age_ev = np.array([age_map.get(a, 50.0) for a in eval_cols.get('age', [])], dtype=np.float64)
    add_num('age_numeric', age_tr, age_ev)
    
    diag_tr = np.nan_to_num(train.get('number_diagnoses', 0.0), nan=0.0)
    diag_ev = np.nan_to_num(eval_cols.get('number_diagnoses', 0.0), nan=0.0)
    add_num('inpatient_x_diagnoses', inpat_tr * diag_tr, inpat_ev * diag_ev)
    add_num('inpatient_x_age', inpat_tr * age_tr, inpat_ev * age_ev)

    th_tr = np.maximum(np.nan_to_num(train.get('time_in_hospital', 1.0), nan=1.0), 1.0)
    th_ev = np.maximum(np.nan_to_num(eval_cols.get('time_in_hospital', 1.0), nan=1.0), 1.0)
    
    labs_tr = np.nan_to_num(train.get('num_lab_procedures', 0.0))
    labs_ev = np.nan_to_num(eval_cols.get('num_lab_procedures', 0.0))
    add_num('labs_per_day', labs_tr / th_tr, labs_ev / th_ev)
    
    meds_tr = np.nan_to_num(train.get('num_medications', 0.0))
    meds_ev = np.nan_to_num(eval_cols.get('num_medications', 0.0))
    add_num('meds_per_day', meds_tr / th_tr, meds_ev / th_ev)
    
    procs_tr = np.nan_to_num(train.get('num_procedures', 0.0))
    procs_ev = np.nan_to_num(eval_cols.get('num_procedures', 0.0))
    add_num('procs_per_day', procs_tr / th_tr, procs_ev / th_ev)
    add_num('diag_per_day', diag_tr / th_tr, diag_ev / th_ev)

    # Comorbidity cross-product indicators
    if 'cm_chf' in cm_tr and 'cm_renal' in cm_tr:
        add_num('feat_cardiorenal', cm_tr['cm_chf'] * cm_tr['cm_renal'], cm_ev['cm_chf'] * cm_ev['cm_renal'])
    if 'cm_chf' in cm_tr and 'cm_pulmonary' in cm_tr:
        add_num('feat_cardiopulmonary', cm_tr['cm_chf'] * cm_tr['cm_pulmonary'], cm_ev['cm_chf'] * cm_ev['cm_pulmonary'])
    if 'cm_renal' in cm_tr and 'cm_diabetes_complicated' in cm_tr:
        add_num('feat_renaldiabetes', cm_tr['cm_renal'] * cm_tr['cm_diabetes_complicated'], cm_ev['cm_renal'] * cm_ev['cm_diabetes_complicated'])
    if 'cm_count' in cm_tr:
        add_num('feat_multi_comorbid', (cm_tr['cm_count'] >= 2.0).astype(np.float64), (cm_ev['cm_count'] >= 2.0).astype(np.float64))

    # Medication Complexity, Glycemic Volatility and Missingness Flags
    meds_list = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]

    def get_med_features(data):
        n = len(data.get("time_in_hospital", []))
        active_cnt = np.zeros(n, dtype=np.float64)
        change_cnt = np.zeros(n, dtype=np.float64)
        for m in meds_list:
            if m in data:
                val = data[m]
                active_cnt += np.isin(val, ["Steady", "Up", "Down"]).astype(np.float64)
                change_cnt += np.isin(val, ["Up", "Down"]).astype(np.float64)
        
        ins = data.get("insulin")
        on_ins = np.isin(ins, ["Steady", "Up", "Down"]).astype(np.float64) if ins is not None else np.zeros(n)
        ins_up = (ins == "Up").astype(np.float64) if ins is not None else np.zeros(n)
        ins_down = (ins == "Down").astype(np.float64) if ins is not None else np.zeros(n)
        
        ch = data.get("change")
        ch_val = (ch == "Ch").astype(np.float64) if ch is not None else np.zeros(n)
        
        dm = data.get("diabetesMed")
        dm_val = (dm == "Yes").astype(np.float64) if dm is not None else np.zeros(n)
        
        a1c = data.get("A1Cresult")
        a1c_tested = (~np.isin(a1c, ["None", None])).astype(np.float64) if a1c is not None else np.zeros(n)
        a1c_high = np.isin(a1c, [">7", ">8"]).astype(np.float64) if a1c is not None else np.zeros(n)
        
        return {
            "num_active_meds": active_cnt,
            "num_med_changes": change_cnt,
            "on_insulin": on_ins,
            "insulin_up": ins_up,
            "insulin_down": ins_down,
            "med_change_binary": ch_val,
            "diabetes_med_binary": dm_val,
            "a1c_tested": a1c_tested,
            "a1c_high": a1c_high,
        }

    med_tr = get_med_features(train)
    med_ev = get_med_features(eval_cols)
    for k in med_tr:
        add_num(k, med_tr[k], med_ev[k])

    def get_adm_disch_missing(data):
        n = len(data.get("time_in_hospital", []))
        disch = data.get("discharge_disposition_id")
        adm = data.get("admission_type_id")
        spec = data.get("medical_specialty")
        payer = data.get("payer_code")
        
        disch_home = (disch == "1").astype(np.float64) if disch is not None else np.zeros(n)
        disch_snf = (disch == "3").astype(np.float64) if disch is not None else np.zeros(n)
        disch_facility = np.isin(disch, ["2", "3", "4", "5", "22"]).astype(np.float64) if disch is not None else np.zeros(n)
        
        adm_emergency = (adm == "1").astype(np.float64) if adm is not None else np.zeros(n)
        
        missing_specialty = np.isin(spec, [None]).astype(np.float64) if spec is not None else np.zeros(n)
        missing_payer = np.isin(payer, [None]).astype(np.float64) if payer is not None else np.zeros(n)
        
        return {
            "disch_home": disch_home,
            "disch_snf": disch_snf,
            "disch_facility": disch_facility,
            "adm_emergency": adm_emergency,
            "missing_specialty": missing_specialty,
            "missing_payer": missing_payer,
        }

    adm_tr = get_adm_disch_missing(train)
    adm_ev = get_adm_disch_missing(eval_cols)
    for k in adm_tr:
        add_num(k, adm_tr[k], adm_ev[k])

    a1c_tr = train.get("A1Cresult")
    a1c_ev = eval_cols.get("A1Cresult")
    ch_tr = train.get("change")
    ch_ev = eval_cols.get("change")
    if a1c_tr is not None and ch_tr is not None and a1c_ev is not None and ch_ev is not None:
        add_num('a1c_high_no_change', 
                ((a1c_tr == ">8") & (ch_tr == "No")).astype(np.float64),
                ((a1c_ev == ">8") & (ch_ev == "No")).astype(np.float64))

    # 6. Fit TabularEncoder on Clean Signal Matrix
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_feats,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 7. 5-Architecture Complementary LightGBM Ensemble
    lgb_models = [
        # Model 1: The Deep Explorer
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.024,
            num_leaves=36,
            max_depth=7,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=3.0,
            reg_alpha=0.2,
            cat_smooth=30.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: The Regularized Shallow
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.015,
            num_leaves=16,
            max_depth=4,
            min_child_samples=200,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_lambda=8.0,
            reg_alpha=1.0,
            cat_smooth=60.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: The Wide Generalist
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.022,
            num_leaves=48,
            max_depth=6,
            min_child_samples=130,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_lambda=4.0,
            reg_alpha=0.3,
            cat_smooth=40.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: The Extreme Bagging
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.02,
            num_leaves=24,
            max_depth=5,
            min_child_samples=120,
            subsample=0.60,
            subsample_freq=1,
            colsample_bytree=0.45,
            reg_lambda=5.0,
            reg_alpha=0.5,
            cat_smooth=45.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 5: The Aggressive Learner
        lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.035,
            num_leaves=54,
            max_depth=8,
            min_child_samples=90,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=15.0,
            reg_alpha=0.0,
            cat_smooth=25.0,
            random_state=888,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = []
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds.append(model.predict_proba(x_eval)[:, 1])

    # 8. Consensus Rank-Averaged Probability Blending:
    # Blends predictions in rank space (optimal for ROC-AUC) while preserving calibrated probability distribution
    ranks = np.mean([rankdata(p) for p in preds], axis=0)
    avg_probs = np.mean(preds, axis=0)
    final_pred = np.sort(avg_probs)[np.argsort(np.argsort(ranks))]
    return np.clip(np.nan_to_num(final_pred, nan=0.114), 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
