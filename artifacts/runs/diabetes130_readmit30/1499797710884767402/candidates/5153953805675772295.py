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


def _fallback_icd9_group(code):
    """Fallback chapter grouping if icd9_group is not available."""
    if code is None or code == "?" or str(code).strip() == "":
        return "missing"
    c = str(code).strip()
    if c.startswith("250"):
        return "diabetes"
    try:
        val = float(c.split(".")[0])
        if 390 <= val <= 459 or val == 785:
            return "circulatory"
        elif 460 <= val <= 519 or val == 786:
            return "respiratory"
        elif 520 <= val <= 579 or val == 787:
            return "digestive"
        elif 580 <= val <= 629 or val == 788:
            return "genitourinary"
        elif 140 <= val <= 239:
            return "neoplasms"
        elif 710 <= val <= 739:
            return "musculoskeletal"
        elif 800 <= val <= 999:
            return "injury"
        else:
            return "other"
    except Exception:
        return "other"


def _compute_oof_target_encoding(
    train_cat: np.ndarray,
    y: np.ndarray,
    eval_cat: np.ndarray,
    n_splits: int = 5,
    m: float = 20.0,
    seed: int = 42,
):
    """Compute K-fold out-of-fold target encoding for train and full encoding for eval."""
    n_train = len(train_cat)
    n_eval = len(eval_cat)
    oof_train = np.empty(n_train, dtype=np.float64)
    overall_mean = float(np.mean(y))

    train_str = np.array(
        [str(x) if (x is not None and str(x) != "?") else "__MISSING__" for x in train_cat]
    )
    eval_str = np.array(
        [str(x) if (x is not None and str(x) != "?") else "__MISSING__" for x in eval_cat]
    )

    rng = np.random.RandomState(seed)
    shuffled_idx = rng.permutation(n_train)
    folds = np.array_split(shuffled_idx, n_splits)

    for val_idx in folds:
        train_mask = np.ones(n_train, dtype=bool)
        train_mask[val_idx] = False

        tr_cat = train_str[train_mask]
        tr_y = y[train_mask]
        fold_prior = float(np.mean(tr_y))

        cats, counts = np.unique(tr_cat, return_counts=True)
        idx = np.searchsorted(cats, tr_cat)
        y_sums = np.bincount(idx, weights=tr_y, minlength=len(cats))
        te_map = {
            cat: (y_sum + m * fold_prior) / (cnt + m)
            for cat, y_sum, cnt in zip(cats, y_sums, counts)
        }

        oof_train[val_idx] = [te_map.get(c, fold_prior) for c in train_str[val_idx]]

    cats, counts = np.unique(train_str, return_counts=True)
    idx = np.searchsorted(cats, train_str)
    y_sums = np.bincount(idx, weights=y, minlength=len(cats))
    full_te_map = {
        cat: (y_sum + m * overall_mean) / (cnt + m)
        for cat, y_sum, cnt in zip(cats, y_sums, counts)
    }
    eval_encoded = np.array(
        [full_te_map.get(c, overall_mean) for c in eval_str], dtype=np.float64
    )

    return oof_train, eval_encoded


def _extract_engineered_features(task: TabularTask, cols: Dict[str, np.ndarray]):
    """Extract clinical, utilisation, intensity, and demographic features."""
    n_rows = len(cols["time_in_hospital"])
    num_feats = {}
    cat_feats = {}

    icd9_fn = globals().get("icd9_group", _fallback_icd9_group)

    # 1. Diagnosis grouping and flags
    d1_grp = np.array([icd9_fn(c) for c in cols["diag_1"]], dtype=object)
    d2_grp = np.array([icd9_fn(c) for c in cols["diag_2"]], dtype=object)
    d3_grp = np.array([icd9_fn(c) for c in cols["diag_3"]], dtype=object)
    cat_feats["diag_1_group"] = d1_grp
    cat_feats["diag_2_group"] = d2_grp
    cat_feats["diag_3_group"] = d3_grp

    num_feats["diag_circ_any"] = (
        (d1_grp == "circulatory") | (d2_grp == "circulatory") | (d3_grp == "circulatory")
    ).astype(np.float64)
    num_feats["diag_diab_any"] = (
        (d1_grp == "diabetes") | (d2_grp == "diabetes") | (d3_grp == "diabetes")
    ).astype(np.float64)
    num_feats["diag_resp_any"] = (
        (d1_grp == "respiratory") | (d2_grp == "respiratory") | (d3_grp == "respiratory")
    ).astype(np.float64)
    num_feats["diag_neop_any"] = (
        (d1_grp == "neoplasms") | (d2_grp == "neoplasms") | (d3_grp == "neoplasms")
    ).astype(np.float64)
    num_feats["diag_distinct_groups"] = np.array(
        [
            float(len({g for g in (g1, g2, g3) if g not in ("missing", "other")}))
            for g1, g2, g3 in zip(d1_grp, d2_grp, d3_grp)
        ],
        dtype=np.float64,
    )

    # 2. Utilisation features
    inpatient = np.nan_to_num(cols["number_inpatient"].astype(np.float64), nan=0.0)
    emergency = np.nan_to_num(cols["number_emergency"].astype(np.float64), nan=0.0)
    outpatient = np.nan_to_num(cols["number_outpatient"].astype(np.float64), nan=0.0)
    total_visits = inpatient + emergency + outpatient

    num_feats["total_visits"] = total_visits
    num_feats["has_inpatient"] = (inpatient > 0).astype(np.float64)
    num_feats["has_emergency"] = (emergency > 0).astype(np.float64)
    num_feats["has_outpatient"] = (outpatient > 0).astype(np.float64)
    num_feats["high_utilizer"] = ((inpatient + emergency) >= 2).astype(np.float64)
    num_feats["very_high_utilizer"] = (inpatient >= 2).astype(np.float64)
    num_feats["log_inpatient"] = np.log1p(inpatient)
    num_feats["log_emergency"] = np.log1p(emergency)
    num_feats["log_outpatient"] = np.log1p(outpatient)
    num_feats["log_total_visits"] = np.log1p(total_visits)
    num_feats["inpatient_ratio"] = inpatient / (total_visits + 1.0)
    num_feats["emergency_ratio"] = emergency / (total_visits + 1.0)

    # 3. Hospital stay intensity and ratios
    time_hosp = np.maximum(1.0, np.nan_to_num(cols["time_in_hospital"].astype(np.float64), nan=1.0))
    num_lab = np.nan_to_num(cols["num_lab_procedures"].astype(np.float64), nan=0.0)
    num_proc = np.nan_to_num(cols["num_procedures"].astype(np.float64), nan=0.0)
    num_med = np.nan_to_num(cols["num_medications"].astype(np.float64), nan=0.0)
    num_diag = np.maximum(1.0, np.nan_to_num(cols["number_diagnoses"].astype(np.float64), nan=1.0))

    num_feats["proc_intensity"] = (num_lab + num_proc + num_med) / time_hosp
    num_feats["med_per_day"] = num_med / time_hosp
    num_feats["lab_per_day"] = num_lab / time_hosp
    num_feats["proc_per_day"] = num_proc / time_hosp
    num_feats["med_per_diag"] = num_med / num_diag
    num_feats["lab_per_diag"] = num_lab / num_diag
    num_feats["diag_intensity"] = num_diag * time_hosp
    num_feats["proc_per_diag"] = num_proc / num_diag

    # 4. Medication dynamics across medication columns
    non_med_cols = {
        "race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
        "max_glu_serum", "A1Cresult", "change", "diabetesMed"
    }
    med_cols = [c for c in task.categorical_columns if c not in non_med_cols and c in cols]

    num_active = np.zeros(n_rows, dtype=np.float64)
    num_up = np.zeros(n_rows, dtype=np.float64)
    num_down = np.zeros(n_rows, dtype=np.float64)

    for m in med_cols:
        arr = cols[m]
        num_up += (arr == "Up").astype(np.float64)
        num_down += (arr == "Down").astype(np.float64)
        num_active += np.isin(arr, ("Steady", "Up", "Down")).astype(np.float64)

    num_feats["num_active_meds"] = num_active
    num_feats["num_up_meds"] = num_up
    num_feats["num_down_meds"] = num_down
    num_feats["num_med_changes"] = num_up + num_down

    if "insulin" in cols:
        num_feats["insulin_active"] = np.isin(cols["insulin"], ("Steady", "Up", "Down")).astype(np.float64)
        num_feats["insulin_changed"] = np.isin(cols["insulin"], ("Up", "Down")).astype(np.float64)

    if "change" in cols:
        num_feats["change_med"] = (cols["change"] == "Ch").astype(np.float64)
    if "diabetesMed" in cols:
        num_feats["any_diabetes_med"] = (cols["diabetesMed"] == "Yes").astype(np.float64)

    # 5. Lab test indicators
    if "A1Cresult" in cols:
        num_feats["a1c_tested"] = np.isin(cols["A1Cresult"], (">7", ">8", "Norm")).astype(np.float64)
        num_feats["a1c_high"] = np.isin(cols["A1Cresult"], (">7", ">8")).astype(np.float64)
    if "max_glu_serum" in cols:
        num_feats["glu_tested"] = np.isin(cols["max_glu_serum"], (">200", ">300", "Norm")).astype(np.float64)
        num_feats["glu_high"] = np.isin(cols["max_glu_serum"], (">200", ">300")).astype(np.float64)

    # 6. Age midpoint and interactions
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0, "[40-50)": 45.0,
        "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0, "[80-90)": 85.0, "[90-100)": 95.0,
    }
    age_num = np.array([age_map.get(str(a).strip(), 65.0) if a is not None else 65.0 for a in cols["age"]], dtype=np.float64)
    num_feats["age_num"] = age_num
    num_feats["age_x_inpatient"] = age_num * np.log1p(inpatient)
    num_feats["age_x_diag"] = age_num * num_diag
    num_feats["inpatient_x_hosp"] = np.log1p(inpatient) * np.log1p(time_hosp)
    num_feats["diag_x_med"] = num_diag * num_med

    return num_feats, cat_feats


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit an ensemble on `train`/`y_train` and return readmission probabilities for `eval_cols`."""
    train_dict = dict(train)
    eval_dict = dict(eval_cols)

    # 1. Extract domain features
    tr_num, tr_cat = _extract_engineered_features(task, train_dict)
    ev_num, ev_cat = _extract_engineered_features(task, eval_dict)

    for k, v in tr_num.items():
        train_dict[k] = v
        eval_dict[k] = ev_num[k]

    for k, v in tr_cat.items():
        train_dict[k] = v
        eval_dict[k] = ev_cat[k]

    # 2. Out-of-fold smoothed target encoding for high-cardinality features
    te_columns = [
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "diag_1_group",
        "payer_code",
    ]
    for col in te_columns:
        if col in train_dict and col in eval_dict:
            oof_tr, full_ev = _compute_oof_target_encoding(
                train_dict[col], y_train, eval_dict[col], n_splits=5, m=20.0, seed=42
            )
            te_name = f"te_{col}"
            train_dict[te_name] = oof_tr
            eval_dict[te_name] = full_ev
            tr_num[te_name] = oof_tr

    extra_num_names = tuple(tr_num.keys())
    extra_cat_names = tuple(tr_cat.keys())

    # 3. Fit TabularEncoder with registered derived features
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_dict,
        extra_numeric=extra_num_names,
        extra_categorical=extra_cat_names,
    )
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # 4. Stratified 5-Fold Cross-Validation Ensemble
    n_splits = 5
    rng = np.random.RandomState(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    pos_splits = np.array_split(rng.permutation(pos_idx), n_splits)
    neg_splits = np.array_split(rng.permutation(neg_idx), n_splits)

    eval_preds_cv = np.zeros(len(x_eval), dtype=np.float64)

    for fold in range(n_splits):
        val_idx = np.concatenate([pos_splits[fold], neg_splits[fold]])
        train_mask = np.ones(len(y_train), dtype=bool)
        train_mask[val_idx] = False
        tr_idx = np.where(train_mask)[0]

        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]

        # Model A: Medium depth with moderate subsampling
        m_a = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=24,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=3.0,
            reg_alpha=0.5,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_a.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)

        # Model B: Shallower, highly regularized trees
        m_b = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=16,
            max_depth=5,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            reg_alpha=1.0,
            random_state=100 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_b.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)

        fold_preds = 0.5 * m_a.predict_proba(x_eval)[:, 1] + 0.5 * m_b.predict_proba(x_eval)[:, 1]
        eval_preds_cv += fold_preds / n_splits

    # 5. Full-training data model
    m_full = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=20,
        max_depth=5,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=4.0,
        reg_alpha=0.8,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    m_full.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    full_preds = m_full.predict_proba(x_eval)[:, 1]

    # Weighted blend of 5-fold CV bagged ensemble and full model
    final_preds = 0.75 * eval_preds_cv + 0.25 * full_preds
    return np.clip(np.nan_to_num(final_preds, nan=0.114), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
