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

"""Initial baseline seed program for clinical Health Clinical Risk & Care Allocation.

Contains the two-stage strategy evolved by AlphaEvolve:
1. `fit_and_score_risk`: Refines patient event probabilities over legacy baseline API scores + EHR/claims features.
2. `allocate_interventions`: Assigns care-management interventions under slot, budget, and clinical eligibility rules.
"""

from __future__ import annotations

from typing import Dict, Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionTier,
    PatientRecord,
)
from clinical_toolkit import (
    CareAllocationState,
    ClinicalFeatureBuilder,
    safe_logit,
    safe_sigmoid,
)


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Domain mechanics and feasibility checks live in `clinical_toolkit` (`ClinicalFeatureBuilder`,
# `RiskBlender`, `CareAllocationState`, `safe_logit`, `safe_sigmoid`). Pre-imported libraries available in sandbox:
# `numpy`, `scipy`, `sklearn`, `lightgbm`, `xgboost`.


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model ensemble with low-rank tensor interaction decomposition and calibrated blending."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import TruncatedSVD
    from scipy.optimize import minimize
    import lightgbm as lgb
    import xgboost as xgb

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)
    y_train = np.asarray(y_train, dtype=np.int32)

    # 1. Meta-feature extraction & legacy discrepancy isolation
    def extract_meta(records: Sequence[PatientRecord]) -> np.ndarray:
        feats = []
        for p in records:
            b_prob = float(p.baseline_api_prob)
            b_logit = safe_logit(b_prob)
            sub = getattr(p, "baseline_submodel_scores", {}) or {}
            c_lin = float(sub.get("claims_linear", 0.0))
            e_vit = float(sub.get("ehr_vitals", 0.0))
            u_rul = float(sub.get("utilization_rules", 0.0))
            acuity = float(getattr(p, "acuity_score", 0.0))
            n_contra = float(len(getattr(p, "contraindications", []) or []))

            diff_ehr_claims = e_vit - c_lin
            diff_util_claims = u_rul - c_lin
            diff_acuity_base = acuity - b_prob

            feats.append([
                b_prob, b_logit, c_lin, e_vit, u_rul, acuity, n_contra,
                diff_ehr_claims, diff_util_claims, diff_acuity_base,
            ])
        return np.array(feats, dtype=np.float32)

    meta_tr = extract_meta(train_records)
    meta_ev = extract_meta(eval_records)

    # 2. Standardize clinical features & rank-r tensor interaction decomposition
    scaler_x = StandardScaler()
    X_tr_sc = scaler_x.fit_transform(x_train)
    X_ev_sc = scaler_x.transform(x_eval)

    n_comp = min(8, max(2, X_tr_sc.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(X_tr_sc)
    V = svd.components_.T  # Shape: (D, r)

    # Rank-r latent syndrome projections
    Z_tr = X_tr_sc @ V
    Z_ev = X_ev_sc @ V

    # Closed-form order-2 symmetric tensor interaction decomposition
    H_tr = (Z_tr ** 2) - ((X_tr_sc ** 2) @ (V ** 2))
    H_ev = (Z_ev ** 2) - ((X_ev_sc ** 2) @ (V ** 2))

    norm_H_tr = np.linalg.norm(H_tr, axis=1, keepdims=True)
    norm_H_ev = np.linalg.norm(H_ev, axis=1, keepdims=True)

    # Full augmented representation
    X_tr_full = np.hstack([X_tr_sc, Z_tr, H_tr, norm_H_tr, meta_tr])
    X_ev_full = np.hstack([X_ev_sc, Z_ev, H_ev, norm_H_ev, meta_ev])
    np.nan_to_num(X_tr_full, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.nan_to_num(X_ev_full, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

    # 3. Stratified 5-Fold Cross-Validation & Out-of-Fold Estimation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_train = len(train_records)
    n_eval = len(eval_records)

    oof_lgb = np.zeros(n_train, dtype=np.float64)
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    oof_lr = np.zeros(n_train, dtype=np.float64)
    oof_base = np.zeros(n_train, dtype=np.float64)

    eval_lgb_folds = np.zeros((n_splits, n_eval), dtype=np.float64)
    eval_xgb_folds = np.zeros((n_splits, n_eval), dtype=np.float64)
    eval_lr_folds = np.zeros((n_splits, n_eval), dtype=np.float64)
    eval_base_folds = np.zeros((n_splits, n_eval), dtype=np.float64)

    for fold_idx, (idx_tr, idx_val) in enumerate(skf.split(X_tr_full, y_train)):
        y_fold_tr = y_train[idx_tr]

        # Model A: LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=110,
            learning_rate=0.035,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42 + fold_idx,
            verbose=-1,
            n_jobs=2,
        )
        clf_lgb.fit(X_tr_full[idx_tr], y_fold_tr)
        oof_lgb[idx_val] = clf_lgb.predict_proba(X_tr_full[idx_val])[:, 1]
        eval_lgb_folds[fold_idx] = clf_lgb.predict_proba(X_ev_full)[:, 1]

        # Model B: XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=90,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42 + fold_idx,
            n_jobs=2,
        )
        clf_xgb.fit(X_tr_full[idx_tr], y_fold_tr)
        oof_xgb[idx_val] = clf_xgb.predict_proba(X_tr_full[idx_val])[:, 1]
        eval_xgb_folds[fold_idx] = clf_xgb.predict_proba(X_ev_full)[:, 1]

        # Model C: Regularized Logistic Regression
        scaler_lr = StandardScaler()
        X_tr_lr = scaler_lr.fit_transform(X_tr_full[idx_tr])
        X_val_lr = scaler_lr.transform(X_tr_full[idx_val])
        X_ev_lr = scaler_lr.transform(X_ev_full)

        clf_lr = LogisticRegression(C=0.5, max_iter=300, penalty="l2", solver="lbfgs", random_state=42 + fold_idx)
        clf_lr.fit(X_tr_lr, y_fold_tr)
        oof_lr[idx_val] = clf_lr.predict_proba(X_val_lr)[:, 1]
        eval_lr_folds[fold_idx] = clf_lr.predict_proba(X_ev_lr)[:, 1]

        # Model D: Calibrated Baseline Prior
        base_tr_logits = np.array([safe_logit(float(train_records[i].baseline_api_prob)) for i in idx_tr]).reshape(-1, 1)
        base_val_logits = np.array([safe_logit(float(train_records[i].baseline_api_prob)) for i in idx_val]).reshape(-1, 1)
        base_ev_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in eval_records]).reshape(-1, 1)

        clf_base = LogisticRegression(C=1.0, max_iter=200, random_state=42 + fold_idx)
        clf_base.fit(base_tr_logits, y_fold_tr)
        oof_base[idx_val] = clf_base.predict_proba(base_val_logits)[:, 1]
        eval_base_folds[fold_idx] = clf_base.predict_proba(base_ev_logits)[:, 1]

    # Fold averaging for eval predictions
    eval_lgb = np.mean(eval_lgb_folds, axis=0)
    eval_xgb = np.mean(eval_xgb_folds, axis=0)
    eval_lr = np.mean(eval_lr_folds, axis=0)
    eval_base = np.mean(eval_base_folds, axis=0)

    # 4. Constrained Optimization for Ensemble Blending Weights
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_lr, oof_base])
    eval_matrix = np.column_stack([eval_lgb, eval_xgb, eval_lr, eval_base])

    eps = 1e-6
    def loss_fn(w):
        p = np.clip(oof_matrix @ w, eps, 1.0 - eps)
        return -np.mean(y_train * np.log(p) + (1.0 - y_train) * np.log(1.0 - p))

    w_init = np.array([0.35, 0.30, 0.20, 0.15])
    bounds = [(0.02, 0.70) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_fn, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x) if (res.success and np.all(res.x >= 0)) else w_init

    oof_blended = np.clip(oof_matrix @ best_weights, eps, 1.0 - eps)
    eval_blended = np.clip(eval_matrix @ best_weights, eps, 1.0 - eps)

    # 5. Blend with toolkit RiskBlender if available
    try:
        blender = RiskBlender(random_state=42)
        rb_dict = blender.fit_predict(
            cohort_spec, train_records, eval_records,
            x_train=x_train, y_train=y_train, x_eval=x_eval
        )
        rb_eval_probs = np.array([float(rb_dict.get(p.patient_id, p.baseline_api_prob)) for p in eval_records])
        z_our = np.array([safe_logit(p) for p in eval_blended])
        z_rb = np.array([safe_logit(p) for p in rb_eval_probs])
        eval_blended = np.array([safe_sigmoid(0.70 * zo + 0.30 * zr) for zo, zr in zip(z_our, z_rb)])
    except Exception:
        pass

    # 6. Monotonic Platt Calibration to Minimize Brier Score
    oof_logits = np.array([safe_logit(p) for p in oof_blended]).reshape(-1, 1)
    eval_logits = np.array([safe_logit(p) for p in eval_blended]).reshape(-1, 1)

    platt = LogisticRegression(C=1e3, max_iter=200, random_state=42)
    platt.fit(oof_logits, y_train)

    if platt.coef_[0, 0] > 0.05:
        eval_calibrated = platt.predict_proba(eval_logits)[:, 1]
    else:
        eval_calibrated = eval_blended

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(eval_calibrated[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Knapsack allocation prioritizing maximum expected net benefit."""
    state = CareAllocationState(cohort_spec, eval_records)
    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    # Pre-calculate peak expected net benefit for each patient across all tiers
    patient_max_benefit = {}
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_b = 0.0
        for t in candidate_tiers:
            if state.fits(p.patient_id, t):
                b = state.expected_net_benefit(p.patient_id, risk, t)
                if b > best_b:
                    best_b = b
        patient_max_benefit[p.patient_id] = best_b

    # Rank patients by descending peak net benefit potential
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -patient_max_benefit[p.patient_id],
    )

    # Greedily assign the tier that currently delivers the highest positive net benefit
    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_tier = None
        best_benefit = 0.0
        for tier in candidate_tiers:
            if state.fits(p.patient_id, tier):
                benefit = state.expected_net_benefit(p.patient_id, risk, tier)
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_tier = tier

        if best_tier is not None and best_benefit > 0.0:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
