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
    """Stage 1: Multi-model ensemble with latent syndrome SVD factors and metric-optimized blending."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.optimize import minimize
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    if len(eval_records) == 0:
        return {}

    y_train = np.asarray([int(p.outcome_label) for p in train_records], dtype=np.int32)
    if len(np.unique(y_train)) < 2:
        return {p.patient_id: float(p.baseline_api_prob) for p in eval_records}

    # Extract interaction-augmented clinical syndrome features
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, _ = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize continuous features
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train)
    x_ev_scaled = scaler.transform(x_eval)

    # Latent syndrome factor decomposition via TruncatedSVD
    n_svd = min(6, max(0, x_train.shape[1] - 1))
    if n_svd >= 2:
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        x_tr_svd = svd.fit_transform(x_tr_scaled)
        x_ev_svd = svd.transform(x_ev_scaled)
    else:
        x_tr_svd = np.empty((len(train_records), 0), dtype=np.float32)
        x_ev_svd = np.empty((len(eval_records), 0), dtype=np.float32)

    # Extract legacy submodel signals and clinical discrepancy features
    def extract_meta_features(records):
        feats = []
        for p in records:
            base_p = float(p.baseline_api_prob)
            base_l = safe_logit(base_p)
            sub = getattr(p, "baseline_submodel_scores", {}) or {}
            c_lin = float(sub.get("claims_linear", base_p))
            e_vit = float(sub.get("ehr_vitals", base_p))
            u_rul = float(sub.get("utilization_rules", base_p))
            acuity = float(getattr(p, "acuity_score", 0.5))
            contra = float(len(getattr(p, "contraindications", ())))
            feats.append([
                base_p,
                base_l,
                c_lin,
                safe_logit(c_lin),
                e_vit,
                safe_logit(e_vit),
                u_rul,
                safe_logit(u_rul),
                acuity,
                contra,
                e_vit - c_lin,
                u_rul - c_lin,
                acuity - base_p,
            ])
        return np.asarray(feats, dtype=np.float32)

    meta_tr = extract_meta_features(train_records)
    meta_ev = extract_meta_features(eval_records)
    meta_scaler = StandardScaler()
    meta_tr_scaled = meta_scaler.fit_transform(meta_tr)
    meta_ev_scaled = meta_scaler.transform(meta_ev)

    # Concatenate representations
    x_tr_tree = np.hstack([x_train, x_tr_svd, meta_tr])
    x_ev_tree = np.hstack([x_eval, x_ev_svd, meta_ev])
    x_tr_lin = np.hstack([x_tr_scaled, x_tr_svd, meta_tr_scaled])
    x_ev_lin = np.hstack([x_ev_scaled, x_ev_svd, meta_ev_scaled])

    base_tr_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in train_records]).reshape(-1, 1)
    base_ev_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in eval_records]).reshape(-1, 1)

    # Out-of-fold cross validation for calibration and ensemble weight optimization
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    n_splits = min(5, max(2, min(n_pos, n_neg)))

    model_keys = ["base", "lgbm", "xgb", "hgb", "lr"]
    oof_preds = {k: np.zeros(len(train_records), dtype=np.float64) for k in model_keys}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(x_tr_tree, y_train):
        y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]

        # Calibrated baseline
        b_clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        b_clf.fit(base_tr_logits[tr_idx], y_tr_f)
        oof_preds["base"][val_idx] = b_clf.predict_proba(base_tr_logits[val_idx])[:, 1]

        # LightGBM
        m_lgb = LGBMClassifier(
            n_estimators=100, learning_rate=0.03, num_leaves=15, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
            random_state=42, verbose=-1, n_jobs=1,
        )
        m_lgb.fit(x_tr_tree[tr_idx], y_tr_f)
        oof_preds["lgbm"][val_idx] = m_lgb.predict_proba(x_tr_tree[val_idx])[:, 1]

        # XGBoost
        m_xgb = XGBClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=1, verbosity=0,
        )
        m_xgb.fit(x_tr_tree[tr_idx], y_tr_f)
        oof_preds["xgb"][val_idx] = m_xgb.predict_proba(x_tr_tree[val_idx])[:, 1]

        # HistGradientBoosting
        m_hgb = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.03, max_leaf_nodes=15,
            min_samples_leaf=15, random_state=42,
        )
        m_hgb.fit(x_tr_tree[tr_idx], y_tr_f)
        oof_preds["hgb"][val_idx] = m_hgb.predict_proba(x_tr_tree[val_idx])[:, 1]

        # Regularized Logistic Regression
        m_lr = LogisticRegression(C=0.2, max_iter=300, random_state=42)
        m_lr.fit(x_tr_lin[tr_idx], y_tr_f)
        oof_preds["lr"][val_idx] = m_lr.predict_proba(x_tr_lin[val_idx])[:, 1]

    # Full models evaluated on eval_records
    eval_preds = {}
    base_clf_full = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    base_clf_full.fit(base_tr_logits, y_train)
    eval_preds["base"] = base_clf_full.predict_proba(base_ev_logits)[:, 1]

    m_lgb_full = LGBMClassifier(
        n_estimators=100, learning_rate=0.03, num_leaves=15, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
        random_state=42, verbose=-1, n_jobs=1,
    )
    m_lgb_full.fit(x_tr_tree, y_train)
    eval_preds["lgbm"] = m_lgb_full.predict_proba(x_ev_tree)[:, 1]

    m_xgb_full = XGBClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=1, verbosity=0,
    )
    m_xgb_full.fit(x_tr_tree, y_train)
    eval_preds["xgb"] = m_xgb_full.predict_proba(x_ev_tree)[:, 1]

    m_hgb_full = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.03, max_leaf_nodes=15,
        min_samples_leaf=15, random_state=42,
    )
    m_hgb_full.fit(x_tr_tree, y_train)
    eval_preds["hgb"] = m_hgb_full.predict_proba(x_ev_tree)[:, 1]

    m_lr_full = LogisticRegression(C=0.2, max_iter=300, random_state=42)
    m_lr_full.fit(x_tr_lin, y_train)
    eval_preds["lr"] = m_lr_full.predict_proba(x_ev_lin)[:, 1]

    # Construct logit matrices
    oof_logits = np.column_stack([
        np.array([safe_logit(p) for p in oof_preds[k]])
        for k in model_keys
    ])
    eval_logits = np.column_stack([
        np.array([safe_logit(p) for p in eval_preds[k]])
        for k in model_keys
    ])

    def sens_at_spec80(y_true, scores):
        neg_scores = scores[y_true == 0]
        pos_scores = scores[y_true == 1]
        if len(neg_scores) == 0 or len(pos_scores) == 0:
            return 0.5
        thresh = float(np.percentile(neg_scores, 80.0))
        return float(np.mean(pos_scores >= thresh))

    prior_w = np.array([0.08, 0.35, 0.30, 0.15, 0.12], dtype=np.float64)
    prior_w /= prior_w.sum()

    # Optimization of ensemble weights directly against CMS HRRP fitness objective
    def objective(theta):
        exp_t = np.exp(theta - np.max(theta))
        w = exp_t / np.sum(exp_t)
        b_logits = oof_logits @ w
        b_probs = np.array([safe_sigmoid(z) for z in b_logits])

        try:
            auc_roc = float(roc_auc_score(y_train, b_probs))
        except Exception:
            auc_roc = 0.5
        try:
            auc_pr = float(average_precision_score(y_train, b_probs))
        except Exception:
            auc_pr = 0.5
        sens80 = sens_at_spec80(y_train, b_probs)
        brier = float(np.mean((b_probs - y_train) ** 2))

        metric = 0.50 * auc_roc + 0.25 * auc_pr + 0.15 * sens80 + 0.10 * (1.0 - brier)
        reg = 0.05 * float(np.sum((w - prior_w) ** 2))
        return -(metric - reg)

    theta0 = np.log(prior_w)
    try:
        opt_res = minimize(
            objective,
            theta0,
            method="Nelder-Mead",
            options={"maxiter": 100, "xatol": 1e-3, "fatol": 1e-4},
        )
        best_exp = np.exp(opt_res.x - np.max(opt_res.x))
        opt_w = best_exp / np.sum(best_exp)
    except Exception:
        opt_w = prior_w

    oof_blend_logits = oof_logits @ opt_w
    eval_blend_logits = eval_logits @ opt_w

    # Monotonic Platt calibration to minimize Brier score while preserving rank order
    try:
        calibrator = LogisticRegression(C=5.0, max_iter=200, random_state=42)
        calibrator.fit(oof_blend_logits.reshape(-1, 1), y_train)
        if calibrator.coef_[0][0] > 0.0:
            eval_cal_probs = calibrator.predict_proba(eval_blend_logits.reshape(-1, 1))[:, 1]
        else:
            eval_cal_probs = np.array([safe_sigmoid(z) for z in eval_blend_logits])
    except Exception:
        eval_cal_probs = np.array([safe_sigmoid(z) for z in eval_blend_logits])

    out: Dict[str, float] = {}
    for p, prob in zip(eval_records, eval_cal_probs):
        clean_p = float(np.clip(prob, 1e-4, 1.0 - 1e-4))
        out[p.patient_id] = round(clean_p, 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Maximal net-benefit intervention allocation with clinical constraint enforcement."""
    state = CareAllocationState(cohort_spec, eval_records)
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )
    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_tier = None
        best_net_benefit = 0.0

        for tier in candidate_tiers:
            if state.fits(p.patient_id, tier):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > best_net_benefit:
                    best_net_benefit = enb
                    best_tier = tier

        if best_tier is not None:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
