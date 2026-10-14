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
    """Stage 1: Stratified multi-family ensemble (Logistic Regression + LightGBM + XGBoost) with

    clinical discordance features, bounded OOF optimization, and monotonic Platt calibration.
    """
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import xgboost as xgb

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_tr, y_train = builder.fit_transform(cohort_spec, train_records)
    x_ev = builder.transform(cohort_spec, eval_records)

    def extract_prior_features(records: Sequence[PatientRecord]) -> np.ndarray:
        feats = []
        for p in records:
            base_p = float(getattr(p, "baseline_api_prob", 0.5))
            sub = getattr(p, "baseline_submodel_scores", {}) or {}
            c_lin = float(sub.get("claims_linear", base_p))
            e_vit = float(sub.get("ehr_vitals", base_p))
            u_rul = float(sub.get("utilization_rules", base_p))

            l_base = safe_logit(base_p)
            l_claims = safe_logit(c_lin)
            l_vitals = safe_logit(e_vit)
            l_rules = safe_logit(u_rul)

            # Clinical discordance and syndromic interaction indicators
            diff_vit_claims = l_vitals - l_claims
            diff_base_rules = l_base - l_rules
            diff_base_claims = l_base - l_claims

            all_logits = [l_base, l_claims, l_vitals, l_rules]
            mean_prior = float(np.mean(all_logits))
            std_prior = float(np.std(all_logits))
            max_prior = float(np.max(all_logits))
            min_prior = float(np.min(all_logits))

            acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)
            ci = getattr(p, "contraindications", ()) or ()
            n_ci = float(len(ci))

            feats.append([
                l_base, l_claims, l_vitals, l_rules,
                base_p, c_lin, e_vit, u_rul,
                diff_vit_claims, diff_base_rules, diff_base_claims,
                mean_prior, std_prior, max_prior, min_prior,
                acuity, n_ci,
            ])
        arr = np.array(feats, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=6.0, neginf=-6.0)

    priors_tr = extract_prior_features(train_records)
    priors_ev = extract_prior_features(eval_records)

    x_train_stacked = np.nan_to_num(np.hstack([priors_tr, x_tr]), nan=0.0, posinf=1e4, neginf=-1e4)
    x_eval_stacked = np.nan_to_num(np.hstack([priors_ev, x_ev]), nan=0.0, posinf=1e4, neginf=-1e4)

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train_stacked)
    x_ev_scaled = scaler.transform(x_eval_stacked)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    min_class = min(n_pos, n_neg)

    # 5-fold stratified CV for out-of-fold blending and variance reduction
    n_splits = min(5, max(2, min_class)) if min_class >= 2 else 1

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_lr = np.zeros(len(train_records), dtype=float)
        oof_lgb = np.zeros(len(train_records), dtype=float)
        oof_xgb = np.zeros(len(train_records), dtype=float)

        eval_lr_preds = np.zeros(len(eval_records), dtype=float)
        eval_lgb_preds = np.zeros(len(eval_records), dtype=float)
        eval_xgb_preds = np.zeros(len(eval_records), dtype=float)

        for train_idx, val_idx in skf.split(x_tr_scaled, y_train):
            X_f_tr, y_f_tr = x_tr_scaled[train_idx], y_train[train_idx]
            X_f_val = x_tr_scaled[val_idx]

            # 1. Regularized Linear Stacking (captures calibrated main effects)
            lr = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            lr.fit(X_f_tr, y_f_tr)
            oof_lr[val_idx] = lr.predict_proba(X_f_val)[:, 1]
            eval_lr_preds += lr.predict_proba(x_ev_scaled)[:, 1] / n_splits

            # 2. LightGBM (captures deep non-linear multi-organ syndrome interactions)
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.04,
                num_leaves=15,
                max_depth=4,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            )
            lgb_model.fit(X_f_tr, y_f_tr)
            oof_lgb[val_idx] = lgb_model.predict_proba(X_f_val)[:, 1]
            eval_lgb_preds += lgb_model.predict_proba(x_ev_scaled)[:, 1] / n_splits

            # 3. XGBoost (adds depth-wise regularized gradient boosted diversity)
            xgb_model = xgb.XGBClassifier(
                n_estimators=90,
                learning_rate=0.04,
                max_depth=3,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=1,
                eval_metric="logloss",
            )
            xgb_model.fit(X_f_tr, y_f_tr)
            oof_xgb[val_idx] = xgb_model.predict_proba(X_f_val)[:, 1]
            eval_xgb_preds += xgb_model.predict_proba(x_ev_scaled)[:, 1] / n_splits

        z_oof_lr = np.array([safe_logit(p) for p in oof_lr])
        z_oof_lgb = np.array([safe_logit(p) for p in oof_lgb])
        z_oof_xgb = np.array([safe_logit(p) for p in oof_xgb])
        z_train_base = priors_tr[:, 0]

        z_ev_lr = np.array([safe_logit(p) for p in eval_lr_preds])
        z_ev_lgb = np.array([safe_logit(p) for p in eval_lgb_preds])
        z_ev_xgb = np.array([safe_logit(p) for p in eval_xgb_preds])
        z_eval_base = priors_ev[:, 0]

        # Bounded Brier-minimization to find optimal blending weights across model families
        def brier_objective(w: np.ndarray) -> float:
            w_pos = np.maximum(w, 0.0)
            s = np.sum(w_pos)
            if s <= 1e-9:
                return 1e6
            w_norm = w_pos / s
            z_b = (
                w_norm[0] * z_oof_lr
                + w_norm[1] * z_oof_lgb
                + w_norm[2] * z_oof_xgb
                + w_norm[3] * z_train_base
            )
            p_b = np.array([safe_sigmoid(z) for z in z_b])
            return float(np.mean((p_b - y_train) ** 2))

        init_w = np.array([0.38, 0.32, 0.22, 0.08])
        bounds = [(0.10, 0.65), (0.10, 0.55), (0.05, 0.45), (0.02, 0.20)]
        try:
            res = minimize(brier_objective, init_w, method="L-BFGS-B", bounds=bounds)
            if res.success and np.sum(res.x) > 0:
                opt_w = np.maximum(res.x, 0.0)
                opt_w = opt_w / np.sum(opt_w)
            else:
                opt_w = init_w / np.sum(init_w)
        except Exception:
            opt_w = init_w / np.sum(init_w)

        z_ev_blend = (
            opt_w[0] * z_ev_lr
            + opt_w[1] * z_ev_lgb
            + opt_w[2] * z_ev_xgb
            + opt_w[3] * z_eval_base
        )
        z_oof_blend = (
            opt_w[0] * z_oof_lr
            + opt_w[1] * z_oof_lgb
            + opt_w[2] * z_oof_xgb
            + opt_w[3] * z_train_base
        )

        # Strictly monotonic Platt calibration (strictly preserves ROC-AUC & PR-AUC rankings)
        calibrator = LogisticRegression(C=1e3, solver="lbfgs", max_iter=200, random_state=42)
        calibrator.fit(z_oof_blend.reshape(-1, 1), y_train)

        if calibrator.coef_[0][0] > 0.0:
            calibrated_probs = calibrator.predict_proba(z_ev_blend.reshape(-1, 1))[:, 1]
        else:
            calibrated_probs = np.array([safe_sigmoid(z) for z in z_ev_blend])
    else:
        # Fallback for degenerate small cohorts
        lr = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        lr.fit(x_tr_scaled, y_train)
        raw_probs = lr.predict_proba(x_ev_scaled)[:, 1]
        calibrated_probs = np.zeros(len(eval_records), dtype=float)
        for idx, p in enumerate(eval_records):
            base_p = float(getattr(p, "baseline_api_prob", 0.5))
            blended = 0.90 * safe_logit(float(raw_probs[idx])) + 0.10 * safe_logit(base_p)
            calibrated_probs[idx] = safe_sigmoid(blended)

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(calibrated_probs[idx])
        if not np.isfinite(prob):
            prob = float(getattr(p, "baseline_api_prob", 0.5))
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: First-fit highest-risk patients into the highest-efficacy feasible intervention tier."""
    state = CareAllocationState(cohort_spec, eval_records)
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5))),
    )
    tiers_by_rrr = [
        t
        for t, s in sorted(
            cohort_spec.interventions.items(),
            key=lambda kv: -kv[1].relative_risk_reduction,
        )
        if t != InterventionTier.NONE
    ]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        for tier in tiers_by_rrr:
            if state.fits(p.patient_id, tier) and state.expected_net_benefit(p.patient_id, risk, tier) > 0.0:
                state.assign(p.patient_id, tier)
                break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
