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
import numpy as np
import scipy
from scipy.optimize import minimize
from typing import Dict, Sequence, List, Tuple, Any, Optional

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from clinical_toolkit import (
    ClinicalFeatureBuilder,
    RiskBlender,
    CareAllocationState,
    InterventionTier,
    safe_logit,
    safe_sigmoid,
)


def _bounded_logit(val: float) -> float:
    return float(safe_logit(max(1e-4, min(1.0 - 1e-4, float(val)))))


def _extract_tensor_dissonance_features(records: Sequence[Any]) -> np.ndarray:
    """Extract multimodal clinical tensor interactions, divergence, and syndrome discordance."""
    feats: List[List[float]] = []
    for p in records:
        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        api_prob = float(getattr(p, "baseline_api_prob", 0.2))
        claims = float(sub.get("claims_linear", api_prob))
        vitals = float(sub.get("ehr_vitals", api_prob))
        util = float(sub.get("utilization_rules", api_prob))
        acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)

        # Logit-space submodel projections
        l_api = _bounded_logit(api_prob)
        l_claims = _bounded_logit(claims)
        l_vitals = _bounded_logit(vitals)
        l_util = _bounded_logit(util)

        # Pairwise tensions & discordance
        diff_vitals_claims = vitals - claims      # Acute bedside decompensation vs chronic baseline
        diff_util_claims = util - claims          # SDoH / recurrent utilization vs chronic coding
        diff_vitals_util = vitals - util          # Acute physiology vs routine care
        diff_api_submean = api_prob - (claims + vitals + util) / 3.0

        # Multi-model dispersion tensor metrics
        sub_arr = np.array([claims, vitals, util], dtype=np.float32)
        spread = float(np.ptp(sub_arr))
        variance = float(np.var(sub_arr))

        # Bilinear tensor outer-product interaction terms
        inter_cv = l_claims * l_vitals
        inter_cu = l_claims * l_util
        inter_vu = l_vitals * l_util

        # Non-linear Acuity Modulations
        acuity_vitals = acuity * vitals
        acuity_claims = acuity * claims
        acuity_util = acuity * util
        acuity_sqrt = float(np.sqrt(max(0.0, acuity)))

        # Specific Clinical Syndrome Signatures
        # 1. Acute Shock Divergence: High vitals risk over chronic baseline under elevated acuity
        shock_divergence = max(0.0, diff_vitals_claims) * (1.0 + acuity)
        # 2. SDoH Adherence Spiral: High utilization over claims under instability
        sdoh_spiral = max(0.0, diff_util_claims) * (1.0 + acuity)
        # 3. Elderly Low-Acuity Elective Correction: High claims, stable vitals, low acuity
        elective_correction = max(0.0, claims - vitals) / (acuity + 0.2)

        feats.append([
            api_prob, claims, vitals, util, acuity, acuity_sqrt,
            l_api, l_claims, l_vitals, l_util,
            diff_vitals_claims, diff_util_claims, diff_vitals_util, diff_api_submean,
            spread, variance,
            inter_cv, inter_cu, inter_vu,
            acuity_vitals, acuity_claims, acuity_util,
            shock_divergence, sdoh_spiral, elective_correction,
        ])
    return np.array(feats, dtype=np.float32)


def _optimize_blend_weights(z_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Convex optimization over probability simplex with temperature scaling to minimize OOF log-loss."""
    n_models = z_train.shape[1]
    init_params = np.zeros(n_models + 2, dtype=np.float64)
    init_params[n_models + 1] = 1.0  # initial beta = 1.0

    def loss(params: np.ndarray) -> float:
        theta = params[:n_models]
        alpha = params[n_models]
        beta = params[n_models + 1]

        # Softmax transformation guarantees non-negative weights summing to 1
        exp_th = np.exp(theta - np.max(theta))
        w = exp_th / np.sum(exp_th)

        logits = alpha + beta * (z_train @ w)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -14.0, 14.0)))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

        bce = -np.mean(y_train * np.log(probs) + (1.0 - y_train) * np.log(1.0 - probs))
        l2_reg = 0.005 * np.sum(theta ** 2)
        return float(bce + l2_reg)

    try:
        res = minimize(loss, init_params, method="L-BFGS-B", options={"maxiter": 70})
        if res.success or res.fun < 10.0:
            opt_theta = res.x[:n_models]
            exp_th = np.exp(opt_theta - np.max(opt_theta))
            opt_w = exp_th / np.sum(exp_th)
            opt_alpha = float(res.x[n_models])
            opt_beta = float(res.x[n_models + 1])
            return opt_w, opt_alpha, opt_beta
    except Exception:
        pass

    return np.ones(n_models, dtype=np.float32) / float(n_models), 0.0, 1.0


def fit_and_score_risk(
    cohort_spec: Any,
    train_records: Sequence[Any],
    eval_records: Sequence[Any],
) -> Dict[str, float]:
    """Stage 1: Multimodal Tensor Factorization & Stratified Convex-Calibrated Ensemble."""
    # 1. Feature Engineering with Interaction Syndromes + Discordance Tensor
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    x_train_extra = _extract_tensor_dissonance_features(train_records)
    x_eval_extra = _extract_tensor_dissonance_features(eval_records)

    x_tr_merged = np.nan_to_num(np.hstack([x_train_raw, x_train_extra]), nan=0.0, posinf=1.0, neginf=-1.0)
    x_ev_merged = np.nan_to_num(np.hstack([x_eval_raw, x_eval_extra]), nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.asarray(y_train, dtype=np.int32)

    # 2. Latent Factor Tensor Decomposition (Truncated SVD)
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr_merged)
    x_ev_scaled = scaler.transform(x_ev_merged)

    n_latent = min(6, x_tr_scaled.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_latent, random_state=42)
    svd_tr = svd.fit_transform(x_tr_scaled)
    svd_ev = svd.transform(x_ev_scaled)

    # Full Feature Tensor: Merged Raw + Latent Syndrome Components
    x_tr_full = np.hstack([x_tr_merged, svd_tr])
    x_ev_full = np.hstack([x_ev_merged, svd_ev])
    x_tr_scaled_full = np.hstack([x_tr_scaled, svd_tr])
    x_ev_scaled_full = np.hstack([x_ev_scaled, svd_ev])

    # 3. Accelerated Failure-Time (AFT) Hazard Kernel Sample Weights
    sample_weights = np.ones(len(train_records), dtype=np.float32)
    for i, p in enumerate(train_records):
        if int(getattr(p, "outcome_label", 0) or 0) == 1:
            tt = float(getattr(p, "time_to_event_days", 15.0) or 15.0)
            # High weight on rapid early readmissions (acute failure within 10 days)
            sample_weights[i] = 1.15 + 0.35 * np.exp(-max(0.0, tt) / 10.0)

    # 4. Stratified 5-Fold Cross-Validated Stacking (OOF Generation & Bagging)
    n_pos = int(np.sum(y_train == 1))
    n_splits = max(2, min(5, n_pos))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(train_records), dtype=np.float32)
    oof_xgb = np.zeros(len(train_records), dtype=np.float32)
    oof_et = np.zeros(len(train_records), dtype=np.float32)
    oof_lr = np.zeros(len(train_records), dtype=np.float32)

    test_lgb = np.zeros(len(eval_records), dtype=np.float32)
    test_xgb = np.zeros(len(eval_records), dtype=np.float32)
    test_et = np.zeros(len(eval_records), dtype=np.float32)
    test_lr = np.zeros(len(eval_records), dtype=np.float32)

    for tr_idx, va_idx in skf.split(x_tr_full, y_train):
        x_tr_f, y_tr_f = x_tr_full[tr_idx], y_train[tr_idx]
        x_va_f = x_tr_full[va_idx]
        w_tr_f = sample_weights[tr_idx]

        # Model 1: LightGBM Classifier
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=20,
            max_depth=5,
            min_child_samples=16,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr_f, y_tr_f, sample_weight=w_tr_f)
        oof_lgb[va_idx] = clf_lgb.predict_proba(x_va_f)[:, 1]
        test_lgb += clf_lgb.predict_proba(x_ev_full)[:, 1] / float(n_splits)

        # Model 2: XGBoost Classifier
        clf_xgb = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        clf_xgb.fit(x_tr_f, y_tr_f, sample_weight=w_tr_f)
        oof_xgb[va_idx] = clf_xgb.predict_proba(x_va_f)[:, 1]
        test_xgb += clf_xgb.predict_proba(x_ev_full)[:, 1] / float(n_splits)

        # Model 3: ExtraTrees Randomized Ensemble
        clf_et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,
        )
        clf_et.fit(x_tr_f, y_tr_f, sample_weight=w_tr_f)
        oof_et[va_idx] = clf_et.predict_proba(x_va_f)[:, 1]
        test_et += clf_et.predict_proba(x_ev_full)[:, 1] / float(n_splits)

        # Model 4: L2-Regularized Logistic Regression on Scaled Latent Representation
        clf_lr = LogisticRegression(C=0.5, max_iter=300, solver="lbfgs", random_state=42)
        clf_lr.fit(x_tr_scaled_full[tr_idx], y_tr_f, sample_weight=w_tr_f)
        oof_lr[va_idx] = clf_lr.predict_proba(x_tr_scaled_full[va_idx])[:, 1]
        test_lr += clf_lr.predict_proba(x_ev_scaled_full)[:, 1] / float(n_splits)

    # 5. Optimal Convex Simplex Meta-Learner
    train_api = np.array([float(getattr(p, "baseline_api_prob", 0.2)) for p in train_records], dtype=np.float32)
    eval_api = np.array([float(getattr(p, "baseline_api_prob", 0.2)) for p in eval_records], dtype=np.float32)

    z_tr = np.column_stack([
        [_bounded_logit(v) for v in oof_lgb],
        [_bounded_logit(v) for v in oof_xgb],
        [_bounded_logit(v) for v in oof_et],
        [_bounded_logit(v) for v in oof_lr],
        [_bounded_logit(v) for v in train_api],
    ])
    z_ev = np.column_stack([
        [_bounded_logit(v) for v in test_lgb],
        [_bounded_logit(v) for v in test_xgb],
        [_bounded_logit(v) for v in test_et],
        [_bounded_logit(v) for v in test_lr],
        [_bounded_logit(v) for v in eval_api],
    ])

    opt_weights, opt_alpha, opt_beta = _optimize_blend_weights(z_tr, y_train)
    eval_logits = opt_alpha + opt_beta * (z_ev @ opt_weights)
    stacked_probs = 1.0 / (1.0 + np.exp(-np.clip(eval_logits, -14.0, 14.0)))

    # 6. Native Toolkit RiskBlender Hybridization
    blender_preds = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.38,
            xgb_weight=0.22,
            baseline_prior_weight=0.05,
            random_state=42,
        )
        blender_preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_raw,
            y_train=y_train,
            x_eval=x_eval_raw,
        )
    except Exception:
        blender_preds = None

    # 7. Dual-Engine Synthesis & Prevalence Calibration Centering
    train_prevalence = float(np.mean(y_train))
    comb_logits = []
    for idx, p in enumerate(eval_records):
        p_stk = float(stacked_probs[idx])
        if blender_preds is not None and p.patient_id in blender_preds:
            p_bln = float(blender_preds[p.patient_id])
            l_val = 0.65 * _bounded_logit(p_stk) + 0.35 * _bounded_logit(p_bln)
        else:
            l_val = _bounded_logit(p_stk)
        comb_logits.append(l_val)

    comb_probs = np.array([safe_sigmoid(lv) for lv in comb_logits], dtype=np.float32)
    mean_comb = float(np.mean(comb_probs))
    if abs(mean_comb - train_prevalence) > 0.008:
        # Monotonic logit shift aligns calibration without disturbing ROC-AUC ranking
        shift = _bounded_logit(train_prevalence) - _bounded_logit(mean_comb)
        comb_probs = np.array([safe_sigmoid(_bounded_logit(pr) + 0.40 * shift) for pr in comb_probs])

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(comb_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: Any,
    eval_records: Sequence[Any],
    predicted_risks: Dict[str, float],
) -> Any:
    """Stage 2: Multi-heuristic Knapsack Optimization maximizing cohort Net ROI subject to all hard constraints."""
    strategies = ["damped_roi", "marginal_roi", "pure_benefit", "risk_cascade"]
    best_state: Optional[Any] = None
    best_total_benefit = -1.0

    for strat in strategies:
        state = CareAllocationState(cohort_spec, eval_records)
        assigned = set()
        total_benefit = 0.0

        if strat == "risk_cascade":
            ranked_patients = sorted(
                eval_records,
                key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
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
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                for tier in tiers_by_rrr:
                    if state.fits(p.patient_id, tier):
                        enb = state.expected_net_benefit(p.patient_id, risk, tier)
                        if enb > 0.0:
                            if state.assign(p.patient_id, tier):
                                assigned.add(p.patient_id)
                                total_benefit += enb
                            break
        else:
            candidates = []
            for p in eval_records:
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                for tier, spec in cohort_spec.interventions.items():
                    if tier == InterventionTier.NONE:
                        continue
                    enb = state.expected_net_benefit(p.patient_id, risk, tier)
                    if enb > 0.0:
                        cost = max(1.0, float(getattr(spec, "cost_usd", 1.0) or 1.0))
                        if strat == "marginal_roi":
                            prio = enb / (cost + 1.0)
                        elif strat == "damped_roi":
                            prio = enb / np.sqrt(cost + 5.0)
                        else:  # pure_benefit
                            prio = enb
                        candidates.append((prio, enb, p.patient_id, tier))

            candidates.sort(key=lambda x: -x[0])
            for prio, enb, pid, tier in candidates:
                if pid in assigned:
                    continue
                if state.fits(pid, tier):
                    if state.assign(pid, tier):
                        assigned.add(pid)
                        total_benefit += enb

        if state.all_ok() and total_benefit > best_total_benefit:
            best_total_benefit = total_benefit
            best_state = state

    if best_state is not None:
        return best_state.to_plan()

    fallback = CareAllocationState(cohort_spec, eval_records)
    return fallback.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
