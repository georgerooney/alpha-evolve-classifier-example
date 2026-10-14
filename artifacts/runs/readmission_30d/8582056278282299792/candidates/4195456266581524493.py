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
    """Stage 1: Chrono-syndromic hazard weighting, submodel discordance tensor, and multi-model logit fusion."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    # 1. Base syndrome features from frozen toolkit
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)
    y_train = np.asarray(y_train_raw, dtype=np.float64)

    # 2. Extract Tri-Domain Clinical Discordance and Acuity Interaction Features
    def _extract_engineered_signals(records: Sequence[PatientRecord]) -> np.ndarray:
        feats = []
        for p in records:
            b_prob = float(getattr(p, "baseline_api_prob", 0.2))
            sub = getattr(p, "baseline_submodel_scores", {}) or {}
            c_lin = float(sub.get("claims_linear", b_prob))
            e_vit = float(sub.get("ehr_vitals", b_prob))
            u_rul = float(sub.get("utilization_rules", b_prob))
            acuity = float(getattr(p, "acuity_score", 0.5))

            l_base = safe_logit(max(1e-4, min(1.0 - 1e-4, b_prob)))
            l_c = safe_logit(max(1e-4, min(1.0 - 1e-4, c_lin)))
            l_v = safe_logit(max(1e-4, min(1.0 - 1e-4, e_vit)))
            l_u = safe_logit(max(1e-4, min(1.0 - 1e-4, u_rul)))

            # Tri-Domain Discordance Tensor
            d_vc = l_v - l_c
            d_uc = l_u - l_c
            d_vu = l_v - l_u
            disc_norm = (d_vc * d_vc + d_uc * d_uc + d_vu * d_vu) ** 0.5

            # Acuity Interactions (counteract elderly elective over-prediction)
            acuity_vitals = acuity * l_v
            acuity_claims = acuity * l_c
            acuity_base = acuity * l_base

            feats.append([
                b_prob, c_lin, e_vit, u_rul, acuity,
                l_base, l_c, l_v, l_u,
                d_vc, d_uc, d_vu, disc_norm,
                acuity_vitals, acuity_claims, acuity_base,
            ])
        return np.array(feats, dtype=np.float64)

    f_train = _extract_engineered_signals(train_records)
    f_eval = _extract_engineered_signals(eval_records)

    x_train_aug = np.nan_to_num(np.hstack([x_train, f_train]), nan=0.0, posinf=1.0, neginf=-1.0)
    x_eval_aug = np.nan_to_num(np.hstack([x_eval, f_eval]), nan=0.0, posinf=1.0, neginf=-1.0)

    # 3. Chrono-Velocity Survival Hazard Weighting
    sample_weights = np.ones(len(train_records), dtype=np.float64)
    for i, p in enumerate(train_records):
        if y_train[i] > 0.5:
            t = getattr(p, "time_to_event_days", None)
            if t is not None:
                sample_weights[i] = 1.0 + 0.8 * np.exp(-max(0.0, float(t)) / 10.0)
            else:
                sample_weights[i] = 1.4
    sample_weights = sample_weights / max(1e-6, float(np.mean(sample_weights)))

    # Scaled representations for linear components
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train_aug)
    x_ev_scaled = scaler.transform(x_eval_aug)

    model_preds = []
    model_weights = []

    # 4. Model A: clinical_toolkit RiskBlender
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.08,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
        )
        rb_probs = np.array([float(rb_dict[p.patient_id]) for p in eval_records], dtype=np.float64)
        model_preds.append(rb_probs)
        model_weights.append(0.35)
    except Exception:
        pass

    # 5. Model B: LightGBM with tree depth regularization and survival weighting
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=5,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        lgb_clf.fit(x_train_aug, y_train, sample_weight=sample_weights)
        lgb_probs = lgb_clf.predict_proba(x_eval_aug)[:, 1]
        model_preds.append(lgb_probs)
        model_weights.append(0.25)
    except Exception:
        pass

    # 6. Model C: HistGradientBoosting
    try:
        hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        )
        hgb.fit(x_train_aug, y_train, sample_weight=sample_weights)
        hgb_probs = hgb.predict_proba(x_eval_aug)[:, 1]
        model_preds.append(hgb_probs)
        model_weights.append(0.15)
    except Exception:
        pass

    # 7. Model D: Acuity-Recalibrated Baseline Logistic Model
    try:
        scaler_f = StandardScaler()
        f_tr_s = scaler_f.fit_transform(f_train)
        f_ev_s = scaler_f.transform(f_eval)
        lr_base = LogisticRegression(C=0.5, max_iter=300, random_state=42)
        lr_base.fit(f_tr_s, y_train, sample_weight=sample_weights)
        recal_base_probs = lr_base.predict_proba(f_ev_s)[:, 1]
        model_preds.append(recal_base_probs)
        model_weights.append(0.15)
    except Exception:
        pass

    # 8. Model E: Regularized Logistic Regression on Full Scaled Space
    try:
        lr_all = LogisticRegression(C=0.25, max_iter=300, random_state=42)
        lr_all.fit(x_tr_scaled, y_train, sample_weight=sample_weights)
        lr_probs = lr_all.predict_proba(x_ev_scaled)[:, 1]
        model_preds.append(lr_probs)
        model_weights.append(0.10)
    except Exception:
        pass

    # Fallback guard
    if not model_preds:
        model_preds.append(np.array([float(p.baseline_api_prob) for p in eval_records], dtype=np.float64))
        model_weights.append(1.0)

    # 9. Multi-Model Logit Fusion & Prevalence Calibration
    norm_w = np.array(model_weights, dtype=np.float64) / sum(model_weights)
    blended_logits = np.zeros(len(eval_records), dtype=np.float64)
    for w, preds in zip(norm_w, model_preds):
        clipped = np.clip(preds, 1e-4, 1.0 - 1e-4)
        logits = np.log(clipped / (1.0 - clipped))
        blended_logits += w * logits

    # Monotonic prevalence logit shift (sharpens Brier score while preserving exact AUC ranks)
    y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))
    target_logit = safe_logit(y_mean)
    current_mean_logit = float(np.mean(blended_logits))
    calibrated_logits = blended_logits + 0.45 * (target_logit - current_mean_logit)
    final_probs = safe_sigmoid(calibrated_logits)

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        p_val = float(final_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, p_val)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Marginal Net Benefit Maximizing Knapsack Allocation."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_tier = None
        best_net_benefit = 0.0
        best_cost = float("inf")

        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                cost = cohort_spec.interventions[tier].cost_usd
                if nb > 0.0:
                    # Select the tier maximizing expected net benefit; break ties with lower cost
                    if nb > best_net_benefit + 1e-3:
                        best_net_benefit = nb
                        best_tier = tier
                        best_cost = cost
                    elif abs(nb - best_net_benefit) <= 1e-3 and cost < best_cost:
                        best_net_benefit = nb
                        best_tier = tier
                        best_cost = cost

        if best_tier is not None and best_net_benefit > 0.0:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
