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


import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def _extract_discrepancy_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract clinical submodel discrepancy, geometric consensus, and syndrome interaction features."""
    rows = []
    for p in records:
        b_api = float(getattr(p, "baseline_api_prob", 0.5))
        subs = getattr(p, "baseline_submodel_scores", {}) or {}
        claims = float(subs.get("claims_linear", b_api))
        vitals = float(subs.get("ehr_vitals", b_api))
        util = float(subs.get("utilization_rules", b_api))
        acuity = float(getattr(p, "acuity_score", 0.5))
        n_contra = float(len(getattr(p, "contraindications", ())))

        # Acute bedside shock vs chronic billing discrepancy
        vitals_claims_diff = vitals - claims
        # SDoH / behavioral utilization vs clinical claims discrepancy
        util_claims_diff = util - claims
        # Bedside instability vs utilization discrepancy
        vitals_util_diff = vitals - util

        # Elderly low-acuity elective correction signal:
        # High claims comorbidity + low acuity + low bedside distress = elective over-prediction!
        elective_overpred = claims * (1.0 - vitals) * max(0.0, 1.0 - acuity)
        elective_overpred_sq = claims * (max(0.0, 1.0 - acuity) ** 2)
        claims_acuity_ratio = claims / (acuity + 0.05)

        # Acute crash signal: high bedside vitals + high acuity
        acute_crash = vitals * acuity
        vitals_claims_ratio = vitals / (claims + 0.05)

        # SDoH spiral signal: high utilization + high baseline risk
        sdoh_spiral = util * b_api
        sdoh_util_spike = util * (1.0 - min(claims, vitals))
        sdoh_claims_ratio = util / (claims + 0.05)

        # Multi-model consensus geometry
        sub_mean = (claims + vitals + util) / 3.0
        sub_std = float(np.std([claims, vitals, util]))
        sub_spread = max(claims, vitals, util) - min(claims, vitals, util)
        sub_min = min(claims, vitals, util)
        sub_max = max(claims, vitals, util)
        tri_synergy = claims * vitals * util

        # Safe logit features (log-odds evidence)
        l_api = safe_logit(max(1e-4, min(1.0 - 1e-4, b_api)))
        l_claims = safe_logit(max(1e-4, min(1.0 - 1e-4, claims)))
        l_vitals = safe_logit(max(1e-4, min(1.0 - 1e-4, vitals)))
        l_util = safe_logit(max(1e-4, min(1.0 - 1e-4, util)))
        l_acuity = safe_logit(max(1e-4, min(1.0 - 1e-4, acuity)))

        # Logit-domain relative evidence
        dl_vitals_claims = l_vitals - l_claims
        dl_util_claims = l_util - l_claims
        dl_vitals_util = l_vitals - l_util
        bayesian_evidence_sum = l_claims + l_vitals + l_util

        # Contraindication interaction with clinical acuity
        contra_acuity = n_contra * acuity
        contra_vitals = n_contra * vitals

        rows.append([
            b_api, claims, vitals, util, acuity, n_contra,
            vitals_claims_diff, util_claims_diff, vitals_util_diff,
            elective_overpred, elective_overpred_sq, claims_acuity_ratio,
            acute_crash, vitals_claims_ratio,
            sdoh_spiral, sdoh_util_spike, sdoh_claims_ratio,
            sub_mean, sub_std, sub_spread, sub_min, sub_max, tri_synergy,
            l_api, l_claims, l_vitals, l_util, l_acuity,
            dl_vitals_claims, dl_util_claims, dl_vitals_util, bayesian_evidence_sum,
            contra_acuity, contra_vitals,
        ])
    return np.array(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Disjoint Temporal-Syndromic Factorization (DTSF) with Transductive Rank-Matching Calibration."""
    # 1. Non-linear cross-organ syndrome interaction features
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    # 2. Extract discrepancy and submodel consensus features
    x_train_disc = _extract_discrepancy_features(train_records)
    x_eval_disc = _extract_discrepancy_features(eval_records)

    x_train_full = np.nan_to_num(np.hstack([x_train_raw, x_train_disc]), nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_full = np.nan_to_num(np.hstack([x_eval_raw, x_eval_disc]), nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Disjoint Temporal Factorization: partition 30-day readmissions into Acute vs Subacute
    # Early Acute Crash (t <= 10 days) vs Delayed Subacute SDoH breakdown (10 < t <= 30 days)
    y_train_acute = np.zeros(len(train_records), dtype=np.int32)
    y_train_subacute = np.zeros(len(train_records), dtype=np.int32)
    for idx, p in enumerate(train_records):
        if int(p.outcome_label) == 1:
            t_days = float(getattr(p, "time_to_event_days", 15.0) or 15.0)
            if t_days <= 10.0:
                y_train_acute[idx] = 1
            else:
                y_train_subacute[idx] = 1

    # 4. Fit RiskBlender baseline ensemble
    blender_preds = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.35,
            xgb_weight=0.20,
            baseline_prior_weight=0.10,
            random_state=42,
        )
        blender_preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_full,
            y_train=y_train,
            x_eval=x_eval_full,
        )
    except Exception:
        try:
            blender = RiskBlender(random_state=42)
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

    # 5. Model Head 1: Holistic Generalist LightGBM
    lgb_full_probs = None
    try:
        lgb_full = lgb.LGBMClassifier(
            n_estimators=110,
            learning_rate=0.04,
            num_leaves=14,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        lgb_full.fit(x_train_full, y_train)
        lgb_full_probs = lgb_full.predict_proba(x_eval_full)[:, 1]
    except Exception:
        pass

    # 6. Model Head 2: Holistic Generalist XGBoost
    xgb_probs = None
    try:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=90,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
        xgb_clf.fit(x_train_full, y_train)
        xgb_probs = xgb_clf.predict_proba(x_eval_full)[:, 1]
    except Exception:
        pass

    # 7. Model Head 3 & 4: Specialist Disjoint Temporal Decomposition
    # Specialist Head A: Acute Bedside Crash Hazard (t <= 10 days)
    acute_probs = None
    if np.sum(y_train_acute) >= 4:
        try:
            lgb_acute = lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.045,
                num_leaves=10,
                max_depth=3,
                subsample=0.80,
                colsample_bytree=0.80,
                min_child_samples=8,
                random_state=101,
                verbose=-1,
                n_jobs=1,
            )
            lgb_acute.fit(x_train_full, y_train_acute)
            acute_probs = lgb_acute.predict_proba(x_eval_full)[:, 1]
        except Exception:
            pass

    # Specialist Head B: Subacute SDoH Breakdown Hazard (t > 10 days)
    subacute_probs = None
    if np.sum(y_train_subacute) >= 4:
        try:
            lgb_subacute = lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.045,
                num_leaves=10,
                max_depth=3,
                subsample=0.80,
                colsample_bytree=0.80,
                min_child_samples=8,
                random_state=202,
                verbose=-1,
                n_jobs=1,
            )
            lgb_subacute.fit(x_train_full, y_train_subacute)
            subacute_probs = lgb_subacute.predict_proba(x_eval_full)[:, 1]
        except Exception:
            pass

    # Reconstructed disjoint probability: P(Acute U Subacute) = P(Acute) + P(Subacute)
    disjoint_probs = None
    if acute_probs is not None and subacute_probs is not None:
        disjoint_probs = np.clip(acute_probs + subacute_probs, 1e-4, 1.0 - 1e-4)

    # 8. Regularized Logistic Regression on Scaled Syndrome Space
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train_full)
    x_ev_scaled = scaler.transform(x_eval_full)
    lr_clf = LogisticRegression(C=0.30, max_iter=300, random_state=42)
    lr_clf.fit(x_tr_scaled, y_train)
    lr_probs = lr_clf.predict_proba(x_ev_scaled)[:, 1]

    # 9. Empirical Target Rate Anchor
    train_mean_rate = float(np.clip(np.mean(y_train), 0.05, 0.95))

    # 10. Multi-View Ensemble Consensus Blending
    out: Dict[str, float] = {}
    raw_ensemble_probs = []

    for idx, p in enumerate(eval_records):
        logit_lr = safe_logit(float(lr_probs[idx]))
        logit_lgb = safe_logit(float(lgb_full_probs[idx])) if lgb_full_probs is not None else logit_lr
        logit_xgb = safe_logit(float(xgb_probs[idx])) if xgb_probs is not None else logit_lr
        logit_disjoint = safe_logit(float(disjoint_probs[idx])) if disjoint_probs is not None else logit_lgb

        b_api_val = float(getattr(p, "baseline_api_prob", 0.5))
        acuity_val = float(getattr(p, "acuity_score", 0.5))
        # Correct elderly low-acuity elective overprediction on baseline prior
        subs = getattr(p, "baseline_submodel_scores", {}) or {}
        claims_val = float(subs.get("claims_linear", b_api_val))
        vitals_val = float(subs.get("ehr_vitals", b_api_val))
        elective_discount = max(0.0, claims_val - max(vitals_val, acuity_val)) * 0.4
        corrected_b_api = max(1e-4, min(1.0 - 1e-4, b_api_val - elective_discount))
        logit_base = safe_logit(corrected_b_api)

        if blender_preds is not None and p.patient_id in blender_preds:
            logit_blender = safe_logit(float(blender_preds[p.patient_id]))
            blended = (
                0.28 * logit_blender
                + 0.22 * logit_lgb
                + 0.20 * logit_xgb
                + 0.16 * logit_disjoint
                + 0.08 * logit_lr
                + 0.06 * logit_base
            )
        else:
            blended = (
                0.32 * logit_lgb
                + 0.26 * logit_xgb
                + 0.22 * logit_disjoint
                + 0.12 * logit_lr
                + 0.08 * logit_base
            )

        prob = safe_sigmoid(blended)
        raw_ensemble_probs.append(float(prob))

    # 11. Transductive Rank-Preserving Calibration:
    # Calibrate probability mean to match cohort baseline empirical rate, minimizing Brier score
    raw_ensemble_probs = np.array(raw_ensemble_probs, dtype=np.float32)
    current_eval_mean = float(np.mean(raw_ensemble_probs))
    if current_eval_mean > 1e-4 and (1.0 - current_eval_mean) > 1e-4:
        # Centering adjustment in logit domain (temperature & shift invariant ranking)
        shift = safe_logit(train_mean_rate) - safe_logit(current_eval_mean)
        calibrated_logits = np.array([safe_logit(float(pr)) + 0.55 * shift for pr in raw_ensemble_probs])
        calibrated_probs = safe_sigmoid(calibrated_logits)
    else:
        calibrated_probs = raw_ensemble_probs

    for idx, p in enumerate(eval_records):
        prob_val = float(calibrated_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob_val)), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Multiple-Choice Knapsack Optimization via marginal net-benefit-to-cost efficiency."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]
    if not active_tiers:
        return state.to_plan()

    tier_specs = {t: cohort_spec.interventions[t] for t in active_tiers}
    costs = {t: float(getattr(tier_specs[t], "cost_usd", 0.0)) for t in active_tiers}
    costs[InterventionTier.NONE] = 0.0
    max_slots = {t: int(getattr(tier_specs[t], "max_slots", 0)) for t in active_tiers}
    total_budget = float(cohort_spec.total_budget_usd)

    # Determine eligibility and positive net benefits for each patient
    patient_tier_benefits: Dict[str, Dict[InterventionTier, float]] = {}
    for p in eval_records:
        p_acuity = float(getattr(p, "acuity_score", 0.0))
        p_contra = set(getattr(p, "contraindications", ()))
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))

        p_benefits = {InterventionTier.NONE: 0.0}
        for t in active_tiers:
            spec = tier_specs[t]
            min_acuity = float(getattr(spec, "min_acuity_score", 0.0))
            contra_flags = set(getattr(spec, "contraindicated_flags", ()))
            if p_acuity >= min_acuity and not (p_contra & contra_flags):
                nb = state.expected_net_benefit(p.patient_id, risk, t)
                if nb > 0.0:
                    p_benefits[t] = float(nb)
        patient_tier_benefits[p.patient_id] = p_benefits

    # Track current assignments, slot consumption, and budget expenditure
    assigned: Dict[str, InterventionTier] = {p.patient_id: InterventionTier.NONE for p in eval_records}
    current_slots: Dict[InterventionTier, int] = {t: 0 for t in active_tiers}
    current_spent = 0.0

    # Iterative Multiple-Choice Knapsack Greedy Upgrade Loop
    while True:
        best_p_id = None
        best_cand_tier = None
        best_delta_b = 0.0
        best_delta_c = 0.0
        best_efficiency = -1.0

        for p in eval_records:
            pid = p.patient_id
            curr_tier = assigned[pid]
            curr_b = patient_tier_benefits[pid].get(curr_tier, 0.0)
            curr_c = costs[curr_tier]

            for cand_tier, cand_b in patient_tier_benefits[pid].items():
                if cand_tier == curr_tier:
                    continue
                delta_b = cand_b - curr_b
                if delta_b <= 0.0:
                    continue

                delta_c = costs[cand_tier] - curr_c
                if current_spent + delta_c > total_budget:
                    continue
                if cand_tier != InterventionTier.NONE and current_slots[cand_tier] + 1 > max_slots[cand_tier]:
                    continue

                eff = delta_b / ((delta_c + 15.0) ** 0.8) if delta_c > 0.0 else delta_b * 1e6
                if eff > best_efficiency:
                    best_efficiency = eff
                    best_p_id = pid
                    best_cand_tier = cand_tier
                    best_delta_b = delta_b
                    best_delta_c = delta_c

        if best_p_id is None or best_cand_tier is None or best_efficiency <= 0.0:
            break

        old_tier = assigned[best_p_id]
        if old_tier != InterventionTier.NONE:
            current_slots[old_tier] -= 1
        if best_cand_tier != InterventionTier.NONE:
            current_slots[best_cand_tier] += 1

        assigned[best_p_id] = best_cand_tier
        current_spent += best_delta_c

    # Commit feasible assignments to CareAllocationState prioritized by net benefit
    sorted_pids = sorted(
        eval_records,
        key=lambda p: patient_tier_benefits[p.patient_id].get(assigned[p.patient_id], 0.0),
        reverse=True,
    )
    for p in sorted_pids:
        tier = assigned[p.patient_id]
        if tier != InterventionTier.NONE:
            if state.fits(p.patient_id, tier):
                state.assign(p.patient_id, tier)

    plan = state.to_plan()
    # Guarantee constraint 1: Every patient in eval_records is assigned a valid tier
    for p in eval_records:
        if p.patient_id not in plan.assignments:
            plan.assignments[p.patient_id] = InterventionTier.NONE
    return plan


# EVOLVE-BLOCK-END
# =====================================================================
