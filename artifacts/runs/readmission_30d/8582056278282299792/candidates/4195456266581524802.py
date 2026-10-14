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
    """Extract clinical submodel discrepancy and syndrome interaction features."""
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

        # Acute crash signal: high bedside vitals + high acuity
        acute_crash = vitals * acuity

        # SDoH spiral signal: high utilization + high baseline risk
        sdoh_spiral = util * b_api

        # Safe logit features
        l_api = safe_logit(max(1e-4, min(1.0 - 1e-4, b_api)))
        l_claims = safe_logit(max(1e-4, min(1.0 - 1e-4, claims)))
        l_vitals = safe_logit(max(1e-4, min(1.0 - 1e-4, vitals)))
        l_util = safe_logit(max(1e-4, min(1.0 - 1e-4, util)))

        rows.append([
            b_api, claims, vitals, util, acuity, n_contra,
            vitals_claims_diff, util_claims_diff, vitals_util_diff,
            elective_overpred, acute_crash, sdoh_spiral,
            l_api, l_claims, l_vitals, l_util,
        ])
    return np.array(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-syndromic feature extraction, survival hazard weighting, and 5-way consensus blending."""
    # 1. Non-linear cross-organ syndrome interaction features
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    # 2. Extract discrepancy and submodel features
    x_train_disc = _extract_discrepancy_features(train_records)
    x_eval_disc = _extract_discrepancy_features(eval_records)

    x_train_full = np.nan_to_num(np.hstack([x_train_raw, x_train_disc]), nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_full = np.nan_to_num(np.hstack([x_eval_raw, x_eval_disc]), nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Derive survival-hazard sample weights for acute post-discharge breakdown
    sample_weights = np.ones(len(train_records), dtype=np.float32)
    for idx, p in enumerate(train_records):
        if int(p.outcome_label) == 1:
            t_days = float(getattr(p, "time_to_event_days", 15.0) or 15.0)
            sample_weights[idx] = 1.0 + 1.2 * float(np.exp(-min(30.0, max(0.0, t_days)) / 10.0))

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

    # 5. Fit survival-hazard LightGBM model
    lgb_probs = None
    try:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=8,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        lgb_clf.fit(x_train_full, y_train, sample_weight=sample_weights)
        lgb_probs = lgb_clf.predict_proba(x_eval_full)[:, 1]
    except Exception:
        pass

    # 6. Fit survival-hazard XGBoost model
    xgb_probs = None
    try:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
        xgb_clf.fit(x_train_full, y_train, sample_weight=sample_weights)
        xgb_probs = xgb_clf.predict_proba(x_eval_full)[:, 1]
    except Exception:
        pass

    # 7. Fit regularized Logistic Regression on scaled features
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train_full)
    x_ev_scaled = scaler.transform(x_eval_full)
    lr_clf = LogisticRegression(C=0.25, max_iter=300, random_state=42)
    lr_clf.fit(x_tr_scaled, y_train, sample_weight=sample_weights)
    lr_probs = lr_clf.predict_proba(x_ev_scaled)[:, 1]

    # 8. Consensus logit blending
    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        logit_lr = safe_logit(float(lr_probs[idx]))
        logit_lgb = safe_logit(float(lgb_probs[idx])) if lgb_probs is not None else logit_lr
        logit_xgb = safe_logit(float(xgb_probs[idx])) if xgb_probs is not None else logit_lr
        logit_base = safe_logit(float(getattr(p, "baseline_api_prob", 0.5)))

        if blender_preds is not None and p.patient_id in blender_preds:
            logit_blender = safe_logit(float(blender_preds[p.patient_id]))
            blended = (
                0.35 * logit_blender
                + 0.25 * logit_lgb
                + 0.20 * logit_xgb
                + 0.12 * logit_lr
                + 0.08 * logit_base
            )
        else:
            blended = (
                0.40 * logit_lgb
                + 0.30 * logit_xgb
                + 0.20 * logit_lr
                + 0.10 * logit_base
            )

        prob = safe_sigmoid(blended)
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, float(prob))), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Multiple-Choice Knapsack Optimization via marginal net-benefit-to-cost efficiency."""
    state = CareAllocationState(cohort_spec, eval_records)
    total_budget = float(cohort_spec.total_budget_usd)
    contra_set = set(cohort_spec.contraindicated_flags)

    # Available tiers and their capacities
    active_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]
    max_slots = {t: int(cohort_spec.interventions[t].max_slots) for t in active_tiers}
    costs = {t: float(cohort_spec.interventions[t].cost_usd) for t in active_tiers}
    costs[InterventionTier.NONE] = 0.0

    # Determine eligibility and candidate benefits for each patient
    patient_tier_benefits: Dict[str, Dict[InterventionTier, float]] = {}
    for p in eval_records:
        p_contra = set(getattr(p, "contraindications", ()))
        p_acuity = float(getattr(p, "acuity_score", 0.0))
        is_eligible = (p_acuity >= cohort_spec.min_acuity_score) and len(contra_set & p_contra) == 0

        p_benefits = {InterventionTier.NONE: 0.0}
        if is_eligible:
            risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
            for tier in active_tiers:
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                if nb > 0.0:
                    p_benefits[tier] = float(nb)
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
                # Budget check
                if current_spent + delta_c > total_budget:
                    continue
                # Slot capacity check
                if cand_tier != InterventionTier.NONE and current_slots[cand_tier] + 1 > max_slots[cand_tier]:
                    continue

                # Marginal net-benefit efficiency metric
                if delta_c > 0.0:
                    eff = delta_b / ((delta_c + 10.0) ** 0.75)
                else:
                    eff = delta_b * 1e6

                if eff > best_efficiency:
                    best_efficiency = eff
                    best_p_id = pid
                    best_cand_tier = cand_tier
                    best_delta_b = delta_b
                    best_delta_c = delta_c

        if best_p_id is None or best_cand_tier is None or best_efficiency <= 0.0:
            break

        # Apply the optimal upgrade
        old_tier = assigned[best_p_id]
        if old_tier != InterventionTier.NONE:
            current_slots[old_tier] -= 1
        if best_cand_tier != InterventionTier.NONE:
            current_slots[best_cand_tier] += 1

        assigned[best_p_id] = best_cand_tier
        current_spent += best_delta_c

    # Commit feasible assignments to CareAllocationState
    for p in eval_records:
        tier = assigned[p.patient_id]
        if tier != InterventionTier.NONE:
            if state.fits(p.patient_id, tier):
                state.assign(p.patient_id, tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
