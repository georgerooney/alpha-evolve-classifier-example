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

from typing import Dict, Sequence, Optional
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def _vec_logit(p: float, eps: float = 1e-5) -> float:
    """Robust scalar logit transformation."""
    p_clamped = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p_clamped / (1.0 - p_clamped)))


def _extract_discrepancy_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract clinical submodel discrepancy, geometric consensus, and syndrome interaction features."""
    rows = []
    for p in records:
        b_api = float(getattr(p, "baseline_api_prob", 0.5) or 0.5)
        subs = getattr(p, "baseline_submodel_scores", {}) or {}
        claims = float(subs.get("claims_linear", b_api) or b_api)
        vitals = float(subs.get("ehr_vitals", b_api) or b_api)
        util = float(subs.get("utilization_rules", b_api) or b_api)
        acuity = float(getattr(p, "acuity_score", 0.5) or 0.5)
        contra = getattr(p, "contraindications", ()) or ()
        n_contra = float(len(contra))

        # Acute bedside shock vs chronic billing discrepancy
        vitals_claims_diff = vitals - claims
        # SDoH / behavioral utilization vs clinical claims discrepancy
        util_claims_diff = util - claims
        # Bedside instability vs utilization discrepancy
        vitals_util_diff = vitals - util

        # Elderly low-acuity elective correction signal:
        elective_overpred = claims * (1.0 - vitals) * max(0.0, 1.0 - acuity)
        elective_overpred_sq = claims * (max(0.0, 1.0 - acuity) ** 2)
        claims_acuity_ratio = claims / (acuity + 0.05)

        # Acute crash signal: high bedside vitals + high acuity
        acute_crash = vitals * acuity * (1.0 + max(0.0, vitals - claims))
        vitals_claims_ratio = vitals / (claims + 0.05)

        # SDoH spiral signal: high utilization + high baseline risk
        sdoh_spiral = util * b_api * (1.0 + max(0.0, util - claims))
        sdoh_util_spike = util * (1.0 - min(claims, vitals))
        sdoh_claims_ratio = util / (claims + 0.05)

        # Multi-model consensus geometry
        sub_mean = (claims + vitals + util) / 3.0
        sub_std = float(np.std([claims, vitals, util]))
        sub_spread = max(claims, vitals, util) - min(claims, vitals, util)
        sub_min = min(claims, vitals, util)
        sub_max = max(claims, vitals, util)
        tri_synergy = claims * vitals * util
        harmonic_mean = 3.0 / (1.0 / (claims + 0.02) + 1.0 / (vitals + 0.02) + 1.0 / (util + 0.02))

        # Safe logit features (log-odds evidence)
        l_api = _vec_logit(b_api)
        l_claims = _vec_logit(claims)
        l_vitals = _vec_logit(vitals)
        l_util = _vec_logit(util)
        l_acuity = _vec_logit(acuity)

        # Logit-domain relative evidence
        dl_vitals_claims = l_vitals - l_claims
        dl_util_claims = l_util - l_claims
        dl_vitals_util = l_vitals - l_util
        bayesian_evidence_sum = l_claims + l_vitals + l_util

        # Contraindication interaction with clinical acuity
        contra_acuity = n_contra * acuity
        contra_vitals = n_contra * vitals
        contra_acuity_ratio = n_contra / (acuity + 0.1)

        rows.append([
            b_api, claims, vitals, util, acuity, n_contra,
            vitals_claims_diff, util_claims_diff, vitals_util_diff,
            elective_overpred, elective_overpred_sq, claims_acuity_ratio,
            acute_crash, vitals_claims_ratio,
            sdoh_spiral, sdoh_util_spike, sdoh_claims_ratio,
            sub_mean, sub_std, sub_spread, sub_min, sub_max, tri_synergy, harmonic_mean,
            l_api, l_claims, l_vitals, l_util, l_acuity,
            dl_vitals_claims, dl_util_claims, dl_vitals_util, bayesian_evidence_sum,
            contra_acuity, contra_vitals, contra_acuity_ratio,
        ])
    return np.array(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Transductive Survival-Manifold & Discrepancy Ensemble (T-SMART) with Monotonic Rank Calibration."""
    # 1. Non-linear cross-organ syndrome interaction features from frozen toolkit
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    x_train_raw = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_raw = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.array([int(getattr(p, "outcome_label", 0) or 0) for p in train_records], dtype=np.int32)

    # 2. Extract discrepancy, geometric consensus, and log-odds evidence features
    x_train_disc = _extract_discrepancy_features(train_records)
    x_eval_disc = _extract_discrepancy_features(eval_records)

    x_train_full = np.nan_to_num(np.hstack([x_train_raw, x_train_disc]), nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_full = np.nan_to_num(np.hstack([x_eval_raw, x_eval_disc]), nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Continuous Survival Hazard Targets (Decomposing readmission by temporal urgency)
    y_hazard = np.zeros(len(train_records), dtype=np.float32)
    y_acute = np.zeros(len(train_records), dtype=np.int32)
    for idx, p in enumerate(train_records):
        if y_train[idx] == 1:
            t_days = float(getattr(p, "time_to_event_days", 15.0) or 15.0)
            # Hazard target: early acute crash gets much higher target to sharply separate top-risk tier
            y_hazard[idx] = 1.0 + 8.0 / np.sqrt(max(1.0, t_days))
            if t_days <= 10.0:
                y_acute[idx] = 1

    # Standardized feature space for linear models and manifold geometry
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train_full)
    x_ev_scaled = scaler.transform(x_eval_full)

    # 4. Transductive Nearest-Neighbor Manifold Estimator
    knn_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        k_neighbors = min(15, max(3, len(train_records) // 8))
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")
        nn.fit(x_tr_scaled)
        distances, indices = nn.kneighbors(x_ev_scaled)
        weights = np.exp(-2.5 * np.clip(distances, 0.0, 2.0))
        for i in range(len(eval_records)):
            neighbor_labels = y_train[indices[i]]
            p_knn = float(np.sum(weights[i] * neighbor_labels) / (np.sum(weights[i]) + 1e-8))
            knn_logits[i] = _vec_logit(p_knn)
    except Exception:
        knn_logits = None

    # 5. Model Head 1: RiskBlender baseline anchor
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
        pass

    # 6. Model Head 2: Holistic Binary LightGBM
    lgb_clf_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.035,
            num_leaves=10,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        lgb_clf.fit(x_train_full, y_train)
        probs = lgb_clf.predict_proba(x_eval_full)[:, 1]
        lgb_clf_logits = np.array([_vec_logit(pr) for pr in probs], dtype=np.float32)
    except Exception:
        lgb_clf_logits = None

    # 7. Model Head 3: Holistic Binary XGBoost
    xgb_clf_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=85,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
        xgb_clf.fit(x_train_full, y_train)
        probs = xgb_clf.predict_proba(x_eval_full)[:, 1]
        xgb_clf_logits = np.array([_vec_logit(pr) for pr in probs], dtype=np.float32)
    except Exception:
        xgb_clf_logits = None

    # 8. Model Head 4: Continuous Survival Hazard LightGBM Regressor
    lgb_haz_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        lgb_haz = lgb.LGBMRegressor(
            n_estimators=90,
            learning_rate=0.035,
            num_leaves=10,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=10,
            random_state=123,
            verbose=-1,
            n_jobs=1,
        )
        lgb_haz.fit(x_train_full, y_hazard)
        haz_preds = lgb_haz.predict(x_eval_full)
        # Standardize hazard predictions directly into logit domain
        haz_std = float(np.std(haz_preds)) + 1e-4
        lgb_haz_logits = np.clip((haz_preds - float(np.mean(haz_preds))) / haz_std * 1.5, -6.0, 6.0)
    except Exception:
        lgb_haz_logits = None

    # 9. Model Head 5: Continuous Survival Hazard XGBoost Regressor
    xgb_haz_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        xgb_haz = xgb.XGBRegressor(
            n_estimators=75,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=456,
            n_jobs=1,
        )
        xgb_haz.fit(x_train_full, y_hazard)
        haz_preds_xgb = xgb_haz.predict(x_eval_full)
        haz_std_xgb = float(np.std(haz_preds_xgb)) + 1e-4
        xgb_haz_logits = np.clip((haz_preds_xgb - float(np.mean(haz_preds_xgb))) / haz_std_xgb * 1.5, -6.0, 6.0)
    except Exception:
        xgb_haz_logits = None

    # 10. Model Head 6: Acute Specialist LightGBM (t <= 10 days)
    acute_logits = np.zeros(len(eval_records), dtype=np.float32)
    if np.sum(y_acute) >= 4:
        try:
            lgb_acute = lgb.LGBMClassifier(
                n_estimators=70,
                learning_rate=0.04,
                num_leaves=8,
                max_depth=3,
                subsample=0.80,
                colsample_bytree=0.80,
                min_child_samples=8,
                random_state=789,
                verbose=-1,
                n_jobs=1,
            )
            lgb_acute.fit(x_train_full, y_acute)
            probs = lgb_acute.predict_proba(x_eval_full)[:, 1]
            acute_logits = np.array([_vec_logit(pr) for pr in probs], dtype=np.float32)
        except Exception:
            acute_logits = None
    else:
        acute_logits = None

    # 11. Model Head 7: Regularized Logistic Regression on Scaled Features
    lr_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        lr_clf = LogisticRegression(C=0.25, max_iter=300, random_state=42)
        lr_clf.fit(x_tr_scaled, y_train)
        probs = lr_clf.predict_proba(x_ev_scaled)[:, 1]
        lr_logits = np.array([_vec_logit(pr) for pr in probs], dtype=np.float32)
    except Exception:
        lr_logits = None

    # 12. Model Head 8: Scaled Ridge Hazard Estimator
    ridge_logits = np.zeros(len(eval_records), dtype=np.float32)
    try:
        ridge_reg = Ridge(alpha=5.0, random_state=42)
        ridge_reg.fit(x_tr_scaled, y_hazard)
        ridge_preds = ridge_reg.predict(x_ev_scaled)
        r_std = float(np.std(ridge_preds)) + 1e-4
        ridge_logits = np.clip((ridge_preds - float(np.mean(ridge_preds))) / r_std * 1.5, -6.0, 6.0)
    except Exception:
        ridge_logits = None

    # 13. Blend all model logits per patient
    raw_logits_list = []
    for idx, p in enumerate(eval_records):
        b_api_val = float(getattr(p, "baseline_api_prob", 0.5) or 0.5)
        acuity_val = float(getattr(p, "acuity_score", 0.5) or 0.5)
        subs = getattr(p, "baseline_submodel_scores", {}) or {}
        claims_val = float(subs.get("claims_linear", b_api_val) or b_api_val)
        vitals_val = float(subs.get("ehr_vitals", b_api_val) or b_api_val)

        # Correct elderly low-acuity elective overprediction on baseline prior
        elective_discount = max(0.0, claims_val - max(vitals_val, acuity_val)) * 0.4
        corrected_b_api = max(1e-4, min(1.0 - 1e-4, b_api_val - elective_discount))
        logit_base = _vec_logit(corrected_b_api)

        # Assemble component logits with weights
        components = []
        if blender_preds is not None and p.patient_id in blender_preds:
            components.append((0.20, _vec_logit(float(blender_preds[p.patient_id]))))
        if lgb_clf_logits is not None:
            components.append((0.18, float(lgb_clf_logits[idx])))
        if xgb_clf_logits is not None:
            components.append((0.15, float(xgb_clf_logits[idx])))
        if lgb_haz_logits is not None:
            components.append((0.14, float(lgb_haz_logits[idx])))
        if xgb_haz_logits is not None:
            components.append((0.10, float(xgb_haz_logits[idx])))
        if acute_logits is not None:
            components.append((0.08, float(acute_logits[idx])))
        if lr_logits is not None:
            components.append((0.06, float(lr_logits[idx])))
        if ridge_logits is not None:
            components.append((0.04, float(ridge_logits[idx])))
        if knn_logits is not None:
            components.append((0.03, float(knn_logits[idx])))
        components.append((0.02, logit_base))

        tot_w = sum(w for w, _ in components)
        blended_logit = sum(w * val for w, val in components) / tot_w
        raw_logits_list.append(blended_logit)

    raw_logits_arr = np.array(raw_logits_list, dtype=np.float32)

    # 14. Transductive Monotonic Rank-Preserving Calibration via Bisection:
    # Solves for delta such that mean(sigmoid(logit + delta)) == target_prevalence.
    # Strictly rank-preserving: ROC-AUC and PR-AUC are mathematically invariant!
    target_prevalence = float(np.clip(np.mean(y_train), 0.05, 0.95))
    low_shift, high_shift = -6.0, 6.0
    for _ in range(16):
        mid_shift = (low_shift + high_shift) / 2.0
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(raw_logits_arr + mid_shift, -30.0, 30.0)))
        if np.mean(p_cal) < target_prevalence:
            low_shift = mid_shift
        else:
            high_shift = mid_shift
    best_shift = (low_shift + high_shift) / 2.0

    # Calibrate with 80% strength towards target prevalence for optimal Brier score
    effective_shift = 0.80 * best_shift
    calibrated_probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits_arr + effective_shift, -30.0, 30.0)))

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob_val = float(calibrated_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob_val)), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Multi-Temperature Multiple-Choice Knapsack Optimization with Strict Feasibility Guarantees."""
    ref_state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]
    if not active_tiers:
        return ref_state.to_plan()

    tier_specs = {t: cohort_spec.interventions[t] for t in active_tiers}
    costs = {t: float(getattr(tier_specs[t], "cost_usd", 0.0)) for t in active_tiers}
    costs[InterventionTier.NONE] = 0.0
    max_slots = {t: int(getattr(tier_specs[t], "max_slots", 0)) for t in active_tiers}
    total_budget = float(cohort_spec.total_budget_usd)

    # Determine eligibility and positive net benefits for each patient
    patient_tier_benefits: Dict[str, Dict[InterventionTier, float]] = {}
    for p in eval_records:
        p_acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)
        p_contra = set(getattr(p, "contraindications", ()) or ())
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5) or 0.5))

        p_benefits = {InterventionTier.NONE: 0.0}
        for t in active_tiers:
            spec = tier_specs[t]
            min_acuity = float(getattr(spec, "min_acuity_score", 0.0) or 0.0)
            contra_flags = set(getattr(spec, "contraindicated_flags", ()) or ())
            if p_acuity >= min_acuity and not (p_contra & contra_flags):
                nb = ref_state.expected_net_benefit(p.patient_id, risk, t)
                if nb > 0.0:
                    p_benefits[t] = float(nb)
        patient_tier_benefits[p.patient_id] = p_benefits

    # Multi-Temperature Knapsack Search: Evaluate candidate solutions across different cost sensitivities
    best_overall_assignments: Optional[Dict[str, InterventionTier]] = None
    best_overall_benefit = -1e9

    for gamma in (0.70, 0.85, 1.00, 1.15):
        assigned: Dict[str, InterventionTier] = {p.patient_id: InterventionTier.NONE for p in eval_records}
        current_slots: Dict[InterventionTier, int] = {t: 0 for t in active_tiers}
        current_spent = 0.0

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

                    eff = delta_b / ((delta_c + 12.0) ** gamma) if delta_c > 0.0 else delta_b * 1e6
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

        total_b = sum(patient_tier_benefits[pid].get(assigned[pid], 0.0) for pid in assigned)
        if total_b > best_overall_benefit:
            best_overall_benefit = total_b
            best_overall_assignments = assigned.copy()

    # Commit optimal assignments to a fresh CareAllocationState prioritized by expected net benefit
    final_state = CareAllocationState(cohort_spec, eval_records)
    if best_overall_assignments is not None:
        sorted_eval = sorted(
            eval_records,
            key=lambda p: patient_tier_benefits[p.patient_id].get(best_overall_assignments.get(p.patient_id, InterventionTier.NONE), 0.0),
            reverse=True,
        )
        for p in sorted_eval:
            tier = best_overall_assignments.get(p.patient_id, InterventionTier.NONE)
            if tier != InterventionTier.NONE:
                if final_state.fits(p.patient_id, tier):
                    final_state.assign(p.patient_id, tier)

    plan = final_state.to_plan()
    # Guarantee constraint: Every patient in eval_records is assigned a valid tier
    for p in eval_records:
        if p.patient_id not in plan.assignments:
            plan.assignments[p.patient_id] = InterventionTier.NONE
    return plan


# EVOLVE-BLOCK-END
# =====================================================================
