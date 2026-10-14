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
    """Stage 1: Multi-modal clinical ensemble with syndrome interactions and log-odds calibration."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    def _logit(p_arr: np.ndarray) -> np.ndarray:
        p_c = np.clip(p_arr, 1e-5, 1.0 - 1e-5)
        return np.log(p_c / (1.0 - p_c))

    def _sigmoid(z_arr: np.ndarray) -> np.ndarray:
        z_c = np.clip(z_arr, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_c))

    if not eval_records:
        return {}

    # Extract non-linear syndrome interaction features from frozen toolkit
    builder = ClinicalFeatureBuilder(include_interactions=True)
    try:
        x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
        x_eval_raw = builder.transform(cohort_spec, eval_records)
    except Exception:
        x_train_raw = np.zeros((len(train_records), 1), dtype=np.float64)
        y_train_raw = np.array([getattr(r, "outcome_label", 0) or 0 for r in train_records], dtype=np.int32)
        x_eval_raw = np.zeros((len(eval_records), 1), dtype=np.float64)

    x_train = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.array(y_train_raw, dtype=np.int32)

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2 or len(train_records) < 4:
        out: Dict[str, float] = {}
        for p in eval_records:
            val = float(getattr(p, "baseline_api_prob", 0.5) or 0.5)
            out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, val)), 6)
        return out

    # Clinical legacy submodel features & acute/chronic discrepancy terms
    def extract_submodel_features(records: Sequence[PatientRecord]) -> np.ndarray:
        feats = []
        for r in records:
            api = float(getattr(r, "baseline_api_prob", 0.5) or 0.5)
            api = max(1e-4, min(1.0 - 1e-4, api))
            sub = getattr(r, "baseline_submodel_scores", {}) or {}
            claims = float(sub.get("claims_linear", api) or api)
            claims = max(1e-4, min(1.0 - 1e-4, claims))
            vitals = float(sub.get("ehr_vitals", api) or api)
            vitals = max(1e-4, min(1.0 - 1e-4, vitals))
            util = float(sub.get("utilization_rules", api) or api)
            util = max(1e-4, min(1.0 - 1e-4, util))
            acuity = float(getattr(r, "acuity_score", 0.5) or 0.5)

            l_api = float(np.log(api / (1.0 - api)))
            l_claims = float(np.log(claims / (1.0 - claims)))
            l_vitals = float(np.log(vitals / (1.0 - vitals)))
            l_util = float(np.log(util / (1.0 - util)))

            # Discrepancy signals: acute vitals vs chronic claims, and utilization discordance
            disc_ac_chr = l_vitals - l_claims
            disc_ac_utl = l_vitals - l_util
            disc_chr_utl = l_claims - l_util
            acuity_api = acuity * l_api
            acuity_vitals = acuity * l_vitals

            feats.append([
                api, l_api,
                claims, l_claims,
                vitals, l_vitals,
                util, l_util,
                acuity,
                disc_ac_chr, disc_ac_utl, disc_chr_utl,
                acuity_api, acuity_vitals,
            ])
        return np.array(feats, dtype=np.float64)

    sub_train = extract_submodel_features(train_records)
    sub_eval = extract_submodel_features(eval_records)

    x_train_aug = np.hstack([x_train, sub_train])
    x_eval_aug = np.hstack([x_eval, sub_eval])

    # 1. Primary Model: RiskBlender (blending LightGBM, XGBoost, Linear, and Prior)
    rb_probs_eval = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.06,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_aug,
            y_train=y_train,
            x_eval=x_eval_aug,
        )
        if rb_dict and len(rb_dict) == len(eval_records):
            rb_probs_eval = np.array([float(rb_dict[p.patient_id]) for p in eval_records], dtype=np.float64)
    except Exception:
        rb_probs_eval = None

    # Fallback to standard features if augmented call failed
    if rb_probs_eval is None:
        try:
            blender = RiskBlender(
                linear_weight=0.40,
                lgbm_weight=0.35,
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
            if rb_dict and len(rb_dict) == len(eval_records):
                rb_probs_eval = np.array([float(rb_dict[p.patient_id]) for p in eval_records], dtype=np.float64)
        except Exception:
            rb_probs_eval = None

    # 2. Complementary Model A: Balanced L2-Regularized Logistic Regression
    lr_probs_eval = None
    try:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(x_train_aug)
        X_ev_s = scaler.transform(x_eval_aug)

        lr = LogisticRegression(
            C=0.20,
            class_weight="balanced",
            max_iter=300,
            random_state=42,
        )
        lr.fit(X_tr_s, y_train)
        lr_probs_eval = lr.predict_proba(X_ev_s)[:, 1]
    except Exception:
        lr_probs_eval = None

    # 3. Complementary Model B: Low-depth Random Forest (safe single-threaded Cython)
    rf_probs_eval = None
    try:
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        rf.fit(x_train_aug, y_train)
        rf_probs_eval = rf.predict_proba(x_eval_aug)[:, 1]
    except Exception:
        rf_probs_eval = None

    api_eval = np.array([float(getattr(p, "baseline_api_prob", 0.5) or 0.5) for p in eval_records], dtype=np.float64)

    # 4. Multi-modal Log-odds Ensembling
    ensemble_logits = np.zeros(len(eval_records), dtype=np.float64)
    total_weight = 0.0

    if rb_probs_eval is not None:
        ensemble_logits += 0.55 * _logit(rb_probs_eval)
        total_weight += 0.55
    if lr_probs_eval is not None:
        ensemble_logits += 0.20 * _logit(lr_probs_eval)
        total_weight += 0.20
    if rf_probs_eval is not None:
        ensemble_logits += 0.15 * _logit(rf_probs_eval)
        total_weight += 0.15

    ensemble_logits += 0.10 * _logit(api_eval)
    total_weight += 0.10

    blended_logits = ensemble_logits / max(1e-4, total_weight)
    p_blended = _sigmoid(blended_logits)

    # 5. Cohort Base-Rate Calibration for Brier Score Minimization
    # Strict monotonic logit shifts preserve AUC-ROC, AUC-PR, and Sens@Spec_0.80 exactly
    y_prev = float(np.mean(y_train))
    p_mean = float(np.mean(p_blended))
    if 0.01 < y_prev < 0.99 and 0.01 < p_mean < 0.99:
        shift = float(_logit(np.array([y_prev]))[0] - _logit(np.array([p_mean]))[0])
        # Moderate shift based on training sample confidence
        sample_shrinkage = min(1.0, len(train_records) / 40.0)
        calibrated_logits = blended_logits + (shift * sample_shrinkage)
        p_final = _sigmoid(calibrated_logits)
    else:
        p_final = p_blended

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        val = float(p_final[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, val)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Adaptive multi-policy knapsack allocation maximizing expected net savings with 0 violations."""
    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    ref_state = CareAllocationState(cohort_spec, eval_records)
    if not candidate_tiers or not eval_records:
        return ref_state.to_plan()

    # Pre-filter eligible patient-tier pairs and compute expected net benefit
    tier_candidates = []
    for p in eval_records:
        r = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5) or 0.5))
        r = max(1e-4, min(1.0 - 1e-4, r))
        acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)
        contra = set(getattr(p, "contraindications", []) or [])

        for tier in candidate_tiers:
            spec = cohort_spec.interventions[tier]
            if acuity < spec.min_acuity_score:
                continue
            if contra & set(spec.contraindicated_flags):
                continue

            net_ben = ref_state.expected_net_benefit(p.patient_id, r, tier)
            if net_ben > 0.0:
                cost = max(1.0, float(spec.cost_usd))
                tier_candidates.append({
                    "pid": p.patient_id,
                    "tier": tier,
                    "net_ben": net_ben,
                    "eff": net_ben / cost,
                })

    if not tier_candidates:
        return ref_state.to_plan()

    # Explore efficiency-benefit tradeoff policies to maximize total net dollar savings
    best_state = None
    best_total_benefit = -1.0

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0, 1.35]
    for alpha in alphas:
        trial_state = CareAllocationState(cohort_spec, eval_records)
        scored = []
        for c in tier_candidates:
            score = c["net_ben"] * (c["eff"] ** alpha)
            scored.append((score, c["pid"], c["tier"], c["net_ben"]))
        scored.sort(key=lambda x: -x[0])

        assigned_pids = set()
        tot_benefit = 0.0
        for _, pid, tier, ben in scored:
            if pid in assigned_pids:
                continue
            if trial_state.fits(pid, tier):
                if trial_state.assign(pid, tier):
                    assigned_pids.add(pid)
                    tot_benefit += ben

        if tot_benefit > best_total_benefit or best_state is None:
            best_total_benefit = tot_benefit
            best_state = trial_state

    plan = best_state.to_plan() if best_state is not None else ref_state.to_plan()

    # Guarantee all eval patients have an assigned tier
    for p in eval_records:
        if p.patient_id not in plan.assignments or plan.assignments[p.patient_id] is None:
            plan.assignments[p.patient_id] = InterventionTier.NONE

    return plan


# EVOLVE-BLOCK-END
# =====================================================================
