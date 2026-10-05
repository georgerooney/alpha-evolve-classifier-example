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

"""Exact clinical predictive metrics (ROC-AUC, PR-AUC, Sensitivity@Specificity, Brier) and Stage 2 ROI evaluator."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionTier,
    PatientRecord,
)
from harness_config import DEFAULT_WEIGHTS, EvaluationWeights


def compute_roc_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Exact Mann-Whitney U ROC-AUC with average-rank tie handling.

    Vectorized with `scipy.stats.rankdata` (identical semantics to the original pure-Python rank loop) because the
    real-data track scores ~20k-row eval sets and runs 1000-resample paired bootstraps on them.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    n = y.shape[0]
    if n == 0 or p.shape[0] != n:
        return 0.5
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = float(rankdata(p)[pos].sum())
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(max(0.0, min(1.0, u_stat / (n_pos * n_neg))))


def paired_bootstrap_auc_diff(
    y_true: Sequence[int],
    p_a: Sequence[float],
    p_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap of `AUC(p_a) - AUC(p_b)` on the same rows.

    Why paired: both models are scored on identical patients, so resampling rows jointly cancels the shared
    sampling noise and gives a much tighter (and honest) CI than comparing two independent AUC intervals.
    Returns the point delta, a 95% percentile CI, and the one-sided bootstrap p-value for `delta <= 0`.
    """
    y = np.asarray(y_true)
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    delta = compute_roc_auc(y, a) - compute_roc_auc(y, b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, y.shape[0], y.shape[0])
        deltas[i] = compute_roc_auc(y[idx], a[idx]) - compute_roc_auc(y[idx], b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value_one_sided": float(np.mean(deltas <= 0.0)) if delta != 0.0 else 1.0,
    }


def compute_pr_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Average Precision (area under Precision-Recall curve) sorted by descending probability."""
    n = len(y_true)
    if n == 0 or len(y_prob) != n:
        return 0.0
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return 0.0

    # Group by distinct score thresholds to stay invariant to tie order
    paired = sorted(zip(y_prob, y_true), key=lambda x: -x[0])
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < n:
        j = i
        pos_in_group = 0
        neg_in_group = 0
        while j < n and paired[j][0] == paired[i][0]:
            if paired[j][1] == 1:
                pos_in_group += 1
            else:
                neg_in_group += 1
            j += 1
        tp += pos_in_group
        fp += neg_in_group
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(max(0.0, min(1.0, ap)))


def compute_sensitivity_at_specificity(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_specificity: float = 0.80,
) -> float:
    """Maximum sensitivity achieved at any threshold whose specificity >= target_specificity."""
    n = len(y_true)
    if n == 0 or len(y_prob) != n:
        return 0.0
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    max_fp = math.floor((1.0 - target_specificity + 1e-9) * n_neg)
    paired = sorted(zip(y_prob, y_true), key=lambda x: -x[0])
    tp = 0
    fp = 0
    best_sens = 0.0
    i = 0
    while i < n:
        j = i
        pos_g = 0
        neg_g = 0
        while j < n and paired[j][0] == paired[i][0]:
            if paired[j][1] == 1:
                pos_g += 1
            else:
                neg_g += 1
            j += 1
        if fp + neg_g <= max_fp:
            tp += pos_g
            fp += neg_g
            best_sens = max(best_sens, tp / n_pos)
        else:
            break
        i = j
    return float(best_sens)


def compute_brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Mean squared probability calibration error (lower is better)."""
    n = len(y_true)
    if n == 0 or len(y_prob) != n:
        return 1.0
    total = 0.0
    for y, p in zip(y_true, y_prob):
        p_clamped = max(0.0, min(1.0, float(p) if math.isfinite(p) else 0.5))
        total += (float(y) - p_clamped) ** 2
    return float(total / n)


def _composite_predictive_unit(
    roc_auc: float,
    pr_auc: float,
    sens_at_spec: float,
    brier: float,
    weights: EvaluationWeights,
) -> float:
    return (
        weights.roc_auc_weight * roc_auc
        + weights.pr_auc_weight * pr_auc
        + weights.sens_at_spec_weight * sens_at_spec
        + weights.brier_calibration_weight * max(0.0, 1.0 - brier)
    )


def evaluate_cohort_predictions_and_allocation(
    cohort_spec: CohortSpec,
    eval_records_with_gt: Sequence[PatientRecord],
    predicted_risks: Mapping[str, float],
    allocation_plan: CareAllocationPlan,
    weights: EvaluationWeights = DEFAULT_WEIGHTS,
) -> Dict[str, Any]:
    """Evaluates Stage 1 risk predictions and Stage 2 care-management allocation against ground truth."""
    violations: list[str] = []
    patient_by_id: Dict[str, PatientRecord] = {p.patient_id: p for p in eval_records_with_gt}

    # --- 1. Validate Stage 1 risk dictionary ---
    y_true: list[int] = []
    y_prob: list[float] = []
    y_base: list[float] = []

    if not isinstance(predicted_risks, Mapping):
        violations.append("Stage 1 output must be a dict mapping patient_id -> float probability")
        predicted_risks = {}

    missing_risk_ids = 0
    invalid_prob_ids = 0
    for p in eval_records_with_gt:
        gt = int(p.outcome_label or 0)
        y_true.append(gt)
        y_base.append(float(p.baseline_api_prob))

        raw_p = predicted_risks.get(p.patient_id)
        if raw_p is None:
            missing_risk_ids += 1
            y_prob.append(0.5)
        else:
            try:
                val = float(raw_p)
            except (TypeError, ValueError):
                val = float("nan")
            if not math.isfinite(val) or val < 0.0 or val > 1.0:
                invalid_prob_ids += 1
                y_prob.append(max(0.0, min(1.0, val if math.isfinite(val) else 0.5)))
            else:
                y_prob.append(val)

    if missing_risk_ids > 0:
        violations.append(f"Stage 1 missing risk predictions for {missing_risk_ids} patients")
    if invalid_prob_ids > 0:
        violations.append(f"Stage 1 produced {invalid_prob_ids} probabilities outside [0.0, 1.0] or non-finite")

    # --- 2. Compute Stage 1 Predictive Metrics ---
    roc_auc = compute_roc_auc(y_true, y_prob)
    pr_auc = compute_pr_auc(y_true, y_prob)
    sens_at_spec = compute_sensitivity_at_specificity(
        y_true, y_prob, cohort_spec.target_specificity
    )
    brier = compute_brier_score(y_true, y_prob)

    base_roc_auc = compute_roc_auc(y_true, y_base)
    base_pr_auc = compute_pr_auc(y_true, y_base)
    base_sens_at_spec = compute_sensitivity_at_specificity(
        y_true, y_base, cohort_spec.target_specificity
    )
    base_brier = compute_brier_score(y_true, y_base)

    auc_uplift = roc_auc - base_roc_auc
    pr_auc_uplift = pr_auc - base_pr_auc

    # --- 3. Validate & Score Stage 2 Care Allocation Plan ---
    assignments = (
        allocation_plan.assignments
        if isinstance(allocation_plan, CareAllocationPlan)
        and isinstance(allocation_plan.assignments, Mapping)
        else {}
    )
    if not isinstance(allocation_plan, CareAllocationPlan):
        violations.append("Stage 2 output must be a CareAllocationPlan instance")

    slot_counts: Dict[InterventionTier, int] = {t: 0 for t in InterventionTier}
    total_cost_usd = 0.0
    gross_savings_usd = 0.0
    prevented_events = 0.0
    missing_alloc = 0
    unknown_tier = 0
    acuity_violations = 0
    contraindication_violations = 0

    unit_event_value = cohort_spec.event_cost_usd * cohort_spec.hrrp_penalty_multiplier

    for p in eval_records_with_gt:
        raw_tier = assignments.get(p.patient_id)
        if raw_tier is None:
            missing_alloc += 1
            tier = InterventionTier.NONE
        else:
            try:
                tier = (
                    raw_tier
                    if isinstance(raw_tier, InterventionTier)
                    else InterventionTier(str(raw_tier))
                )
            except ValueError:
                unknown_tier += 1
                tier = InterventionTier.NONE

        spec = cohort_spec.interventions.get(tier)
        if spec is None:
            unknown_tier += 1
            spec = cohort_spec.interventions[InterventionTier.NONE]
            tier = InterventionTier.NONE

        slot_counts[tier] = slot_counts.get(tier, 0) + 1
        total_cost_usd += spec.cost_usd

        if tier != InterventionTier.NONE:
            if p.acuity_score + 1e-9 < spec.min_acuity_score:
                acuity_violations += 1
            if set(p.contraindications) & set(spec.contraindicated_flags):
                contraindication_violations += 1

            # Realized clinical benefit on true outcome events (with time-to-event urgency weighting)
            if p.outcome_label == 1:
                urgency_factor = 1.0
                if p.time_to_event_days is not None and cohort_spec.horizon_days > 0:
                    urgency_factor = 1.0 + 0.15 * max(
                        0.0,
                        1.0 - float(p.time_to_event_days) / float(cohort_spec.horizon_days),
                    )
                eff_rrr = spec.relative_risk_reduction * urgency_factor
                prevented_events += eff_rrr
                gross_savings_usd += eff_rrr * unit_event_value

    if missing_alloc > 0:
        violations.append(f"Stage 2 missing intervention assignments for {missing_alloc} patients")
    if unknown_tier > 0:
        violations.append(f"Stage 2 used {unknown_tier} unsupported intervention tiers")
    if acuity_violations > 0:
        violations.append(f"Stage 2 assigned {acuity_violations} low-acuity patients above their eligibility tier")
    if contraindication_violations > 0:
        violations.append(f"Stage 2 assigned {contraindication_violations} contraindicated interventions")

    for tier, count in slot_counts.items():
        if tier == InterventionTier.NONE:
            continue
        spec = cohort_spec.interventions.get(tier)
        if spec is not None and count > spec.max_slots:
            violations.append(
                f"Stage 2 exceeded slot capacity for {tier.value}: {count} > {spec.max_slots}"
            )

    if total_cost_usd > cohort_spec.total_budget_usd + 1e-6:
        violations.append(
            f"Stage 2 exceeded intervention budget: ${total_cost_usd:,.2f} > ${cohort_spec.total_budget_usd:,.2f}"
        )

    net_savings_usd = gross_savings_usd - total_cost_usd
    actuarial_auc_roi_usd = max(0.0, auc_uplift * 100.0) * cohort_spec.roi_per_1pct_auc_usd

    # --- 4. Combine Primary Predictive Fitness + Stage 2 ROI Tie-Breaker ---
    pred_unit = _composite_predictive_unit(roc_auc, pr_auc, sens_at_spec, brier, weights)
    base_pred_unit = _composite_predictive_unit(
        base_roc_auc, base_pr_auc, base_sens_at_spec, base_brier, weights
    )
    predictive_score = weights.predictive_scale * pred_unit
    baseline_predictive_score = weights.predictive_scale * base_pred_unit

    roi_tiebreaker = weights.roi_tiebreaker_cap * math.tanh(
        net_savings_usd / max(1.0, weights.roi_target_savings_usd)
    )
    hard_violation_count = len(violations)
    total_score = (
        predictive_score
        + roi_tiebreaker
        - hard_violation_count * weights.hard_violation_penalty
    )

    return {
        "scenario_id": cohort_spec.scenario_id,
        "scenario_name": cohort_spec.scenario_name,
        "is_feasible": hard_violation_count == 0,
        "hard_violation_count": hard_violation_count,
        "violations": violations,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sens_at_spec": sens_at_spec,
        "brier_score": brier,
        "baseline_roc_auc": base_roc_auc,
        "baseline_pr_auc": base_pr_auc,
        "baseline_sens_at_spec": base_sens_at_spec,
        "baseline_brier_score": base_brier,
        "auc_uplift": auc_uplift,
        "pr_auc_uplift": pr_auc_uplift,
        "target_auc": cohort_spec.target_auc,
        "meets_target_auc": roc_auc >= cohort_spec.target_auc,
        "gross_savings_usd": gross_savings_usd,
        "intervention_cost_usd": total_cost_usd,
        "net_savings_usd": net_savings_usd,
        "prevented_events": prevented_events,
        "actuarial_auc_roi_usd": actuarial_auc_roi_usd,
        "predictive_score": predictive_score,
        "baseline_predictive_score": baseline_predictive_score,
        "roi_tiebreaker": roi_tiebreaker,
        "total_score": total_score,
    }
