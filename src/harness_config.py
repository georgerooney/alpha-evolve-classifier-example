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

"""Immutable human-configured evaluation weights for the AlphaEvolve clinical portfolio harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationWeights:
    """Weights governing the primary predictive fitness score, Stage 2 ROI tie-breaker, and hard penalties.

    Why predictive power is primary and ROI is a bounded tie-breaker:
    Stage 1 optimizes model discrimination and calibration (ROC-AUC, PR-AUC, Sensitivity@Specificity, Brier).
    Scaling composite predictive power by 10,000.0 means a +1.0% uplift equals +100.0 fitness points. Stage 2
    net dollar savings is reported in full in evaluator insights and contributes a bounded tanh tie-breaker
    in [-5.0, +5.0] so intervention cost variance never drowns out genuine predictive AUC lift.
    """

    roc_auc_weight: float = 0.50
    pr_auc_weight: float = 0.25
    sens_at_spec_weight: float = 0.15
    brier_calibration_weight: float = 0.10
    predictive_scale: float = 10000.0
    roi_tiebreaker_cap: float = 5.0
    roi_target_savings_usd: float = 50000.0
    hard_violation_penalty: float = 1e6


DEFAULT_WEIGHTS = EvaluationWeights()
