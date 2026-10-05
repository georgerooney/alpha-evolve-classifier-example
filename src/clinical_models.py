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

"""Data models for Multi-Model Clinical Risk & Care Allocation Portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class InterventionTier(str, Enum):
    """Care-management and acute-resource intervention tiers."""

    NONE = "NONE"
    AUTOMATED_OUTREACH = "AUTOMATED_OUTREACH"
    PHARMACY_MED_RECON = "PHARMACY_MED_RECON"
    TELEMONITORING = "TELEMONITORING"
    INTENSIVE_RN_CARE_MGMT = "INTENSIVE_RN_CARE_MGMT"
    ACUTE_BED_NAVIGATION = "ACUTE_BED_NAVIGATION"


@dataclass(frozen=True)
class InterventionSpec:
    """Resource cost, clinical efficacy, capacity, and safety rules for an intervention tier."""

    tier: InterventionTier
    cost_usd: float
    relative_risk_reduction: float
    max_slots: int
    min_acuity_score: float = 0.0
    contraindicated_flags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PatientRecord:
    """Clinical, demographic, SDoH, and legacy baseline API score record for a single member/encounter.

    Note: `patient_id` is unique within a scenario. When `eval_records` are sent to the untrusted candidate worker in
    Stage 1/Stage 2, `outcome_label` and `time_to_event_days` are stripped (`None`) so ground-truth labels never exist
    in child process memory.
    """

    patient_id: str
    age: float
    sex: int
    charlson_index: float
    prior_ed_visits_6m: int
    prior_ip_admissions_12m: int
    length_of_stay_days: float
    sdoh_deprivation_index: float
    acuity_score: float
    features: Dict[str, float]
    baseline_api_prob: float
    baseline_submodel_scores: Dict[str, float] = field(default_factory=dict)
    contraindications: Tuple[str, ...] = ()
    outcome_label: Optional[int] = None
    time_to_event_days: Optional[float] = None


@dataclass(frozen=True)
class CohortSpec:
    """Clinical cohort specification, target performance gates, and Stage 2 care-program constraints."""

    problem_id: str
    scenario_id: str
    scenario_name: str
    horizon_days: int
    target_auc: float
    target_specificity: float
    min_sensitivity_floor: float
    event_cost_usd: float
    hrrp_penalty_multiplier: float
    roi_per_1pct_auc_usd: float
    total_budget_usd: float
    interventions: Dict[InterventionTier, InterventionSpec]
    feature_names: Tuple[str, ...]


@dataclass
class CareAllocationPlan:
    """Stage 2 candidate output mapping every evaluation patient_id to an assigned InterventionTier."""

    scenario_id: str
    assignments: Dict[str, InterventionTier] = field(default_factory=dict)


@dataclass(frozen=True)
class ClinicalBenchmarkInstance:
    """Complete train + evaluation scenario for a clinical portfolio problem."""

    problem_id: str
    scenario_id: str
    scenario_name: str
    description: str
    cohort_spec: CohortSpec
    train_records: Tuple[PatientRecord, ...]
    eval_records: Tuple[PatientRecord, ...]


@dataclass(frozen=True)
class TabularTask:
    """Schema handed to `fit_and_score_risk` for real-data tabular classification problems (e.g. Diabetes 130).

    Why a separate type from `CohortSpec`: real datasets have arbitrary columns, informative missingness and
    high-cardinality codes, so the candidate receives raw columns (`Dict[str, np.ndarray]`) described by this schema
    instead of the fixed synthetic `PatientRecord` fields. Numeric columns are float64 with NaN for missing; categorical
    columns are object arrays of `str` with `None` for missing. Rows are shuffled and carry no IDs or labels.
    """

    problem_id: str
    round_id: str
    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]
    n_train: int
    n_eval: int
    label_description: str = ""

