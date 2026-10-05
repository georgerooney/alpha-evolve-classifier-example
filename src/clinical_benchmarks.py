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

"""Physiologically realistic clinical cohort benchmark generators, seed perturbation, and held-out suites."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clinical_models import (
    ClinicalBenchmarkInstance,
    CohortSpec,
    InterventionSpec,
    InterventionTier,
    PatientRecord,
)
from data_adapters import SimulatedVertexModelAdapter, load_records_from_csv

HERE = pathlib.Path(__file__).resolve().parent
PROBLEMS_ROOT = HERE.parent / "problems"

DEFAULT_PERTURB_SEED = 2027
DEFAULT_HELDOUT_SEED = 2026

COMMON_CLINICAL_FEATURES: Tuple[str, ...] = (
    "egfr",
    "hba1c",
    "hemoglobin",
    "sodium",
    "nt_probnp",
    "lvef",
    "loop_diuretic_dose_mg",
    "weight_gain_7d_kg",
    "lactate_peak",
    "lactate_clearance_6h",
    "shock_index",
    "sepsis_organ_dysfunction_count",
    "vital_instability_count",
    "ed_boarding_hours",
    "med_adherence_pdc",
    "discharge_to_snf",
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


DEFAULT_PROBLEM_CATALOG: Dict[str, Dict[str, Any]] = {
    "readmission_30d": {
        "problem_id": "readmission_30d",
        "title": "30-Day All-Cause Hospital Readmission Reduction (CMS HRRP)",
        "horizon_days": 30,
        "target_auc": 0.70,
        "target_specificity": 0.80,
        "min_sensitivity_floor": 0.55,
        "event_cost_usd": 16800.0,
        "hrrp_penalty_multiplier": 1.35,
        "roi_per_1pct_auc_usd": 420000.0,
        "base_seed": 11000,
        "scenarios": [
            {
                "scenario_id": "READM-01",
                "scenario_name": "Medicare HRRP General Med/Surg Discharges",
                "description": "High-volume general medicine and surgical discharges subject to CMS HRRP penalties.",
                "n_train": 360,
                "n_eval": 180,
                "budget_usd": 28000.0,
                "shift": "standard",
            },
            {
                "scenario_id": "READM-02",
                "scenario_name": "High-SDoH Deprivation & Medication Non-Adherence Cohort",
                "description": "Safety-net and high social vulnerability discharges with post-discharge pharmacy gaps.",
                "n_train": 340,
                "n_eval": 170,
                "budget_usd": 26000.0,
                "shift": "high_sdoh",
            },
            {
                "scenario_id": "READM-03",
                "scenario_name": "Multi-Morbid Diabetic & Renal Complex Cohort",
                "description": "Patients with concurrent CKD and glycemic instability where linear claims models miscalibrate.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 25000.0,
                "shift": "renal_metabolic",
            },
            {
                "scenario_id": "READM-04",
                "scenario_name": "Post-Acute SNF & Home Health Transition Cohort",
                "description": "Complex transitions to skilled nursing and home health under tight RN care-management capacity.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 22000.0,
                "shift": "snf_transition",
            },
        ],
    },
    "inpatient_admission": {
        "problem_id": "inpatient_admission",
        "title": "Inpatient Admissions & Acute Bed Capacity Optimization",
        "horizon_days": 14,
        "target_auc": 0.78,
        "target_specificity": 0.80,
        "min_sensitivity_floor": 0.60,
        "event_cost_usd": 14200.0,
        "hrrp_penalty_multiplier": 1.0,
        "roi_per_1pct_auc_usd": 510000.0,
        "base_seed": 22000,
        "scenarios": [
            {
                "scenario_id": "INPAT-01",
                "scenario_name": "Metro Acute Care ED-to-Inpatient Triage",
                "description": "Predicting unplanned inpatient admissions from ED and urgent ambulatory encounters.",
                "n_train": 400,
                "n_eval": 200,
                "budget_usd": 30000.0,
                "shift": "standard",
            },
            {
                "scenario_id": "INPAT-02",
                "scenario_name": "Winter Respiratory & ED Boarding Surge",
                "description": "High ED boarding hours and vital-sign instability during seasonal capacity crunch.",
                "n_train": 380,
                "n_eval": 190,
                "budget_usd": 24000.0,
                "shift": "ed_surge",
            },
            {
                "scenario_id": "INPAT-03",
                "scenario_name": "Chronic High-Utilizer Polychronic Population",
                "description": "Frequent ED utilizers with compounding comorbidity velocity and electrolyte derangements.",
                "n_train": 360,
                "n_eval": 180,
                "budget_usd": 27000.0,
                "shift": "high_utilizer",
            },
            {
                "scenario_id": "INPAT-04",
                "scenario_name": "Regional Bed-Capacity Constrained Network",
                "description": "Severe acute bed navigation slot constraints requiring high specificity.",
                "n_train": 360,
                "n_eval": 180,
                "budget_usd": 21000.0,
                "shift": "capacity_crunch",
            },
        ],
    },
    "sepsis_90d": {
        "problem_id": "sepsis_90d",
        "title": "Sepsis Early Detection & 90-Day Survivor Trajectory",
        "horizon_days": 90,
        "target_auc": 0.80,
        "target_specificity": 0.80,
        "min_sensitivity_floor": 0.65,
        "event_cost_usd": 28500.0,
        "hrrp_penalty_multiplier": 1.15,
        "roi_per_1pct_auc_usd": 680000.0,
        "base_seed": 33000,
        "scenarios": [
            {
                "scenario_id": "SEPSIS-01",
                "scenario_name": "Post-ICU Sepsis Survivor 90-Day Trajectory",
                "description": "Sepsis survivors discharged after ICU stay with residual organ dysfunction and shock index risk.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 32000.0,
                "shift": "sepsis_core",
            },
            {
                "scenario_id": "SEPSIS-02",
                "scenario_name": "Incomplete Lactate Clearance & Hemodynamic Fragility",
                "description": "Subtle shock-index x impaired 6-hour lactate clearance interactions missed by linear scores.",
                "n_train": 300,
                "n_eval": 150,
                "budget_usd": 30000.0,
                "shift": "lactate_shock",
            },
            {
                "scenario_id": "SEPSIS-03",
                "scenario_name": "Immunocompromised & Renal-Impaired Sepsis Cohort",
                "description": "Multi-organ renal + hematologic vulnerability post-sepsis hospitalization.",
                "n_train": 300,
                "n_eval": 150,
                "budget_usd": 29000.0,
                "shift": "renal_metabolic",
            },
            {
                "scenario_id": "SEPSIS-04",
                "scenario_name": "Post-Sepsis SNF & Rural Tele-Followup Cohort",
                "description": "Discharges to SNF and rural zip codes with high SDoH barriers to early sepsis follow-up.",
                "n_train": 300,
                "n_eval": 150,
                "budget_usd": 26000.0,
                "shift": "snf_transition",
            },
        ],
    },
    "chf_30d": {
        "problem_id": "chf_30d",
        "title": "Congestive Heart Failure (CHF) 30-Day Readmission Propensity & Timing",
        "horizon_days": 30,
        "target_auc": 0.76,
        "target_specificity": 0.80,
        "min_sensitivity_floor": 0.60,
        "event_cost_usd": 19400.0,
        "hrrp_penalty_multiplier": 1.30,
        "roi_per_1pct_auc_usd": 490000.0,
        "base_seed": 44000,
        "scenarios": [
            {
                "scenario_id": "CHF-01",
                "scenario_name": "HFrEF Acute Decompensated Heart Failure Discharges",
                "description": "Reduced ejection fraction (~20% readmission rate) with fluid weight and NT-proBNP elevation.",
                "n_train": 340,
                "n_eval": 170,
                "budget_usd": 28000.0,
                "shift": "chf_hfref",
            },
            {
                "scenario_id": "CHF-02",
                "scenario_name": "Cardiorenal Syndrome & Loop Diuretic Resistance",
                "description": "High loop diuretic dose + 7-day weight gain + eGFR drop driving rapid 14-day readmission.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 26000.0,
                "shift": "cardiorenal",
            },
            {
                "scenario_id": "CHF-03",
                "scenario_name": "HFpEF Elderly Multi-Morbid Cohort",
                "description": "Preserved ejection fraction with atrial/renal comorbidities and medication adherence gaps.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 25000.0,
                "shift": "high_sdoh",
            },
            {
                "scenario_id": "CHF-04",
                "scenario_name": "High-Volume Community Hospital HF Network",
                "description": "Broad community CHF discharges under telemonitoring and RN home-visit slot caps.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 23000.0,
                "shift": "capacity_crunch",
            },
        ],
    },
}


def load_problem_config(problem_id: str) -> Dict[str, Any]:
    """Loads problem configuration from `problems/<problem_id>/problem_config.json` or fallback catalog."""
    clean_id = pathlib.Path(problem_id).name
    cfg_path = PROBLEMS_ROOT / clean_id / "problem_config.json"
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Tabular (real-data) problems carry `task_type` + column schema instead of synthetic `scenarios`.
        if "problem_id" in data and ("scenarios" in data or "task_type" in data):
            return data
    if clean_id in DEFAULT_PROBLEM_CATALOG:
        return DEFAULT_PROBLEM_CATALOG[clean_id]
    raise ValueError(f"Unknown problem_id '{problem_id}' and no config found at {cfg_path}")


def _build_interventions(n_eval: int, shift: str) -> Dict[InterventionTier, InterventionSpec]:
    tight = 0.70 if shift == "capacity_crunch" else 1.0
    return {
        InterventionTier.NONE: InterventionSpec(
            tier=InterventionTier.NONE,
            cost_usd=0.0,
            relative_risk_reduction=0.0,
            max_slots=n_eval,
            min_acuity_score=0.0,
        ),
        InterventionTier.AUTOMATED_OUTREACH: InterventionSpec(
            tier=InterventionTier.AUTOMATED_OUTREACH,
            cost_usd=65.0,
            relative_risk_reduction=0.07,
            max_slots=max(10, int(0.45 * n_eval * tight)),
            min_acuity_score=1.0,
        ),
        InterventionTier.PHARMACY_MED_RECON: InterventionSpec(
            tier=InterventionTier.PHARMACY_MED_RECON,
            cost_usd=220.0,
            relative_risk_reduction=0.17,
            max_slots=max(6, int(0.22 * n_eval * tight)),
            min_acuity_score=2.0,
        ),
        InterventionTier.TELEMONITORING: InterventionSpec(
            tier=InterventionTier.TELEMONITORING,
            cost_usd=480.0,
            relative_risk_reduction=0.26,
            max_slots=max(5, int(0.16 * n_eval * tight)),
            min_acuity_score=2.6,
            contraindicated_flags=("NO_HOME_CONNECTIVITY",),
        ),
        InterventionTier.INTENSIVE_RN_CARE_MGMT: InterventionSpec(
            tier=InterventionTier.INTENSIVE_RN_CARE_MGMT,
            cost_usd=950.0,
            relative_risk_reduction=0.38,
            max_slots=max(4, int(0.10 * n_eval * tight)),
            min_acuity_score=3.4,
            contraindicated_flags=("PALLIATIVE_HOSPICE",),
        ),
        InterventionTier.ACUTE_BED_NAVIGATION: InterventionSpec(
            tier=InterventionTier.ACUTE_BED_NAVIGATION,
            cost_usd=1350.0,
            relative_risk_reduction=0.46,
            max_slots=max(3, int(0.06 * n_eval * tight)),
            min_acuity_score=4.2,
            contraindicated_flags=("PALLIATIVE_HOSPICE", "DECLINED_ACUTE_PROTOCOL"),
        ),
    }


def _latent_nonlinear_log_odds(
    problem_id: str,
    *,
    age: float,
    charlson: float,
    ed_6m: int,
    ip_12m: int,
    los: float,
    sdoh: float,
    acuity: float,
    feats: Mapping[str, float],
) -> float:
    """True underlying physiological risk log-odds combining main effects and strong clinical interactions."""
    egfr = feats["egfr"]
    hba1c = feats["hba1c"]
    hb = feats["hemoglobin"]
    na = feats["sodium"]
    bnp = feats["nt_probnp"]
    lvef = feats["lvef"]
    diuretic = feats["loop_diuretic_dose_mg"]
    wt_gain = feats["weight_gain_7d_kg"]
    lactate = feats["lactate_peak"]
    lac_clear = feats["lactate_clearance_6h"]
    shock = feats["shock_index"]
    organ_dys = feats["sepsis_organ_dysfunction_count"]
    vital_instab = feats["vital_instability_count"]
    boarding = feats["ed_boarding_hours"]
    adherence = feats["med_adherence_pdc"]
    snf = feats["discharge_to_snf"]

    # Non-linear clinical syndrome terms (the signal AlphaEvolve can unlock via feature engineering + GBDTs)
    cardiorenal_syndrome = (
        2.10 * max(0.0, (65.0 - egfr) / 28.0) * math.log1p(max(0.0, bnp - 450.0) / 800.0)
    )
    diuretic_resistance = (
        1.75 * max(0.0, wt_gain - 0.6) * max(0.0, (diuretic - 25.0) / 50.0) * (1.3 if lvef < 42.0 else 0.85)
    )
    sepsis_hemodynamic_failure = (
        2.35 * max(0.0, shock - 0.76) * max(0.0, 0.42 - lac_clear) * 3.2
        + 0.90 * max(0.0, lactate - 1.8) * (1.0 + 0.45 * organ_dys)
    )
    sdoh_adherence_spiral = (
        2.65 * max(0.0, sdoh - 0.32) * max(0.0, 0.82 - adherence) * (1.0 + 0.32 * charlson)
    )
    ed_boarding_velocity = (
        2.20 * math.log1p(ed_6m + ip_12m) * max(0.0, (boarding - 2.2) / 4.5) * (1.0 + 0.35 * vital_instab)
    )
    electrolyte_anemia_fragility = (
        1.45 * (1.0 if abs(na - 139.0) > 3.5 else 0.0) * max(0.0, (13.0 - hb) / 2.2)
        + 1.15 * max(0.0, hba1c - 6.8) * max(0.0, (70.0 - egfr) / 22.0)
    )
    snf_deconditioning = 1.65 * snf * max(0.0, (los - 2.5) / 4.0) * (1.0 if acuity > 3.5 else 0.5)

    # Subpopulation correction where legacy linear claims score over-scores low-acuity elderly elective stays
    elderly_elective_correction = -1.55 if (age > 70.0 and acuity < 3.2 and vital_instab == 0) else 0.0

    # Weight syndrome emphasis by clinical problem domain while keeping shared pathophysiology
    w_chf = 1.50 if "chf" in problem_id else 0.95
    w_sep = 1.55 if "sepsis" in problem_id else 0.90
    w_inp = 1.65 if "inpatient" in problem_id else 1.05
    w_rdm = 1.60 if "readmission" in problem_id else 1.05

    log_odds = (
        -2.75
        + 0.14 * acuity
        + 0.10 * charlson
        + w_chf * (cardiorenal_syndrome + diuretic_resistance)
        + w_sep * sepsis_hemodynamic_failure
        + w_rdm * (sdoh_adherence_spiral + snf_deconditioning)
        + w_inp * (ed_boarding_velocity + electrolyte_anemia_fragility)
        + elderly_elective_correction
    )
    return log_odds


def _generate_cohort_records(
    rng: random.Random,
    problem_id: str,
    scenario_id: str,
    count: int,
    shift: str,
    horizon_days: int,
    prefix: str,
) -> Tuple[PatientRecord, ...]:
    adapter = SimulatedVertexModelAdapter(problem_id=problem_id)
    records: List[PatientRecord] = []

    for i in range(count):
        age = max(22.0, min(94.0, rng.gauss(66.0 if shift != "ed_surge" else 59.0, 14.0)))
        sex = 1 if rng.random() < 0.48 else 0
        charlson = max(0.0, min(12.0, round(rng.gammavariate(2.2, 1.4), 1)))
        ed_6m = min(12, int(rng.expovariate(0.75 if shift != "high_utilizer" else 0.42)))
        ip_12m = min(8, int(rng.expovariate(1.15 if shift != "high_utilizer" else 0.65)))
        los = max(1.0, min(28.0, round(rng.lognormvariate(1.25, 0.55), 1)))
        sdoh = max(
            0.02,
            min(
                0.98,
                round(
                    rng.betavariate(
                        2.8 if shift == "high_sdoh" else 1.8,
                        1.6 if shift == "high_sdoh" else 2.6,
                    ),
                    3,
                ),
            ),
        )

        egfr = max(
            10.0,
            min(
                115.0,
                round(
                    rng.gauss(52.0 if shift in ("renal_metabolic", "cardiorenal") else 68.0, 22.0),
                    1,
                ),
            ),
        )
        hba1c = max(4.8, min(13.5, round(rng.lognormvariate(1.88, 0.22), 2)))
        hemoglobin = max(7.2, min(16.8, round(rng.gauss(12.1, 1.9), 2)))
        sodium = max(124.0, min(152.0, round(rng.gauss(138.5, 3.8), 1)))

        is_hf_context = "chf" in problem_id or shift in ("chf_hfref", "cardiorenal")
        nt_probnp = max(
            50.0,
            min(
                28000.0,
                round(rng.lognormvariate(7.3 if is_hf_context else 6.1, 1.05), 1),
            ),
        )
        lvef = max(
            15.0,
            min(
                70.0,
                round(rng.gauss(36.0 if shift == "chf_hfref" else 51.0, 12.5), 1),
            ),
        )
        loop_diuretic = max(
            0.0,
            min(240.0, round(rng.expovariate(1.0 / (65.0 if is_hf_context else 28.0)), 1)),
        )
        weight_gain = max(-2.0, min(8.5, round(rng.gauss(1.1 if is_hf_context else 0.3, 1.4), 2)))

        is_sepsis_context = "sepsis" in problem_id or shift in ("sepsis_core", "lactate_shock")
        lactate_peak = max(
            0.6,
            min(11.0, round(rng.lognormvariate(0.82 if is_sepsis_context else 0.38, 0.52), 2)),
        )
        lactate_clearance = max(
            -0.20,
            min(0.85, round(rng.gauss(0.26 if is_sepsis_context else 0.42, 0.21), 3)),
        )
        shock_index = max(
            0.42,
            min(1.65, round(rng.gauss(0.84 if is_sepsis_context else 0.71, 0.20), 3)),
        )
        organ_dys = float(
            min(5, int(rng.expovariate(0.75 if is_sepsis_context else 1.45)))
        )

        vital_instab = float(min(6, int(rng.expovariate(0.85))))
        ed_boarding = max(
            0.5,
            min(30.0, round(rng.lognormvariate(1.45 if shift == "ed_surge" else 1.05, 0.60), 2)),
        )
        med_adherence = max(0.10, min(1.0, round(rng.betavariate(3.2, 1.7), 3)))
        discharge_snf = 1.0 if rng.random() < (0.38 if shift == "snf_transition" else 0.18) else 0.0

        acuity = max(
            0.5,
            min(
                9.8,
                round(
                    1.2
                    + 0.32 * charlson
                    + 0.45 * vital_instab
                    + 0.35 * organ_dys
                    + (1.1 if egfr < 45.0 else 0.0)
                    + rng.gauss(0.0, 0.7),
                    2,
                ),
            ),
        )

        feats: Dict[str, float] = {
            "egfr": egfr,
            "hba1c": hba1c,
            "hemoglobin": hemoglobin,
            "sodium": sodium,
            "nt_probnp": nt_probnp,
            "lvef": lvef,
            "loop_diuretic_dose_mg": loop_diuretic,
            "weight_gain_7d_kg": weight_gain,
            "lactate_peak": lactate_peak,
            "lactate_clearance_6h": lactate_clearance,
            "shock_index": shock_index,
            "sepsis_organ_dysfunction_count": organ_dys,
            "vital_instability_count": vital_instab,
            "ed_boarding_hours": ed_boarding,
            "med_adherence_pdc": med_adherence,
            "discharge_to_snf": discharge_snf,
        }

        base_prob, submodels = adapter.score_patient(
            age=age,
            charlson_index=charlson,
            prior_ed_visits_6m=ed_6m,
            prior_ip_admissions_12m=ip_12m,
            length_of_stay_days=los,
            sdoh_deprivation_index=sdoh,
            acuity_score=acuity,
            features=feats,
        )

        # True outcome generated from latent nonlinear pathophysiology with realistic low noise
        log_odds = _latent_nonlinear_log_odds(
            problem_id,
            age=age,
            charlson=charlson,
            ed_6m=ed_6m,
            ip_12m=ip_12m,
            los=los,
            sdoh=sdoh,
            acuity=acuity,
            feats=feats,
        )
        true_prob = _sigmoid(log_odds + rng.gauss(0.0, 0.38))
        outcome = 1 if rng.random() < true_prob else 0
        tte: Optional[float] = None
        if outcome == 1:
            tte = round(max(1.0, min(float(horizon_days), rng.weibullvariate(horizon_days * 0.45, 1.3))), 1)

        contraindications: List[str] = []
        if rng.random() < 0.05:
            contraindications.append("PALLIATIVE_HOSPICE")
        if sdoh > 0.82 and rng.random() < 0.25:
            contraindications.append("NO_HOME_CONNECTIVITY")
        if rng.random() < 0.03:
            contraindications.append("DECLINED_ACUTE_PROTOCOL")

        records.append(
            PatientRecord(
                patient_id=f"{scenario_id}-{prefix}-{i:04d}",
                age=age,
                sex=sex,
                charlson_index=charlson,
                prior_ed_visits_6m=ed_6m,
                prior_ip_admissions_12m=ip_12m,
                length_of_stay_days=los,
                sdoh_deprivation_index=sdoh,
                acuity_score=acuity,
                features=feats,
                baseline_api_prob=base_prob,
                baseline_submodel_scores=submodels,
                contraindications=tuple(contraindications),
                outcome_label=outcome,
                time_to_event_days=tte,
            )
        )
    return tuple(records)


def perturb_instances(
    instances: Sequence[ClinicalBenchmarkInstance],
    seed: int,
) -> List[ClinicalBenchmarkInstance]:
    """Anonymizes patient IDs, shuffles patient order, and applies small physiological jitter (+/-1.5%)."""
    rng = random.Random(seed)
    out: List[ClinicalBenchmarkInstance] = []

    for inst in instances:
        def _jitter_records(recs: Sequence[PatientRecord], tag: str) -> Tuple[PatientRecord, ...]:
            idxs = list(range(len(recs)))
            rng.shuffle(idxs)
            jittered: List[PatientRecord] = []
            for new_i, old_i in enumerate(idxs):
                p = recs[old_i]
                digest = hashlib.sha256(f"{seed}:{inst.scenario_id}:{tag}:{p.patient_id}".encode()).hexdigest()[:10]
                opaque_id = f"MBR-{tag}-{new_i:04d}-{digest}"
                new_feats = {
                    k: (
                        v
                        if k in ("discharge_to_snf", "sepsis_organ_dysfunction_count", "vital_instability_count")
                        else round(v * (1.0 + rng.uniform(-0.015, 0.015)), 4)
                    )
                    for k, v in p.features.items()
                }
                jittered.append(
                    dataclasses.replace(
                        p,
                        patient_id=opaque_id,
                        features=new_feats,
                    )
                )
            return tuple(jittered)

        out.append(
            dataclasses.replace(
                inst,
                train_records=_jitter_records(inst.train_records, "TR"),
                eval_records=_jitter_records(inst.eval_records, "EV"),
            )
        )
    return out


def build_benchmark_instances(
    problem_id: str = "readmission_30d",
    *,
    perturb: bool = False,
    perturb_seed: Optional[int] = None,
) -> List[ClinicalBenchmarkInstance]:
    """Builds the public benchmark scenario suite for `problem_id`."""
    cfg = load_problem_config(problem_id)
    base_seed = int(cfg.get("base_seed", 11000))
    horizon_days = int(cfg.get("horizon_days", 30))
    feature_names = tuple(cfg.get("feature_names", COMMON_CLINICAL_FEATURES))

    instances: List[ClinicalBenchmarkInstance] = []
    for idx, sc in enumerate(cfg["scenarios"]):
        sc_rng = random.Random(base_seed + idx * 101)
        scenario_id = str(sc["scenario_id"])
        n_train = int(sc.get("n_train", 320))
        n_eval = int(sc.get("n_eval", 160))
        shift = str(sc.get("shift", "standard"))

        spec = CohortSpec(
            problem_id=str(cfg["problem_id"]),
            scenario_id=scenario_id,
            scenario_name=str(sc["scenario_name"]),
            horizon_days=horizon_days,
            target_auc=float(cfg.get("target_auc", 0.75)),
            target_specificity=float(cfg.get("target_specificity", 0.80)),
            min_sensitivity_floor=float(cfg.get("min_sensitivity_floor", 0.55)),
            event_cost_usd=float(cfg.get("event_cost_usd", 16000.0)),
            hrrp_penalty_multiplier=float(cfg.get("hrrp_penalty_multiplier", 1.1)),
            roi_per_1pct_auc_usd=float(cfg.get("roi_per_1pct_auc_usd", 450000.0)),
            total_budget_usd=float(sc.get("budget_usd", 25000.0)),
            interventions=_build_interventions(n_eval, shift),
            feature_names=feature_names,
        )

        # Support optional CSV override in scenario config for Kaggle / Silver table drop-in
        if sc.get("train_csv") and sc.get("eval_csv"):
            train_recs = tuple(
                load_records_from_csv(
                    sc["train_csv"],
                    feature_columns=feature_names,
                    include_labels=True,
                    problem_id=problem_id,
                )
            )
            eval_recs = tuple(
                load_records_from_csv(
                    sc["eval_csv"],
                    feature_columns=feature_names,
                    include_labels=True,
                    problem_id=problem_id,
                )
            )
        else:
            train_recs = _generate_cohort_records(
                sc_rng, problem_id, scenario_id, n_train, shift, horizon_days, "T"
            )
            eval_recs = _generate_cohort_records(
                sc_rng, problem_id, scenario_id, n_eval, shift, horizon_days, "E"
            )

        instances.append(
            ClinicalBenchmarkInstance(
                problem_id=str(cfg["problem_id"]),
                scenario_id=scenario_id,
                scenario_name=str(sc["scenario_name"]),
                description=str(sc.get("description", "")),
                cohort_spec=spec,
                train_records=train_recs,
                eval_records=eval_recs,
            )
        )

    if perturb:
        eff_seed = (
            perturb_seed
            if perturb_seed is not None
            else int(os.environ.get("AE_PERTURB_SEED") or DEFAULT_PERTURB_SEED)
        )
        return perturb_instances(instances, eff_seed)
    return instances


def build_heldout_benchmark_instances(
    problem_id: str = "readmission_30d",
    *,
    heldout_seed: Optional[int] = None,
) -> List[ClinicalBenchmarkInstance]:
    """Builds a stratified 3-scenario held-out test suite generated from `AE_HELDOUT_SEED`."""
    cfg = load_problem_config(problem_id)
    eff_seed = (
        heldout_seed
        if heldout_seed is not None
        else int(os.environ.get("AE_HELDOUT_SEED") or DEFAULT_HELDOUT_SEED)
    )
    horizon_days = int(cfg.get("horizon_days", 30))
    feature_names = tuple(cfg.get("feature_names", COMMON_CLINICAL_FEATURES))

    heldout_shifts = [
        ("HO-01", "Held-Out Stratified Multi-Morbid & SDoH Stress Cohort", "high_sdoh", 360, 180, 25000.0),
        ("HO-02", "Held-Out Cardiorenal & Hemodynamic Decompensation Cohort", "cardiorenal", 340, 170, 24000.0),
        ("HO-03", "Held-Out Capacity-Constrained SNF & Acute Triage Cohort", "capacity_crunch", 340, 170, 20000.0),
    ]
    instances: List[ClinicalBenchmarkInstance] = []
    for idx, (suffix, name, shift, n_tr, n_ev, budget) in enumerate(heldout_shifts):
        rng = random.Random(eff_seed + idx * 997 + int(cfg.get("base_seed", 11000)))
        sc_id = f"{problem_id.upper()[:6]}-{suffix}"
        spec = CohortSpec(
            problem_id=str(cfg["problem_id"]),
            scenario_id=sc_id,
            scenario_name=name,
            horizon_days=horizon_days,
            target_auc=float(cfg.get("target_auc", 0.75)),
            target_specificity=float(cfg.get("target_specificity", 0.80)),
            min_sensitivity_floor=float(cfg.get("min_sensitivity_floor", 0.55)),
            event_cost_usd=float(cfg.get("event_cost_usd", 16000.0)),
            hrrp_penalty_multiplier=float(cfg.get("hrrp_penalty_multiplier", 1.1)),
            roi_per_1pct_auc_usd=float(cfg.get("roi_per_1pct_auc_usd", 450000.0)),
            total_budget_usd=budget,
            interventions=_build_interventions(n_ev, shift),
            feature_names=feature_names,
        )
        train_recs = _generate_cohort_records(rng, problem_id, sc_id, n_tr, shift, horizon_days, "HT")
        eval_recs = _generate_cohort_records(rng, problem_id, sc_id, n_ev, shift, horizon_days, "HE")
        instances.append(
            ClinicalBenchmarkInstance(
                problem_id=str(cfg["problem_id"]),
                scenario_id=sc_id,
                scenario_name=name,
                description=f"Stratified held-out scenario ({shift})",
                cohort_spec=spec,
                train_records=train_recs,
                eval_records=eval_recs,
            )
        )
    return perturb_instances(instances, eff_seed ^ 0x5A5A5A)
