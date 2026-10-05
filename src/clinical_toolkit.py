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

"""Frozen candidate-facing toolkit (`clinical_toolkit.py`) for feature engineering, OOF GBDT stacking, and care allocation.

Why this module exists:
Per the AlphaEvolve engineering playbook, domain mechanics, feature matrix extraction, out-of-fold GBDT stacking,
and Stage 2 resource feasibility checks belong in a frozen toolkit so candidate `EVOLVE-BLOCK` code stays concise
and never crashes on shape mismatches or non-atomic budget/slot checks.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionTier,
    PatientRecord,
    TabularTask,
)


def safe_logit(p: float, eps: float = 1e-4) -> float:
    """Clamped log-odds transform `log(p / (1 - p))`."""
    c = max(eps, min(1.0 - eps, float(p)))
    return math.log(c / (1.0 - c))


def safe_sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-min(float(x), 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(float(x), -60.0))
    return z / (1.0 + z)


def compute_urgency_sample_weights(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    early_event_boost: float = 0.30,
) -> np.ndarray:
    """Weights positive training cases by time-to-event urgency so early readmissions/admissions get stronger signal."""
    weights = np.ones(len(train_records), dtype=np.float64)
    horizon = float(max(1, cohort_spec.horizon_days))
    for i, p in enumerate(train_records):
        if p.outcome_label == 1 and p.time_to_event_days is not None:
            urgency = max(0.0, min(1.0, 1.0 - float(p.time_to_event_days) / horizon))
            weights[i] = 1.0 + early_event_boost * urgency
    return weights


class ClinicalFeatureBuilder:
    """Transforms `PatientRecord` sequences into deterministic numpy feature matrices `(X, y)`.

    When `include_interactions=True`, appends engineered clinical syndrome interaction features:
      - Cardiorenal syndrome index (`(65 - egfr)_+ * log1p((nt_probnp - 450)_+ / 800)`)
      - Loop diuretic resistance (`(weight_gain_7d_kg - 0.6)_+ * (loop_diuretic_dose_mg - 25)_+ / 50`)
      - Sepsis hemodynamic shock x impaired lactate clearance
      - SDoH deprivation x medication non-adherence x Charlson comorbidity spiral
      - Acute ED velocity x ED boarding hours x vital instability
      - Electrolyte/anemia fragility & SNF deconditioning interactions
      - Elderly low-acuity elective stay indicator (correcting legacy baseline over-prediction)
      - Legacy submodel disagreement dispersion (`max - min` and `ehr - claims` log-odds delta)
    """

    def __init__(self, include_interactions: bool = True) -> None:
        self.include_interactions = include_interactions
        self.feature_Keys: Tuple[str, ...] = ()

    def _extract_row(self, p: PatientRecord, feature_names: Tuple[str, ...]) -> List[float]:
        f = p.features
        base_prob = float(p.baseline_api_prob)
        claims_s = float(p.baseline_submodel_scores.get("claims_linear", base_prob))
        ehr_s = float(p.baseline_submodel_scores.get("ehr_vitals", base_prob))
        rules_s = float(p.baseline_submodel_scores.get("utilization_rules", base_prob))

        row: List[float] = [
            float(p.age),
            float(p.sex),
            float(p.charlson_index),
            float(p.prior_ed_visits_6m),
            float(p.prior_ip_admissions_12m),
            float(p.length_of_stay_days),
            float(p.sdoh_deprivation_index),
            float(p.acuity_score),
            base_prob,
            safe_logit(base_prob),
            claims_s,
            ehr_s,
            rules_s,
        ]
        for name in feature_names:
            row.append(float(f.get(name, 0.0)))

        if self.include_interactions:
            egfr = float(f.get("egfr", 70.0))
            hba1c = float(f.get("hba1c", 5.6))
            hb = float(f.get("hemoglobin", 12.5))
            na = float(f.get("sodium", 139.0))
            bnp = float(f.get("nt_probnp", 400.0))
            lvef = float(f.get("lvef", 50.0))
            diuretic = float(f.get("loop_diuretic_dose_mg", 20.0))
            wt_gain = float(f.get("weight_gain_7d_kg", 0.0))
            lactate = float(f.get("lactate_peak", 1.4))
            lac_clear = float(f.get("lactate_clearance_6h", 0.35))
            shock = float(f.get("shock_index", 0.70))
            organ_dys = float(f.get("sepsis_organ_dysfunction_count", 0.0))
            vital_instab = float(f.get("vital_instability_count", 0.0))
            boarding = float(f.get("ed_boarding_hours", 2.0))
            adherence = float(f.get("med_adherence_pdc", 0.80))
            snf = float(f.get("discharge_to_snf", 0.0))

            cardiorenal = max(0.0, (65.0 - egfr) / 28.0) * math.log1p(max(0.0, bnp - 450.0) / 800.0)
            diuretic_res = max(0.0, wt_gain - 0.6) * max(0.0, (diuretic - 25.0) / 50.0) * (1.3 if lvef < 42.0 else 0.85)
            shock_clearance = max(0.0, shock - 0.76) * max(0.0, 0.42 - lac_clear) * 3.2
            lactate_organ = max(0.0, lactate - 1.8) * (1.0 + 0.45 * organ_dys)
            sdoh_nonadherence = max(0.0, p.sdoh_deprivation_index - 0.32) * max(0.0, 0.82 - adherence) * (1.0 + 0.32 * p.charlson_index)
            ed_boarding_vel = math.log1p(p.prior_ed_visits_6m + p.prior_ip_admissions_12m) * max(0.0, (boarding - 2.2) / 4.5) * (1.0 + 0.35 * vital_instab)
            na_anemia = (1.0 if abs(na - 139.0) > 3.5 else 0.0) * max(0.0, (13.0 - hb) / 2.2)
            glycemic_renal = max(0.0, hba1c - 6.8) * max(0.0, (70.0 - egfr) / 22.0)
            snf_decond = snf * max(0.0, (p.length_of_stay_days - 2.5) / 4.0) * (1.0 if p.acuity_score > 3.5 else 0.5)
            elderly_elective = 1.0 if (p.age > 70.0 and p.acuity_score < 3.2 and vital_instab == 0.0) else 0.0
            submodel_spread = max(claims_s, ehr_s, rules_s) - min(claims_s, ehr_s, rules_s)
            ehr_vs_claims_logit = safe_logit(ehr_s) - safe_logit(claims_s)

            row.extend(
                [
                    cardiorenal,
                    diuretic_res,
                    shock_clearance,
                    lactate_organ,
                    sdoh_nonadherence,
                    ed_boarding_vel,
                    na_anemia,
                    glycemic_renal,
                    snf_decond,
                    elderly_elective,
                    submodel_spread,
                    ehr_vs_claims_logit,
                ]
            )
        return row

    def fit_transform(
        self,
        cohort_spec: CohortSpec,
        train_records: Sequence[PatientRecord],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts `(X_train, y_train)` numpy arrays from `train_records`."""
        self.feature_Keys = tuple(cohort_spec.feature_names)
        x_rows = [self._extract_row(p, self.feature_Keys) for p in train_records]
        y_rows = [int(p.outcome_label or 0) for p in train_records]
        return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int32)

    def transform(
        self,
        cohort_spec: CohortSpec,
        records: Sequence[PatientRecord],
    ) -> np.ndarray:
        """Extracts `X` numpy array from `records` using the cohort's feature order."""
        f_names = self.feature_Keys or tuple(cohort_spec.feature_names)
        x_rows = [self._extract_row(p, f_names) for p in records]
        return np.asarray(x_rows, dtype=np.float64)


class RiskBlender:
    """Multi-model clinical risk stacker combining L2 Logistic, GradientBoosting, LightGBM, and XGBoost with OOF meta-calibration."""

    def __init__(
        self,
        *,
        linear_weight: float = 0.35,
        lgbm_weight: float = 0.30,
        xgb_weight: float = 0.25,
        hgb_weight: float = 0.10,
        baseline_prior_weight: float = 0.05,
        use_oof_meta_learner: bool = True,
        n_folds: int = 3,
        random_state: int = 42,
    ) -> None:
        self.linear_weight = linear_weight
        self.lgbm_weight = lgbm_weight
        self.xgb_weight = xgb_weight
        self.hgb_weight = hgb_weight
        self.baseline_prior_weight = baseline_prior_weight
        self.use_oof_meta_learner = use_oof_meta_learner
        self.n_folds = n_folds
        self.random_state = random_state

    def _build_models(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        import lightgbm as lgb
        import xgboost as xgb

        lr = LogisticRegression(C=0.40, max_iter=300, random_state=self.random_state)
        lgbm = lgb.LGBMClassifier(
            n_estimators=65,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.055,
            min_child_samples=16,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.1,
            reg_lambda=2.2,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=1,
        )
        xgbm = xgb.XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.055,
            min_child_weight=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=2.2,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        hgb = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.06,
            min_samples_leaf=18,
            subsample=0.85,
            random_state=self.random_state,
        )
        return lr, lgbm, xgbm, hgb

    def fit_predict(
        self,
        cohort_spec: CohortSpec,
        train_records: Sequence[PatientRecord],
        eval_records: Sequence[PatientRecord],
        *,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        x_eval: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        include_interactions: bool = True,
    ) -> Dict[str, float]:
        """Fits the base learners + optional Stratified K-Fold OOF meta-calibrator and returns `{patient_id: prob}`."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        if x_train is None or y_train is None or x_eval is None:
            builder = ClinicalFeatureBuilder(include_interactions=include_interactions)
            x_train, y_train = builder.fit_transform(cohort_spec, train_records)
            x_eval = builder.transform(cohort_spec, eval_records)

        if len(np.unique(y_train)) < 2:
            return {p.patient_id: float(p.baseline_api_prob) for p in eval_records}

        if sample_weight is None:
            sample_weight = compute_urgency_sample_weights(cohort_spec, train_records)

        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(x_train)
        x_ev_scaled = scaler.transform(x_eval)

        lr, lgbm, xgbm, hgb = self._build_models()
        lr.fit(x_tr_scaled, y_train, sample_weight=sample_weight)
        lgbm.fit(x_train, y_train, sample_weight=sample_weight)
        xgbm.fit(x_train, y_train, sample_weight=sample_weight)
        hgb.fit(x_train, y_train, sample_weight=sample_weight)

        p_lr = np.clip(lr.predict_proba(x_ev_scaled)[:, 1], 1e-4, 1.0 - 1e-4)
        p_lgb = np.clip(lgbm.predict_proba(x_eval)[:, 1], 1e-4, 1.0 - 1e-4)
        p_xgb = np.clip(xgbm.predict_proba(x_eval)[:, 1], 1e-4, 1.0 - 1e-4)
        p_hgb = np.clip(hgb.predict_proba(x_eval)[:, 1], 1e-4, 1.0 - 1e-4)
        p_base_ev = np.array([float(p.baseline_api_prob) for p in eval_records], dtype=np.float64)

        w_total = max(
            1e-9,
            self.linear_weight + self.lgbm_weight + self.xgb_weight + self.hgb_weight + self.baseline_prior_weight,
        )
        prior_logits_ev = (
            self.linear_weight * np.log(p_lr / (1.0 - p_lr))
            + self.lgbm_weight * np.log(p_lgb / (1.0 - p_lgb))
            + self.xgb_weight * np.log(p_xgb / (1.0 - p_xgb))
            + self.hgb_weight * np.log(p_hgb / (1.0 - p_hgb))
            + self.baseline_prior_weight * np.log(np.clip(p_base_ev, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(p_base_ev, 1e-4, 1.0 - 1e-4)))
        ) / w_total

        if self.use_oof_meta_learner and int(np.sum(y_train == 1)) >= self.n_folds:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            oof_meta = np.zeros((len(train_records), 5), dtype=np.float64)
            p_base_tr = np.clip(
                np.array([float(p.baseline_api_prob) for p in train_records], dtype=np.float64),
                1e-4,
                1.0 - 1e-4,
            )
            oof_meta[:, 4] = np.log(p_base_tr / (1.0 - p_base_tr))

            for tr_idx, val_idx in skf.split(x_train, y_train):
                f_lr, f_lgb, f_xgb, f_hgb = self._build_models()
                sw_fold = sample_weight[tr_idx]
                f_lr.fit(x_tr_scaled[tr_idx], y_train[tr_idx], sample_weight=sw_fold)
                f_lgb.fit(x_train[tr_idx], y_train[tr_idx], sample_weight=sw_fold)
                f_xgb.fit(x_train[tr_idx], y_train[tr_idx], sample_weight=sw_fold)
                f_hgb.fit(x_train[tr_idx], y_train[tr_idx], sample_weight=sw_fold)

                for col_i, (m, x_v) in enumerate(
                    [
                        (f_lr, x_tr_scaled[val_idx]),
                        (f_lgb, x_train[val_idx]),
                        (f_xgb, x_train[val_idx]),
                        (f_hgb, x_train[val_idx]),
                    ]
                ):
                    pv = np.clip(m.predict_proba(x_v)[:, 1], 1e-4, 1.0 - 1e-4)
                    oof_meta[val_idx, col_i] = np.log(pv / (1.0 - pv))

            meta_clf = LogisticRegression(C=0.5, max_iter=200, random_state=self.random_state)
            meta_clf.fit(oof_meta, y_train)

            ev_meta = np.column_stack(
                [
                    np.log(p_lr / (1.0 - p_lr)),
                    np.log(p_lgb / (1.0 - p_lgb)),
                    np.log(p_xgb / (1.0 - p_xgb)),
                    np.log(p_hgb / (1.0 - p_hgb)),
                    np.log(np.clip(p_base_ev, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(p_base_ev, 1e-4, 1.0 - 1e-4))),
                ]
            )
            meta_probs = np.clip(meta_clf.predict_proba(ev_meta)[:, 1], 1e-4, 1.0 - 1e-4)
            meta_logits = np.log(meta_probs / (1.0 - meta_probs))
            final_logits = 0.65 * prior_logits_ev + 0.35 * meta_logits
        else:
            final_logits = prior_logits_ev

        out: Dict[str, float] = {}
        for idx, p in enumerate(eval_records):
            prob = max(1e-4, min(1.0 - 1e-4, safe_sigmoid(float(final_logits[idx]))))
            out[p.patient_id] = round(prob, 6)
        return out


class CareAllocationState:
    """Incremental, constraint-safe Stage 2 care-intervention allocator.

    Mirrors every Stage 2 constraint enforced by `clinical_metrics.evaluate_cohort_predictions_and_allocation`:
      - Per-tier slot capacity (`spec.max_slots`)
      - Cohort total budget (`cohort_spec.total_budget_usd`)
      - Patient clinical acuity floor (`p.acuity_score >= spec.min_acuity_score`)
      - Patient safety contraindications (`set(p.contraindications) & set(spec.contraindicated_flags) == empty`)
    """

    def __init__(
        self,
        cohort_spec: CohortSpec,
        eval_records: Sequence[PatientRecord],
    ) -> None:
        self.cohort_spec = cohort_spec
        self.patients_by_id: Dict[str, PatientRecord] = {p.patient_id: p for p in eval_records}
        self.assignments: Dict[str, InterventionTier] = {
            p.patient_id: InterventionTier.NONE for p in eval_records
        }
        self.slot_counts: Dict[InterventionTier, int] = {t: 0 for t in InterventionTier}
        self.slot_counts[InterventionTier.NONE] = len(self.patients_by_id)
        self.spent_usd: float = 0.0

    def remaining_budget(self) -> float:
        """Remaining Stage 2 intervention budget in USD."""
        return float(self.cohort_spec.total_budget_usd - self.spent_usd)

    def remaining_slots(self, tier: InterventionTier) -> int:
        """Remaining capacity slots for `tier`."""
        spec = self.cohort_spec.interventions.get(tier)
        if spec is None:
            return 0
        if tier == InterventionTier.NONE:
            return len(self.patients_by_id)
        return max(0, spec.max_slots - self.slot_counts.get(tier, 0))

    def is_clinically_eligible(self, patient_id: str, tier: InterventionTier) -> bool:
        """Checks clinical acuity floor and safety contraindications (ignoring budget/slots)."""
        p = self.patients_by_id.get(patient_id)
        spec = self.cohort_spec.interventions.get(tier)
        if p is None or spec is None:
            return False
        if tier == InterventionTier.NONE:
            return True
        if p.acuity_score + 1e-9 < spec.min_acuity_score:
            return False
        if set(p.contraindications) & set(spec.contraindicated_flags):
            return False
        return True

    def fits(self, patient_id: str, tier: InterventionTier) -> bool:
        """Returns True iff assigning `patient_id` to `tier` satisfies clinical, slot, and budget rules."""
        if not self.is_clinically_eligible(patient_id, tier):
            return False
        cur_tier = self.assignments.get(patient_id, InterventionTier.NONE)
        if cur_tier == tier:
            return True
        new_spec = self.cohort_spec.interventions[tier]
        old_spec = self.cohort_spec.interventions[cur_tier]
        if tier != InterventionTier.NONE and self.remaining_slots(tier) <= 0:
            return False
        delta_cost = new_spec.cost_usd - old_spec.cost_usd
        if self.spent_usd + delta_cost > self.cohort_spec.total_budget_usd + 1e-6:
            return False
        return True

    def assign(self, patient_id: str, tier: InterventionTier) -> bool:
        """Assigns `patient_id` to `tier` if feasible; returns False without mutating state otherwise."""
        if not self.fits(patient_id, tier):
            return False
        cur_tier = self.assignments[patient_id]
        if cur_tier == tier:
            return True
        old_spec = self.cohort_spec.interventions[cur_tier]
        new_spec = self.cohort_spec.interventions[tier]
        self.slot_counts[cur_tier] -= 1
        self.slot_counts[tier] = self.slot_counts.get(tier, 0) + 1
        self.spent_usd += new_spec.cost_usd - old_spec.cost_usd
        self.assignments[patient_id] = tier
        return True

    def remove(self, patient_id: str) -> InterventionTier:
        """Resets `patient_id` to `InterventionTier.NONE` and returns its previous tier."""
        prev = self.assignments.get(patient_id, InterventionTier.NONE)
        if prev != InterventionTier.NONE:
            self.assign(patient_id, InterventionTier.NONE)
        return prev

    def assign_all(self, pairs: Iterable[Tuple[str, InterventionTier]]) -> bool:
        """Atomically assigns multiple `(patient_id, tier)` pairs; rolls back completely if any pair fails."""
        snapshot: List[Tuple[str, InterventionTier]] = []
        for pid, tier in pairs:
            if pid not in self.assignments:
                for old_pid, old_tier in reversed(snapshot):
                    self.assign(old_pid, old_tier)
                return False
            prev = self.assignments[pid]
            if not self.assign(pid, tier):
                for old_pid, old_tier in reversed(snapshot):
                    self.assign(old_pid, old_tier)
                return False
            snapshot.append((pid, prev))
        return True

    def expected_net_benefit(
        self,
        patient_id: str,
        risk_prob: float,
        tier: InterventionTier,
    ) -> float:
        """Expected dollar savings minus intervention cost for assigning `patient_id` to `tier`."""
        p = self.patients_by_id.get(patient_id)
        spec = self.cohort_spec.interventions.get(tier)
        if p is None or spec is None or tier == InterventionTier.NONE:
            return 0.0
        unit_event_val = self.cohort_spec.event_cost_usd * self.cohort_spec.hrrp_penalty_multiplier
        urgency_boost = 1.0 + 0.08 * min(1.0, max(0.0, (p.acuity_score - 3.0) / 6.0))
        expected_gross = float(risk_prob) * spec.relative_risk_reduction * urgency_boost * unit_event_val
        return float(expected_gross - spec.cost_usd)

    def all_ok(self) -> bool:
        """Verifies all current assignments satisfy every Stage 2 hard constraint."""
        if self.spent_usd > self.cohort_spec.total_budget_usd + 1e-6:
            return False
        for tier, count in self.slot_counts.items():
            if tier == InterventionTier.NONE:
                continue
            spec = self.cohort_spec.interventions.get(tier)
            if spec is None or count > spec.max_slots:
                return False
        for pid, tier in self.assignments.items():
            if not self.is_clinically_eligible(pid, tier):
                return False
        return len(self.assignments) == len(self.patients_by_id)

    def to_plan(self) -> CareAllocationPlan:
        """Builds a complete `CareAllocationPlan` covering every evaluation patient."""
        return CareAllocationPlan(
            scenario_id=self.cohort_spec.scenario_id,
            assignments=dict(self.assignments),
        )


# =====================================================================================================================
# Real-data tabular track (e.g. `diabetes130_readmit30`)
#
# Why only generic mechanics live here: the synthetic portfolio's `ClinicalFeatureBuilder` encodes the generator's
# latent risk formula, which makes "evolved" lift circular. For real data the toolkit must not pre-chew domain signal;
# it only removes plumbing footguns (NaN handling, unseen categories, stable code assignment, ICD-9 chapter lookup).
# =====================================================================================================================

ICD9_GROUPS: Tuple[str, ...] = (
    "circulatory",
    "respiratory",
    "digestive",
    "diabetes",
    "injury",
    "musculoskeletal",
    "genitourinary",
    "neoplasms",
    "other",
    "missing",
)


def icd9_group(code: Optional[str]) -> str:
    """Maps an ICD-9 diagnosis code to the chapter groups used by Strack et al. (2014), the Diabetes 130 paper.

    circulatory 390-459,785 | respiratory 460-519,786 | digestive 520-579,787 | diabetes 250.xx | injury 800-999 |
    musculoskeletal 710-739 | genitourinary 580-629,788 | neoplasms 140-239 | other (incl. V/E codes) | missing.
    Never raises.
    """
    if code is None:
        return "missing"
    s = str(code).strip()
    if not s or s == "?":
        return "missing"
    if s[0] in "VvEe":
        return "other"
    try:
        v = float(s)
    except ValueError:
        return "other"
    major = int(v)
    if major == 250:
        return "diabetes"
    if 390 <= major <= 459 or major == 785:
        return "circulatory"
    if 460 <= major <= 519 or major == 786:
        return "respiratory"
    if 520 <= major <= 579 or major == 787:
        return "digestive"
    if 800 <= major <= 999:
        return "injury"
    if 710 <= major <= 739:
        return "musculoskeletal"
    if 580 <= major <= 629 or major == 788:
        return "genitourinary"
    if 140 <= major <= 239:
        return "neoplasms"
    return "other"


# Charlson comorbidity groups by ICD-9 major code (simplified from Quan et al., 2005). A fixed clinical prior, never
# fitted to data. Inclusive (lo, hi) ranges of the 3-digit major code.
_CHARLSON_ICD9: Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...] = (
    ("mi", ((410, 410), (412, 412))),
    ("chf", ((425, 425), (428, 428))),
    ("pvd", ((440, 441),)),
    ("cerebrovascular", ((430, 438),)),
    ("dementia", ((290, 290),)),
    ("pulmonary", ((490, 505),)),
    ("rheumatic", ((710, 710), (714, 714), (725, 725))),
    ("peptic_ulcer", ((531, 534),)),
    ("liver", ((570, 573),)),
    ("hemiplegia", ((342, 342), (344, 344))),
    ("renal", ((582, 583), (585, 586), (588, 588))),
    ("cancer", ((140, 195), (200, 208))),
    ("metastatic", ((196, 199),)),
    ("hiv", ((42, 44),)),
)


def _charlson_conditions(code: Optional[str]) -> Tuple[str, ...]:
    s = "" if code is None else str(code).strip()
    if not s or s[0] in "VvEe?":
        return ()
    try:
        v = float(s)
    except ValueError:
        return ()
    major = int(v)
    found = [name for name, ranges in _CHARLSON_ICD9 if any(lo <= major <= hi for lo, hi in ranges)]
    if major == 250 and int(round((v - 250) * 100)) // 10 >= 4:  # 250.4x-250.9x: diabetes with complications
        found.append("diabetes_complicated")
    return tuple(found)


def icd9_comorbidities(*diag_cols: np.ndarray) -> Dict[str, np.ndarray]:
    """Charlson-style comorbidity flags across one or more ICD-9 diagnosis columns.

    Returns `{"cm_<condition>": 0/1 float64, ..., "cm_count": number of distinct conditions}`. A condition coded in
    several columns counts once. V/E codes and missing values contribute nothing. Never raises.
    """
    names = [n for n, _ in _CHARLSON_ICD9] + ["diabetes_complicated"]
    n = len(diag_cols[0]) if diag_cols else 0
    out = {f"cm_{k}": np.zeros(n, dtype=np.float64) for k in names}
    cache: Dict[Optional[str], Tuple[str, ...]] = {}
    for col in diag_cols:
        for i, code in enumerate(np.asarray(col, dtype=object).tolist()):
            if code not in cache:
                cache[code] = _charlson_conditions(code)
            for cond in cache[code]:
                out[f"cm_{cond}"][i] = 1.0
    out["cm_count"] = np.sum([out[f"cm_{k}"] for k in names], axis=0) if n else np.zeros(0)
    return out


def oof_target_encode(
    train_col: np.ndarray,
    y_train: np.ndarray,
    eval_col: np.ndarray,
    *,
    n_folds: int = 5,
    smoothing: float = 20.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed mean-target encoding of one categorical column, using only `y_train`.

    Training rows are encoded out-of-fold, so a row's value never depends on its own label (in-fold encoding leaks
    the label and makes GBDTs over-trust the feature). Eval rows use statistics from all training rows. Encoding is
    `(sum_y + smoothing * prior) / (count + smoothing)`; unseen levels get the prior. `None` is its own level.
    Returns `(train_encoded, eval_encoded)` as float64.
    """
    y = np.asarray(y_train, dtype=np.float64)
    keys = [("__NA__" if v is None else str(v)) for v in np.asarray(train_col, dtype=object).tolist()]
    keys += [("__NA__" if v is None else str(v)) for v in np.asarray(eval_col, dtype=object).tolist()]
    _, inv = np.unique(np.array(keys, dtype=object), return_inverse=True)
    n, k = len(y), int(inv.max()) + 1 if len(inv) else 0
    tr_codes, ev_codes = inv[:n], inv[n:]

    def _encode(fit_idx: np.ndarray, codes: np.ndarray) -> np.ndarray:
        prior = float(y[fit_idx].mean()) if len(fit_idx) else 0.0
        s = np.bincount(tr_codes[fit_idx], weights=y[fit_idx], minlength=k)
        c = np.bincount(tr_codes[fit_idx], minlength=k)
        return ((s + smoothing * prior) / (c + smoothing))[codes]

    folds = np.random.default_rng(seed).permutation(n) % max(2, int(n_folds))
    tr_enc = np.empty(n, dtype=np.float64)
    for f in np.unique(folds):
        held = folds == f
        tr_enc[held] = _encode(np.flatnonzero(~held), tr_codes[held])
    return tr_enc, _encode(np.arange(n), ev_codes)


class TabularEncoder:
    """Turns raw column dicts (`Dict[str, np.ndarray]`) into a float64 feature matrix with stable column order.

    - Numeric columns pass through unchanged; NaN is preserved (GBDTs route missing values natively, and missingness
      is informative in EHR/claims data). Impute yourself before linear models.
    - Categorical columns (default, `one_hot=False`) become float codes learnt on the training data: values seen at
      least `min_count` times get codes `1..K` (most frequent first), rare **and unseen** values share code `0`
      (OTHER), missing (`None`) stays NaN. `categorical_indices` lists those matrix columns for
      `lightgbm.LGBMClassifier.fit(..., categorical_feature=enc.categorical_indices)`.
    - `one_hot=True` expands each categorical into 0/1 indicator columns (frequent values + OTHER); missing gives all
      zeros. `categorical_indices` is then empty.
    - Add your own derived columns to the dicts and register them via `extra_numeric` / `extra_categorical` in `fit`.
      The same columns must be present when calling `transform`.
    """

    OTHER = "__OTHER__"

    def __init__(self, min_count: int = 20, one_hot: bool = False) -> None:
        self.min_count = int(min_count)
        self.one_hot = bool(one_hot)
        self.numeric_columns: Tuple[str, ...] = ()
        self.categorical_columns: Tuple[str, ...] = ()
        self.vocab: Dict[str, Dict[str, int]] = {}
        self.feature_names: List[str] = []
        self.categorical_indices: List[int] = []

    def fit(
        self,
        task: TabularTask,
        cols: Dict[str, np.ndarray],
        *,
        extra_numeric: Sequence[str] = (),
        extra_categorical: Sequence[str] = (),
    ) -> "TabularEncoder":
        self.numeric_columns = tuple(task.numeric_columns) + tuple(extra_numeric)
        self.categorical_columns = tuple(task.categorical_columns) + tuple(extra_categorical)
        self.vocab = {}
        for name in self.categorical_columns:
            counts: Dict[str, int] = {}
            for v in np.asarray(cols[name], dtype=object).tolist():
                if v is not None:
                    counts[str(v)] = counts.get(str(v), 0) + 1
            frequent = sorted((k for k, c in counts.items() if c >= self.min_count), key=lambda k: (-counts[k], k))
            self.vocab[name] = {k: i + 1 for i, k in enumerate(frequent)}

        self.feature_names = list(self.numeric_columns)
        self.categorical_indices = []
        for name in self.categorical_columns:
            if self.one_hot:
                levels = list(self.vocab[name]) + [self.OTHER]
                self.feature_names.extend(f"{name}={lv}" for lv in levels)
            else:
                self.categorical_indices.append(len(self.feature_names))
                self.feature_names.append(name)
        return self

    def transform(self, cols: Dict[str, np.ndarray]) -> np.ndarray:
        blocks: List[np.ndarray] = [
            np.asarray(cols[name], dtype=np.float64).reshape(-1, 1) for name in self.numeric_columns
        ]
        for name in self.categorical_columns:
            vocab = self.vocab[name]
            raw = np.asarray(cols[name], dtype=object).tolist()
            codes = np.array(
                [np.nan if v is None else float(vocab.get(str(v), 0)) for v in raw], dtype=np.float64
            )
            if not self.one_hot:
                blocks.append(codes.reshape(-1, 1))
                continue
            # Indicator layout: frequent levels 1..K first, then OTHER (code 0); missing rows stay all-zero.
            k = len(vocab)
            onehot = np.zeros((len(raw), k + 1), dtype=np.float64)
            valid = ~np.isnan(codes)
            col_idx = np.where(codes[valid] == 0.0, k, codes[valid] - 1.0).astype(int)
            onehot[np.flatnonzero(valid), col_idx] = 1.0
            blocks.append(onehot)
        if not blocks:
            return np.zeros((0, 0), dtype=np.float64)
        return np.hstack(blocks)

    def fit_transform(
        self,
        task: TabularTask,
        cols: Dict[str, np.ndarray],
        *,
        extra_numeric: Sequence[str] = (),
        extra_categorical: Sequence[str] = (),
    ) -> np.ndarray:
        return self.fit(task, cols, extra_numeric=extra_numeric, extra_categorical=extra_categorical).transform(cols)
