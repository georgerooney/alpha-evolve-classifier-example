# AlphaEvolve Clinical Problem Specification — Inpatient Admissions & Acute Bed Capacity Optimization (`inpatient_admission`)

## 1. Clinical & Business Objective
Optimize **Inpatient Admissions & Acute Bed Capacity Optimization** across a multi-scenario clinical cohort over a **14-day horizon**.
- **Target Discrimination Gate**: $\text{ROC-AUC} > 0.78$ (at target specificity $0.80$).
- **Financial Value Model**: Each unplanned clinical event costs **\$14,200** (before CMS HRRP multiplier), and each $+1.0\%$ uplift in portfolio ROC-AUC delivers **\$510,000** in actuarial plan value.

---

## 2. Two-Stage Functional Contract (`EVOLVE-BLOCK`)

Your candidate program must define two functions with exact signatures:

```python
def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    ...

def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    ...
```

### Stage 1: `fit_and_score_risk`
- `train_records` contain labeled `PatientRecord` instances (`outcome_label` $\in \{0, 1\}$, `time_to_event_days`).
- `eval_records` have `outcome_label=None` and `time_to_event_days=None` (ground-truth labels never enter worker memory).
- Must return a dictionary `{patient_id: probability}` covering **every** `patient_id` in `eval_records` with a finite float in $[0.0, 1.0]$.
- Legacy production model predictions are provided on every record (`p.baseline_api_prob` and `p.baseline_submodel_scores` with keys `"claims_linear"`, `"ehr_vitals"`, `"utilization_rules"`). These legacy linear models capture main effects but miss non-linear cross-organ syndromes, SDoH adherence spirals, and elderly low-acuity elective over-prediction.

### Stage 2: `allocate_interventions`
- Executed only after Stage 1 probabilities are committed.
- Must return a `CareAllocationPlan(scenario_id=cohort_spec.scenario_id, assignments={patient_id: InterventionTier})`.
- Hard constraints enforced by the evaluator ($-10^6$ penalty per violation):
  1. Every `patient_id` in `eval_records` must be assigned a valid `InterventionTier`.
  2. Assigned count for each tier must not exceed `cohort_spec.interventions[tier].max_slots`.
  3. Total cost across all assigned interventions must not exceed `cohort_spec.total_budget_usd`.
  4. Clinical acuity eligibility: `p.acuity_score >= spec.min_acuity_score`.
  5. Clinical safety contraindications: `set(p.contraindications) & set(spec.contraindicated_flags)` must be empty.

---

## 3. Fitness Function

$$\text{Fitness} = 10{,}000 \times \left( 0.50 \cdot \text{AUC}_{\text{ROC}} + 0.25 \cdot \text{AUC}_{\text{PR}} + 0.15 \cdot \text{Sens@Spec}_{0.80} + 0.10 \cdot (1 - \text{Brier}) \right) + 5.0 \cdot \tanh\!\left(\frac{\text{NetSavings}_{\$}}{50{,}000}\right) - 10^6 \cdot N_{\text{violations}}$$

---

## 4. Frozen Toolkit API (`clinical_toolkit`)

Always use `clinical_toolkit` rather than re-implementing constraint bookkeeping:

- **`ClinicalFeatureBuilder(include_interactions: bool = True)`**:
  - `fit_transform(cohort_spec, train_records) -> Tuple[np.ndarray, np.ndarray]`
  - `transform(cohort_spec, eval_records) -> np.ndarray`
  - Setting `include_interactions=True` adds 12 engineered features: 10 non-linear syndrome terms (cardiorenal index, loop diuretic resistance, shock-index $\times$ lactate clearance, peak lactate $\times$ organ dysfunction, SDoH $\times$ non-adherence spiral, ED boarding velocity, electrolyte/anemia fragility, glycemic-renal interaction, SNF deconditioning, elderly elective correction) plus 2 legacy-submodel disagreement terms (submodel spread, EHR-vs-claims logit delta).
- **`RiskBlender(linear_weight=0.35, lgbm_weight=0.30, xgb_weight=0.25, hgb_weight=0.10, baseline_prior_weight=0.05, use_oof_meta_learner=True, n_folds=3, random_state=42)`**:
  - `fit_predict(cohort_spec, train_records, eval_records, x_train=..., y_train=..., x_eval=...) -> Dict[str, float]`
- **`CareAllocationState(cohort_spec, eval_records)`**:
  - `state.fits(patient_id, tier) -> bool`: checks clinical eligibility, slot capacity, and remaining budget.
  - `state.assign(patient_id, tier) -> bool`: mutates state iff `fits` is True.
  - `state.assign_all(pairs) -> bool`: atomic all-or-nothing multi-patient assignment (rolls back on any overflow).
  - `state.expected_net_benefit(patient_id, risk_prob, tier) -> float`: expected avoided event value minus intervention cost.
  - `state.remaining_budget() -> float`, `state.remaining_slots(tier) -> int`, `state.all_ok() -> bool`, `state.to_plan() -> CareAllocationPlan`.
- **Pre-imported Sandbox Stack**: `numpy`, `scipy`, `sklearn` (`linear_model`, `ensemble`, `calibration`, `preprocessing`), `lightgbm`, and `xgboost` are pre-imported in the worker process. Do not write to disk, spawn subprocesses, or open network sockets.
