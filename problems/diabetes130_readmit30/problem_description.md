# Diabetes 130-US Hospitals: 30-Day Readmission (Real Data)

## 1. Task
Predict whether a diabetic inpatient encounter is followed by an **inpatient readmission within 30 days**. The data
is the public UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset (Strack et al., 2014): about 99k encounters
from about 70k patients at 130 US hospitals. About **11.4%** of encounters are positive.

This is real, noisy, claims/EHR-style data. Published gradient-boosting models reach roughly **0.65–0.70 ROC-AUC**,
so gains of +0.005 AUC are meaningful. Optimise genuine generalisation, not quirks of one sample.

## 2. Contract
Define exactly one function inside the EVOLVE-BLOCK:

```python
def fit_and_score_risk(task: TabularTask, train: Dict[str, np.ndarray], y_train: np.ndarray,
                       eval_cols: Dict[str, np.ndarray]) -> Sequence[float]
```

- **Return value:** one probability in `[0, 1]` per evaluation row, **in the same row order as `eval_cols`**.
- **Columns:** `train` / `eval_cols` map column name to array. Every column in `task.numeric_columns` and
  `task.categorical_columns` is present.
  - **Numeric** columns are `float64`, with `NaN` for missing.
  - **Categorical** columns are object arrays of `str`, with `None` for missing (the raw file's `?`). In
    `max_glu_serum` / `A1Cresult`, the literal string `"None"` means *test not performed*. It is a real category,
    not missing data.
- **Labels:** `y_train` is an int array of 0/1. Evaluation labels are never available to the candidate.
- **Hidden fields:** rows are shuffled, and there are no encounter or patient IDs.
- **Round sizes:** `task.n_train` / `task.n_eval` give the round sizes. Each fitness round uses about 20k training
  rows and about 6.6k evaluation rows.
- **Time limit:** about 120 s CPU per round, on one thread. Keep `n_jobs=1` and `verbose=-1`.
- **Output errors:** returning the wrong length, NaN, or values outside `[0, 1]` costs one hard violation for that
  round.
- **Crashes:** an exception costs one hard violation for that round only.
- **Sandbox:** file, network, subprocess and new-module imports are blocked. A sandbox breach scores the whole
  program as invalid.

### Columns
- **Numeric:**
  - `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`.
  - `number_outpatient` / `number_emergency` / `number_inpatient`: visit counts in the prior year.
  - `number_diagnoses`.
- **Categorical:**
  - **Demographics:** `race`, `gender`, `age` (10-year bands such as `"[70-80)"`), `weight` (about 97% missing).
  - **Admission and discharge:** `admission_type_id`, `discharge_disposition_id`, `admission_source_id` (numeric
    codes stored as strings).
  - **Payer and specialty:** `payer_code`, `medical_specialty` (high missingness, which may itself be informative).
  - **Diagnoses:** `diag_1`, `diag_2`, `diag_3`, as ICD-9 codes such as `"428"`, `"250.83"`, `"V57"`. These are high
    cardinality (700+ levels).
  - **Lab results:** `max_glu_serum`, `A1Cresult`.
  - **Medications:** 23 columns (`metformin`, ..., `insulin`, ...), each with values `No` / `Steady` / `Up` /
    `Down`. Also `change` (`Ch`/`No`) and `diabetesMed` (`Yes`/`No`).

## 3. Fitness
$$\text{Fitness} = 10{,}000 \times \overline{\text{ROC-AUC}}_{\text{rounds}} - 10^6 \times N_{\text{violations}}$$

- **Rounds:** each round trains on a fresh random 20k-row subsample of the training patients and scores a disjoint
  shard of held-out selection patients. +0.01 AUC is worth +100 points.
- **Patient grouping:** patients never appear in both train and eval.
- **Insights:** they also report PR-AUC, Brier score, mean prediction vs prevalence, and the worst/best subgroup AUC
  by race, gender and age. These are diagnostics, not fitness terms.
- **Final validation:** finalists are re-scored once on a **locked test set** of separate patients, trained on all
  training patients, and compared against a tuned LightGBM baseline with a paired bootstrap CI. Overfitting the
  selection shards will show up there.

## 4. Frozen toolkit (`clinical_toolkit`)
- **`TabularEncoder(min_count=20, one_hot=False)`**
  - **Methods:** `fit(task, cols, *, extra_numeric=(), extra_categorical=())`, `transform(cols) -> np.ndarray`,
    `fit_transform(...)`. Attributes `feature_names` and `categorical_indices`.
  - **Numeric** columns pass through, with NaN preserved.
  - **Categorical** columns become integer codes, learnt on the data passed to `fit`:
    - values seen at least `min_count` times get codes `1..K`;
    - rare and unseen values share code `0` (OTHER);
    - missing stays NaN.
  - **LightGBM:** pass `categorical_feature=enc.categorical_indices` to `LGBMClassifier.fit`.
  - **One-hot:** `one_hot=True` emits 0/1 indicators (missing gives all zeros). Use it for linear models, and
    impute numeric NaN yourself.
  - **Derived columns:** to add your own, put the new arrays into **both** dicts and register them with
    `extra_numeric` / `extra_categorical`.
- **`icd9_group(code) -> str`**
  - Maps an ICD-9 code to the Strack et al. (2014) chapter groups: `circulatory`, `respiratory`, `digestive`,
    `diabetes`, `injury`, `musculoskeletal`, `genitourinary`, `neoplasms`, `other` (including V/E codes) and
    `missing`.
  - Never raises.
- **`icd9_comorbidities(*diag_cols) -> Dict[str, np.ndarray]`**
  - Charlson-style 0/1 flags across any number of diagnosis columns: `cm_mi`, `cm_chf`, `cm_pvd`,
    `cm_cerebrovascular`, `cm_dementia`, `cm_pulmonary`, `cm_rheumatic`, `cm_peptic_ulcer`, `cm_liver`,
    `cm_hemiplegia`, `cm_renal`, `cm_cancer`, `cm_metastatic`, `cm_hiv`, `cm_diabetes_complicated`.
  - Also `cm_count`, the number of distinct conditions.
  - All values are float64. It is a fixed clinical mapping, not fitted to data.
- **`oof_target_encode(train_col, y_train, eval_col, *, n_folds=5, smoothing=20.0, seed=0) -> (train_enc, eval_enc)`**
  - Smoothed mean-target encoding of one categorical column, using only `y_train`.
  - Training rows are encoded **out-of-fold**, so a row never sees its own label. Eval rows use all training rows.
  - Unseen levels get the prior, and `None` is its own level.
- **`clinical_models.TabularTask`**
  - Fields: `problem_id`, `round_id`, `numeric_columns`, `categorical_columns`, `n_train`, `n_eval`,
    `label_description`.

## 5. Ideas the harness does not do for you
**The seed is already a tuned LightGBM.** Its hyper-parameters came from random search on these fitness rounds, so
re-tuning alone will not move the score. Look for structural gains:

- **Features:** comorbidity patterns and interactions (`icd9_comorbidities`), target encodings of high-cardinality
  codes (`oof_target_encode`), utilisation intensity (`number_inpatient` is a strong signal), and
  medication-change patterns, including insulin dosing direction.
- **Model families:** both `lightgbm` and `xgboost` are pre-imported. They split differently: XGBoost treats
  category codes as ordinal unless you one-hot or target-encode them. sklearn linear models need imputation and
  `one_hot=True`.
- **Ensembling:** blend heterogeneous models, using out-of-fold stacking within `train` only. Remember the time
  limit: a 5-fold stack of two GBDTs costs about 12 fits per round.
- **Objective:** class weighting and monotone constraints. The fitness is AUC, so calibration does not change it.

## 6. Data provenance & setup
- **Download:** `scripts/fetch_diabetes130.sh` downloads the UCI archive (CC BY 4.0) into
  `data/diabetes130/`, which is gitignored.
- **Excluded encounters:** those discharged as expired or to hospice (`discharge_disposition_id` ∈ {11, 13, 14, 19,
  20, 21}) are dropped, because they cannot be readmitted.
- **Splits:** patients (`patient_nbr`) are hashed into train (60%), selection (20%) and locked test (20%).
