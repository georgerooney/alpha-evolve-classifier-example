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

"""Pluggable dataset loaders (CSV / Kaggle / Snowflake on AWS) and legacy baseline model API adapters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import pathlib
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from clinical_models import PatientRecord

_SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_ident(name: str, field_label: str) -> str:
    if not _SAFE_SQL_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Unsafe SQL identifier for {field_label}: {name!r}")
    return name


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


class SimulatedVertexModelAdapter:
    """Simulates existing legacy baseline models accessed via GCP / Vertex AI / Snowflake APIs.

    Why three submodels:
    A typical healthcare production stack typically combines separate claims-history utilization scores,
    EHR vital/lab linear scores, and rules-based acuity indices. These legacy linear models capture main
    effects well (~0.64-0.73 AUC) but miss non-linear cross-domain interactions (e.g., cardiorenal
    decompensation, shock-index x lactate clearance, or SDoH deprivation x medication non-adherence).
    """

    def __init__(self, problem_id: str = "readmission_30d") -> None:
        self.problem_id = problem_id

    def score_patient(
        self,
        *,
        age: float,
        charlson_index: float,
        prior_ed_visits_6m: int,
        prior_ip_admissions_12m: int,
        length_of_stay_days: float,
        sdoh_deprivation_index: float,
        acuity_score: float,
        features: Mapping[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Returns `(baseline_api_prob, baseline_submodel_scores)`."""
        claims_logit = (
            -2.15
            + 0.018 * (age - 60.0)
            + 0.16 * charlson_index
            + 0.21 * prior_ed_visits_6m
            + 0.25 * prior_ip_admissions_12m
            + 0.05 * length_of_stay_days
        )
        claims_prob = _sigmoid(claims_logit)

        egfr = float(features.get("egfr", 70.0))
        nt_probnp = float(features.get("nt_probnp", 400.0))
        lactate = float(features.get("lactate_peak", 1.5))
        shock_idx = float(features.get("shock_index", 0.7))
        vital_instab = float(features.get("vital_instability_count", 0.0))

        ehr_logit = (
            -1.95
            + 0.14 * acuity_score
            - 0.012 * (egfr - 65.0)
            + 0.00012 * (nt_probnp - 500.0)
            + 0.18 * (lactate - 1.5)
            + 0.45 * (shock_idx - 0.7)
            + 0.15 * vital_instab
        )
        ehr_prob = _sigmoid(ehr_logit)

        rules_logit = (
            -2.05
            + 0.19 * acuity_score
            + 0.15 * prior_ip_admissions_12m
            + 0.20 * sdoh_deprivation_index
        )
        rules_prob = _sigmoid(rules_logit)

        composite = max(
            1e-4,
            min(1.0 - 1e-4, 0.45 * claims_prob + 0.35 * ehr_prob + 0.20 * rules_prob),
        )
        submodels = {
            "claims_linear": round(claims_prob, 6),
            "ehr_vitals": round(ehr_prob, 6),
            "utilization_rules": round(rules_prob, 6),
        }
        return round(composite, 6), submodels


def _row_to_patient_record(
    row: Mapping[str, Any],
    idx: int,
    feature_columns: Sequence[str],
    model_adapter: SimulatedVertexModelAdapter,
    include_labels: bool,
) -> PatientRecord:
    """Normalizes a case-insensitive CSV or Snowflake row dict into a `PatientRecord`."""
    lower_row = {str(k).lower(): v for k, v in row.items()}
    pid = str(lower_row.get("patient_id") or f"ROW-{idx:05d}").strip()
    age = float(lower_row.get("age") or 62.0)
    sex = int(float(lower_row.get("sex") or 0))
    charlson = float(lower_row.get("charlson_index") or 2.0)
    ed_6m = int(float(lower_row.get("prior_ed_visits_6m") or 0))
    ip_12m = int(float(lower_row.get("prior_ip_admissions_12m") or 0))
    los = float(lower_row.get("length_of_stay_days") or 3.0)
    sdoh = float(lower_row.get("sdoh_deprivation_index") or 0.3)
    acuity = float(lower_row.get("acuity_score") or 3.0)

    feats: Dict[str, float] = {}
    for col in feature_columns:
        raw_val = lower_row.get(col.lower())
        feats[col] = float(raw_val) if raw_val not in (None, "") else 0.0

    if lower_row.get("baseline_api_prob") not in (None, ""):
        base_prob = float(lower_row["baseline_api_prob"])
        submodels = {
            "claims_linear": float(lower_row.get("claims_linear") or base_prob),
            "ehr_vitals": float(lower_row.get("ehr_vitals") or base_prob),
            "utilization_rules": float(lower_row.get("utilization_rules") or base_prob),
        }
    else:
        base_prob, submodels = model_adapter.score_patient(
            age=age,
            charlson_index=charlson,
            prior_ed_visits_6m=ed_6m,
            prior_ip_admissions_12m=ip_12m,
            length_of_stay_days=los,
            sdoh_deprivation_index=sdoh,
            acuity_score=acuity,
            features=feats,
        )

    raw_contra = str(lower_row.get("contraindications") or "").strip()
    contraindications = (
        tuple(x.strip() for x in raw_contra.split("|") if x.strip())
        if raw_contra
        else ()
    )

    outcome: Optional[int] = None
    tte: Optional[float] = None
    if include_labels:
        raw_out = lower_row.get("outcome_label")
        if raw_out not in (None, ""):
            outcome = int(float(raw_out))
        raw_tte = lower_row.get("time_to_event_days")
        if raw_tte not in (None, ""):
            tte = float(raw_tte)

    return PatientRecord(
        patient_id=pid,
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
        contraindications=contraindications,
        outcome_label=outcome,
        time_to_event_days=tte,
    )


class CSVDatasetAdapter:
    """Loads clinical cohort records from a local CSV (e.g., Kaggle or Clinical Silver table export)."""

    def __init__(
        self,
        csv_path: pathlib.Path | str,
        feature_columns: Sequence[str],
        model_adapter: Optional[SimulatedVertexModelAdapter] = None,
        allowed_root: Optional[pathlib.Path] = None,
    ) -> None:
        resolved = pathlib.Path(csv_path).resolve()
        if allowed_root is not None:
            root_resolved = pathlib.Path(allowed_root).resolve()
            if not str(resolved).startswith(str(root_resolved) + "/"):
                raise ValueError(f"CSV path '{resolved}' is outside allowed root '{root_resolved}'")
        if not resolved.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {resolved}")
        self.csv_path = resolved
        self.feature_columns = tuple(feature_columns)
        self.model_adapter = model_adapter or SimulatedVertexModelAdapter()

    def load_records(self, include_labels: bool = True) -> List[PatientRecord]:
        records: List[PatientRecord] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                records.append(
                    _row_to_patient_record(
                        row, idx, self.feature_columns, self.model_adapter, include_labels
                    )
                )
        return records


@dataclass(frozen=True)
class SnowflakeConfig:
    """Connection & table coordinates for an AWS Snowflake Silver environment."""

    account: str
    warehouse: str
    database: str
    schema: str
    table_or_view: str
    role: str = "RL_CLINICAL_DS_READONLY"
    authenticator: str = "externalbrowser"


class SnowflakeDatasetAdapter:
    """Loads clinical cohort records from Snowflake on AWS in the parent evaluator process.

    Why in the parent process:
    Snowflake is queried once at the start of an experiment inside the AWS VPC (via AWS PrivateLink).
    Records are cached in parent RAM and piped into the air-gapped candidate worker (`socket.*` blocked)
    with `outcome_label` stripped from `eval_records`.
    """

    def __init__(
        self,
        config: SnowflakeConfig,
        feature_columns: Sequence[str],
        model_adapter: Optional[SimulatedVertexModelAdapter] = None,
        row_fetcher: Optional[Callable[[str], Sequence[Mapping[str, Any]]]] = None,
    ) -> None:
        _validate_sql_ident(config.warehouse, "warehouse")
        _validate_sql_ident(config.database, "database")
        _validate_sql_ident(config.schema, "schema")
        _validate_sql_ident(config.table_or_view, "table_or_view")
        self.config = config
        self.feature_columns = tuple(_validate_sql_ident(c, "feature_column") for c in feature_columns)
        self.model_adapter = model_adapter or SimulatedVertexModelAdapter()
        self.row_fetcher = row_fetcher

    def build_select_query(self) -> str:
        """Builds a strictly validated SELECT query against the configured Snowflake view/table."""
        core_cols = (
            "PATIENT_ID",
            "AGE",
            "SEX",
            "CHARLSON_INDEX",
            "PRIOR_ED_VISITS_6M",
            "PRIOR_IP_ADMISSIONS_12M",
            "LENGTH_OF_STAY_DAYS",
            "SDOH_DEPRIVATION_INDEX",
            "ACUITY_SCORE",
            "OUTCOME_LABEL",
            "TIME_TO_EVENT_DAYS",
        )
        feat_cols = tuple(c.upper() for c in self.feature_columns)
        all_cols = ", ".join(core_cols + feat_cols)
        fqn = f"{self.config.database}.{self.config.schema}.{self.config.table_or_view}"
        return f"SELECT {all_cols} FROM {fqn}"  # noqa: S608 - identifiers strictly validated via _SAFE_SQL_IDENTIFIER

    def _default_snowflake_fetch(self, sql: str) -> Sequence[Mapping[str, Any]]:
        import os
        import snowflake.connector  # type: ignore[import-not-found]

        conn = snowflake.connector.connect(
            account=self.config.account,
            user=os.environ.get("SNOWFLAKE_USER", ""),
            authenticator=self.config.authenticator,
            role=self.config.role,
            warehouse=self.config.warehouse,
            database=self.config.database,
            schema=self.config.schema,
        )
        try:
            with conn.cursor(snowflake.connector.DictCursor) as cur:
                cur.execute(sql)
                return list(cur.fetchall())
        finally:
            conn.close()

    def load_records(self, include_labels: bool = True) -> List[PatientRecord]:
        sql = self.build_select_query()
        fetcher = self.row_fetcher or self._default_snowflake_fetch
        raw_rows = fetcher(sql)
        return [
            _row_to_patient_record(r, idx, self.feature_columns, self.model_adapter, include_labels)
            for idx, r in enumerate(raw_rows)
        ]


def load_records_from_csv(
    csv_path: pathlib.Path | str,
    feature_columns: Sequence[str],
    include_labels: bool = True,
    problem_id: str = "readmission_30d",
    allowed_root: Optional[pathlib.Path] = None,
) -> List[PatientRecord]:
    """Convenience wrapper for loading a CSV dataset into `PatientRecord` objects."""
    adapter = CSVDatasetAdapter(
        csv_path=csv_path,
        feature_columns=feature_columns,
        model_adapter=SimulatedVertexModelAdapter(problem_id=problem_id),
        allowed_root=allowed_root,
    )
    return adapter.load_records(include_labels=include_labels)
