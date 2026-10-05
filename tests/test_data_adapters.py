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

"""Behavioral tests for CSV/Kaggle and Clinical Silver/Vertex API data adapters."""

from __future__ import annotations

import csv
import pathlib

from data_adapters import (
    CSVDatasetAdapter,
    SimulatedVertexModelAdapter,
    load_records_from_csv,
)


def test_csv_dataset_adapter_loads_and_scores_records(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "kaggle_sample.csv"
    fieldnames = [
        "patient_id",
        "age",
        "sex",
        "charlson_index",
        "prior_ed_visits_6m",
        "prior_ip_admissions_12m",
        "length_of_stay_days",
        "sdoh_deprivation_index",
        "acuity_score",
        "egfr",
        "hba1c",
        "nt_probnp",
        "outcome_label",
        "time_to_event_days",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "patient_id": "KAG-001",
                "age": "71",
                "sex": "1",
                "charlson_index": "4.0",
                "prior_ed_visits_6m": "3",
                "prior_ip_admissions_12m": "1",
                "length_of_stay_days": "5.5",
                "sdoh_deprivation_index": "0.75",
                "acuity_score": "6.2",
                "egfr": "38.0",
                "hba1c": "8.4",
                "nt_probnp": "4200.0",
                "outcome_label": "1",
                "time_to_event_days": "12.0",
            }
        )
        writer.writerow(
            {
                "patient_id": "KAG-002",
                "age": "49",
                "sex": "0",
                "charlson_index": "1.0",
                "prior_ed_visits_6m": "0",
                "prior_ip_admissions_12m": "0",
                "length_of_stay_days": "2.0",
                "sdoh_deprivation_index": "0.15",
                "acuity_score": "1.8",
                "egfr": "92.0",
                "hba1c": "5.4",
                "nt_probnp": "180.0",
                "outcome_label": "0",
                "time_to_event_days": "",
            }
        )

    adapter = CSVDatasetAdapter(
        csv_path=csv_path,
        feature_columns=("egfr", "hba1c", "nt_probnp"),
        model_adapter=SimulatedVertexModelAdapter(problem_id="readmission_30d"),
    )
    records = adapter.load_records(include_labels=True)
    assert len(records) == 2
    assert records[0].patient_id == "KAG-001"
    assert records[0].outcome_label == 1
    assert 0.0 < records[0].baseline_api_prob < 1.0
    assert records[0].baseline_api_prob > records[1].baseline_api_prob

    eval_records = load_records_from_csv(
        csv_path,
        feature_columns=("egfr", "hba1c", "nt_probnp"),
        include_labels=False,
    )
    assert eval_records[0].outcome_label is None
    assert eval_records[0].time_to_event_days is None


def test_snowflake_dataset_adapter_contract() -> None:
    from data_adapters import SnowflakeConfig, SnowflakeDatasetAdapter

    cfg = SnowflakeConfig(
        account="clinical-edw.us-east-1.privatelink",
        warehouse="WH_CLINICAL_ML",
        database="PROD_CLINICAL_SILVER",
        schema="CARE_MGMT",
        table_or_view="VW_READMISSION_30D_COHORT",
    )
    fake_rows = [
        {
            "PATIENT_ID": "SF-1001",
            "AGE": 73.0,
            "SEX": 1,
            "CHARLSON_INDEX": 5.0,
            "PRIOR_ED_VISITS_6M": 4,
            "PRIOR_IP_ADMISSIONS_12M": 2,
            "LENGTH_OF_STAY_DAYS": 6.0,
            "SDOH_DEPRIVATION_INDEX": 0.82,
            "ACUITY_SCORE": 6.8,
            "EGFR": 34.0,
            "HBA1C": 8.9,
            "NT_PROBNP": 5100.0,
            "OUTCOME_LABEL": 1,
            "TIME_TO_EVENT_DAYS": 11.0,
        }
    ]
    adapter = SnowflakeDatasetAdapter(
        config=cfg,
        feature_columns=("egfr", "hba1c", "nt_probnp"),
        row_fetcher=lambda _sql: fake_rows,
    )
    assert "SELECT" in adapter.build_select_query()
    assert "PROD_CLINICAL_SILVER.CARE_MGMT.VW_READMISSION_30D_COHORT" in adapter.build_select_query()

    recs = adapter.load_records(include_labels=True)
    assert len(recs) == 1
    assert recs[0].patient_id == "SF-1001"
    assert recs[0].features["egfr"] == 34.0
    assert recs[0].outcome_label == 1

    masked = adapter.load_records(include_labels=False)
    assert masked[0].outcome_label is None

