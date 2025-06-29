"""Airflow DAG: generate -> features -> train -> evaluate (+ simulate) -> register.

Each task shells out to ``python -m menumatch.cli <stage>`` so the Airflow
environment only needs Airflow itself; the project interpreter is chosen with
``MENUMATCH_PYTHON`` (defaults to ``python``). Stage outputs are exchanged via
files under ``$MENUMATCH_DATA_DIR/runs/<run_id>`` and models are registered to the
artifact store (S3 mock / local) configured through the usual env vars.
"""
from __future__ import annotations

import os
from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PYTHON = os.environ.get("MENUMATCH_PYTHON", "python")
# {{ run_id }} is sanitised by the CLI (':' and '+' are not path safe)
RUN_ID = "{{ run_id }}"


def stage(name: str) -> BashOperator:
    return BashOperator(
        task_id=name,
        bash_command=f'{PYTHON} -m menumatch.cli {name} --run-id "{RUN_ID}"',
        cwd=os.environ.get("MENUMATCH_HOME") or None,
    )


with DAG(
    dag_id="menumatch_training",
    description="Daily retrain of the MenuMatch add-on recommender",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["menumatch", "recsys"],
) as dag:
    generate = stage("generate")
    features = stage("features")
    train = stage("train")
    evaluate = stage("evaluate")
    simulate = stage("simulate")
    register = stage("register")

    generate >> features >> train >> evaluate >> register
    train >> simulate >> register
