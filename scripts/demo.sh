#!/usr/bin/env bash
# ~60-second demo for screen recording: run the Airflow DAG against a local S3 mock,
# serve the promoted model, and call /recommend.  Needs apache-airflow (see README).
# Usage: scripts/demo.sh   (from the repo root, venv active; Airflow in $AIRFLOW_BIN's env)
set -euo pipefail
AIRFLOW_BIN=${AIRFLOW_BIN:-airflow}
export MENUMATCH_STORE=s3 MENUMATCH_S3_ENDPOINT=http://localhost:5000
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1   # dummy moto creds
export AIRFLOW_HOME=${AIRFLOW_HOME:-$PWD/airflow_home} AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags AIRFLOW__CORE__LOAD_EXAMPLES=False
export MENUMATCH_PYTHON=$(command -v python) MENUMATCH_HOME=$PWD
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

moto_server -p 5000 >/dev/null 2>&1 & sleep 3
$AIRFLOW_BIN db migrate >/dev/null 2>&1
$AIRFLOW_BIN dags test menumatch_training 2>&1 | grep -E "DAG TEST|menumatch registered|DagRun Finished"
uvicorn menumatch.serving.app:app --port 8000 --log-level warning & sleep 6
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"restaurant_id":0,"cart_item_ids":[0],"k":3,"hour":19}' | python -m json.tool
