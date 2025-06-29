PY ?= python
PORT ?= 8000

.PHONY: install test pipeline serve s3 airflow-test lint
install:
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
test:
	$(PY) -m pytest
pipeline:  ## generate -> features -> train -> evaluate -> simulate -> register
	$(PY) -m menumatch.cli all
serve:
	$(PY) -m uvicorn menumatch.serving.app:app --port $(PORT)
s3:  ## local S3 mock (then export MENUMATCH_STORE=s3 MENUMATCH_S3_ENDPOINT=http://localhost:5000)
	moto_server -p 5000
airflow-test:  ## run the DAG once without a scheduler (needs apache-airflow installed)
	AIRFLOW_HOME=$(CURDIR)/airflow_home AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags AIRFLOW__CORE__LOAD_EXAMPLES=False \
	MENUMATCH_PYTHON=$(shell which $(PY)) MENUMATCH_HOME=$(CURDIR) airflow dags test menumatch_training
