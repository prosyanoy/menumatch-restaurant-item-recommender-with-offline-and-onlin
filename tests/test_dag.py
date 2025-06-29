import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("airflow")


def _load_dag():
    path = Path(__file__).parents[1] / "dags" / "menumatch_dag.py"
    spec = importlib.util.spec_from_file_location("menumatch_dag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises on any import/definition error
    return module.dag


def test_dag_structure():
    dag = _load_dag()
    assert dag.dag_id == "menumatch_training"
    tasks = dag.task_dict
    assert set(tasks) == {"generate", "features", "train", "evaluate", "simulate", "register"}
    order = ["generate", "features", "train", "evaluate", "register"]
    for up, down in zip(order, order[1:]):
        assert down in tasks[up].downstream_task_ids
    assert "simulate" in tasks["train"].downstream_task_ids
    assert tasks["register"].upstream_task_ids == {"evaluate", "simulate"}
    assert "menumatch.cli register" in tasks["register"].bash_command
