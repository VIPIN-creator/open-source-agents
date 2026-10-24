import json
import sys
from pathlib import Path

import pytest

# Ensure module is found when running pytest from any directory or CLI entrypoint
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_eval_lite as ev


def test_normalize_host():
    assert ev.normalize_host("localhost:11434") == "http://localhost:11434"
    assert ev.normalize_host("http://localhost:11434/") == "http://localhost:11434"


def test_normalize_host_rejects_empty():
    with pytest.raises(ValueError):
        ev.normalize_host(" ")


def test_dataset_loads_example():
    example_path = Path(__file__).resolve().parent.parent / "examples" / "cases.json"
    cases = ev.load_dataset(example_path)
    assert len(cases) == 3
    assert cases[0]["id"] == "capital-france"


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "same", "prompt": "one"},
        {"id": "same", "prompt": "two"},
    ]))
    with pytest.raises(ValueError, match="Duplicate"):
        ev.load_dataset(path)


def test_dataset_rejects_empty_prompt(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "a", "prompt": " "}]))
    with pytest.raises(ValueError, match="prompt"):
        ev.load_dataset(path)


def test_substring_check_is_case_insensitive():
    result = ev.check_response(
        {"expected_substrings": ["paris", "FRANCE"]},
        "PARIS is the capital of France.",
    )
    assert result["passed"] is True
    assert result["failures"] == []


def test_missing_expected_substring_fails():
    result = ev.check_response(
        {"expected_substrings": ["Paris"]}, "London"
    )
    assert result["passed"] is False
    assert "Paris" in result["failures"][0]


def test_json_validation_and_required_keys():
    case = {"expected_json": True, "expected_json_keys": ["status"]}
    assert ev.check_response(case, '{"status":"ok"}')["passed"] is True
    assert ev.check_response(case, "not json")["passed"] is False
    assert ev.check_response(case, '{"other":1}')["passed"] is False


def test_evaluate_case_uses_mocked_ollama(monkeypatch):
    def fake_post(url, payload, timeout):
        assert url == "http://localhost:11434/api/generate"
        assert payload["model"] == "fake-model"
        return {
            "response": "Paris",
            "eval_count": 5,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 10,
        }

    monkeypatch.setattr(ev, "post_json", fake_post)
    result = ev.evaluate_case(
        {"id": "capital", "prompt": "capital?", "expected_substrings": ["Paris"]},
        "fake-model",
        "http://localhost:11434",
        10,
        0.0,
    )
    assert result["status"] == "PASS"
    assert result["tokens_per_second"] == 5.0


def test_evaluate_case_captures_api_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(ev, "post_json", fail)
    result = ev.evaluate_case(
        {"id": "x", "prompt": "hi"},
        "model",
        "http://localhost:11434",
        1,
        0.0,
    )
    assert result["status"] == "ERROR"
    assert result["passed"] is False


def test_summary_counts():
    summary = ev.summarize([
        {"status": "PASS", "latency_seconds": 1.0, "tokens_per_second": 10.0},
        {"status": "FAIL", "latency_seconds": 3.0, "tokens_per_second": 20.0},
        {"status": "ERROR", "latency_seconds": 2.0, "tokens_per_second": None},
    ])
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["errors"] == 1
    assert summary["pass_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["median_latency_seconds"] == 2.0


def test_run_evaluation_multiple_models(monkeypatch):
    def fake_eval(case, model, host, timeout, temperature):
        return {
            "id": case["id"], "model": model, "status": "PASS",
            "latency_seconds": 1.0, "tokens_per_second": 5.0,
            "passed": True, "failures": [], "response": "ok",
        }

    monkeypatch.setattr(ev, "evaluate_case", fake_eval)
    report = ev.run_evaluation(
        [{"id": "one", "prompt": "hi"}],
        ["model-a", "model-b"],
        "localhost:11434",
    )
    assert report["models"] == ["model-a", "model-b"]
    assert report["summary_by_model"]["model-a"]["passed"] == 1
    assert len(report["results"]) == 2
