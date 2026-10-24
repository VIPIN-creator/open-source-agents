# Agent Eval Lite

A small CLI for evaluating local Ollama models against a repeatable JSON test set.

It runs the same cases against one or more models, applies deterministic checks, and reports pass rate, latency, output token counts, and generation speed.

## What it checks

- Expected substrings (case-insensitive)
- Valid JSON responses
- Required keys in JSON objects
- Per-case latency and Ollama token statistics when provided
- Side-by-side summaries for multiple models
- JSON report output for later comparison or CI

This is a lightweight smoke-test harness, not a semantic judge. A substring match does not prove an answer is factually correct.

## Requirements

- Python 3.10+
- Ollama installed and running
- At least one model downloaded locally

Runtime uses only the Python standard library. `pytest` is used for development tests.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2
```

## Run

From this directory:

```bash
python agent_eval_lite.py
```

Run a different model:

```bash
python agent_eval_lite.py --models qwen3:8b
```

Compare models against the same cases:

```bash
python agent_eval_lite.py --models llama3.2,qwen3:8b --output comparison.json
```

Use a custom dataset or server:

```bash
python agent_eval_lite.py \
  --dataset examples/cases.json \
  --host http://127.0.0.1:11434 \
  --timeout 120 \
  --temperature 0 \
  --output eval-report.json
```

Suppress model responses in the saved report:

```bash
python agent_eval_lite.py --no-save-responses
```

Exit code is `0` when all cases pass, `1` when any case fails or errors, and `2` for invalid configuration/dataset or report-writing errors.

## Dataset format

```json
[
  {
    "id": "capital-france",
    "prompt": "What is the capital of France?",
    "expected_substrings": ["Paris"],
    "tags": ["factual"]
  },
  {
    "id": "json-status",
    "prompt": "Return only JSON with key status and value ok.",
    "expected_json": true,
    "expected_json_keys": ["status"]
  }
]
```

Supported fields:
- `id` (required, unique string)
- `prompt` (required, non-empty string)
- `expected_substrings` (optional list of strings; every item must appear)
- `expected_json` (optional boolean)
- `expected_json_keys` (optional list of keys required in a JSON object)
- `tags` (optional descriptive metadata; currently not used for scoring)

## Tests

```bash
python -m pytest -v
```

Unit tests mock Ollama responses and do not require a running server or downloaded model.

## Security and privacy

- Prompts are sent to the configured Ollama endpoint. With the default host, requests stay on the local machine.
- Reports include generated responses by default. Review them before sharing; use `--no-save-responses` to omit response text from the saved report.
- Do not use private or production prompts without authorization.
- For remote hosts, confirm the endpoint is trusted before sending prompts.

## Limitations

- The checks are deterministic and intentionally simple; they do not assess semantic equivalence, hallucination rate, or safety comprehensively.
- Generation can vary even at temperature zero, depending on model/runtime.
- Latency includes model loading and machine load and is not a controlled benchmark.
- Token/s is calculated from Ollama's `eval_count` and `eval_duration` when present.
- Models must already be installed; this tool does not pull or modify models.

## License

This tool is intended to be contributed under the parent repository's license. Check the parent repository's current `LICENSE` and contribution instructions before submitting.
