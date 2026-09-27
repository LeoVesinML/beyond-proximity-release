import json
from pathlib import Path

import pytest

from scripts.adapt_conceptgraphs_output import adapt_prediction as adapt_conceptgraphs
from scripts.adapt_langsplat_output import adapt_prediction as adapt_langsplat
from scripts.baseline_adapter_common import BaselineAdapterError, write_adapter_run


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _benchmark(path: Path) -> None:
    _write(path, {"queries": [{
        "query_id": "q001",
        "scene_id": "ConferenceHall",
        "query": "Find a chair",
        "query_type": "simple_object",
    }]})


def _native(path: Path, baseline: str, prediction: dict) -> None:
    artifact = path.parent / "native_artifact.bin"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"native fixture")
    _write(path, {
        "provenance": {
            "native_execution": True,
            "baseline": baseline,
            "scene_id": "ConferenceHall-capture-pilot",
            "repository": "https://example.invalid/baseline",
            "revision": "fixture-revision",
            "command": "fixture native command",
            "checkpoints": ["fixture-checkpoint"],
            "native_artifacts": [artifact.name],
        },
        "predictions": [prediction],
    })


@pytest.mark.parametrize(
    ("method", "adapter", "prediction"),
    [
        ("conceptgraphs", adapt_conceptgraphs, {
            "query_id": "q001",
            "found": True,
            "matched_object": "chair",
            "selected_object_id": "object_1",
            "selected_view_ids": ["v001"],
            "trace_node_ids": ["object_1"],
            "confidence": 0.8,
            "explanation": "Native object match.",
            "metrics": {"runtime_seconds": 0.2, "model_call_count": 0},
        }),
        ("langsplat", adapt_langsplat, {
            "query_id": "q001",
            "found": True,
            "matched_label": "chair",
            "selected_view_id": "v001",
            "trace": ["language_field"],
            "relevancy_score": 0.7,
            "explanation": "Native relevancy match.",
            "metrics": {"runtime_seconds": 0.1, "model_call_count": 0},
        }),
    ],
)
def test_baseline_adapter_writes_canonical_outputs(tmp_path, method, adapter, prediction):
    benchmark = tmp_path / "benchmark.json"
    native = tmp_path / "native.json"
    output = tmp_path / "output"
    _benchmark(benchmark)
    _native(native, method, prediction)

    write_adapter_run(
        native_path=native,
        benchmark_path=benchmark,
        output_dir=output,
        run_id=f"{method}_fixture",
        method=method,
        mode="live",
        scene_id="ConferenceHall-capture-pilot",
        benchmark_scene_id="ConferenceHall",
        adapt_prediction=adapter,
    )

    result = json.loads((output / "query_results" / "q001.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == "semanticsplat.query_result.v1"
    assert result["method"] == method
    assert result["mode"] == "live"
    assert result["result"]["found"] is True
    assert (output / "run_config.json").is_file()
    assert (output / "logs.json").is_file()
    assert (output / "baseline_summary.md").is_file()


def test_baseline_adapter_refuses_missing_native_evidence_without_output(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    _benchmark(benchmark)
    output = tmp_path / "output"

    with pytest.raises(BaselineAdapterError, match="Missing native input"):
        write_adapter_run(
            native_path=tmp_path / "missing.json",
            benchmark_path=benchmark,
            output_dir=output,
            run_id="blocked",
            method="conceptgraphs",
            mode="live",
            scene_id="ConferenceHall-capture-pilot",
            benchmark_scene_id="ConferenceHall",
            adapt_prediction=adapt_conceptgraphs,
        )

    assert not output.exists()
