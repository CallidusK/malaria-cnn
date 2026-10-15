"""
test_compare.py: automatic checks for the model comparison (src/compare.py).

Where this fits in the pipeline:
    data -> model -> train -> [evaluate: compare the models] -> predict
The result files written here are made up, inside a temporary folder, only to
check that the table is built correctly. They never go near results/.

Run them with:
    pytest
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import compare, evaluate
from src.models import SimpleCNN, count_parameters


def write_made_up_results(results_dir: Path, model_name: str) -> None:
    """Write a small, made-up metrics + history pair for one model."""
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.3, 0.2, 0.6])
    metrics = {
        "checkpoint_epoch": 2,
        "num_test_images": 4,
        "test_at_default_threshold": evaluate.compute_metrics(labels, probabilities, 0.5),
        "test_at_tuned_threshold": evaluate.compute_metrics(labels, probabilities, 0.25),
        "test_roc_auc": 0.75,
    }
    history = [{"seconds": 60.0}, {"seconds": 60.0}, {"seconds": 60.0}]
    (results_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics))
    (results_dir / f"{model_name}_history.json").write_text(json.dumps(history))


def test_comparison_table_uses_the_result_files(tmp_path: Path) -> None:
    write_made_up_results(tmp_path, "simple_cnn")
    table = compare.write_comparison(tmp_path).read_text()

    assert "SimpleCNN" in table
    assert "ResNet18" not in table                         # no ResNet18 files -> no ResNet18 column
    assert f"{count_parameters(SimpleCNN()):,}" in table   # parameter count is computed, not typed in
    assert "3.0 min" in table                              # 3 epochs x 60 seconds
    assert "not measured" in table                         # no speed_benchmark.json yet
    assert "50.00%" in table                               # recall at 0.5: caught 1 of 2 infected


def test_comparison_without_any_results_raises_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare.write_comparison(tmp_path)


def test_speed_benchmark_returns_a_positive_speed() -> None:
    images_per_second = compare.benchmark_inference_speed(
        "simple_cnn", torch.device("cpu"), batch_size=2, warmup_batches=1, timed_batches=2
    )
    assert images_per_second > 0


def test_models_tested_on_different_test_sets_are_not_compared(tmp_path: Path) -> None:
    write_made_up_results(tmp_path, "simple_cnn")
    write_made_up_results(tmp_path, "resnet18")
    # Pretend ResNet18 was tested on a different number of cells.
    resnet_metrics_path = tmp_path / "resnet18_metrics.json"
    resnet_metrics = json.loads(resnet_metrics_path.read_text())
    resnet_metrics["num_test_images"] = 11
    resnet_metrics_path.write_text(json.dumps(resnet_metrics))

    with pytest.raises(ValueError, match="different test sets"):
        compare.write_comparison(tmp_path)
