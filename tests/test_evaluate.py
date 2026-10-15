"""
test_evaluate.py: automatic checks for the evaluation code (src/evaluate.py).

Where this fits in the pipeline:
    data -> model -> train -> [evaluate] -> predict
The metric tests use tiny examples small enough to check by hand, so you can
see exactly why each number is what it is.

Run them with:
    pytest
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import config, data, evaluate, train
from src.make_fake_data import create_fake_dataset
from src.utils import set_seed

# 1 = infected, 0 = healthy. Seven cells, checked by hand at threshold 0.5.
# The example is deliberately lopsided so that every metric has a DIFFERENT
# value; otherwise mixing up e.g. recall and specificity could go unnoticed.
#   cell:         A    B    C    D    E    F    G
#   truth:        1    1    1    1    0    0    0
#   probability: 0.9  0.6  0.2  0.8  0.1  0.7  0.55
#   prediction:   1    1    0    1    0    1    1
#   outcome:      TP   TP   FN   TP   TN   FP   FP
LABELS = np.array([1, 1, 1, 1, 0, 0, 0])
PROBABILITIES = np.array([0.9, 0.6, 0.2, 0.8, 0.1, 0.7, 0.55])


def test_confusion_counts_match_the_hand_worked_example() -> None:
    metrics = evaluate.compute_metrics(LABELS, PROBABILITIES, threshold=0.5)
    assert metrics["true_positives"] == 3   # A, B, D
    assert metrics["false_negatives"] == 1  # C: infected but missed
    assert metrics["true_negatives"] == 1   # E
    assert metrics["false_positives"] == 2  # F, G: healthy but flagged


def test_metric_formulas_match_the_hand_worked_example() -> None:
    metrics = evaluate.compute_metrics(LABELS, PROBABILITIES, threshold=0.5)
    assert metrics["recall"] == pytest.approx(3 / 4)       # caught 3 of 4 infected
    assert metrics["specificity"] == pytest.approx(1 / 3)  # cleared 1 of 3 healthy
    assert metrics["precision"] == pytest.approx(3 / 5)    # 3 of 5 "infected" calls were right
    assert metrics["accuracy"] == pytest.approx(4 / 7)     # 4 of 7 cells right
    assert metrics["f1"] == pytest.approx(2 / 3)           # 2 x 0.6 x 0.75 / (0.6 + 0.75)


def test_probability_equal_to_threshold_counts_as_infected() -> None:
    predictions = evaluate.predict_infected(np.array([0.49, 0.5, 0.51]), threshold=0.5)
    assert predictions.tolist() == [0, 1, 1]


def test_lowering_the_threshold_never_lowers_recall() -> None:
    recalls = [evaluate.compute_metrics(LABELS, PROBABILITIES, t)["recall"] for t in (0.85, 0.5, 0.15)]
    assert recalls == sorted(recalls)  # 1/4, 3/4, 4/4


def test_chosen_threshold_is_the_highest_that_reaches_the_target() -> None:
    # Infected probabilities, highest first: 0.9, 0.8, 0.6, 0.2.
    # Target 0.7: 2 of 4 (0.5) isn't enough, 3 of 4 (0.75) is -> threshold 0.6.
    # Target 1.0: all 4 must be caught -> threshold 0.2.
    assert evaluate.choose_threshold_for_recall(LABELS, PROBABILITIES, target_recall=0.7) == 0.6
    assert evaluate.choose_threshold_for_recall(LABELS, PROBABILITIES, target_recall=1.0) == 0.2


def test_chosen_threshold_really_reaches_the_target() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    labels = rng.integers(0, 2, size=500)
    probabilities = rng.uniform(size=500)
    threshold = evaluate.choose_threshold_for_recall(labels, probabilities, config.TARGET_RECALL)
    recall = evaluate.compute_metrics(labels, probabilities, threshold)["recall"]
    assert recall >= config.TARGET_RECALL


def test_train_then_evaluate_end_to_end(tmp_path: Path) -> None:
    """Train for one epoch on fake data, then evaluate: every output file must appear."""
    set_seed(config.RANDOM_SEED)
    data_dir = tmp_path / "fake_data"
    output_dir = tmp_path / "outputs"
    create_fake_dataset(data_dir)
    split_table = data.split_by_patient(data.find_images(data_dir))
    cpu = torch.device("cpu")

    train.train_model("simple_cnn", split_table, data_dir, output_dir, num_epochs=1, device=cpu)
    metrics = evaluate.evaluate_model(output_dir / "simple_cnn_best.pt", split_table, data_dir, output_dir, cpu)

    assert metrics["validation_at_tuned_threshold"]["recall"] >= config.TARGET_RECALL
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    for suffix in ("metrics.json", "confusion_matrices.png", "roc_curve.png", "misclassified.png"):
        assert (output_dir / f"simple_cnn_{suffix}").exists(), suffix
    saved = json.loads((output_dir / "simple_cnn_metrics.json").read_text())
    assert saved["tuned_threshold"] == pytest.approx(metrics["tuned_threshold"])
