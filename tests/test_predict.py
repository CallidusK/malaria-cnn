"""
test_predict.py: automatic checks for the prediction script (src/predict.py).

Where this fits in the pipeline:
    data -> model -> train -> evaluate -> [predict]
The most important check: predict.py must give exactly the same probability
as the evaluation code for the same cell, or its answers couldn't be trusted
to match the results we report.

Run them with:
    pytest
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src import config, data, evaluate, predict, train
from src.make_fake_data import create_fake_dataset
from src.utils import set_seed

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def trained_setup(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Fake data and a SimpleCNN trained for one epoch, shared by the tests below."""
    set_seed(config.RANDOM_SEED)
    folder = tmp_path_factory.mktemp("predict")
    data_dir = folder / "fake_data"
    create_fake_dataset(data_dir)
    split_table = data.split_by_patient(data.find_images(data_dir))
    train.train_model("simple_cnn", split_table, data_dir, folder, num_epochs=1, device=CPU)
    return {"folder": folder, "data_dir": data_dir, "split_table": split_table,
            "weights": folder / "simple_cnn_best.pt"}


def test_junk_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "cell.png").write_bytes(b"")
    (tmp_path / "cell.JPG").write_bytes(b"")
    (tmp_path / "Thumbs.db").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    found = [path.name for path in predict.find_image_files(tmp_path)]
    assert found == ["cell.JPG", "cell.png"]


def test_threshold_override_wins(tmp_path: Path) -> None:
    threshold, reason = predict.choose_threshold("simple_cnn", 0.3, tmp_path)
    assert threshold == 0.3
    assert "--threshold" in reason


def test_tuned_threshold_is_used_when_available(tmp_path: Path) -> None:
    (tmp_path / "simple_cnn_metrics.json").write_text(json.dumps({"tuned_threshold": 0.12}))
    threshold, reason = predict.choose_threshold("simple_cnn", None, tmp_path)
    assert threshold == 0.12
    assert "tuned" in reason


def test_default_threshold_is_the_fallback(tmp_path: Path) -> None:
    threshold, reason = predict.choose_threshold("simple_cnn", None, tmp_path)  # no metrics file here
    assert threshold == config.DEFAULT_THRESHOLD
    assert "default" in reason


def test_predict_matches_evaluation_exactly(trained_setup: dict) -> None:
    """The same cells through predict.py and through the evaluation code give identical probabilities."""
    data_dir = trained_setup["data_dir"]
    test_rows = trained_setup["split_table"][trained_setup["split_table"]["split"] == "test"]
    paths = [data_dir / path for path in test_rows["path"]]

    model, checkpoint = evaluate.load_checkpoint(trained_setup["weights"], CPU)
    dataset = data.MalariaDataset(test_rows, data_dir, data.build_eval_transform(checkpoint["image_size"]))
    evaluation_probabilities, _ = evaluate.predict_probabilities(model, DataLoader(dataset, batch_size=8), CPU)

    predictions, _, _ = predict.predict_files(trained_setup["weights"], paths, results_dir=trained_setup["folder"],
                                              device=CPU)
    assert np.allclose(predictions["probability_infected"].to_numpy(), evaluation_probabilities, atol=1e-6)


def test_predictions_follow_the_threshold(trained_setup: dict) -> None:
    data_dir = trained_setup["data_dir"]
    paths = predict.find_image_files(data_dir / "cell_images" / "Parasitized")[:5]
    predictions, threshold, _ = predict.predict_files(trained_setup["weights"], paths, threshold_override=0.5,
                                                      results_dir=trained_setup["folder"], device=CPU)
    assert len(predictions) == 5
    for _, row in predictions.iterrows():
        expected = "Parasitized" if row["probability_infected"] >= threshold else "Uninfected"
        assert row["prediction"] == expected
