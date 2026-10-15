"""
compare.py: puts SimpleCNN and ResNet18 side by side in a comparison table (comparison.md).

Where this fits in the pipeline:
    data -> model -> train -> [evaluate: compare the models] -> predict

Two jobs:
    1. benchmark_inference_speed()  how many cells per second each model can classify.
                                    Measured with random images: speed depends on the
                                    model's size, not on what its weights have learned,
                                    so the trained models aren't needed for this.
    2. write_comparison()           a markdown table built ONLY from real result files
                                    (<model>_metrics.json, <model>_history.json and
                                    speed_benchmark.json). Nothing is typed in by hand.

How to run it:
    python -m src.compare --benchmark --results-dir /kaggle/working/outputs   # Kaggle (GPU)
    python -m src.compare                                                      # laptop: rebuild from results/
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import config
from src.models import MODEL_NAMES, build_model, count_parameters, image_size_for
from src.utils import get_device, set_seed

# How each model is written in the table.
DISPLAY_NAMES: dict[str, str] = {"simple_cnn": "SimpleCNN", "resnet18": "ResNet18"}

SPEED_FILENAME = "speed_benchmark.json"
COMPARISON_FILENAME = "comparison.md"

# Weights are stored as 32-bit numbers: 4 bytes each.
BYTES_PER_PARAMETER = 4


# ===========================================================================
# 1. Speed benchmark
# ===========================================================================

def wait_for_device(device: torch.device) -> None:
    """
    GPUs work asynchronously: Python hands them work and moves on before it's
    finished. Waiting here makes the stopwatch measure the GPU's real work.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_inference_speed(
    model_name: str,
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
    warmup_batches: int = config.SPEED_WARMUP_BATCHES,
    timed_batches: int = config.SPEED_TIMED_BATCHES,
) -> float:
    """
    Return how many images per second the model classifies on `device`.
    This times the model alone (not loading images from disk).
    """
    model = build_model(model_name, pretrained=False).to(device)
    model.eval()
    image_size = image_size_for(model_name)
    images = torch.randn(batch_size, config.NUM_CHANNELS, image_size, image_size, device=device)  # (batch, 3, size, size)

    with torch.no_grad():
        # Warm-up: the first batches are slow while the GPU sets things up. Not timed.
        for _ in range(warmup_batches):
            model(images)
        wait_for_device(device)

        start_time = time.perf_counter()
        for _ in range(timed_batches):
            model(images)
        wait_for_device(device)
        seconds = time.perf_counter() - start_time

    return timed_batches * batch_size / seconds


def run_speed_benchmark(results_dir: Path, device: torch.device) -> Path:
    """Benchmark every model and save the results to speed_benchmark.json."""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = str(device)

    images_per_second = {}
    for model_name in MODEL_NAMES:
        images_per_second[model_name] = benchmark_inference_speed(model_name, device)
        print(f"{DISPLAY_NAMES[model_name]:>10}: {images_per_second[model_name]:,.0f} images/second on {device_name}")

    results_dir.mkdir(parents=True, exist_ok=True)
    speed_path = results_dir / SPEED_FILENAME
    with open(speed_path, "w") as speed_file:
        json.dump({"device": device_name, "batch_size": config.BATCH_SIZE,
                   "images_per_second": images_per_second}, speed_file, indent=2)
    return speed_path


# ===========================================================================
# 2. The comparison table
# ===========================================================================

def summarise_model(model_name: str, results_dir: Path) -> dict | None:
    """
    Gather one model's numbers from its result files.
    Returns None if this model hasn't been trained and evaluated yet.
    """
    metrics_path = results_dir / f"{model_name}_metrics.json"
    history_path = results_dir / f"{model_name}_history.json"
    if not (metrics_path.exists() and history_path.exists()):
        return None

    metrics = json.loads(metrics_path.read_text())
    history = json.loads(history_path.read_text())

    total_training_seconds = 0.0
    for record in history:
        total_training_seconds += record["seconds"]

    num_parameters = count_parameters(build_model(model_name, pretrained=False))
    return {
        "parameters": num_parameters,
        "size_mb": num_parameters * BYTES_PER_PARAMETER / 1_000_000,
        "image_size": image_size_for(model_name),
        "epochs_run": len(history),
        "best_epoch": metrics["checkpoint_epoch"],
        "training_minutes": total_training_seconds / 60,
        "metrics": metrics,
    }


def add_test_rows(rows: list[tuple[str, str]], results: dict) -> None:
    """Append the test-set rows for one threshold (accuracy ... false alarms)."""
    rows.append(("accuracy", f"{results['accuracy']:.2%}"))
    rows.append(("precision", f"{results['precision']:.2%}"))
    rows.append(("recall (sensitivity)", f"{results['recall']:.2%}"))
    rows.append(("specificity", f"{results['specificity']:.2%}"))
    rows.append(("F1", f"{results['f1']:.4f}"))
    rows.append(("missed infections", f"{results['false_negatives']:,}"))
    rows.append(("false alarms", f"{results['false_positives']:,}"))


def table_column(model_name: str, summary: dict, speed: dict | None) -> list[tuple[str, str]]:
    """
    One model's column of the comparison table, as (row label, text) pairs.
    Section headings have empty text.
    """
    if speed is not None and model_name in speed["images_per_second"]:
        speed_text = f"{speed['images_per_second'][model_name]:,.0f}"
    else:
        speed_text = "not measured"
    at_default = summary["metrics"]["test_at_default_threshold"]
    at_tuned = summary["metrics"]["test_at_tuned_threshold"]

    rows = []
    rows.append(("**Size**", ""))
    rows.append(("parameters", f"{summary['parameters']:,}"))
    rows.append(("size of the weights", f"{summary['size_mb']:.1f} MB"))
    rows.append(("input image size", f"{summary['image_size']}×{summary['image_size']}"))

    rows.append(("**Training**", ""))
    rows.append(("training time", f"{summary['training_minutes']:.1f} min"))
    rows.append(("epochs run (best epoch)", f"{summary['epochs_run']} ({summary['best_epoch']})"))

    rows.append(("**Speed**", ""))
    rows.append(("cells classified per second", speed_text))

    rows.append((f"**Test set, default threshold {config.DEFAULT_THRESHOLD}**", ""))
    add_test_rows(rows, at_default)

    rows.append((f"**Test set, threshold tuned for {config.TARGET_RECALL:.0%} recall on validation**", ""))
    rows.append(("threshold", f"{at_tuned['threshold']:.4f}"))
    add_test_rows(rows, at_tuned)

    rows.append(("**Threshold-free**", ""))
    rows.append(("ROC-AUC", f"{summary['metrics']['test_roc_auc']:.4f}"))
    return rows


def write_comparison(results_dir: Path) -> Path:
    """Write comparison.md into results_dir, using only the result files found there."""
    summaries = {}
    for model_name in MODEL_NAMES:
        summary = summarise_model(model_name, results_dir)
        if summary is None:
            print(f"Skipping {model_name}: no metrics/history files in {results_dir}")
        else:
            summaries[model_name] = summary
    if not summaries:
        raise FileNotFoundError(f"No model results found in {results_dir}")

    # A fair comparison needs every model tested on the SAME test cells.
    # Different test-set sizes prove they weren't (e.g. real results next to fake ones).
    test_set_sizes = {}
    for model_name, summary in summaries.items():
        test_set_sizes[model_name] = summary["metrics"]["num_test_images"]
    if len(set(test_set_sizes.values())) > 1:
        raise ValueError(f"The models were tested on different test sets {test_set_sizes}, "
                         "so comparing them would be unfair.")

    speed_path = results_dir / SPEED_FILENAME
    speed = None
    if speed_path.exists():
        speed = json.loads(speed_path.read_text())

    # One column of text per model. Every column has the same row labels, in the same order.
    columns = {}
    for model_name, summary in summaries.items():
        columns[model_name] = table_column(model_name, summary, speed)

    model_names = list(summaries)
    num_test_images = summaries[model_names[0]]["metrics"]["num_test_images"]
    lines = [
        "# Model comparison: " + " vs ".join(DISPLAY_NAMES[name] for name in model_names),
        "",
        # Deliberately no claim like "from a real Kaggle run" here: this script
        # can't know where the files came from (the tests feed it fake ones).
        "*Generated by `python -m src.compare` from the result files in this folder "
        "(`<model>_metrics.json`, `<model>_history.json`, `speed_benchmark.json`).*",
        "",
        f"- **Test set:** {num_test_images:,} cells from patients that no model ever trained on, "
        "and that were never used to make any choice.",
        "- **Thresholds** were tuned on the *validation* set only.",
        "- **Training time** is the sum of all epochs, as measured during training.",
    ]
    if speed is not None:
        lines.append(f"- **Speed** is the model alone (no image loading), measured on: {speed['device']}, "
                     f"in batches of {speed['batch_size']}.")
    lines += ["", "| | " + " | ".join(DISPLAY_NAMES[name] for name in model_names) + " |",
              "|---|" + "---|" * len(model_names)]

    first_column = columns[model_names[0]]
    for row_number, (label, _) in enumerate(first_column):
        cells = []
        for model_name in model_names:
            _, text = columns[model_name][row_number]
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    comparison_path = results_dir / COMPARISON_FILENAME
    comparison_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {comparison_path}")
    return comparison_path


# ===========================================================================
# Command line
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the trained models.")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR,
                        help="Folder with <model>_metrics.json and <model>_history.json files")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure each model's speed first (do this on the Kaggle GPU)")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    if args.benchmark:
        run_speed_benchmark(args.results_dir, get_device())
    write_comparison(args.results_dir)
    print((args.results_dir / COMPARISON_FILENAME).read_text())


if __name__ == "__main__":
    main()
