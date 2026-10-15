"""
evaluate.py: honest scores for a trained model, measured on the untouched test set.

Where this fits in the pipeline:
    data -> model -> train -> [evaluate] -> predict

What happens, in the order of this file:
    1. load_checkpoint()              rebuild the trained model from its saved weights
    2. predict_probabilities()        the model's "probability infected" for every cell
    3. compute_metrics()              accuracy, precision, recall, specificity, F1 at a threshold
    4. choose_threshold_for_recall()  on VALIDATION data only: the highest threshold that
                                      still catches TARGET_RECALL of the infected cells
    5. plots                          confusion matrices, ROC curve, misclassified cells
    6. evaluate_model()               all of the above, plus metrics.json

The golden rule: the test set is only ever MEASURED. Every choice (the best
epoch, the threshold) is made on validation data. If we chose anything by
looking at test results, the test score would no longer be an honest estimate.

How to run it:
    python -m src.train --smoke-test && python -m src.evaluate --smoke-test   # laptop
    python -m src.evaluate --model simple_cnn --data-dir /kaggle/input         # Kaggle
"""

import argparse
import json
from pathlib import Path

import matplotlib

# Draw plots straight into image files (there's no screen on Kaggle).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data import default_splits_csv_for, get_dataloaders, load_or_create_splits
from src.models import MODEL_NAMES, build_model
from src.utils import get_device, set_seed


# ===========================================================================
# 1. Load the trained model
# ===========================================================================

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """
    Rebuild the model saved by src/train.py and put it in evaluation mode.
    Returns (model, checkpoint), where checkpoint also holds the epoch, image size etc.
    """
    # weights_only=True: only load numbers and plain values, never code.
    # A safe habit for any file that could come from somewhere else.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # pretrained=False: no need to download ImageNet weights, because our own
    # trained weights replace every one of them on the next line.
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # dropout off, BatchNorm uses its learned statistics
    return model, checkpoint


# ===========================================================================
# 2. Predict
# ===========================================================================

def predict_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over every image in the loader.
    Returns (probabilities, labels): one entry per image, in the loader's order.
    """
    model.eval()
    probability_batches = []
    label_batches = []

    with torch.no_grad():  # only measuring, not learning
        for images, labels in tqdm(loader, desc="  predicting", leave=False):
            logits = model(images.to(device))      # (batch,)
            probabilities = torch.sigmoid(logits)  # (batch,) each between 0 and 1
            probability_batches.append(probabilities.cpu().numpy())
            label_batches.append(labels.numpy())

    probabilities = np.concatenate(probability_batches)          # (num_images,)
    labels = np.concatenate(label_batches).astype(int)           # (num_images,) 0 or 1
    return probabilities, labels


def predict_infected(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """The decision rule: probability at or above the threshold -> infected (1), else healthy (0)."""
    return (probabilities >= threshold).astype(int)


# ===========================================================================
# 3. Metrics
# ===========================================================================

def safe_divide(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of crashing when there is nothing to divide by."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """
    Compare predictions at `threshold` with the true labels.

    The four possible outcomes for each cell (the "confusion matrix"):
        true positive   infected, and we said infected          (caught)
        false negative  infected, but we said healthy           (MISSED: the dangerous mistake)
        true negative   healthy, and we said healthy            (correctly cleared)
        false positive  healthy, but we said infected           (false alarm: a human re-checks)
    """
    predictions = predict_infected(probabilities, threshold)
    is_infected = labels == config.LABEL_PARASITIZED
    is_predicted_infected = predictions == config.LABEL_PARASITIZED

    true_positives = int(np.sum(is_infected & is_predicted_infected))
    false_negatives = int(np.sum(is_infected & ~is_predicted_infected))
    true_negatives = int(np.sum(~is_infected & ~is_predicted_infected))
    false_positives = int(np.sum(~is_infected & is_predicted_infected))

    # Recall (sensitivity): of all infected cells, the share we caught.
    recall = safe_divide(true_positives, true_positives + false_negatives)
    # Specificity: of all healthy cells, the share we correctly called healthy.
    specificity = safe_divide(true_negatives, true_negatives + false_positives)
    # Precision: of all cells we CALLED infected, the share that really were.
    precision = safe_divide(true_positives, true_positives + false_positives)
    # F1: one number balancing precision and recall (their "harmonic mean").
    f1 = safe_divide(2 * precision * recall, precision + recall)
    # Accuracy: of all cells, the share we got right.
    accuracy = safe_divide(true_positives + true_negatives, len(labels))

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
    }


# ===========================================================================
# 4. Tune the threshold (on validation data only!)
# ===========================================================================

def choose_threshold_for_recall(labels: np.ndarray, probabilities: np.ndarray, target_recall: float) -> float:
    """
    Find the HIGHEST threshold that still catches at least `target_recall` of the infected cells.

    Lowering the threshold catches more infected cells (higher recall) but also
    raises more false alarms. So we want the highest threshold that reaches the
    target: enough recall, with as few false alarms as possible.

    Only these values can change the recall: the probabilities given to the
    infected cells. So we try them as thresholds, from highest to lowest, and
    stop at the first one that reaches the target.
    """
    infected_probabilities = probabilities[labels == config.LABEL_PARASITIZED]
    candidate_thresholds = np.sort(np.unique(infected_probabilities))[::-1]  # highest first

    for threshold in candidate_thresholds:
        recall = compute_metrics(labels, probabilities, threshold)["recall"]
        if recall >= target_recall:
            return float(threshold)

    # The lowest candidate catches every infected cell (recall 1.0), so the loop
    # always returns before getting here. This line only satisfies Python.
    return float(candidate_thresholds[-1])


# ===========================================================================
# 5. Plots
# ===========================================================================

def plot_confusion_matrices(metrics_by_title: dict[str, dict], path: Path) -> None:
    """One confusion matrix per threshold, side by side. Rows = truth, columns = prediction."""
    figure, axes = plt.subplots(1, len(metrics_by_title), figsize=(5.5 * len(metrics_by_title), 4.5))
    class_names = config.CLASS_NAMES  # ["Uninfected", "Parasitized"], in label order 0, 1

    for axis, (title, metrics) in zip(np.atleast_1d(axes), metrics_by_title.items()):
        matrix = np.array([
            [metrics["true_negatives"], metrics["false_positives"]],   # truly Uninfected
            [metrics["false_negatives"], metrics["true_positives"]],   # truly Parasitized
        ])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                # Dark squares get white text, light squares black text.
                text_colour = "white" if matrix[row, column] > matrix.max() / 2 else "black"
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center",
                          fontsize=14, color=text_colour)
        axis.set_xticks([0, 1], labels=[f"predicted\n{name}" for name in class_names])
        axis.set_yticks([0, 1], labels=[f"truly\n{name}" for name in class_names])
        axis.set_title(title)

    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def plot_roc_curve(
    labels: np.ndarray, probabilities: np.ndarray, metrics_by_title: dict[str, dict], roc_auc: float, path: Path
) -> None:
    """
    The ROC curve: recall (y) against false-alarm rate (x) for EVERY possible threshold.
    A perfect model hugs the top-left corner; random guessing is the diagonal.
    The dots mark the thresholds we actually report.
    """
    false_alarm_rates, recalls, _ = roc_curve(labels, probabilities)

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot(false_alarm_rates, recalls, label=f"model (AUC = {roc_auc:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random guessing (AUC = 0.5)")
    for title, metrics in metrics_by_title.items():
        false_alarm_rate = 1 - metrics["specificity"]
        axis.scatter(false_alarm_rate, metrics["recall"], s=60, zorder=3, label=title)
    axis.set_xlabel("false-alarm rate (1 - specificity)")
    axis.set_ylabel("recall (sensitivity)")
    axis.set_title("ROC curve, test set")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def plot_misclassified(
    test_rows: pd.DataFrame,
    data_dir: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    path: Path,
    max_per_row: int = 8,
) -> None:
    """
    Show the test cells the model got wrong at `threshold`, with their probability of infection.
    Top row: missed infections (most confidently wrong first). Bottom row: false alarms.
    """
    predictions = predict_infected(probabilities, threshold)
    is_infected = labels == config.LABEL_PARASITIZED

    missed_indices = np.where(is_infected & (predictions == 0))[0]
    false_alarm_indices = np.where(~is_infected & (predictions == 1))[0]
    # Most confidently wrong first: missed infections with the LOWEST probability,
    # false alarms with the HIGHEST.
    missed_indices = missed_indices[np.argsort(probabilities[missed_indices])]
    false_alarm_indices = false_alarm_indices[np.argsort(-probabilities[false_alarm_indices])]

    rows_to_show = [
        (f"Missed infections ({len(missed_indices)} in total)", missed_indices[:max_per_row]),
        (f"False alarms ({len(false_alarm_indices)} in total)", false_alarm_indices[:max_per_row]),
    ]

    # Two stacked "sub-figures", one per kind of mistake, each with its title on top.
    figure = plt.figure(figsize=(2 * max_per_row, 6), layout="constrained")
    figure.suptitle(f"Misclassified test cells at threshold {threshold:.3f} "
                    "(p = predicted probability of infection)", fontsize=13)
    row_figures = figure.subfigures(2, 1)

    for row_figure, (row_title, indices) in zip(row_figures, rows_to_show):
        row_figure.suptitle(row_title, fontweight="bold")
        axes = row_figure.subplots(1, max_per_row)
        for column, axis in enumerate(axes):
            axis.axis("off")
            if column < len(indices):
                index = indices[column]
                with Image.open(data_dir / test_rows["path"].iloc[index]) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_title(f"p = {probabilities[index]:.3f}", fontsize=10)
        if len(indices) == 0:
            axes[0].text(0, 0.5, "none", va="center", fontsize=12)

    figure.savefig(path, dpi=100)
    plt.close(figure)


# ===========================================================================
# 6. Put it all together
# ===========================================================================

def print_comparison(metrics_by_title: dict[str, dict]) -> None:
    """Print the metrics for each threshold as side-by-side columns."""
    rows = [
        ("accuracy", "accuracy", "{:.4f}"),
        ("precision", "precision", "{:.4f}"),
        ("recall (sensitivity)", "recall", "{:.4f}"),
        ("specificity", "specificity", "{:.4f}"),
        ("F1", "f1", "{:.4f}"),
        ("missed infections (FN)", "false_negatives", "{:,}"),
        ("false alarms (FP)", "false_positives", "{:,}"),
    ]
    titles = list(metrics_by_title)
    column_width = 30  # wide enough for a title like "threshold 0.500 (default)" plus a gap
    print(f"\n{'':<24}" + "".join(f"{title:>{column_width}}" for title in titles))
    for row_name, key, number_format in rows:
        values = "".join(f"{number_format.format(metrics_by_title[title][key]):>{column_width}}" for title in titles)
        print(f"{row_name:<24}{values}")
    print()


def evaluate_model(
    checkpoint_path: Path,
    split_table: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Evaluate one trained model, save all plots and metrics.json, and return the metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model_name = checkpoint["model_name"]
    print(f"Evaluating {model_name} from epoch {checkpoint['epoch']} "
          f"(validation loss {checkpoint['val_loss']:.4f})")

    dataloaders = get_dataloaders(split_table, data_dir, checkpoint["image_size"])
    val_probabilities, val_labels = predict_probabilities(model, dataloaders["val"], device)
    test_probabilities, test_labels = predict_probabilities(model, dataloaders["test"], device)

    # Step 1: choose the threshold using VALIDATION data only.
    tuned_threshold = choose_threshold_for_recall(val_labels, val_probabilities, config.TARGET_RECALL)
    val_at_tuned = compute_metrics(val_labels, val_probabilities, tuned_threshold)
    print(f"Tuned threshold (chosen on validation): {tuned_threshold:.4f} "
          f"-> validation recall {val_at_tuned['recall']:.4f} (target {config.TARGET_RECALL})")

    # Step 2: measure the TEST set at both thresholds. Nothing is chosen here.
    default_title = f"threshold {config.DEFAULT_THRESHOLD:.3f} (default)"
    tuned_title = f"threshold {tuned_threshold:.3f} (tuned)"
    test_metrics_by_title = {
        default_title: compute_metrics(test_labels, test_probabilities, config.DEFAULT_THRESHOLD),
        tuned_title: compute_metrics(test_labels, test_probabilities, tuned_threshold),
    }
    # ROC-AUC: the chance that a random infected cell gets a higher probability
    # than a random healthy cell. 1.0 = perfect ranking, 0.5 = coin flip.
    # It looks at all thresholds at once, so it doesn't depend on our choice.
    test_roc_auc = float(roc_auc_score(test_labels, test_probabilities))

    print(f"TEST SET ({len(test_labels):,} cells), measured once:")
    print_comparison(test_metrics_by_title)
    print(f"ROC-AUC: {test_roc_auc:.4f}")

    # Save everything.
    test_rows = split_table[split_table["split"] == "test"].reset_index(drop=True)  # same order as the loader
    plot_confusion_matrices(test_metrics_by_title, output_dir / f"{model_name}_confusion_matrices.png")
    plot_roc_curve(test_labels, test_probabilities, test_metrics_by_title, test_roc_auc,
                   output_dir / f"{model_name}_roc_curve.png")
    plot_misclassified(test_rows, data_dir, test_labels, test_probabilities, tuned_threshold,
                       output_dir / f"{model_name}_misclassified.png")

    metrics = {
        "model_name": model_name,
        "checkpoint_epoch": checkpoint["epoch"],
        "target_recall": config.TARGET_RECALL,
        "tuned_threshold": tuned_threshold,
        "validation_at_tuned_threshold": val_at_tuned,
        "test_at_default_threshold": test_metrics_by_title[default_title],
        "test_at_tuned_threshold": test_metrics_by_title[tuned_title],
        "test_roc_auc": test_roc_auc,
        "num_test_images": int(len(test_labels)),
    }
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"Saved metrics and plots to {output_dir}")
    return metrics


# ===========================================================================
# Command line
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained malaria cell classifier.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="simple_cnn")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Folder with the cell images. Not needed with --smoke-test.")
    parser.add_argument("--splits-csv", type=Path, default=None,
                        help="The saved split used for training (default: chosen from --data-dir)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained model file (default: <output dir>/<model>_best.pt)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Evaluate the smoke-test model on the fake data.")
    args = parser.parse_args()

    if args.smoke_test:
        data_dir = args.data_dir or config.FAKE_DATA_DIR
        output_dir = config.OUTPUT_DIR / "smoke_test"
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required (or use --smoke-test).")
        data_dir = args.data_dir
        output_dir = config.OUTPUT_DIR

    checkpoint_path = args.checkpoint or output_dir / f"{args.model}_best.pt"
    if not checkpoint_path.exists():
        smoke_flag = " --smoke-test" if args.smoke_test else ""
        raise FileNotFoundError(f"No trained model at {checkpoint_path}. Train one first: "
                                f"python -m src.train --model {args.model}{smoke_flag}")

    # Evaluation must use the SAME split as training, so it must already exist.
    splits_csv = args.splits_csv or default_splits_csv_for(data_dir)
    if not splits_csv.exists():
        raise FileNotFoundError(f"No saved split at {splits_csv}. Evaluation must reuse the "
                                "split the model was trained with.")

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")
    split_table = load_or_create_splits(data_dir, splits_csv)
    evaluate_model(checkpoint_path, split_table, data_dir, output_dir, device)


if __name__ == "__main__":
    main()
