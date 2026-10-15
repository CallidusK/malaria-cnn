"""
predict.py: classifies cell images from the command line with a trained model.

Where this fits in the pipeline:
    data -> model -> train -> evaluate -> [predict]

This is the step a user of the model would run: give it one image of a single
red blood cell (or a folder of them), get back "Parasitized" or "Uninfected"
and the predicted probability of infection.

Which threshold? By default, the TUNED threshold from Phase 5
(results/<model>_metrics.json): chosen on validation data to catch at least
98% of infected cells. That's the one we'd actually use, because a missed
infection is worse than a false alarm. --threshold overrides it.

How to run it:
    python -m src.predict --weights outputs/resnet18_best.pt --image cell.png
    python -m src.predict --weights outputs/resnet18_best.pt --folder my_cells/ --output-csv predictions.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from src import config
from src.data import build_eval_transform
from src.evaluate import load_checkpoint, predict_infected
from src.utils import get_device

# The label printed for each decision (0 = healthy, 1 = infected).
LABEL_NAMES: dict[int, str] = {
    config.LABEL_UNINFECTED: "Uninfected",
    config.LABEL_PARASITIZED: "Parasitized",
}


def find_image_files(folder: Path) -> list[Path]:
    """Every image file in the folder (and its sub-folders), skipping junk like Thumbs.db."""
    image_files = []
    for path in sorted(folder.rglob("*")):  # sorted: the same order every time
        if path.is_file() and path.suffix.lower() in config.PREDICT_EXTENSIONS:
            image_files.append(path)
    return image_files


def choose_threshold(model_name: str, threshold_override: float | None, results_dir: Path) -> tuple[float, str]:
    """
    Decide which threshold to use, and say why.
    Returns (threshold, explanation).
    """
    if threshold_override is not None:
        return threshold_override, "set with --threshold"

    metrics_path = results_dir / f"{model_name}_metrics.json"
    if metrics_path.exists():
        tuned_threshold = json.loads(metrics_path.read_text())["tuned_threshold"]
        return tuned_threshold, f"tuned for {config.TARGET_RECALL:.0%} recall on validation ({metrics_path})"

    return config.DEFAULT_THRESHOLD, f"default: no tuned threshold found at {metrics_path}"


def predict_probabilities_for_files(
    model: nn.Module, image_paths: list[Path], image_size: int, device: torch.device
) -> np.ndarray:
    """The model's probability of infection for each image file, in the same order."""
    transform = build_eval_transform(image_size)  # exactly the preparation used in validation and testing
    model.eval()
    all_probabilities = []

    with torch.no_grad():  # only predicting, not learning
        for start in range(0, len(image_paths), config.BATCH_SIZE):
            batch_paths = image_paths[start:start + config.BATCH_SIZE]
            image_list = []
            for path in batch_paths:
                with Image.open(path) as image:
                    image_list.append(transform(image.convert("RGB")))  # (3, image_size, image_size)
            images = torch.stack(image_list).to(device)  # (batch, 3, image_size, image_size)
            probabilities = torch.sigmoid(model(images))  # (batch,)
            all_probabilities.append(probabilities.cpu().numpy())

    return np.concatenate(all_probabilities)  # (num_images,)


def predict_files(
    weights_path: Path,
    image_paths: list[Path],
    threshold_override: float | None = None,
    results_dir: Path = config.RESULTS_DIR,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, float, str]:
    """
    Classify the given image files with the model saved at weights_path.
    Returns (table with one row per image, threshold used, why that threshold).
    """
    if device is None:
        device = get_device()
    model, checkpoint = load_checkpoint(weights_path, device)
    threshold, threshold_reason = choose_threshold(checkpoint["model_name"], threshold_override, results_dir)

    probabilities = predict_probabilities_for_files(model, image_paths, checkpoint["image_size"], device)
    decisions = predict_infected(probabilities, threshold)  # the same rule as in evaluation

    rows = []
    for path, probability, decision in zip(image_paths, probabilities, decisions):
        rows.append({
            "image": str(path),
            "prediction": LABEL_NAMES[int(decision)],
            "probability_infected": float(probability),
        })
    return pd.DataFrame(rows), threshold, threshold_reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify red blood cell images as Parasitized or Uninfected.")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Trained model file, e.g. outputs/resnet18_best.pt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One cell image")
    source.add_argument("--folder", type=Path, help="A folder of cell images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the decision threshold (default: the model's tuned threshold)")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR,
                        help="Where <model>_metrics.json (with the tuned threshold) lives")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Also save the predictions to this CSV file")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"No model at {args.weights}. Train one, or download the released weights "
                                "(see README.md, 'Trained weights').")

    if args.image is not None:
        if not args.image.is_file():
            raise FileNotFoundError(f"No image at {args.image}")
        image_paths = [args.image]
    else:
        image_paths = find_image_files(args.folder)
        if not image_paths:
            raise FileNotFoundError(f"No images ({', '.join(config.PREDICT_EXTENSIONS)}) found in {args.folder}")

    predictions, threshold, threshold_reason = predict_files(args.weights, image_paths, args.threshold,
                                                             args.results_dir)

    print(f"Threshold {threshold:.4f} ({threshold_reason})")
    print("Infected if the probability of infection is at or above it.\n")
    # Make the name column as wide as the longest file name, so the columns line up.
    names = [Path(image).name for image in predictions["image"]]
    name_width = max(len(name) for name in names)
    for name, (_, row) in zip(names, predictions.iterrows()):
        print(f"{name:<{name_width}}  {row['prediction']:<12} p(infected) = {row['probability_infected']:.3f}")

    if len(predictions) > 1:
        counts = predictions["prediction"].value_counts()
        print(f"\n{len(predictions)} images: {counts.get('Parasitized', 0)} Parasitized, "
              f"{counts.get('Uninfected', 0)} Uninfected")

    if args.output_csv is not None:
        predictions.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
