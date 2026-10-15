"""
train.py: teaches a model to tell infected cells from healthy ones.

Where this fits in the pipeline:
    data -> model -> [train] -> evaluate -> predict

What happens, in the order of this file:
    1. train_one_epoch()    one pass over all training images, updating the weights
    2. validate()           measure loss, accuracy and recall on the validation images
    3. train_model()        repeat 1 + 2 for many epochs, with a learning-rate scheduler,
                            early stopping, and saving the best model so far
    4. save/plot history    a JSON file and a picture of the loss/accuracy curves
    5. overfit_one_batch()  sanity check: can the model memorise just 16 images?

How to run it:
    python -m src.train --smoke-test                    # 1 epoch on fake data (laptop)
    python -m src.train --overfit-batch                 # sanity check on fake data
    python -m src.train --model simple_cnn --data-dir /kaggle/input/...   # real training (Kaggle)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

# Draw plots straight into image files. There's no screen to show them on
# when this runs as a script on Kaggle.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
from matplotlib.ticker import MaxNLocator  # noqa: E402

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data import (
    MalariaDataset,
    build_eval_transform,
    default_splits_csv_for,
    get_dataloaders,
    load_or_create_splits,
)
from src.make_fake_data import create_fake_dataset
from src.models import MODEL_NAMES, build_model, count_parameters, image_size_for
from src.utils import get_device, set_seed


# ===========================================================================
# 1. One epoch of training
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Show the model every training image once, adjusting its weights after each batch.
    Returns the average training loss over the epoch.
    """
    # Training mode: dropout is ON, and BatchNorm uses the current batch's statistics.
    model.train()

    total_loss = 0.0
    total_images = 0
    for images, labels in tqdm(train_loader, desc="  training", leave=False):
        images = images.to(device)  # (batch, 3, height, width)
        labels = labels.to(device)  # (batch,) each 0.0 or 1.0

        # The 5 steps of every training iteration:
        # Step 1, forward pass: the model scores every image in the batch.
        logits = model(images)  # (batch,)

        # Step 2, loss: one number saying how wrong those scores are.
        # BCEWithLogitsLoss = Binary Cross-Entropy, the standard loss for yes/no
        # questions. It applies the sigmoid itself ("with logits"), which is more
        # numerically stable than doing sigmoid first.
        loss = loss_function(logits, labels)  # a single number

        # Step 3, zero the gradients: PyTorch ADDS new gradients to old ones,
        # so we clear the previous batch's gradients first.
        optimiser.zero_grad()

        # Step 4, backward pass: work out, for every weight, which direction
        # would make the loss smaller (the "gradient").
        loss.backward()

        # Step 5, optimiser step: nudge every weight a little in that direction.
        optimiser.step()

        # loss.item() is the AVERAGE loss over this batch; weight it by the batch
        # size so the last (often smaller) batch counts fairly.
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


# ===========================================================================
# 2. Validation
# ===========================================================================

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Measure the model on images it does NOT learn from.

    Returns:
        loss      average validation loss (what early stopping watches)
        accuracy  share of all cells classified correctly
        recall    share of truly infected cells that were caught (the medical priority)
    """
    # Evaluation mode: dropout OFF, BatchNorm uses the statistics it learned
    # during training. Without this, results would be random and wrong.
    model.eval()

    total_loss = 0.0
    total_images = 0
    num_correct = 0
    num_infected = 0         # truly infected cells seen so far
    num_infected_caught = 0  # ...of which the model called infected

    # no_grad: we're only measuring, not learning, so PyTorch can skip the
    # bookkeeping needed for gradients. Faster, and uses less memory.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)  # (batch, 3, height, width)
            labels = labels.to(device)  # (batch,)

            logits = model(images)  # (batch,)
            loss = loss_function(logits, labels)

            # Sigmoid squashes any logit into a probability between 0 and 1.
            probabilities = torch.sigmoid(logits)  # (batch,)
            # At or above the threshold -> predict infected (1.0), else healthy (0.0).
            # (Same rule as src/evaluate.py -> predict_infected().)
            predictions = (probabilities >= config.DEFAULT_THRESHOLD).float()  # (batch,)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size
            num_correct += (predictions == labels).sum().item()

            is_infected = labels == config.LABEL_PARASITIZED  # (batch,) True/False
            num_infected += is_infected.sum().item()
            num_infected_caught += (predictions[is_infected] == config.LABEL_PARASITIZED).sum().item()

    if num_infected > 0:
        recall = num_infected_caught / num_infected
    else:
        recall = 0.0  # can't happen with our split checks, but avoids dividing by zero

    return {
        "loss": total_loss / total_images,
        "accuracy": num_correct / total_images,
        "recall": recall,
    }


# ===========================================================================
# 3. The full training run
# ===========================================================================

def save_checkpoint(model: nn.Module, model_name: str, epoch: int, val_loss: float, path: Path) -> None:
    """
    Save the model's learned weights, plus what's needed to use them later.
    A "state dict" is a dictionary of every weight tensor in the model, by layer name.
    """
    torch.save(
        {
            "model_name": model_name,
            "image_size": image_size_for(model_name),
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def print_epoch(record: dict) -> None:
    """One readable line per epoch."""
    print(
        f"epoch {record['epoch']:>2} | "
        f"train loss {record['train_loss']:.4f} | "
        f"val loss {record['val_loss']:.4f} | "
        f"val acc {record['val_accuracy']:.3f} | "
        f"val recall {record['val_recall']:.3f} | "
        f"lr {record['learning_rate']:.0e} | "
        f"{record['seconds']:.0f}s"
    )


def train_model(
    model_name: str,
    split_table: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    num_epochs: int,
    device: torch.device,
) -> list[dict]:
    """
    Train a fresh model for up to num_epochs, keeping the best version.

    "Best" = lowest validation loss. Training stops early if the validation
    loss hasn't improved for EARLY_STOPPING_PATIENCE epochs in a row.
    Returns the history: one dictionary of numbers per epoch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = image_size_for(model_name)
    dataloaders = get_dataloaders(split_table, data_dir, image_size)

    model = build_model(model_name).to(device)
    print(f"Model: {model_name} with {count_parameters(model):,} learnable parameters")

    loss_function = nn.BCEWithLogitsLoss()

    # Adam: an optimiser that adapts the step size for each weight automatically.
    # A reliable default that needs little tuning.
    optimiser = torch.optim.Adam(model.parameters(), lr=config.SIMPLE_CNN_LEARNING_RATE)

    # When validation loss hasn't improved for LR_SCHEDULER_PATIENCE epochs,
    # multiply the learning rate by LR_SCHEDULER_FACTOR: smaller, more careful
    # steps once we're close to a good solution.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",  # "min" because lower validation loss is better
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    checkpoint_path = output_dir / f"{model_name}_best.pt"
    best_val_loss = math.inf  # anything beats infinity, so epoch 1 is always "best so far"
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        learning_rate = optimiser.param_groups[0]["lr"]  # the rate used during THIS epoch

        train_loss = train_one_epoch(model, dataloaders["train"], loss_function, optimiser, device)
        val_metrics = validate(model, dataloaders["val"], loss_function, device)
        scheduler.step(val_metrics["loss"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_recall": val_metrics["recall"],
            "learning_rate": learning_rate,
            "seconds": time.time() - start_time,
        }
        history.append(record)
        print_epoch(record)

        # Early stopping: keep the best model; give up after too many epochs without progress.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(model, model_name, epoch, best_val_loss, checkpoint_path)
            print(f"          new best validation loss: saved to {checkpoint_path.name}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping: no improvement for {epochs_without_improvement} epochs.")
                break

    save_history(history, output_dir / f"{model_name}_history.json")
    plot_history(history, output_dir / f"{model_name}_curves.png")
    print(f"Best validation loss {best_val_loss:.4f}. Outputs are in {output_dir}")
    return history


# ===========================================================================
# 4. Saving and plotting the history
# ===========================================================================

def save_history(history: list[dict], path: Path) -> None:
    """Write the per-epoch numbers to a JSON file (readable as plain text)."""
    with open(path, "w") as history_file:
        json.dump(history, history_file, indent=2)


def plot_history(history: list[dict], path: Path) -> None:
    """
    Draw the training curves.
    Left: train vs validation loss (a widening gap = overfitting).
    Right: validation accuracy and recall.
    """
    epochs = [record["epoch"] for record in history]
    train_losses = [record["train_loss"] for record in history]
    val_losses = [record["val_loss"] for record in history]
    val_accuracies = [record["val_accuracy"] for record in history]
    val_recalls = [record["val_recall"] for record in history]

    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epochs, train_losses, marker="o", label="train loss")
    loss_axis.plot(epochs, val_losses, marker="o", label="validation loss")
    loss_axis.set_title("Loss (lower is better)")
    loss_axis.set_xlabel("epoch")
    loss_axis.legend()

    metric_axis.plot(epochs, val_accuracies, marker="o", label="validation accuracy")
    metric_axis.plot(epochs, val_recalls, marker="o", label="validation recall")
    metric_axis.set_title("Validation accuracy and recall (higher is better)")
    metric_axis.set_xlabel("epoch")
    # No fixed 0-to-1 range: matplotlib zooms in on the actual values, so small
    # but real differences (e.g. 94% vs 96%) are visible instead of a flat line.
    metric_axis.legend()

    # Epochs are whole numbers, so don't show ticks like 2.5.
    for axis in (loss_axis, metric_axis):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


# ===========================================================================
# 5. Sanity check: overfit one small batch
# ===========================================================================

def get_fixed_batch(
    split_table: pd.DataFrame, data_dir: Path, image_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pick `batch_size` training images (half infected, half healthy) as one fixed batch.
    No augmentation, so the images are exactly the same at every step.
    """
    training_rows = split_table[split_table["split"] == "train"]
    images_per_class = batch_size // 2
    chosen_rows = training_rows.groupby("label").sample(n=images_per_class, random_state=config.RANDOM_SEED)

    dataset = MalariaDataset(chosen_rows, data_dir, build_eval_transform(image_size))
    image_list = []
    label_list = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        image_list.append(image)
        label_list.append(label)

    images = torch.stack(image_list)  # (batch_size, 3, image_size, image_size)
    labels = torch.stack(label_list)  # (batch_size,)
    return images, labels


def overfit_one_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    num_steps: int = config.OVERFIT_STEPS,
) -> list[float]:
    """
    Train on the SAME small batch over and over, and return the loss at every step.

    A healthy model + training loop can memorise a handful of images, so the
    loss should drop close to zero. If it can't even do that, something is
    broken (wrong labels, a layer not connected, learning rate far off...),
    and there's no point starting a long training run.
    """
    model = model.to(device)
    model.train()
    images = images.to(device)  # (batch, 3, height, width)
    labels = labels.to(device)  # (batch,)

    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.SIMPLE_CNN_LEARNING_RATE)

    losses = []
    for _ in range(num_steps):
        logits = model(images)                 # step 1: forward
        loss = loss_function(logits, labels)   # step 2: loss
        optimiser.zero_grad()                  # step 3: zero gradients
        loss.backward()                        # step 4: backward
        optimiser.step()                       # step 5: update weights
        losses.append(loss.item())
    return losses


def run_overfit_check(model_name: str, split_table: pd.DataFrame, data_dir: Path, device: torch.device) -> bool:
    """Run the overfit-one-batch check, print the loss as it falls, and report pass/fail."""
    images, labels = get_fixed_batch(split_table, data_dir, image_size_for(model_name), config.OVERFIT_BATCH_SIZE)
    print(f"Overfitting one batch of {len(labels)} images for {config.OVERFIT_STEPS} steps...")

    losses = overfit_one_batch(build_model(model_name), images, labels, device)
    print_every = 10
    for step in range(0, len(losses), print_every):
        print(f"  step {step + 1:>3}: loss {losses[step]:.4f}")

    final_loss = losses[-1]
    passed = final_loss < config.OVERFIT_TARGET_LOSS
    if passed:
        print(f"PASSED: final loss {final_loss:.4f} < {config.OVERFIT_TARGET_LOSS}. The model can learn.")
    else:
        print(f"FAILED: final loss {final_loss:.4f} >= {config.OVERFIT_TARGET_LOSS}. Something is broken.")
    return passed


# ===========================================================================
# Command line
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a malaria cell classifier.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="simple_cnn")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Folder with the cell images. Not needed with --smoke-test or --overfit-batch.")
    parser.add_argument("--splits-csv", type=Path, default=None,
                        help="Saved train/val/test split (default: chosen from --data-dir)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick check that everything runs: 1 epoch on the fake data.")
    parser.add_argument("--overfit-batch", action="store_true",
                        help="Sanity check: try to memorise one small batch.")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")

    # Without --data-dir, the quick checks use the fake dataset (created if missing).
    data_dir = args.data_dir
    if data_dir is None:
        if not (args.smoke_test or args.overfit_batch):
            parser.error("--data-dir is required for real training (or use --smoke-test).")
        data_dir = config.FAKE_DATA_DIR
        if not data_dir.exists():
            print("No fake data found. Creating it.")
            create_fake_dataset(data_dir)

    splits_csv = args.splits_csv or default_splits_csv_for(data_dir)
    split_table = load_or_create_splits(data_dir, splits_csv)

    if args.overfit_batch:
        passed = run_overfit_check(args.model, split_table, data_dir, device)
        sys.exit(0 if passed else 1)  # exit code 1 tells scripts/notebooks the check failed

    if args.smoke_test:
        num_epochs = config.SMOKE_TEST_EPOCHS
        output_dir = config.OUTPUT_DIR / "smoke_test"  # keep smoke outputs apart from real ones
    else:
        num_epochs = config.MAX_EPOCHS
        output_dir = config.OUTPUT_DIR

    train_model(args.model, split_table, data_dir, output_dir, num_epochs, device)


if __name__ == "__main__":
    main()
