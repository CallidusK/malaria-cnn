"""
test_models.py: automatic checks for the model (src/models.py) and training code (src/train.py).

Where this fits in the pipeline:
    data -> [model] -> [train] -> evaluate -> predict
The most important test here is the overfit-one-batch check: if the model
can't memorise 16 images, a long training run would only waste GPU time.

Run them with:
    pytest
"""

from pathlib import Path

import pytest
import torch

from src import config, data, train
from src.make_fake_data import create_fake_dataset
from src.models import SimpleCNN, build_model, count_parameters
from src.utils import set_seed

# Tests run on the CPU: always available, and gives the same numbers on every machine.
CPU = torch.device("cpu")


def test_simple_cnn_gives_one_logit_per_image() -> None:
    model = SimpleCNN()
    images = torch.randn(4, config.NUM_CHANNELS, config.SIMPLE_CNN_IMAGE_SIZE, config.SIMPLE_CNN_IMAGE_SIZE)  # (4, 3, 128, 128)
    logits = model(images)
    assert logits.shape == (4,)  # one score per image, same shape as a batch of labels


def test_simple_cnn_accepts_other_image_sizes() -> None:
    # Global average pooling squeezes any height/width down to 1x1,
    # so the model isn't tied to 128x128 inputs.
    model = SimpleCNN()
    bigger_size = 2 * config.SIMPLE_CNN_IMAGE_SIZE
    images = torch.randn(2, config.NUM_CHANNELS, bigger_size, bigger_size)  # (2, 3, 256, 256)
    assert model(images).shape == (2,)


def test_unknown_model_name_raises_an_error() -> None:
    with pytest.raises(ValueError):
        build_model("not_a_real_model")


def test_model_has_learnable_parameters() -> None:
    assert count_parameters(SimpleCNN()) > 0


def test_saved_checkpoint_reloads_the_same_model(tmp_path: Path) -> None:
    set_seed(config.RANDOM_SEED)
    model = SimpleCNN()
    checkpoint_path = tmp_path / "model.pt"
    train.save_checkpoint(model, "simple_cnn", epoch=1, val_loss=0.5, path=checkpoint_path)

    reloaded = SimpleCNN()
    checkpoint = torch.load(checkpoint_path)
    reloaded.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation mode switches off dropout, so the same input must give the same output.
    model.eval()
    reloaded.eval()
    images = torch.randn(2, config.NUM_CHANNELS, config.SIMPLE_CNN_IMAGE_SIZE, config.SIMPLE_CNN_IMAGE_SIZE)
    assert torch.allclose(model(images), reloaded(images))
    assert checkpoint["image_size"] == config.SIMPLE_CNN_IMAGE_SIZE


def test_model_can_overfit_one_small_batch(tmp_path: Path) -> None:
    """
    The key sanity check. Train repeatedly on the same 16 fake images: a working
    model and training loop must drive the loss close to zero.
    """
    set_seed(config.RANDOM_SEED)
    create_fake_dataset(tmp_path)
    split_table = data.split_by_patient(data.find_images(tmp_path))
    images, labels = train.get_fixed_batch(
        split_table, tmp_path, config.SIMPLE_CNN_IMAGE_SIZE, config.OVERFIT_BATCH_SIZE
    )
    assert images.shape == (config.OVERFIT_BATCH_SIZE, config.NUM_CHANNELS,
                            config.SIMPLE_CNN_IMAGE_SIZE, config.SIMPLE_CNN_IMAGE_SIZE)  # (16, 3, 128, 128)
    assert set(labels.tolist()) == {0.0, 1.0}  # both classes in the batch

    losses = train.overfit_one_batch(SimpleCNN(), images, labels, CPU)

    assert losses[-1] < losses[0]                     # it learned something
    assert losses[-1] < config.OVERFIT_TARGET_LOSS    # ...enough to memorise the batch
