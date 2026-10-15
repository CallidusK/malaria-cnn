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
from torch.utils.data import DataLoader, TensorDataset

from src import config, data, train
from src.make_fake_data import create_fake_dataset
from src.models import (
    ResNet18Classifier,
    SimpleCNN,
    build_model,
    count_parameters,
    count_trainable_parameters,
    freeze_pretrained_layers,
    keep_batchnorm_statistics_frozen,
    unfreeze_all_layers,
)
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
    images = torch.randn(2, config.NUM_CHANNELS, config.RESNET_IMAGE_SIZE, config.RESNET_IMAGE_SIZE)  # (2, 3, 224, 224)
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


# ---------------------------------------------------------------------------
# ResNet18 (Phase 6). pretrained=False everywhere: the tests never download anything.
# ---------------------------------------------------------------------------


def test_resnet18_gives_one_logit_per_image() -> None:
    model = ResNet18Classifier(pretrained=False)
    images = torch.randn(2, config.NUM_CHANNELS, config.RESNET_IMAGE_SIZE, config.RESNET_IMAGE_SIZE)  # (2, 3, 224, 224)
    assert model(images).shape == (2,)


def test_freezing_leaves_only_the_new_last_layer_trainable() -> None:
    model = ResNet18Classifier(pretrained=False)
    freeze_pretrained_layers(model)
    # The new last layer: 512 weights + 1 bias.
    assert count_trainable_parameters(model) == config.RESNET_FEATURE_SIZE + 1
    unfreeze_all_layers(model)
    assert count_trainable_parameters(model) == count_parameters(model)


def test_batchnorm_statistics_only_change_when_not_frozen() -> None:
    model = ResNet18Classifier(pretrained=False)
    first_batchnorm = model.backbone.bn1
    images = torch.randn(4, config.NUM_CHANNELS, 64, 64)  # (4, 3, 64, 64): small, to keep the test fast

    # Frozen: training mode, but BatchNorm switched back to evaluation mode.
    model.train()
    keep_batchnorm_statistics_frozen(model)
    mean_before = first_batchnorm.running_mean.clone()
    model(images)
    assert torch.equal(first_batchnorm.running_mean, mean_before)

    # Not frozen: plain training mode updates the running statistics (the trap we avoid).
    model.train()
    model(images)
    assert not torch.equal(first_batchnorm.running_mean, mean_before)


def test_frozen_stage_only_changes_the_last_layer() -> None:
    set_seed(config.RANDOM_SEED)
    model = ResNet18Classifier(pretrained=False)
    freeze_pretrained_layers(model)
    first_conv_before = model.backbone.conv1.weight.clone()
    batchnorm_mean_before = model.backbone.bn1.running_mean.clone()
    last_layer_before = model.classifier().weight.clone()

    images = torch.randn(8, config.NUM_CHANNELS, 64, 64)  # (8, 3, 64, 64)
    labels = torch.tensor([0.0, 1.0] * 4)                 # (8,)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    optimiser = torch.optim.Adam(model.classifier().parameters(), lr=config.RESNET_HEAD_LEARNING_RATE)
    train.train_one_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), optimiser, CPU, batchnorm_frozen=True)

    assert torch.equal(model.backbone.conv1.weight, first_conv_before)                # pretrained weights: untouched
    assert torch.equal(model.backbone.bn1.running_mean, batchnorm_mean_before)        # BatchNorm statistics: untouched
    assert not torch.equal(model.classifier().weight, last_layer_before)              # new layer: learned


def test_resnet18_trains_in_two_stages(tmp_path: Path) -> None:
    set_seed(config.RANDOM_SEED)
    data_dir = tmp_path / "fake_data"
    create_fake_dataset(data_dir)
    split_table = data.split_by_patient(data.find_images(data_dir))

    history = train.train_model("resnet18", split_table, data_dir, tmp_path / "outputs",
                                num_epochs=1, device=CPU, pretrained=False)

    stages = [record["stage"] for record in history]
    assert stages == ["frozen", "fine-tune"]
    assert [record["epoch"] for record in history] == [1, 2]  # epoch numbers continue across stages
    assert (tmp_path / "outputs" / "resnet18_best.pt").exists()
