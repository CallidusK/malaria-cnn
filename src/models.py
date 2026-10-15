"""
models.py: the neural networks that look at a cell image and score how likely it is infected.

Where this fits in the pipeline:
    data -> [model] -> train -> evaluate -> predict

    SimpleCNN   our own small CNN, built from scratch (Phase 3)
    ResNet18    a bigger pretrained model (added in Phase 6)

Both take a batch of images (batch, 3, height, width) and return ONE number per
image, called a "logit": a raw score where positive means "probably infected"
and negative means "probably healthy". The sigmoid function turns a logit into
a probability between 0 and 1.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src import config

# The models that build_model() knows about.
MODEL_NAMES: list[str] = ["simple_cnn", "resnet18"]


class ConvBlock(nn.Module):
    """
    One building block of SimpleCNN: Conv -> BatchNorm -> ReLU -> MaxPool.

    Input  (batch, in_channels,  height,     width)
    Output (batch, out_channels, height / 2, width / 2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Convolution: slides `out_channels` small filters (3x3 pixels each)
        # over the image. Each filter learns to light up for one visual
        # pattern, producing one "feature map" showing where that pattern is.
        # The padding adds a 1-pixel border so the output keeps the same height/width.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.CONV_KERNEL_SIZE,
            padding=config.CONV_KERNEL_SIZE // 2,
            bias=False,  # BatchNorm right after adds its own shift, so a bias here would be wasted
        )

        # BatchNorm: rescales each feature map to a steady range (mean ~0,
        # spread ~1) across the batch. This keeps training stable and faster.
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # ReLU: keeps positive values, turns negatives into 0. This simple
        # bend is what lets stacked layers learn complicated, non-straight-line patterns.
        self.relu = nn.ReLU()

        # Max pooling: in every 2x2 square, keep only the largest value.
        # Halves the height and width: less to compute, and the next block
        # sees a wider area of the cell with each of its 3x3 filters.
        self.pool = nn.MaxPool2d(kernel_size=config.POOL_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)        # (batch, out_channels, height, width)
        x = self.batch_norm(x)  # same shape
        x = self.relu(x)        # same shape
        x = self.pool(x)        # (batch, out_channels, height / 2, width / 2)
        return x


class SimpleCNN(nn.Module):
    """
    Our own CNN: 4 ConvBlocks, then global average pooling, dropout and one output.

    Shapes for a batch of 128x128 images (the channel counts come from
    SIMPLE_CNN_CHANNELS in config.py):
        input             (batch,   3, 128, 128)
        after block 1     (batch,  32,  64,  64)
        after block 2     (batch,  64,  32,  32)
        after block 3     (batch, 128,  16,  16)
        after block 4     (batch, 256,   8,   8)
        global avg pool   (batch, 256,   1,   1)
        flatten           (batch, 256)
        linear            (batch,   1)
        output            (batch,)          one logit per image
    """

    def __init__(self, dropout_probability: float = config.DROPOUT_PROBABILITY):
        super().__init__()
        channels_1, channels_2, channels_3, channels_4 = config.SIMPLE_CNN_CHANNELS

        self.block1 = ConvBlock(config.NUM_CHANNELS, channels_1)  # finds edges and colour patches
        self.block2 = ConvBlock(channels_1, channels_2)           # combines them into small shapes
        self.block3 = ConvBlock(channels_2, channels_3)           # larger shapes (blobs, rings)
        self.block4 = ConvBlock(channels_3, channels_4)           # whole-cell patterns (e.g. "blob inside cell")

        # Global average pooling: average each of the 256 feature maps down to
        # one number. "How strongly does pattern k appear anywhere in the cell?"
        # It also means the model works for any input size, not just 128x128.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Dropout: during training, randomly zero 30% of the 256 numbers, so the
        # final layer can't depend on any single feature. Switched off automatically
        # when the model is in evaluation mode (model.eval()).
        self.dropout = nn.Dropout(p=dropout_probability)

        # Linear layer: a weighted sum of the 256 features -> 1 score (the logit).
        self.classifier = nn.Linear(channels_4, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.block1(images)       # (batch,  32, 64, 64)
        x = self.block2(x)            # (batch,  64, 32, 32)
        x = self.block3(x)            # (batch, 128, 16, 16)
        x = self.block4(x)            # (batch, 256,  8,  8)
        x = self.global_pool(x)       # (batch, 256,  1,  1)
        x = torch.flatten(x, start_dim=1)  # (batch, 256): drop the two size-1 dimensions
        x = self.dropout(x)           # (batch, 256)
        logits = self.classifier(x)   # (batch, 1)
        # Remove the last size-1 dimension so the shape matches the labels: (batch,)
        return logits.squeeze(dim=1)  # (batch,)


class ResNet18Classifier(nn.Module):
    """
    ResNet18, pretrained on ImageNet (1.2 million everyday photos in 1,000 classes),
    with its last layer replaced so it answers our yes/no question.

    Why can photos of dogs and cars help with blood cells? The early layers of a
    trained CNN detect general things (edges, textures, colour blobs) that are
    useful for almost any image. We reuse those and only teach it what's new.

    Shapes for a batch of 224x224 images:
        input                        (batch,   3, 224, 224)
        ResNet18 layers (conv/pool)  (batch, 512,   7,   7)   inside the backbone
        its global average pool      (batch, 512)
        our new last layer           (batch,   1)
        output                       (batch,)                 one logit per image
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # pretrained=True downloads ImageNet weights (about 45 MB, needs Internet).
        # pretrained=False gives random weights: used by the tests, and when
        # loading our own trained model (whose weights replace them anyway).
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        # ResNet18's last layer ("fc") maps 512 features to 1,000 ImageNet classes.
        # Swap it for a fresh layer mapping the same 512 features to 1 logit.
        self.backbone.fc = nn.Linear(config.RESNET_FEATURE_SIZE, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(images)  # (batch, 1)
        return logits.squeeze(dim=1)    # (batch,)

    def classifier(self) -> nn.Module:
        """The new last layer: the only part that learns during the frozen stage."""
        return self.backbone.fc


def freeze_pretrained_layers(model: ResNet18Classifier) -> None:
    """
    Lock every pretrained weight, so only the new last layer learns.

    requires_grad=False tells PyTorch not to compute gradients for a weight,
    so the optimiser can't change it.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier().parameters():
        parameter.requires_grad = True


def unfreeze_all_layers(model: nn.Module) -> None:
    """Let every weight learn again (the fine-tune stage)."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def keep_batchnorm_statistics_frozen(model: nn.Module) -> None:
    """
    Stop BatchNorm layers from updating their running statistics.

    The subtle trap: freezing weights (requires_grad=False) does NOT freeze
    BatchNorm's running mean and variance. In training mode they keep updating
    from every batch, so the "frozen" pretrained layers would quietly change
    anyway. Putting just the BatchNorm layers into evaluation mode stops that.
    Call this AFTER model.train(), because model.train() switches them back on.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Create a fresh model by name: "simple_cnn" or "resnet18".
    `pretrained` only matters for ResNet18 (SimpleCNN always starts from random weights).
    """
    if model_name == "simple_cnn":
        return SimpleCNN()
    if model_name == "resnet18":
        return ResNet18Classifier(pretrained=pretrained)
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {MODEL_NAMES}")


def image_size_for(model_name: str) -> int:
    """The image size each model expects (set in config.py)."""
    if model_name == "simple_cnn":
        return config.SIMPLE_CNN_IMAGE_SIZE
    if model_name == "resnet18":
        return config.RESNET_IMAGE_SIZE
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    """How many numbers (weights) the model has. A rough measure of its size."""
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()  # numel = number of elements in this weight tensor
    return total


def count_trainable_parameters(model: nn.Module) -> int:
    """How many of those weights are currently allowed to learn (not frozen)."""
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += parameter.numel()
    return total
