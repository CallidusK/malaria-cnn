"""
models.py: the neural networks that look at a cell image and score how likely it is infected.

Where this fits in the pipeline:
    data -> [model] -> train -> evaluate -> predict

    SimpleCNN   our own small CNN, built from scratch (Phase 3)

It takes a batch of images (batch, 3, height, width) and returns ONE number per
image, called a "logit": a raw score where positive means "probably infected"
and negative means "probably healthy". The sigmoid function turns a logit into
a probability between 0 and 1.
"""

import torch
from torch import nn

from src import config

# The models that build_model() knows about. Model files saved by train.py
# record this name, so evaluate.py and predict.py know what to rebuild.
MODEL_NAMES: list[str] = ["simple_cnn"]


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


def build_model(model_name: str) -> nn.Module:
    """Create a fresh model (random weights) by name. Today the only one is "simple_cnn"."""
    if model_name == "simple_cnn":
        return SimpleCNN()
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {MODEL_NAMES}")


def image_size_for(model_name: str) -> int:
    """The image size each model expects (set in config.py)."""
    if model_name == "simple_cnn":
        return config.SIMPLE_CNN_IMAGE_SIZE
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    """How many numbers (weights) the model has. A rough measure of its size."""
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()  # numel = number of elements in this weight tensor
    return total

