"""
train.py: the training loop. Teaches a model using the training images, checks it on
the validation images, stops early when it stops improving, and saves the best version.

Where this fits in the pipeline:
    data -> model -> [train] -> evaluate -> predict

Built in Phase 3.
"""
