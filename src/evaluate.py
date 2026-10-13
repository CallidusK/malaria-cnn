"""
evaluate.py: scores the trained model on the untouched test set (accuracy, recall,
specificity, ROC curve, confusion matrix) and tunes the decision threshold on validation data.

Where this fits in the pipeline:
    data -> model -> train -> [evaluate] -> predict

Built in Phase 5.
"""
