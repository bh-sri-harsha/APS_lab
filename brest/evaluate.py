import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def evaluate_thresholds(y_test, probabilities, thresholds):
    actual = y_test.values
    results = []

    for threshold in thresholds:
        predicted = (probabilities[:, 0] >= threshold).astype(int)

        tn = np.sum((actual == 0) & (predicted == 0))
        fp = np.sum((actual == 0) & (predicted == 1))
        fn = np.sum((actual == 1) & (predicted == 0))
        tp = np.sum((actual == 1) & (predicted == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        cm = confusion_matrix(actual, predicted)

        results.append({
            "Threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ConfusionMatrix": cm
        })

    return pd.DataFrame(results)