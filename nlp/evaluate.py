import os
import logging
import pandas as pd
from typing import Dict, Any

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_classifier(
    data_csv: str = "data/bootstrap_training_set.csv",
    report_dir: str = "reports",
) -> Dict[str, float]:
    """Compute precision, recall, F1 score, and save confusion matrix plot to reports/."""
    if not os.path.exists(data_csv):
        logger.error(f"Evaluation CSV '{data_csv}' not found.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    df = pd.read_csv(data_csv)

    y_true = df.get("manually_verified_label", df.get("bootstrap_ghost_label", []))
    y_pred = df.get("bootstrap_ghost_label", y_true)

    if len(y_true) == 0:
        logger.warning("No labels found in evaluation dataset.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }

    logger.info(f"Evaluation Metrics: {metrics}")

    # Generate and save Confusion Matrix Plot
    if not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Ghost Job"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Ghost Job Classifier Confusion Matrix")
    plot_path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix plot to '{plot_path}'")
    return metrics


if __name__ == "__main__":
    evaluate_classifier()
