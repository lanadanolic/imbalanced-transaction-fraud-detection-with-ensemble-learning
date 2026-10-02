from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_metrics(
    y_true,
    y_pred
) -> dict:
   
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred
        ),
        "Recall": recall_score(
            y_true,
            y_pred
        ),
        "F1-score": f1_score(
            y_true,
            y_pred
        )
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    save_path: Path | None = None
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Predicted 0",
            "Predicted 1"
        ],
        yticklabels=[
            "Actual 0",
            "Actual 1"
        ]
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def create_metrics_table(
    model_results: dict
) -> pd.DataFrame:

    rows = []

    for model_name, metrics in model_results.items():

        row = {
            "Model": model_name,
            **metrics
        }

        rows.append(row)

    return pd.DataFrame(rows)


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    save_path: Path | None = None
):

    plot_df = metrics_df.set_index(
        "Model"
    )[
        [
            "Precision",
            "Recall",
            "F1-score"
        ]
    ]

    plot_df.plot(
        kind="bar",
        figsize=(9, 5)
    )

    plt.title(
        "Model Performance Comparison"
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()