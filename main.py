import argparse
import os

import matplotlib
import pandas as pd
from kagglehub.datasets import dataset_download
from matplotlib import pyplot as plt
from scipy.stats import norm

from src.plot import (
    plot_anomaly_scores,
    plot_confusion_matrix,
    plot_feature,
    plot_simulation,
)


def get_dataset() -> pd.DataFrame:
    # https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
    path = dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    return pd.read_csv(f"{path}/creditcard_2023.csv")


def generate_plots(data: pd.DataFrame, percent: float) -> None:
    """Generate plots for the dataset."""
    plot_anomaly_scores(data, percent, "data/anomaly_scores.png")
    plot_confusion_matrix(data["Class"], data["Prediction"], "data/confusion_matrix.png")
    plot_simulation(data, 200, percent, "data/simulation.png")

    for feature in range(1, 29):
        plot_feature(data[f"V{feature}"], f"data/V{feature}.png")


def print_statistics(data: pd.DataFrame, normal: pd.Series, fraud: pd.Series, percent: float) -> None:
    """Prints statistics about fraud detection results, including 95% confidence intervals."""
    caught = (data["Prediction"] & (data["Class"] == 1)).sum()
    missed = len(fraud) - caught
    false_alarms = (data["Prediction"] & (data["Class"] == 0)).sum()
    precision = caught / (caught + false_alarms) if (caught + false_alarms) > 0 else 0.0
    n_predicted_fraud = caught + false_alarms

    z = norm.ppf(0.975)
    se_precision = (precision * (1 - precision) / n_predicted_fraud) ** 0.5
    ci_precision = (precision - z * se_precision, precision + z * se_precision)

    print(f"Total Transactions: {len(data)}")
    print(f"Normal Transactions: {len(normal)}")
    print(f"Fraud Transactions: {len(fraud)}")
    print(f"Frauds Caught: {caught}")
    print(f"Frauds Missed: {missed}")
    print(f"False Alarms: {false_alarms}")
    print(f"Percent of frauds caught: {caught / len(fraud) * 100:.2f}%")
    print(f"Precision: {precision:.2%} (95% CI: {ci_precision[0]:.2%} - {ci_precision[1]:.2%})")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")
    parser.add_argument(
        "-p", "--percent", type=float, default=85, help="Percent of frauds to catch (default: 85, range: 1-100)"
    )
    args = parser.parse_args()
    percent = args.percent
    if not (1 <= percent <= 100):
        raise ValueError("Invalid percent value")

    # Set up matplotlib
    matplotlib.use("agg")
    plt.rcParams.update({"figure.max_open_warning": 0})

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Load the dataset
    data = get_dataset()
    fraud = data[data["Class"] == 1]
    normal = data[data["Class"] == 0]

    # Calculate anomaly scores
    features = data.drop(columns=["id", "Class"])
    normal_features = normal.drop(columns=["id", "Class"])
    data_z = (features - normal_features.mean()) / normal_features.std()
    data["Anomaly"] = data_z.abs().sum(axis=1)

    # Predict anomalies based on the specified threshold
    threshold = data[data["Class"] == 1]["Anomaly"].quantile(1 - percent / 100)
    data["Prediction"] = data["Anomaly"] > threshold

    # Print statistics
    print_statistics(data, normal, fraud, percent)

    # Generate plots
    generate_plots(data, percent)


if __name__ == "__main__":
    main()
