import pandas as pd
from kagglehub.datasets import dataset_download


def get_dataset() -> pd.DataFrame:
    # https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
    path = dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    return pd.read_csv(f"{path}/creditcard_2023.csv")


def main() -> None:
    # Download the dataset
    data = get_dataset()

    # Split the dataset into normal and fraud transactions
    fraud = data[data["Class"] == 1]
    normal = data[data["Class"] == 0]

    # Calculate the z-scores for the features
    features = data.drop(columns=["id", "Class"])
    normal_features = normal.drop(columns=["id", "Class"])
    data_z = (features - normal_features.mean()) / normal_features.std()

    # Calculate the anomaly score by summing the absolute z-scores
    data["Anomaly"] = data_z.abs().sum(axis=1)

    # Predict anomalies based on the anomaly scores
    PERCENT_TO_CATCH = 85  # Catch 85% of the frauds
    threshold = data[data["Class"] == 1]["Anomaly"].quantile(1 - PERCENT_TO_CATCH / 100)
    data["Prediction"] = data["Anomaly"] > threshold

    # Calculate the results
    caught = (data["Prediction"] & (data["Class"] == 1)).sum()
    missed = len(fraud) - caught
    false_alarms = (data["Prediction"] & (data["Class"] == 0)).sum()
    precision = caught / (caught + false_alarms) if (caught + false_alarms) > 0 else 0

    # Print the results
    print(f"Total Transactions: {len(data)}")
    print(f"Frauds: {len(fraud)}")
    print(f"Frauds Caught: {caught}")
    print(f"Frauds Missed: {missed}")
    print(f"False Alarms: {false_alarms}")
    print(f"Percent of frauds caught: {PERCENT_TO_CATCH}%")
    print(f"Precision: {precision:.2%}")


if __name__ == "__main__":
    main()
