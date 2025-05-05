import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


def plot_anomaly_scores(data: pd.DataFrame, percent: float, filename: str) -> None:
    """Plot the distribution of anomaly scores for normal and fraud transactions."""
    data = data[(data["Anomaly"] >= 0) & (data["Anomaly"] <= 150)]
    normal = data[data["Class"] == 0]
    fraud = data[data["Class"] == 1]
    threshold = fraud["Anomaly"].quantile(1 - percent / 100)

    ax = sns.kdeplot(normal["Anomaly"], fill=True, alpha=0.3, label="Normal", color="blue")
    sns.kdeplot(fraud["Anomaly"], fill=True, alpha=0.3, ax=ax, label="Fraud", color="red")
    ax.axvline(threshold, color="black", linestyle="--", label=f"{percent:.0f}% catch threshold")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend()

    fig = ax.get_figure()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    fig.clf()


def plot_feature(data: pd.Series, filename: str) -> None:
    """Plot the distribution of a feature with a normal distribution overlay."""
    z_score = (data - data.mean()) / data.std()
    z_score = z_score[(z_score >= -5) & (z_score <= 5)]

    ax = sns.kdeplot(z_score, fill=True, alpha=0.3)
    x = np.linspace(-5, 5, 100)
    ax.plot(x, norm.pdf(x, z_score.mean(), z_score.std()), "k--", label="Normal Distribution")

    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.legend()

    fig = ax.get_figure()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    fig.clf()


def plot_confusion_matrix(real: pd.Series, prediction: pd.Series, filename: str) -> None:
    """Plot the confusion matrix for the predictions."""
    tn = ((real == 0) & (prediction == 0)).sum()
    fp = ((real == 0) & (prediction == 1)).sum()
    fn = ((real == 1) & (prediction == 0)).sum()
    tp = ((real == 1) & (prediction == 1)).sum()
    cm = np.array([[tn, fp], [fn, tp]])
    cm = cm.astype("float") / cm.sum()

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig = ax.get_figure()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    fig.clf()


def plot_simulation(data: pd.DataFrame, size: int, percent: float, filename: str) -> None:
    """Plot a scatter plot of anomaly scores with colors based on class labels."""
    sample = data.sample(size)
    scores = sample["Anomaly"]
    labels = sample["Class"]
    threshold = sample[sample["Class"] == 1]["Anomaly"].quantile(1 - percent / 100)

    x = np.random.rand(len(sample))
    df_plot = pd.DataFrame({"x": x, "Anomaly": scores, "Class": labels.map({0: "Normal", 1: "Fraud"})})
    ax = sns.scatterplot(
        data=df_plot, x="x", y="Anomaly", hue="Class", palette={"Normal": "blue", "Fraud": "red"}, alpha=0.6
    )
    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        label=f"{percent:.0f}% catch threshold",
    )

    ax.set_ylabel("Anomaly Score")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.legend()

    fig = ax.get_figure()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    fig.clf()
