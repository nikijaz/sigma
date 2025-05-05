# Sigma

This program serves as a proof of concept for detecting credit card fraud using purely statistical methods. The approach
focuses on analyzing transaction patterns to identify anomalies without relying on machine learning. For a detailed
explanation of the methodology, results, and insights, please refer to the [report](REPORT.pdf).

## Installation

0. Ensure [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed.
1. Clone and navigate to the repository:

    ```shell
    git clone https://github.com/nikijaz/sigma.git
    cd sigma/
    ```

2. Install dependencies:

    ```shell
    uv sync
    ```

3. Run the script:

    ```shell
    # <PERCENT> - Percent of frauds to catch (default: 85)  
    uv run main.py -p <PERCENT>
    ```

## Technologies Used

**Python** powers the program's logic, with **uv** as the package manager. The algorithm uses **pandas** for data
manipulation and **scipy** for numerical operations. **Seaborn** and **matplotlib** are used for data visualization.

## Dataset

The dataset used is the
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
from Kaggle.
