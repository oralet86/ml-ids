# **ml-ids**

This is a repository for a school project dedicated to the comparative analysis of machine learning and deep learning models for Network Intrusion Detection Systems (IDS).

## Comparative Analysis & Machine Learning for Network Intrusion Detection Systems (NIDS)

The project aims to benchmark standard algorithms against major datasets (CICIDS2017, UNSW-NB15, ToN-IoT) and presents a novel architecture to improve traffic classification performance.

## **📂 Project Structure**

To ensure the provided `utils.py` scripts work correctly, your project directory must look like this:

```text
ml-ids/
├── datasets/                   # Dataset root directory
│   ├── cicids2017/             # Place all CSVs here
│   ├── ton_iot/                # Place 'train_test_network.csv' here (can be in subfolders)
│   └── unsw_nb15/              # Place Training/Testing set CSVs here
├── hyperparams/                # Stores hyperparameter configurations
├── models/                     # Stores trained model artifacts (.pkl, .h5)
├── logs.log                    # Execution logs (auto-generated)
├── utils.py                    # The core utility script (Paths, Logger, Data Loaders)
└── main.py                     # Entry point
```

-----

## **🚀 Getting Started**

### **Prerequisites**

* **Python 3.13.7**

### **Installation**

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ml-ids.git
    cd ml-ids
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

-----

## **🛠 Utility Module Documentation (`utils.py`)**

This project includes a utility module (`utils.py`) that handles **path management**, **logging**, and **data loading**. Below is a guide on how to use these functions in your scripts.

### **1. Automatic Setup**

Just by importing `utils`, the script automatically checks and creates the necessary folder structure (`datasets/`, `models/`, `hyperparams/`) and the log file.

```python
import utils  # This triggers the directory creation logic immediately
```

### **2. Using the Logger**

Instead of using standard `print()` statements, use the pre-configured logger. It writes to both the console and `logs.log` simultaneously.

```python
from utils import logger

logger.info("Starting the training process...")
logger.warning("Dataset contains NaN values, dropping them.")
logger.error("Model training failed due to memory error.")
```

### **3. Loading Datasets**

The module includes optimized loaders for the three major datasets. These functions handle file searching (globbing), concatenation, and cleaning automatically.

#### **A. CICIDS2017**

Loads all CSV files found in `datasets/cicids2017/`.

* **Usage:**

    ```python
    from utils import get_cicids2017

    # Returns a single combined DataFrame of all CSVs
    df_cic = get_cicids2017()

    print(df_cic.shape)
    ```

#### **B. ToN-IoT**

Recursively searches for `train_test_network.csv` inside `datasets/ton_iot/`.

* **Usage:**

    ```python
    from utils import get_toniot

    # Loads the specific Network dataset file
    df_ton = get_toniot()
    ```

#### **C. UNSW-NB15**

Automatically finds the *Training* and *Testing* set files, merges them into one DataFrame, and cleans the headers.

* **Usage:**

    ```python
    from utils import get_unsw_nb15

    # Returns combined Train + Test set
    df_unsw = get_unsw_nb15()
    ```

-----

## **📊 Supported Datasets**

| Dataset | Description | Target Use Case |
| :--- | :--- | :--- |
| **TON-IOT** | Telemetry and Network data from IoT devices. | IoT Security, Heterogeneous Networks |
| **UNSW-NB15** | Comprehensive modern attack dataset. | General Network Intrusion Detection |
| **CICIDS2017** | Real-world PCAPs converted to CSV flows. | Traffic Analysis, Flow Classification |

-----

## **🧠 Models**

The project implements and benchmarks the following AI models:

* **Random Forest:** Robust baseline for classification.
* **XGBoost / LGBM / CatBoost:** Gradient boosting machines for high performance on tabular data.
* **SVC:** C-Support Vector Classification model for margin-based classification.
* **MLP (Multi-Layer Perceptron):** Standard dense neural networks.
* **CNN-LSTM:** A hybrid model using Convolutional layers for feature extraction and LSTM for temporal/sequence dependency analysis.
