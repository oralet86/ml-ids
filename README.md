# **ml-ids**

This is a repository for a school project dedicated to the comparative analysis of machine learning and deep learning models for Network Intrusion Detection Systems (IDS).

## Comparative Analysis & Machine Learning for Network Intrusion Detection Systems (NIDS)

The project aims to benchmark standard algorithms against major datasets (CICIDS2017, UNSW-NB15, ToN-IoT) and presents a novel architecture to improve traffic classification performance.

---

## **📌 Project Scope**

This project focuses on:

* Binary intrusion detection (benign vs attack)
* Consistent preprocessing across datasets
* Fair comparison between:

  * Traditional ML models (tree-based, linear, etc.)
  * Deep Learning models (MLP, CNN, RNN-style, Transformer-ready)
* Reproducible experimentation with:

  * Fixed random seeds
  * Stratified splits
  * Centralized hyperparameters
* Dataset-agnostic evaluation metrics

The framework currently supports **three major IDS datasets**:

* **CICIDS2017**
* **UNSW-NB15**
* **TON_IoT**

---

## **📂 Repository Structure**

```text
ml-ids/
├── datasets/                  # (Expected) raw dataset directory
│   ├── cicids2017/
│   ├── unsw_nb15/
│   └── ton_iot/
│
├── cicids2017.py              # CICIDS dataset loading & preprocessing
├── cicids_ml.py               # CICIDS – traditional ML pipeline
├── cicids_dl.py               # CICIDS – deep learning pipeline
│
├── unsw_nb15.py               # UNSW-NB15 loading & preprocessing
├── unsw_nb15_ml.py            # UNSW-NB15 – traditional ML pipeline
├── unsw_nb15_dl.py            # UNSW-NB15 – deep learning pipeline
│
├── ton_iot.py                 # TON_IoT loading & preprocessing
├── ton_iot_ml.py              # TON_IoT – traditional ML pipeline
├── ton_iot_dl.py              # TON_IoT – deep learning pipeline
│
├── traditional_models.py      # Classical ML model definitions
├── deep_models.py             # Deep learning model definitions
│
├── utils.py                   # Shared utilities (logging, metrics, IO)
├── params.py                  # Centralized experiment configuration
│
├── requirements.txt
├── logs.log                   # Runtime logs
└── README.md
```

---

## **🧠 Model Categories**

### **Traditional Machine Learning**

Implemented in `traditional_models.py`, including:

* Tree-based models
* Ensemble models
* Linear classifiers

Used via:

* `*_ml.py` scripts

Key features:

* SMOTE support
* Feature scaling and imputation
* Feature selection (where applicable)

---

### **Deep Learning**

Implemented in `deep_models.py`, used by:

* `*_dl.py` scripts

Characteristics:

* PyTorch-based
* Raw logits returned (no implicit sigmoid/softmax)
* Explicit training loops
* Early stopping support
* GPU acceleration (CUDA / TF32 enabled where available)

Sequence handling:

* Current DL pipelines support **tabular inputs**
* Sequence length can be explicitly controlled (e.g., `seq_len=1`)
* Ready for extension to temporal models (RNN / Transformer)

---

## **📊 Datasets**

### **CICIDS2017**

* Multiple CSV files (traffic by day)
* Large-scale, highly imbalanced
* Custom train/validation/test splitting logic

### **UNSW-NB15**

* Predefined features
* Clean binary labeling
* Stratified holdout supported

### **TON_IoT**

* IoT-focused traffic
* High-dimensional tabular data
* Explicit ML and DL pipelines

Each dataset has:

* A dedicated loader
* Dataset-specific preprocessing
* Unified output format for downstream models

---

## **⚙️ Configuration (`params.py`)**

All global experiment settings are defined in **one place**, including:

* Random seeds
* Train/validation split ratios
* Batch size
* Epoch count
* Early stopping patience
* SMOTE parameters
* Dataset usage percentage (for fast experiments)

This ensures:

* Reproducibility
* Easy ablation studies
* Consistent comparisons across datasets

---

## **📈 Metrics & Evaluation**

All models are evaluated using **binary classification metrics**:

* Accuracy
* Precision
* Recall
* F1-score
* Training time

Metrics are:

* Computed consistently across ML and DL
* Logged via `utils.py`
* Saved for later analysis

---

## **🛠 Utilities (`utils.py`)**

The utility module provides:

* Dataset path resolution
* Centralized logging
* Metrics computation
* Stratified splitting helpers
* Result saving utilities

Importing `utils.py` ensures:

* Required directories exist
* Logging is initialized consistently

---

## **🚀 Running Experiments**

Examples:

```bash
# CICIDS – traditional ML
python cicids_ml.py

# CICIDS – deep learning
python cicids_dl.py

# UNSW-NB15 – ML
python unsw_nb15_ml.py

# TON_IoT – DL
python ton_iot_dl.py
```

Each script is **self-contained** and dataset-specific.

---

## **📦 Installation**

```bash
pip install -r requirements.txt
```

Python 3.13.7 recommended.

---

## **🧭 How to Use & Extend the Framework**

This section describes how to **run experiments**, **add new models**, and **tune hyperparameters** within the existing architecture.

The framework is intentionally explicit: there is **no hidden automation**, and each dataset–model combination is controlled by a single entry script.

---

## **Running an Experiment**

Each dataset has two entry points:

| Task           | Script    |
| -------------- | --------- |
| Traditional ML | `*_ml.py` |
| Deep Learning  | `*_dl.py` |

Example:

```bash
python cicids_ml.py
python unsw_nb15_dl.py
```

What happens internally:

1. Dataset is loaded and preprocessed
2. Train/validation/test splits are created
3. Models defined in the model registry are iterated
4. Each model is trained and evaluated
5. Metrics are logged and saved

There is **no global runner** by design; this avoids implicit coupling between datasets.

---

## **🎛 Hyperparameter Tuning**

### **Global Hyperparameters**

**Location**

```text
params.py
```

Common tuning knobs include:

| Parameter                | Purpose             |
| ------------------------ | ------------------- |
| `BATCH_SIZE`             | DL batch size       |
| `EPOCHS`                 | Max training epochs |
| `PATIENCE`               | Early stopping      |
| `RANDOM_STATE`           | Reproducibility     |
| `VAL_FRAC` / `TEST_SIZE` | Split ratios        |
| `*_DATA_PCT`             | Dataset subsampling |

Changing values here affects **all experiments consistently**.

---

### **Model-Specific Hyperparameters**

Model-specific parameters should be:

* Passed through the model constructor
* Defined explicitly in the dataset script

Example:

```python
model = MyClassifier(
    n_estimators=200,
    max_depth=12,
)
```

> There is currently **no automatic grid search**.

---

## **🧪 Dataset Subsampling (Fast Experiments)**

Each dataset supports partial usage for quick iteration:

```python
CICIDS_DATA_PCT = 0.05
UNSW_DATA_PCT = 0.1
TON_IOT_DATA_PCT = 0.2
```

This happens **before splitting**, ensuring label ratios remain meaningful.

---

## **📐 Preprocessing & Feature Handling**

Preprocessing is dataset-specific and located in:

```text
cicids2017.py
unsw_nb15.py
ton_iot.py
```

Common steps include:

* NaN / Inf removal
* Duplicate removal
* Feature scaling
* Label normalization

### **Important Rules**

* Do **not** add preprocessing inside models
* Do **not** change label semantics in training scripts
* Feature count consistency is required for DL models

---

## **⏱ Training Loop (DL)**

Deep learning scripts (`*_dl.py`) handle:

* TensorDataset creation
* DataLoader construction
* Loss computation (`BCEWithLogitsLoss`)
* Optimizer setup
* Early stopping
* Metric aggregation

If modifying training behavior, do so **in the dataset DL script**, not the model.

---

## **📊 Results & Logging**

* Logs are written via `logger` from `utils.py`
* Metrics are saved per model and trial
* Training time is tracked explicitly

Results are structured for:

* CSV extraction
* Later aggregation
* External plotting

Here is a **clean, professional contributors section** you can append to the **very end** of the README.
It matches software documentation tone and avoids academic fluff.

---

## **👥 Contributors**

This project is developed and maintained by:

* **Mert Fıçıcı**
* **Efekan Çelik**
* **Cemre Sude Akdağ**

Contributions include dataset engineering, model implementation, experimental design, and evaluation infrastructure.
