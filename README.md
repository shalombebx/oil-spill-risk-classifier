# Oil Spill Risk Classifier

> Predictive Risk Classification of Oil Spill Severity in the Niger Delta  
> Using Machine Learning and Explainable Artificial Intelligence (SHAP)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Overview

This project builds a **multi-class machine learning classifier** that predicts oil spill severity
(Major, Medium, or Minor) from incident attributes recorded by NOSDRA (National Oil Spill Detection
and Response Agency). An **Explainable AI (XAI)** layer using SHAP values translates model decisions
into actionable policy insights for environmental regulators.

**Domain:** Applied Data Science & Explainable Artificial Intelligence  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Academic Context:** SIWES Industrial Training 2026 — University of Uyo, Nigeria

---

## Problem Statement

Oil spill response prioritisation in the Niger Delta is largely reactive. This project provides a
predictive tool that classifies incident severity from attributes available at the time of reporting —
enabling proactive, data-driven resource allocation by agencies such as NDDC and NOSDRA.

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng/oilspillmonitor.html) |
| **Download** | Click "Download complete dataset as CSV" on the source page |
| **Target Variable** | `Category` — Major / Medium / Minor |
| **Key Features** | Cause, company, state, LGA, spill location type, JIV status, year, month |

> **Note:** The raw dataset is not included in this repository.  
> Download it directly from the NOSDRA portal and place it in `data/raw/`.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| pandas | 2.0+ | Data manipulation and analysis |
| NumPy | 1.24+ | Numerical computing |
| scikit-learn | 1.3+ | Preprocessing, Random Forest, evaluation |
| XGBoost | 2.0+ | Gradient boosting classifier |
| SHAP | 0.44+ | Explainability layer (XAI) |
| Matplotlib | 3.7+ | Visualisation |
| Seaborn | 0.12+ | Statistical plots |
| Google Colab | — | Primary development environment |

---

## Project Structure

```
oil-spill-risk-classifier/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
├── .gitignore                       ← Files excluded from Git
│
├── notebooks/                       ← One notebook per project phase
│   ├── 01_eda.ipynb                 ← Phase 3: Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       ← Phase 4: Data Preparation
│   ├── 03_modeling.ipynb            ← Phase 5: Model Training & Evaluation
│   └── 04_explainability.ipynb      ← Phase 6: SHAP Explainability
│
├── src/                             ← Reusable Python modules (refactored from notebooks)
│   ├── __init__.py
│   ├── preprocess.py                ← Data cleaning and encoding functions
│   ├── model.py                     ← Training, evaluation, and persistence
│   └── explain.py                   ← SHAP wrappers and plot helpers
│
├── reports/
│   └── figures/                     ← All saved plots and charts
│       ├── 01_severity.png
│       ├── 02_causes.png
│       ├── 03_companies.png
│       ├── 04_time.png
│       ├── 05_confusion_matrix.png
│       ├── 06_shap_global.png
│       ├── 07_shap_class.png
│       └── 08_shap_sample.png
│
├── data/                            ← NOT tracked by Git (see .gitignore)
│   ├── raw/                         ← Original NOSDRA CSV — never edit
│   └── processed/                   ← Cleaned data after preprocessing
│
└── models/                          ← NOT tracked by Git (generated locally)
    ├── xgb_oil_spill.pkl
    ├── rf_oil_spill.pkl
    ├── label_encoders.pkl
    └── target_encoder.pkl
```

---

## Setup & Usage

### Option A — Google Colab (Recommended while on placement)

1. Download the NOSDRA dataset from the link above
2. Open [Google Colab](https://colab.research.google.com)
3. Upload `oil_spill_starter_notebook.ipynb` via **File → Upload Notebook**
4. Run all cells top-to-bottom with `Shift + Enter`

### Option B — Local Environment

```bash
# Clone the repository
git clone https://github.com/shalombebx/oil-spill-risk-classifier.git
cd oil-spill-risk-classifier

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Notebook Execution Order

Run notebooks strictly in sequence — each notebook depends on the outputs of the previous one.

| Order | Notebook | Phase | Description |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Phase 3 | Load data, inspect columns, exploratory analysis |
| 2 | `02_preprocessing.ipynb` | Phase 4 | Clean, encode, feature engineer, split |
| 3 | `03_modeling.ipynb` | Phase 5 | Train models, cross-validate, evaluate |
| 4 | `04_explainability.ipynb` | Phase 6 | SHAP global and local explanations |

---

## Key Results

> _To be updated after model training is complete_

| Model | Weighted F1-Score | Cross-Val F1 (5-Fold) |
|---|---|---|
| Dummy Baseline | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

**Top SHAP Features:** _To be updated after explainability phase_

---

## Methodology

This project follows the **CRISP-DM** framework:

```
Business Understanding (Phase 1)
        ↓
Data Understanding (Phase 2)
        ↓
Data Preparation (Phase 3–4)
        ↓
Modelling (Phase 5)
        ↓
Evaluation (Phase 5–6)
        ↓
Deployment / Reporting (Phase 7–9)
```

---

## Repository Notes

- `data/` and `models/` are excluded from Git via `.gitignore` — download/generate locally
- All saved figures are committed to `reports/figures/` so visualisations are visible on GitHub
- Notebooks are committed with outputs cleared to keep file sizes small

---

## Author

**Shalom Bebebaraseigha** (Shards)  
Computer Science — University of Uyo | Class of 2027  
SIWES Industrial Training: NDDC, Bayelsa State, 2026  
GitHub: [@shalombebx](https://github.com/shalombebx)

---

## Acknowledgements

- Dataset: [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng)
- Explainability: [SHAP by Scott Lundberg et al.](https://github.com/slundberg/shap)
- Methodology: CRISP-DM framework
