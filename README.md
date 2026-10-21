# Oil Spill Risk Classifier

> Predictive Risk Classification of Oil Spill Severity in the Niger Delta  
> Using Machine Learning and Explainable Artificial Intelligence (SHAP)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-blueviolet.svg)
![SHAP](https://img.shields.io/badge/SHAP-latest-red.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Abstract

Oil spills ruin traditional livelihoods and cause untold damage to the environment across the Niger Delta. Several hundred oil spills are recorded in Nigeria each year, exacerbated by organised oil theft and artisanal refining. Despite this, operational responses by regulatory bodies remain reactive, lacking predictive severity tooling. This applied data science project develops an end-to-end multi-class predictive risk classifier leveraging historical NOSDRA monitoring data. Following the CRISP-DM framework, raw incident records were sanitised, and an 80/20 stratified split was applied to preserve class distribution. 

To overcome severe target class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was applied dynamically to the training space. Candidate classifiers including a Dummy baseline, Random Forest, and a SMOTE-Optimized XGBoost were evaluated. To demystify the model, Explainable AI (XAI) was integrated using SHAP feature attribution maps to extract business-level insights and actionable policy recommendations.

---

## Problem Statement

The primary problem is that oil spill response prioritisation in the Niger Delta is currently reactive; no predictive severity tooling exists at the time of incident reporting. This project proposes a multi-class classifier predicting severity (Major / Medium / Minor) from incident attributes available at the point of reporting (cause, company, location, habitat), enabling proactive resource allocation for agencies such as NDDC and NOSDRA.

**Domain:** Applied Data Science & Explainable Artificial Intelligence (XAI)  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Academic Context:** SIWES Industrial Training — University of Uyo, Nigeria

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng/oilspillmonitor.html) |
| **File** | `nosdra_2026-08-23_15_32_02UTC_complete.csv` |
| **Raw Size** | 21,107 rows × 42 columns |
| **Usable Records** | 13,202 rows (after Unknown severity removal) |
| **Final Model Dataset** | 13,202 rows × 10 columns (9 features + 1 target) |
| **Train / Test Split** | 10,561 train / 2,641 test (80/20 stratified) |

> **Note:** The raw dataset is not included in this repository.  
> Download it directly from the NOSDRA portal and place it in `data/raw/`.

### Known Data Limitations

1. **Temporal Sparsity:** Published data does not reliably include spills prior to 2006.
2. **Review Status:** Spills currently under active NOSDRA review are excluded from the download.
3. **Data Latency:** The most recent incidents are omitted due to reporting latency.

---

## Target Variable — Severity Engineering

> **Important:** The raw NOSDRA dataset does not contain a pre-labelled severity column.
> The target variable (`severity`) was engineered using official NOSDRA barrel volume
> thresholds combined with habitat type; a key methodological contribution of this project.

```text
Inland water habitats:
  Major  → > 250 barrels spilled
  Medium → 25 – 250 barrels spilled
  Minor  → < 25 barrels spilled

Land / Swamp / Shoreline / Open Sea:
  Major  → > 2,500 barrels spilled
  Medium → 250 – 2,500 barrels spilled
  Minor  → < 250 barrels spilled
```

### Severity Distribution (After Engineering)

| Class | Count | Share | Test Support |
|---|---|---|---|
| **Minor** | 12,678 | ~96.0% | 2,536 |
| **Medium** | 472 | ~3.6% | 95 |
| **Major** | 52 | ~0.4% | 10 |

> ⚠️ **Severe class imbalance** — Minor incidents dominate the dataset by a significant margin. SMOTE was actively applied during the Phase 5 modelling stage to synthesize artificial minority records, preventing the classifier from collapsing into majority-class predictions.

---

## Features Used in Model

After preprocessing, 34 columns were excluded due to high missingness (>20%), non-predictive identifiers (IDs, certificate numbers), or free-text descriptions. The 9 features retained:

| Feature | Type | Description |
|---|---|---|
| `company` | Categorical | Operating company responsible for the spill |
| `cause` | Categorical | Reported cause (e.g., Sabotage, Equipment Failure) |
| `statesaffected` | Categorical | Nigerian state where spill occurred |
| `spillareahabitat` | Categorical | Environmental type (inland, swamp, open sea, etc.) |
| `incident_year` | Numeric | Year extracted from `incidentdate` |
| `incident_month` | Numeric | Month extracted from `incidentdate` |
| *(+ 3 others)* | Mixed | Remaining columns not excluded by business logic |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | Preprocessing, Random Forest, evaluation |
| XGBoost | Gradient boosting classifier (GridSearchCV tuned) |
| imbalanced-learn | Applying SMOTE for synthetic minority oversampling |
| SHAP | Explainability layer (TreeExplainer) |
| Matplotlib / Seaborn | Visualisation (all figures at 300 DPI) |
| Google Colab + Drive | Development and persistence environment |

---

## Project Structure

```text
oil-spill-risk-classifier/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies (exact versions)
├── .gitignore                         ← Excludes data/, models/, __pycache__/
│
├── notebooks/                         ← Executed notebooks (one per phase)
│   ├── 01_eda.ipynb                   ← Phase 2: Data Understanding
│   ├── 02_eda_part_b.ipynb            ← Phase 3: Exploratory Data Analysis
│   ├── 03_preprocessing.ipynb         ← Phase 4: Data Preparation
│   ├── 04_modeling.ipynb              ← Phase 5: Modelling, SMOTE & Evaluation
│   ├── 05_explainability.ipynb        ← Phase 6: SHAP XAI
│
├── src/                               ← Reusable Python modules
│   ├── __init__.py
│   ├── preprocess.py                  ← Cleaning, encoding, severity engineering
│   ├── model.py                       ← Training, SMOTE logic, cross-validation
│   └── explain.py                     ← SHAP wrappers and plot helpers
│
├── reports/
│   └── figures/                       ← All saved plots (300 DPI)
│       ├── 01_top_12_companies.png    
│       ├── 02_cause_vs_severity.png   
│       ├── 03_severity_distribution.png 
│       ├── 04_spills_per_year.png     
│       ├── 05_jiv_status.png          
│       ├── shap_global_bar.png        
│       ├── shap_beeswarm_0.png        
│       ├── shap_beeswarm_1.png
│       ├── shap_beeswarm_2.png
│       ├── shap_force_plot_idx0_correct.png 
│       ├── shap_force_plot_idx1_correct.png 
│       └── shap_force_plot_idx21_error.png  
│
├── data/                              ← NOT tracked by Git
│   ├── raw/                           ← Place nosdra_2026-08-23_15_32_02UTC_complete.csv here
│   └── processed/                     ← X_train, X_test, y_train, y_test CSVs
│
└── models/                            ← NOT tracked by Git
    ├── xgboost_model.pkl
    ├── target_encoder.pkl
    └── label_encoders.pkl
```

---

## Notebook Execution Order

Run notebooks strictly in sequence — each one depends on outputs from the previous.

| Order | Notebook | Phase | Key Output |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Phase 2 — Data Understanding | Data dictionary, quality report |
| 2 | `02_eda_part_b.ipynb` | Phase 3 — EDA | 5 figures, research question answers |
| 3 | `03_preprocessing.ipynb` | Phase 4 — Preprocessing | X_train/X_test CSVs, encoders |
| 4 | `04_modeling.ipynb` | Phase 5 — Modelling | Trained SMOTE XGBoost model, metrics |
| 5 | `05_explainability.ipynb` | Phase 6 — XAI + Report | SHAP plots, IEEE report draft |

---

## Results

### Model Comparison

| Model | Weighted F1 | Notes |
|---|---|---|
| Dummy Classifier (Baseline) | 0.9408 | Majority-class prediction only |
| Random Forest | 0.9418 | class_weight='balanced' |
| XGBoost (Base) | 0.9426 | Suffered from ~0.00 recall on minority classes |
| **XGBoost (SMOTE-Optimized)** | **0.8984** | **Synthetic oversampling restored minority recall** |

**Best XGBoost Hyperparameters:** `learning_rate=0.1`, `max_depth=7`, `n_estimators=100`

### Per-Class Performance (SMOTE-Optimized XGBoost on Test Set)

| Severity Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Major** | 0.12 | 0.20 | 0.15 | 10 |
| **Medium** | 0.12 | 0.40 | 0.18 | 95 |
| **Minor** | 0.97 | 0.89 | 0.93 | 2,536 |
| **Weighted Avg** | **0.94** | **0.87** | **0.90** | 2,641 |

> ⚠️ **Critical Interpretation:** While overall accuracy dropped to 87% (Weighted F1: 0.8984) compared to the base model, applying SMOTE successfully broke the model's absolute reliance on the 96% majority class. The model now effectively catches 40% of Medium spills and 20% of Major spills, whereas the base model missed almost all of them. The corresponding drop in precision indicates a higher rate of false positives—aggressively flagging lower-severity spills as high-risk. In environmental risk triage, triggering a Joint Investigation Visit (JIV) unnecessarily is vastly preferable to systematically waving through a catastrophic spill as routine.

---

## SHAP Explainability Findings

### Top Feature Drivers (Global Importance)

1. **`cause`** — Incidents attributed to Sabotage/Theft and Equipment Failure are the strongest predictors of Major severity classifications, as identified in the global SHAP bar chart.
2. **`spillareahabitat`:** Open sea and sensitive inland swamp environments disproportionately push the model toward Major predictions, reflecting the lower volume thresholds for inland habitats in NOSDRA's severity definition.
3. **`company`:** Certain operating companies carry an inherent risk baseline. Historical data from operators such as NAOC and SPDC shows higher-volume incident patterns, causing the model to assign higher severity probabilities to these operators.
4. **`incident_year`:** More recent years (post-2020) act as a mitigating factor in local force plots, likely reflecting improved containment technologies or shifts in reporting behaviour.

### Policy Recommendations

Based on SHAP feature attribution maps, the following recommendations are made for NOSDRA and NDDC:

1. **Risk-Based JIV Prioritisation:** Integrate the model into the incident reporting pipeline. When a report involves high-risk causes (Sabotage) in sensitive habitats (inland swamps), automatically flag the incident as probable Major and prioritise immediate JIV deployment before the official barrel volume is assessed.
2. **Targeted Operator Audits:** Transition from random compliance checks to targeted operational audits for high-risk operators identified by SHAP company attribution scores.
3. **Enhanced Habitat Monitoring:** Increase proactive surveillance (drone patrols, community monitoring) specifically in geographical zones historically linked to Major classifications based on `spillareahabitat` feature importance.

---

## Limitations

1. **Precision Trade-off:** While SMOTE effectively restored the model's ability to recall rare Major and Medium spills, it inherently reduced class precision, producing false positives that may trigger unnecessary resource allocations. 
2. **Engineered Target:** The severity label is derived from barrel volume thresholds, not an independently assessed ground truth. If `estimatedquantity` itself is inaccurate, the target variable inherits that noise.
3. **Data Latency & Sparsity:** Pre-2006 records are excluded due to known publication gaps. Active-review and very-recent incidents are also excluded, introducing a systematic recency bias.
4. **Missing High-Value Features:** `lga`, `typeoffacility`, and `jivpresent` were excluded due to >20% missing values. These columns may contain strong predictive signal and warrant future imputation strategies.

---

## Future Work

- Deploy the serialised SMOTE-Optimized XGBoost model and encoders as a **real-time REST API** using FastAPI
- Build a **Streamlit dashboard** for non-technical regulatory users
- Integrate **geospatial features** using the excluded `latitude` and `longitude` columns
- Explore **deep learning** comparisons (TabNet, LightGBM) to capture more complex feature interactions in highly skewed data spaces

---

## Author

**Shalom "Vantage" Bolo Bebebaraseigha**  
Computer Science — University of Uyo  
Matric No.: 23/SC/CO/010  
SIWES Industrial Training: NDDC, Bayelsa State (Completed September 2026)  
GitHub: [@shalombebx](https://github.com/shalombebx)

---

## Acknowledgements

- Dataset: [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng)
- Explainability: [SHAP — Lundberg et al.](https://github.com/slundberg/shap)
- Methodology: CRISP-DM framework
- Development: Google Colab + Google Drive
