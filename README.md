# Oil Spill Risk Classifier

> Predictive Risk Classification of Oil Spill Severity in the Niger Delta  
> Using Machine Learning and Explainable Artificial Intelligence (SHAP)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-latest-red.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Abstract

Oil spills ruin traditional livelihoods and cause untold damage to the environment across the Niger
Delta. Several hundred oil spills are recorded in Nigeria each year, exacerbated by organised oil
theft and artisanal refining. Despite this, operational responses by regulatory bodies remain
reactive, lacking predictive severity tooling. This applied data science project develops an
end-to-end multi-class predictive risk classifier leveraging historical NOSDRA monitoring data.
Following the CRISP-DM framework, raw incident records were sanitised, and an 80/20 stratified
split was applied to preserve class distribution. Candidate classifiers including a Dummy
baseline, Random Forest, and XGBoost were evaluated using the Weighted F1 Score. To demystify
the model, Explainable AI (XAI) was integrated using SHAP feature attribution maps to extract
business-level insights and actionable policy recommendations.

---

## Problem Statement

The primary problem is that oil spill response prioritisation in the Niger Delta is currently
reactive, no predictive severity tooling exists at the time of incident reporting. This project
proposes a multi-class classifier predicting severity (Major / Medium / Minor) from incident
attributes available at the point of reporting (cause, company, location, habitat), enabling
proactive resource allocation for agencies such as NDDC and NOSDRA.

**Domain:** Applied Data Science & Explainable Artificial Intelligence (XAI)  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Academic Context:** SIWES Industrial Training 2026 — University of Uyo, Nigeria

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

```
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

> ⚠️ **Severe class imbalance** — Minor incidents dominate the dataset by a significant margin.
> This directly impacts classifier performance on the minority classes (see Results).

---

## Features Used in Model

After preprocessing, 34 columns were excluded due to high missingness (>20%), non-predictive
identifiers (IDs, certificate numbers), or free-text descriptions. The 9 features retained:

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
| SHAP | Explainability layer (TreeExplainer) |
| Matplotlib / Seaborn | Visualisation (all figures at 300 DPI) |
| Google Colab + Drive | Development and persistence environment |

---

## Project Structure

```
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
│   ├── 04_modeling.ipynb              ← Phase 5: Modelling & Evaluation
│   └── 04_explainability.ipynb        ← Phase 6: SHAP XAI + Report Generation
│
├── src/                               ← Reusable Python modules
│   ├── __init__.py
│   ├── preprocess.py                  ← Cleaning, encoding, severity engineering
│   ├── model.py                       ← Training, evaluation, cross-validation
│   └── explain.py                     ← SHAP wrappers and plot helpers
│
├── reports/
│   └── figures/                       ← All saved plots (300 DPI)
│       ├── 01_top_12_companies.png    ← Top 12 operators by spill count
│       ├── 02_cause_vs_severity.png   ← Top causes cross-tabulated with severity
│       ├── 03_severity_distribution.png ← Class imbalance bar + pie charts
│       ├── 04_spills_per_year.png     ← Temporal trend 2006–2025
│       ├── 05_jiv_status.png          ← JIV status vs severity
│       ├── shap_global_bar.png        ← Global SHAP feature importance
│       ├── shap_beeswarm_*.png        ← Per-class beeswarm plots
│       └── shap_force_plot_*.png      ← Local force plots (correct + error instances)
│
├── data/                              ← NOT tracked by Git
│   ├── raw/                           ← Place NOSDRA CSV here
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
| 4 | `04_modeling.ipynb` | Phase 5 — Modelling | Trained XGBoost model, metrics |
| 5 | `04_explainability.ipynb` | Phase 6 — XAI + Report | SHAP plots, IEEE report draft |

---

## Setup & Usage

### Google Colab (Recommended — no local install needed)

1. Download the NOSDRA CSV from the source link above
2. Upload it to your Google Drive
3. Open [Google Colab](https://colab.research.google.com) and load each notebook via **File → Upload Notebook**
4. Update the `file_path` variable in each notebook to match your Drive path
5. Run all cells top-to-bottom with `Shift + Enter`

### Local Environment

```bash
git clone https://github.com/shalombebx/oil-spill-risk-classifier.git
cd oil-spill-risk-classifier
pip install -r requirements.txt
jupyter notebook
```

---

## Results

### Model Comparison

| Model | Weighted F1 | Notes |
|---|---|---|
| Dummy Classifier (Baseline) | 0.9408 | Majority-class prediction only |
| Random Forest | 0.9418 | class_weight='balanced' |
| **XGBoost (Selected)** | **0.9426** | GridSearchCV tuned |

**Best XGBoost Hyperparameters:** `learning_rate=0.1`, `max_depth=7`, `n_estimators=100`

### Per-Class Performance (XGBoost on Test Set)

| Severity Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Major | 0.00 | 0.00 | 0.00 | 10 |
| Medium | 0.33 | 0.03 | 0.06 | 95 |
| **Minor** | **0.96** | **1.00** | **0.98** | 2,536 |
| **Weighted Avg** | **0.94** | **0.96** | **0.94** | 2,641 |

> ⚠️ **Critical Interpretation:** The high Weighted F1 of 0.9426 is driven almost entirely by the
> Minor class (support = 2,536). The model currently fails to identify Major spills (F1 = 0.00)
> and struggles significantly with Medium (F1 = 0.06) due to the extreme class imbalance. This is
> an acknowledged limitation of the current iteration, see Future Work below.

---

## SHAP Explainability Findings

### Top Feature Drivers (Global Importance)

1. **`cause`** — Incidents attributed to Sabotage/Theft and Equipment Failure are the strongest
   predictors of Major severity classifications, as identified in the global SHAP bar chart.

2. **`spillareahabitat`:** Open sea and sensitive inland swamp environments disproportionately
   push the model toward Major predictions, reflecting the lower volume thresholds for inland
   habitats in NOSDRA's severity definition.

3. **`company`:** Certain operating companies carry an inherent risk baseline. Historical
   data from operators such as NAOC and SPDC shows higher-volume incident patterns,
   causing the model to assign higher severity probabilities to these operators.

4. **`incident_year`:** More recent years (post-2020) act as a mitigating factor in local
   force plots, likely reflecting improved containment technologies or shifts in reporting behaviour.

### Policy Recommendations

Based on SHAP feature attribution maps, the following recommendations are made for NOSDRA and NDDC:

1. **Risk-Based JIV Prioritisation:** Integrate the model into the incident reporting pipeline.
   When a report involves high-risk causes (Sabotage) in sensitive habitats (inland swamps),
   automatically flag the incident as probable Major and prioritise immediate JIV deployment
   before the official barrel volume is assessed.

2. **Targeted Operator Audits:** Transition from random compliance checks to targeted operational
   audits for high-risk operators identified by SHAP company attribution scores.

3. **Enhanced Habitat Monitoring:** Increase proactive surveillance (drone patrols, community
   monitoring) specifically in geographical zones historically linked to Major classifications
   based on `spillareahabitat` feature importance.

---

## EDA Key Findings

| Research Question | Finding |
|---|---|
| Which companies report the most spills? | NAOC and SPDC account for a disproportionate majority; OANDO and RENAISSANCE also feature prominently |
| What are the primary causes? | Sabotage/oil theft and equipment failure dominate; Sabotage also correlates most strongly with Major severity |
| What is the severity distribution? | Severe imbalance: Minor spills (96%) overwhelm Medium (3.6%) and Major (0.4%) |
| How has frequency trended over time? | Notable spike in 2013–2014; Major proportion remains consistently low year-over-year |
| Does JIV status correlate with severity? | Majority of categorised spills received a formal JIV, consistent with NOSDRA protocols |

---

## Limitations

1. **Extreme Class Imbalance:** The current XGBoost model effectively predicts Minor-only on
   imbalanced data. Future iterations must apply SMOTE, `scale_pos_weight`, or cost-sensitive
   learning to improve Recall for Major and Medium classes.

2. **Engineered Target:** The severity label is derived from barrel volume thresholds, not
   an independently assessed ground truth. If `estimatedquantity` itself is inaccurate, the
   target variable inherits that noise.

3. **Data Latency & Sparsity:** Pre-2006 records are excluded due to known publication gaps.
   Active-review and very-recent incidents are also excluded, introducing a systematic recency
   bias.

4. **Missing High-Value Features:** `lga`, `typeoffacility`, and `jivpresent` were excluded
   due to >20% missing values. These columns may contain strong predictive signal and warrant
   future imputation strategies.

---

## Future Work

- Apply **SMOTE** or XGBoost `scale_pos_weight` to address class imbalance and improve
  Major/Medium recall
- Deploy the serialised XGBoost model and encoders as a **real-time REST API** using FastAPI
- Build a **Streamlit dashboard** for non-technical regulatory users
- Integrate **geospatial features** using the excluded `latitude` and `longitude` columns
- Explore **deep learning** comparisons (TabNet, LightGBM)

---

## Author

**Shalom Bebebaraseigha** 
Computer Science — University of Uyo | '023 
Matric No.: 23/SC/CO/010  
SIWES Industrial Training: NDDC, Bayelsa State, 2026  
GitHub: [@shalombebx](https://github.com/shalombebx)

---

## Acknowledgements

- Dataset: [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng)
- Explainability: [SHAP — Lundberg et al.](https://github.com/slundberg/shap)
- Methodology: CRISP-DM framework
- Development: Google Colab + Google Drive
