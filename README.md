# Oil Spill Risk Classifier

> Predictive Risk Classification of Oil Spill Severity in the Niger Delta  
> Using Machine Learning and Explainable Artificial Intelligence (SHAP)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTENC-blueviolet.svg)
![SHAP](https://img.shields.io/badge/SHAP-latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Abstract

Oil spills damage livelihoods and the environment across the Niger Delta. Several hundred spills are recorded in Nigeria each year, driven partly by organised oil theft and artisanal refining, yet regulatory response remains reactive and no predictive severity tooling exists at the point of reporting. This applied data science project builds a multi-class severity classifier (Major / Medium / Minor) from historical NOSDRA incident records, following the CRISP-DM framework. Raw records were cleaned, a severity target was engineered from official barrel-volume thresholds, and an 80/20 stratified split was applied.

The target is severely imbalanced (about 96% Minor). Three XGBoost configurations were compared: an unweighted baseline, a SMOTENC-oversampled model, and a class-weighted model. Only class weighting meaningfully recovered minority-class recall (Medium 0.40, Major 0.20, versus 0.03 and 0.00 for the baseline), at a large cost in precision. SMOTENC gave a marginal gain only. SHAP is used to explain the final model's predictions.

---

## Problem Statement

Oil spill response prioritisation in the Niger Delta is currently reactive. This project proposes a multi-class classifier that predicts severity from attributes available at the point of reporting (cause, company, location, habitat), to support triage and resource allocation by agencies such as NDDC and NOSDRA.

**Domain:** Applied Data Science & Explainable AI  
**Methodology:** CRISP-DM  
**Academic Context:** SIWES Industrial Training, University of Uyo, Nigeria

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng/oilspillmonitor.html) |
| **File** | `nosdra_2026-08-23_15_32_02UTC_complete.csv` |
| **Raw size** | 21,107 rows x 42 columns |
| **Usable records** | 13,202 rows (after removing Unknown severity) |
| **Model dataset** | 13,202 rows x 10 columns (9 features + 1 target) |
| **Train / test** | 10,561 / 2,641 (80/20 stratified, `random_state=42`) |

> The raw dataset is not included in this repository. Download it from the NOSDRA portal and place it in `data/raw/`.

### Known data limitations

1. **Temporal sparsity:** published data does not reliably include spills before 2006.
2. **Review status:** spills under active NOSDRA review are excluded from the download.
3. **Data latency:** the most recent incidents are omitted due to reporting lag.

---

## Target Variable: Severity Engineering

The raw dataset has no severity label. `severity` was engineered from official NOSDRA barrel-volume thresholds combined with habitat type.

```text
Inland water habitats:
  Major  -> > 250 barrels
  Medium -> 25 - 250 barrels
  Minor  -> < 25 barrels

Land / Swamp / Shoreline / Open Sea:
  Major  -> > 2,500 barrels
  Medium -> 250 - 2,500 barrels
  Minor  -> < 250 barrels
```

### Class distribution (after removing Unknown)

| Class | Count (full dataset) | Share | Test support |
|---|---|---|---|
| **Minor** | 12,678 | ~96.0% | 2,536 |
| **Medium** | 472 | ~3.6% | 95 |
| **Major** | 52 | ~0.4% | 10 |

The Major class has only 10 test samples. Any Major metric moves in steps of 10 percentage points per prediction and should be read as indicative, not reliable.

---

## Features

34 columns were excluded for high missingness (>20%), being identifiers, or being free text. The 9 retained features:

| Feature | Type | Description |
|---|---|---|
| `company` | Categorical | Operating company |
| `cause` | Categorical | Reported cause (e.g. sabotage, equipment failure) |
| `statesaffected` | Categorical | State where the spill occurred |
| `spillareahabitat` | Categorical | Habitat type (inland, swamp, open sea, etc.) |
| `incident_year` | Numeric | Year from `incidentdate` |
| `incident_month` | Numeric | Month from `incidentdate` |
| *(+ 3 others)* | Mixed | Remaining columns not excluded by the filters above |

Categorical features are label-encoded, which is why plain SMOTE is unsuitable (see Imbalance Handling).

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | Preprocessing, Random Forest, evaluation, class weights |
| XGBoost | Gradient boosting classifier |
| imbalanced-learn | SMOTENC oversampling (resampling inside a CV pipeline) |
| SHAP | Explainability (TreeExplainer) |
| Matplotlib / Seaborn | Visualisation (figures at 300 DPI) |
| Google Colab + Drive | Development environment |

---

## Project Structure

```text
oil-spill-risk-classifier/
|
|-- README.md
|-- requirements.txt
|-- .gitignore                         <- excludes data/, models/, __pycache__/
|-- streamlit_app.py                   <- incident assessment UI
|
|-- notebooks/
|   |-- 01_eda.ipynb                   <- Phase 2: Data Understanding
|   |-- 02_eda_part_b.ipynb            <- Phase 3: EDA
|   |-- 03_preprocessing.ipynb         <- Phase 4: Data Preparation
|   |-- 04_modeling.ipynb              <- Phase 5: Modelling, imbalance handling, comparison
|   `-- 05_explainability.ipynb        <- Phase 6: SHAP
|
|-- src/
|   |-- preprocess.py                  <- cleaning, encoding, severity engineering
|   |-- model.py                       <- baseline training (see note below)
|   `-- explain.py                     <- SHAP helpers
|
|-- reports/figures/                   <- saved plots (300 DPI)
|
|-- data/                              <- not tracked by Git
|   |-- raw/
|   `-- processed/
|
`-- models/                            <- not tracked by Git
    |-- xgboost_model.pkl              <- baseline (unweighted)
    |-- xgboost_smote_model.pkl        <- SMOTENC-trained
    |-- xgboost_weighted_model.pkl     <- class-weighted (final model)
    |-- target_encoder.pkl
    `-- label_encoders.pkl
```

---

## Notebook Execution Order

Run in sequence; each depends on the previous outputs.

| Order | Notebook | Phase | Key output |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Data Understanding | Data dictionary, quality report |
| 2 | `02_eda_part_b.ipynb` | EDA | 5 figures |
| 3 | `03_preprocessing.ipynb` | Preprocessing | train/test CSVs, encoders |
| 4 | `04_modeling.ipynb` | Modelling | Three models, comparison table |
| 5 | `05_explainability.ipynb` | XAI | SHAP plots |

---

## Imbalance Handling

Two approaches were tested against an unweighted baseline, all with the same split and the same tuned XGBoost settings family.

1. **SMOTENC.** Plain SMOTE interpolates between values, which produces meaningless codes (e.g. `company = 12.4`) for label-encoded categoricals. `SMOTENC` handles categorical columns correctly. It was placed inside an `imblearn` Pipeline so resampling happens on training folds only during cross-validation, and capped at 600 rows each for Major and Medium (full balancing from about 42 Major training rows would be mostly synthetic noise). Tuned with `GridSearchCV` on macro F1; best parameters: `learning_rate=0.1`, `max_depth=7`, `n_estimators=200`.
2. **Class weighting.** `compute_sample_weight("balanced")` passed to `XGBClassifier.fit`, with `learning_rate=0.1`, `max_depth=7`, `n_estimators=100`.

---

## Results

### Model comparison (test set, n = 2,641)

| Model | Macro F1 | Weighted F1 | Accuracy | Major recall | Medium recall | Minor recall | Major precision | Medium precision |
|---|---|---|---|---|---|---|---|---|
| Baseline XGBoost | 0.346 | 0.943 | 0.959 | 0.000 | 0.032 | 0.998 | 0.000 | 0.333 |
| SMOTENC + XGBoost | 0.371 | 0.944 | 0.958 | 0.000 | 0.084 | 0.994 | 0.000 | 0.348 |
| **Class-weighted XGBoost** | **0.420** | 0.898 | 0.866 | **0.200** | **0.400** | 0.886 | 0.118 | 0.119 |

Reference baselines (weighted F1 only): Dummy majority-class 0.9408, Random Forest (`class_weight="balanced"`) 0.9418.

**Macro F1 is the headline metric.** Weighted F1 is dominated by the 96% Minor class, which is why the Dummy classifier scores 0.94 and the best minority-recall model scores lowest on it.

### Final model: class-weighted XGBoost

| Severity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Major | 0.12 | 0.20 | 0.15 | 10 |
| Medium | 0.12 | 0.40 | 0.18 | 95 |
| Minor | 0.97 | 0.89 | 0.93 | 2,536 |
| Weighted avg | 0.94 | 0.87 | 0.90 | 2,641 |

### Interpretation

- **SMOTENC did not solve the problem.** Medium recall rose only from 0.03 to 0.08 and Major stayed at 0.00. With about 42 Major and about 377 Medium training rows and nine features that cannot observe actual spill volume, oversampling adds no new signal.
- **Class weighting changes the decision boundary far more.** It recovers 40% of Medium and 20% of Major spills (2 of 10), but precision for both falls to about 0.12, meaning roughly 8 of every 9 high-severity flags are false alarms.
- **Why class weighting is the final model.** In triage, unnecessarily triggering a Joint Investigation Visit costs less than waving a serious spill through as routine, so recall is prioritised. This is a use-case judgement, not a claim that the model is accurate: overall accuracy fell from 95.9% to 86.6%.
- **The model is a screening aid, not a decision-maker.** Predicted severity should prompt human review, not replace volume assessment.

---

## SHAP Explainability

SHAP analysis should be run against the final class-weighted model (`models/xgboost_weighted_model.pkl`). The feature-driver observations below were drawn from earlier analysis and must be re-verified against the final model's SHAP plots before being cited.

Candidate drivers to confirm:

1. `cause`: sabotage/theft and equipment failure as drivers of higher severity.
2. `spillareahabitat`: habitat sensitivity, partly mechanical because inland habitats have lower volume thresholds in the severity definition.
3. `company`: operator-level historical incident patterns.
4. `incident_year`: possible downward shift in recent years.

### Policy-oriented suggestions (conditional on verified SHAP results)

1. **Risk-based JIV prioritisation:** flag reports combining high-risk causes and sensitive habitats for early investigation visits.
2. **Targeted operator audits** informed by operator-level attribution.
3. **Habitat-focused monitoring** in areas historically linked to higher-severity predictions.

---

## Limitations

1. **Weak minority-class performance.** Even the best model has Major precision 0.12 and Medium precision 0.12. Major results rest on 10 test samples.
2. **Engineered target.** Severity is derived from `estimatedquantity`, so label noise in reported volumes carries directly into the target.
3. **Limited features.** The model cannot observe actual spill volume at reporting time, which caps achievable minority-class discrimination.
4. **Data latency and sparsity.** Pre-2006, under-review, and very recent incidents are excluded, introducing recency bias.
5. **Excluded features.** `lga`, `typeoffacility`, and `jivpresent` were dropped for >20% missingness and may carry signal.
6. **Single split.** Test metrics come from one stratified split; confidence intervals for the rare classes would be wide.

---

## Future Work

- Collect or impute high-signal excluded features (`lga`, `typeoffacility`, geospatial coordinates).
- Evaluate threshold tuning and cost-sensitive learning to trade precision for recall explicitly.
- Compare LightGBM / CatBoost (native categorical handling, avoiding label encoding).
- Repeated stratified cross-validation to report uncertainty on Major and Medium recall.
- Serve the final model through the Streamlit app or a FastAPI endpoint.

---

## Author

**Shalom "Vantage" Bolo Bebebaraseigha**  
Computer Science, University of Uyo  
Matric No.: 23/SC/CO/010  
SIWES Industrial Training: NDDC, Bayelsa State  
GitHub: [@shalombebx](https://github.com/shalombebx)

---

## Acknowledgements

- Dataset: [NOSDRA Oil Spill Monitor](https://nosdra.oilspillmonitor.ng)
- Explainability: [SHAP, Lundberg et al.](https://github.com/slundberg/shap)
- Methodology: CRISP-DM
- Development: Google Colab + Google Drive
