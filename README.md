# Credit Risk Classification — Lending Club Loan Default Prediction

A probability-of-default (PD) model built on Lending Club's public loan dataset, comparing a logistic regression baseline against gradient-boosted tree models (XGBoost, LightGBM), with threshold selection and SHAP-based interpretability.

## Project overview

This project predicts whether a loan will default, using borrower and loan characteristics available at the time of issuance. The approach mirrors how a bank credit risk team would build and evaluate a PD model:

- Start simple and interpretable (logistic regression) before adding model complexity
- Evaluate using metrics suited to imbalanced classification (ROC-AUC, PR-AUC) rather than raw accuracy
- Select a decision threshold based on the real-world cost asymmetry between missed defaults and false alarms
- Explain the model's decisions using SHAP, rather than treating it as a black box

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.713 | 0.381 |
| XGBoost | 0.727 | 0.404 |
| LightGBM | 0.726 | 0.402 |

Base default rate in the dataset: ~20.3% (the PR-AUC floor for a no-skill model).

**XGBoost** was selected as the final model. At a threshold tuned for **85% recall**, the model achieves 28% precision on the default class — a deliberate trade-off, since in lending a missed default is substantially costlier than flagging a safe loan for additional review.

## Data

This repo does **not** include the dataset — it's downloaded via the Kaggle API at setup time (see below). The dataset used is the [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (`wordsforthewise/lending-club`), ~1.1GB, ~2.2M loan records (2007–2018).

## Setup

### 1. Clone the repo and install dependencies

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

### 2. Get a Kaggle API token

1. Create a free account at [kaggle.com](https://kaggle.com) and verify your phone number.
2. Go to Account Settings → API → **Generate New Token**.
3. Copy the token value shown.

### 3. Set your token as an environment variable

```bash
# Mac/Linux
export KAGGLE_API_TOKEN=your_token_here

# Windows (Command Prompt / Anaconda Prompt)
set KAGGLE_API_TOKEN=your_token_here
```

(See Kaggle's docs for making this permanent on your system if you'll be re-running this across multiple sessions.)

### 4. Download the dataset

```bash
python download_data.py
```

This pulls the dataset into a local `data/` folder (gitignored — it never gets committed).

## Running the project

Open the main notebook and run cells in order:

```bash
jupyter notebook credit_risk_model.ipynb
```

The notebook is organized into sections: data preparation → feature engineering → encoding → model training (Logistic Regression, XGBoost, LightGBM) → model comparison → threshold selection → SHAP interpretability.

## Methodology notes

- **Target definition:** loans are labeled default (1) if `loan_status` is "Charged Off" or "Default"; "Fully Paid" loans are labeled 0. In-progress loans (Current, Late, In Grace Period) are excluded, since their final outcome isn't yet known.
- **Leakage avoidance:** fields only populated after the loan outcome is known (e.g. recovery amounts, last payment info) are dropped before modeling.
- **Train/test split:** stratified 80/20 split, done before any encoding that depends on the training distribution (e.g. `addr_state` frequency encoding is fit on the training set only).
- **Class imbalance:** handled via class weighting (`class_weight='balanced'` / `scale_pos_weight`), applied to the training set only.
- **Threshold selection:** chosen from the precision-recall curve to prioritize recall over precision, reflecting the higher real-world cost of a missed default versus a false alarm.

## Limitations & possible extensions

- No macroeconomic context (e.g. unemployment rate, interest rate environment at time of issuance)
- Could extend to LGD (loss given default) and EAD (exposure at default) modeling for a fuller expected-loss estimate
- A true out-of-time validation (training on earlier loan vintages, testing on later ones) would better reflect real-world deployment than a random stratified split

## Requirements

See `requirements.txt` for exact package versions. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `matplotlib`, `seaborn`, `kaggle`.
