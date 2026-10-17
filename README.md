# Ames Housing Price Prediction

An end-to-end data science project predicting residential property sale prices using the Ames, Iowa Housing dataset. Built with Python, Pandas, scikit-learn, and Tableau.

## 📄 Full Report

**[Click here to view the full 12-page analytical report (PDF)](./Ames_Iowa_housing_dataset.pdf)**

## 📓 Notebook

**[Click here to view the Jupyter Notebook](ames_housing_ml.ipynb)**

## Project Overview

- **Dataset:** 2,930 home sale records from Ames, Iowa (2006–2010), 82 variables covering physical structure, lot characteristics, and sale details.
- **Goal:** Predict `SalePrice` from property features and identify which factors most influence price.
- **Approach:** Data wrangling → feature engineering → visualization → model training → evaluation.

## Data Wrangling

- Investigated all 27 columns with missing values and classified them as true missing data vs. "not applicable" (e.g., `PoolQC` is `NA` because the house has no pool).
- Imputed `LotFrontage` using the **median within each neighborhood** to preserve local spatial patterns, with a global median fallback.
- Engineered a new **`TotalSF`** feature combining basement, first-floor, and second-floor square footage, since total livable space drives price more than floor distribution.

## Visualizations

- **Histogram** of `SalePrice` — right-skewed, with most homes clustered around $150,000.
- **Scatter plots** of `GrLivArea` and `TotalSF` vs. `SalePrice` — both show a clear positive relationship, with `TotalSF` stronger.
- **Correlation heatmap** — `OverallQual` (≈0.80) is the single strongest predictor, followed by living area features.
- **Box plot** of `SalePrice` by `OverallQual` — strong monotonic relationship from quality rating 1 to 10.

## Models & Results

Three regression models were trained and evaluated on a 20% held-out test set (292 samples):

| Model                   | MAE                  | RMSE                 | R²              |
| ----------------------- | -------------------- | -------------------- | ---------------- |
| **Random Forest** | **$19,178.80** | **$32,009.16** | **0.8270** |
| Linear Regression       | $21,260.53           | $47,160.57           | 0.6244           |
| Ridge Regression        | $21,203.57           | $47,030.87           | 0.6265           |

**Winner: Random Forest** — explains ~82.7% of the variance in sale prices and outperforms linear models on every metric. Tree-based methods capture the complex, non-linear relationships typical of real estate data.

## Key Findings

- **`OverallQual`** (material and finish quality) is the strongest single predictor of price.
- **`TotalSF`** (total livable area including basement) outperforms above-ground-only area as a predictor.
- Linear and Ridge Regression both plateau around R² ≈ 0.62, confirming that price is not a linear function of the features.
- `GarageCars` and `GarageArea` are highly multicollinear (≈0.88) — Ridge Regression's regularization provides a small edge over standard Linear Regression.

## Tools Used

- **Python** — Pandas, NumPy, scikit-learn
- **Visualization** — Matplotlib, Seaborn, Tableau
- **Environment** — GitHub Codespaces / Google Colab

## Dataset Source

Ames Iowa Housing Dataset — compiled by the Ames City Assessor's Office, covering sales from 2006–2010.
