# PharmaPulse Predictive Modeling Report
Generated on: 2026-09-06 23:57:05

## Overview
This report summarizes the predictive modeling work for adherence and discontinuation risk prediction.

## Models Trained
- Logistic Regression
- XGBoost
- LightGBM

## Targets Predicted
1. Adherence Risk (binary: high adherence >=0.8 vs low adherence <0.8)
2. Discontinuation Risk (binary: discontinued=1 vs not discontinued=0)

## ADHERENCE RISK PREDICTION

| Model | Test ROC-AUC | Test PR-AUC | Test F1 | CV ROC-AUC Mean (±Std) |
|-------|--------------|-------------|---------|------------------------|
| Logistic Regression | 0.714 | 0.359 | 0.200 | 0.800 (±0.011) |
| Xgboost | 0.695 | 0.318 | 0.157 | 0.780 (±0.030) |
| Lightgbm | 0.695 | 0.354 | 0.000 | 0.783 (±0.028) |

## DISCONTINUATION RISK PREDICTION

| Model | Test ROC-AUC | Test PR-AUC | Test F1 | CV ROC-AUC Mean (±Std) |
|-------|--------------|-------------|---------|------------------------|
| Logistic Regression | 0.949 | 0.322 | 0.000 | 0.663 (±0.145) |
| Xgboost | 0.663 | 0.276 | 0.000 | 0.661 (±0.177) |
| Lightgbm | 0.307 | 0.040 | 0.000 | 0.649 (±0.128) |

## Files Generated
- `models/predictive/`: Trained models and preprocessors
- `reports/figures/`: Model comparison plots and SHAP summary plots
- `reports/models/`: Detailed model results (JSON format)

## Next Steps
1. Implement uplift/causal modeling for heterogeneous treatment effects
2. Develop survival analysis for time-to-event data
3. Build next-best-action engine combining risk and uplift estimates
4. Create FastAPI backend for model serving
5. Develop Streamlit dashboard for visualization and interaction
