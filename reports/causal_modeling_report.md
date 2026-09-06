# PharmaPulse Uplift/Causal Modeling Report
Generated on: 2026-09-06 23:58:10

## Overview
This report summarizes the uplift modeling work for estimating heterogeneous treatment effects (HTE)
of Treatment A and B vs Control on adherence and discontinuation outcomes.

## Approach
We estimated the conditional average treatment effect (CATE) using three approaches:
1. Uplift Random Forest (from causalml library)
2. XGBoost Uplift (training separate models for treatment and control and taking the difference)
3. LightGBM Uplift (training separate models for treatment and control and taking the difference)

## Evaluation Metric
We used a simplified version of the Qini coefficient, specifically the uplift in the top 20%
of predicted treatment effect minus the uplift in the bottom 20%. This measures how well the model
separates those who are positively affected by the treatment from those who are not or negatively affected.

## Results
### Treatment A vs Control on ADHERENCE

| Model | Test Uplift Qini (20%) |
|-------|------------------------|
| Xgboost Uplift | 0.2837 |
| Lightgbm Uplift | 0.2574 |

### Treatment A vs Control on DISCONTINUATION

| Model | Test Uplift Qini (20%) |
|-------|------------------------|
| Xgboost Uplift | -0.1333 |
| Lightgbm Uplift | -0.0714 |

### Treatment B vs Control on ADHERENCE

| Model | Test Uplift Qini (20%) |
|-------|------------------------|
| Xgboost Uplift | -0.5222 |
| Lightgbm Uplift | -0.4167 |

### Treatment B vs Control on DISCONTINUATION

| Model | Test Uplift Qini (20%) |
|-------|------------------------|
| Xgboost Uplift | 0.1111 |
| Lightgbm Uplift | 0.0000 |

## Files Generated
- `models/causal/`: Trained uplift models and preprocessors
- `reports/figures/causal/`: Uplift comparison plots
- `reports/causal/`: Detailed model results (JSON format)

## Next Steps
1. Integrate uplift estimates with risk models to build a next-best-action engine
2. Use uplift models to personalize treatment recommendations
3. Validate uplift models using holdout datasets or simulation
4. Extend to multi-treatment uplift modeling (more than two treatment arms)
