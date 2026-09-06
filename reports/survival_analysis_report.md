# PharmaPulse Survival Analysis Report
Generated on: 2026-09-06 23:57:21

## Overview
This report summarizes the survival analysis for time-to-discontinuation.

## Data Summary
- Number of patients: 1000
- Number of discontinuation events: 22
- Event rate: 0.022

## Overall Survival
- Median survival time: Not reached (median survival > max follow-up)

## Log-Rank Test (Cluster Comparison)
- Chi-square statistic: 2103.337
- p-value: 0.000
- Degrees of freedom: 1

## Survival by Cluster

- Cluster 0:
  - Median survival: Not reached
  - Events: 5/482

- Cluster 1:
  - Median survival: Not reached
  - Events: 5/129

- Cluster 2:
  - Median survival: Not reached
  - Events: 12/389

## Cox Proportional Hazards Model
- Concordance index: 0.878
- Log-likelihood: -117.066
- AIC: 282.133
- Number of covariates: 24

## Files Generated
- `reports/figures/survival/`: Kaplan-Meier plots, CoxMS hazard ratio plot
- `models/survival/`: Trained survival models (KaplanMeierFitter, CoxPHFitter)
- `reports/figures/survival/km_results.json`: Kaplan-Meier results
- `reports/figures/survival/cph_results.json`: CoxPH results

## Next Steps
1. Integrate survival predictions with risk models for dynamic prognostication
2. Develop cure models if appropriate (many patients may not experience event)
3. Competing risks analysis (if other events like death are considered)
4. Dynamic survival predictions using longitudinal updates
