# OpenFE + LightGBM validation report

## Executive Summary

OpenFE (Zhang et al., ICML 2023, pip `openfe==0.0.12`) was **fit on TRAIN only**, then LightGBM `boosting_type=gbdt` was tuned on the frozen OpenFE+base-20 matrix with StratifiedKFold 5 (`shuffle=True`, `random_state=42`) maximizing OOF AUC. Test labels were not used to fit or select. This eval script **never fits FE or the model**.

- Device: **cpu** (4 CPUs). This VM has no GPU; OpenFE and LightGBM run on CPU.
- gender in model / FE: **NO**
- employment_status in model: **YES** (LGBM gain rank **3**, TRAIN permutation rank **3**)
- monotone_constraints: **false**; interaction_constraints: **false**; IV/VIF drop: **false**
- Freeze UTC: `2026-09-13T15:24:52.579483+00:00`
- OpenFE+LGBM Test AUC **0.8887** / KS **0.5806** / Gini **0.7774** / PSI **0.003**
- Baseline raw-20 LGBM Test AUC **0.8921** (delta OpenFE−baseline = **-0.0034**)
- Gates: AUC>=0.7 YES; KS>=0.3 YES; PSI_WATCH=0.1 exceeded=NO

## Train / Test Split

Source: `scripts/train.py::load_and_split` **UNCHANGED** (`test_size=0.3`, `stratify=default`, `random_state=42`).

| split | n | bad_rate | n_bads |
| --- | --- | --- | --- |
| Train | 14000 | 0.2001 | 2801 |
| Test | 6000 | 0.2002 | 1201 |

Expected: Train n=14000 bad_rate≈0.2001 (2801 bads); Test n=6000 bad_rate≈0.2002 (1201 bads). Confirmed against a fresh `load_and_split` call in this eval.

## Feature Engineering & Selection

- Base features: the exact remaining **20** original columns (gender dropped; employment_status kept in native CSV order). No IV/VIF drop.
- OpenFE version: **0.0.12**; n_jobs: **4**
- Candidates generated (order=1): **1935**
- Stage-2 ranked: **762**; selected: **50**
- Base 20 kept: **YES**
- Selection method: OpenFE two-stage selection on TRAIN only: stage1=predictive successive feature-wise halving (n_data_blocks=8, min_candidate_features=2000); stage2=gain_importance. Keep formulas with stage2 gain>0, capped at TOP_K=50. Original base 20 columns are KEPT.

Selected formulas:

| autoFE | formula |
| --- | --- |
| autoFE_f_0 | GroupByThenNUnique(num_of_open_accounts,employment_status) |
| autoFE_f_1 | (debt_to_income_ratio/credit_score) |
| autoFE_f_2 | (credit_score+age) |
| autoFE_f_3 | CombineThenFreq(employment_status,age) |
| autoFE_f_4 | GroupByThenRank(credit_score,age) |
| autoFE_f_5 | (credit_score-age) |
| autoFE_f_6 | GroupByThenRank(credit_score,marital_status) |
| autoFE_f_7 | CombineThenFreq(employment_status,grade_subgrade) |
| autoFE_f_8 | (debt_to_income_ratio*interest_rate) |
| autoFE_f_9 | (credit_score-num_of_open_accounts) |
| autoFE_f_10 | Combine(employment_status,public_records) |
| autoFE_f_11 | GroupByThenMin(interest_rate,grade_subgrade) |
| autoFE_f_12 | GroupByThenRank(credit_score,num_of_open_accounts) |
| autoFE_f_13 | (credit_score+interest_rate) |
| autoFE_f_14 | GroupByThenRank(debt_to_income_ratio,age) |
| autoFE_f_15 | (credit_score+loan_term) |
| autoFE_f_16 | (credit_score/interest_rate) |
| autoFE_f_17 | (credit_score-loan_term) |
| autoFE_f_18 | GroupByThenRank(credit_score,delinquency_history) |
| autoFE_f_19 | GroupByThenRank(credit_score,num_of_delinquencies) |
| autoFE_f_20 | GroupByThenRank(credit_score,public_records) |
| autoFE_f_21 | GroupByThenMax(num_of_delinquencies,employment_status) |
| autoFE_f_22 | (annual_income+total_credit_limit) |
| autoFE_f_23 | (debt_to_income_ratio*num_of_delinquencies) |
| autoFE_f_24 | (debt_to_income_ratio*total_credit_limit) |
| autoFE_f_25 | GroupByThenRank(debt_to_income_ratio,num_of_delinquencies) |
| autoFE_f_26 | (debt_to_income_ratio-credit_score) |
| autoFE_f_27 | (debt_to_income_ratio*credit_score) |
| autoFE_f_28 | residual(installment) |
| autoFE_f_29 | Combine(grade_subgrade,num_of_open_accounts) |
| autoFE_f_30 | (current_balance/num_of_open_accounts) |
| autoFE_f_31 | (annual_income/total_credit_limit) |
| autoFE_f_32 | (total_credit_limit/current_balance) |
| autoFE_f_33 | CombineThenFreq(education_level,employment_status) |
| autoFE_f_34 | GroupByThenRank(debt_to_income_ratio,loan_purpose) |
| autoFE_f_35 | (current_balance*num_of_delinquencies) |
| autoFE_f_36 | (debt_to_income_ratio*num_of_open_accounts) |
| autoFE_f_37 | (total_credit_limit/num_of_open_accounts) |
| autoFE_f_38 | Combine(loan_purpose,num_of_open_accounts) |
| autoFE_f_39 | min(credit_score,loan_amount) |
| autoFE_f_40 | GroupByThenRank(debt_to_income_ratio,delinquency_history) |
| autoFE_f_41 | (monthly_income/debt_to_income_ratio) |
| autoFE_f_42 | (debt_to_income_ratio*age) |
| autoFE_f_43 | (credit_score-interest_rate) |
| autoFE_f_44 | GroupByThenMean(interest_rate,age) |
| autoFE_f_45 | (debt_to_income_ratio-loan_term) |
| autoFE_f_46 | GroupByThenMedian(interest_rate,grade_subgrade) |
| autoFE_f_47 | GroupByThenRank(debt_to_income_ratio,num_of_open_accounts) |
| autoFE_f_48 | (monthly_income/num_of_delinquencies) |
| autoFE_f_49 | CombineThenFreq(grade_subgrade,age) |

Leakage note: OpenFE.fit used TRAIN rows and TRAIN labels only. openfe.transform applies the frozen formulas; it does not refit or use labels. OpenFE's official transform concatenates train+test solely so GroupByThen*/freq operators can evaluate (library design). Test labels were never used. Feature selection never included test rows or test labels.

## Hyperparameter Tuning (Optuna)

Optuna TPESampler seed=42, 25 trials, StratifiedKFold(5, shuffle=True, random_state=42), maximize OOF AUC. n_estimators_final = mean fold best_iteration (early_stopping=50, n_estimators_tune=1000).

Best hyperparameters (OpenFE+LGBM):

| param | value |
| --- | --- |
| learning_rate | 0.04318321693373985 |
| num_leaves | 57.0 |
| max_depth | 3.0 |
| min_child_samples | 61.0 |
| subsample | 0.962029747855441 |
| colsample_bytree | 0.6571958731133117 |
| reg_alpha | 0.004663260912619185 |
| reg_lambda | 0.0021051189482196196 |

- n_estimators_final (OpenFE+LGBM): **96**
- fold best_iterations: [102, 128, 97, 87, 66]
- Baseline raw-20 n_estimators_final: **195**

## Model Performance & Discrimination

### OOF (train folds, pre-freeze)

| model | AUC | Gini | KS_Statistic | Optimal_Cutoff_Probability | Validation_Rating |
| --- | --- | --- | --- | --- | --- |
| openfe_lgbm | 0.8829 | 0.7658 | 0.5611 | 0.1798 | Good (0.40 - 0.60) |
| baseline_raw20 | 0.8918 | 0.7836 | 0.584 | 0.1954 | Good (0.40 - 0.60) |

### After freeze — Train refit (not used for selection)

| model | AUC | Gini | KS_Statistic | Optimal_Cutoff_Probability | Validation_Rating |
| --- | --- | --- | --- | --- | --- |
| openfe_lgbm | 0.9116 | 0.8232 | 0.6368 | 0.1862 | Suspicious / Extreme (>0.60) |
| baseline_raw20 | 0.9222 | 0.8444 | 0.665 | 0.2034 | Suspicious / Extreme (>0.60) |

### After freeze — Test

| model | AUC | KS_Statistic | Gini | Validation_Rating | Optimal_Cutoff_Probability | PSI | PSI_Status | psi_above_0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openfe_lgbm | 0.8887 | 0.5806 | 0.7774 | Good (0.40 - 0.60) | 0.1807 | 0.003 | Stable (<0.10) - No action needed | False |
| baseline_raw20 | 0.8921 | 0.5822 | 0.7842 | Good (0.40 - 0.60) | 0.183 | 0.0027 | Stable (<0.10) - No action needed | False |

## Population Stability Index (PSI)

Score PSI uses train refit PD as expected and test PD as actual (`calculate_psi`, 10 bins).

| model | PSI | status | psi_above_0.10 |
| --- | --- | --- | --- |
| openfe_lgbm | 0.003 | Stable (<0.10) - No action needed | NO |
| baseline_raw20 | 0.0027 | Stable (<0.10) - No action needed | NO |

## Explainability & Governance

- gender absent from base X, OpenFE formulas, transformed columns, and `feature_name_`: **YES**
- employment_status in-model: **YES**
- TRAIN LGBM gain rank of employment_status: **3**
- TRAIN permutation rank of employment_status: **3**

Top 20 TRAIN LGBM gain:

| feature | importance |
| --- | --- |
| autoFE_f_1 | 122 |
| autoFE_f_29 | 57 |
| employment_status | 55 |
| autoFE_f_0 | 50 |
| autoFE_f_4 | 39 |
| autoFE_f_8 | 38 |
| autoFE_f_2 | 29 |
| autoFE_f_5 | 21 |
| autoFE_f_38 | 20 |
| autoFE_f_15 | 18 |
| autoFE_f_10 | 17 |
| debt_to_income_ratio | 15 |
| autoFE_f_3 | 14 |
| autoFE_f_13 | 14 |
| autoFE_f_12 | 12 |
| autoFE_f_34 | 12 |
| autoFE_f_17 | 10 |
| autoFE_f_43 | 10 |
| autoFE_f_42 | 8 |
| autoFE_f_47 | 8 |

Top 20 TRAIN permutation (roc_auc, sample=4000):

| feature | importance |
| --- | --- |
| autoFE_f_0 | 0.06814720530149206 |
| autoFE_f_1 | 0.0452442477935705 |
| employment_status | 0.025852720617699276 |
| autoFE_f_29 | 0.013403036639694554 |
| autoFE_f_4 | 0.0064508221286110334 |
| autoFE_f_2 | 0.006404487950083082 |
| autoFE_f_8 | 0.00637916578274796 |
| autoFE_f_3 | 0.0025881140709762183 |
| autoFE_f_38 | 0.0024455421873370207 |
| autoFE_f_5 | 0.002410118091543715 |
| autoFE_f_10 | 0.0021522495311022314 |
| autoFE_f_12 | 0.0017034545228018134 |
| autoFE_f_15 | 0.001361739956157626 |
| autoFE_f_13 | 0.000819199 |
| debt_to_income_ratio | 0.000784381 |
| autoFE_f_17 | 0.000757645 |
| autoFE_f_9 | 0.00075293 |
| autoFE_f_43 | 0.000718853 |
| autoFE_f_16 | 0.000562879 |
| autoFE_f_24 | 0.000535739 |

## beats_* vs comparators (Test AUC)

| comparator | auc | openfe_lgbm_beats | baseline_beats |
| --- | --- | --- | --- |
| last_run_0.8858 | 0.8858 | YES | YES |
| pr2_0.8885 | 0.8885 | YES | YES |
| pr4_0.8859 | 0.8859 | YES | YES |
| pr6_stack_0.8883 | 0.8883 | YES | YES |
| pr6_gbdt_0.8895 | 0.8895 | NO | YES |
| pr7_stack_0.8951 | 0.8951 | NO | NO |
| pr7_ft_0.8963 | 0.8963 | NO | NO |
| pr8_danet_0.8655 | 0.8655 | YES | YES |
| pr9_ag_0.8967 | 0.8967 | NO | NO |
| pr9_avg_stack_0.8970 | 0.897 | NO | NO |

## Package versions

| package | version |
| --- | --- |
| python | 3.12.3 |
| openfe | 0.0.12 |
| lightgbm | 4.7.0 |
| sklearn | 1.9.1 |
| pandas | 3.0.5 |
| numpy | 2.4.4 |
| joblib | 1.6.0 |
| optuna | 5.0.0 |

## Artifacts

Persisted OpenFE object / formulas / transformed matrices / LGBM models so a validator can SHA-match and rerun this eval **without refitting FE**.

- `artifacts/openfe_lgbm_model.joblib`
- `artifacts/openfe_lgbm_openfe.joblib`
- `artifacts/openfe_lgbm_features.joblib`
- `artifacts/openfe_lgbm_formulas.json`
- `artifacts/openfe_lgbm_X_train.joblib` / `artifacts/openfe_lgbm_X_test.joblib`
- `artifacts/openfe_lgbm_test_metrics.json`
- `artifacts/openfe_lgbm_run_report.md`

Did not write or overwrite prior-pack prefixes (`lgbm_linear_tree_`, `stack_lr_rf_lgbm_`, `lgbm_emp_overlay_`, `lgbm_emp_in_`, `dart_monotone_`, `ensemble_no_gender_`, `ensemble_fm_ft_`, `danet_`, `ag_realmlp_`, `autogluon_`, `realmlp_`) or `artifacts/lgbm_model.joblib`.
