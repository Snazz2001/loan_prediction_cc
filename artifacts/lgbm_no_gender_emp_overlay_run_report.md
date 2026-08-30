# LightGBM (no gender; employment_status overlay-only) — run report

One model: `lightgbm.LGBMClassifier` (`objective=binary`, `boosting_type=gbdt`, `random_state=42`).
Not a stack. LogisticRegression and RandomForest were not trained as competing models.
Frozen split reused from `scripts/train.py::load_and_split` (unchanged). Last-run
`artifacts/lgbm_model.joblib`, PR #1 linear-tree artifacts, and PR #2 stack artifacts
were not written.

## 1. Script paths

- Train script: `scripts/train_lgbm_no_gender_emp_overlay.py` (tune + freeze on TRAIN only)
- Eval script: `scripts/eval_lgbm_no_gender_emp_overlay.py` (**never fits**; loads artifacts; scores Test; writes this report)
- Eval confirmation: no `.fit(`, no Optuna study, no `LGBMClassifier(` constructor in this file.

## 2. Exact in-model feature list

`gender` and `employment_status` are **ABSENT** from the in-model feature list, the WOE
encoder keys, the encoded matrices, the LightGBM `feature_name_`, and the SHAP table.
`loan_paid_back` is not a feature (`default = 1 - loan_paid_back` is the target only).
No IV drop. No VIF drop. `credit_score` and all other original columns except the two
policy drops remain.

In-model features (19): `age, marital_status, education_level, annual_income, monthly_income, debt_to_income_ratio, credit_score, loan_amount, loan_purpose, interest_rate, loan_term, installment, grade_subgrade, num_of_open_accounts, total_credit_limit, current_balance, delinquency_history, public_records, num_of_delinquencies`

| order | feature |
| --- | --- |
| 1 | age |
| 2 | marital_status |
| 3 | education_level |
| 4 | annual_income |
| 5 | monthly_income |
| 6 | debt_to_income_ratio |
| 7 | credit_score |
| 8 | loan_amount |
| 9 | loan_purpose |
| 10 | interest_rate |
| 11 | loan_term |
| 12 | installment |
| 13 | grade_subgrade |
| 14 | num_of_open_accounts |
| 15 | total_credit_limit |
| 16 | current_balance |
| 17 | delinquency_history |
| 18 | public_records |
| 19 | num_of_delinquencies |

- `gender` in model? **NO**
- `employment_status` in model? **NO** (policy overlay only; see §6)
- `loan_paid_back` in model? **NO**

## 3. Train OOF AUC vs last-run CV 0.8862 and vs stack OOF 0.8891

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **0.7094** (raw 0.7093987806008206)
- Train OOF KS (monitoring only, not used to select): 0.3192
- Last-run CV AUC (compare only, not retrained): 0.8862
- Beat last-run CV 0.8862? **NO**
- PR #2 stack Train OOF AUC (compare only, not retrained): 0.8891
- Beat stack OOF 0.8891? **NO**
- A drop vs the stack OOF is expected (this is a single LightGBM without gender / employment_status) and is not a reason to restore those features.
- n_trials completed: 50 / 50
- n_estimators_final = max(50, round(mean best_iteration)) = **253**
- Fold best_iterations: [466, 308, 170, 169, 154]
- Fold AUCs (winning params re-run): [0.720054, 0.699912, 0.720085, 0.70783, 0.701236]
- Fold KS (monitoring): [0.345089, 0.300446, 0.338839, 0.329018, 0.305353]

Best hyperparameters (search params; fixed `objective=binary`, `boosting_type=gbdt`, `random_state=42`, monotone_constraints all −1 on WOE columns):

| hyperparameter | value |
| --- | --- |
| learning_rate | 0.018426932029835672 |
| num_leaves | 17.0 |
| max_depth | 3.0 |
| min_child_samples | 169.0 |
| subsample | 0.6209197183594259 |
| colsample_bytree | 0.668048776781886 |
| reg_alpha | 0.19906320633702526 |
| reg_lambda | 0.004007312868129964 |
| n_estimators_final | 253.0 |

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 4. After freeze: Test metrics vs last-run champion and vs PR #2 stack

Test AUC 0.6995 did NOT beat last-run Test AUC 0.8858.

| metric | this candidate (Test) | last-run champion (Test, compare only) | beat last-run 0.8858? | PR #2 stack (Test, compare only) | beat stack 0.8885? |
| --- | --- | --- | --- | --- | --- |
| AUC | **0.6995** | 0.8858 | **NO** | 0.8885 | NO |
| KS | **0.2848** | 0.5777 | n/a (selection was OOF AUC) | 0.5795 | n/a |
| Gini | **0.3989** | 0.7715 | n/a | 0.7769 | n/a |
| PSI (train PD vs test PD) | **0.0011** | 0.0021 | n/a | 0.0028 | n/a |

- PSI status: Stable (<0.10) - No action needed
- This candidate Train refit (not OOF) AUC/KS: 0.7209 / 0.3312
- Last-run and stack numbers are documented comparators; they were not retrained in this run.
- A drop vs PR #2 stack Test AUC 0.8885 is expected and is **not** a reason to put `gender` or `employment_status` back into the model.

| model | sample | AUC | Gini | KS | PSI |
| --- | --- | --- | --- | --- | --- |
| LightGBM (this candidate; no gender / no employment_status) | Train (refit, not OOF) | 0.7209 | 0.4419 | 0.3312 | n/a |
| LightGBM (this candidate; no gender / no employment_status) | Test (after freeze) | 0.6995 | 0.3989 | 0.2848 | 0.0011 |
| Last-run champion LightGBM (compare only, not retrained) | Test | 0.8858 | 0.7715 | 0.5777 | 0.0021 |
| PR #2 stack LR+RF+LGBM (compare only, not retrained) | Test | 0.8885 | 0.7769 | 0.5795 | 0.0028 |

## 5. Train mean |SHAP| by in-model feature

Computed **after freeze**, on **TRAIN**, in-model features only. Method: shap.TreeExplainer on all TRAIN rows, in-model WOE features only (non-linear_tree LGBM; path-dependent TreeSHAP). utils.risk_skills.generate_shap_summary also run on TRAIN.
n_rows=14000.

| feature | mean_abs_shap |
| --- | --- |
| debt_to_income_ratio | 0.569143 |
| credit_score | 0.32396 |
| grade_subgrade | 0.243119 |
| loan_purpose | 0.042492 |
| interest_rate | 0.036853 |
| education_level | 0.031956 |
| delinquency_history | 0.028118 |
| installment | 0.022608 |
| loan_amount | 0.019769 |
| age | 0.018741 |
| num_of_open_accounts | 0.017697 |
| annual_income | 0.017174 |
| current_balance | 0.016511 |
| total_credit_limit | 0.010681 |
| num_of_delinquencies | 0.009189 |
| loan_term | 0.00789 |
| marital_status | 0.00783 |
| monthly_income | 0.005894 |
| public_records | 0.000931 |

- `gender` present in SHAP table? **NO**
- `employment_status` present in SHAP table? **NO**

`utils.risk_skills.generate_shap_summary` on TRAIN (same frozen model, no refit):
`{'xai_method': 'SHAP (Kernel/Tree)', 'n_rows': 2000, 'global_importance_ranking': [['debt_to_income_ratio', 0.5653], ['credit_score', 0.3115], ['grade_subgrade', 0.247], ['loan_purpose', 0.0416], ['interest_rate', 0.0388], ['education_level', 0.032], ['delinquency_history', 0.0295], ['installment', 0.0229], ['loan_amount', 0.0199], ['age', 0.018], ['annual_income', 0.0169], ['num_of_open_accounts', 0.0169], ['current_balance', 0.0168], ['total_credit_limit', 0.0114], ['num_of_delinquencies', 0.0084], ['loan_term', 0.0078], ['marital_status', 0.0075], ['monthly_income', 0.0058], ['public_records', 0.0009]]}`

## 6. Overlay exhibit (TRAIN only; NOT a model input)

**Explicit statement:** `employment_status` is a human-review / policy overlay. It is **not** a
model input. It was not WOE-encoded into the LightGBM feature matrix, not passed to
`LGBMClassifier`, and not consumed on the eval model path. Do not restore it to chase AUC.

Overlay IV (`calculate_woe_iv` on TRAIN): **1.9096** (Suspicious / Overfitting (>0.5))

| employment_status | n | n_default | default_rate | woe |
| --- | --- | --- | --- | --- |
| Unemployed | 1475 | 1210 | 0.8203 | -2.9045 |
| Student | 566 | 332 | 0.5866 | -1.7357 |
| Employed | 9086 | 1029 | 0.1133 | 0.6721 |
| Self-employed | 2021 | 224 | 0.1108 | 0.6964 |
| Retired | 852 | 6 | 0.007 | 3.5629 |

Share of TRAIN defaults in Unemployed vs Retired (counts and % of all train defaults; train defaults total = 2801):

| employment_status | n | n_default | share_of_all_train_defaults | share_of_all_train_defaults_pct |
| --- | --- | --- | --- | --- |
| Unemployed | 1475 | 1210 | 0.431989 | 43.1989 |
| Retired | 852 | 6 | 0.002142 | 0.2142 |

- Unemployed: n=1475, n_default=1210, share of all train defaults = 43.1989%
- Retired: n=852, n_default=6, share of all train defaults = 0.2142%

This employment_status WOE table is a TRAIN-only policy / human-review overlay. It is NOT a model input. It is not passed to LightGBM, not present in the WOE feature matrix, and not consumed by the eval model path.

## 7. Artifact paths + freeze-before-test confirmation

- `artifacts/lgbm_no_gender_emp_overlay_model.joblib`
- `artifacts/lgbm_no_gender_emp_overlay_encoders.joblib`
- `artifacts/lgbm_no_gender_emp_overlay_meta.json`
- `artifacts/lgbm_no_gender_emp_overlay_overlay.json`
- `artifacts/lgbm_no_gender_emp_overlay_test_metrics.json`
- `artifacts/lgbm_no_gender_emp_overlay_run_report.md`
- `artifacts/lgbm_no_gender_emp_overlay_X_train.csv`
- `artifacts/lgbm_no_gender_emp_overlay_y_train.csv`
- `artifacts/lgbm_no_gender_emp_overlay_X_test.csv`
- `artifacts/lgbm_no_gender_emp_overlay_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`.

- Freeze timestamp (UTC), written by the train script before this eval script ran: **2026-08-30T13:09:20.179500+00:00**
- `freeze.test_looked_at` at freeze time: `False`
- `freeze.test_metrics` at freeze time: `None`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select=False`)
- Eval script never calls `fit` / Optuna / early stopping. It only `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split: `scripts/train.py::load_and_split` unchanged (`test_size=0.3`, `stratify=default`, `random_state=42`).
  Train n=14000 bad_rate=0.2001;
  Test n=6000 bad_rate=0.2002.
- WOE 5-bin encoders (`fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS` from `utils.config`) fit on TRAIN only.

## 8. Plain statement on Test AUC vs 0.8858

Test AUC 0.6995 did NOT beat last-run Test AUC 0.8858.

Beat last-run Test AUC 0.8858? **NO**
