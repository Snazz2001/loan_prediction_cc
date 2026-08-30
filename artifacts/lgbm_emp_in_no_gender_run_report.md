# LightGBM (employment_status in-model; gender out) — run report

One model: `lightgbm.LGBMClassifier` (`objective=binary`, `boosting_type=gbdt`, `random_state=42`).
Not a stack. LogisticRegression and RandomForest were not trained. Not `linear_tree`.
No `interaction_constraints`. Frozen split reused from `scripts/train.py::load_and_split`
(unchanged). Last-run `artifacts/lgbm_model.joblib`, PR #1 linear-tree artifacts, PR #2
stack artifacts, and PR #3 overlay artifacts were not written.

## 1. Script paths

- Train script: `scripts/train_lgbm_emp_in_no_gender.py` (tune + freeze on TRAIN only)
- Eval script: `scripts/eval_lgbm_emp_in_no_gender.py` (**never fits**; loads artifacts; scores Test; writes this report)
- Eval confirmation: no `.fit(`, no Optuna study, no `LGBMClassifier(` constructor in this file.

## 2. Exact in-model feature list (20)

`gender` is **ABSENT** from the in-model feature list, the WOE encoder keys, the encoded
matrices, the LightGBM `feature_name_`, and the SHAP table.
`employment_status` is a **normal in-model WOE feature** (not overlay, not policy table,
not an interaction-constraint singleton). It is present in X, encoders, `feature_name_`,
and SHAP.
`loan_paid_back` is not a feature (`default = 1 - loan_paid_back` is the target only).
No IV drop. No VIF drop. `credit_score` stays. All remaining original columns except
`gender` remain.

In-model features (20): `age, marital_status, education_level, annual_income, monthly_income, employment_status, debt_to_income_ratio, credit_score, loan_amount, loan_purpose, interest_rate, loan_term, installment, grade_subgrade, num_of_open_accounts, total_credit_limit, current_balance, delinquency_history, public_records, num_of_delinquencies`

| order | feature |
| --- | --- |
| 1 | age |
| 2 | marital_status |
| 3 | education_level |
| 4 | annual_income |
| 5 | monthly_income |
| 6 | employment_status |
| 7 | debt_to_income_ratio |
| 8 | credit_score |
| 9 | loan_amount |
| 10 | loan_purpose |
| 11 | interest_rate |
| 12 | loan_term |
| 13 | installment |
| 14 | grade_subgrade |
| 15 | num_of_open_accounts |
| 16 | total_credit_limit |
| 17 | current_balance |
| 18 | delinquency_history |
| 19 | public_records |
| 20 | num_of_delinquencies |

- `gender_in_model` = **false**
- `employment_status_in_model` = **true**
- `loan_paid_back` in model? **NO**

## 3. Train OOF AUC/KS vs last CV 0.8862, vs stack OOF 0.8891, vs overlay OOF 0.7094

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **0.8882** (raw 0.8881846982372291)
- Train OOF KS (monitoring only, not used to select): 0.5825
- Last-run CV AUC (compare only, not retrained): 0.8862
- Beat last-run CV 0.8862? **YES**
- PR #2 stack Train OOF AUC (compare only, not retrained): 0.8891
- Beat stack OOF 0.8891? **NO**
- PR #3 overlay Train OOF AUC (compare only, not retrained): 0.7094
- Beat overlay OOF 0.7094? **YES**
- n_trials completed: 50 / 50
- n_estimators_final = max(50, round(mean best_iteration)) = **161**
- Fold best_iterations: [212, 153, 137, 129, 172]
- Fold AUCs (winning params re-run): [0.882159, 0.88713, 0.894003, 0.890117, 0.887842]
- Fold KS (monitoring): [0.584821, 0.597768, 0.593304, 0.598214, 0.580284]
- OOF PDs persisted: `artifacts/lgbm_emp_in_no_gender_oof_pds.csv`

Best hyperparameters (search params; fixed `objective=binary`, `boosting_type=gbdt`, `random_state=42`, monotone_constraints all −1 on WOE columns):

| hyperparameter | value |
| --- | --- |
| learning_rate | 0.059145813889703885 |
| num_leaves | 33.0 |
| max_depth | 5.0 |
| min_child_samples | 94.0 |
| subsample | 0.7864816397178762 |
| colsample_bytree | 0.7567007984472038 |
| reg_alpha | 0.056561467651267544 |
| reg_lambda | 0.0010869536551841176 |
| n_estimators_final | 161.0 |

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 4. After freeze: Test metrics vs last-run / stack / overlay

Plain yes/no on beat 0.8858 / 0.8885 / 0.6995 (no go/no-go declaration):

- Beat last-run Test AUC 0.8858? **YES**. YES — Test AUC 0.8859 beat last-run Test AUC 0.8858.
- Beat stack Test AUC 0.8885? **NO**. NO — Test AUC 0.8859 did not beat stack Test AUC 0.8885.
- Beat overlay Test AUC 0.6995? **YES**. YES — Test AUC 0.8859 beat overlay Test AUC 0.6995.

| metric | this candidate (Test) | last-run champion (Test) | beat 0.8858? | PR #2 stack (Test) | beat 0.8885? | PR #3 overlay (Test) | beat 0.6995? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUC | **0.8859** | 0.8858 | **YES** | 0.8885 | **NO** | 0.6995 | **YES** |
| KS | **0.5735** | 0.5777 | n/a | 0.5795 | n/a | 0.2848 | n/a |
| Gini | **0.7718** | 0.7715 | n/a | 0.7769 | n/a | n/a | n/a |
| PSI (train PD vs test PD) | **0.0019** | 0.0021 | n/a | 0.0028 | n/a | n/a | n/a |

- PSI status: Stable (<0.10) - No action needed
- This candidate Train refit (not OOF) AUC/KS: 0.898 / 0.6067
- Last-run, stack, and overlay numbers are documented comparators; they were not retrained in this run.

| model | sample | AUC | Gini | KS | PSI |
| --- | --- | --- | --- | --- | --- |
| LightGBM (this candidate; emp in, gender out) | Train (refit, not OOF) | 0.898 | 0.796 | 0.6067 | n/a |
| LightGBM (this candidate; emp in, gender out) | Test (after freeze) | 0.8859 | 0.7718 | 0.5735 | 0.0019 |
| Last-run champion LightGBM (compare only, not retrained) | Test | 0.8858 | 0.7715 | 0.5777 | 0.0021 |
| PR #2 stack LR+RF+LGBM (compare only, not retrained) | Test | 0.8885 | 0.7769 | 0.5795 | 0.0028 |
| PR #3 overlay LightGBM (gender out, emp overlay-only; compare only) | Test | 0.6995 | n/a | 0.2848 | n/a |

## 5. Train mean |SHAP| by in-model feature

Computed **after freeze**, on **TRAIN**, in-model features only. Method: shap.TreeExplainer on all TRAIN rows, in-model WOE features only (non-linear_tree LGBM; path-dependent TreeSHAP). utils.risk_skills.generate_shap_summary also run on TRAIN.
n_rows=14000.

Top features by mean |SHAP|: employment_status (1.297588), debt_to_income_ratio (0.959357), credit_score (0.569642), grade_subgrade (0.430064), interest_rate (0.078837), current_balance (0.064007), education_level (0.060312), loan_purpose (0.055864)

| feature | mean_abs_shap |
| --- | --- |
| employment_status | 1.297588 |
| debt_to_income_ratio | 0.959357 |
| credit_score | 0.569642 |
| grade_subgrade | 0.430064 |
| interest_rate | 0.078837 |
| current_balance | 0.064007 |
| education_level | 0.060312 |
| loan_purpose | 0.055864 |
| loan_amount | 0.046562 |
| age | 0.043506 |
| marital_status | 0.040253 |
| annual_income | 0.036397 |
| monthly_income | 0.033499 |
| delinquency_history | 0.031774 |
| num_of_open_accounts | 0.031711 |
| total_credit_limit | 0.025045 |
| loan_term | 0.018765 |
| num_of_delinquencies | 0.017694 |
| installment | 0.016042 |
| public_records | 0.002958 |

- `employment_status` present in SHAP table? **YES**
- `gender` present in SHAP table? **NO**

`utils.risk_skills.generate_shap_summary` on TRAIN (same frozen model, no refit):
`{'xai_method': 'SHAP (Kernel/Tree)', 'n_rows': 2000, 'global_importance_ranking': [['employment_status', 1.2891], ['debt_to_income_ratio', 0.9475], ['credit_score', 0.5437], ['grade_subgrade', 0.4337], ['interest_rate', 0.0793], ['current_balance', 0.0641], ['education_level', 0.0594], ['loan_purpose', 0.0541], ['loan_amount', 0.0482], ['age', 0.0439], ['marital_status', 0.0409], ['annual_income', 0.0365], ['monthly_income', 0.0319], ['delinquency_history', 0.0317], ['num_of_open_accounts', 0.0294], ['total_credit_limit', 0.0258], ['loan_term', 0.0199], ['num_of_delinquencies', 0.017], ['installment', 0.0154], ['public_records', 0.0037]]}`

## 6. Artifact paths + freeze-before-test confirmation

- `artifacts/lgbm_emp_in_no_gender_model.joblib`
- `artifacts/lgbm_emp_in_no_gender_encoders.joblib`
- `artifacts/lgbm_emp_in_no_gender_meta.json`
- `artifacts/lgbm_emp_in_no_gender_oof_pds.csv`
- `artifacts/lgbm_emp_in_no_gender_test_metrics.json`
- `artifacts/lgbm_emp_in_no_gender_run_report.md`
- `artifacts/lgbm_emp_in_no_gender_X_train.csv`
- `artifacts/lgbm_emp_in_no_gender_y_train.csv`
- `artifacts/lgbm_emp_in_no_gender_X_test.csv`
- `artifacts/lgbm_emp_in_no_gender_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`.

- Freeze timestamp (UTC), written by the train script before this eval script ran: **2026-08-30T13:25:39.431309+00:00**
- `freeze.test_looked_at` at freeze time: `False`
- `freeze.test_metrics` at freeze time: `None`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select=False`)
- Eval script never calls `fit` / Optuna / early stopping. It only `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split: `scripts/train.py::load_and_split` unchanged (`test_size=0.3`, `stratify=default`, `random_state=42`).
  Train n=14000 bad_rate=0.2001 (bads=2801);
  Test n=6000 bad_rate=0.2002 (bads=1201).
- WOE 5-bin encoders (`fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS` from `utils.config`) fit on TRAIN only.
