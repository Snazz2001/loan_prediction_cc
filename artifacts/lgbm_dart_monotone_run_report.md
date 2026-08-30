# LightGBM DART monotone (6-feature, emp singleton) — run report

One model: `lightgbm.LGBMClassifier` (`objective=cross_entropy`, `boosting_type=dart`, `random_state=42`).
Not a stack. LogisticRegression and RandomForest were not trained. Not `linear_tree`.
`employment_status` is in-model, additive-only (singleton interaction group). No overlay.
No policy table as a model input. Frozen split reused from `scripts/train.py::load_and_split`
(unchanged). Last-run `artifacts/lgbm_model.joblib` and PR #1–#4 artifact names were not written.
Early stopping was **not** used (incompatible with DART). `n_estimators` was searched.

## 1. Script paths

- Train script: `scripts/train_lgbm_dart_monotone.py` (tune + freeze on TRAIN only)
- Eval script: `scripts/eval_lgbm_dart_monotone.py` (**never fits**; loads artifacts; applies frozen WOE to Test; writes this report)
- Eval confirmation: no `.fit(`, no Optuna study, no `LGBMClassifier(` constructor in this file.

## 2. Exact 6 in-model features in order

`gender` is **ABSENT** from the in-model feature list, the WOE encoder keys, the encoded
matrices, the LightGBM `feature_name_`, and the SHAP table.
`employment_status` is **IN-MODEL**, additive-only singleton interaction group.
`loan_paid_back` is not a feature (`default = 1 - loan_paid_back` is the target only).
No IV/VIF re-run. No extra fields. WOE 5-bin TRAIN-ONLY via `fit_woe_encoder` /
`apply_woe_encoder` (`WOE_BINS` from `utils.config`).

In-model features (6, asserted exact order): `employment_status, debt_to_income_ratio, interest_rate, grade_subgrade, delinquency_history, num_of_delinquencies`

| order | feature |
| --- | --- |
| 1 | employment_status |
| 2 | debt_to_income_ratio |
| 3 | interest_rate |
| 4 | grade_subgrade |
| 5 | delinquency_history |
| 6 | num_of_delinquencies |

- `gender_in_model` = **false**
- `employment_status_in_model` = **true**
- `boosting_type` = **dart**
- `loan_paid_back` in model? **NO**
- monotone_constraints = `[-1, -1, -1, -1, -1, -1]` (aligned to the 6-column order; signs not searched)
- interaction_constraints = `[['employment_status'], ['debt_to_income_ratio', 'interest_rate', 'grade_subgrade', 'delinquency_history', 'num_of_delinquencies']]` (not relaxed)

## 3. Train OOF AUC/KS vs last CV 0.8862

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **0.7707** (raw 0.7706795938166943)
- Train OOF KS (monitoring only, not used to select): 0.506
- Last-run CV AUC (compare only, not retrained): 0.8862
- Beat last-run CV 0.8862? **NO**
- n_trials completed: 50 / 50
- n_estimators (searched; used for final refit): **406**
- Fold AUCs (winning params re-run): [0.744013, 0.766684, 0.778335, 0.791048, 0.769138]
- Fold KS (monitoring): [0.462054, 0.514286, 0.523214, 0.516964, 0.513265]
- OOF PDs persisted: `artifacts/lgbm_dart_monotone_oof_pds.csv`
- early_stopping used? **NO**

Best hyperparameters (search params; fixed `objective=cross_entropy`, `boosting_type=dart`,
`random_state=42`, monotone all −1, emp singleton interaction):

| hyperparameter | value |
| --- | --- |
| learning_rate | 0.018623353562709655 |
| num_leaves | 48.0 |
| max_depth | 5.0 |
| min_child_samples | 752.0 |
| feature_fraction | 0.6787252920759664 |
| colsample_bytree | 0.6787252920759664 |
| reg_alpha | 3.0124989986733963 |
| reg_lambda | 0.1671382022795037 |
| drop_rate | 0.20531181715063013 |
| skip_drop | 0.6097155734197113 |
| n_estimators | 406.0 |

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 4. After freeze: Test metrics vs last-run / stack / PR #4

Plain yes/no on beat 0.8858 / 0.8885 / 0.8859 (no go/no-go declaration):

- Beat last-run Test AUC 0.8858? **NO**. NO — Test AUC 0.7645 did not beat last-run Test AUC 0.8858.
- Beat stack Test AUC 0.8885? **NO**. NO — Test AUC 0.7645 did not beat stack Test AUC 0.8885.
- Beat PR #4 Test AUC 0.8859? **NO**. NO — Test AUC 0.7645 did not beat PR #4 Test AUC 0.8859.

Test AUC 0.7645 did not beat last-run champion Test AUC 0.8858.

| metric | this candidate (Test) | last-run champion (Test) | beat 0.8858? | PR #2 stack (Test) | beat 0.8885? | PR #4 emp-in no-gender (Test) | beat 0.8859? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUC | **0.7645** | 0.8858 | **NO** | 0.8885 | **NO** | 0.8859 | **NO** |
| KS | **0.4989** | 0.5777 | n/a | n/a | n/a | n/a | n/a |
| Gini | **0.5289** | 0.7715 | n/a | n/a | n/a | n/a | n/a |
| PSI (train PD vs test PD) | **0.0011** | 0.0021 | n/a | n/a | n/a | n/a | n/a |

- PSI status: Stable (<0.10) - No action needed
- This candidate Train refit (not OOF) AUC/KS: 0.7728 / 0.506
- Last-run, stack, and PR #4 numbers are documented comparators; they were not retrained in this run.
- PR #4 is not a go champion.

| model | sample | AUC | Gini | KS | PSI |
| --- | --- | --- | --- | --- | --- |
| LightGBM DART monotone (this candidate) | Train (refit, not OOF) | 0.7728 | 0.5455 | 0.506 | n/a |
| LightGBM DART monotone (this candidate) | Test (after freeze) | 0.7645 | 0.5289 | 0.4989 | 0.0011 |
| Last-run champion LightGBM (compare only, not retrained) | Test | 0.8858 | 0.7715 | 0.5777 | 0.0021 |
| PR #2 stack LR+RF+LGBM (compare only, not retrained) | Test | 0.8885 | n/a | n/a | n/a |
| PR #4 emp-in no-gender LightGBM (compare only, not a go champion) | Test | 0.8859 | n/a | n/a | n/a |

## 5. Train mean |SHAP| ranking

Computed **after freeze**, on **TRAIN**, in-model features only. Method: shap.TreeExplainer (TreeSHAP) on all TRAIN rows
n_rows=14000.

Ranking by mean |SHAP|: employment_status (0.082269), debt_to_income_ratio (0.0), interest_rate (0.0), grade_subgrade (0.0), delinquency_history (0.0), num_of_delinquencies (0.0)

| feature | mean_abs_shap |
| --- | --- |
| employment_status | 0.082269 |
| debt_to_income_ratio | 0.0 |
| interest_rate | 0.0 |
| grade_subgrade | 0.0 |
| delinquency_history | 0.0 |
| num_of_delinquencies | 0.0 |

- `employment_status` present in SHAP table? **YES**
- `employment_status` rank (1 = highest mean |SHAP|): **1**
- `gender` present in SHAP table? **NO**
- Collapsed to an employment lookup (other 5 features ~0 SHAP)? **YES**. YES — the model collapsed to an employment lookup (other 5 features ~0 SHAP). This is reported, not fixed (interaction_constraints / emp / monotone left as written).

`utils.risk_skills.generate_shap_summary` on TRAIN (same frozen model, no refit):
`{'xai_method': 'SHAP (Kernel/Tree)', 'global_importance_ranking': [['employment_status', 0.0804], ['debt_to_income_ratio', 0.0], ['interest_rate', 0.0], ['grade_subgrade', 0.0], ['delinquency_history', 0.0], ['num_of_delinquencies', 0.0]]}`

## 6. Artifact paths + freeze-before-test confirmation

- `artifacts/lgbm_dart_monotone_model.joblib`
- `artifacts/lgbm_dart_monotone_encoders.joblib`
- `artifacts/lgbm_dart_monotone_meta.json`
- `artifacts/lgbm_dart_monotone_oof_pds.csv`
- `artifacts/lgbm_dart_monotone_test_metrics.json`
- `artifacts/lgbm_dart_monotone_run_report.md`
- `artifacts/lgbm_dart_monotone_X_train.csv`
- `artifacts/lgbm_dart_monotone_y_train.csv`
- `artifacts/lgbm_dart_monotone_X_test.csv`
- `artifacts/lgbm_dart_monotone_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`.

- Freeze timestamp (UTC), written by the train script before this eval script ran: **2026-08-30T15:57:35.847264+00:00**
- `freeze.test_looked_at` at freeze time: `False`
- `freeze.test_metrics` at freeze time: `None`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select=False`)
- Freeze before test look? **YES**
- Eval script never calls `fit` / Optuna / early stopping. It only applies the frozen encoder, then `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split: `scripts/train.py::load_and_split` unchanged (`test_size=0.3`, `stratify=default`, `random_state=42`).
  Train n=14000 bad_rate=0.2001 (bads=2801);
  Test n=6000 bad_rate=0.2002 (bads=1201).
- WOE 5-bin encoders (`fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS` from `utils.config`) fit on TRAIN only; applied to Test in this eval script.
- HEAD of this PR branch is the commit that contains these scripts/artifacts.

## 7. Outcome (plain)

- Beat last-run Test AUC 0.8858? **NO**
- Beat stack Test AUC 0.8885? **NO**
- Beat PR #4 Test AUC 0.8859? **NO**
- Employment-lookup collapse? **YES**
