# Ensemble no-gender (LR + LGBM gbdt + DART + CatBoost) — run report

Four-base ensemble on the frozen `scripts/train.py` split. Gender is out. `employment_status` is a normal in-model WOE feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#5 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `scripts/train_ensemble_no_gender.py` (tune / fit on TRAIN only, then freeze)
- Eval script: `scripts/eval_ensemble_no_gender.py` (**never fits**; `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- WOE: `utils.woe_encoding.fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS=5` from `utils.config` (thresholds unchanged: KS_MIN=0.3, AUC_MIN=0.7)

Eval never fits. Encoders, four bases, and stack meta are loaded from freeze artifacts.

Split verified: Train n=14000 bad_rate=0.2001 (bads=2801); Test n=6000 bad_rate=0.2002 (bads=1201).

## 2. Exact 20 in-model features

`gender_in_model=false`. `employment_status_in_model=true`. `monotone_constraints=false`. `interaction_constraints=false`.

1. age
2. marital_status
3. education_level
4. annual_income
5. monthly_income
6. employment_status
7. debt_to_income_ratio
8. credit_score
9. loan_amount
10. loan_purpose
11. interest_rate
12. loan_term
13. installment
14. grade_subgrade
15. num_of_open_accounts
16. total_credit_limit
17. current_balance
18. delinquency_history
19. public_records
20. num_of_delinquencies

- `loan_paid_back` in feature matrix: **false**
- IV-drop: **false**. VIF-drop: **false**. `credit_score` stays.
- Confirmed absent from `feature_name_` / `coef`, encoder keys, persisted X_train/X_test, and SHAP/coef tables: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model WOE feature, not overlay, not singleton interaction group)
- All four bases actually trained: `['lr', 'lgbm_gbdt', 'lgbm_dart', 'catboost']`

## 3. OOF AUC / KS (train only; used to select average vs stack)

| model | OOF_AUC | OOF_KS | OOF_Gini |
| --- | --- | --- | --- |
| LR (WOE) | 0.8806 | 0.5644 | 0.7613 |
| LightGBM gbdt | 0.8893 | 0.5823 | 0.7785 |
| LightGBM DART | 0.8875 | 0.5799 | 0.775 |
| CatBoost | 0.8877 | 0.5785 | 0.7754 |
| Simple average | 0.8872 | 0.5785 | 0.7743 |
| Stack meta-LR | 0.8885 | 0.5811 | 0.777 |

- Which ensemble OOF selects: **stack** (average OOF AUC=0.8872, stack OOF AUC=0.8885)
- Selection used Train OOF AUC only. Test was not used to pick among bases or between average vs stack.
- LR C grid (train OOF AUC): [{'C': 0.01, 'oof_auc': 0.877693, 'oof_ks': 0.553196}, {'C': 0.03, 'oof_auc': 0.879027, 'oof_ks': 0.557396}, {'C': 0.1, 'oof_auc': 0.879838, 'oof_ks': 0.559043}, {'C': 0.3, 'oof_auc': 0.880348, 'oof_ks': 0.561171}, {'C': 1.0, 'oof_auc': 0.880563, 'oof_ks': 0.563043}, {'C': 3.0, 'oof_auc': 0.880619, 'oof_ks': 0.563938}, {'C': 10.0, 'oof_auc': 0.88064, 'oof_ks': 0.564385}, {'C': 30.0, 'oof_auc': 0.880638, 'oof_ks': 0.564116}, {'C': 100.0, 'oof_auc': 0.880636, 'oof_ks': 0.563938}]
- GBDT n_estimators_final = max(50, round(mean best_iteration)) = **152** (fold best_iterations = [147, 132, 159, 171, 153])
- GBDT trials completed: 50/50; DART: 50/50; CatBoost: 50/50
- DART: no early_stopping. GBDT used train-fold early_stopping=30. CatBoost used searched `iterations` with no monotone.

### 3.1 Winning hyperparameters

- LR: {'penalty': 'l2', 'C': 10.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
- LGBM gbdt final: {'objective': 'binary', 'boosting_type': 'gbdt', 'random_state': 42, 'verbosity': -1, 'n_jobs': 4, 'subsample_freq': 1, 'n_estimators': 152, 'learning_rate': 0.05435025690302303, 'num_leaves': 53, 'max_depth': 3, 'min_child_samples': 103, 'subsample': 0.7312202514997275, 'colsample_bytree': 0.9728984059205478, 'reg_alpha': 0.4366724940335208, 'reg_lambda': 0.042964739661628434}
- LGBM DART final: {'objective': 'binary', 'boosting_type': 'dart', 'random_state': 42, 'verbosity': -1, 'n_jobs': 4, 'n_estimators': 499, 'learning_rate': 0.054048018851067, 'num_leaves': 64, 'max_depth': 3, 'min_child_samples': 294, 'feature_fraction': 0.8835726202569505, 'reg_alpha': 3.1471274492014585, 'reg_lambda': 0.029616152297803622, 'drop_rate': 0.24478158628129132, 'skip_drop': 0.6928734309755}
- CatBoost final: {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_state': 42, 'verbose': 0, 'allow_writing_files': False, 'thread_count': 4, 'depth': 4, 'learning_rate': 0.03258548935284139, 'l2_leaf_reg': 2.596756291129093, 'iterations': 479, 'min_data_in_leaf': 38, 'subsample': 0.9721880459666276, 'bootstrap_type': 'Bernoulli'}
- Stack meta: {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42, 'fitted_on': 'OOF PDs of four bases (not full-train in-sample PDs)'}

## 4. After freeze: Test AUC / KS / Gini / PSI

PSI train side = OOF PD of that same model/ensemble (not in-sample full-train PD).

| model | Test_AUC | Test_KS | Test_Gini | Test_PSI |
| --- | --- | --- | --- | --- |
| LR (WOE) | 0.8823 | 0.5695 | 0.7646 | 0.0027 |
| LightGBM gbdt | 0.8895 | 0.5816 | 0.779 | 0.0091 |
| LightGBM DART | 0.8869 | 0.5779 | 0.7738 | 0.0038 |
| CatBoost | 0.8878 | 0.5804 | 0.7756 | 0.0044 |
| Simple average | 0.8871 | 0.5764 | 0.7743 | 0.0024 |
| Stack meta-LR | 0.8883 | 0.5795 | 0.7766 | 0.0035 |

Internal gates (unchanged, not a go/no-go): AUC ≥ 0.7, KS ≥ 0.3. OOF-selected ensemble `stack` Test AUC=0.8883 (PASS vs AUC_MIN), KS=0.5795 (PASS vs KS_MIN).

### 4.1 Plain yes/no vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858** / KS 0.5777 / Gini 0.7715 / PSI 0.0021 (CV 0.8862); PR #2 stack Test AUC **0.8885** (had gender); PR #4 Test AUC **0.8859** (emp in, gender out, monotone LGBM); PR #5 DART Test AUC 0.7645.

| ensemble | Test_AUC | beat_0.8858 | beat_0.8885 | beat_0.8859 |
| --- | --- | --- | --- | --- |
| average | 0.8871 | YES | NO | YES |
| stack | 0.8883 | YES | NO | YES |

- Did **average** beat 0.8858? **YES**
- Did **average** beat 0.8885? **NO**
- Did **average** beat 0.8859? **YES**
- Did **stack** beat 0.8858? **YES**
- Did **stack** beat 0.8885? **NO**
- Did **stack** beat 0.8859? **YES**

## 5. TRAIN explains after freeze (in-model features only)

- LR abs-coef top: 1. debt_to_income_ratio (1.602244); 2. current_balance (1.582467); 3. education_level (1.300558); 4. employment_status (1.288471); 5. loan_amount (1.171731); 6. marital_status (0.978944); 7. grade_subgrade (0.94344); 8. loan_purpose (0.916411). employment_status rank = **4**. gender absent: **YES**.
- LGBM gbdt mean |SHAP| top: 1. employment_status (1.185698); 2. debt_to_income_ratio (0.830874); 3. credit_score (0.500282); 4. grade_subgrade (0.354993); 5. interest_rate (0.060224); 6. annual_income (0.042879); 7. current_balance (0.038794); 8. loan_amount (0.033436). employment_status rank = **1**. gender absent: **YES**. Method: shap.TreeExplainer (TreeSHAP) on all TRAIN rows — lgbm_gbdt
- LGBM DART mean |SHAP| top: 1. employment_status (1.24186); 2. debt_to_income_ratio (0.877479); 3. credit_score (0.575471); 4. grade_subgrade (0.327437); 5. interest_rate (0.066664); 6. loan_amount (0.036351); 7. education_level (0.034107); 8. current_balance (0.029571). employment_status rank = **1**. gender absent: **YES**. Method: shap.TreeExplainer (TreeSHAP) on all TRAIN rows — lgbm_dart
- CatBoost mean |SHAP| top: 1. employment_status (1.465034); 2. debt_to_income_ratio (1.073518); 3. credit_score (0.639332); 4. grade_subgrade (0.459358); 5. interest_rate (0.089816); 6. education_level (0.054397); 7. loan_purpose (0.054129); 8. current_balance (0.052272). employment_status rank = **1**. gender absent: **YES**. Method: shap.TreeExplainer (TreeSHAP) on all TRAIN rows — catboost

## 6. Stack meta coefs and freeze-before-test

| base_pd | meta_coef |
| --- | --- |
| pd_lr | 0.434013 |
| pd_lgbm_gbdt | 2.379526 |
| pd_lgbm_dart | 2.116687 |
| pd_catboost | 2.507628 |

- Stack meta intercept: **-3.193918**
- Freeze timestamp UTC: **2026-08-31T09:14:06.319383+00:00**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- `computed_after_freeze=true` in `artifacts/ensemble_no_gender_test_metrics.json`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)

Artifacts:

- `artifacts/ensemble_no_gender_encoders.joblib`
- `artifacts/ensemble_no_gender_lr.joblib`
- `artifacts/ensemble_no_gender_lgbm_gbdt.joblib`
- `artifacts/ensemble_no_gender_lgbm_dart.joblib`
- `artifacts/ensemble_no_gender_catboost.joblib`
- `artifacts/ensemble_no_gender_stack_meta.joblib`
- `artifacts/ensemble_no_gender_meta.json`
- `artifacts/ensemble_no_gender_oof_pds.csv`
- `artifacts/ensemble_no_gender_test_metrics.json`
- `artifacts/ensemble_no_gender_run_report.md`
- `artifacts/ensemble_no_gender_X_train.csv` / `artifacts/ensemble_no_gender_y_train.csv` / `artifacts/ensemble_no_gender_X_test.csv` / `artifacts/ensemble_no_gender_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`, `artifacts/lgbm_dart_monotone_*`.

## 7. Base install / runtime

All four bases were actually trained (LR, LightGBM gbdt, LightGBM DART, CatBoost). No base was dropped. CatBoost import succeeded. DART runtime completed without early_stopping.

Governance thresholds in `utils/config.py` were not changed.
