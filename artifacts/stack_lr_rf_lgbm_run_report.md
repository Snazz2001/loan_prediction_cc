# Stack LR + RF + LightGBM — run report

Leak-free stacking candidate on the frozen `scripts/train.py` split. One stack only (three bases + one meta-learner). No CatBoost / EBM / AutoML. Last-run `artifacts/lgbm_model.joblib` and PR #1 linear-tree artifacts were not written.

## 1. Paths

- Train script: `scripts/train_stack_lr_rf_lgbm.py` (tune / fit on TRAIN only, then freeze)
- Eval script: `scripts/eval_stack_lr_rf_lgbm.py` (**never fits**; `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- WOE: `utils.woe_encoding.fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS=5` from `utils.config` (thresholds unchanged)

Eval never fits. Encoders, bases, and meta are loaded from freeze artifacts.

## 2. Train OOF stack AUC vs last-run CV 0.8862

- **Train OOF stack AUC = 0.8891** (raw 0.8890707491957113)
- Train OOF KS = 0.5824; Gini = 0.7781
- Last-run CV AUC = 0.8862
- **Beat last-run CV 0.8862? YES**
- Optuna TPESampler seed=42, 30/30 trials, 5-fold StratifiedKFold
- Objective = TRAIN OOF stack AUC (meta fitted on OOF base PDs vs train y, scored on those OOF PDs)
- LGBM `n_estimators_final` = max(50, round(mean OOF best_iteration)) = **225** (fold best_iterations = [218, 278, 179, 213, 236])
- Meta LR C selected on the frozen OOF PD matrix only: **10.0** (grid scores: {'0.1': 0.885739, '0.5': 0.887675, '1.0': 0.888327, '2.0': 0.888705, '10.0': 0.889071})

Selection used Train OOF stack AUC only. Test was not used to choose this candidate.

### 2.1 Winning hyperparameters

| component   | hyperparameter   | value               |
|:------------|:-----------------|:--------------------|
| lr          | penalty          | l2                  |
| lr          | C                | 19.832294665045822  |
| lr          | solver           | lbfgs               |
| lr          | max_iter         | 1000                |
| lr          | random_state     | 42                  |
| rf          | n_estimators     | 337                 |
| rf          | max_depth        | 5                   |
| rf          | min_samples_leaf | 72                  |
| rf          | n_jobs           | -1                  |
| rf          | random_state     | 42                  |
| lgbm_tune   | objective        | binary              |
| lgbm_tune   | boosting_type    | gbdt                |
| lgbm_tune   | verbosity        | -1                  |
| lgbm_tune   | random_state     | 42                  |
| lgbm_tune   | n_estimators     | 1000                |
| lgbm_tune   | learning_rate    | 0.03139974180336646 |
| lgbm_tune   | num_leaves       | 18                  |
| lgbm_tune   | max_depth        | 4                   |
| lgbm_final  | objective        | binary              |
| lgbm_final  | boosting_type    | gbdt                |
| lgbm_final  | verbosity        | -1                  |
| lgbm_final  | random_state     | 42                  |
| lgbm_final  | n_estimators     | 225                 |
| lgbm_final  | learning_rate    | 0.03139974180336646 |
| lgbm_final  | num_leaves       | 18                  |
| lgbm_final  | max_depth        | 4                   |
| meta        | penalty          | l2                  |
| meta        | C                | 10.0                |
| meta        | solver           | lbfgs               |
| meta        | max_iter         | 1000                |
| meta        | random_state     | 42                  |

## 3. After freeze: Test metrics vs last-run champion

**Test AUC 0.8885 DID beat last-run Test AUC 0.8858.**

| metric                                        |   this_candidate_test |   last_run_champion | beat_last_run   |
|:----------------------------------------------|----------------------:|--------------------:|:----------------|
| AUC                                           |                0.8885 |              0.8858 | YES             |
| KS                                            |                0.5795 |              0.5777 | n/a             |
| Gini                                          |                0.7769 |              0.7715 | n/a             |
| PSI (train OOF stacked PD vs test stacked PD) |                0.0028 |              0.0021 | n/a             |

- Test Validation_Rating: Good (0.40 - 0.60)
- Internal gates (unchanged): AUC ≥ 0.7, KS ≥ 0.3. This Test AUC PASS, KS PASS.
- PSI status: Stable (<0.10) - No action needed (below watch threshold, no action needed)
- Confusion @ KS cutoff 0.1312: TN=3760 FP=1039 FN=246 TP=955; accuracy=0.7858 precision=0.4789 recall=0.7952 specificity=0.7835

Train OOF stacked PD (meta on OOF 14000×3, not full-train in-sample base PDs): AUC=0.8891 KS=0.5824 Gini=0.7781.

## 4. Per-base OOF AUC (did the stack add anything?)

| model                          |   OOF_AUC |   OOF_KS |   OOF_Gini |
|:-------------------------------|----------:|---------:|-----------:|
| LR (base)                      |    0.8806 |   0.5643 |     0.7612 |
| RandomForest (base)            |    0.8699 |   0.5432 |     0.7399 |
| LightGBM (base)                |    0.8892 |   0.5815 |     0.7784 |
| Stack meta-LR (this candidate) |    0.8891 |   0.5824 |     0.7781 |

Last-run CV AUC = 0.8862. Compare each base and the stack against that number and against each other. Stack OOF AUC minus best base OOF AUC = -0.0001.

## 5. Artifact paths, freeze-before-test, no test-label leakage

- `artifacts/stack_lr_rf_lgbm_bases.joblib` — dict of lr / rf / lgbm fitted on all 14000 train rows
- `artifacts/stack_lr_rf_lgbm_meta.joblib` — meta LR fitted on OOF PDs (not refit on in-sample full-train base PDs)
- `artifacts/stack_lr_rf_lgbm_encoders.joblib` — WOE encoders fit on TRAIN only
- `artifacts/stack_lr_rf_lgbm_meta.json` — freeze record
- `artifacts/stack_lr_rf_lgbm_oof_pds.csv` — 14000 × 3 base PDs + y
- `artifacts/stack_lr_rf_lgbm_test_metrics.json` — test metrics written by this eval script only
- `artifacts/stack_lr_rf_lgbm_run_report.md` — this report

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`.

- Freeze timestamp UTC: **2026-08-29T05:29:17.340260+00:00**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- Frozen split verified again at eval: Train n=14000 bad_rate=0.2001; Test n=6000 bad_rate=0.2002
- `loan_paid_back` in feature matrix: **false** (dropped in `load_and_split`; target is `default = 1 - loan_paid_back`)
- IV-drop: **false**. VIF-drop: **false**. All 21 features kept: age, gender, marital_status, education_level, annual_income, monthly_income, employment_status, debt_to_income_ratio, credit_score, loan_amount, loan_purpose, interest_rate, loan_term, installment, grade_subgrade, num_of_open_accounts, total_credit_limit, current_balance, delinquency_history, public_records, num_of_delinquencies
- Test labels were never used to fit or select. Eval does not call `fit`.

## 6. Plain statement on Test AUC vs 0.8858

Test AUC 0.8885 DID beat last-run Test AUC 0.8858.

## 7. Split / features (diagnostics)

| sample | n | bad_rate |
| --- | --- | --- |
| Train | 14000 | 0.2001 |
| Test | 6000 | 0.2002 |

IV table computed on TRAIN as a diagnostic only (no screening drop):

| feature              |     iv | predictive_power                |
|:---------------------|-------:|:--------------------------------|
| employment_status    | 1.9096 | Suspicious / Overfitting (>0.5) |
| debt_to_income_ratio | 0.3211 | Strong (0.3 - 0.5)              |
| grade_subgrade       | 0.2755 | Medium (0.1 - 0.3)              |
| credit_score         | 0.2601 | Medium (0.1 - 0.3)              |
| interest_rate        | 0.08   | Weak (0.02 - 0.1)               |
| delinquency_history  | 0.041  | Weak (0.02 - 0.1)               |
| num_of_delinquencies | 0.0317 | Weak (0.02 - 0.1)               |
| loan_purpose         | 0.0073 | Unpredictive (<0.02)            |
| education_level      | 0.0037 | Unpredictive (<0.02)            |
| monthly_income       | 0.003  | Unpredictive (<0.02)            |
| annual_income        | 0.003  | Unpredictive (<0.02)            |
| current_balance      | 0.0022 | Unpredictive (<0.02)            |
| loan_amount          | 0.0022 | Unpredictive (<0.02)            |
| num_of_open_accounts | 0.0021 | Unpredictive (<0.02)            |
| installment          | 0.002  | Unpredictive (<0.02)            |
| marital_status       | 0.0019 | Unpredictive (<0.02)            |
| total_credit_limit   | 0.0016 | Unpredictive (<0.02)            |
| age                  | 0.0015 | Unpredictive (<0.02)            |
| gender               | 0.0004 | Unpredictive (<0.02)            |
| public_records       | 0.0002 | Unpredictive (<0.02)            |
| loan_term            | 0.0001 | Unpredictive (<0.02)            |

`employment_status` IV is in the Suspicious / Overfitting band and is **kept** for evaluation; it is **not cleared for production** pending independent validation and fair-lending review.

Governance thresholds in `utils/config.py` were not changed.
