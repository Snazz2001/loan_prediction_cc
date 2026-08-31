# DANet candidate (no gender, frozen split) — run report

ONE DANet (Deep Abstract Networks, Chen/Huang et al. AAAI 2022) on the frozen `scripts/train.py` split. Gender is out. `employment_status` is a normal in-model feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#7 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `scripts/train_danet.py` (tune / fit on TRAIN only, then freeze)
- Eval script: `scripts/eval_danet.py` (**never fits**; torch forward + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- Vendored official DANet: `third_party/danet/` (AbstractLayer / LearnableLocality / Entmax15)

Eval never fits. Preprocess and DANet weights are loaded from freeze artifacts.

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
- Confirmed absent from persisted X_train/X_test, preprocess feature_order / cat maps, and permutation table: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model feature, not overlay, not singleton interaction group)

## 3. neural_impl

- library: `vendored WhatAShot/DANet (official)`
- class: `third_party.danet.DANet.DANet`
- version / upstream commit: `b007c57121ec9082f6ef19ec7465d9df70767c26`
- constructor: `DANet(input_dim=20, num_classes=2, layer_num=2, base_outdim=16, k=8, virtual_batch_size=256, drop_rate=0.2)`
- source: `https://github.com/WhatAShot/DANet`
- ABSTLAY: AbstractLayer + LearnableLocality(Entmax15 k-masks)
- official vs pytorch-tabular fallback: **official vendored** (`not used; official vendored subset imported successfully`)
- objective: CrossEntropyLoss over 2 classes; PD = softmax(logits)[:, 1]
- optimizer: Adam (official repo uses QHAdam; Adam used here to avoid qhoptim on CPU)
- `shrunk_for_cpu`: **True**
- winner hyperparams: {'layer_num': 2, 'k': 8, 'base_outdim': 16, 'drop_rate': 0.2, 'lr': 0.003}
- n_epochs_final: **14**
- n_layers / k / width: 2 / 8 / 16
- notes: Official default is layer=20, k=5, base_outdim=64, max_epochs=4000, batch=8192. CPU grid: layer_num in {2,4}, k in {3,5,8}, base_outdim in {16,32}, max_epochs=20, patience=5, batch=512, virtual_batch_size=256, grid n=6 (not 50 Optuna trials). n_epochs_final = mean fold best epoch.
- device: cpu (torch cuda available: False)
- Package versions: {'python': '3.12.3', 'torch': '2.13.0+cpu', 'numpy': '2.4.4', 'pandas': '3.0.5', 'sklearn': '1.9.0', 'joblib': '1.6.0', 'danet_library': 'vendored WhatAShot/DANet (official)', 'danet_upstream_commit': 'b007c57121ec9082f6ef19ec7465d9df70767c26', 'danet_source_url': 'https://github.com/WhatAShot/DANet'}

## 4. OOF AUC / KS / Gini (train only)

| split | AUC | KS | Gini | PSI |
| --- | --- | --- | --- | --- |
| OOF (train folds) | 0.8624 | 0.5222 | 0.7247 |  |
| Test (after freeze) | 0.8655 | 0.5412 | 0.731 | 0.0329 |

- OOF AUC = **0.8624**, OOF KS = **0.5222**, OOF Gini = **0.7247**
- Fold AUCs: [0.8650980548469387, 0.8656560905612246, 0.8644180484693877, 0.8709219547193878, 0.8615835468947415]
- Fold KS: [0.553125, 0.540625, 0.5236607142857143, 0.5397321428571429, 0.530728560862812]
- Fold best epochs: [11, 14, 11, 18, 17]
- Selection used Train OOF AUC only. Test was not used to pick hyperparameters.

### 4.1 Grid scores (train OOF)

| layer_num | k | base_outdim | drop_rate | lr | oof_auc | oof_ks | fold_aucs | fold_best_epochs | n_epochs_final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 5 | 16 | 0.1 | 0.001 | 0.855594 | 0.520349 | [0.854019, 0.853665, 0.85672, 0.860867, 0.855138] | [18, 18, 20, 19, 20] | 19 |
| 2 | 5 | 32 | 0.1 | 0.001 | 0.859765 | 0.523817 | [0.858867, 0.85727, 0.8635, 0.86471, 0.863176] | [18, 19, 20, 20, 20] | 19 |
| 2 | 3 | 16 | 0.0 | 0.0003 | 0.841602 | 0.490818 | [0.831683, 0.843671, 0.842643, 0.849458, 0.848692] | [20, 19, 19, 20, 18] | 19 |
| 4 | 5 | 16 | 0.1 | 0.001 | 0.853899 | 0.517064 | [0.850907, 0.851929, 0.860322, 0.859039, 0.855435] | [18, 19, 18, 19, 15] | 18 |
| 2 | 8 | 16 | 0.2 | 0.003 | 0.862352 | 0.52224 | [0.865098, 0.865656, 0.864418, 0.870922, 0.861584] | [11, 14, 11, 18, 17] | 14 |
| 4 | 5 | 32 | 0.2 | 0.0003 | 0.84785 | 0.505366 | [0.847436, 0.847359, 0.848162, 0.849665, 0.848736] | [18, 19, 20, 20, 19] | 19 |

## 5. After freeze: Test AUC / KS / Gini / PSI

PSI expected side = OOF PD; actual side = Test PD (not in-sample full-train PD).

- Test AUC = **0.8655**
- Test KS = **0.5412**
- Test Gini = **0.731**
- Test PSI = **0.0329** (Stable (<0.10) - No action needed)
- Test PSI above 0.10: **NO** (fact; not a go/no-go)
- Internal bars (unchanged, not a go/no-go): AUC ≥ 0.7 → YES; KS ≥ 0.3 → YES

### 5.1 Plain YES/NO vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858** / KS 0.5777 / Gini 0.7715 / PSI 0.0021 (CV 0.8862); PR #2 stack **0.8885** (had gender); PR #4 **0.8859**; PR #6 stack **0.8883**; PR #6 gbdt-alone **0.8895**; PR #7 stack **0.8951**; PR #7 FT-alone **0.8963**.

| comparator | beat |
| --- | --- |
| last-run 0.8858 | NO |
| PR#2 stack 0.8885 | NO |
| PR#4 0.8859 | NO |
| PR#6 stack 0.8883 | NO |
| PR#6 gbdt-alone 0.8895 | NO |
| PR#7 stack 0.8951 | NO |
| PR#7 FT-alone 0.8963 | NO |

- Beat 0.8858? **NO**
- Beat 0.8885? **NO**
- Beat 0.8859? **NO**
- Beat 0.8883? **NO**
- Beat 0.8895? **NO**
- Beat 0.8951? **NO**
- Beat 0.8963? **NO**

## 6. TRAIN permutation (after freeze, in-model features only)

- Method: permutation importance on TRAIN sample n=500, seed=42, metric=AUC drop vs baseline AUC=0.910556 — danet
- Top features: 1. employment_status (0.25933); 2. debt_to_income_ratio (0.101776); 3. credit_score (0.09256); 4. delinquency_history (0.009404); 5. installment (0.00669); 6. current_balance (0.006233); 7. grade_subgrade (0.004621); 8. monthly_income (0.003358)
- employment_status rank = **1**
- gender absent: **YES**
- skip_reason: None

## 7. Freeze-before-test

- Freeze timestamp UTC: **2026-08-31T12:30:49.710780+00:00**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- `computed_after_freeze=true` in `artifacts/danet_test_metrics.json`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)

Artifacts:

- `artifacts/danet_model.pt`
- `artifacts/danet_preprocess.joblib`
- `artifacts/danet_meta.json`
- `artifacts/danet_oof_pds.csv`
- `artifacts/danet_test_metrics.json`
- `artifacts/danet_run_report.md`
- `artifacts/danet_requirements.txt`
- `artifacts/danet_permutation_importance.json`
- `artifacts/danet_X_train.csv` / `artifacts/danet_y_train.csv` / `artifacts/danet_X_test.csv` / `artifacts/danet_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`, `artifacts/lgbm_dart_monotone_*`, `artifacts/ensemble_no_gender_*`, `artifacts/ensemble_fm_ft_*`.

Governance thresholds in `utils/config.py` were not changed.
