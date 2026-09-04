# AutoGluon + RealMLP pack (no gender, frozen split) — run report

Two models on the frozen `scripts/train.py::load_and_split` split, reported separately, plus an equal-weight average and an OOF logistic stack when leak-free OOFs exist. Gender is out. `employment_status` is a normal in-model feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#8 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `scripts/train_autogluon_realmlp.py` (fit on TRAIN only, then freeze)
- Eval script: `scripts/eval_autogluon_realmlp.py` (**never fits**; loads frozen AutoGluon predictor + RealMLP weights + stack meta; `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`

Eval never fits. Split verified: Train n=14000 bad_rate=0.2001 (bads=2801); Test n=6000 bad_rate=0.2002 (bads=1201).

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
- Confirmed absent from persisted X_train/X_test and importance tables: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model feature)

## 3. AutoGluon

- ran: **YES**
- failure: none
- library/class: `autogluon.tabular.TabularPredictor` / `autogluon.tabular.TabularPredictor`
- version: **1.6.1**
- preset requested: `best_quality`
- preset actually run: `best_quality`
- extreme attempted: **NO** — No GPU (torch.cuda.is_available() is false); extreme requires CUDA.
- time_limit: **3600** seconds
- problem_type=`binary`, eval_metric=`roc_auc`, test passed to fit: **false**
- best_model: `WeightedEnsemble_L3`
- TabPFN / TabM / RealMLP seats in the AG ensemble: **TabPFN=NO; TabM=NO; RealMLP=NO**
- TabPFN names: []
- TabM names: []
- RealMLP names: []
- predictor path: `artifacts/autogluon_predictor`
- leaderboard (validation, no test):

| model | score_val | pred_time_val | fit_time | stack_level | can_infer |
| --- | --- | --- | --- | --- | --- |
| WeightedEnsemble_L3 | 0.8986851385051561 | 4.0216288566589355 | 593.8685681819916 | 3 | True |
| WeightedEnsemble_L2 | 0.8978476395942299 | 0.6065733432769775 | 325.08123564720154 | 2 | True |
| CatBoost_r69_BAG_L2 | 0.8968026388595733 | 3.4970993995666504 | 495.48402428627014 | 2 | True |
| CatBoost_r177_BAG_L2 | 0.8967552982222651 | 3.4974679946899414 | 487.85805082321167 | 2 | True |
| CatBoost_r9_BAG_L2 | 0.8964307359135543 | 3.514003038406372 | 530.0087788105011 | 2 | True |
| CatBoost_r13_BAG_L2 | 0.8962597039141207 | 3.4996068477630615 | 515.705041885376 | 2 | True |
| CatBoost_r137_BAG_L2 | 0.8962310763772164 | 3.4992175102233887 | 489.3272774219513 | 2 | True |
| CatBoost_BAG_L2 | 0.8961759890901668 | 3.5044445991516113 | 496.44949316978455 | 2 | True |
| CatBoost_r167_BAG_L2 | 0.8960317037538319 | 3.4974637031555176 | 500.2105073928833 | 2 | True |
| CatBoost_r50_BAG_L2 | 0.8960248656617764 | 3.5010719299316406 | 485.52045130729675 | 2 | True |

## 4. RealMLP

- ran: **YES**
- failure: none
- library: `pytabkit`
- class: `pytabkit.RealMLP_TD_Classifier`
- version: **1.7.3**
- paper: Holzmüller, Grinsztajn, Steinwart — Better by Default (NeurIPS 2024)
- config: **RealMLP-TD (published tuned default)**
- HPO run: **NO**
- HPO note: Library exposes RealMLP_HPO_Classifier, but this pack uses the published RealMLP-TD default explicitly (paper-faithful). CPU-only 4-core budget cannot complete paper-scale HPO (n_hyperopt_steps=50 × n_cv=5) without cutting the search short. OOF AUC is reported from StratifiedKFold 5 on the TD default; it is not used to pick among HPO configs.
- constructor kwargs: `{'device': 'cpu', 'random_state': 42, 'n_cv': 1, 'n_refit': 0, 'verbosity': 1}`
- published TD defaults: `{'hidden_sizes': [256, 256, 256], 'max_one_hot_cat_size': 9, 'embedding_size': 8, 'weight_param': 'ntk', 'bias_lr_factor': 0.1, 'act': 'selu', 'use_parametric_act': True, 'act_lr_factor': 0.1, 'block_str': 'w-b-a-d', 'p_drop': 0.15, 'p_drop_sched': 'flat_cos', 'add_front_scale': True, 'scale_lr_factor': 6.0, 'bias_init_mode': 'he+5', 'weight_init_mode': 'std', 'wd': 0.02, 'wd_sched': 'flat_cos', 'bias_wd_factor': 0.0, 'use_ls': True, 'ls_eps': 0.1, 'num_emb_type': 'pbld', 'plr_sigma': 0.1, 'plr_hidden_1': 16, 'plr_hidden_2': 4, 'plr_lr_factor': 0.1, 'lr': 0.04, 'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'], 'n_epochs': 256, 'lr_sched': 'coslog4', 'opt': 'adam', 'sq_mom': 0.95}`
- device: cpu
- model SHA-256: `b3acdb40a1ffaa29df84c9a6d3725de124846e9c54cc60cbda653689e0a515d8`

## 5. OOF AUC / KS / Gini (train only)

| model | OOF_AUC | OOF_KS | OOF_Gini |
| --- | --- | --- | --- |
| AutoGluon | 0.8987 | 0.5957 | 0.7974 |
| RealMLP-TD | 0.8871 | 0.5721 | 0.7742 |
| Equal-weight average | 0.8958 | 0.5922 | 0.7917 |
| OOF logistic stack | 0.8983 | 0.5962 | 0.7966 |

- AutoGluon OOF available: **YES**
- RealMLP OOF available: **YES**
- RealMLP fold AUCs: [0.874785, 0.886934, 0.892975, 0.892602, 0.890439]
- RealMLP fold KS: [0.550893, 0.576339, 0.590625, 0.600446, 0.588838]
- Selection used Train OOF only. Test was not used to pick hyperparameters or to fit the stack.

## 6. After freeze: Test AUC / KS / Gini / PSI

PSI expected side = OOF PD; actual side = Test PD (not in-sample full-train PD).

| model | Test_AUC | Test_KS | Test_Gini | Test_PSI | psi_above_0.10 |
| --- | --- | --- | --- | --- | --- |
| AutoGluon | 0.8967 | 0.5949 | 0.7934 | 0.0034 | NO |
| RealMLP-TD | 0.8951 | 0.5979 | 0.7902 | 0.1355 | YES |
| Equal-weight average | 0.897 | 0.5962 | 0.7939 | 0.0804 | NO |
| OOF logistic stack | 0.897 | 0.5924 | 0.7939 | 0.028 | NO |

- Freeze timestamp UTC: **2026-09-04T05:25:32.620406+00:00**
- `computed_after_freeze=true`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)
- Internal bars (unchanged, not a go/no-go): AUC_MIN=0.7, KS_MIN=0.3, PSI_WATCH=0.1

### 6.1 Stack meta coefs

- built: **YES**
- pd_autogluon coef: **7.046578399425561**
- pd_realmlp coef: **1.1158232524297897**
- intercept: **-3.22827479626338**
- reason: OOF logistic stack fitted on train OOFs only (columns: pd_autogluon, pd_realmlp)

### 6.2 Plain YES/NO vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858**; PR #2 stack **0.8885**; PR #4 **0.8859**; PR #6 stack **0.8883**; PR #6 gbdt-alone **0.8895**; PR #7 stack **0.8951**; PR #7 FT-alone **0.8963**; PR #8 DANet **0.8655**.

| comparator | autogluon | realmlp | equal_weight_average | stack_lr |
| --- | --- | --- | --- | --- |
| last-run 0.8858 | YES | YES | YES | YES |
| PR#2 stack 0.8885 | YES | YES | YES | YES |
| PR#4 0.8859 | YES | YES | YES | YES |
| PR#6 stack 0.8883 | YES | YES | YES | YES |
| PR#6 gbdt-alone 0.8895 | YES | YES | YES | YES |
| PR#7 stack 0.8951 | YES | NO | YES | YES |
| PR#7 FT-alone 0.8963 | YES | NO | YES | YES |
| PR#8 DANet 0.8655 | YES | YES | YES | YES |

Best reported Test AUC among scorers that ran: AutoGluon=0.8967 RealMLP=0.8951 average=0.897 stack=0.897

## 7. TRAIN explainability (after freeze, in-model features only)

- AutoGluon permutation top: 1. employment_status (0.264972); 2. debt_to_income_ratio (0.148473); 3. credit_score (0.104705); 4. grade_subgrade (0.013676); 5. interest_rate (0.012279); 6. current_balance (0.011231); 7. age (0.009941); 8. loan_purpose (0.009511)
- AutoGluon permutation employment_status rank: **1**
- AutoGluon feature_importance employment_status rank: **None**
- RealMLP permutation top: 1. employment_status (0.244472); 2. debt_to_income_ratio (0.110965); 3. credit_score (0.098659); 4. grade_subgrade (0.012386); 5. delinquency_history (0.001639); 6. annual_income (0.000752); 7. num_of_open_accounts (0.000699); 8. total_credit_limit (0.000457)
- RealMLP permutation employment_status rank: **1**
- gender absent: **YES**

## 8. Package versions / device

- python: 3.12.3
- autogluon.tabular: 1.6.1
- torch: 2.14.0+cpu
- pytabkit: 1.7.3
- sklearn: 1.9.0
- pandas: 2.3.3
- numpy: 2.4.4
- joblib: 1.6.0
- device: {'device': 'cpu', 'torch_cuda_available': False, 'torch_version': '2.14.0+cpu', 'n_cpus': 4, 'note': 'This VM has no GPU; AutoGluon extreme preset was not run.'}

## 9. Artifacts

- `artifacts/autogluon_predictor`
- `artifacts/autogluon_leaderboard.csv`
- `artifacts/autogluon_feature_importance.csv`
- `artifacts/realmlp_model.joblib`
- `artifacts/realmlp_preprocess.joblib`
- `artifacts/realmlp_permutation_importance.json`
- `artifacts/ag_realmlp_stack_meta.joblib`
- `artifacts/ag_realmlp_meta.json`
- `artifacts/ag_realmlp_oof_pds.csv`
- `artifacts/ag_realmlp_test_metrics.json`
- `artifacts/ag_realmlp_run_report.md`
- `artifacts/ag_realmlp_requirements.txt`
- `artifacts/ag_realmlp_X_train.csv` / `artifacts/ag_realmlp_y_train.csv` / `artifacts/ag_realmlp_X_test.csv` / `artifacts/ag_realmlp_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `lgbm_linear_tree_*`, `linear_tree_*`, `stack_lr_rf_lgbm_*`, `lgbm_no_gender_emp_overlay_*`, `lgbm_emp_overlay_*`, `lgbm_emp_in_*`, `lgbm_dart_monotone_*`, `dart_monotone_*`, `ensemble_no_gender_*`, `ensemble_fm_ft_*`, `danet_*`.

Governance thresholds in `utils/config.py` were not changed.
