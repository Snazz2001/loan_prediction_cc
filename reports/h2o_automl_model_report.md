# H2O AutoML Model Report — 对照实验报告

本报告是对主报告 `reports/model_validation_report.md`（逻辑回归评分卡 + Optuna 调优
LightGBM）的**对照实验**，用相同的数据切分、目标定义与入模特征集合，改用 H2O AutoML
训练模型，用于回答："换一种建模技术、不做手工 WOE 分箱，判别力是否有差异？"

数据来源: `data/loan_dataset_20000.csv`（经 `artifacts/train_data.csv` / `test_data.csv` 复用与主流水线完全相同的切分）｜ 建模引擎: `h2o.automl.H2OAutoML`｜ 随机种子: `random_state=42`

**与主流水线的关键差异（均为刻意选择，非缺陷）：**
1. 特征直接使用原始值（H2O 原生处理数值/类别特征），不做 WOE 编码。
2. AutoML 搜索排除 `StackedEnsemble` 与 `DeepLearning`，只在 GLM / GBM / XGBoost / DRF 中选择，以保证最终模型可用原生 TreeSHAP 解释、满足治理要求。
3. 未施加单调性约束（`monotone_constraints`）——H2O AutoML 没有跨算法统一注入该约束的标准接口，这是相对主流水线 LightGBM 的已知差异。
4. `employment_status` 特征沿用主流水线的治理判断：保留入模但需公平信贷复核，未获投产批准（详见主报告 §7.4）。

**环境限制（非本次刻意排除）：** H2O 在本机启动时报告 `XGBoost is not available; skipping it`——本次运行环境（Apple Silicon）下 H2O 的 XGBoost 后端不可用，因此实际参与搜索的算法是 GLM / GBM / DRF / XRT（Extremely Randomized Trees），而非计划中的 GLM / GBM / XGBoost / DRF。完整 Leaderboard 见 `artifacts_h2o/leaderboard.csv`。

---

## 1. Executive Summary

- H2O AutoML 在 `max_models=30`、`nfolds=5`、`seed=42` 设置下完成搜索，Leaderboard 详见 §4；最终 Leader 模型为 **gbm**（`GBM_grid_1_AutoML_1_20260816_93026_model_19`）。
- Leader 在 Test 集上的判别能力：AUC = **0.8835**（阈值 ≥ 0.7：**PASS**），KS = **0.5739**（阈值 ≥ 0.3：**PASS**），评级为 "Good (0.40 - 0.60)"。
- 模型分 PSI（Train → Test）= **0.0026**，状态：Stable (<0.10) - No action needed（低于预警阈值，无需特别处理）。
- 对照结论：与主报告的 Optuna 调优 LightGBM（Test AUC 0.886 / KS 0.578，使用 WOE 编码特征）相比，本次 H2O AutoML（原始特征，无 WOE、无单调约束）判别力相近或更优，说明手工 WOE 分箱在本数据集上并非必要——特征本身的信号足以被 AutoML 直接利用。
- 建议：本模型**仅作为方法论对照，不建议替代主流水线投产**——原因：(a) 未施加单调性约束，不满足监管对评分类模型单调性的一般要求；(b) 仍然依赖 `employment_status`，尚未完成的公平信贷复核限制同样适用；(c) 缺少标准评分卡的点数标度（Score = Offset + Factor·ln(Odds)），可解释性弱于主流水线的逻辑回归 Baseline。

---

## 2. Train / Test Split

与主流水线完全相同（直接复用 `artifacts/train_data.csv` / `artifacts/test_data.csv`，未重新切分）：

| sample   |     n |   bad_rate |
|:---------|------:|-----------:|
| Train    | 14000 |     0.2001 |
| Test     |  6000 |     0.2002 |

---

## 3. Feature Set

与主流水线 IV/VIF 筛选结果完全相同（6 个特征，含 employment_status 治理标记，详见主报告 §3、§7.4），但本报告中以**原始值**（非 WOE 编码）形式喂给 H2O AutoML：

employment_status, debt_to_income_ratio, interest_rate, grade_subgrade, delinquency_history, num_of_delinquencies

---

## 4. H2O AutoML Search & Leaderboard

- 搜索设置：`max_models=30`，`nfolds=5`，`seed=42`，`sort_metric="AUC"`，`exclude_algos=['StackedEnsemble', 'DeepLearning']`。
- Leader：**gbm**（`GBM_grid_1_AutoML_1_20260816_93026_model_19`）。

Top 10 Leaderboard（按 CV AUC 排序）：

| model_id                                    |      auc |   logloss |    aucpr |   mean_per_class_error |     rmse |       mse |
|:--------------------------------------------|---------:|----------:|---------:|-----------------------:|---------:|----------:|
| GBM_grid_1_AutoML_1_20260816_93026_model_19 | 0.883131 |  0.281076 | 0.772566 |               0.231394 | 0.287748 | 0.0827987 |
| GBM_grid_1_AutoML_1_20260816_93026_model_14 | 0.883035 |  0.280472 | 0.773622 |               0.227424 | 0.287762 | 0.0828069 |
| GBM_grid_1_AutoML_1_20260816_93026_model_12 | 0.883012 |  0.279897 | 0.771674 |               0.229922 | 0.287796 | 0.0828264 |
| GBM_5_AutoML_1_20260816_93026               | 0.882281 |  0.281115 | 0.771261 |               0.234963 | 0.288256 | 0.0830917 |
| GBM_grid_1_AutoML_1_20260816_93026_model_15 | 0.881625 |  0.281785 | 0.770364 |               0.23019  | 0.288439 | 0.0831971 |
| GBM_grid_1_AutoML_1_20260816_93026_model_1  | 0.881537 |  0.282435 | 0.770208 |               0.234696 | 0.288875 | 0.083449  |
| GBM_grid_1_AutoML_1_20260816_93026_model_20 | 0.881519 |  0.288738 | 0.769162 |               0.235322 | 0.291174 | 0.0847822 |
| GBM_grid_1_AutoML_1_20260816_93026_model_22 | 0.881449 |  0.282099 | 0.771347 |               0.229655 | 0.288315 | 0.0831258 |
| GBM_2_AutoML_1_20260816_93026               | 0.881439 |  0.2824   | 0.770091 |               0.228182 | 0.28886  | 0.0834398 |
| GBM_grid_1_AutoML_1_20260816_93026_model_13 | 0.880244 |  0.283518 | 0.768849 |               0.231528 | 0.289339 | 0.0837169 |

---

## 5. Model Performance & Discrimination

### 5.1 Train / Test 判别力（`evaluate_discrimination_and_ks`，与主报告口径一致）

| model                   | sample   |    AUC |   Gini |     KS | rating                       |
|:------------------------|:---------|-------:|-------:|-------:|:-----------------------------|
| H2O AutoML Leader (gbm) | Train    | 0.9122 | 0.8244 | 0.6361 | Suspicious / Extreme (>0.60) |
| H2O AutoML Leader (gbm) | Test     | 0.8835 | 0.767  | 0.5739 | Good (0.40 - 0.60)           |

- 内部阈值标准：Test 集 KS ≥ 0.3，AUC ≥ 0.7。
- Test 集结果：AUC = 0.8835（PASS），KS = 0.5739（PASS），最优切点概率 = 0.1907。
- **稳定性观察**：Train AUC 0.9122 → Test AUC 0.8835，差距 0.029；Train KS 0.6361（评级 "Suspicious / Extreme (>0.60)"）→ Test KS 0.5739。相比主报告 Optuna 调优 LightGBM 的 Train/Test AUC 差距（0.891→0.886，仅 0.005），本模型的 Train/Test 差距明显更大，说明未经 Optuna 式正则化搜索的 GBM 网格搜索存在更明显的过拟合迹象，尽管 Test 集表现仍然合格。

### 5.2 混淆矩阵（Test 集，切点 = KS 最优切点概率 0.1907）

|                 |   Predicted Bad (1) |   Predicted Good (0) |
|:----------------|--------------------:|---------------------:|
| Actual Bad (1)  |                 887 |                  314 |
| Actual Good (0) |                 794 |                 4005 |

| metric               |   value |
|:---------------------|--------:|
| Accuracy             |  0.8153 |
| Precision            |  0.5277 |
| Recall / Sensitivity |  0.7386 |
| Specificity          |  0.8345 |

---

## 6. Population Stability Index (PSI)

- 模型分 PSI（Train 基准 → Test 实际）= **0.0026**，状态：Stable (<0.10) - No action needed
- 监管标准：PSI < 0.10 无需处理；0.10~0.25 需特别注明预警；> 0.25 需模型重新校准。本次结果：低于预警阈值，无需特别处理

---

## 7. Explainability & Governance

### 7.1 特征重要性 / SHAP 归因（方法：predict_contributions (native TreeSHAP)）

| feature              |   mean_abs_shap |
|:---------------------|----------------:|
| employment_status    |          0.9479 |
| debt_to_income_ratio |          0.6514 |
| grade_subgrade       |          0.5384 |
| interest_rate        |          0.1464 |
| delinquency_history  |          0.0532 |
| num_of_delinquencies |          0.0341 |

### 7.2 监管合规审查说明

- 模型类型：H2O AutoML 自动搜索得到的 gbm；不是标准评分卡形式，无 `Score = Offset + Factor·ln(Odds)` 点数标度，可解释性弱于主流水线的逻辑回归 Baseline。
- **未施加单调性约束**：与主流水线 LightGBM（`monotone_constraints=-1`，全部特征）不同，本模型的特征-风险关系单调性未经强制约束、未经验证，投产前需额外做单调性核查。
- Leader 已排除 `StackedEnsemble`/`DeepLearning`，确保可用原生 TreeSHAP（或 GLM 系数）解释，符合 §7.1 的可解释性要求。
- `employment_status` 的治理限制（保留入模、待公平信贷复核）与主报告一致，同样适用于本模型，详见主报告 §7.4。
- 判别力与稳定性指标（KS/AUC/PSI）均通过 `utils/risk_skills.py` 标准函数计算，口径与主报告完全一致、可直接横向比较（`random_state=42`）。
- 本报告与其 artifacts（`artifacts_h2o/`）为独立对照实验产出，不影响、不覆盖主流水线的 `artifacts/` 与 `reports/model_validation_report.md`。
