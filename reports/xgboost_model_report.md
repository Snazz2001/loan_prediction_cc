# XGBoost Model Report — 对照实验报告

本报告是对主报告 `reports/model_validation_report.md`（逻辑回归评分卡 + Optuna 调优
LightGBM）的**对照实验**：除建模算法从 LightGBM 换成 XGBoost 外，数据切分、WOE 特征
矩阵、Optuna 调优设置（50 轮、5 折 CV、优化目标 AUC）、单调性约束方向均与主流水线
逐项对齐，是三个模型中与主流水线设置最接近的一个对照（比 H2O AutoML 对照更贴近，因为
XGBoost 原生支持 `monotone_constraints` 与 `shap` 库）。

数据来源: `data/loan_dataset_20000.csv`（经 `artifacts/train_data.csv` / `test_data.csv` 复用与主流水线完全相同的切分）｜ 建模引擎: `xgboost.XGBClassifier` + `scripts/tune_xgb_skill.py`｜ 随机种子: `random_state=42`

---

## 1. Executive Summary

- 使用 `scripts/tune_xgb_skill.py` 的 `tune_xgboost_optuna`（50 轮 Optuna，5 折交叉验证，优化目标 AUC）训练单调约束 XGBoost 模型。
- 最佳 CV AUC = **0.8866**。
- Test 集判别能力：AUC = **0.8852**（阈值 ≥ 0.7：**PASS**），KS = **0.5716**（阈值 ≥ 0.3：**PASS**），评级为 "Good (0.40 - 0.60)"。
- Train → Test AUC 差距 = 0.0063（主报告 LightGBM 为 0.005；H2O AutoML GBM 为 0.029）。
- 模型分 PSI（Train → Test）= **0.0018**，状态：Stable (<0.10) - No action needed（低于预警阈值，无需特别处理）。
- 对照结论：三个模型（LightGBM / H2O AutoML GBM / XGBoost）在同一 WOE 特征集合、同等调优强度下判别力高度接近（AUC 均在 0.88 附近），说明本数据集在当前 6 特征、含 `employment_status` 的设置下，判别力上限已趋于收敛，模型算法选择本身不是当前的瓶颈；真正影响判别力的是**特征选择（尤其是 employment_status 与 credit_score 的取舍）**，而非建模技术。
- 建议：与主报告一致——本模型同样**不建议替代主流水线投产**：仍依赖 `employment_status`（详见主报告 §7.4，公平信贷复核未完成），且不是标准评分卡形式，评分点数标度缺失，可解释性弱于 LR Baseline。

---

## 2. Train / Test Split & Feature Set

与主流水线完全相同（直接复用 `artifacts/train_data.csv` / `test_data.csv` / `feature_selection.json` / `woe_encoders.joblib`，未重新切分、未重新拟合编码器）：

| sample   |     n |   bad_rate |
|:---------|------:|-----------:|
| Train    | 14000 |     0.2001 |
| Test     |  6000 |     0.2002 |

入模特征（6 个，WOE 编码，与主流水线 LightGBM 使用的矩阵完全相同）：employment_status, debt_to_income_ratio, interest_rate, grade_subgrade, delinquency_history, num_of_delinquencies

---

## 3. Hyperparameter Tuning (Optuna)

- 调优函数：`scripts/tune_xgb_skill.py` 的 `tune_xgboost_optuna`（TPE 采样器，`random_state=42`），参数含义与 `tune_lightgbm_optuna` 逐项对齐。
- 搜索空间：`learning_rate` ∈ [0.01, 0.1]（log），`max_depth` ∈ [3, 8]，`min_child_weight` ∈ [1, 20]，`subsample` ∈ [0.6, 1.0]，`colsample_bytree` ∈ [0.6, 1.0]，`reg_alpha`/`reg_lambda` ∈ [1e-3, 10]（log）。
- 交叉验证：5 折 `StratifiedKFold`（`shuffle=True, random_state=42`），每折内以验证集早停（`early_stopping_rounds=30`）；`monotone_constraints=(-1, -1, -1, -1, -1, -1)` 在每次试验中保持固定，方向与 WOE 编码定义一致（WOE 越高代表历史违约率越低）。
- 试验轮数：完成 50 轮。

### 3.1 最佳超参数配置

| hyperparameter   |     value |
|:-----------------|----------:|
| learning_rate    | 0.0882469 |
| max_depth        | 5         |
| min_child_weight | 3         |
| subsample        | 0.607107  |
| colsample_bytree | 0.967421  |
| reg_alpha        | 0.0372538 |
| reg_lambda       | 0.0125657 |

### 3.2 特征重要度

| feature              |   importance |
|:---------------------|-------------:|
| employment_status    |    0.668287  |
| debt_to_income_ratio |    0.152097  |
| grade_subgrade       |    0.123334  |
| interest_rate        |    0.0260972 |
| delinquency_history  |    0.0196671 |
| num_of_delinquencies |    0.0105173 |

- **已知局限**（与主报告 LightGBM 一致，刻意保留以保证"同样设置"）：最终模型使用 CV 得到的最佳超参数在全量 Train 上重新拟合，重新拟合时固定 `n_estimators=1000` 且未设置 `eval_set`/早停，与各折验证时实际使用的早停树数不一致，可能引入额外方差。

---

## 4. Model Performance & Discrimination

### 4.1 Train / Test 判别力（`evaluate_discrimination_and_ks`，与主报告口径一致）

| model                          | sample   |    AUC |   Gini |     KS | rating             |
|:-------------------------------|:---------|-------:|-------:|-------:|:-------------------|
| Optuna-Tuned Monotonic XGBoost | Train    | 0.8915 | 0.783  | 0.5809 | Good (0.40 - 0.60) |
| Optuna-Tuned Monotonic XGBoost | Test     | 0.8852 | 0.7703 | 0.5716 | Good (0.40 - 0.60) |

- 内部阈值标准：Test 集 KS ≥ 0.3，AUC ≥ 0.7。
- Test 集结果：AUC = 0.8852（PASS），KS = 0.5716（PASS），最优切点概率 = 0.206。

### 4.2 混淆矩阵（Test 集，切点 = KS 最优切点概率 0.206）

|                 |   Predicted Bad (1) |   Predicted Good (0) |
|:----------------|--------------------:|---------------------:|
| Actual Bad (1)  |                 878 |                  323 |
| Actual Good (0) |                 769 |                 4030 |

| metric               |   value |
|:---------------------|--------:|
| Accuracy             |  0.818  |
| Precision            |  0.5331 |
| Recall / Sensitivity |  0.7311 |
| Specificity          |  0.8398 |

---

## 5. Population Stability Index (PSI)

- 模型分 PSI（Train 基准 → Test 实际）= **0.0018**，状态：Stable (<0.10) - No action needed
- 监管标准：PSI < 0.10 无需处理；0.10~0.25 需特别注明预警；> 0.25 需模型重新校准。本次结果：低于预警阈值，无需特别处理

---

## 6. Explainability & Governance

### 6.1 全局 SHAP 特征重要性排序（`generate_shap_summary`，Test 抽样 2000 条）

| feature              |   mean_abs_shap |
|:---------------------|----------------:|
| employment_status    |          2.215  |
| debt_to_income_ratio |          1.555  |
| grade_subgrade       |          1.3299 |
| interest_rate        |          0.2721 |
| delinquency_history  |          0.0948 |
| num_of_delinquencies |          0.0836 |

### 6.2 监管合规审查说明

- 模型类型：Optuna 调优单调约束 XGBoost；与主流水线的 LightGBM 挑战者模型定位相同（判别力优先的树模型），不是标准评分卡形式，无点数标度，可解释性弱于 LR Baseline。
- 已对全部入模特征施加单调递减约束（`monotone_constraints=-1`），约束方向依据 WOE 编码定义，在 Optuna 全部 50 轮试验中保持固定，与主流水线 LightGBM 的处理方式完全一致。
- `employment_status` 的治理限制（保留入模、待公平信贷复核）与主报告一致，同样适用于本模型，详见主报告 §7.4。
- 判别力与稳定性指标（KS/AUC/PSI）、SHAP 解释均通过 `utils/risk_skills.py` 标准函数计算，口径与主报告完全一致、可直接横向比较（`random_state=42`）。
- 本报告与其 artifacts（`artifacts_xgboost/`）为独立对照实验产出，不影响、不覆盖主流水线的 `artifacts/`、`reports/model_validation_report.md`，也不影响 H2O 对照实验的 `artifacts_h2o/`、`reports/h2o_automl_model_report.md`。
