# Model Validation Report — 信用评分卡建模与验证报告

数据来源: `data/loan_dataset_20000.csv` ｜ 建模函数来源: `utils/risk_skills.py` ｜ 调优来源: `scripts/tune_lgb_skill.py` ｜ 随机种子: `random_state=42`
流水线: `scripts/train.py`（训练 + Optuna 调优，仅接触 Train 集）→ `scripts/test.py`（评估、混淆矩阵与报告生成，仅接触 Test 集与已持久化模型）

---

## 1. Executive Summary

- 本次基于 `data/loan_dataset_20000.csv`（20,000 笔贷款样本）构建了标准信用评分卡模型（逻辑回归，`train_scorecard_model`，Baseline），并通过 Optuna 贝叶斯超参数寻优（`tune_lightgbm_optuna`，50 轮，5 折交叉验证，优化目标 AUC）得到单调约束 LightGBM 最优模型（Champion）。
- 目标变量 `default` = 1 - `loan_paid_back`（1 = 违约/未还清，0 = 正常还清）。
- **最优模型（Optuna 调优 LightGBM）** 在 Test 集上的判别能力：AUC = **0.8858**（阈值 ≥ 0.7：**PASS**），KS = **0.5777**（阈值 ≥ 0.3：**PASS**），评级为 "Good (0.40 - 0.60)"；混淆矩阵详见 §5.3。
- 基线评分卡（逻辑回归）Test 集：AUC = 0.8812，KS = 0.5717，作为可解释性对照。
- 最优模型预测违约概率在 Train → Test 上的 PSI = **0.0021**，稳定性状态："Stable (<0.10) - No action needed"（低于预警阈值，无需特别处理）。
- 特征工程阶段基于 IV 剔除 14 个弱预测特征；基于迭代式 VIF 剔除 1 个共线性特征；最终入模特征 6 个（详见 §3）。
- **治理标记（保留，非剔除）**：特征 `employment_status` 的 IV 落入 "Suspicious / Overfitting (>0.5)" 区间（详见 §3.1、§7.4），存在近似确定性分组（如 employment_status=Unemployed 违约率 81.8%、Retired 违约率仅 0.5%）。经复核判断为起贷时可得的合法申请人属性而非标签泄漏，予以保留入模；但其对模型判别力的贡献占比过高（详见 §7.4 的敏感性对比），存在集中度风险与潜在公平信贷（fair-lending）合规风险，**在独立验证与公平信贷复核完成前不建议直接投产使用**。
- **方法论修正记录**：此前版本的多重共线性过滤为一次性剔除所有 VIF 超阈值特征，导致 `credit_score` 因与 `grade_subgrade` 共同存在而被误伤剔除（二者共线，但只需剔除其一）；现改为迭代式 VIF（每轮只剔除当前 VIF 最高的单个特征后重新计算，详见 §3.2），该问题已修正。
- **调优脚本已知局限**：`tune_lightgbm_optuna` 的最终模型使用交叉验证得到的最佳超参数在全量 Train 上重新拟合，但重新拟合时固定 `n_estimators=1000` 且未设置 `eval_set`/早停，与各折验证时实际使用的早停树数不一致，可能引入额外方差；如需更严谨的复现，建议后续为最终拟合补充早停或将 CV 中位数最优轮数固定为 `n_estimators`。
- 建议：该最优模型判别能力与稳定性均满足内部风控标准；在完成 §7.4 所述的独立验证与公平信贷复核前，暂不建议投产。

---

## 2. Train / Test Split

- 数据源：`data/loan_dataset_20000.csv`，共 20,000 条样本，22 个原始字段（含目标衍生前）。
- 目标变量：`default`（由 `loan_paid_back` 取反得到，1 = 违约）。
- 切分方式：`sklearn.model_selection.train_test_split`（`scripts/train.py`），`test_size=0.3`，按目标变量分层抽样，`random_state=42`；切分结果持久化为 `artifacts/train_data.csv` / `artifacts/test_data.csv`，`scripts/test.py` 直接加载，不重新切分。
- 分层校验：Train / Test 违约率差异 < 0.001（见下表），分层抽样有效。
- 说明：原始数据不含时间/放款日期字段，无法构建真正意义上的跨时间 OOT 样本；本报告中的 "Test" 集为分层随机切分得到的样本外测试集，Optuna 调优过程中的 5 折交叉验证同样只在 Train 集内部进行，Test 集全程未参与调参或训练。

| sample   |     n |   bad_rate |
|:---------|------:|-----------:|
| Train    | 14000 |     0.2001 |
| Test     |  6000 |     0.2002 |

---

## 3. Feature Engineering & Selection

### 3.1 WOE / IV 筛选表（基于 Train 集，`calculate_woe_iv`，等频 5 分箱，`scripts/train.py`）

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
| loan_amount          | 0.0022 | Unpredictive (<0.02)            |
| current_balance      | 0.0022 | Unpredictive (<0.02)            |
| num_of_open_accounts | 0.0021 | Unpredictive (<0.02)            |
| installment          | 0.002  | Unpredictive (<0.02)            |
| marital_status       | 0.0019 | Unpredictive (<0.02)            |
| total_credit_limit   | 0.0016 | Unpredictive (<0.02)            |
| age                  | 0.0015 | Unpredictive (<0.02)            |
| gender               | 0.0004 | Unpredictive (<0.02)            |
| public_records       | 0.0002 | Unpredictive (<0.02)            |
| loan_term            | 0.0001 | Unpredictive (<0.02)            |

- 剔除标准：IV < 0.02 → 直接剔除（预测力不足）。IV > 0.5 → **不自动剔除**，仅标记为需要业务/公平信贷复核（"Suspicious / Overfitting"，可能是合法强预测因子，也可能是泄漏，取舍属于治理判断而非统计判断）。
- 弱预测力剔除（14 个）：loan_purpose, education_level, monthly_income, annual_income, loan_amount, current_balance, num_of_open_accounts, installment, marital_status, total_credit_limit, age, gender, public_records, loan_term。
- 标记但保留入模（1 个）：`employment_status`——详见 §7.4 的治理复核说明。

### 3.2 多重共线性过滤（迭代式 VIF，内核为 `compute_vif_filter`，在 IV 筛选后特征的 WOE 编码矩阵上计算，`scripts/train.py`）

最终一轮（全部特征 VIF ≤ 5.0）：

| feature              |   vif |
|:---------------------|------:|
| delinquency_history  |  3.86 |
| num_of_delinquencies |  3.81 |
| grade_subgrade       |  1.35 |
| interest_rate        |  1.33 |
| debt_to_income_ratio |  1.05 |
| employment_status    |  1.01 |

- 剔除标准：VIF > 5.0，**每轮只剔除当前 VIF 最高的单个特征后重新计算**，而非一次性剔除所有超阈值特征（一次性剔除会在多个特征互相拖累对方 VIF 时过度剔除——本报告曾因此误伤 `credit_score`，见 §1）。
- 本次剔除特征（1 个，按剔除顺序）：credit_score。
- 完整迭代过程（共 2 轮）：

轮次 1：

| feature              |   vif |
|:---------------------|------:|
| credit_score         |  7.17 |
| grade_subgrade       |  6.98 |
| delinquency_history  |  3.86 |
| num_of_delinquencies |  3.81 |
| interest_rate        |  1.37 |
| debt_to_income_ratio |  1.05 |
| employment_status    |  1.01 |

轮次 2：

| feature              |   vif |
|:---------------------|------:|
| delinquency_history  |  3.86 |
| num_of_delinquencies |  3.81 |
| grade_subgrade       |  1.35 |
| interest_rate        |  1.33 |
| debt_to_income_ratio |  1.05 |
| employment_status    |  1.01 |

### 3.3 最终入模特征（6 个）

employment_status, debt_to_income_ratio, interest_rate, grade_subgrade, delinquency_history, num_of_delinquencies

---

## 4. Hyperparameter Tuning (Optuna)

- 调优函数：`scripts/tune_lgb_skill.py` 的 `tune_lightgbm_optuna`（TPE 采样器，`random_state=42`）。
- 搜索空间：`learning_rate` ∈ [0.01, 0.1]（log），`num_leaves` ∈ [15, 127]，`max_depth` ∈ [3, 8]，`min_child_samples` ∈ [20, 300]，`subsample` ∈ [0.6, 1.0]，`colsample_bytree` ∈ [0.6, 1.0]，`reg_alpha`/`reg_lambda` ∈ [1e-3, 10]（log）。
- 交叉验证：5 折 `StratifiedKFold`（`shuffle=True, random_state=42`），每折内以验证集早停（`stopping_rounds=30`）；单调性约束 `monotone_constraints=[-1, -1, -1, -1, -1, -1]` 在每次试验中保持固定，与入模特征的 WOE 编码方向一致。
- 试验轮数：请求 50 轮，完成 50 轮。
- 优化目标：AUC（OOF，跨 5 折）。
- 最佳 CV AUC = **0.8862**。

### 4.1 最佳超参数配置

| hyperparameter    |       value |
|:------------------|------------:|
| learning_rate     |  0.054694   |
| num_leaves        | 87          |
| max_depth         |  4          |
| min_child_samples | 67          |
| subsample         |  0.888127   |
| colsample_bytree  |  0.896228   |
| reg_alpha         |  0.00104463 |
| reg_lambda        |  0.156011   |

### 4.2 最优模型特征重要度

| feature              |   importance |
|:---------------------|-------------:|
| grade_subgrade       |          912 |
| debt_to_income_ratio |          562 |
| employment_status    |          419 |
| interest_rate        |          184 |
| delinquency_history  |          121 |
| num_of_delinquencies |           41 |

---

## 5. Model Performance & Discrimination

### 5.1 Train / Test 判别力对比表（`evaluate_discrimination_and_ks`，`scripts/test.py`）

| model                                      | sample   |    AUC |   Gini |     KS | rating             |
|:-------------------------------------------|:---------|-------:|-------:|-------:|:-------------------|
| Logistic Regression Scorecard (Baseline)   | Train    | 0.8794 | 0.7588 | 0.5602 | Good (0.40 - 0.60) |
| Logistic Regression Scorecard (Baseline)   | Test     | 0.8812 | 0.7625 | 0.5717 | Good (0.40 - 0.60) |
| Optuna-Tuned Monotonic LightGBM (Champion) | Train    | 0.8908 | 0.7817 | 0.5819 | Good (0.40 - 0.60) |
| Optuna-Tuned Monotonic LightGBM (Champion) | Test     | 0.8858 | 0.7715 | 0.5777 | Good (0.40 - 0.60) |

- 内部阈值标准：Test 集 KS ≥ 0.3，AUC ≥ 0.7。
- 最优模型（Optuna 调优 LightGBM）Test 集结果：AUC = 0.8858（PASS），KS = 0.5777（PASS），最优切点概率 = 0.2363。

### 5.2 评分卡标尺参数（Baseline，`train_scorecard_model`，`scripts/train.py`）

- Base Points = 600，Base Odds = 50.0，PDO = 20
- Factor = 28.8539，Offset = 487.1229，Intercept = -1.4628

| feature              |   logistic_coef |   score_impact_factor |
|:---------------------|----------------:|----------------------:|
| grade_subgrade       |         -1.6041 |                 46.29 |
| debt_to_income_ratio |         -1.5791 |                 45.56 |
| employment_status    |         -1.2759 |                 36.82 |
| delinquency_history  |         -0.3528 |                 10.18 |
| interest_rate        |         -0.3228 |                  9.31 |
| num_of_delinquencies |          0.2283 |                 -6.59 |

### 5.3 混淆矩阵（最优模型，Test 集，切点 = KS 最优切点概率 0.2363）

|                 |   Predicted Bad (1) |   Predicted Good (0) |
|:----------------|--------------------:|---------------------:|
| Actual Bad (1)  |                 823 |                  378 |
| Actual Good (0) |                 519 |                 4280 |

| metric               |   value |
|:---------------------|--------:|
| Accuracy             |  0.8505 |
| Precision            |  0.6133 |
| Recall / Sensitivity |  0.6853 |
| Specificity          |  0.8919 |

- 切点选取说明：Test 集违约率约 20%，固定 0.5 切点会导致模型几乎不触发正类预测；因此采用 `evaluate_discrimination_and_ks` 基于 ROC 曲线给出的 KS 最优切点概率（`argmax(TPR-FPR)`）作为混淆矩阵切点，而非任意的 0.5。

---

## 6. Population Stability Index (PSI)

### 6.1 模型分 PSI（最优模型预测违约概率，Train 作为基准分布，Test 作为实际分布，`scripts/test.py`）

- PSI = **0.0021**，状态：Stable (<0.10) - No action needed
- 监管标准：PSI < 0.10 无需处理；0.10~0.25 需特别注明预警；> 0.25 需模型重新校准。本次结果：低于预警阈值，无需特别处理

### 6.2 关键特征 PSI（Train vs Test，Top 3 入模特征）

| feature              |    PSI | status                            |
|:---------------------|-------:|:----------------------------------|
| debt_to_income_ratio | 0.0046 | Stable (<0.10) - No action needed |
| interest_rate        | 0.0044 | Stable (<0.10) - No action needed |
| delinquency_history  | 0.0005 | Stable (<0.10) - No action needed |

---

## 7. Explainability & Governance

### 7.1 全局 SHAP 特征重要性排序（`generate_shap_summary`，基于最优模型 LightGBM，Test 抽样 2000 条，`scripts/test.py`）

| feature              |   mean_abs_shap |
|:---------------------|----------------:|
| employment_status    |          1.7561 |
| debt_to_income_ratio |          1.1506 |
| grade_subgrade       |          1.0311 |
| interest_rate        |          0.13   |
| delinquency_history  |          0.0363 |
| num_of_delinquencies |          0.0117 |

### 7.2 Top 3 特征业务解释

1. **employment_status**（⚠ 本报告 §7.4 单独讨论其治理风险）（平均 |SHAP| = 1.7561）：借款人就业状态（Employed / Self-employed / Retired / Student / Unemployed），起贷时可得的申请人属性，非模型输出的衍生字段；WOE 编码值越低代表原始取值落入历史违约率更高的分箱，对应模型预测违约概率上升。作为全局重要性排名第 1 的特征，建议在授信审批规则与额度定价中重点参考。
2. **debt_to_income_ratio**（平均 |SHAP| = 1.1506）：借款人月负债与月收入的比值，比值越高代表偿债压力越大；WOE 编码值越低代表原始取值落入历史违约率更高的分箱，对应模型预测违约概率上升。作为全局重要性排名第 2 的特征，建议在授信审批规则与额度定价中重点参考。
3. **grade_subgrade**（平均 |SHAP| = 1.0311）：内部预分配风险评级子等级（30 档），与 credit_score/DTI/interest_rate 等原始征信变量存在结构性重叠；WOE 编码值越低代表原始取值落入历史违约率更高的分箱，对应模型预测违约概率上升。作为全局重要性排名第 3 的特征，建议在授信审批规则与额度定价中重点参考。

### 7.3 敏感性分析：治理标记特征的边际贡献

对已保留但标记为 "Suspicious / Overfitting" 的特征（`employment_status`），重新拟合 Baseline 逻辑回归、剔除该特征后在同一 Test 集上重新评估，以量化其对判别力的边际贡献（口径对齐 `reports/PD_Model_Development_Document.docx` §7.3 的 variant 敏感性分析）：

| variant                            |   n_features |    AUC |   Gini |     KS |
|:-----------------------------------|-------------:|-------:|-------:|-------:|
| Full set (incl. employment_status) |            6 | 0.8812 | 0.7625 | 0.5717 |
| Excluding employment_status        |            5 | 0.6979 | 0.3958 | 0.2764 |

该特征贡献了约 0.1833 的 Test AUC——是本模型判别力的主要来源，而非边际增益。这也是本模型判别力显著高于早前排除该特征版本（Test AUC ≈0.665）的核心原因。

### 7.4 治理标记特征明细（按类别）

**employment_status**（借款人就业状态（Employed / Self-employed / Retired / Student / Unemployed），起贷时可得的申请人属性，非模型输出的衍生字段）：

| employment_status   |    n |   default_rate |     woe |
|:--------------------|-----:|---------------:|--------:|
| Unemployed          | 1475 |         0.8203 | -2.9045 |
| Student             |  566 |         0.5866 | -1.7357 |
| Employed            | 9086 |         0.1133 |  0.6721 |
| Self-employed       | 2021 |         0.1108 |  0.6964 |
| Retired             |  852 |         0.007  |  3.5629 |



**治理结论**：该特征在起贷时可得、非模型输出的衍生字段，不构成技术意义上的标签泄漏；但其对模型判别力的贡献占比过高（集中度风险），且类别与借款人的就业/收入状态强相关，在放贷决策中使用可能构成对特定群体的间接歧视（disparate impact）。本报告将其保留入模用于性能评估，**但明确不建议在完成独立验证与公平信贷（fair-lending）合规复核之前将其用于实际授信决策**。参考文档：`reports/PD_Model_Development_Document.docx` §8（同样保留该特征，同样将其列为待决限制项，尚未获得独立验证与合规批准）。

### 7.5 监管合规审查说明

- 模型类型：标准评分卡（L2 正则逻辑回归，`penalty='l2', C=1.0`）作为可解释性 Baseline，系数与标度均可追溯至 `Score = Offset + Factor * ln(Odds)` 公式；最优决策模型为 Optuna 调优后的单调约束 LightGBM。
- 最优模型（LightGBM）已对全部入模特征施加单调递减约束（`monotone_constraints=-1`），约束方向依据 WOE 编码定义（WOE 越高代表历史违约率越低），且该约束在 Optuna 全部 50 轮试验中保持固定，符合监管对评分类模型单调性的一般要求。
- 特征筛选与共线性处理均通过标准化流程（`calculate_woe_iv` / 迭代式 `compute_vif_filter`）完成，超参数寻优通过标准化流程（`tune_lightgbm_optuna`）完成，未使用未经验证的第三方黑盒库直接输出模型结论。
- 判别力与稳定性指标（KS/AUC/PSI）均通过 `utils/risk_skills.py` 标准函数计算，口径统一、可复现（`random_state=42`）。
- 训练/调参/测试边界可审计：`scripts/train.py` 只写 `artifacts/`（含 Optuna 调优过程），从不读取 Test 集；`scripts/test.py` 只读 `artifacts/` 与 Test 集，从不重新拟合分箱、编码器、模型参数或超参数。
- IV > 0.5 的特征不再自动剔除，改为保留入模 + 强制标记 + §7.3/§7.4 的量化敏感性分析与治理结论，取舍留待模型使用方与合规团队决策，而非由脚本单方面决定（详见 CLAUDE.md 中记录的方法论修正）。
