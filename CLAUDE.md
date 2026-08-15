# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository State & Workflow

* **Dependencies:** `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `optuna`, `shap`, `joblib`, `statsmodels`.
* **Execution Pipeline:**
  ```bash
  python3 scripts/train.py
  python3 scripts/test.py
  
`train.py` loads `data/loan_dataset_20000.csv` and persists all artifacts under `artifacts/`.

`test.py` strictly reads `artifacts/` (never refits) and generates `reports/model_validation_report.md`.

---

### Directory & Module Layout

* **`utils/risk_skills.py`**: Core risk-modeling functions (`calculate_woe_iv`, `compute_vif_filter`, `train_scorecard_model`, `evaluate_discrimination_and_ks`, `calculate_psi`, `generate_shap_summary`). Must be reused directly.
* **`utils/config.py`**: Single source of truth for paths, `random_state=42`, and governance thresholds (IV/VIF/KS/AUC/PSI).
* **`utils/woe_encoding.py`**: Fit-on-train / transform-on-test WOE encoder ensuring data consistency across splits.
* **`scripts/tune_lgb_skill.py`**: Optuna-based Bayesian hyperparameter search (`tune_lightgbm_optuna`). Integrated directly into `train.py`.
* **`scripts/train.py`**:
  * Loads `data/loan_dataset_20000.csv`, derives target (`default = 1 - loan_paid_back`).
  * Performs stratified split (`random_state=42`).
  * Filters features: IV screening (drop IV < 0.02; **flag, don't drop**, IV > 0.5 — see Known Pitfalls below), WOE encoding, then `iterative_vif_filter` (drops the single highest-VIF feature and recomputes, repeating until all remaining features are ≤ threshold — see Known Pitfalls).
  * Trains Baseline Scorecard & calls `tune_lightgbm_optuna` (5-fold CV, target: AUC/KS).
  * Persists all encoders, models, and intermediate metrics into `artifacts/`.
* **`scripts/test.py`**: Evaluates frozen artifacts on the Test split (KS, AUC, PSI, Confusion Matrix, SHAP) and renders the final markdown report. Also runs a sensitivity variant (refit Baseline LR excluding any IV-flagged feature) and a per-category breakdown for flagged categorical features, to give the governance/fair-lending reviewer concrete numbers rather than just a flag.
* **`data/loan_dataset_20000.csv`**: Primary dataset. Note: `employment_status` has IV ≈ 1.9 — **kept in the model, flagged for governance review, not auto-excluded** (see Known Pitfalls below; this changed from an earlier version of this pipeline that auto-excluded it).
* **`artifacts/`**: Serialized pipeline state (`.joblib`, `.json`, `.csv`). Auto-generated.
* **`reports/model_validation_report.md`**: Final compliance and validation report. Auto-generated.
* **`reports/PD_Model_Development_Document.docx`**: An independently authored reference PD model development document for this same dataset, reporting Test AUC 0.887 / KS 0.577 with a 5-variable WOE logistic scorecard. Treat it as ground truth for "what a well-built model on this dataset should score" — if a change to this pipeline drops discrimination well below that (e.g. back down to ~0.66-0.70 AUC), suspect a feature-selection regression before assuming the data has no more signal to give. See Known Pitfalls below for the two concrete regressions this already caused once.

---

### 银行风控模型开发与验证规范

#### 1. 架构与工具要求
* 必须复用 `utils/risk_skills.py` 中的标准函数进行特征粗筛、VIF 过滤、指标评估（KS/AUC/PSI）与 SHAP 解释。
* 严禁使用未经验证的第三方黑盒库直接输出模型。

#### 2. 建模与调优规范 (Optuna + LightGBM)
* 模型训练与调优脚本必须统一存放于 `scripts/` 目录下。
* 模型超参数寻优必须使用 `scripts/tune_lgb_skill.py` 中的 `tune_lightgbm_optuna`。
* 调优优化目标默认设定为 `auc` 或 `ks`，交叉验证折数设置为 5 折。
* 最终输出文档中必须包含：
  * 最佳超参数配置表（Learning Rate, Num Leaves, Regularization 等）；
  * 5 折交叉验证的 OOF 评估得分与收敛情况；
  * 特征重要性（Feature Importance）排序与 SHAP 分析结果。

#### 3. 文档交付标准
每次完成建模必须在 `reports/` 目录下输出完整的 Markdown 报告：`model_validation_report.md`，必须包含以下章节：
1. **Executive Summary**（核心业务结论与上线建议）
2. **Train / Test Split**（样本切分量、违约率分布与分层校验）
3. **Feature Engineering & Selection**（WOE/IV 筛选表、共线性 VIF 剔除清单及合规过滤说明）
4. **Hyperparameter Tuning (Optuna)**（超参数搜索空间、最佳参数表与 5 折 CV 得分）
5. **Model Performance & Discrimination**（Train 与 Test 集上的 KS、AUC、Gini、Confusion Matrix 对比）
6. **Population Stability Index (PSI)**（模型预测分与核心特征的 PSI 稳定性分析）
7. **Explainability & Governance**（SHAP 全局特征贡献排序、单调性合规与审计总结；对任何被标记为 "Suspicious / Overfitting" (IV > 0.5) 但保留入模的特征，必须额外包含：剔除该特征后的敏感性对比 [AUC/KS delta]，以及按类别/分箱的样本量-违约率-WOE 明细表，供公平信贷与集中度风险复核）

---

## Known Pitfalls (read before touching feature selection)

Two feature-selection bugs previously suppressed this model's discrimination from ~0.88 AUC down to ~0.66 AUC. Both were only caught by diffing against `reports/PD_Model_Development_Document.docx`, an independent reference build on the same dataset. If you're re-deriving `scripts/train.py`'s feature-selection logic from scratch, re-introducing either of these is the most likely way to quietly regress model quality.

1. **One-shot VIF dropping over-prunes collinear features.** `utils/risk_skills.py`'s `compute_vif_filter` computes VIF for a feature set in a single pass and returns *every* feature above the threshold as `recommended_drops`. Dropping all of them at once is wrong: two features can each show high VIF purely because *the other* is present (e.g. `credit_score` and `grade_subgrade` here — both proxy the same underlying risk score). Dropping one of them, alone, is often enough to bring the other's VIF back under threshold. `scripts/train.py`'s `iterative_vif_filter` fixes this: drop only the single highest-VIF feature, recompute VIF on the survivors, repeat until everyone remaining is ≤ threshold. Verified on this dataset: one-shot dropped both `credit_score` (VIF 7.17) and `grade_subgrade` (VIF 6.98); iterative correctly drops only one (recomputed VIF of the other then falls to ~1.3-1.4). Losing `credit_score` this way cost ~0.035 AUC. **Rule: any VIF/correlation-based multicollinearity filter must re-check after each single removal, never drop a whole batch in one pass.**
2. **Auto-excluding IV > 0.5 ("Suspicious / Overfitting") features throws away real signal without review.** `calculate_woe_iv`'s own rating band labels IV > 0.5 as "Suspicious / Overfitting" — but that label means *needs a leakage review*, not *auto-drop*. This pipeline previously auto-excluded any such feature as presumed leakage. On this dataset that feature is `employment_status` (IV ≈ 1.9), which turned out to be a legitimate origination-time attribute — `reports/PD_Model_Development_Document.docx` (§5.1, §7.3, §8) keeps it, explicitly labeling it a governance/fair-lending risk requiring review rather than a leakage bug, and its own sensitivity analysis shows removing it drops Test AUC from 0.887 to 0.705. Reproduced here: removing it drops Test AUC from ~0.88 to ~0.70. **Rule: IV > 0.5 is a flag for human governance review (data provenance + fair-lending), not an automatic exclusion rule a script should apply silently.** `scripts/train.py` now keeps such features in the model (`flagged_as_suspicious` in `feature_selection.json`) and `scripts/test.py` is required to report, for each one: (a) a sensitivity AUC/KS delta with vs. without it, and (b) a per-category/bin breakdown — so a reviewer sees the actual tradeoff instead of a silently-vanished column. Current status for `employment_status` specifically: kept in the model for evaluation purposes, but **not yet cleared for production use** pending real independent validation and fair-lending review — don't strip that caveat out of the report without an actual sign-off to point to.