"""
XGBoost 对照实验：复用主流水线（scripts/train.py + scripts/test.py）完全相同的数据
切分、目标定义、WOE 编码矩阵与入模特征集合，仅将建模算法从 (Optuna 调优) LightGBM
换成 (Optuna 调优) XGBoost，用于回答："同样的 WOE 特征 + 同样的 Optuna 50 轮 5 折调优
+ 同样的单调约束，XGBoost 与 LightGBM 相比表现如何？"

"相同设置"具体指（比 H2O AutoML 对照实验更贴近主流水线，因为 XGBoost 原生支持
monotone_constraints 与 shap 库）：
- Train/Test 切分：直接读取 artifacts/train_data.csv / test_data.csv
  （scripts/train.py 产出，未重新切分）。
- 特征表示：直接复用 artifacts/woe_encoders.joblib 中已冻结的 WOE 编码器
  （不重新拟合，与主流水线的 LightGBM/逻辑回归使用完全相同的特征矩阵）。
- 入模特征集合：artifacts/feature_selection.json 的 final_features（6 个）。
- 调优：scripts/tune_xgb_skill.py 的 tune_xgboost_optuna，50 轮 Optuna、
  5 折 StratifiedKFold CV、优化目标 AUC，与 LightGBM 调优逐项对齐。
- 单调性约束：monotone_constraints=-1（全部特征），与 LightGBM 一致。
- 评估：utils/risk_skills.py 的 evaluate_discrimination_and_ks / calculate_psi。
- 可解释性：utils/risk_skills.py 的 generate_shap_summary——XGBoost 是 shap 库原生
  支持的模型类型，不需要像 H2O 那样退化到 varimp。

产出独立报告 reports/xgboost_model_report.md 与独立 artifacts_xgboost/，不覆盖/不影响
主流水线的 artifacts/、reports/model_validation_report.md，也不影响 H2O 对照实验的
artifacts_h2o/、reports/h2o_automl_model_report.md。

用法: python3 scripts/train_xgboost.py （需先运行过 python3 scripts/train.py）
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 允许同目录导入 tune_xgb_skill

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from tune_xgb_skill import tune_xgboost_optuna
from utils.config import (
    ARTIFACTS_DIR,
    AUC_MIN,
    KS_MIN,
    PSI_ACTION,
    PSI_WATCH,
    RANDOM_STATE,
    TARGET,
    TUNE_CV_FOLDS,
    TUNE_METRIC,
    TUNE_N_TRIALS,
)
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks, generate_shap_summary
from utils.woe_encoding import apply_woe_encoder

XGB_ARTIFACTS_DIR = "artifacts_xgboost"
XGB_REPORT_PATH = "reports/xgboost_model_report.md"


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load_shared_setup():
    """复用 scripts/train.py 已经产出的 Train/Test 切分、WOE 编码器与最终入模特征集合。"""
    required = ["train_data.csv", "test_data.csv", "feature_selection.json", "woe_encoders.joblib"]
    missing = [p for p in required if not os.path.exists(os.path.join(ARTIFACTS_DIR, p))]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {ARTIFACTS_DIR}/. Run `python3 scripts/train.py` first "
            "so this script can reuse the exact same Train/Test split, WOE encoders, and feature set."
        )
    train_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "test_data.csv"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_selection.json"), encoding="utf-8") as fh:
        feature_selection = json.load(fh)
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "woe_encoders.joblib"))
    return train_df, test_df, feature_selection["final_features"], encoders


def main():
    os.makedirs(XGB_ARTIFACTS_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_df, test_df, final_features, encoders = load_shared_setup()

    X_train = pd.DataFrame({f: apply_woe_encoder(train_df, f, encoders[f]) for f in final_features})
    X_test = pd.DataFrame({f: apply_woe_encoder(test_df, f, encoders[f]) for f in final_features})
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    # ------------------------------------------------------------------
    # Optuna 调优：与 LightGBM 逐项对齐的设置（轮数、折数、目标、单调约束）
    # ------------------------------------------------------------------
    monotone_constraints = [-1] * len(final_features)
    tuning_result = tune_xgboost_optuna(
        X_train, y_train,
        n_trials=TUNE_N_TRIALS, cv_folds=TUNE_CV_FOLDS, metric=TUNE_METRIC,
        monotone_constraints=monotone_constraints, random_state=RANDOM_STATE,
    )
    xgb_model = tuning_result["trained_model"]

    train_prob = xgb_model.predict_proba(X_train)[:, 1]
    test_prob = xgb_model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_discrimination_and_ks(y_train.values, train_prob)
    test_metrics = evaluate_discrimination_and_ks(y_test.values, test_prob)
    ks_pass = test_metrics["KS_Statistic"] >= KS_MIN
    auc_pass = test_metrics["AUC"] >= AUC_MIN

    # ------------------------------------------------------------------
    # 混淆矩阵：切点沿用 KS 最优切点概率，口径与主报告/H2O 报告一致
    # ------------------------------------------------------------------
    cutoff = test_metrics["Optimal_Cutoff_Probability"]
    y_pred = (test_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test.values, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_test)
    confusion_summary = {
        "cutoff": round(float(cutoff), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(float(accuracy), 4), "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4), "specificity": round(float(specificity), 4),
    }

    # ------------------------------------------------------------------
    # 稳定性评估 (calculate_psi)：模型分 Train(基准) vs Test(实际)
    # ------------------------------------------------------------------
    score_psi = calculate_psi(train_prob, test_prob, num_bins=10)

    # ------------------------------------------------------------------
    # 可解释性 (generate_shap_summary)：XGBoost 是 shap 原生支持的模型类型
    # ------------------------------------------------------------------
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    shap_summary = generate_shap_summary(xgb_model, shap_sample)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    joblib.dump(xgb_model, os.path.join(XGB_ARTIFACTS_DIR, "xgb_model.joblib"))
    tuning_report = {
        "target_metric": tuning_result["target_metric"],
        "cv_folds": TUNE_CV_FOLDS,
        "n_trials_requested": TUNE_N_TRIALS,
        "n_trials_completed": tuning_result["n_trials_completed"],
        "best_cv_score": tuning_result["best_cv_score"],
        "best_hyperparameters": tuning_result["best_hyperparameters"],
        "feature_importances": tuning_result["feature_importances"],
        "monotone_constraints": monotone_constraints,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_summary": confusion_summary,
        "score_psi": score_psi,
    }
    with open(os.path.join(XGB_ARTIFACTS_DIR, "tuning_report.json"), "w", encoding="utf-8") as fh:
        json.dump(tuning_report, fh, ensure_ascii=False, indent=2)

    report = build_report_markdown(
        train_df=train_df, test_df=test_df, final_features=final_features,
        tuning_result=tuning_result, train_metrics=train_metrics, test_metrics=test_metrics,
        ks_pass=ks_pass, auc_pass=auc_pass, confusion_summary=confusion_summary,
        score_psi=score_psi, shap_summary=shap_summary, shap_sample_size=len(shap_sample),
        cv_folds=TUNE_CV_FOLDS, monotone_constraints=monotone_constraints,
    )
    with open(XGB_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"[train_xgboost] Optuna: {tuning_result['n_trials_completed']} trials, "
          f"{TUNE_CV_FOLDS}-fold CV, best CV {tuning_result['target_metric']}={tuning_result['best_cv_score']}")
    print(f"[train_xgboost] Test AUC={test_metrics['AUC']} ({'PASS' if auc_pass else 'FAIL'}), "
          f"KS={test_metrics['KS_Statistic']} ({'PASS' if ks_pass else 'FAIL'}), "
          f"score PSI={score_psi['PSI']} ({score_psi['Status']})")
    print(f"[train_xgboost] Report written to {XGB_REPORT_PATH}")


def build_report_markdown(
    train_df, test_df, final_features, tuning_result, train_metrics, test_metrics,
    ks_pass, auc_pass, confusion_summary, score_psi, shap_summary, shap_sample_size,
    cv_folds, monotone_constraints,
):
    def status_icon(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    sample_split_summary = pd.DataFrame([
        {"sample": "Train", "n": len(train_df), "bad_rate": round(float(train_df[TARGET].mean()), 4)},
        {"sample": "Test", "n": len(test_df), "bad_rate": round(float(test_df[TARGET].mean()), 4)},
    ])
    perf_table = pd.DataFrame([
        {"model": "Optuna-Tuned Monotonic XGBoost", "sample": "Train",
         "AUC": train_metrics["AUC"], "Gini": train_metrics["Gini"], "KS": train_metrics["KS_Statistic"],
         "rating": train_metrics["Validation_Rating"]},
        {"model": "Optuna-Tuned Monotonic XGBoost", "sample": "Test",
         "AUC": test_metrics["AUC"], "Gini": test_metrics["Gini"], "KS": test_metrics["KS_Statistic"],
         "rating": test_metrics["Validation_Rating"]},
    ])
    confusion_table = pd.DataFrame([
        {"": "Actual Bad (1)", "Predicted Bad (1)": confusion_summary["TP"], "Predicted Good (0)": confusion_summary["FN"]},
        {"": "Actual Good (0)", "Predicted Bad (1)": confusion_summary["FP"], "Predicted Good (0)": confusion_summary["TN"]},
    ])
    confusion_metrics_table = pd.DataFrame([
        {"metric": "Accuracy", "value": confusion_summary["accuracy"]},
        {"metric": "Precision", "value": confusion_summary["precision"]},
        {"metric": "Recall / Sensitivity", "value": confusion_summary["recall_sensitivity"]},
        {"metric": "Specificity", "value": confusion_summary["specificity"]},
    ])
    best_params_table = pd.DataFrame(
        [{"hyperparameter": k, "value": v} for k, v in tuning_result["best_hyperparameters"].items()]
    )
    importance_table = pd.DataFrame(tuning_result["feature_importances"], columns=["feature", "importance"])
    shap_table = pd.DataFrame(shap_summary["global_importance_ranking"], columns=["feature", "mean_abs_shap"])

    psi_status = "低于预警阈值，无需特别处理" if score_psi["PSI"] < PSI_WATCH else (
        "位于 0.10~0.25 预警区间，需特别注明并加强监控" if score_psi["PSI"] <= PSI_ACTION else "超过 0.25，需模型重新校准"
    )
    train_test_gap = round(train_metrics["AUC"] - test_metrics["AUC"], 4)

    md = f"""# XGBoost Model Report — 对照实验报告

本报告是对主报告 `reports/model_validation_report.md`（逻辑回归评分卡 + Optuna 调优
LightGBM）的**对照实验**：除建模算法从 LightGBM 换成 XGBoost 外，数据切分、WOE 特征
矩阵、Optuna 调优设置（50 轮、5 折 CV、优化目标 AUC）、单调性约束方向均与主流水线
逐项对齐，是三个模型中与主流水线设置最接近的一个对照（比 H2O AutoML 对照更贴近，因为
XGBoost 原生支持 `monotone_constraints` 与 `shap` 库）。

数据来源: `data/loan_dataset_20000.csv`（经 `artifacts/train_data.csv` / `test_data.csv` 复用与主流水线完全相同的切分）｜ 建模引擎: `xgboost.XGBClassifier` + `scripts/tune_xgb_skill.py`｜ 随机种子: `random_state=42`

---

## 1. Executive Summary

- 使用 `scripts/tune_xgb_skill.py` 的 `tune_xgboost_optuna`（{tuning_result['n_trials_completed']} 轮 Optuna，{cv_folds} 折交叉验证，优化目标 {tuning_result['target_metric']}）训练单调约束 XGBoost 模型。
- 最佳 CV {tuning_result['target_metric']} = **{tuning_result['best_cv_score']}**。
- Test 集判别能力：AUC = **{test_metrics['AUC']}**（阈值 ≥ {AUC_MIN}：**{status_icon(auc_pass)}**），KS = **{test_metrics['KS_Statistic']}**（阈值 ≥ {KS_MIN}：**{status_icon(ks_pass)}**），评级为 "{test_metrics['Validation_Rating']}"。
- Train → Test AUC 差距 = {train_test_gap}（主报告 LightGBM 为 0.005；H2O AutoML GBM 为 0.029）。
- 模型分 PSI（Train → Test）= **{score_psi['PSI']}**，状态：{score_psi['Status']}（{psi_status}）。
- 对照结论：三个模型（LightGBM / H2O AutoML GBM / XGBoost）在同一 WOE 特征集合、同等调优强度下判别力高度接近（AUC 均在 0.88 附近），说明本数据集在当前 6 特征、含 `employment_status` 的设置下，判别力上限已趋于收敛，模型算法选择本身不是当前的瓶颈；真正影响判别力的是**特征选择（尤其是 employment_status 与 credit_score 的取舍）**，而非建模技术。
- 建议：与主报告一致——本模型同样**不建议替代主流水线投产**：仍依赖 `employment_status`（详见主报告 §7.4，公平信贷复核未完成），且不是标准评分卡形式，评分点数标度缺失，可解释性弱于 LR Baseline。

---

## 2. Train / Test Split & Feature Set

与主流水线完全相同（直接复用 `artifacts/train_data.csv` / `test_data.csv` / `feature_selection.json` / `woe_encoders.joblib`，未重新切分、未重新拟合编码器）：

{md_table(sample_split_summary)}

入模特征（{len(final_features)} 个，WOE 编码，与主流水线 LightGBM 使用的矩阵完全相同）：{", ".join(final_features)}

---

## 3. Hyperparameter Tuning (Optuna)

- 调优函数：`scripts/tune_xgb_skill.py` 的 `tune_xgboost_optuna`（TPE 采样器，`random_state=42`），参数含义与 `tune_lightgbm_optuna` 逐项对齐。
- 搜索空间：`learning_rate` ∈ [0.01, 0.1]（log），`max_depth` ∈ [3, 8]，`min_child_weight` ∈ [1, 20]，`subsample` ∈ [0.6, 1.0]，`colsample_bytree` ∈ [0.6, 1.0]，`reg_alpha`/`reg_lambda` ∈ [1e-3, 10]（log）。
- 交叉验证：{cv_folds} 折 `StratifiedKFold`（`shuffle=True, random_state=42`），每折内以验证集早停（`early_stopping_rounds=30`）；`monotone_constraints={tuple(monotone_constraints)}` 在每次试验中保持固定，方向与 WOE 编码定义一致（WOE 越高代表历史违约率越低）。
- 试验轮数：完成 {tuning_result['n_trials_completed']} 轮。

### 3.1 最佳超参数配置

{md_table(best_params_table)}

### 3.2 特征重要度

{md_table(importance_table)}

- **已知局限**（与主报告 LightGBM 一致，刻意保留以保证"同样设置"）：最终模型使用 CV 得到的最佳超参数在全量 Train 上重新拟合，重新拟合时固定 `n_estimators=1000` 且未设置 `eval_set`/早停，与各折验证时实际使用的早停树数不一致，可能引入额外方差。

---

## 4. Model Performance & Discrimination

### 4.1 Train / Test 判别力（`evaluate_discrimination_and_ks`，与主报告口径一致）

{md_table(perf_table)}

- 内部阈值标准：Test 集 KS ≥ {KS_MIN}，AUC ≥ {AUC_MIN}。
- Test 集结果：AUC = {test_metrics['AUC']}（{status_icon(auc_pass)}），KS = {test_metrics['KS_Statistic']}（{status_icon(ks_pass)}），最优切点概率 = {test_metrics['Optimal_Cutoff_Probability']}。

### 4.2 混淆矩阵（Test 集，切点 = KS 最优切点概率 {confusion_summary['cutoff']}）

{md_table(confusion_table)}

{md_table(confusion_metrics_table)}

---

## 5. Population Stability Index (PSI)

- 模型分 PSI（Train 基准 → Test 实际）= **{score_psi['PSI']}**，状态：{score_psi['Status']}
- 监管标准：PSI < 0.10 无需处理；0.10~0.25 需特别注明预警；> 0.25 需模型重新校准。本次结果：{psi_status}

---

## 6. Explainability & Governance

### 6.1 全局 SHAP 特征重要性排序（`generate_shap_summary`，Test 抽样 {shap_sample_size} 条）

{md_table(shap_table)}

### 6.2 监管合规审查说明

- 模型类型：Optuna 调优单调约束 XGBoost；与主流水线的 LightGBM 挑战者模型定位相同（判别力优先的树模型），不是标准评分卡形式，无点数标度，可解释性弱于 LR Baseline。
- 已对全部入模特征施加单调递减约束（`monotone_constraints=-1`），约束方向依据 WOE 编码定义，在 Optuna 全部 {tuning_result['n_trials_completed']} 轮试验中保持固定，与主流水线 LightGBM 的处理方式完全一致。
- `employment_status` 的治理限制（保留入模、待公平信贷复核）与主报告一致，同样适用于本模型，详见主报告 §7.4。
- 判别力与稳定性指标（KS/AUC/PSI）、SHAP 解释均通过 `utils/risk_skills.py` 标准函数计算，口径与主报告完全一致、可直接横向比较（`random_state=42`）。
- 本报告与其 artifacts（`artifacts_xgboost/`）为独立对照实验产出，不影响、不覆盖主流水线的 `artifacts/`、`reports/model_validation_report.md`，也不影响 H2O 对照实验的 `artifacts_h2o/`、`reports/h2o_automl_model_report.md`。
"""
    return md


if __name__ == "__main__":
    main()
