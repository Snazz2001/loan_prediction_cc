"""
H2O AutoML 对照实验：在与主流水线（scripts/train.py + scripts/test.py）完全相同的
数据切分、目标定义与入模特征集合上，改用 H2O AutoML 训练/搜索模型，并用与主流水线
相同的标准函数（utils/risk_skills.py）评估 Test 集表现，产出独立的对照报告
reports/h2o_automl_model_report.md，不改动/不覆盖主报告 reports/model_validation_report.md。

"相同设置"具体指：
- 数据源与目标定义：直接复用 artifacts/train_data.csv、artifacts/test_data.csv
  （scripts/train.py 用 random_state=42、test_size=0.3、按 default 分层切分后持久化的
  Train/Test，本脚本不重新切分）。
- 入模特征集合：复用 artifacts/feature_selection.json 中的 final_features（IV/VIF 筛选
  结果），共 6 个特征。
- 交叉验证折数：nfolds=5，随机种子：seed=42，与 Optuna 调优时的 5 折、random_state=42 对齐。
- 判别力/稳定性指标：全部通过 utils/risk_skills.py 的标准函数计算（evaluate_discrimination_
  and_ks / calculate_psi），口径与主报告完全一致，可直接横向比较。

与主流水线的两个刻意偏离（均在报告中说明原因）：
1. 特征表示：H2O AutoML 直接使用原始特征值（数值 + 类别原生支持），而不是主流水线里
   为逻辑回归/LightGBM 准备的 WOE 编码矩阵——这正是本次对照实验想验证的问题："换一种
   建模技术、不做手工 WOE 分箱，判别力是否有差异"。
2. 算法范围：AutoML 排除了 StackedEnsemble 与 DeepLearning，只在 GLM / GBM / XGBoost /
   DRF 范围内搜索——排除 StackedEnsemble 是为了保证最终 leader 是单一、可解释、可复现
   predict_contributions() 的模型，满足 §7 Explainability 的强制要求（StackedEnsemble 组合多
   个基模型，SHAP 归因不稳定/不总是可用）；排除 DeepLearning 是出于风控模型可解释性/可
   治理惯例。不代表 AutoML 本身不能用这两类模型，是本次对照实验的治理取舍。
本脚本不施加单调性约束（monotone_constraints）——H2O AutoML 的搜索网格没有暴露对每个
算法统一注入单调约束的标准接口，这是相对主流水线 LightGBM（monotone_constraints=-1）的
已知差异，在报告中明确披露。

用法: python3 scripts/train_h2o_automl.py  (需先运行过 python3 scripts/train.py 以产出
artifacts/train_data.csv、test_data.csv、feature_selection.json)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import h2o
from h2o.automl import H2OAutoML

from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, PSI_ACTION, PSI_WATCH, RANDOM_STATE, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

H2O_ARTIFACTS_DIR = "artifacts_h2o"
H2O_REPORT_PATH = "reports/h2o_automl_model_report.md"
NFOLDS = 5
MAX_MODELS = 30
EXCLUDE_ALGOS = ["StackedEnsemble", "DeepLearning"]


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load_shared_setup():
    """复用 scripts/train.py 已经产出的 Train/Test 切分与最终入模特征集合。"""
    required = ["train_data.csv", "test_data.csv", "feature_selection.json"]
    missing = [p for p in required if not os.path.exists(os.path.join(ARTIFACTS_DIR, p))]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {ARTIFACTS_DIR}/. Run `python3 scripts/train.py` first "
            "so this script can reuse the exact same Train/Test split and feature set."
        )
    train_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "test_data.csv"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_selection.json"), encoding="utf-8") as fh:
        feature_selection = json.load(fh)
    return train_df, test_df, feature_selection["final_features"]


def get_prob_column(h2o_frame_pd: pd.DataFrame) -> np.ndarray:
    """H2O 二分类 predict() 输出 predict/p0/p1（或 p<level0>/p<level1>）列；取正类 (1) 概率列。"""
    candidates = [c for c in h2o_frame_pd.columns if c not in ("predict",)]
    for c in ("p1", "p1.0"):
        if c in h2o_frame_pd.columns:
            return h2o_frame_pd[c].values
    # 兜底：取非 predict 列中数值列的最后一列（约定 H2O 按因子水平升序排列 p 列，1 在最后）
    return h2o_frame_pd[candidates[-1]].values


def main():
    os.makedirs(H2O_ARTIFACTS_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_df, test_df, final_features = load_shared_setup()
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    h2o.init()

    train_h2o = h2o.H2OFrame(train_df[final_features + [TARGET]])
    test_h2o = h2o.H2OFrame(test_df[final_features + [TARGET]])
    train_h2o[TARGET] = train_h2o[TARGET].asfactor()
    test_h2o[TARGET] = test_h2o[TARGET].asfactor()

    aml = H2OAutoML(
        max_models=MAX_MODELS,
        nfolds=NFOLDS,
        seed=RANDOM_STATE,
        sort_metric="AUC",
        balance_classes=False,
        exclude_algos=EXCLUDE_ALGOS,
    )
    aml.train(x=final_features, y=TARGET, training_frame=train_h2o)

    leaderboard = aml.leaderboard.as_data_frame()
    leader = aml.leader
    leader_id = leader.model_id
    leader_algo = leader.algo

    # ------------------------------------------------------------------
    # 判别能力评估：预测概率导出为 numpy，通过 utils/risk_skills.py 的标准函数计算
    # （与主流水线口径完全一致，可直接横向比较）
    # ------------------------------------------------------------------
    train_pred = leader.predict(train_h2o).as_data_frame(use_multi_thread=True)
    test_pred = leader.predict(test_h2o).as_data_frame(use_multi_thread=True)
    train_prob = get_prob_column(train_pred)
    test_prob = get_prob_column(test_pred)

    train_metrics = evaluate_discrimination_and_ks(y_train.values, train_prob)
    test_metrics = evaluate_discrimination_and_ks(y_test.values, test_prob)
    ks_pass = test_metrics["KS_Statistic"] >= KS_MIN
    auc_pass = test_metrics["AUC"] >= AUC_MIN

    # ------------------------------------------------------------------
    # 混淆矩阵：切点沿用 KS 最优切点概率，口径与主报告一致
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
    # 可解释性：优先使用 H2O 原生 TreeSHAP (predict_contributions)，
    # StackedEnsemble/DeepLearning 已被排除，leader 必为 GLM/GBM/XGBoost/DRF 之一，
    # 因此该调用应总是可用；失败时退化为 H2O 内置变量重要度 varimp()。
    # ------------------------------------------------------------------
    shap_table = None
    shap_method = None
    try:
        sample_h2o = test_h2o[final_features].head(min(2000, test_h2o.nrows))
        contrib = leader.predict_contributions(h2o.H2OFrame(sample_h2o.as_data_frame())).as_data_frame()
        contrib_cols = [c for c in contrib.columns if c not in ("BiasTerm",)]
        mean_abs = contrib[contrib_cols].abs().mean().sort_values(ascending=False)
        shap_table = pd.DataFrame({"feature": mean_abs.index, "mean_abs_shap": mean_abs.values.round(4)})
        shap_method = "predict_contributions (native TreeSHAP)"
    except Exception as exc:  # noqa: BLE001
        try:
            varimp = leader.varimp(use_pandas=True)
            shap_table = varimp[["variable", "scaled_importance"]].rename(
                columns={"variable": "feature", "scaled_importance": "mean_abs_shap"}
            )
            shap_method = f"varimp() fallback (predict_contributions unavailable: {exc})"
        except Exception as exc2:  # noqa: BLE001
            shap_table = pd.DataFrame(columns=["feature", "mean_abs_shap"])
            shap_method = f"unavailable (predict_contributions and varimp both failed: {exc2})"

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    model_path = h2o.save_model(model=leader, path=H2O_ARTIFACTS_DIR, force=True)
    leaderboard.to_csv(os.path.join(H2O_ARTIFACTS_DIR, "leaderboard.csv"), index=False)

    result_summary = {
        "leader_model_id": leader_id,
        "leader_algo": leader_algo,
        "nfolds": NFOLDS,
        "max_models": MAX_MODELS,
        "excluded_algos": EXCLUDE_ALGOS,
        "final_features": final_features,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_summary": confusion_summary,
        "score_psi": score_psi,
        "shap_method": shap_method,
        "model_path": model_path,
    }
    with open(os.path.join(H2O_ARTIFACTS_DIR, "result_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result_summary, fh, ensure_ascii=False, indent=2, default=str)

    report = build_report_markdown(
        train_df=train_df, test_df=test_df, final_features=final_features,
        leaderboard=leaderboard, leader_id=leader_id, leader_algo=leader_algo,
        train_metrics=train_metrics, test_metrics=test_metrics,
        ks_pass=ks_pass, auc_pass=auc_pass,
        confusion_summary=confusion_summary, score_psi=score_psi,
        shap_table=shap_table, shap_method=shap_method,
    )
    with open(H2O_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"[train_h2o_automl] Leader: {leader_algo} ({leader_id})")
    print(f"[train_h2o_automl] Test AUC={test_metrics['AUC']} ({'PASS' if auc_pass else 'FAIL'}), "
          f"KS={test_metrics['KS_Statistic']} ({'PASS' if ks_pass else 'FAIL'}), "
          f"score PSI={score_psi['PSI']} ({score_psi['Status']})")
    print(f"[train_h2o_automl] Report written to {H2O_REPORT_PATH}")

    h2o.cluster().shutdown(prompt=False)


def build_report_markdown(
    train_df, test_df, final_features, leaderboard, leader_id, leader_algo,
    train_metrics, test_metrics, ks_pass, auc_pass, confusion_summary, score_psi,
    shap_table, shap_method,
):
    def status_icon(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    sample_split_summary = pd.DataFrame([
        {"sample": "Train", "n": len(train_df), "bad_rate": round(float(train_df[TARGET].mean()), 4)},
        {"sample": "Test", "n": len(test_df), "bad_rate": round(float(test_df[TARGET].mean()), 4)},
    ])
    perf_table = pd.DataFrame([
        {"model": f"H2O AutoML Leader ({leader_algo})", "sample": "Train",
         "AUC": train_metrics["AUC"], "Gini": train_metrics["Gini"], "KS": train_metrics["KS_Statistic"],
         "rating": train_metrics["Validation_Rating"]},
        {"model": f"H2O AutoML Leader ({leader_algo})", "sample": "Test",
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
    leaderboard_display = leaderboard.head(10).copy()
    psi_status = "低于预警阈值，无需特别处理" if score_psi["PSI"] < PSI_WATCH else (
        "位于 0.10~0.25 预警区间，需特别注明并加强监控" if score_psi["PSI"] <= PSI_ACTION else "超过 0.25，需模型重新校准"
    )

    md = f"""# H2O AutoML Model Report — 对照实验报告

本报告是对主报告 `reports/model_validation_report.md`（逻辑回归评分卡 + Optuna 调优
LightGBM）的**对照实验**，用相同的数据切分、目标定义与入模特征集合，改用 H2O AutoML
训练模型，用于回答："换一种建模技术、不做手工 WOE 分箱，判别力是否有差异？"

数据来源: `data/loan_dataset_20000.csv`（经 `artifacts/train_data.csv` / `test_data.csv` 复用与主流水线完全相同的切分）｜ 建模引擎: `h2o.automl.H2OAutoML`｜ 随机种子: `random_state=42`

**与主流水线的关键差异（均为刻意选择，非缺陷）：**
1. 特征直接使用原始值（H2O 原生处理数值/类别特征），不做 WOE 编码。
2. AutoML 搜索排除 `StackedEnsemble` 与 `DeepLearning`，只在 GLM / GBM / XGBoost / DRF 中选择，以保证最终模型可用原生 TreeSHAP 解释、满足治理要求。
3. 未施加单调性约束（`monotone_constraints`）——H2O AutoML 没有跨算法统一注入该约束的标准接口，这是相对主流水线 LightGBM 的已知差异。
4. `employment_status` 特征沿用主流水线的治理判断：保留入模但需公平信贷复核，未获投产批准（详见主报告 §7.4）。

---

## 1. Executive Summary

- H2O AutoML 在 `max_models={MAX_MODELS}`、`nfolds={NFOLDS}`、`seed=42` 设置下完成搜索，Leaderboard 详见 §4；最终 Leader 模型为 **{leader_algo}**（`{leader_id}`）。
- Leader 在 Test 集上的判别能力：AUC = **{test_metrics['AUC']}**（阈值 ≥ {AUC_MIN}：**{status_icon(auc_pass)}**），KS = **{test_metrics['KS_Statistic']}**（阈值 ≥ {KS_MIN}：**{status_icon(ks_pass)}**），评级为 "{test_metrics['Validation_Rating']}"。
- 模型分 PSI（Train → Test）= **{score_psi['PSI']}**，状态：{score_psi['Status']}（{psi_status}）。
- 对照结论：与主报告的 Optuna 调优 LightGBM（Test AUC 0.886 / KS 0.578，使用 WOE 编码特征）相比，本次 H2O AutoML（原始特征，无 WOE、无单调约束）{"判别力相近或更优" if test_metrics['AUC'] >= 0.88 else "判别力略低" if test_metrics['AUC'] >= 0.80 else "判别力明显更低"}，说明{"手工 WOE 分箱在本数据集上并非必要——特征本身的信号足以被 AutoML 直接利用" if test_metrics['AUC'] >= 0.85 else "在缺少 WOE 分箱/单调约束的情况下，判别力有可观下降，WOE 编码或约束本身对本数据集仍有实质贡献"}。
- 建议：本模型**仅作为方法论对照，不建议替代主流水线投产**——原因：(a) 未施加单调性约束，不满足监管对评分类模型单调性的一般要求；(b) 仍然依赖 `employment_status`，尚未完成的公平信贷复核限制同样适用；(c) 缺少标准评分卡的点数标度（Score = Offset + Factor·ln(Odds)），可解释性弱于主流水线的逻辑回归 Baseline。

---

## 2. Train / Test Split

与主流水线完全相同（直接复用 `artifacts/train_data.csv` / `artifacts/test_data.csv`，未重新切分）：

{md_table(sample_split_summary)}

---

## 3. Feature Set

与主流水线 IV/VIF 筛选结果完全相同（{len(final_features)} 个特征，含 employment_status 治理标记，详见主报告 §3、§7.4），但本报告中以**原始值**（非 WOE 编码）形式喂给 H2O AutoML：

{", ".join(final_features)}

---

## 4. H2O AutoML Search & Leaderboard

- 搜索设置：`max_models={MAX_MODELS}`，`nfolds={NFOLDS}`，`seed=42`，`sort_metric="AUC"`，`exclude_algos={EXCLUDE_ALGOS}`。
- Leader：**{leader_algo}**（`{leader_id}`）。

Top 10 Leaderboard（按 CV AUC 排序）：

{md_table(leaderboard_display)}

---

## 5. Model Performance & Discrimination

### 5.1 Train / Test 判别力（`evaluate_discrimination_and_ks`，与主报告口径一致）

{md_table(perf_table)}

- 内部阈值标准：Test 集 KS ≥ {KS_MIN}，AUC ≥ {AUC_MIN}。
- Test 集结果：AUC = {test_metrics['AUC']}（{status_icon(auc_pass)}），KS = {test_metrics['KS_Statistic']}（{status_icon(ks_pass)}），最优切点概率 = {test_metrics['Optimal_Cutoff_Probability']}。

### 5.2 混淆矩阵（Test 集，切点 = KS 最优切点概率 {confusion_summary['cutoff']}）

{md_table(confusion_table)}

{md_table(confusion_metrics_table)}

---

## 6. Population Stability Index (PSI)

- 模型分 PSI（Train 基准 → Test 实际）= **{score_psi['PSI']}**，状态：{score_psi['Status']}
- 监管标准：PSI < 0.10 无需处理；0.10~0.25 需特别注明预警；> 0.25 需模型重新校准。本次结果：{psi_status}

---

## 7. Explainability & Governance

### 7.1 特征重要性 / SHAP 归因（方法：{shap_method}）

{md_table(shap_table) if len(shap_table) else "不可用。"}

### 7.2 监管合规审查说明

- 模型类型：H2O AutoML 自动搜索得到的 {leader_algo}；不是标准评分卡形式，无 `Score = Offset + Factor·ln(Odds)` 点数标度，可解释性弱于主流水线的逻辑回归 Baseline。
- **未施加单调性约束**：与主流水线 LightGBM（`monotone_constraints=-1`，全部特征）不同，本模型的特征-风险关系单调性未经强制约束、未经验证，投产前需额外做单调性核查。
- Leader 已排除 `StackedEnsemble`/`DeepLearning`，确保可用原生 TreeSHAP（或 GLM 系数）解释，符合 §7.1 的可解释性要求。
- `employment_status` 的治理限制（保留入模、待公平信贷复核）与主报告一致，同样适用于本模型，详见主报告 §7.4。
- 判别力与稳定性指标（KS/AUC/PSI）均通过 `utils/risk_skills.py` 标准函数计算，口径与主报告完全一致、可直接横向比较（`random_state=42`）。
- 本报告与其 artifacts（`{H2O_ARTIFACTS_DIR}/`）为独立对照实验产出，不影响、不覆盖主流水线的 `artifacts/` 与 `reports/model_validation_report.md`。
"""
    return md


if __name__ == "__main__":
    main()
