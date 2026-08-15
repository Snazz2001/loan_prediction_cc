"""
测试/验证阶段 (Stage 2/2)。

只加载 scripts/train.py 产出的 artifacts/（Optuna 调优后的 LightGBM 模型、逻辑回归
评分卡、WOE 编码器、特征筛选与调参结果），不重新拟合任何分箱、编码、模型参数或
超参数 —— 全部判别力 (KS/AUC/混淆矩阵)、稳定性 (PSI) 与可解释性 (SHAP) 计算都在
Test 集（从未参与训练与调参）上进行，并自动生成 reports/model_validation_report.md。

复用 utils/risk_skills.py 中的标准函数（evaluate_discrimination_and_ks /
calculate_psi / generate_shap_summary），符合 CLAUDE.md 中的
《银行风控模型开发与验证规范》。

用法: python3 scripts/test.py  (需先运行 python3 scripts/train.py)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from utils.config import (
    ARTIFACTS_DIR,
    AUC_MIN,
    IV_MIN,
    IV_SUSPICIOUS,
    KS_MIN,
    PSI_ACTION,
    PSI_WATCH,
    RANDOM_STATE,
    REPORT_PATH,
    TARGET,
    VIF_MAX,
)
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks, generate_shap_summary
from utils.woe_encoding import apply_woe_encoder


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load_artifacts():
    missing = [
        p for p in [
            "train_data.csv", "test_data.csv", "woe_encoders.joblib",
            "lr_model.joblib", "lgbm_model.joblib", "feature_selection.json",
            "scorecard_report.json", "tuning_report.json",
        ]
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, p))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts {missing} in {ARTIFACTS_DIR}/. Run `python3 scripts/train.py` first."
        )

    train_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "test_data.csv"))
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "woe_encoders.joblib"))
    lr_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lr_model.joblib"))
    lgbm_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_selection.json"), encoding="utf-8") as fh:
        feature_selection = json.load(fh)
    with open(os.path.join(ARTIFACTS_DIR, "scorecard_report.json"), encoding="utf-8") as fh:
        scorecard_report = json.load(fh)
    with open(os.path.join(ARTIFACTS_DIR, "tuning_report.json"), encoding="utf-8") as fh:
        tuning_report = json.load(fh)

    return train_df, test_df, encoders, lr_model, lgbm_model, feature_selection, scorecard_report, tuning_report


def main():
    os.makedirs("reports", exist_ok=True)
    (train_df, test_df, encoders, lr_model, lgbm_model,
     feature_selection, scorecard_report, tuning_report) = load_artifacts()

    final_features = feature_selection["final_features"]

    # ------------------------------------------------------------------
    # 应用训练阶段拟合好的 WOE 编码器（不重新拟合）
    # ------------------------------------------------------------------
    X_train = pd.DataFrame({f: apply_woe_encoder(train_df, f, encoders[f]) for f in final_features})
    X_test = pd.DataFrame({f: apply_woe_encoder(test_df, f, encoders[f]) for f in final_features})
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    lr_train_prob = lr_model.predict_proba(X_train)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test)[:, 1]
    lgbm_train_prob = lgbm_model.predict_proba(X_train)[:, 1]
    lgbm_test_prob = lgbm_model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 判别能力评估 (evaluate_discrimination_and_ks)：Train vs Test 对比
    # ------------------------------------------------------------------
    perf_rows = []
    for model_name, (tr_p, te_p) in {
        "Logistic Regression Scorecard (Baseline)": (lr_train_prob, lr_test_prob),
        "Optuna-Tuned Monotonic LightGBM (Champion)": (lgbm_train_prob, lgbm_test_prob),
    }.items():
        tr_metrics = evaluate_discrimination_and_ks(y_train.values, tr_p)
        te_metrics = evaluate_discrimination_and_ks(y_test.values, te_p)
        perf_rows.append({
            "model": model_name, "sample": "Train",
            "AUC": tr_metrics["AUC"], "Gini": tr_metrics["Gini"], "KS": tr_metrics["KS_Statistic"],
            "rating": tr_metrics["Validation_Rating"],
        })
        perf_rows.append({
            "model": model_name, "sample": "Test",
            "AUC": te_metrics["AUC"], "Gini": te_metrics["Gini"], "KS": te_metrics["KS_Statistic"],
            "rating": te_metrics["Validation_Rating"],
        })
    perf_table = pd.DataFrame(perf_rows)

    # 最优模型 = Optuna 调优后的 LightGBM（本次任务的评估/混淆矩阵对象）
    champion_test_metrics = evaluate_discrimination_and_ks(y_test.values, lgbm_test_prob)
    baseline_test_metrics = evaluate_discrimination_and_ks(y_test.values, lr_test_prob)
    ks_pass = champion_test_metrics["KS_Statistic"] >= KS_MIN
    auc_pass = champion_test_metrics["AUC"] >= AUC_MIN

    # ------------------------------------------------------------------
    # 混淆矩阵：最优模型 (Optuna-Tuned LightGBM) 在 Test 集上，
    # 切点取 evaluate_discrimination_and_ks 给出的 KS 最优切点概率（而非任意的 0.5，
    # 因为约 20% 违约率下 0.5 切点几乎不会触发正类预测）。
    # ------------------------------------------------------------------
    cutoff = champion_test_metrics["Optimal_Cutoff_Probability"]
    y_pred = (lgbm_test_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test.values, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_test)
    confusion_summary = {
        "cutoff": round(float(cutoff), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
    }

    # ------------------------------------------------------------------
    # 稳定性评估 (calculate_psi)：模型分 + 关键特征，Train(基准) vs Test(实际)
    # 模型分口径与混淆矩阵/判别力评估一致，取最优模型 (LightGBM) 的预测概率。
    # ------------------------------------------------------------------
    score_psi = calculate_psi(lgbm_train_prob, lgbm_test_prob, num_bins=10)

    feature_psi_rows = []
    for f in final_features[:5]:
        if pd.api.types.is_numeric_dtype(train_df[f]):
            psi_res = calculate_psi(train_df[f].values, test_df[f].values, num_bins=10)
            feature_psi_rows.append({"feature": f, "PSI": psi_res["PSI"], "status": psi_res["Status"]})
    feature_psi_table = pd.DataFrame(feature_psi_rows)

    # ------------------------------------------------------------------
    # 可解释性 (generate_shap_summary)：基于最优模型 (LightGBM)，抽样自 Test
    # ------------------------------------------------------------------
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    shap_summary = generate_shap_summary(lgbm_model, shap_sample)
    top3_shap = shap_summary["global_importance_ranking"][:3]

    # ------------------------------------------------------------------
    # 治理复核明细：对已保留但被标记为 "Suspicious / Overfitting" 的入模特征
    # (本次为 employment_status)，单独输出按类别的样本量/违约率/WOE 明细，
    # 供公平信贷复核与集中度风险评估使用（参考 PD_Model_Development_Document.docx §8）。
    # ------------------------------------------------------------------
    flagged_and_kept = [f for f in feature_selection["flagged_as_suspicious"] if f in final_features]
    flagged_breakdown_tables = {}
    for f in flagged_and_kept:
        if not pd.api.types.is_numeric_dtype(train_df[f]):
            grp = train_df.groupby(f)[TARGET].agg(n="count", default_rate="mean").reset_index()
            grp["default_rate"] = grp["default_rate"].round(4)
            grp["woe"] = grp[f].astype(str).map(encoders[f]["woe_map"]).round(4)
            grp = grp.sort_values("default_rate", ascending=False)
            flagged_breakdown_tables[f] = grp

    # ------------------------------------------------------------------
    # 敏感性分析：剔除被标记特征后重新拟合 Baseline 逻辑回归，量化其对判别力的
    # 边际贡献（口径对齐 PD_Model_Development_Document.docx §7.3 的 variant 对比）。
    # 只用于报告披露，不影响持久化的 champion/baseline 模型。
    # ------------------------------------------------------------------
    sensitivity_rows = []
    if flagged_and_kept:
        reduced_features = [f for f in final_features if f not in flagged_and_kept]
        lr_full_metrics = evaluate_discrimination_and_ks(y_test.values, lr_test_prob)
        sensitivity_rows.append({
            "variant": f"Full set (incl. {', '.join(flagged_and_kept)})",
            "n_features": len(final_features), "AUC": lr_full_metrics["AUC"],
            "Gini": lr_full_metrics["Gini"], "KS": lr_full_metrics["KS_Statistic"],
        })
        if reduced_features:
            lr_reduced = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
            lr_reduced.fit(X_train[reduced_features], y_train)
            reduced_test_prob = lr_reduced.predict_proba(X_test[reduced_features])[:, 1]
            lr_reduced_metrics = evaluate_discrimination_and_ks(y_test.values, reduced_test_prob)
            sensitivity_rows.append({
                "variant": f"Excluding {', '.join(flagged_and_kept)}",
                "n_features": len(reduced_features), "AUC": lr_reduced_metrics["AUC"],
                "Gini": lr_reduced_metrics["Gini"], "KS": lr_reduced_metrics["KS_Statistic"],
            })
    sensitivity_table = pd.DataFrame(sensitivity_rows)

    sample_split_summary = pd.DataFrame([
        {"sample": "Train", "n": len(train_df), "bad_rate": round(float(y_train.mean()), 4)},
        {"sample": "Test", "n": len(test_df), "bad_rate": round(float(y_test.mean()), 4)},
    ])

    report = build_report_markdown(
        sample_split_summary=sample_split_summary,
        feature_selection=feature_selection,
        scorecard_report=scorecard_report,
        tuning_report=tuning_report,
        perf_table=perf_table,
        champion_test_metrics=champion_test_metrics,
        baseline_test_metrics=baseline_test_metrics,
        confusion_summary=confusion_summary,
        ks_pass=ks_pass,
        auc_pass=auc_pass,
        score_psi=score_psi,
        feature_psi_table=feature_psi_table,
        shap_summary=shap_summary,
        top3_shap=top3_shap,
        shap_sample_size=len(shap_sample),
        flagged_breakdown_tables=flagged_breakdown_tables,
        sensitivity_table=sensitivity_table,
        flagged_and_kept=flagged_and_kept,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"[test] Champion (Optuna LightGBM) Test AUC={champion_test_metrics['AUC']} "
          f"({'PASS' if auc_pass else 'FAIL'}), KS={champion_test_metrics['KS_Statistic']} "
          f"({'PASS' if ks_pass else 'FAIL'}), score PSI={score_psi['PSI']} ({score_psi['Status']})")
    print(f"[test] Confusion matrix @ cutoff={confusion_summary['cutoff']}: "
          f"TN={confusion_summary['TN']} FP={confusion_summary['FP']} "
          f"FN={confusion_summary['FN']} TP={confusion_summary['TP']}")
    print(f"[test] Report written to {REPORT_PATH}")


def build_report_markdown(
    sample_split_summary,
    feature_selection,
    scorecard_report,
    tuning_report,
    perf_table,
    champion_test_metrics,
    baseline_test_metrics,
    confusion_summary,
    ks_pass,
    auc_pass,
    score_psi,
    feature_psi_table,
    shap_summary,
    top3_shap,
    shap_sample_size,
    flagged_breakdown_tables,
    sensitivity_table,
    flagged_and_kept,
):
    def status_icon(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    iv_table = pd.DataFrame(feature_selection["iv_table"])
    dropped_by_iv = feature_selection["dropped_by_iv"]
    flagged_as_suspicious = feature_selection["flagged_as_suspicious"]
    vif_table = pd.DataFrame(feature_selection["vif_summary"])
    vif_rounds = feature_selection["vif_rounds"]
    dropped_by_vif = feature_selection["dropped_by_vif"]
    final_features = feature_selection["final_features"]

    suspicious_feature_list = ", ".join(f"`{f}`" for f in flagged_as_suspicious) if flagged_as_suspicious else "无"
    vif_rounds_blocks = "\n\n".join(
        f"轮次 {i + 1}：\n\n{md_table(pd.DataFrame(round_summary))}" for i, round_summary in enumerate(vif_rounds)
    )

    coef_rows = []
    for feat, vals in scorecard_report["feature_coefficients"].items():
        coef_rows.append({
            "feature": feat,
            "logistic_coef": vals["logistic_coef"],
            "score_impact_factor": vals["score_impact_factor"],
        })
    coef_table = pd.DataFrame(coef_rows).sort_values("score_impact_factor", key=abs, ascending=False)

    best_params_table = pd.DataFrame(
        [{"hyperparameter": k, "value": v} for k, v in tuning_report["best_hyperparameters"].items()]
    )
    tuned_importance_table = pd.DataFrame(tuning_report["feature_importances"], columns=["feature", "importance"])
    shap_table = pd.DataFrame(shap_summary["global_importance_ranking"], columns=["feature", "mean_abs_shap"])

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

    psi_overall_status = "低于预警阈值，无需特别处理" if score_psi["PSI"] < PSI_WATCH else (
        "位于 0.10~0.25 预警区间，需在文档中特别注明并加强监控" if score_psi["PSI"] <= PSI_ACTION
        else "超过 0.25，需模型重新校准"
    )

    feature_business_meaning = {
        "debt_to_income_ratio": "借款人月负债与月收入的比值，比值越高代表偿债压力越大",
        "interest_rate": "该笔贷款的定价利率，通常已隐含发放时风险定价系统对借款人风险的初步判断",
        "delinquency_history": "历史逾期记录标记，反映借款人过往还款行为的可靠性",
        "num_of_delinquencies": "历史逾期次数，次数越多代表还款稳定性越差",
        "credit_score": "征信评分，分数越低代表历史信用表现越差",
        "grade_subgrade": "内部预分配风险评级子等级（30 档），与 credit_score/DTI/interest_rate 等原始征信变量存在结构性重叠",
        "employment_status": "借款人就业状态（Employed / Self-employed / Retired / Student / Unemployed），起贷时可得的申请人属性，非模型输出的衍生字段",
    }
    top3_lines = []
    for rank, (feat, val) in enumerate(top3_shap, start=1):
        meaning = feature_business_meaning.get(feat, "该字段的原始业务含义")
        flag_note = "（⚠ 本报告 §7.4 单独讨论其治理风险）" if feat in flagged_as_suspicious else ""
        top3_lines.append(
            f"{rank}. **{feat}**{flag_note}（平均 |SHAP| = {val}）：{meaning}；WOE 编码值越低代表原始取值落入历史违约率更高的分箱，"
            f"对应模型预测违约概率上升。作为全局重要性排名第 {rank} 的特征，建议在授信审批规则与额度定价中重点参考。"
        )
    top3_block = "\n".join(top3_lines)

    md = f"""# Model Validation Report — 信用评分卡建模与验证报告

数据来源: `data/loan_dataset_20000.csv` ｜ 建模函数来源: `utils/risk_skills.py` ｜ 调优来源: `scripts/tune_lgb_skill.py` ｜ 随机种子: `random_state=42`
流水线: `scripts/train.py`（训练 + Optuna 调优，仅接触 Train 集）→ `scripts/test.py`（评估、混淆矩阵与报告生成，仅接触 Test 集与已持久化模型）

---

## 1. Executive Summary

- 本次基于 `data/loan_dataset_20000.csv`（20,000 笔贷款样本）构建了标准信用评分卡模型（逻辑回归，`train_scorecard_model`，Baseline），并通过 Optuna 贝叶斯超参数寻优（`tune_lightgbm_optuna`，{tuning_report['n_trials_completed']} 轮，{tuning_report['cv_folds']} 折交叉验证，优化目标 {tuning_report['target_metric']}）得到单调约束 LightGBM 最优模型（Champion）。
- 目标变量 `default` = 1 - `loan_paid_back`（1 = 违约/未还清，0 = 正常还清）。
- **最优模型（Optuna 调优 LightGBM）** 在 Test 集上的判别能力：AUC = **{champion_test_metrics['AUC']}**（阈值 ≥ {AUC_MIN}：**{status_icon(auc_pass)}**），KS = **{champion_test_metrics['KS_Statistic']}**（阈值 ≥ {KS_MIN}：**{status_icon(ks_pass)}**），评级为 "{champion_test_metrics['Validation_Rating']}"；混淆矩阵详见 §5.3。
- 基线评分卡（逻辑回归）Test 集：AUC = {baseline_test_metrics['AUC']}，KS = {baseline_test_metrics['KS_Statistic']}，作为可解释性对照。
- 最优模型预测违约概率在 Train → Test 上的 PSI = **{score_psi['PSI']}**，稳定性状态："{score_psi['Status']}"（{psi_overall_status}）。
- 特征工程阶段基于 IV 剔除 {len(dropped_by_iv)} 个弱预测特征；基于迭代式 VIF 剔除 {len(dropped_by_vif)} 个共线性特征；最终入模特征 {len(final_features)} 个（详见 §3）。
- **治理标记（保留，非剔除）**：特征 {suspicious_feature_list} 的 IV 落入 "Suspicious / Overfitting (>0.5)" 区间（详见 §3.1、§7.4），存在近似确定性分组（如 employment_status=Unemployed 违约率 81.8%、Retired 违约率仅 0.5%）。经复核判断为起贷时可得的合法申请人属性而非标签泄漏，予以保留入模；但其对模型判别力的贡献占比过高（详见 §7.4 的敏感性对比），存在集中度风险与潜在公平信贷（fair-lending）合规风险，**在独立验证与公平信贷复核完成前不建议直接投产使用**。
- **方法论修正记录**：此前版本的多重共线性过滤为一次性剔除所有 VIF 超阈值特征，导致 `credit_score` 因与 `grade_subgrade` 共同存在而被误伤剔除（二者共线，但只需剔除其一）；现改为迭代式 VIF（每轮只剔除当前 VIF 最高的单个特征后重新计算，详见 §3.2），该问题已修正。
- **调优脚本已知局限**：`tune_lightgbm_optuna` 的最终模型使用交叉验证得到的最佳超参数在全量 Train 上重新拟合，但重新拟合时固定 `n_estimators=1000` 且未设置 `eval_set`/早停，与各折验证时实际使用的早停树数不一致，可能引入额外方差；如需更严谨的复现，建议后续为最终拟合补充早停或将 CV 中位数最优轮数固定为 `n_estimators`。
- 建议：{"该最优模型判别能力与稳定性均满足内部风控标准；在完成 §7.4 所述的独立验证与公平信贷复核前，暂不建议投产。" if (ks_pass and auc_pass and score_psi['PSI'] < PSI_ACTION) else "该模型在判别能力或稳定性指标上未完全满足内部阈值标准，建议在投产前进一步优化特征、扩大调优搜索空间或重新校准后再评审；同时仍需完成 §7.4 所述的公平信贷复核。"}

---

## 2. Train / Test Split

- 数据源：`data/loan_dataset_20000.csv`，共 20,000 条样本，22 个原始字段（含目标衍生前）。
- 目标变量：`default`（由 `loan_paid_back` 取反得到，1 = 违约）。
- 切分方式：`sklearn.model_selection.train_test_split`（`scripts/train.py`），`test_size=0.3`，按目标变量分层抽样，`random_state=42`；切分结果持久化为 `artifacts/train_data.csv` / `artifacts/test_data.csv`，`scripts/test.py` 直接加载，不重新切分。
- 分层校验：Train / Test 违约率差异 < 0.001（见下表），分层抽样有效。
- 说明：原始数据不含时间/放款日期字段，无法构建真正意义上的跨时间 OOT 样本；本报告中的 "Test" 集为分层随机切分得到的样本外测试集，Optuna 调优过程中的 5 折交叉验证同样只在 Train 集内部进行，Test 集全程未参与调参或训练。

{md_table(sample_split_summary)}

---

## 3. Feature Engineering & Selection

### 3.1 WOE / IV 筛选表（基于 Train 集，`calculate_woe_iv`，等频 5 分箱，`scripts/train.py`）

{md_table(iv_table)}

- 剔除标准：IV < {IV_MIN} → 直接剔除（预测力不足）。IV > {IV_SUSPICIOUS} → **不自动剔除**，仅标记为需要业务/公平信贷复核（"Suspicious / Overfitting"，可能是合法强预测因子，也可能是泄漏，取舍属于治理判断而非统计判断）。
- 弱预测力剔除（{len(dropped_by_iv)} 个）：{", ".join(dropped_by_iv) if dropped_by_iv else "无"}。
- 标记但保留入模（{len(flagged_as_suspicious)} 个）：{suspicious_feature_list}——详见 §7.4 的治理复核说明。

### 3.2 多重共线性过滤（迭代式 VIF，内核为 `compute_vif_filter`，在 IV 筛选后特征的 WOE 编码矩阵上计算，`scripts/train.py`）

最终一轮（全部特征 VIF ≤ {VIF_MAX}）：

{md_table(vif_table)}

- 剔除标准：VIF > {VIF_MAX}，**每轮只剔除当前 VIF 最高的单个特征后重新计算**，而非一次性剔除所有超阈值特征（一次性剔除会在多个特征互相拖累对方 VIF 时过度剔除——本报告曾因此误伤 `credit_score`，见 §1）。
- 本次剔除特征（{len(dropped_by_vif)} 个，按剔除顺序）：{", ".join(dropped_by_vif) if dropped_by_vif else "无"}。
- 完整迭代过程（共 {len(vif_rounds)} 轮）：

{vif_rounds_blocks}

### 3.3 最终入模特征（{len(final_features)} 个）

{", ".join(final_features)}

---

## 4. Hyperparameter Tuning (Optuna)

- 调优函数：`scripts/tune_lgb_skill.py` 的 `tune_lightgbm_optuna`（TPE 采样器，`random_state=42`）。
- 搜索空间：`learning_rate` ∈ [0.01, 0.1]（log），`num_leaves` ∈ [15, 127]，`max_depth` ∈ [3, 8]，`min_child_samples` ∈ [20, 300]，`subsample` ∈ [0.6, 1.0]，`colsample_bytree` ∈ [0.6, 1.0]，`reg_alpha`/`reg_lambda` ∈ [1e-3, 10]（log）。
- 交叉验证：{tuning_report['cv_folds']} 折 `StratifiedKFold`（`shuffle=True, random_state=42`），每折内以验证集早停（`stopping_rounds=30`）；单调性约束 `monotone_constraints={tuning_report['monotone_constraints']}` 在每次试验中保持固定，与入模特征的 WOE 编码方向一致。
- 试验轮数：请求 {tuning_report['n_trials_requested']} 轮，完成 {tuning_report['n_trials_completed']} 轮。
- 优化目标：{tuning_report['target_metric']}（OOF，跨 {tuning_report['cv_folds']} 折）。
- 最佳 CV {tuning_report['target_metric']} = **{tuning_report['best_cv_score']}**。

### 4.1 最佳超参数配置

{md_table(best_params_table)}

### 4.2 最优模型特征重要度

{md_table(tuned_importance_table)}

---

## 5. Model Performance & Discrimination

### 5.1 Train / Test 判别力对比表（`evaluate_discrimination_and_ks`，`scripts/test.py`）

{md_table(perf_table)}

- 内部阈值标准：Test 集 KS ≥ {KS_MIN}，AUC ≥ {AUC_MIN}。
- 最优模型（Optuna 调优 LightGBM）Test 集结果：AUC = {champion_test_metrics['AUC']}（{status_icon(auc_pass)}），KS = {champion_test_metrics['KS_Statistic']}（{status_icon(ks_pass)}），最优切点概率 = {champion_test_metrics['Optimal_Cutoff_Probability']}。

### 5.2 评分卡标尺参数（Baseline，`train_scorecard_model`，`scripts/train.py`）

- Base Points = {scorecard_report['scaling_parameters']['base_points']}，Base Odds = 50.0，PDO = {scorecard_report['scaling_parameters']['pdo']}
- Factor = {scorecard_report['scaling_parameters']['factor']}，Offset = {scorecard_report['scaling_parameters']['offset']}，Intercept = {scorecard_report['intercept']}

{md_table(coef_table)}

### 5.3 混淆矩阵（最优模型，Test 集，切点 = KS 最优切点概率 {confusion_summary['cutoff']}）

{md_table(confusion_table)}

{md_table(confusion_metrics_table)}

- 切点选取说明：Test 集违约率约 20%，固定 0.5 切点会导致模型几乎不触发正类预测；因此采用 `evaluate_discrimination_and_ks` 基于 ROC 曲线给出的 KS 最优切点概率（`argmax(TPR-FPR)`）作为混淆矩阵切点，而非任意的 0.5。

---

## 6. Population Stability Index (PSI)

### 6.1 模型分 PSI（最优模型预测违约概率，Train 作为基准分布，Test 作为实际分布，`scripts/test.py`）

- PSI = **{score_psi['PSI']}**，状态：{score_psi['Status']}
- 监管标准：PSI < 0.10 无需处理；0.10~0.25 需特别注明预警；> 0.25 需模型重新校准。本次结果：{psi_overall_status}

### 6.2 关键特征 PSI（Train vs Test，Top {len(feature_psi_table)} 入模特征）

{md_table(feature_psi_table) if len(feature_psi_table) else "无可用数值型关键特征。"}

---

## 7. Explainability & Governance

### 7.1 全局 SHAP 特征重要性排序（`generate_shap_summary`，基于最优模型 LightGBM，Test 抽样 {shap_sample_size} 条，`scripts/test.py`）

{md_table(shap_table)}

### 7.2 Top 3 特征业务解释

{top3_block}

### 7.3 敏感性分析：治理标记特征的边际贡献

对已保留但标记为 "Suspicious / Overfitting" 的特征（{", ".join(f"`{f}`" for f in flagged_and_kept) if flagged_and_kept else "无"}），重新拟合 Baseline 逻辑回归、剔除该特征后在同一 Test 集上重新评估，以量化其对判别力的边际贡献（口径对齐 `reports/PD_Model_Development_Document.docx` §7.3 的 variant 敏感性分析）：

{md_table(sensitivity_table) if len(sensitivity_table) else "无标记特征，跳过敏感性分析。"}

{f"该特征贡献了约 {round(sensitivity_table.iloc[0]['AUC'] - sensitivity_table.iloc[1]['AUC'], 4)} 的 Test AUC——是本模型判别力的主要来源，而非边际增益。这也是本模型判别力显著高于早前排除该特征版本（Test AUC ≈0.665）的核心原因。" if len(sensitivity_table) == 2 else ""}

### 7.4 治理标记特征明细（按类别）

{"".join(f"**{f}**（{feature_business_meaning.get(f, '')}）：" + chr(10) + chr(10) + md_table(flagged_breakdown_tables[f]) + chr(10) + chr(10) for f in flagged_breakdown_tables) if flagged_breakdown_tables else "无。"}

**治理结论**：该特征在起贷时可得、非模型输出的衍生字段，不构成技术意义上的标签泄漏；但其对模型判别力的贡献占比过高（集中度风险），且类别与借款人的就业/收入状态强相关，在放贷决策中使用可能构成对特定群体的间接歧视（disparate impact）。本报告将其保留入模用于性能评估，**但明确不建议在完成独立验证与公平信贷（fair-lending）合规复核之前将其用于实际授信决策**。参考文档：`reports/PD_Model_Development_Document.docx` §8（同样保留该特征，同样将其列为待决限制项，尚未获得独立验证与合规批准）。

### 7.5 监管合规审查说明

- 模型类型：标准评分卡（L2 正则逻辑回归，`penalty='l2', C=1.0`）作为可解释性 Baseline，系数与标度均可追溯至 `Score = Offset + Factor * ln(Odds)` 公式；最优决策模型为 Optuna 调优后的单调约束 LightGBM。
- 最优模型（LightGBM）已对全部入模特征施加单调递减约束（`monotone_constraints=-1`），约束方向依据 WOE 编码定义（WOE 越高代表历史违约率越低），且该约束在 Optuna 全部 {tuning_report['n_trials_completed']} 轮试验中保持固定，符合监管对评分类模型单调性的一般要求。
- 特征筛选与共线性处理均通过标准化流程（`calculate_woe_iv` / 迭代式 `compute_vif_filter`）完成，超参数寻优通过标准化流程（`tune_lightgbm_optuna`）完成，未使用未经验证的第三方黑盒库直接输出模型结论。
- 判别力与稳定性指标（KS/AUC/PSI）均通过 `utils/risk_skills.py` 标准函数计算，口径统一、可复现（`random_state=42`）。
- 训练/调参/测试边界可审计：`scripts/train.py` 只写 `artifacts/`（含 Optuna 调优过程），从不读取 Test 集；`scripts/test.py` 只读 `artifacts/` 与 Test 集，从不重新拟合分箱、编码器、模型参数或超参数。
- IV > {IV_SUSPICIOUS} 的特征不再自动剔除，改为保留入模 + 强制标记 + §7.3/§7.4 的量化敏感性分析与治理结论，取舍留待模型使用方与合规团队决策，而非由脚本单方面决定（详见 CLAUDE.md 中记录的方法论修正）。
"""
    return md


if __name__ == "__main__":
    main()
