"""
训练阶段 (Stage 1/2)。

只接触 Train 集，不接触 Test 集 —— 数据切分、IV 筛选、WOE 编码拟合、VIF 过滤、
模型训练全部基于 Train，避免信息泄漏。产出的模型、编码器与筛选结果全部写入
artifacts/，供 scripts/test.py 独立加载评估，便于审计"训练用了什么数据、
产出了什么模型"这条边界。

复用 utils/risk_skills.py 中的标准函数（calculate_woe_iv / compute_vif_filter /
train_scorecard_model）与 scripts/tune_lgb_skill.py 中的 tune_lightgbm_optuna，
符合 CLAUDE.md 中的《银行风控模型开发与验证规范》。

用法: python3 scripts/train.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 允许同目录导入 tune_lgb_skill

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tune_lgb_skill import tune_lightgbm_optuna
from utils.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    IV_MIN,
    IV_SUSPICIOUS,
    RANDOM_STATE,
    RAW_TARGET_COL,
    TARGET,
    TEST_SIZE,
    TUNE_CV_FOLDS,
    TUNE_METRIC,
    TUNE_N_TRIALS,
    VIF_MAX,
    WOE_BINS,
)
from utils.risk_skills import calculate_woe_iv, compute_vif_filter, train_scorecard_model
from utils.woe_encoding import apply_woe_encoder, fit_woe_encoder


def load_and_split():
    raw = pd.read_csv(DATA_PATH)
    raw[TARGET] = 1 - raw[RAW_TARGET_COL]
    raw = raw.drop(columns=[RAW_TARGET_COL])
    feature_cols = [c for c in raw.columns if c != TARGET]

    train_df, test_df = train_test_split(
        raw, test_size=TEST_SIZE, stratify=raw[TARGET], random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), feature_cols


def screen_features(train_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """基于 Train 集调用 calculate_woe_iv 计算全部候选特征的 IV。"""
    records = []
    for f in feature_cols:
        res = calculate_woe_iv(train_df, f, TARGET, bins=WOE_BINS)
        records.append({"feature": res["feature"], "iv": res["iv"], "predictive_power": res["predictive_power"]})
    return pd.DataFrame(records).sort_values("iv", ascending=False).reset_index(drop=True)


def iterative_vif_filter(X: pd.DataFrame, features: list, threshold: float):
    """
    迭代式 VIF 过滤：每轮只剔除当前 VIF 最高的单个特征，然后在剩余特征上重新
    计算 VIF，直至全部特征的 VIF 都不超过阈值。

    compute_vif_filter 本身只做单轮计算，一次性返回所有超过阈值的特征作为
    "recommended_drops"；如果直接把该列表整体剔除，会在多个特征互相拖累对方
    VIF 的情况下过度剔除——例如 A、B 两个特征只要同时在场就会互相推高 VIF，
    单独剔除任意一个后另一个的 VIF 通常会回落到阈值以下。标准做法是每轮只删
    "当前最严重的那一个"、重新计算，逐步收敛，而不是一次性删光所有超阈值特征。
    """
    remaining = list(features)
    dropped = []
    rounds = []
    while len(remaining) > 1:
        result = compute_vif_filter(X[remaining], remaining, threshold=threshold)
        rounds.append(result["vif_summary"])
        offenders = [r for r in result["vif_summary"] if r["vif"] > threshold]
        if not offenders:
            break
        worst = max(offenders, key=lambda r: r["vif"])
        dropped.append(worst["feature"])
        remaining.remove(worst["feature"])
    return remaining, dropped, rounds


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 数据切分：仅在此处切分一次，随后 Train/Test 各自持久化，
    #    scripts/test.py 直接读取持久化结果，不重新切分。
    # ------------------------------------------------------------------
    train_df, test_df, feature_cols = load_and_split()
    train_df.to_csv(os.path.join(ARTIFACTS_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(ARTIFACTS_DIR, "test_data.csv"), index=False)

    # ------------------------------------------------------------------
    # 2. 特征筛选 (calculate_woe_iv)，仅使用 Train
    #
    #    IV < IV_MIN：预测力不足，直接剔除。
    #    IV > IV_SUSPICIOUS："Suspiciously strong"——只是一个需要人工复核的标记，
    #    不再自动剔除。这类特征（本数据集中为 employment_status）可能是合法的
    #    强预测因子，也可能是泄漏；剔除与否是治理/合规判断，不应由脚本单方面
    #    静默决定。保留入模，但在 feature_selection.json 与最终报告中显式标记，
    #    要求业务/公平信贷复核（见 CLAUDE.md 及 reports/PD_Model_Development_
    #    Document.docx §5.1、§8 对该问题的处理方式）。
    # ------------------------------------------------------------------
    iv_table = screen_features(train_df, feature_cols)
    dropped_by_iv = iv_table.loc[iv_table["iv"] < IV_MIN, "feature"].tolist()
    flagged_as_suspicious = iv_table.loc[iv_table["iv"] > IV_SUSPICIOUS, "feature"].tolist()
    kept_after_iv = [f for f in feature_cols if f not in dropped_by_iv]

    # ------------------------------------------------------------------
    # 3. WOE 编码：在 Train 上拟合分箱与 WOE 映射
    # ------------------------------------------------------------------
    encoders = {f: fit_woe_encoder(train_df, f, TARGET, bins=WOE_BINS) for f in kept_after_iv}
    X_train_woe = pd.DataFrame({f: apply_woe_encoder(train_df, f, encoders[f]) for f in kept_after_iv})
    y_train = train_df[TARGET]

    # ------------------------------------------------------------------
    # 4. 多重共线性过滤：迭代式 VIF (iterative_vif_filter 包装 compute_vif_filter)，
    #    仅使用 Train 的 WOE 编码矩阵
    # ------------------------------------------------------------------
    final_features, dropped_by_vif, vif_rounds = iterative_vif_filter(X_train_woe, kept_after_iv, VIF_MAX)

    X_train = X_train_woe[final_features]

    # ------------------------------------------------------------------
    # 5a. 逻辑回归评分卡 (标准产出模型)
    #
    #     train_scorecard_model 只返回摘要 dict (系数/标度参数)，不返回已训练
    #     模型对象；因此这里额外用相同超参数在本地拟合一份可持久化、可在
    #     test.py 中预测概率的模型实例。
    # ------------------------------------------------------------------
    scorecard_report = train_scorecard_model(X_train, y_train, base_points=600, base_odds=50.0, pdo=20)
    lr_model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5b. 单调约束 LightGBM：Optuna 贝叶斯超参数寻优 (scripts/tune_lgb_skill.py)
    #
    #     WOE 编码后特征的方向已与"风险"对齐 (WOE 越高代表该分箱历史违约率越低)，
    #     因此对全部入模特征施加单调递减约束，符合监管对评分卡单调性的要求。
    #     CV 目标函数固定为 AUC (TUNE_METRIC)；KS 作为 Train/Test 事后监控指标，
    #     由 scripts/test.py 通过 evaluate_discrimination_and_ks 单独计算并报告。
    # ------------------------------------------------------------------
    monotone_constraints = [-1] * len(final_features)
    tuning_result = tune_lightgbm_optuna(
        X_train,
        y_train,
        n_trials=TUNE_N_TRIALS,
        cv_folds=TUNE_CV_FOLDS,
        metric=TUNE_METRIC,
        monotone_constraints=monotone_constraints,
        random_state=RANDOM_STATE,
    )
    lgbm_model = tuning_result["trained_model"]

    # ------------------------------------------------------------------
    # 6. 持久化全部产出，供 scripts/test.py 加载评估与生成报告
    # ------------------------------------------------------------------
    joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, "woe_encoders.joblib"))
    joblib.dump(lr_model, os.path.join(ARTIFACTS_DIR, "lr_model.joblib"))
    joblib.dump(lgbm_model, os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib"))

    feature_selection = {
        "feature_cols": feature_cols,
        "iv_table": iv_table.to_dict(orient="records"),
        "dropped_by_iv": dropped_by_iv,
        "flagged_as_suspicious": flagged_as_suspicious,
        "kept_after_iv": kept_after_iv,
        "vif_rounds": vif_rounds,
        "vif_summary": vif_rounds[-1] if vif_rounds else [],
        "dropped_by_vif": dropped_by_vif,
        "final_features": final_features,
    }
    with open(os.path.join(ARTIFACTS_DIR, "feature_selection.json"), "w", encoding="utf-8") as fh:
        json.dump(feature_selection, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "scorecard_report.json"), "w", encoding="utf-8") as fh:
        json.dump(scorecard_report, fh, ensure_ascii=False, indent=2)

    tuning_report = {
        "target_metric": tuning_result["target_metric"],
        "cv_folds": TUNE_CV_FOLDS,
        "n_trials_requested": TUNE_N_TRIALS,
        "n_trials_completed": tuning_result["n_trials_completed"],
        "best_cv_score": tuning_result["best_cv_score"],
        "best_hyperparameters": tuning_result["best_hyperparameters"],
        "feature_importances": tuning_result["feature_importances"],
        "monotone_constraints": monotone_constraints,
    }
    with open(os.path.join(ARTIFACTS_DIR, "tuning_report.json"), "w", encoding="utf-8") as fh:
        json.dump(tuning_report, fh, ensure_ascii=False, indent=2)

    print(f"[train] Train n={len(train_df)}, bad_rate={train_df[TARGET].mean():.4f}")
    print(f"[train] Test  n={len(test_df)}, bad_rate={test_df[TARGET].mean():.4f} (held out, untouched)")
    print(f"[train] IV screening: kept {len(kept_after_iv)}/{len(feature_cols)} "
          f"(dropped {len(dropped_by_iv)} weak; {len(flagged_as_suspicious)} flagged as suspicious but KEPT "
          f"pending governance review: {flagged_as_suspicious})")
    print(f"[train] Iterative VIF filter ({len(vif_rounds)} rounds): dropped {dropped_by_vif}")
    print(f"[train] Final features ({len(final_features)}): {final_features}")
    print(f"[train] Optuna tuning: {tuning_result['n_trials_completed']} trials, "
          f"{TUNE_CV_FOLDS}-fold CV, best {tuning_result['target_metric']}={tuning_result['best_cv_score']}")
    print(f"[train] Best hyperparameters: {tuning_result['best_hyperparameters']}")
    print(f"[train] Artifacts written to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
