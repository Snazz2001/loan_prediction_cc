"""
Optuna 贝叶斯超参数寻优 —— XGBoost 版本，结构与参数含义对齐 scripts/tune_lgb_skill.py
的 tune_lightgbm_optuna，便于与 LightGBM 结果直接横向比较。

已知局限（与 tune_lightgbm_optuna 完全一致，为保持"同样设置"刻意保留）：
最终模型使用交叉验证得到的最佳超参数在全量 Train 上重新拟合，但重新拟合时固定
n_estimators=1000 且未设置 eval_set/早停，与各折验证时实际使用的早停树数不一致。
"""

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from typing import Any, Dict, List, Optional
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算 KS 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def tune_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    metric: str = "auc",  # 支持 "auc" 或 "ks"
    monotone_constraints: Optional[List[int]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    使用 Optuna 对 XGBoost 进行贝叶斯超参数寻优（针对二分类/风控场景）。
    参数含义与 tune_lightgbm_optuna 一致，便于直接对照。
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    mc_tuple = tuple(monotone_constraints) if monotone_constraints is not None else None

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            # 全部特征均为 WOE 编码后的浮点数，非原生类别特征；shap 的 TreeExplainer
            # 不兼容 xgboost>=2.x 默认开启的 enable_categorical=True，显式关闭。
            "enable_categorical": False,
            "random_state": random_state,
            "n_estimators": 1000,  # 配合 early_stopping 使用
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "monotone_constraints": mc_tuple,
        }

        oof_preds = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]

        if metric == "ks":
            score = _calculate_ks(y_train.values, oof_preds)
        else:
            score = roc_auc_score(y_train, oof_preds)

        return score

    # 创建 Optuna Study 并执行寻优
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "enable_categorical": False,
        "random_state": random_state,
        "n_estimators": 1000,
        "monotone_constraints": mc_tuple,
    })

    # 用全量训练集及最佳参数重新拟合最终模型
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    feature_importances = dict(zip(X_train.columns, [float(x) for x in final_model.feature_importances_]))
    sorted_importance = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    return {
        "best_cv_score": round(float(study.best_value), 4),
        "target_metric": metric.upper(),
        "best_hyperparameters": study.best_params,
        "n_trials_completed": len(study.trials),
        "feature_importances": sorted_importance,
        "trained_model": final_model,
    }
