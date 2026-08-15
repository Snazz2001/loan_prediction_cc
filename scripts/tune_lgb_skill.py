import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

# 关闭 Optuna 默认的冗长日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算 KS 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def tune_lightgbm_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    metric: str = "auc",  # 支持 "auc" 或 "ks"
    monotone_constraints: Optional[List[int]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    使用 Optuna 对 LightGBM 进行贝叶斯超参数寻优（针对二分类/风控场景）。
    
    Args:
        X_train: 训练特征集
        y_train: 目标二分类标签 (0/1)
        n_trials: 寻优轮数
        cv_folds: 分层交叉验证折数
        metric: 优化目标 ('auc' 或 'ks')
        monotone_constraints: 单调性约束列表 (-1, 0, 1)
        random_state: 随机种子
        
    Returns:
        包含最佳参数、最佳 CV 得分、训练好的最终模型及历史指标的字典
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": random_state,
            "n_estimators": 1000,  # 配合 early_stopping 使用
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "monotone_constraints": monotone_constraints
        }

        oof_preds = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
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

    best_params = study.best_params
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": random_state,
        "n_estimators": 1000,
        "monotone_constraints": monotone_constraints
    })

    # 用全量训练集及最佳参数重新拟合最终模型
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    feature_importances = dict(zip(X_train.columns, [int(x) for x in final_model.feature_importances_]))
    sorted_importance = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    return {
        "best_cv_score": round(float(study.best_value), 4),
        "target_metric": metric.upper(),
        "best_hyperparameters": study.best_params,
        "n_trials_completed": len(study.trials),
        "feature_importances": sorted_importance,
        "trained_model": final_model
    }