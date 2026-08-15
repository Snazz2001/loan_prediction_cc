import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import shap

# ----------------------------------------------------
# 1. 数据审查与特征工程 Skill
# ----------------------------------------------------

def calculate_woe_iv(
    data: pd.DataFrame, 
    feature: str, 
    target: str, 
    bins: int = 5
) -> Dict[str, Any]:
    """
    计算特征的 WOE (Weight of Evidence) 与 IV (Information Value)。
    
    Args:
        data: 包含特征和目标列的 DataFrame
        feature: 待分箱评估的特征名
        target: 二分类标签列名 (0/1)
        bins: 连续变量分箱数
        
    Returns:
        包含 IV 值、各箱分值及 WOE 映射字典的结构化数据
    """
    df = data[[feature, target]].dropna().copy()
    
    # 连续变量等频分箱，离散变量直接分组
    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > bins:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop').astype(str)
    else:
        df['bin'] = df[feature].astype(str)
        
    grouped = df.groupby('bin').agg(
        total_count=(target, 'count'),
        bad_count=(target, 'sum')
    ).reset_index()
    
    grouped['good_count'] = grouped['total_count'] - grouped['bad_count']
    
    total_bad = grouped['bad_count'].sum()
    total_good = grouped['good_count'].sum()
    
    # 平滑处理防止除零
    grouped['bad_dist'] = np.where(grouped['bad_count'] == 0, 0.5, grouped['bad_count']) / total_bad
    grouped['good_dist'] = np.where(grouped['good_count'] == 0, 0.5, grouped['good_count']) / total_good
    
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['iv_component'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    
    total_iv = grouped['iv_component'].sum()
    
    # 评判 IV 预测能力
    if total_iv < 0.02:
        predictive_power = "Unpredictive (<0.02)"
    elif total_iv < 0.1:
        predictive_power = "Weak (0.02 - 0.1)"
    elif total_iv < 0.3:
        predictive_power = "Medium (0.1 - 0.3)"
    elif total_iv < 0.5:
        predictive_power = "Strong (0.3 - 0.5)"
    else:
        predictive_power = "Suspicious / Overfitting (>0.5)"

    return {
        "feature": feature,
        "iv": round(float(total_iv), 4),
        "predictive_power": predictive_power,
        "bin_details": grouped[['bin', 'total_count', 'bad_count', 'woe', 'iv_component']].to_dict(orient='records')
    }


def compute_vif_filter(
    data: pd.DataFrame, 
    features: List[str], 
    threshold: float = 5.0
) -> Dict[str, Any]:
    """
    计算特征集的方差膨胀因子 (VIF)，识别并推荐剔除多重共线性特征。
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X = data[features].dropna()
    vif_data = []
    
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": features[i], "vif": round(float(vif), 2)})
        
    dropped_candidates = [item["feature"] for item in vif_data if item["vif"] > threshold]
    
    return {
        "vif_summary": sorted(vif_data, key=lambda x: x['vif'], reverse=True),
        "recommended_drops": dropped_candidates,
        "status": "PASS" if not dropped_candidates else "WARNING_HIGH_COLLINEARITY"
    }

# ----------------------------------------------------
# 2. 核心算法与评分卡构建 Skill
# ----------------------------------------------------

def train_scorecard_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    base_points: int = 600, 
    base_odds: float = 50.0, 
    pdo: int = 20
) -> Dict[str, Any]:
    """
    训练用于标准风控评分卡的带惩罚项逻辑回归模型，并转换对数几率到标尺分值。
    Scale: Score = Offset + Factor * ln(Odds)
    """
    # 计算标尺因子
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    
    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    coefficients = {}
    for col, coef in zip(X_train.columns, model.coef_[0]):
        coefficients[col] = {
            "logistic_coef": round(float(coef), 4),
            "score_impact_factor": round(float(-coef * factor), 2)
        }
        
    return {
        "model_type": "Standard Risk Scorecard (Logistic Regression)",
        "intercept": round(float(model.intercept_[0]), 4),
        "scaling_parameters": {
            "factor": round(float(factor), 4),
            "offset": round(float(offset), 4),
            "base_points": base_points,
            "pdo": pdo
        },
        "feature_coefficients": coefficients
    }


def train_constrained_lightgbm(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    monotone_constraints: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    训练满足监管单调性约束 (Monotonic Constraints) 的 LightGBM 模型。
    -1: 单调递减, 0: 无约束, 1: 单调递增
    """
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        monotone_constraints=monotone_constraints,
        random_state=42,
        verbosity=-1
    )
    clf.fit(X_train, y_train)
    
    importance = dict(zip(X_train.columns, [float(x) for x in clf.feature_importances_]))
    
    return {
        "model_type": "Monotonic Constrained LightGBM",
        "n_features": len(X_train.columns),
        "feature_importances": sorted(importance.items(), key=lambda x: x[1], reverse=True)
    }

# ----------------------------------------------------
# 3. 模型验证与 XAI (可解释性) Skill
# ----------------------------------------------------

def evaluate_discrimination_and_ks(
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    计算银行模型验证必需的判别能力指标：AUC-ROC、KS 统计量及 Gini 系数。
    """
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    max_ks_idx = np.argmax(ks_values)
    ks_stat = ks_values[max_ks_idx]
    optimal_cutoff = thresholds[max_ks_idx]
    
    # 评判标准
    if ks_stat < 0.2:
        ks_rating = "Poor (<0.20)"
    elif ks_stat < 0.4:
        ks_rating = "Acceptable (0.20 - 0.40)"
    elif ks_stat < 0.6:
        ks_rating = "Good (0.40 - 0.60)"
    else:
        ks_rating = "Suspicious / Extreme (>0.60)"
        
    return {
        "AUC": round(float(auc), 4),
        "Gini": round(float(gini), 4),
        "KS_Statistic": round(float(ks_stat), 4),
        "Optimal_Cutoff_Probability": round(float(optimal_cutoff), 4),
        "Validation_Rating": ks_rating
    }


def calculate_psi(
    expected: np.ndarray, 
    actual: np.ndarray, 
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    计算群体稳定性指标 (PSI)，用于检验开发样本与 OOT (跨时间样本) / 生产样本的漂移程度。
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # 基于基准样本确定分箱分位数
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # 平滑微量
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_vector = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = np.sum(psi_vector)
    
    # 监管标准评估
    if total_psi < 0.1:
        stability = "Stable (<0.10) - No action needed"
    elif total_psi <= 0.25:
        stability = "Slight Shift (0.10 - 0.25) - Close monitoring advised"
    else:
        stability = "Significant Shift (>0.25) - Model recalibration required"
        
    return {
        "PSI": round(float(total_psi), 4),
        "Status": stability
    }


def generate_shap_summary(
    model: Any, 
    X_sample: pd.DataFrame
) -> Dict[str, Any]:
    """
    通过 Tree/Linear SHAP 生成模型全局可解释性指标与各特征的平均绝对贡献。
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    
    # 计算每个特征的平均 SHAP 贡献绝对值
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    if mean_abs_shap.ndim > 1:  # 处理多分类/多维输出
        mean_abs_shap = mean_abs_shap[:, 1]
        
    global_importance = {
        col: round(float(val), 4) 
        for col, val in zip(X_sample.columns, mean_abs_shap)
    }
    
    return {
        "xai_method": "SHAP (Kernel/Tree)",
        "global_importance_ranking": sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
    }