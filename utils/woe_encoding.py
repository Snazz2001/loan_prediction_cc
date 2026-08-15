"""
WOE 编码的 fit/transform 封装。

utils/risk_skills.py 中的 calculate_woe_iv 只对单个传入数据集计算分箱与 IV，
不产出可复用于新样本 (Test) 的分箱边界，因此这里在其计算逻辑之上封装一层
fit(train)/transform(test) 的绑定关系：分箱边界与各分箱 WOE 值只在 Train 上拟合，
测试阶段 (scripts/test.py) 直接复用训练阶段拟合好的映射，不重新计算，避免信息泄漏。
"""

import numpy as np
import pandas as pd


def fit_woe_encoder(train_df: pd.DataFrame, feature: str, target: str, bins: int) -> dict:
    s = train_df[feature]
    is_numeric = pd.api.types.is_numeric_dtype(s) and s.nunique() > bins

    if is_numeric:
        _, edges = pd.qcut(s, q=bins, duplicates="drop", retbins=True)
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_series = pd.cut(s, bins=edges, include_lowest=True)
    else:
        edges = None
        bin_series = s.astype(str)

    tmp = pd.DataFrame({"bin": bin_series, target: train_df[target].values})
    grouped = tmp.groupby("bin", observed=True).agg(
        total=(target, "count"), bad=(target, "sum")
    ).reset_index()
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    grouped["bad_dist"] = np.where(grouped["bad"] == 0, 0.5, grouped["bad"]) / total_bad
    grouped["good_dist"] = np.where(grouped["good"] == 0, 0.5, grouped["good"]) / total_good
    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])

    woe_map = dict(zip(grouped["bin"], grouped["woe"]))
    return {"is_numeric": is_numeric, "edges": edges, "woe_map": woe_map, "default_woe": 0.0}


def apply_woe_encoder(df: pd.DataFrame, feature: str, encoder: dict) -> pd.Series:
    if encoder["is_numeric"]:
        bin_series = pd.cut(df[feature], bins=encoder["edges"], include_lowest=True).astype(object)
    else:
        bin_series = df[feature].astype(str)
    return bin_series.map(encoder["woe_map"]).fillna(encoder["default_woe"]).astype(float)
