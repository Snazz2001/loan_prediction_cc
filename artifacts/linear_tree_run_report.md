# Linear-tree LightGBM candidate — run report

Frozen protocol candidate. Dataset, target, split, and preprocessing were reused from
existing `scripts/train.py` / `utils/*` code (unchanged). Last-run champion
`artifacts/lgbm_model.joblib` was not written or overwritten. No extra models.
No CatBoost / EBM / AutoML. `employment_status` was kept. Interaction constraints
and monotonicity were not relaxed.

## 1. Paths + full source

- Train script path: `scripts/train_linear_tree_lgbm.py`
- Eval script path: `scripts/eval_linear_tree_lgbm.py`
- Model artifact: `artifacts/lgbm_linear_tree_model.joblib`
- Freeze meta (written BEFORE test metrics): `artifacts/lgbm_linear_tree_meta.json`
- Test metrics json (eval only): `artifacts/linear_tree_test_metrics.json`
- This report: `artifacts/linear_tree_run_report.md`

Full source of both scripts is at the bottom of this file.

## 2. Train OOF AUC vs last-run CV 0.8862

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **0.7732** (raw 0.7732413439398039)
- Train OOF KS (monitoring only, not used to select): 0.506
- Last-run CV AUC (compare only): 0.8862
- Beat last-run CV 0.8862? **NO**
- n_trials completed: 80
- n_estimators_final = max(50, round(mean best_iteration)) = **50**
- Fold best_iterations: [1, 1, 1, 1, 1]
- Fold AUCs (winning params re-run): [0.747183, 0.779088, 0.781089, 0.777005, 0.774276]
- Fold KS (monitoring): [0.462054, 0.514286, 0.523214, 0.516964, 0.513265]

Best hyperparameters (search params only; fixed `linear_tree=True`, `objective=binary`,
`boosting_type=gbdt`, `random_state=42`, monotone and interaction constraints):

| hyperparameter     |        value |
|:-------------------|-------------:|
| learning_rate      |   0.0138628  |
| num_leaves         |  88          |
| max_depth          |   8          |
| min_child_samples  | 154          |
| subsample          |   0.616395   |
| colsample_bytree   |   0.830193   |
| reg_alpha          |   0.00100901 |
| reg_lambda         |   0.00158896 |
| path_smooth        |   4.68557    |
| min_gain_to_split  |   0.215822   |
| n_estimators_final |  50          |

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 3. After freeze: Test metrics vs last-run champion

Test AUC 0.7645 did NOT beat last-run Test AUC 0.8858.

| metric | this candidate (Test) | last-run champion (Test, compare only) | beat last run? |
| --- | --- | --- | --- |
| AUC | **0.7645** | 0.8858 | NO |
| KS | **0.4989** | 0.5777 | n/a (selection was OOF AUC) |
| Gini | **0.5289** | 0.7715 | n/a |
| PSI (train PD vs test PD) | **0.0011** | 0.0021 | n/a |

- PSI status: Stable (<0.10) - No action needed
- Last-run Train AUC/KS (compare only, not recomputed): 0.8908 / 0.5819
- Last-run LR scorecard Test AUC/KS (compare only, not retuned): 0.8812 / 0.5717
- This candidate Train refit (not OOF) AUC/KS: 0.7728 / 0.506

| model                                                    | sample                 |    AUC |   Gini |     KS | rating                   |
|:---------------------------------------------------------|:-----------------------|-------:|-------:|-------:|:-------------------------|
| Linear-tree LightGBM (this candidate)                    | Train (refit, not OOF) | 0.7728 | 0.5455 | 0.506  | Good (0.40 - 0.60)       |
| Linear-tree LightGBM (this candidate)                    | Test (after freeze)    | 0.7645 | 0.5289 | 0.4989 | Good (0.40 - 0.60)       |
| Last-run champion LightGBM (compare only, not retrained) | Test                   | 0.8858 | 0.7715 | 0.5777 | reported, not recomputed |

## 4. Train SHAP / governance exhibits (computed on TRAIN after freeze, not test)

### (i) mean |SHAP| by feature on train

Computed with Exact SHAP (`shap.explainers.Exact` on `predict_proba`, Independent train background n=200, `interactions=True`) on all 14000 train rows. Native LightGBM `pred_contrib` is not implemented for `linear_tree`.

| feature              |   mean_abs_shap |
|:---------------------|----------------:|
| employment_status    |        0.069619 |
| interest_rate        |        0        |
| grade_subgrade       |        0        |
| debt_to_income_ratio |        0        |
| delinquency_history  |        0        |
| num_of_delinquencies |        0        |

`utils.risk_skills.generate_shap_summary` on train (same model, no refit):
`{'xai_method': 'SHAP (Kernel/Tree)', 'global_importance_ranking': [['employment_status', 0.3252], ['debt_to_income_ratio', 0.0], ['interest_rate', 0.0], ['grade_subgrade', 0.0], ['delinquency_history', 0.0], ['num_of_delinquencies', 0.0]], 'note': 'generate_shap_summary uses shap.Explainer -> TreeExplainer, which is NOT valid for linear_tree (leaf linear models). Exact SHAP above is the exhibit of record.'}`

### (ii) pairwise SHAP interaction of employment_status vs each other feature

Must be ~0 by construction because `interaction_constraints` isolate `employment_status`
from the other five features. Method: `shap.explainers.Exact(predict_proba, Independent(train_background=200), interactions=True) on all train rows. Native LightGBM pred_contrib is not implemented for linear_tree; TreeExplainer would be invalid.`

| feature_a         | feature_b            |   mean_abs_interaction |   mean_interaction |   max_abs_interaction |
|:------------------|:---------------------|-----------------------:|-------------------:|----------------------:|
| employment_status | debt_to_income_ratio |                      0 |                 -0 |                     0 |
| employment_status | interest_rate        |                      0 |                  0 |                     0 |
| employment_status | grade_subgrade       |                      0 |                 -0 |                     0 |
| employment_status | delinquency_history  |                      0 |                 -0 |                     0 |
| employment_status | num_of_delinquencies |                      0 |                  0 |                     0 |

### (iii) WOE bin table for employment_status on TRAIN

Especially Unemployed vs Retired. WOE mapping came from the frozen train-only encoder
(`fit_woe_encoder` / `calculate_woe_iv`), not from test.

| employment_status   |    n |   n_default |   default_rate |     woe |
|:--------------------|-----:|------------:|---------------:|--------:|
| Unemployed          | 1475 |        1210 |         0.8203 | -2.9045 |
| Student             |  566 |         332 |         0.5866 | -1.7357 |
| Employed            | 9086 |        1029 |         0.1133 |  0.6721 |
| Self-employed       | 2021 |         224 |         0.1108 |  0.6964 |
| Retired             |  852 |           6 |         0.007  |  3.5629 |

- Unemployed: {'employment_status': 'Unemployed', 'n': 1475, 'n_default': 1210, 'default_rate': 0.8203, 'woe': -2.9045}
- Retired: {'employment_status': 'Retired', 'n': 852, 'n_default': 6, 'default_rate': 0.007, 'woe': 3.5629}
- IV (train, `calculate_woe_iv`): 1.9096 (Suspicious / Overfitting (>0.5))

`employment_status` remains flagged for governance / fair-lending review and is **not
cleared for production use**.

## 5. Artifact paths (new names only)

- `artifacts/lgbm_linear_tree_model.joblib`
- `artifacts/lgbm_linear_tree_meta.json`
- `artifacts/linear_tree_test_metrics.json`
- `artifacts/linear_tree_run_report.md`
- `artifacts/linear_tree_woe_encoders.joblib`
- `artifacts/linear_tree_feature_selection.json`
- `artifacts/linear_tree_X_train.csv`
- `artifacts/linear_tree_y_train.csv`
- `artifacts/linear_tree_X_test.csv`
- `artifacts/linear_tree_y_test.csv`

Explicitly **not** written: `artifacts/lgbm_model.joblib`.

## 6. Freeze-before-test confirmation

- Freeze timestamp (UTC), written by the train script before this eval script ran: **2026-08-29T04:20:53.928978+00:00**
- `freeze.test_looked_at` at freeze time: `False`
- `freeze.test_metrics` at freeze time: `None`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select=False`)
- Eval script never calls `fit` / Optuna / early stopping. It only `predict_proba` +
  `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split / preprocessing: `scripts/train.py` `load_and_split` + `screen_features` +
  `iterative_vif_filter` and `utils/woe_encoding.py` fit-on-train / transform-on-test.
  Train n=14000 bad_rate=0.2001;
  Test n=6000 bad_rate=0.2002.
- final_features order: ['employment_status', 'debt_to_income_ratio', 'interest_rate', 'grade_subgrade', 'delinquency_history', 'num_of_delinquencies']
- monotone_constraints: [-1, -1, -1, -1, -1, -1]
- interaction_constraints: [['employment_status'], ['debt_to_income_ratio', 'interest_rate', 'grade_subgrade', 'delinquency_history', 'num_of_delinquencies']]
- linear_tree: True

## 7. Plain statement on Test AUC vs 0.8858

Test AUC 0.7645 did NOT beat last-run Test AUC 0.8858.

Beat last-run Test AUC 0.8858? **NO**

---

## Appendix A — full source of `scripts/train_linear_tree_lgbm.py`

```python
"""
Linear-tree LightGBM candidate (train / tune only).

Reproduces the FROZEN split + WOE/IV/VIF by calling scripts/train.py helpers
UNCHANGED (load_and_split, screen_features, iterative_vif_filter) plus the
existing utils/woe_encoding.py and utils/risk_skills.py functions. Does not
rewrite last-run artifacts (never writes artifacts/lgbm_model.joblib).

Train/tune only: Optuna TPE (seed=42), 80 trials, 5-fold StratifiedKFold on
TRAIN. Selection rule is Train OOF AUC vs last-run CV 0.8862. Freeze timestamp
is written BEFORE any Test discrimination metric is computed. Test labels are
persisted for the eval script and are never used to fit or select.

Usage (from repo root): python3 scripts/train_linear_tree_lgbm.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from train import iterative_vif_filter, load_and_split, screen_features
from utils.config import (
    ARTIFACTS_DIR,
    IV_MIN,
    IV_SUSPICIOUS,
    RANDOM_STATE,
    TARGET,
    VIF_MAX,
    WOE_BINS,
)
from utils.risk_skills import calculate_woe_iv, generate_shap_summary
from utils.woe_encoding import apply_woe_encoder, fit_woe_encoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPECTED_FINAL_FEATURES = [
    "employment_status",
    "debt_to_income_ratio",
    "interest_rate",
    "grade_subgrade",
    "delinquency_history",
    "num_of_delinquencies",
]
MONOTONE_CONSTRAINTS = [-1, -1, -1, -1, -1, -1]
INTERACTION_CONSTRAINTS = [
    ["employment_status"],
    [
        "debt_to_income_ratio",
        "interest_rate",
        "grade_subgrade",
        "delinquency_history",
        "num_of_delinquencies",
    ],
]
LAST_RUN_CV_AUC = 0.8862
N_TRIALS = 80
CV_FOLDS = 5
N_ESTIMATORS_TUNE = 1000
EARLY_STOPPING_ROUNDS = 30

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_model.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_meta.json")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_woe_encoders.joblib")
FEATURE_SELECTION_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_feature_selection.json")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_y_test.csv")


def _ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _best_iteration(model: lgb.LGBMClassifier) -> int:
    bi = getattr(model, "best_iteration_", None)
    if bi is None:
        bi = getattr(model, "best_iteration", None)
    if bi is None or int(bi) <= 0:
        n_est = getattr(model, "n_estimators_", None) or model.get_params().get("n_estimators", N_ESTIMATORS_TUNE)
        return int(n_est)
    return int(bi)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def frozen_params_template() -> dict:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "linear_tree": True,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": 1,
        "subsample_freq": 1,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "interaction_constraints": INTERACTION_CONSTRAINTS,
    }


def reproduce_frozen_preprocessing():
    """Call existing train.py / utils preprocessing UNCHANGED. Never invent a split."""
    train_df, test_df, feature_cols = load_and_split()

    iv_table = screen_features(train_df, feature_cols)
    dropped_by_iv = iv_table.loc[iv_table["iv"] < IV_MIN, "feature"].tolist()
    flagged_as_suspicious = iv_table.loc[iv_table["iv"] > IV_SUSPICIOUS, "feature"].tolist()
    kept_after_iv = [f for f in feature_cols if f not in dropped_by_iv]

    encoders = {f: fit_woe_encoder(train_df, f, TARGET, bins=WOE_BINS) for f in kept_after_iv}
    X_train_woe = pd.DataFrame({f: apply_woe_encoder(train_df, f, encoders[f]) for f in kept_after_iv})
    y_train = train_df[TARGET].astype(int)

    final_features, dropped_by_vif, vif_rounds = iterative_vif_filter(X_train_woe, kept_after_iv, VIF_MAX)

    if list(final_features) != EXPECTED_FINAL_FEATURES:
        if set(final_features) != set(EXPECTED_FINAL_FEATURES):
            raise RuntimeError(
                "Frozen preprocessing did not produce the required final_features set. "
                f"got={final_features} expected={EXPECTED_FINAL_FEATURES}"
            )
        final_features = list(EXPECTED_FINAL_FEATURES)

    if "employment_status" not in final_features:
        raise RuntimeError("employment_status was dropped; frozen protocol forbids that.")
    if "gender" in final_features:
        raise RuntimeError("gender should remain IV-dropped.")
    if "credit_score" in final_features:
        raise RuntimeError("credit_score should remain VIF-dropped.")

    X_train = X_train_woe[final_features].copy()
    X_test = pd.DataFrame({f: apply_woe_encoder(test_df, f, encoders[f]) for f in final_features})
    y_test = test_df[TARGET].astype(int)

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
        "protocol_note": (
            "Reproduced by calling scripts/train.py helpers unchanged. "
            "This file is a linear-tree candidate copy; it does not replace "
            "artifacts/feature_selection.json from the last-run champion."
        ),
    }
    return train_df, test_df, encoders, X_train, y_train, X_test, y_test, feature_selection


def fit_fold(params: dict, X_tr, y_tr, X_va, y_va) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_X=X_va,
        eval_y=y_va,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def run_oof_cv(params: dict, X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold):
    oof = np.zeros(len(X), dtype=float)
    best_iters = []
    fold_aucs = []
    fold_ks = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = fit_fold(params, X_tr, y_tr, X_va, y_va)
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        best_iters.append(_best_iteration(model))
        fold_aucs.append(float(roc_auc_score(y_va, pred)))
        fold_ks.append(_ks(y_va.values, pred))
        print(
            f"    fold {fold_i}/{CV_FOLDS}: AUC={fold_aucs[-1]:.6f} "
            f"KS={fold_ks[-1]:.6f} best_iteration={best_iters[-1]}"
        )
    oof_auc = float(roc_auc_score(y, oof))
    oof_ks = _ks(y.values, oof)
    return {
        "oof_preds": oof,
        "oof_auc": oof_auc,
        "oof_ks": oof_ks,
        "best_iterations": best_iters,
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
    }


def employment_status_woe_table(train_df: pd.DataFrame, encoders: dict) -> list:
    grp = (
        train_df.groupby("employment_status", observed=True)[TARGET]
        .agg(n="count", n_default="sum", default_rate="mean")
        .reset_index()
    )
    woe_map = {str(k): float(v) for k, v in encoders["employment_status"]["woe_map"].items()}
    grp["woe"] = grp["employment_status"].astype(str).map(woe_map)
    grp["default_rate"] = grp["default_rate"].astype(float)
    grp = grp.sort_values("default_rate", ascending=False)
    rows = []
    for rec in grp.to_dict(orient="records"):
        rows.append(
            {
                "employment_status": rec["employment_status"],
                "n": int(rec["n"]),
                "n_default": int(rec["n_default"]),
                "default_rate": round(float(rec["default_rate"]), 4),
                "woe": round(float(rec["woe"]), 4),
            }
        )
    return rows


def compute_train_shap_exhibits(model: lgb.LGBMClassifier, X_train: pd.DataFrame) -> dict:
    """
    LightGBM does not implement pred_contrib / pred_interactions for linear_tree
    models, and shap.TreeExplainer treats leaves as constants so it is invalid
    here. With 6 features Exact SHAP is feasible; explanations are computed on
    TRAIN rows only (Independent masker also sampled from TRAIN).
    """
    import shap

    features = list(X_train.columns)

    def predict_pd(X):
        if isinstance(X, pd.DataFrame):
            frame = X[features] if set(features).issubset(X.columns) else X
        else:
            frame = pd.DataFrame(np.asarray(X), columns=features)
        return model.predict_proba(frame)[:, 1]

    background = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.explainers.Exact(predict_pd, masker)
    print(f"[shap] Exact SHAP interactions on all {len(X_train)} train rows (background n={len(background)})...")
    explanation = explainer(X_train, interactions=True)
    inter = np.asarray(explanation.values)
    shap_main = inter.sum(axis=2)
    mean_abs = np.abs(shap_main).mean(axis=0)
    mean_abs_rows = [
        {"feature": f, "mean_abs_shap": round(float(v), 6)}
        for f, v in sorted(zip(features, mean_abs), key=lambda x: x[1], reverse=True)
    ]

    emp_idx = features.index("employment_status")
    pairwise = []
    for j, feat in enumerate(features):
        if feat == "employment_status":
            continue
        pair_vals = inter[:, emp_idx, j] + inter[:, j, emp_idx]
        pairwise.append(
            {
                "feature_a": "employment_status",
                "feature_b": feat,
                "mean_abs_interaction": round(float(np.abs(pair_vals).mean()), 8),
                "mean_interaction": round(float(pair_vals.mean()), 8),
                "max_abs_interaction": round(float(np.abs(pair_vals).max()), 8),
            }
        )

    skill_shap = None
    try:
        skill_shap = generate_shap_summary(model, X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_STATE))
        skill_shap["note"] = (
            "generate_shap_summary uses shap.Explainer -> TreeExplainer, which is NOT "
            "valid for linear_tree (leaf linear models). Exact SHAP above is the exhibit of record."
        )
    except Exception as exc:  # noqa: BLE001
        skill_shap = {
            "error": str(exc),
            "note": "Expected: TreeExplainer/pred_contrib is not implemented for linear_tree.",
        }

    return {
        "mean_abs_shap_by_feature": mean_abs_rows,
        "employment_status_pairwise_interactions": pairwise,
        "interaction_method": (
            "shap.explainers.Exact(predict_proba, Independent(train_background=200), "
            "interactions=True) on all train rows. Native LightGBM pred_contrib is not "
            "implemented for linear_tree; TreeExplainer would be invalid."
        ),
        "generate_shap_summary_train": skill_shap,
    }


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if os.path.exists(os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")):
        print("[train_linear_tree] Note: artifacts/lgbm_model.joblib exists and will NOT be overwritten.")

    print("[train_linear_tree] Reproducing frozen split + WOE/IV/VIF via scripts/train.py helpers...")
    (
        train_df,
        test_df,
        encoders,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_selection,
    ) = reproduce_frozen_preprocessing()

    print(
        f"[train_linear_tree] Train n={len(train_df)}, bad_rate={float(y_train.mean()):.4f} | "
        f"Test n={len(test_df)}, bad_rate={float(y_test.mean()):.4f} (held out, unused for tune/select)"
    )
    print(f"[train_linear_tree] final_features={list(X_train.columns)}")
    print(f"[train_linear_tree] monotone_constraints={MONOTONE_CONSTRAINTS}")
    print(f"[train_linear_tree] interaction_constraints={INTERACTION_CONSTRAINTS}")

    X_train.to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame({TARGET: y_train.values}).to_csv(Y_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    pd.DataFrame({TARGET: y_test.values}).to_csv(Y_TEST_PATH, index=False)
    joblib.dump(encoders, ENCODERS_PATH)
    with open(FEATURE_SELECTION_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(feature_selection), fh, ensure_ascii=False, indent=2)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = frozen_params_template()
        params.update(
            {
                "n_estimators": N_ESTIMATORS_TUNE,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            }
        )
        print(f"[tune] trial {trial.number:03d}/{N_TRIALS - 1} starting")
        cv_res = run_oof_cv(params, X_train, y_train, skf)
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        trial.set_user_attr("fold_best_iterations", cv_res["best_iterations"])
        trial.set_user_attr("fold_aucs", cv_res["fold_aucs"])
        print(
            f"[tune] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f} (KS monitoring only)"
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_search = dict(study.best_params)
    winning_params = frozen_params_template()
    winning_params.update(best_search)
    winning_params["n_estimators"] = N_ESTIMATORS_TUNE

    print("[train_linear_tree] Re-running winning trial's 5 train folds to set n_estimators_final...")
    winner_cv = run_oof_cv(winning_params, X_train, y_train, skf)
    n_estimators_final = max(50, int(round(float(np.mean(winner_cv["best_iterations"])))))
    train_oof_auc = round(float(winner_cv["oof_auc"]), 4)
    train_oof_ks = round(float(winner_cv["oof_ks"]), 4)
    beats_last_cv = bool(winner_cv["oof_auc"] > LAST_RUN_CV_AUC)

    print(
        f"[train_linear_tree] Train OOF AUC={train_oof_auc} (raw={winner_cv['oof_auc']:.6f}) "
        f"vs last CV {LAST_RUN_CV_AUC} -> beat={beats_last_cv}"
    )
    print(
        f"[train_linear_tree] fold best_iterations={winner_cv['best_iterations']} "
        f"-> n_estimators_final={n_estimators_final}"
    )

    final_params = dict(winning_params)
    final_params["n_estimators"] = n_estimators_final
    print(
        f"[train_linear_tree] Refitting on ALL {len(X_train)} train rows with "
        f"n_estimators={n_estimators_final}, no early stopping, no test."
    )
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_linear_tree] FREEZE at {freeze_timestamp} (before any test metrics)")

    print("[train_linear_tree] Computing TRAIN-ONLY governance SHAP exhibits...")
    shap_exhibits = compute_train_shap_exhibits(final_model, X_train)
    emp_woe_table = employment_status_woe_table(train_df, encoders)
    emp_iv = calculate_woe_iv(train_df, "employment_status", TARGET, bins=WOE_BINS)

    feature_importances = sorted(
        zip(list(X_train.columns), [int(x) for x in final_model.feature_importances_]),
        key=lambda x: x[1],
        reverse=True,
    )

    meta = {
        "algorithm": "lightgbm.LGBMClassifier",
        "fixed": {
            "objective": "binary",
            "boosting_type": "gbdt",
            "linear_tree": True,
            "random_state": RANDOM_STATE,
            "monotone_constraints": MONOTONE_CONSTRAINTS,
            "interaction_constraints": INTERACTION_CONSTRAINTS,
            "final_features": EXPECTED_FINAL_FEATURES,
        },
        "search_space": {
            "n_trials": N_TRIALS,
            "cv_folds": CV_FOLDS,
            "objective": "OOF AUC",
            "ks_role": "monitoring only",
            "learning_rate": "log 0.01-0.1",
            "num_leaves": "15-127",
            "max_depth": "3-8",
            "min_child_samples": "20-300",
            "subsample": "0.6-1.0 (subsample_freq=1)",
            "colsample_bytree": "0.6-1.0",
            "reg_alpha": "log 1e-3-10",
            "reg_lambda": "log 1e-3-10",
            "path_smooth": "0-10",
            "min_gain_to_split": "0-0.5",
            "n_estimators_tune": N_ESTIMATORS_TUNE,
            "early_stopping": EARLY_STOPPING_ROUNDS,
        },
        "best_hyperparameters": _json_ready(best_search),
        "n_estimators_final": n_estimators_final,
        "winning_fold_best_iterations": winner_cv["best_iterations"],
        "winning_fold_aucs": [round(x, 6) for x in winner_cv["fold_aucs"]],
        "winning_fold_ks": [round(x, 6) for x in winner_cv["fold_ks"]],
        "train_oof_auc": train_oof_auc,
        "train_oof_auc_raw": float(winner_cv["oof_auc"]),
        "train_oof_ks_monitoring": train_oof_ks,
        "n_trials_completed": len(study.trials),
        "feature_importances": [{"feature": f, "importance": int(v)} for f, v in feature_importances],
        "split": {
            "train_n": int(len(train_df)),
            "train_bad_rate": round(float(y_train.mean()), 4),
            "test_n": int(len(test_df)),
            "test_bad_rate": round(float(y_test.mean()), 4),
            "note": "Test labels persisted for eval only; not used to fit or select.",
        },
        "freeze": {
            "timestamp_utc": freeze_timestamp,
            "selection_rule": "higher train OOF AUC vs last run CV 0.8862",
            "last_run_cv_auc": LAST_RUN_CV_AUC,
            "train_oof_auc": train_oof_auc,
            "beats_last_cv": beats_last_cv,
            "test_looked_at": False,
            "test_metrics": None,
            "test_labels_used_to_fit_or_select": False,
        },
        "train_governance_exhibits": {
            "mean_abs_shap_by_feature": shap_exhibits["mean_abs_shap_by_feature"],
            "employment_status_pairwise_shap_interactions": shap_exhibits[
                "employment_status_pairwise_interactions"
            ],
            "interaction_method": shap_exhibits["interaction_method"],
            "generate_shap_summary_train": shap_exhibits["generate_shap_summary_train"],
            "employment_status_woe_bin_table": emp_woe_table,
            "employment_status_iv": {
                "iv": emp_iv["iv"],
                "predictive_power": emp_iv["predictive_power"],
                "bin_details": emp_iv["bin_details"],
            },
        },
        "artifact_paths": {
            "model": MODEL_PATH,
            "meta": META_PATH,
            "encoders": ENCODERS_PATH,
            "feature_selection": FEATURE_SELECTION_PATH,
            "X_train": X_TRAIN_PATH,
            "y_train": Y_TRAIN_PATH,
            "X_test": X_TEST_PATH,
            "y_test": Y_TEST_PATH,
        },
        "did_not_write": ["artifacts/lgbm_model.joblib"],
    }

    joblib.dump(final_model, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta), fh, ensure_ascii=False, indent=2)

    print(f"[train_linear_tree] Wrote {MODEL_PATH}")
    print(f"[train_linear_tree] Wrote {META_PATH} (freeze complete; no test metrics inside)")
    print("[train_linear_tree] STOPPING. Run python3 scripts/eval_linear_tree_lgbm.py for test metrics.")


if __name__ == "__main__":
    main()

```

## Appendix B — full source of `scripts/eval_linear_tree_lgbm.py`

```python
"""
Linear-tree LightGBM candidate (eval only).

Loads frozen artifacts from scripts/train_linear_tree_lgbm.py. Never fits a
model, never re-tunes, never re-runs Optuna. Computes Test AUC / KS / Gini /
PSI with the same functions as the last run (evaluate_discrimination_and_ks
and calculate_psi from utils/risk_skills.py) and writes
artifacts/linear_tree_run_report.md.

Usage (from repo root): python3 scripts/eval_linear_tree_lgbm.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd

from utils.config import ARTIFACTS_DIR, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_model.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_meta.json")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_y_test.csv")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_run_report.md")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_test_metrics.json")

LAST_RUN_CV_AUC = 0.8862
LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
LAST_RUN_TRAIN_AUC = 0.8908
LAST_RUN_TRAIN_KS = 0.5819
LAST_RUN_LR_TEST_AUC = 0.8812
LAST_RUN_LR_TEST_KS = 0.5717

TRAIN_SCRIPT = "scripts/train_linear_tree_lgbm.py"
EVAL_SCRIPT = "scripts/eval_linear_tree_lgbm.py"


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load_source(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def rows_to_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main() -> None:
    missing = [
        p
        for p in [MODEL_PATH, META_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, TRAIN_SCRIPT, EVAL_SCRIPT]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python3 scripts/train_linear_tree_lgbm.py` first. "
            "This eval script never fits."
        )

    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)

    freeze = meta["freeze"]
    if freeze.get("test_looked_at") is True or freeze.get("test_metrics") is not None:
        raise RuntimeError(
            "Freeze record already contains test metrics. Train script must freeze "
            "BEFORE the eval script looks at test."
        )
    if freeze.get("test_labels_used_to_fit_or_select") is not False:
        raise RuntimeError("Freeze record does not confirm test labels were unused for fit/select.")

    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET]
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET]

    expected = meta["fixed"]["final_features"]
    if list(X_train.columns) != expected or list(X_test.columns) != expected:
        raise RuntimeError(
            f"Encoded matrix column order mismatch. train={list(X_train.columns)} "
            f"test={list(X_test.columns)} expected={expected}"
        )

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_discrimination_and_ks(y_train.values, train_prob)
    test_metrics = evaluate_discrimination_and_ks(y_test.values, test_prob)
    score_psi = calculate_psi(train_prob, test_prob, num_bins=10)

    beats_last_test_auc = bool(test_metrics["AUC"] > LAST_RUN_TEST_AUC)
    beats_last_cv = bool(freeze["beats_last_cv"])

    test_payload = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze["timestamp_utc"],
        "train_refit_metrics": train_metrics,
        "test_metrics": test_metrics,
        "score_psi": score_psi,
        "beats_last_run_test_auc_0_8858": beats_last_test_auc,
        "used_to_select_model": False,
        "note": (
            "Test metrics are reported after freeze. They were not used to choose "
            "hyperparameters, n_estimators_final, or whether to keep the candidate."
        ),
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_payload, fh, ensure_ascii=False, indent=2)

    exhibits = meta["train_governance_exhibits"]
    shap_table = rows_to_df(exhibits["mean_abs_shap_by_feature"])
    inter_table = rows_to_df(exhibits["employment_status_pairwise_shap_interactions"])
    woe_table = rows_to_df(exhibits["employment_status_woe_bin_table"])
    params_table = rows_to_df(
        [{"hyperparameter": k, "value": v} for k, v in meta["best_hyperparameters"].items()]
        + [{"hyperparameter": "n_estimators_final", "value": meta["n_estimators_final"]}]
    )
    perf_table = rows_to_df(
        [
            {
                "model": "Linear-tree LightGBM (this candidate)",
                "sample": "Train (refit, not OOF)",
                "AUC": train_metrics["AUC"],
                "Gini": train_metrics["Gini"],
                "KS": train_metrics["KS_Statistic"],
                "rating": train_metrics["Validation_Rating"],
            },
            {
                "model": "Linear-tree LightGBM (this candidate)",
                "sample": "Test (after freeze)",
                "AUC": test_metrics["AUC"],
                "Gini": test_metrics["Gini"],
                "KS": test_metrics["KS_Statistic"],
                "rating": test_metrics["Validation_Rating"],
            },
            {
                "model": "Last-run champion LightGBM (compare only, not retrained)",
                "sample": "Test",
                "AUC": LAST_RUN_TEST_AUC,
                "Gini": LAST_RUN_TEST_GINI,
                "KS": LAST_RUN_TEST_KS,
                "rating": "reported, not recomputed",
            },
        ]
    )

    unemployed = next((r for r in exhibits["employment_status_woe_bin_table"] if r["employment_status"] == "Unemployed"), None)
    retired = next((r for r in exhibits["employment_status_woe_bin_table"] if r["employment_status"] == "Retired"), None)

    beat_cv_plain = "YES" if beats_last_cv else "NO"
    beat_test_plain = "YES" if beats_last_test_auc else "NO"
    test_auc_plain = (
        f"Test AUC {test_metrics['AUC']} DID beat last-run Test AUC {LAST_RUN_TEST_AUC}."
        if beats_last_test_auc
        else f"Test AUC {test_metrics['AUC']} did NOT beat last-run Test AUC {LAST_RUN_TEST_AUC}."
    )

    train_src = load_source(TRAIN_SCRIPT)
    eval_src = load_source(EVAL_SCRIPT)

    report = f"""# Linear-tree LightGBM candidate — run report

Frozen protocol candidate. Dataset, target, split, and preprocessing were reused from
existing `scripts/train.py` / `utils/*` code (unchanged). Last-run champion
`artifacts/lgbm_model.joblib` was not written or overwritten. No extra models.
No CatBoost / EBM / AutoML. `employment_status` was kept. Interaction constraints
and monotonicity were not relaxed.

## 1. Paths + full source

- Train script path: `{TRAIN_SCRIPT}`
- Eval script path: `{EVAL_SCRIPT}`
- Model artifact: `{MODEL_PATH}`
- Freeze meta (written BEFORE test metrics): `{META_PATH}`
- Test metrics json (eval only): `{TEST_METRICS_PATH}`
- This report: `{REPORT_PATH}`

Full source of both scripts is at the bottom of this file.

## 2. Train OOF AUC vs last-run CV 0.8862

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **{meta['train_oof_auc']}** (raw {meta['train_oof_auc_raw']})
- Train OOF KS (monitoring only, not used to select): {meta['train_oof_ks_monitoring']}
- Last-run CV AUC (compare only): {LAST_RUN_CV_AUC}
- Beat last-run CV 0.8862? **{beat_cv_plain}**
- n_trials completed: {meta['n_trials_completed']}
- n_estimators_final = max(50, round(mean best_iteration)) = **{meta['n_estimators_final']}**
- Fold best_iterations: {meta['winning_fold_best_iterations']}
- Fold AUCs (winning params re-run): {meta['winning_fold_aucs']}
- Fold KS (monitoring): {meta['winning_fold_ks']}

Best hyperparameters (search params only; fixed `linear_tree=True`, `objective=binary`,
`boosting_type=gbdt`, `random_state=42`, monotone and interaction constraints):

{md_table(params_table)}

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 3. After freeze: Test metrics vs last-run champion

{test_auc_plain}

| metric | this candidate (Test) | last-run champion (Test, compare only) | beat last run? |
| --- | --- | --- | --- |
| AUC | **{test_metrics['AUC']}** | {LAST_RUN_TEST_AUC} | {beat_test_plain} |
| KS | **{test_metrics['KS_Statistic']}** | {LAST_RUN_TEST_KS} | n/a (selection was OOF AUC) |
| Gini | **{test_metrics['Gini']}** | {LAST_RUN_TEST_GINI} | n/a |
| PSI (train PD vs test PD) | **{score_psi['PSI']}** | {LAST_RUN_TEST_PSI} | n/a |

- PSI status: {score_psi['Status']}
- Last-run Train AUC/KS (compare only, not recomputed): {LAST_RUN_TRAIN_AUC} / {LAST_RUN_TRAIN_KS}
- Last-run LR scorecard Test AUC/KS (compare only, not retuned): {LAST_RUN_LR_TEST_AUC} / {LAST_RUN_LR_TEST_KS}
- This candidate Train refit (not OOF) AUC/KS: {train_metrics['AUC']} / {train_metrics['KS_Statistic']}

{md_table(perf_table)}

## 4. Train SHAP / governance exhibits (computed on TRAIN after freeze, not test)

### (i) mean |SHAP| by feature on train

Computed with Exact SHAP (`shap.explainers.Exact` on `predict_proba`, Independent train background n=200, `interactions=True`) on all {len(X_train)} train rows. Native LightGBM `pred_contrib` is not implemented for `linear_tree`.

{md_table(shap_table)}

`utils.risk_skills.generate_shap_summary` on train (same model, no refit):
`{exhibits.get('generate_shap_summary_train')}`

### (ii) pairwise SHAP interaction of employment_status vs each other feature

Must be ~0 by construction because `interaction_constraints` isolate `employment_status`
from the other five features. Method: `{exhibits['interaction_method']}`

{md_table(inter_table)}

### (iii) WOE bin table for employment_status on TRAIN

Especially Unemployed vs Retired. WOE mapping came from the frozen train-only encoder
(`fit_woe_encoder` / `calculate_woe_iv`), not from test.

{md_table(woe_table)}

- Unemployed: {unemployed}
- Retired: {retired}
- IV (train, `calculate_woe_iv`): {exhibits['employment_status_iv']['iv']} ({exhibits['employment_status_iv']['predictive_power']})

`employment_status` remains flagged for governance / fair-lending review and is **not
cleared for production use**.

## 5. Artifact paths (new names only)

- `{MODEL_PATH}`
- `{META_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{os.path.join(ARTIFACTS_DIR, 'linear_tree_woe_encoders.joblib')}`
- `{os.path.join(ARTIFACTS_DIR, 'linear_tree_feature_selection.json')}`
- `{X_TRAIN_PATH}`
- `{Y_TRAIN_PATH}`
- `{X_TEST_PATH}`
- `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`.

## 6. Freeze-before-test confirmation

- Freeze timestamp (UTC), written by the train script before this eval script ran: **{freeze['timestamp_utc']}**
- `freeze.test_looked_at` at freeze time: `{freeze['test_looked_at']}`
- `freeze.test_metrics` at freeze time: `{freeze['test_metrics']}`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select={freeze['test_labels_used_to_fit_or_select']}`)
- Eval script never calls `fit` / Optuna / early stopping. It only `predict_proba` +
  `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split / preprocessing: `scripts/train.py` `load_and_split` + `screen_features` +
  `iterative_vif_filter` and `utils/woe_encoding.py` fit-on-train / transform-on-test.
  Train n={meta['split']['train_n']} bad_rate={meta['split']['train_bad_rate']};
  Test n={meta['split']['test_n']} bad_rate={meta['split']['test_bad_rate']}.
- final_features order: {expected}
- monotone_constraints: {meta['fixed']['monotone_constraints']}
- interaction_constraints: {meta['fixed']['interaction_constraints']}
- linear_tree: {meta['fixed']['linear_tree']}

## 7. Plain statement on Test AUC vs 0.8858

{test_auc_plain}

Beat last-run Test AUC 0.8858? **{beat_test_plain}**

---

## Appendix A — full source of `{TRAIN_SCRIPT}`

```python
{train_src}
```

## Appendix B — full source of `{EVAL_SCRIPT}`

```python
{eval_src}
```
"""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_linear_tree] Test AUC={test_metrics['AUC']} KS={test_metrics['KS_Statistic']} "
        f"Gini={test_metrics['Gini']} PSI={score_psi['PSI']}"
    )
    print(f"[eval_linear_tree] Beat last-run Test AUC {LAST_RUN_TEST_AUC}? {beat_test_plain}")
    print(f"[eval_linear_tree] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

```
