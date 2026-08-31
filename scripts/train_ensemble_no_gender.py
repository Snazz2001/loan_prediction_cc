"""
Four-base ensemble candidate (train / tune only) — no gender, no monotone.

Bases (TRAIN OOF only):
  1. sklearn LogisticRegression on WOE (l2, lbfgs, max_iter=1000)
  2. LightGBM gbdt (early stopping on fold valid)
  3. LightGBM DART (n_estimators searched; NO early_stopping)
  4. CatBoost (iterations searched; NO monotone)

Ensembles (meta on OOF PDs only):
  A. Equal-weight average of the four OOF PDs
  B. Logistic stack fitted on the four OOF PD columns (not refit on in-sample)

Frozen split via scripts/train.py::load_and_split UNCHANGED. Drops gender.
employment_status is a normal in-model WOE feature. ALL remaining original
features kept (no IV drop, no VIF drop). WOE 5-bin encoders fit on TRAIN only.

NO monotone_constraints. NO interaction_constraints. NO early_stopping on DART.

Optuna TPE (seed=42, n_trials=50) for gbdt / dart / catboost. LR uses a small
log grid for C. Freeze is written with test_looked_at=false / test_metrics=null
BEFORE any Test discrimination metric is computed.

Does not write last-run artifacts/lgbm_model.joblib or PR #1–#5 artifact prefixes.

If CatBoost import/install or DART runtime fails: STOP, write a failure note,
do not silently drop a base.

Usage (from repo root): python3 scripts/train_ensemble_no_gender.py
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURE_NOTE_PATH = os.path.join("artifacts", "ensemble_no_gender_FAILURE.md")


def _write_failure_and_stop(title: str, detail: str) -> None:
    os.makedirs("artifacts", exist_ok=True)
    body = (
        f"# Ensemble no-gender failure — STOP\n\n"
        f"**{title}**\n\n"
        f"All four bases are required (LR, LightGBM gbdt, LightGBM DART, CatBoost). "
        f"This run stopped rather than silently dropping a base.\n\n"
        f"```\n{detail}\n```\n"
    )
    with open(FAILURE_NOTE_PATH, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"[train_ensemble] FAILURE written to {FAILURE_NOTE_PATH}", flush=True)
    raise RuntimeError(f"{title}\n{detail}")


try:
    from catboost import CatBoostClassifier
except Exception as exc:  # noqa: BLE001
    _write_failure_and_stop(
        "CatBoost import failed",
        f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
    )

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from train import load_and_split
from utils.config import ARTIFACTS_DIR, RANDOM_STATE, RAW_TARGET_COL, TARGET, WOE_BINS
from utils.risk_skills import evaluate_discrimination_and_ks
from utils.woe_encoding import apply_woe_encoder, fit_woe_encoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
CV_FOLDS = 5
N_ESTIMATORS_GBDT_TUNE = 1000
EARLY_STOPPING_ROUNDS = 30
EXPECTED_TRAIN_N = 14000
EXPECTED_TRAIN_BAD_RATE = 0.2001
EXPECTED_TRAIN_N_BADS = 2801
EXPECTED_TEST_N = 6000
EXPECTED_TEST_BAD_RATE = 0.2002
EXPECTED_TEST_N_BADS = 1201

LR_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
N_JOBS = 4

DROPPED_FROM_MODEL = ("gender",)
REQUIRED_IN_MODEL = ("employment_status",)

EXPECTED_IN_MODEL_FEATURES = [
    "age",
    "marital_status",
    "education_level",
    "annual_income",
    "monthly_income",
    "employment_status",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "loan_purpose",
    "interest_rate",
    "loan_term",
    "installment",
    "grade_subgrade",
    "num_of_open_accounts",
    "total_credit_limit",
    "current_balance",
    "delinquency_history",
    "public_records",
    "num_of_delinquencies",
]

BASE_COL_ORDER = ["pd_lr", "pd_lgbm_gbdt", "pd_lgbm_dart", "pd_catboost"]

REQUIRED_PATHS = [
    "data/loan_dataset_20000.csv",
    "scripts/train.py",
    "scripts/test.py",
    "utils/config.py",
    "utils/risk_skills.py",
    "utils/woe_encoding.py",
]

FORBIDDEN_PREFIXES = (
    "lgbm_linear_tree_",
    "linear_tree_",
    "stack_lr_rf_lgbm_",
    "lgbm_no_gender_emp_overlay_",
    "lgbm_emp_in_no_gender_",
    "lgbm_dart_monotone_",
)

ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_encoders.joblib")
LR_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_lr.joblib")
GBDT_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_lgbm_gbdt.joblib")
DART_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_lgbm_dart.joblib")
CATBOOST_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_catboost.joblib")
STACK_META_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_stack_meta.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_y_test.csv")

ALLOWED_WRITE_PATHS = [
    ENCODERS_PATH,
    LR_PATH,
    GBDT_PATH,
    DART_PATH,
    CATBOOST_PATH,
    STACK_META_PATH,
    META_JSON_PATH,
    OOF_PDS_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    FAILURE_NOTE_PATH,
]


def _require_pipeline_files() -> None:
    missing = [p for p in REQUIRED_PATHS if not os.path.exists(p)]
    if missing:
        found = []
        for root, _dirs, files in os.walk("."):
            if any(skip in root.split(os.sep) for skip in (".git", "__pycache__", ".cursor")):
                continue
            for fn in files:
                found.append(os.path.join(root, fn).lstrip("./"))
        raise FileNotFoundError(
            "Required pipeline files missing: "
            + ", ".join(missing)
            + ". Files found: "
            + ", ".join(sorted(found)[:200])
        )


def _assert_allowed_writes(paths: list[str]) -> None:
    allowed_set = {os.path.abspath(p) for p in ALLOWED_WRITE_PATHS}
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path not in allowed_set:
            raise RuntimeError(f"Refusing to write undeclared artifact: {path}")
        base = os.path.basename(path)
        if base == "lgbm_model.joblib" or base.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"Refusing to write forbidden artifact name: {path}")


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj


def _ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _assert_load_and_split_unchanged() -> None:
    import train as train_mod

    src = inspect.getsource(train_mod.load_and_split)
    for needle in (
        "test_size=TEST_SIZE",
        "stratify=raw[TARGET]",
        "random_state=RANDOM_STATE",
        "1 - raw[RAW_TARGET_COL]",
        "raw.drop(columns=[RAW_TARGET_COL])",
    ):
        if needle not in src:
            raise RuntimeError(
                f"scripts/train.py::load_and_split appears changed; missing {needle!r}"
            )


def _assert_in_model_columns(columns: list[str], context: str, *, require_order: bool = True) -> None:
    cols = list(columns)
    if "gender" in cols:
        raise RuntimeError(
            f"{context}: gender must be ABSENT from in-model X / encoders / "
            f"feature_name_ / coef / SHAP. Found gender in {cols}"
        )
    if "employment_status" not in cols:
        raise RuntimeError(
            f"{context}: employment_status must be PRESENT as a normal in-model WOE feature. "
            f"columns={cols}"
        )
    if RAW_TARGET_COL in cols:
        raise RuntimeError(f"{context}: {RAW_TARGET_COL} must never be a feature. columns={cols}")
    if TARGET in cols:
        raise RuntimeError(f"{context}: target {TARGET} present in feature columns: {cols}")
    if len(cols) != 20:
        raise RuntimeError(f"{context}: expected exactly 20 in-model columns, got {len(cols)}: {cols}")
    if set(cols) != set(EXPECTED_IN_MODEL_FEATURES):
        raise RuntimeError(
            f"{context}: in-model column set mismatch. got={cols} expected={EXPECTED_IN_MODEL_FEATURES}"
        )
    if require_order and cols != EXPECTED_IN_MODEL_FEATURES:
        raise RuntimeError(
            f"{context}: in-model column ORDER mismatch. got={cols} expected={EXPECTED_IN_MODEL_FEATURES}"
        )


def _assert_no_constraint_params(params: dict, context: str) -> None:
    for key in ("monotone_constraints", "interaction_constraints"):
        if key not in params:
            continue
        val = params[key]
        if val not in (None, [], (), "", "None"):
            raise RuntimeError(f"{context}: {key}={val!r} is forbidden")


def _assert_fitted_no_constraints(model, context: str) -> None:
    params = model.get_params() if hasattr(model, "get_params") else {}
    _assert_no_constraint_params(params, context)
    for key in ("monotone_constraints", "interaction_constraints"):
        if hasattr(model, key):
            val = getattr(model, key)
            if val not in (None, [], (), "", "None"):
                raise RuntimeError(f"{context}: fitted attribute {key}={val!r} is forbidden")


def _feature_names_from_model(model, fallback: list[str]) -> list[str]:
    if hasattr(model, "feature_name_"):
        names = list(model.feature_name_)
        if names:
            return names
    if hasattr(model, "feature_names_"):
        names = list(model.feature_names_)
        if names:
            return names
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        if names:
            return [str(x) for x in names]
    return list(fallback)


def _best_iteration_lgbm(model: lgb.LGBMClassifier) -> int:
    bi = getattr(model, "best_iteration_", None)
    if bi is None:
        bi = getattr(model, "best_iteration", None)
    if bi is None or int(bi) <= 0:
        booster = getattr(model, "booster_", None)
        if booster is not None:
            bi = getattr(booster, "best_iteration", None)
    if bi is None or int(bi) <= 0:
        n_est = getattr(model, "n_estimators_", None) or model.get_params().get(
            "n_estimators", N_ESTIMATORS_GBDT_TUNE
        )
        return int(n_est)
    return int(bi)


def encode_woe(df: pd.DataFrame, feature_cols: list[str], encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in feature_cols})


def build_in_model_features(feature_cols: list[str]) -> list[str]:
    in_model = [c for c in feature_cols if c not in DROPPED_FROM_MODEL]
    _assert_in_model_columns(in_model, "build_in_model_features")
    return in_model


def verify_frozen_split(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    train_n = len(train_df)
    test_n = len(test_df)
    train_bads = int(train_df[TARGET].sum())
    test_bads = int(test_df[TARGET].sum())
    train_rate = round(float(train_df[TARGET].mean()), 4)
    test_rate = round(float(test_df[TARGET].mean()), 4)
    if train_n != EXPECTED_TRAIN_N or train_rate != EXPECTED_TRAIN_BAD_RATE or train_bads != EXPECTED_TRAIN_N_BADS:
        raise RuntimeError(
            "Could not reconstruct frozen Train split. "
            f"got n={train_n} bad_rate={train_rate} bads={train_bads}; "
            f"expected n={EXPECTED_TRAIN_N} bad_rate={EXPECTED_TRAIN_BAD_RATE} bads={EXPECTED_TRAIN_N_BADS}"
        )
    if test_n != EXPECTED_TEST_N or test_rate != EXPECTED_TEST_BAD_RATE or test_bads != EXPECTED_TEST_N_BADS:
        raise RuntimeError(
            "Could not reconstruct frozen Test split. "
            f"got n={test_n} bad_rate={test_rate} bads={test_bads}; "
            f"expected n={EXPECTED_TEST_N} bad_rate={EXPECTED_TEST_BAD_RATE} bads={EXPECTED_TEST_N_BADS}"
        )
    if RAW_TARGET_COL in train_df.columns or RAW_TARGET_COL in test_df.columns or RAW_TARGET_COL in feature_cols:
        raise RuntimeError(f"{RAW_TARGET_COL} leaked into the split frame or feature_cols")
    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise RuntimeError("default target missing after load_and_split")
    if TARGET in feature_cols:
        raise RuntimeError("target column present in feature_cols")
    if "gender" not in feature_cols or "gender" not in train_df.columns:
        raise RuntimeError("gender missing from original feature_cols; cannot drop it from X by name")
    if "employment_status" not in feature_cols or "employment_status" not in train_df.columns:
        raise RuntimeError("employment_status missing from original feature_cols")


def _rank_of(feature: str, ranking: list[dict], key: str) -> int | None:
    for i, row in enumerate(ranking, start=1):
        if row["feature"] == feature:
            return i
    return None


def _mean_abs_shap_ranking(shap_arr: np.ndarray, columns: list[str], method: str) -> dict:
    if shap_arr.shape != (shap_arr.shape[0], len(columns)):
        raise RuntimeError(f"Unexpected SHAP shape {shap_arr.shape} vs n_features={len(columns)}")
    mean_abs = np.abs(shap_arr).mean(axis=0)
    ranking = sorted(
        (
            {"feature": col, "mean_abs_shap": round(float(val), 6)}
            for col, val in zip(columns, mean_abs)
        ),
        key=lambda r: r["mean_abs_shap"],
        reverse=True,
    )
    shap_features = [r["feature"] for r in ranking]
    _assert_in_model_columns(shap_features, f"SHAP table ({method})", require_order=False)
    if "gender" in shap_features:
        raise RuntimeError("SHAP table contains gender")
    if "employment_status" not in shap_features:
        raise RuntimeError("SHAP table missing employment_status")
    return {
        "method": method,
        "n_rows": int(shap_arr.shape[0]),
        "mean_abs_shap_by_feature": ranking,
        "employment_status_rank": _rank_of("employment_status", ranking, "mean_abs_shap"),
        "gender_present": False,
        "employment_status_present": True,
    }


def _shap_values_to_2d(shap_values, n_rows: int, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[-1])
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
    if arr.shape != (n_rows, n_features):
        raise RuntimeError(f"Unexpected SHAP array shape {arr.shape}")
    return arr


def compute_tree_shap(model, X_train: pd.DataFrame, context: str) -> dict:
    _assert_in_model_columns(list(X_train.columns), f"{context} SHAP input")
    features = list(X_train.columns)
    tree_error = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        arr = _shap_values_to_2d(shap_values, len(X_train), len(features))
        return _mean_abs_shap_ranking(
            arr, features, f"shap.TreeExplainer (TreeSHAP) on all TRAIN rows — {context}"
        )
    except Exception as exc:  # noqa: BLE001
        tree_error = f"{type(exc).__name__}: {exc}"
        print(f"[shap] TreeExplainer failed for {context}: {tree_error}", flush=True)

    if "catboost" in context.lower():
        try:
            sv = model.get_feature_importance(type="ShapValues", data=X_train)
            sv = np.asarray(sv)
            if sv.ndim != 2 or sv.shape[1] != len(features) + 1:
                raise RuntimeError(f"CatBoost native SHAP shape {sv.shape}")
            arr = sv[:, :-1]
            out = _mean_abs_shap_ranking(
                arr, features, f"CatBoost native ShapValues on all TRAIN rows (TreeExplainer failed: {tree_error})"
            )
            out["tree_explainer_error"] = tree_error
            return out
        except Exception as native_exc:  # noqa: BLE001
            raise RuntimeError(
                f"{context}: both TreeExplainer and CatBoost native SHAP failed. "
                f"TreeExplainer={tree_error}; native={type(native_exc).__name__}: {native_exc}"
            ) from native_exc

    def predict_pd(X):
        if isinstance(X, pd.DataFrame):
            frame = X[features] if set(features).issubset(X.columns) else X
        else:
            frame = pd.DataFrame(np.asarray(X), columns=features)
        return model.predict_proba(frame)[:, 1]

    background = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict_pd, masker)
    sample = X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_STATE)
    explanation = explainer(sample)
    arr = np.asarray(explanation.values)
    if arr.ndim > 2:
        arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
    out = _mean_abs_shap_ranking(
        arr,
        features,
        f"shap.Explainer on TRAIN sample n={len(sample)} (TreeExplainer failed: {tree_error}) — {context}",
    )
    out["tree_explainer_error"] = tree_error
    return out


def compute_lr_coef_ranking(model: LogisticRegression, columns: list[str]) -> dict:
    _assert_in_model_columns(columns, "LR coef ranking")
    coefs = np.asarray(model.coef_[0], dtype=float)
    if len(coefs) != len(columns):
        raise RuntimeError(f"LR coef length {len(coefs)} != n_features {len(columns)}")
    ranking = sorted(
        (
            {
                "feature": col,
                "coef": round(float(c), 6),
                "abs_coef": round(float(abs(c)), 6),
            }
            for col, c in zip(columns, coefs)
        ),
        key=lambda r: r["abs_coef"],
        reverse=True,
    )
    names = [r["feature"] for r in ranking]
    _assert_in_model_columns(names, "LR abs-coef table", require_order=False)
    return {
        "method": "absolute logistic coefficient ranking on TRAIN-fitted WOE columns",
        "intercept": round(float(model.intercept_[0]), 6),
        "abs_coef_by_feature": ranking,
        "employment_status_rank": _rank_of("employment_status", ranking, "abs_coef"),
        "gender_present": False,
        "employment_status_present": True,
    }


def gbdt_params_template() -> dict:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": N_JOBS,
        "subsample_freq": 1,
    }


def dart_params_template() -> dict:
    return {
        "objective": "binary",
        "boosting_type": "dart",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": N_JOBS,
    }


def run_lr_oof(C: float, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    params = {
        "penalty": "l2",
        "C": float(C),
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }
    oof = np.zeros(len(X), dtype=float)
    fold_aucs = []
    fold_ks = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = LogisticRegression(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(float(roc_auc_score(y.iloc[va_idx], pred)))
        fold_ks.append(_ks(y.iloc[va_idx].values, pred))
        print(
            f"    [lr C={C}] fold {fold_i}/{CV_FOLDS}: AUC={fold_aucs[-1]:.6f} KS={fold_ks[-1]:.6f}",
            flush=True,
        )
    return {
        "params": params,
        "oof_preds": oof,
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": _ks(y.values, oof),
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
    }


def run_gbdt_oof(params: dict, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    _assert_no_constraint_params(params, "gbdt params")
    if params.get("boosting_type") != "gbdt":
        raise RuntimeError("gbdt boosting_type must be gbdt")
    oof = np.zeros(len(X), dtype=float)
    best_iters = []
    fold_aucs = []
    fold_ks = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        best_iters.append(_best_iteration_lgbm(model))
        fold_aucs.append(float(roc_auc_score(y_va, pred)))
        fold_ks.append(_ks(y_va.values, pred))
        print(
            f"    [gbdt] fold {fold_i}/{CV_FOLDS}: AUC={fold_aucs[-1]:.6f} "
            f"KS={fold_ks[-1]:.6f} best_iteration={best_iters[-1]}",
            flush=True,
        )
    return {
        "oof_preds": oof,
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": _ks(y.values, oof),
        "best_iterations": best_iters,
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
    }


def run_dart_oof(params: dict, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    _assert_no_constraint_params(params, "dart params")
    if params.get("boosting_type") != "dart":
        raise RuntimeError("dart boosting_type must be dart")
    oof = np.zeros(len(X), dtype=float)
    fold_aucs = []
    fold_ks = []
    n_estimators_used = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = lgb.LGBMClassifier(**params)
        # DART: no eval_set, no early_stopping
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        n_est = int(model.get_params().get("n_estimators"))
        n_estimators_used.append(n_est)
        fold_aucs.append(float(roc_auc_score(y.iloc[va_idx], pred)))
        fold_ks.append(_ks(y.iloc[va_idx].values, pred))
        print(
            f"    [dart] fold {fold_i}/{CV_FOLDS}: AUC={fold_aucs[-1]:.6f} "
            f"KS={fold_ks[-1]:.6f} n_estimators={n_est} (no early_stopping)",
            flush=True,
        )
    return {
        "oof_preds": oof,
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": _ks(y.values, oof),
        "n_estimators_used": n_estimators_used,
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
    }


def run_catboost_oof(params: dict, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    _assert_no_constraint_params(params, "catboost params")
    oof = np.zeros(len(X), dtype=float)
    fold_aucs = []
    fold_ks = []
    iterations_used = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        n_iter = int(model.get_params().get("iterations"))
        iterations_used.append(n_iter)
        fold_aucs.append(float(roc_auc_score(y.iloc[va_idx], pred)))
        fold_ks.append(_ks(y.iloc[va_idx].values, pred))
        print(
            f"    [catboost] fold {fold_i}/{CV_FOLDS}: AUC={fold_aucs[-1]:.6f} "
            f"KS={fold_ks[-1]:.6f} iterations={n_iter}",
            flush=True,
        )
    return {
        "oof_preds": oof,
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": _ks(y.values, oof),
        "iterations_used": iterations_used,
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
    }


def tune_lr(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    print("[train_ensemble] Tuning LR C on TRAIN OOF AUC (log grid)...", flush=True)
    best = None
    grid_scores = []
    for C in LR_C_GRID:
        res = run_lr_oof(C, X, y, folds)
        grid_scores.append({"C": C, "oof_auc": round(res["oof_auc"], 6), "oof_ks": round(res["oof_ks"], 6)})
        print(f"[tune_lr] C={C} OOF AUC={res['oof_auc']:.6f} OOF KS={res['oof_ks']:.6f}", flush=True)
        if best is None or res["oof_auc"] > best["oof_auc"]:
            best = res
    return {"winner": best, "grid_scores": grid_scores}


def tune_gbdt(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    print(f"[train_ensemble] Tuning LightGBM gbdt: Optuna TPE n_trials={N_TRIALS}...", flush=True)
    oof_store: dict[int, dict] = {}

    def objective(trial: optuna.Trial) -> float:
        params = gbdt_params_template()
        params.update(
            {
                "n_estimators": N_ESTIMATORS_GBDT_TUNE,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        )
        print(f"[tune_gbdt] trial {trial.number:03d}/{N_TRIALS - 1} starting", flush=True)
        cv_res = run_gbdt_oof(params, X, y, folds)
        oof_store[trial.number] = {**cv_res, "params": params}
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        trial.set_user_attr("fold_best_iterations", cv_res["best_iterations"])
        print(
            f"[tune_gbdt] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f}",
            flush=True,
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    winner = oof_store[study.best_trial.number]
    n_estimators_final = max(50, int(round(float(np.mean(winner["best_iterations"])))))
    winner["n_estimators_final"] = n_estimators_final
    winner["best_search"] = dict(study.best_params)
    winner["n_trials_completed"] = len(study.trials)
    winner["study_best_value"] = float(study.best_value)
    print(
        f"[tune_gbdt] best OOF AUC={winner['oof_auc']:.6f} "
        f"n_estimators_final={n_estimators_final} best_iterations={winner['best_iterations']}",
        flush=True,
    )
    return winner


def tune_dart(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    print(f"[train_ensemble] Tuning LightGBM DART: Optuna TPE n_trials={N_TRIALS} (NO early_stopping)...", flush=True)
    oof_store: dict[int, dict] = {}

    def objective(trial: optuna.Trial) -> float:
        params = dart_params_template()
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 100, 800),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.3),
                "skip_drop": trial.suggest_float("skip_drop", 0.2, 0.8),
            }
        )
        print(
            f"[tune_dart] trial {trial.number:03d}/{N_TRIALS - 1} starting "
            f"n_estimators={params['n_estimators']}",
            flush=True,
        )
        try:
            cv_res = run_dart_oof(params, X, y, folds)
        except Exception as exc:  # noqa: BLE001
            _write_failure_and_stop(
                "LightGBM DART runtime failed",
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            )
        oof_store[trial.number] = {**cv_res, "params": params}
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        print(
            f"[tune_dart] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f}",
            flush=True,
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    winner = oof_store[study.best_trial.number]
    winner["best_search"] = dict(study.best_params)
    winner["n_trials_completed"] = len(study.trials)
    winner["study_best_value"] = float(study.best_value)
    print(f"[tune_dart] best OOF AUC={winner['oof_auc']:.6f} params={winner['best_search']}", flush=True)
    return winner


def catboost_params_from_trial(trial: optuna.Trial) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": N_JOBS,
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "iterations": trial.suggest_int("iterations", 200, 800),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "bootstrap_type": "Bernoulli",
    }


def tune_catboost(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    print(f"[train_ensemble] Tuning CatBoost: Optuna TPE n_trials={N_TRIALS}...", flush=True)
    oof_store: dict[int, dict] = {}

    def objective(trial: optuna.Trial) -> float:
        params = catboost_params_from_trial(trial)
        print(
            f"[tune_catboost] trial {trial.number:03d}/{N_TRIALS - 1} starting "
            f"iterations={params['iterations']} depth={params['depth']}",
            flush=True,
        )
        try:
            cv_res = run_catboost_oof(params, X, y, folds)
        except Exception as exc:  # noqa: BLE001
            _write_failure_and_stop(
                "CatBoost runtime failed",
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            )
        oof_store[trial.number] = {**cv_res, "params": params}
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        print(
            f"[tune_catboost] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f}",
            flush=True,
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    winner = oof_store[study.best_trial.number]
    winner["best_search"] = dict(study.best_params)
    winner["n_trials_completed"] = len(study.trials)
    winner["study_best_value"] = float(study.best_value)
    print(f"[tune_catboost] best OOF AUC={winner['oof_auc']:.6f} params={winner['best_search']}", flush=True)
    return winner


def main() -> None:
    _require_pipeline_files()
    _assert_load_and_split_unchanged()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    _assert_allowed_writes(ALLOWED_WRITE_PATHS)

    if os.path.exists(os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")):
        print("[train_ensemble] Note: artifacts/lgbm_model.joblib exists and will NOT be overwritten.")

    print("[train_ensemble] Loading frozen split via scripts/train.py::load_and_split (unchanged)...")
    train_df, test_df, feature_cols = load_and_split()
    verify_frozen_split(train_df, test_df, feature_cols)

    in_model_features = build_in_model_features(feature_cols)
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    print(
        f"[train_ensemble] Train n={len(train_df)}, bad_rate={float(y_train.mean()):.4f} "
        f"bads={int(y_train.sum())} | Test n={len(test_df)}, bad_rate={float(y_test.mean()):.4f} "
        f"bads={int(y_test.sum())} (held out; unused for tune/select)"
    )
    print(f"[train_ensemble] Original feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"[train_ensemble] DROPPED from X: {list(DROPPED_FROM_MODEL)}")
    print(f"[train_ensemble] REQUIRED in X: {list(REQUIRED_IN_MODEL)}")
    print(f"[train_ensemble] in_model_features ({len(in_model_features)}): {in_model_features}")
    print("[train_ensemble] gender_in_model=false employment_status_in_model=true monotone_constraints=false")

    encoders = {f: fit_woe_encoder(train_df, f, TARGET, bins=WOE_BINS) for f in in_model_features}
    if "gender" in encoders:
        raise RuntimeError("WOE encoders must not include gender")
    if "employment_status" not in encoders:
        raise RuntimeError("WOE encoders must include employment_status")
    _assert_in_model_columns(list(encoders.keys()), "WOE encoders")

    X_train = encode_woe(train_df, in_model_features, encoders)
    X_test = encode_woe(test_df, in_model_features, encoders)
    _assert_in_model_columns(list(X_train.columns), "X_train")
    _assert_in_model_columns(list(X_test.columns), "X_test")
    if RAW_TARGET_COL in X_train.columns or TARGET in X_train.columns:
        raise RuntimeError("target leaked into WOE feature matrix")

    X_train.to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame({TARGET: y_train.values}).to_csv(Y_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    pd.DataFrame({TARGET: y_test.values}).to_csv(Y_TEST_PATH, index=False)
    joblib.dump(encoders, ENCODERS_PATH)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train, y_train))
    if len(folds) != CV_FOLDS:
        raise RuntimeError(f"expected {CV_FOLDS} folds, got {len(folds)}")

    # ------------------------------------------------------------------
    # Tune four bases on TRAIN OOF only. Same folds. Never use test.
    # ------------------------------------------------------------------
    lr_tune = tune_lr(X_train, y_train, folds)
    try:
        gbdt_tune = tune_gbdt(X_train, y_train, folds)
    except Exception as exc:  # noqa: BLE001
        if "FAILURE written" in str(exc):
            raise
        _write_failure_and_stop("LightGBM gbdt runtime failed", f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
    try:
        dart_tune = tune_dart(X_train, y_train, folds)
    except Exception as exc:  # noqa: BLE001
        if "FAILURE written" in str(exc) or "LightGBM DART" in str(exc):
            raise
        _write_failure_and_stop("LightGBM DART runtime failed", f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
    try:
        cat_tune = tune_catboost(X_train, y_train, folds)
    except Exception as exc:  # noqa: BLE001
        if "FAILURE written" in str(exc) or "CatBoost" in str(exc):
            raise
        _write_failure_and_stop("CatBoost runtime failed", f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

    lr_oof = lr_tune["winner"]["oof_preds"]
    gbdt_oof = gbdt_tune["oof_preds"]
    dart_oof = dart_tune["oof_preds"]
    cat_oof = cat_tune["oof_preds"]
    y_np = y_train.to_numpy()

    oof_base = pd.DataFrame(
        {
            "pd_lr": lr_oof,
            "pd_lgbm_gbdt": gbdt_oof,
            "pd_lgbm_dart": dart_oof,
            "pd_catboost": cat_oof,
        }
    )
    if list(oof_base.columns) != BASE_COL_ORDER:
        raise RuntimeError("OOF base column order mismatch")

    oof_average = oof_base[BASE_COL_ORDER].mean(axis=1).to_numpy()

    stack_meta = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )
    stack_meta.fit(oof_base[BASE_COL_ORDER], y_train)
    oof_stack = stack_meta.predict_proba(oof_base[BASE_COL_ORDER])[:, 1]

    def pack_metrics(pd_vec: np.ndarray) -> dict:
        m = evaluate_discrimination_and_ks(y_np, pd_vec)
        return {
            "AUC": m["AUC"],
            "KS_Statistic": m["KS_Statistic"],
            "Gini": m["Gini"],
            "Validation_Rating": m["Validation_Rating"],
            "auc_raw": float(roc_auc_score(y_np, pd_vec)),
            "ks_raw": _ks(y_np, pd_vec),
        }

    oof_metrics = {
        "lr": pack_metrics(lr_oof),
        "lgbm_gbdt": pack_metrics(gbdt_oof),
        "lgbm_dart": pack_metrics(dart_oof),
        "catboost": pack_metrics(cat_oof),
        "average": pack_metrics(oof_average),
        "stack": pack_metrics(oof_stack),
    }

    avg_auc = float(oof_metrics["average"]["auc_raw"])
    stack_auc = float(oof_metrics["stack"]["auc_raw"])
    if stack_auc > avg_auc:
        oof_selected_ensemble = "stack"
    else:
        oof_selected_ensemble = "average"

    stack_meta_coefs = {
        name: round(float(coef), 6)
        for name, coef in zip(BASE_COL_ORDER, stack_meta.coef_[0])
    }
    stack_meta_intercept = round(float(stack_meta.intercept_[0]), 6)

    print(
        f"[train_ensemble] OOF AUC lr={oof_metrics['lr']['AUC']} "
        f"gbdt={oof_metrics['lgbm_gbdt']['AUC']} dart={oof_metrics['lgbm_dart']['AUC']} "
        f"catboost={oof_metrics['catboost']['AUC']} average={oof_metrics['average']['AUC']} "
        f"stack={oof_metrics['stack']['AUC']}",
        flush=True,
    )
    print(
        f"[train_ensemble] OOF-selected ensemble={oof_selected_ensemble} "
        f"(average {avg_auc:.6f} vs stack {stack_auc:.6f}); meta coefs={stack_meta_coefs}",
        flush=True,
    )

    oof_df = oof_base.copy()
    oof_df["pd_average"] = oof_average
    oof_df["pd_stack"] = oof_stack
    oof_df["y"] = y_np
    oof_df.to_csv(OOF_PDS_PATH, index=False)

    # ------------------------------------------------------------------
    # Refit four bases on ALL 14000 train rows. Keep stack meta on OOF.
    # ------------------------------------------------------------------
    print("[train_ensemble] Refitting four bases on ALL 14000 train rows (no test, no early stopping)...", flush=True)

    lr_full = LogisticRegression(**lr_tune["winner"]["params"])
    lr_full.fit(X_train, y_train)
    lr_names = _feature_names_from_model(lr_full, in_model_features)
    _assert_in_model_columns(lr_names, "fitted LR feature_names_in_")
    if "gender" in lr_names:
        raise RuntimeError("Fitted LR contains gender")

    gbdt_final_params = dict(gbdt_tune["params"])
    gbdt_final_params["n_estimators"] = int(gbdt_tune["n_estimators_final"])
    _assert_no_constraint_params(gbdt_final_params, "gbdt final params")
    gbdt_full = lgb.LGBMClassifier(**gbdt_final_params)
    gbdt_full.fit(X_train, y_train)
    _assert_fitted_no_constraints(gbdt_full, "fitted gbdt")
    gbdt_names = _feature_names_from_model(gbdt_full, in_model_features)
    _assert_in_model_columns(gbdt_names, "fitted gbdt feature_name_")

    dart_final_params = dict(dart_tune["params"])
    _assert_no_constraint_params(dart_final_params, "dart final params")
    if dart_final_params.get("boosting_type") != "dart":
        raise RuntimeError("DART final boosting_type is not dart")
    dart_full = lgb.LGBMClassifier(**dart_final_params)
    dart_full.fit(X_train, y_train)
    _assert_fitted_no_constraints(dart_full, "fitted dart")
    dart_names = _feature_names_from_model(dart_full, in_model_features)
    _assert_in_model_columns(dart_names, "fitted dart feature_name_")

    cat_final_params = dict(cat_tune["params"])
    _assert_no_constraint_params(cat_final_params, "catboost final params")
    cat_full = CatBoostClassifier(**cat_final_params)
    cat_full.fit(X_train, y_train, verbose=False)
    _assert_fitted_no_constraints(cat_full, "fitted catboost")
    cat_names = _feature_names_from_model(cat_full, in_model_features)
    _assert_in_model_columns(cat_names, "fitted catboost feature_names_")

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_ensemble] FREEZE at {freeze_timestamp} (before any test metrics)", flush=True)

    print("[train_ensemble] TRAIN-only explains (LR abs coef; TreeSHAP for trees)...", flush=True)
    lr_explain = compute_lr_coef_ranking(lr_full, in_model_features)
    gbdt_explain = compute_tree_shap(gbdt_full, X_train, "lgbm_gbdt")
    dart_explain = compute_tree_shap(dart_full, X_train, "lgbm_dart")
    cat_explain = compute_tree_shap(cat_full, X_train, "catboost")

    for label, explain in (
        ("lr", lr_explain),
        ("gbdt", gbdt_explain),
        ("dart", dart_explain),
        ("catboost", cat_explain),
    ):
        if explain.get("gender_present") is not False:
            raise RuntimeError(f"{label} explain table claims gender present")
        if explain.get("employment_status_present") is not True:
            raise RuntimeError(f"{label} explain table missing employment_status")

    joblib.dump(lr_full, LR_PATH)
    joblib.dump(gbdt_full, GBDT_PATH)
    joblib.dump(dart_full, DART_PATH)
    joblib.dump(cat_full, CATBOOST_PATH)
    joblib.dump(stack_meta, STACK_META_PATH)

    written = [
        ENCODERS_PATH,
        LR_PATH,
        GBDT_PATH,
        DART_PATH,
        CATBOOST_PATH,
        STACK_META_PATH,
        META_JSON_PATH,
        OOF_PDS_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]
    _assert_allowed_writes(written)
    for p in written:
        base = os.path.basename(p)
        if base == "lgbm_model.joblib" or base.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"accidentally wrote forbidden artifact {p}")

    meta_record = {
        "model_type": "ensemble_no_gender_lr_gbdt_dart_catboost",
        "freeze_timestamp_utc": freeze_timestamp,
        "test_looked_at": False,
        "test_metrics": None,
        "test_labels_used_to_fit_or_select": False,
        "monotone_constraints": False,
        "interaction_constraints": False,
        "gender_in_model": False,
        "employment_status_in_model": True,
        "in_model_features": in_model_features,
        "dropped_from_model": list(DROPPED_FROM_MODEL),
        "n_in_model_features": len(in_model_features),
        "loan_paid_back_in_features": False,
        "iv_drop": False,
        "vif_drop": False,
        "woe_bins": WOE_BINS,
        "cv_folds": CV_FOLDS,
        "optuna_sampler": "TPESampler",
        "optuna_seed": RANDOM_STATE,
        "n_trials_requested": N_TRIALS,
        "bases_trained": ["lr", "lgbm_gbdt", "lgbm_dart", "catboost"],
        "split": {
            "source": "scripts/train.py::load_and_split UNCHANGED",
            "test_size": 0.3,
            "stratify": TARGET,
            "random_state": RANDOM_STATE,
            "train_n": int(len(train_df)),
            "train_bad_rate": round(float(y_train.mean()), 4),
            "train_n_bads": int(y_train.sum()),
            "test_n": int(len(test_df)),
            "test_bad_rate": round(float(y_test.mean()), 4),
            "test_n_bads": int(y_test.sum()),
            "note": "Test labels persisted for eval only; not used to fit or select.",
        },
        "hyperparams": {
            "lr": lr_tune["winner"]["params"],
            "lr_c_grid": LR_C_GRID,
            "lr_grid_scores": lr_tune["grid_scores"],
            "lgbm_gbdt_search": gbdt_tune["best_search"],
            "lgbm_gbdt_final": {k: v for k, v in gbdt_final_params.items()},
            "lgbm_gbdt_n_estimators_final": int(gbdt_tune["n_estimators_final"]),
            "lgbm_gbdt_best_iterations": gbdt_tune["best_iterations"],
            "lgbm_gbdt_n_trials_completed": gbdt_tune["n_trials_completed"],
            "lgbm_dart_search": dart_tune["best_search"],
            "lgbm_dart_final": {k: v for k, v in dart_final_params.items()},
            "lgbm_dart_n_trials_completed": dart_tune["n_trials_completed"],
            "catboost_search": cat_tune["best_search"],
            "catboost_final": {k: v for k, v in cat_final_params.items()},
            "catboost_n_trials_completed": cat_tune["n_trials_completed"],
            "stack_meta": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "fitted_on": "OOF PDs of four bases (not full-train in-sample PDs)",
            },
        },
        "oof_metrics": oof_metrics,
        "oof_auc": {
            "lr": oof_metrics["lr"]["AUC"],
            "lgbm_gbdt": oof_metrics["lgbm_gbdt"]["AUC"],
            "lgbm_dart": oof_metrics["lgbm_dart"]["AUC"],
            "catboost": oof_metrics["catboost"]["AUC"],
            "average": oof_metrics["average"]["AUC"],
            "stack": oof_metrics["stack"]["AUC"],
        },
        "oof_ks": {
            "lr": oof_metrics["lr"]["KS_Statistic"],
            "lgbm_gbdt": oof_metrics["lgbm_gbdt"]["KS_Statistic"],
            "lgbm_dart": oof_metrics["lgbm_dart"]["KS_Statistic"],
            "catboost": oof_metrics["catboost"]["KS_Statistic"],
            "average": oof_metrics["average"]["KS_Statistic"],
            "stack": oof_metrics["stack"]["KS_Statistic"],
        },
        "oof_selected_ensemble": oof_selected_ensemble,
        "stack_meta_coefs": stack_meta_coefs,
        "stack_meta_intercept": stack_meta_intercept,
        "train_explains": {
            "lr": lr_explain,
            "lgbm_gbdt": gbdt_explain,
            "lgbm_dart": dart_explain,
            "catboost": cat_explain,
        },
        "artifact_paths": {
            "encoders": ENCODERS_PATH,
            "lr": LR_PATH,
            "lgbm_gbdt": GBDT_PATH,
            "lgbm_dart": DART_PATH,
            "catboost": CATBOOST_PATH,
            "stack_meta": STACK_META_PATH,
            "meta": META_JSON_PATH,
            "oof_pds": OOF_PDS_PATH,
            "X_train": X_TRAIN_PATH,
            "y_train": Y_TRAIN_PATH,
            "X_test": X_TEST_PATH,
            "y_test": Y_TEST_PATH,
        },
        "did_not_write": [
            "artifacts/lgbm_model.joblib",
            "artifacts/lgbm_linear_tree_*",
            "artifacts/linear_tree_*",
            "artifacts/stack_lr_rf_lgbm_*",
            "artifacts/lgbm_no_gender_emp_overlay_*",
            "artifacts/lgbm_emp_in_no_gender_*",
            "artifacts/lgbm_dart_monotone_*",
        ],
        "notes": (
            "Stack meta is fitted on OOF base PDs only and is NOT refit on "
            "full-train in-sample base PDs. PSI train side must be OOF ensemble PD. "
            "No monotone_constraints. No interaction_constraints. gender dropped. "
            "employment_status is a normal in-model WOE feature."
        ),
    }

    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta_record), fh, ensure_ascii=False, indent=2)

    print(f"[train_ensemble] FREEZE {freeze_timestamp}", flush=True)
    print("[train_ensemble] test_looked_at=false test_metrics=null", flush=True)
    print(f"[train_ensemble] artifacts: {written}", flush=True)
    print("[train_ensemble] STOP — eval script computes test metrics after freeze.", flush=True)


if __name__ == "__main__":
    main()
