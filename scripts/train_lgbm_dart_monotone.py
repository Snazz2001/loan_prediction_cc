"""
LightGBM DART monotone candidate (train / tune only).

Frozen split via scripts/train.py::load_and_split UNCHANGED. Reconstructs the
champion 6-feature WOE method: 5-bin TRAIN-ONLY encoders on the six named
columns using utils.woe_encoding.fit_woe_encoder / apply_woe_encoder and
WOE_BINS from utils.config. Does NOT re-run IV/VIF, does NOT call
screen_features / iterative_vif_filter, does NOT add fields, does NOT invent
a new WOE scheme, does NOT load PR #4 20-col encoders.

One model: lightgbm.LGBMClassifier(boosting_type='dart', objective='cross_entropy').
Does not stack. Does not train LogisticRegression or RandomForest.
employment_status stays IN-MODEL as an additive-only singleton interaction
group. gender is absent. No overlay. No policy table as a model input.
DART: n_estimators is searched; early_stopping is NOT used (incompatible).

Optuna TPE (seed=42, n_trials=50) maximises TRAIN OOF AUC (KS monitoring
only). Freeze is written with test_looked_at=false / test_metrics=null
BEFORE any Test discrimination metric is computed.

Does not write last-run artifacts/lgbm_model.joblib or PR #1–#4 artifact names.

Usage (from repo root): python3 scripts/train_lgbm_dart_monotone.py
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
import shap
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from train import load_and_split
from utils.config import ARTIFACTS_DIR, RANDOM_STATE, RAW_TARGET_COL, TARGET, WOE_BINS
from utils.risk_skills import evaluate_discrimination_and_ks, generate_shap_summary
from utils.woe_encoding import apply_woe_encoder, fit_woe_encoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

LAST_RUN_CV_AUC = 0.8862
N_TRIALS = 50
CV_FOLDS = 5
EXPECTED_TRAIN_N = 14000
EXPECTED_TRAIN_BAD_RATE = 0.2001
EXPECTED_TRAIN_N_BADS = 2801
EXPECTED_TEST_N = 6000
EXPECTED_TEST_BAD_RATE = 0.2002
EXPECTED_TEST_N_BADS = 1201

IN_MODEL_FEATURES = [
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

REQUIRED_PATHS = [
    "data/loan_dataset_20000.csv",
    "scripts/train.py",
    "scripts/test.py",
    "utils/config.py",
    "utils/risk_skills.py",
    "utils/woe_encoding.py",
]
FORBIDDEN_WRITE_PATHS = [
    os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib"),
    os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_model.joblib"),
    os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_meta.json"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_run_report.md"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_test_metrics.json"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_woe_encoders.joblib"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_feature_selection.json"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_X_train.csv"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_y_train.csv"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_X_test.csv"),
    os.path.join(ARTIFACTS_DIR, "linear_tree_y_test.csv"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_bases.joblib"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_meta.joblib"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_encoders.joblib"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_meta.json"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_oof_pds.csv"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_test_metrics.json"),
    os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_run_report.md"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_model.joblib"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_encoders.joblib"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_meta.json"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_overlay.json"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_run_report.md"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_test_metrics.json"),
    os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_model.joblib"),
    os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_encoders.joblib"),
    os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_meta.json"),
    os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_run_report.md"),
    os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_test_metrics.json"),
]

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_model.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_encoders.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_y_test.csv")

ALLOWED_WRITE_PATHS = [
    MODEL_PATH,
    ENCODERS_PATH,
    META_PATH,
    OOF_PDS_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
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
    forbidden_set = {os.path.abspath(p) for p in FORBIDDEN_WRITE_PATHS}
    allowed_set = {os.path.abspath(p) for p in ALLOWED_WRITE_PATHS}
    forbidden_prefixes = (
        "lgbm_linear_tree_",
        "linear_tree_",
        "stack_lr_rf_lgbm_",
        "lgbm_no_gender_emp_overlay_",
        "lgbm_emp_in_no_gender_",
    )
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path in forbidden_set:
            raise RuntimeError(f"Refusing to write forbidden artifact: {path}")
        if abs_path not in allowed_set:
            raise RuntimeError(f"Refusing to write undeclared artifact: {path}")
        base = os.path.basename(path)
        if base == "lgbm_model.joblib" or base.startswith(forbidden_prefixes):
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


def _assert_in_model_columns(columns: list[str], context: str) -> None:
    cols = list(columns)
    if cols != IN_MODEL_FEATURES:
        raise RuntimeError(
            f"{context}: expected exact in-model order {IN_MODEL_FEATURES}, got {cols}"
        )
    if "gender" in cols:
        raise RuntimeError(f"{context}: gender must be ABSENT. Found gender in {cols}")
    if "employment_status" not in cols:
        raise RuntimeError(f"{context}: employment_status must be PRESENT. columns={cols}")
    if RAW_TARGET_COL in cols:
        raise RuntimeError(f"{context}: {RAW_TARGET_COL} must never be a feature. columns={cols}")
    if TARGET in cols:
        raise RuntimeError(f"{context}: target {TARGET} present in feature columns: {cols}")


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
    if RAW_TARGET_COL in feature_cols or RAW_TARGET_COL in train_df.columns or RAW_TARGET_COL in test_df.columns:
        raise RuntimeError("loan_paid_back leaked into load_and_split output")
    missing = [f for f in IN_MODEL_FEATURES if f not in train_df.columns]
    if missing:
        raise RuntimeError(f"Named in-model columns missing from split frame: {missing}")


def encode_woe(df: pd.DataFrame, features: list[str], encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in features})


def frozen_params_template() -> dict:
    return {
        "boosting_type": "dart",
        "objective": "cross_entropy",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": 1,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "interaction_constraints": INTERACTION_CONSTRAINTS,
    }


def fit_fold(params: dict, X_tr, y_tr) -> lgb.LGBMClassifier:
    """Fit one DART fold. No eval_set, no early_stopping (incompatible with DART)."""
    if params.get("boosting_type") != "dart":
        raise RuntimeError("boosting_type must be dart")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)
    return model


def run_oof_cv(params: dict, X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold):
    oof = np.zeros(len(X), dtype=float)
    fold_aucs = []
    fold_ks = []
    n_estimators_used = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = fit_fold(params, X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        n_est = int(model.get_params().get("n_estimators"))
        n_estimators_used.append(n_est)
        fold_aucs.append(float(roc_auc_score(y_va, pred)))
        fold_ks.append(_ks(y_va.values, pred))
        print(
            f"    fold {fold_i}/{CV_FOLDS}: AUC={fold_aucs[-1]:.6f} "
            f"KS={fold_ks[-1]:.6f} n_estimators={n_est} (no early_stopping)",
            flush=True,
        )
    oof_auc = float(roc_auc_score(y, oof))
    oof_ks = _ks(y.values, oof)
    return {
        "oof_preds": oof,
        "oof_auc": oof_auc,
        "oof_ks": oof_ks,
        "n_estimators_used": n_estimators_used,
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
    }


def compute_train_shap(model: lgb.LGBMClassifier, X_train: pd.DataFrame) -> dict:
    """
    Train-only mean |SHAP| for the 6 in-model features.
    Prefer TreeExplainer; if invalid for this DART booster, fall back to a
    model-agnostic SHAP explainer still computed on TRAIN only.
    """
    features = list(X_train.columns)
    _assert_in_model_columns(features, "SHAP X_train")
    method = None
    mean_abs = None
    shap_error = None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[-1])
        else:
            arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
        if arr.shape != (len(X_train), len(features)):
            raise RuntimeError(f"Unexpected TreeSHAP shape {arr.shape}")
        mean_abs = np.abs(arr).mean(axis=0)
        method = "shap.TreeExplainer (TreeSHAP) on all TRAIN rows"
    except Exception as exc:  # noqa: BLE001
        shap_error = f"TreeExplainer failed for DART: {type(exc).__name__}: {exc}"
        print(f"[shap] {shap_error}", flush=True)

        def predict_pd(X):
            if isinstance(X, pd.DataFrame):
                frame = X[features] if set(features).issubset(X.columns) else X
            else:
                frame = pd.DataFrame(np.asarray(X), columns=features)
            return model.predict_proba(frame)[:, 1]

        background = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.Explainer(predict_pd, masker)
        explanation = explainer(X_train)
        arr = np.asarray(explanation.values)
        if arr.ndim > 2:
            arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
        mean_abs = np.abs(arr).mean(axis=0)
        method = (
            "shap.Explainer(predict_proba, Independent(train_background=200)) on all "
            "TRAIN rows — TreeExplainer was invalid for this DART model"
        )

    mean_abs_rows = [
        {"feature": f, "mean_abs_shap": round(float(v), 6)}
        for f, v in sorted(zip(features, mean_abs), key=lambda x: x[1], reverse=True)
    ]
    if "gender" in [r["feature"] for r in mean_abs_rows]:
        raise RuntimeError("SHAP table contains gender")
    emp_rank = next(i for i, r in enumerate(mean_abs_rows, start=1) if r["feature"] == "employment_status")
    emp_val = next(r["mean_abs_shap"] for r in mean_abs_rows if r["feature"] == "employment_status")
    other_vals = [r["mean_abs_shap"] for r in mean_abs_rows if r["feature"] != "employment_status"]
    max_other = max(other_vals) if other_vals else 0.0
    collapsed_to_employment_lookup = bool(
        emp_rank == 1 and (emp_val > 0) and (max_other <= max(1e-4, 0.05 * emp_val))
    )

    skill_shap = None
    try:
        skill_shap = generate_shap_summary(
            model, X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_STATE)
        )
        skill_shap_features = [k for k, _ in skill_shap.get("global_importance_ranking", [])]
        if "gender" in skill_shap_features:
            raise RuntimeError("generate_shap_summary ranking contains gender")
    except Exception as exc:  # noqa: BLE001
        skill_shap = {"error": str(exc)}

    return {
        "method": method,
        "n_rows": int(len(X_train)),
        "mean_abs_shap_by_feature": mean_abs_rows,
        "employment_status_rank": emp_rank,
        "collapsed_to_employment_lookup": collapsed_to_employment_lookup,
        "collapse_note": (
            "Other 5 features have ~0 mean |SHAP| relative to employment_status "
            "(max other <= max(1e-4, 5% of emp)). Singleton interaction + monotone "
            "can collapse DART to an employment lookup (same risk as PR #1). Not 'fixed'."
            if collapsed_to_employment_lookup
            else "Not collapsed: remaining features retain non-trivial mean |SHAP|."
        ),
        "tree_explainer_error": shap_error,
        "generate_shap_summary_train": skill_shap,
    }


def main() -> None:
    _require_pipeline_files()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    _assert_allowed_writes(ALLOWED_WRITE_PATHS)

    if os.path.exists(os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")):
        print("[train_lgbm_dart] Note: artifacts/lgbm_model.joblib exists and will NOT be overwritten.")

    print("[train_lgbm_dart] Loading frozen split via scripts/train.py::load_and_split (unchanged)...")
    train_df, test_df, feature_cols = load_and_split()
    verify_frozen_split(train_df, test_df, feature_cols)

    if "gender" in IN_MODEL_FEATURES:
        raise RuntimeError("gender must not be in IN_MODEL_FEATURES")
    if IN_MODEL_FEATURES[0] != "employment_status":
        raise RuntimeError("employment_status must be first in the 6-column order")
    if list(MONOTONE_CONSTRAINTS) != [-1] * 6:
        raise RuntimeError("monotone_constraints must be six -1 aligned to the 6-column order")
    if INTERACTION_CONSTRAINTS != [
        ["employment_status"],
        [
            "debt_to_income_ratio",
            "interest_rate",
            "grade_subgrade",
            "delinquency_history",
            "num_of_delinquencies",
        ],
    ]:
        raise RuntimeError("interaction_constraints must not be relaxed")

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    print(
        f"[train_lgbm_dart] Train n={len(train_df)}, bad_rate={float(y_train.mean()):.4f} "
        f"(bads={int(y_train.sum())}) | Test n={len(test_df)}, "
        f"bad_rate={float(y_test.mean()):.4f} (bads={int(y_test.sum())}; held out; unused for tune/select)"
    )
    print(f"[train_lgbm_dart] in_model_features (exact order): {IN_MODEL_FEATURES}")
    print("[train_lgbm_dart] gender_in_model=false employment_status_in_model=true boosting_type=dart")
    print("[train_lgbm_dart] WOE 5-bin TRAIN-ONLY on the 6 named columns; no IV/VIF, no extra fields")

    encoders = {f: fit_woe_encoder(train_df, f, TARGET, bins=WOE_BINS) for f in IN_MODEL_FEATURES}
    _assert_in_model_columns(list(encoders.keys()), "WOE encoders")
    if "gender" in encoders:
        raise RuntimeError("WOE encoders must not include gender")

    X_train = encode_woe(train_df, IN_MODEL_FEATURES, encoders)
    X_test = encode_woe(test_df, IN_MODEL_FEATURES, encoders)
    _assert_in_model_columns(list(X_train.columns), "X_train")
    _assert_in_model_columns(list(X_test.columns), "X_test")
    if RAW_TARGET_COL in X_train.columns or RAW_TARGET_COL in X_test.columns:
        raise RuntimeError("loan_paid_back present in encoded feature matrix")

    X_train.to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame({TARGET: y_train.values}).to_csv(Y_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    pd.DataFrame({TARGET: y_test.values}).to_csv(Y_TEST_PATH, index=False)
    joblib.dump(encoders, ENCODERS_PATH)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = frozen_params_template()
        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 100, 800),
                # sklearn alias of LightGBM feature_fraction; do not also pass feature_fraction
                # (alias clash with default colsample_bytree) and do not search subsample.
                "colsample_bytree": feature_fraction,
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.3),
                "skip_drop": trial.suggest_float("skip_drop", 0.2, 0.8),
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            }
        )
        print(
            f"[tune] trial {trial.number:03d}/{N_TRIALS - 1} starting "
            f"n_estimators={params['n_estimators']} lr={params['learning_rate']:.5f} "
            f"drop_rate={params['drop_rate']:.4f} skip_drop={params['skip_drop']:.4f}",
            flush=True,
        )
        cv_res = run_oof_cv(params, X_train, y_train, skf)
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        trial.set_user_attr("fold_aucs", cv_res["fold_aucs"])
        trial.set_user_attr("fold_ks", cv_res["fold_ks"])
        trial.set_user_attr("n_estimators", params["n_estimators"])
        trial.set_user_attr("feature_fraction", feature_fraction)
        print(
            f"[tune] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f} (KS monitoring only)",
            flush=True,
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    if len(study.trials) != N_TRIALS:
        raise RuntimeError(f"Expected {N_TRIALS} trials, completed {len(study.trials)}")

    best_search = dict(study.best_params)
    winning_params = frozen_params_template()
    winning_params.update(
        {
            "learning_rate": best_search["learning_rate"],
            "num_leaves": int(best_search["num_leaves"]),
            "max_depth": int(best_search["max_depth"]),
            "min_child_samples": int(best_search["min_child_samples"]),
            "colsample_bytree": float(best_search["feature_fraction"]),
            "reg_alpha": best_search["reg_alpha"],
            "reg_lambda": best_search["reg_lambda"],
            "drop_rate": best_search["drop_rate"],
            "skip_drop": best_search["skip_drop"],
            "n_estimators": int(best_search["n_estimators"]),
        }
    )

    print(
        f"[train_lgbm_dart] Re-running winning trial 5-fold OOF on TRAIN "
        f"(n_estimators={winning_params['n_estimators']}, no early_stopping, no test)..."
    )
    winner_cv = run_oof_cv(winning_params, X_train, y_train, skf)
    n_estimators_final = int(winning_params["n_estimators"])
    train_oof_auc = round(float(winner_cv["oof_auc"]), 4)
    train_oof_ks = round(float(winner_cv["oof_ks"]), 4)
    beats_last_cv = bool(winner_cv["oof_auc"] > LAST_RUN_CV_AUC)

    print(
        f"[train_lgbm_dart] Train OOF AUC={train_oof_auc} (raw={winner_cv['oof_auc']:.6f}) "
        f"vs last CV {LAST_RUN_CV_AUC} -> beat={beats_last_cv}"
    )
    print(f"[train_lgbm_dart] n_estimators_final (searched, not early-stopped)={n_estimators_final}")

    oof_df = pd.DataFrame({"oof_pd": winner_cv["oof_preds"], TARGET: y_train.values})
    oof_df.to_csv(OOF_PDS_PATH, index=False)

    print(
        f"[train_lgbm_dart] Refitting on ALL {len(X_train)} train rows with "
        f"winning params including n_estimators={n_estimators_final}. "
        "No test until freeze. No new holdout. No early_stopping."
    )
    final_model = lgb.LGBMClassifier(**winning_params)
    final_model.fit(X_train, y_train)
    fitted_boosting = final_model.get_params().get("boosting_type")
    if fitted_boosting != "dart":
        raise RuntimeError(f"Fitted boosting_type is {fitted_boosting}, expected dart")
    if final_model.get_params().get("objective") != "cross_entropy":
        raise RuntimeError(f"Fitted objective is {final_model.get_params().get('objective')}, expected cross_entropy")
    model_names = list(getattr(final_model, "feature_name_", list(X_train.columns)))
    _assert_in_model_columns(model_names, "fitted LGBM feature_name_")

    train_refit_prob = final_model.predict_proba(X_train)[:, 1]
    train_refit_metrics = evaluate_discrimination_and_ks(y_train.values, train_refit_prob)

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_lgbm_dart] FREEZE at {freeze_timestamp} (before any test metrics)")

    print("[train_lgbm_dart] Computing TRAIN-ONLY TreeSHAP / fallback SHAP on 6 features...")
    shap_exhibits = compute_train_shap(final_model, X_train)
    shap_features = [r["feature"] for r in shap_exhibits["mean_abs_shap_by_feature"]]
    if "gender" in shap_features:
        raise RuntimeError("SHAP ranking contains gender")
    print(
        f"[train_lgbm_dart] employment_status SHAP rank={shap_exhibits['employment_status_rank']} "
        f"collapsed_to_employment_lookup={shap_exhibits['collapsed_to_employment_lookup']}"
    )

    feature_importances = sorted(
        zip(list(X_train.columns), [int(x) for x in final_model.feature_importances_]),
        key=lambda x: x[1],
        reverse=True,
    )

    best_hyperparameters = {
        "learning_rate": best_search["learning_rate"],
        "num_leaves": int(best_search["num_leaves"]),
        "max_depth": int(best_search["max_depth"]),
        "min_child_samples": int(best_search["min_child_samples"]),
        "feature_fraction": float(best_search["feature_fraction"]),
        "colsample_bytree": float(best_search["feature_fraction"]),
        "reg_alpha": best_search["reg_alpha"],
        "reg_lambda": best_search["reg_lambda"],
        "drop_rate": best_search["drop_rate"],
        "skip_drop": best_search["skip_drop"],
        "n_estimators": n_estimators_final,
    }

    meta = {
        "algorithm": "lightgbm.LGBMClassifier",
        "stacked": False,
        "competing_models": [],
        "boosting_type": "dart",
        "early_stopping_used": False,
        "fixed": {
            "objective": "cross_entropy",
            "boosting_type": "dart",
            "linear_tree": False,
            "random_state": RANDOM_STATE,
            "monotone_constraints": MONOTONE_CONSTRAINTS,
            "interaction_constraints": INTERACTION_CONSTRAINTS,
            "in_model_features": IN_MODEL_FEATURES,
            "dropped_from_model": ["gender"],
            "gender_in_model": False,
            "employment_status_in_model": True,
            "employment_status_role": "additive_only_singleton_interaction_group",
            "iv_drop": False,
            "vif_drop": False,
            "loan_paid_back_in_features": False,
            "n_in_model_features": 6,
            "early_stopping": False,
        },
        "search_space": {
            "n_trials": N_TRIALS,
            "cv_folds": CV_FOLDS,
            "objective": "OOF AUC",
            "ks_role": "monitoring only",
            "learning_rate": "log 0.01-0.1",
            "num_leaves": "16-64",
            "max_depth": "3-6",
            "min_child_samples": "100-800 (min_data_in_leaf)",
            "feature_fraction": "0.5-1.0 (passed as sklearn colsample_bytree)",
            "reg_alpha": "log 1e-3-10",
            "reg_lambda": "log 1e-3-10",
            "drop_rate": "0.05-0.3",
            "skip_drop": "0.2-0.8",
            "n_estimators": "200-800 (searched; DART has no early_stopping)",
            "early_stopping": False,
            "sampler": "TPESampler",
            "sampler_seed": RANDOM_STATE,
            "subsample_searched": False,
        },
        "best_hyperparameters": _json_ready(best_hyperparameters),
        "n_estimators": n_estimators_final,
        "n_estimators_final": n_estimators_final,
        "winning_fold_n_estimators": winner_cv["n_estimators_used"],
        "winning_fold_aucs": [round(x, 6) for x in winner_cv["fold_aucs"]],
        "winning_fold_ks": [round(x, 6) for x in winner_cv["fold_ks"]],
        "train_oof_auc": train_oof_auc,
        "train_oof_auc_raw": float(winner_cv["oof_auc"]),
        "train_oof_ks_monitoring": train_oof_ks,
        "train_refit_metrics": train_refit_metrics,
        "n_trials_requested": N_TRIALS,
        "n_trials_completed": len(study.trials),
        "feature_importances": [{"feature": f, "importance": int(v)} for f, v in feature_importances],
        "in_model_features": IN_MODEL_FEATURES,
        "gender_in_model": False,
        "employment_status_in_model": True,
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
        "freeze": {
            "timestamp_utc": freeze_timestamp,
            "selection_rule": "higher train OOF AUC (test not used)",
            "last_run_cv_auc": LAST_RUN_CV_AUC,
            "train_oof_auc": train_oof_auc,
            "beats_last_cv": beats_last_cv,
            "test_looked_at": False,
            "test_metrics": None,
            "test_labels_used_to_fit_or_select": False,
        },
        "train_shap": shap_exhibits,
        "artifact_paths": {
            "model": MODEL_PATH,
            "encoders": ENCODERS_PATH,
            "meta": META_PATH,
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
        ],
    }

    joblib.dump(final_model, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta), fh, ensure_ascii=False, indent=2)

    print(f"[train_lgbm_dart] Wrote {MODEL_PATH}")
    print(f"[train_lgbm_dart] Wrote {META_PATH} (freeze complete; no test metrics inside)")
    print(f"[train_lgbm_dart] Wrote {OOF_PDS_PATH}")
    print("[train_lgbm_dart] STOPPING. Run python3 scripts/eval_lgbm_dart_monotone.py for test metrics.")


if __name__ == "__main__":
    main()
