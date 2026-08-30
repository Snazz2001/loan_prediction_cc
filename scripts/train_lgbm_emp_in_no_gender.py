"""
LightGBM-only candidate (train / tune only).

Frozen split via scripts/train.py::load_and_split UNCHANGED. Drops gender from
the in-model matrix. Puts employment_status BACK IN as a normal WOE feature
(not overlay, not policy table, not an interaction-constraint singleton).
Keeps ALL remaining original features (no IV drop, no VIF drop). WOE 5-bin
encoders fit on TRAIN only.

One model: lightgbm.LGBMClassifier. Does not stack. Does not train
LogisticRegression or RandomForest. Not linear_tree. No interaction_constraints.

Optuna TPE (seed=42) maximises TRAIN OOF AUC (KS monitoring only). Freeze is
written with test_looked_at=false / test_metrics=null BEFORE any Test
discrimination metric is computed. Test labels are never used to fit or select.

Does not write last-run artifacts/lgbm_model.joblib, PR #1 linear-tree
artifacts, PR #2 stack artifacts, or PR #3 overlay artifacts.

Usage (from repo root): python3 scripts/train_lgbm_emp_in_no_gender.py
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

from train import load_and_split
from utils.config import ARTIFACTS_DIR, RANDOM_STATE, RAW_TARGET_COL, TARGET, WOE_BINS
from utils.risk_skills import evaluate_discrimination_and_ks, generate_shap_summary
from utils.woe_encoding import apply_woe_encoder, fit_woe_encoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

LAST_RUN_CV_AUC = 0.8862
STACK_OOF_AUC = 0.8891
OVERLAY_OOF_AUC = 0.7094
N_TRIALS = 50
CV_FOLDS = 5
N_ESTIMATORS_TUNE = 1000
EARLY_STOPPING_ROUNDS = 30
EXPECTED_TRAIN_N = 14000
EXPECTED_TRAIN_BAD_RATE = 0.2001
EXPECTED_TEST_N = 6000
EXPECTED_TEST_BAD_RATE = 0.2002

DROPPED_FROM_MODEL = ("gender",)
REQUIRED_IN_MODEL = ("employment_status",)

# Original remaining columns after dropping gender only (original CSV order).
# PR #3's 19 plus employment_status in its native position after monthly_income.
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
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_X_train.csv"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_y_train.csv"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_X_test.csv"),
    os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_y_test.csv"),
]

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_model.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_encoders.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_emp_in_no_gender_y_test.csv")

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
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path in forbidden_set:
            raise RuntimeError(f"Refusing to write forbidden artifact: {path}")
        if abs_path not in allowed_set:
            raise RuntimeError(f"Refusing to write undeclared artifact: {path}")
        base = os.path.basename(path)
        if base == "lgbm_model.joblib" or base.startswith(
            (
                "lgbm_linear_tree_",
                "linear_tree_",
                "stack_lr_rf_lgbm_",
                "lgbm_no_gender_emp_overlay_",
            )
        ):
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


def _best_iteration(model: lgb.LGBMClassifier) -> int:
    bi = getattr(model, "best_iteration_", None)
    if bi is None:
        bi = getattr(model, "best_iteration", None)
    if bi is None or int(bi) <= 0:
        booster = getattr(model, "booster_", None)
        if booster is not None:
            bi = getattr(booster, "best_iteration", None)
    if bi is None or int(bi) <= 0:
        n_est = getattr(model, "n_estimators_", None) or model.get_params().get(
            "n_estimators", N_ESTIMATORS_TUNE
        )
        return int(n_est)
    return int(bi)


def _assert_in_model_columns(columns: list[str], context: str) -> None:
    cols = list(columns)
    if "gender" in cols:
        raise RuntimeError(
            f"{context}: gender must be ABSENT from in-model X / encoders / feature_name_ / SHAP. "
            f"Found gender in {cols}"
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
            f"{context}: in-model column set mismatch. "
            f"got={cols} expected={EXPECTED_IN_MODEL_FEATURES}"
        )


def frozen_params_template(n_features: int) -> dict:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
        "subsample_freq": 1,
        "monotone_constraints": [-1] * n_features,
    }


def encode_woe(df: pd.DataFrame, feature_cols: list[str], encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in feature_cols})


def build_in_model_features(feature_cols: list[str]) -> list[str]:
    in_model = [c for c in feature_cols if c not in DROPPED_FROM_MODEL]
    _assert_in_model_columns(in_model, "build_in_model_features")
    if in_model != EXPECTED_IN_MODEL_FEATURES:
        raise RuntimeError(
            f"in_model_features order != expected original-order list. "
            f"got={in_model} expected={EXPECTED_IN_MODEL_FEATURES}"
        )
    return in_model


def verify_frozen_split(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    train_n = len(train_df)
    test_n = len(test_df)
    train_bad = round(float(train_df[TARGET].mean()), 4)
    test_bad = round(float(test_df[TARGET].mean()), 4)
    if train_n != EXPECTED_TRAIN_N or train_bad != EXPECTED_TRAIN_BAD_RATE:
        raise RuntimeError(
            f"Frozen split mismatch on Train: n={train_n} bad_rate={train_bad} "
            f"(expected n={EXPECTED_TRAIN_N} bad_rate={EXPECTED_TRAIN_BAD_RATE})"
        )
    if test_n != EXPECTED_TEST_N or test_bad != EXPECTED_TEST_BAD_RATE:
        raise RuntimeError(
            f"Frozen split mismatch on Test: n={test_n} bad_rate={test_bad} "
            f"(expected n={EXPECTED_TEST_N} bad_rate={EXPECTED_TEST_BAD_RATE})"
        )
    if RAW_TARGET_COL in train_df.columns or RAW_TARGET_COL in test_df.columns or RAW_TARGET_COL in feature_cols:
        raise RuntimeError(f"{RAW_TARGET_COL} leaked into the split frame or feature_cols")
    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise RuntimeError("default target missing after load_and_split")
    if TARGET in feature_cols:
        raise RuntimeError("target column present in feature_cols")
    if "gender" not in feature_cols:
        raise RuntimeError("gender missing from original feature_cols; cannot drop it from X by name")
    if "gender" not in train_df.columns:
        raise RuntimeError("gender missing from train_df (needed for drop assert)")
    if "employment_status" not in feature_cols:
        raise RuntimeError("employment_status missing from original feature_cols; cannot include it in X")
    if "employment_status" not in train_df.columns:
        raise RuntimeError("employment_status missing from train_df")


def fit_fold(params: dict, X_tr, y_tr, X_va, y_va) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def run_oof_cv(params: dict, X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold):
    oof = np.zeros(len(X), dtype=float)
    best_iters: list[int] = []
    fold_aucs: list[float] = []
    fold_ks: list[float] = []
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
            f"KS={fold_ks[-1]:.6f} best_iteration={best_iters[-1]}",
            flush=True,
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


def compute_train_shap(model: lgb.LGBMClassifier, X_train: pd.DataFrame) -> dict:
    _assert_in_model_columns(list(X_train.columns), "SHAP input")
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]
    mean_abs = np.abs(shap_arr).mean(axis=0)
    ranking = sorted(
        (
            {"feature": col, "mean_abs_shap": round(float(val), 6)}
            for col, val in zip(X_train.columns, mean_abs)
        ),
        key=lambda r: r["mean_abs_shap"],
        reverse=True,
    )
    shap_features = [r["feature"] for r in ranking]
    _assert_in_model_columns(shap_features, "SHAP table")
    if "gender" in shap_features:
        raise RuntimeError("SHAP table contains gender")
    if "employment_status" not in shap_features:
        raise RuntimeError("SHAP table missing employment_status")

    skill_shap: dict
    try:
        shap_sample = X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_STATE)
        skill_shap = generate_shap_summary(model, shap_sample)
        skill_feats = [name for name, _val in skill_shap.get("global_importance_ranking", [])]
        _assert_in_model_columns(skill_feats, "generate_shap_summary")
        skill_shap = {
            "xai_method": skill_shap.get("xai_method"),
            "n_rows": int(len(shap_sample)),
            "global_importance_ranking": skill_shap.get("global_importance_ranking"),
        }
    except Exception as exc:  # noqa: BLE001
        skill_shap = {"error": str(exc)}

    return {
        "method": (
            "shap.TreeExplainer on all TRAIN rows, in-model WOE features only "
            "(non-linear_tree LGBM; path-dependent TreeSHAP). "
            "utils.risk_skills.generate_shap_summary also run on TRAIN."
        ),
        "n_rows": int(len(X_train)),
        "mean_abs_shap_by_feature": ranking,
        "gender_present": False,
        "employment_status_present": True,
        "generate_shap_summary_train": skill_shap,
    }


def main() -> None:
    _require_pipeline_files()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    _assert_allowed_writes(ALLOWED_WRITE_PATHS)

    if os.path.exists(os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")):
        print("[train_lgbm_emp_in] Note: artifacts/lgbm_model.joblib exists and will NOT be overwritten.")

    print("[train_lgbm_emp_in] Loading frozen split via scripts/train.py::load_and_split (unchanged)...")
    train_df, test_df, feature_cols = load_and_split()
    verify_frozen_split(train_df, test_df, feature_cols)

    in_model_features = build_in_model_features(feature_cols)
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    print(
        f"[train_lgbm_emp_in] Train n={len(train_df)}, bad_rate={float(y_train.mean()):.4f} | "
        f"Test n={len(test_df)}, bad_rate={float(y_test.mean()):.4f} "
        "(held out; unused for tune/select)"
    )
    print(f"[train_lgbm_emp_in] Original feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"[train_lgbm_emp_in] DROPPED from X (not encoded into the model): {list(DROPPED_FROM_MODEL)}")
    print(f"[train_lgbm_emp_in] REQUIRED in X (normal WOE feature): {list(REQUIRED_IN_MODEL)}")
    print(f"[train_lgbm_emp_in] in_model_features ({len(in_model_features)}): {in_model_features}")
    print(
        f"[train_lgbm_emp_in] gender_in_model=false employment_status_in_model=true "
        f"n_in_model={len(in_model_features)}"
    )

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
    if list(X_train.columns) != list(in_model_features) or list(X_test.columns) != list(in_model_features):
        raise RuntimeError("WOE matrix column order != in_model_features")

    X_train.to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame({TARGET: y_train.values}).to_csv(Y_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    pd.DataFrame({TARGET: y_test.values}).to_csv(Y_TEST_PATH, index=False)
    joblib.dump(encoders, ENCODERS_PATH)

    n_features = len(in_model_features)
    monotone_constraints = [-1] * n_features
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = frozen_params_template(n_features)
        params.update(
            {
                "n_estimators": N_ESTIMATORS_TUNE,
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
        print(f"[tune] trial {trial.number:03d}/{N_TRIALS - 1} starting", flush=True)
        cv_res = run_oof_cv(params, X_train, y_train, skf)
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        trial.set_user_attr("fold_best_iterations", cv_res["best_iterations"])
        trial.set_user_attr("fold_aucs", cv_res["fold_aucs"])
        print(
            f"[tune] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f} (KS monitoring only)",
            flush=True,
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_search = dict(study.best_params)
    winning_params = frozen_params_template(n_features)
    winning_params.update(best_search)
    winning_params["n_estimators"] = N_ESTIMATORS_TUNE

    print("[train_lgbm_emp_in] Re-running winning trial's 5 train folds to set n_estimators_final...")
    winner_cv = run_oof_cv(winning_params, X_train, y_train, skf)
    n_estimators_final = max(50, int(round(float(np.mean(winner_cv["best_iterations"])))))
    train_oof_auc = round(float(winner_cv["oof_auc"]), 4)
    train_oof_ks = round(float(winner_cv["oof_ks"]), 4)
    beats_last_cv = bool(winner_cv["oof_auc"] > LAST_RUN_CV_AUC)
    beats_stack_oof = bool(winner_cv["oof_auc"] > STACK_OOF_AUC)
    beats_overlay_oof = bool(winner_cv["oof_auc"] > OVERLAY_OOF_AUC)

    print(
        f"[train_lgbm_emp_in] Train OOF AUC={train_oof_auc} (raw={winner_cv['oof_auc']:.6f}) "
        f"vs last CV {LAST_RUN_CV_AUC} -> beat={beats_last_cv}; "
        f"vs stack OOF {STACK_OOF_AUC} -> beat={beats_stack_oof}; "
        f"vs overlay OOF {OVERLAY_OOF_AUC} -> beat={beats_overlay_oof}"
    )
    print(
        f"[train_lgbm_emp_in] fold best_iterations={winner_cv['best_iterations']} "
        f"-> n_estimators_final={n_estimators_final}"
    )

    oof_df = pd.DataFrame(
        {
            "oof_pd": winner_cv["oof_preds"],
            TARGET: y_train.values,
        }
    )
    oof_df.to_csv(OOF_PDS_PATH, index=False)

    final_params = dict(winning_params)
    final_params["n_estimators"] = n_estimators_final
    print(
        f"[train_lgbm_emp_in] Refitting on ALL {len(X_train)} train rows with "
        f"n_estimators={n_estimators_final}, no early stopping, no test, no new holdout."
    )
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    model_names = list(getattr(final_model, "feature_name_", list(X_train.columns)))
    _assert_in_model_columns(model_names, "fitted LGBM feature_name_")
    if "gender" in model_names:
        raise RuntimeError("Fitted model feature_name_ contains gender")
    if "employment_status" not in model_names:
        raise RuntimeError("Fitted model feature_name_ missing employment_status")

    train_refit_prob = final_model.predict_proba(X_train)[:, 1]
    train_refit_metrics = evaluate_discrimination_and_ks(y_train.values, train_refit_prob)

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_lgbm_emp_in] FREEZE at {freeze_timestamp} (before any test metrics)")

    print("[train_lgbm_emp_in] Computing TRAIN-ONLY SHAP on in-model features...")
    shap_exhibits = compute_train_shap(final_model, X_train)

    feature_importances = sorted(
        zip(list(X_train.columns), [int(x) for x in final_model.feature_importances_]),
        key=lambda x: x[1],
        reverse=True,
    )

    meta = {
        "algorithm": "lightgbm.LGBMClassifier",
        "stacked": False,
        "competing_models": [],
        "fixed": {
            "objective": "binary",
            "boosting_type": "gbdt",
            "linear_tree": False,
            "random_state": RANDOM_STATE,
            "monotone_constraints": monotone_constraints,
            "interaction_constraints": None,
            "in_model_features": in_model_features,
            "dropped_from_model": list(DROPPED_FROM_MODEL),
            "gender_in_model": False,
            "employment_status_in_model": True,
            "employment_status_role": "normal_in_model_woe_feature",
            "iv_drop": False,
            "vif_drop": False,
            "loan_paid_back_in_features": False,
            "n_in_model_features": len(in_model_features),
        },
        "search_space": {
            "n_trials": N_TRIALS,
            "cv_folds": CV_FOLDS,
            "objective": "OOF AUC",
            "ks_role": "monitoring only",
            "learning_rate": "log 0.01-0.1",
            "num_leaves": "15-63",
            "max_depth": "3-8",
            "min_child_samples": "20-300",
            "subsample": "0.6-1.0 (subsample_freq=1)",
            "colsample_bytree": "0.6-1.0",
            "reg_alpha": "log 1e-3-10",
            "reg_lambda": "log 1e-3-10",
            "n_estimators_tune": N_ESTIMATORS_TUNE,
            "early_stopping": EARLY_STOPPING_ROUNDS,
            "sampler": "TPESampler",
            "sampler_seed": RANDOM_STATE,
        },
        "best_hyperparameters": _json_ready(best_search),
        "n_estimators_final": n_estimators_final,
        "winning_fold_best_iterations": winner_cv["best_iterations"],
        "winning_fold_aucs": [round(x, 6) for x in winner_cv["fold_aucs"]],
        "winning_fold_ks": [round(x, 6) for x in winner_cv["fold_ks"]],
        "train_oof_auc": train_oof_auc,
        "train_oof_auc_raw": float(winner_cv["oof_auc"]),
        "train_oof_ks_monitoring": train_oof_ks,
        "train_refit_metrics": train_refit_metrics,
        "n_trials_requested": N_TRIALS,
        "n_trials_completed": len(study.trials),
        "feature_importances": [{"feature": f, "importance": int(v)} for f, v in feature_importances],
        "in_model_features": in_model_features,
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
            "stack_oof_auc_comparator": STACK_OOF_AUC,
            "overlay_oof_auc_comparator": OVERLAY_OOF_AUC,
            "train_oof_auc": train_oof_auc,
            "beats_last_cv": beats_last_cv,
            "beats_stack_oof": beats_stack_oof,
            "beats_overlay_oof": beats_overlay_oof,
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
        ],
    }

    joblib.dump(final_model, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta), fh, ensure_ascii=False, indent=2)

    print(f"[train_lgbm_emp_in] Wrote {MODEL_PATH}")
    print(f"[train_lgbm_emp_in] Wrote {META_PATH} (freeze complete; no test metrics inside)")
    print(f"[train_lgbm_emp_in] Wrote {OOF_PDS_PATH}")
    print("[train_lgbm_emp_in] STOPPING. Run python3 scripts/eval_lgbm_emp_in_no_gender.py for test metrics.")


if __name__ == "__main__":
    main()
