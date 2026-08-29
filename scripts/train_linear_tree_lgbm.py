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
