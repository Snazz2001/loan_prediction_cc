"""
Leak-free stacking candidate (train / tune only).

Bases: sklearn LogisticRegression, RandomForestClassifier, lightgbm.LGBMClassifier.
Meta: LogisticRegression on 5-fold OOF base PDs.

FROZEN split via scripts/train.py load_and_split UNCHANGED. All features kept
(no IV-drop, no VIF-drop). WOE 5-bin encoders fit on TRAIN only.

Optuna TPE (seed=42) maximises TRAIN OOF stack AUC. Freeze is written with
test_looked_at=false / test_metrics=null BEFORE any Test discrimination metric
is computed. Test labels are never used to fit or select.

Does not write artifacts/lgbm_model.joblib or PR #1 linear-tree artifacts.

Usage (from repo root): python3 scripts/train_stack_lr_rf_lgbm.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import load_and_split
from utils.config import ARTIFACTS_DIR, RANDOM_STATE, RAW_TARGET_COL, TARGET, WOE_BINS
from utils.risk_skills import calculate_woe_iv, evaluate_discrimination_and_ks
from utils.woe_encoding import apply_woe_encoder, fit_woe_encoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

LAST_RUN_CV_AUC = 0.8862
N_TRIALS = 30
CV_FOLDS = 5
N_ESTIMATORS_TUNE = 1000
EARLY_STOPPING_ROUNDS = 30
META_C_GRID = [0.1, 0.5, 1.0, 2.0, 10.0]

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
]

BASES_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_bases.joblib")
META_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_meta.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_encoders.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_oof_pds.csv")


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
    return obj


def _encode_woe(df: pd.DataFrame, feature_cols: list, encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in feature_cols})


def _lr_params(C: float) -> dict:
    return {
        "penalty": "l2",
        "C": float(C),
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }


def _rf_params(n_estimators: int, max_depth: int, min_samples_leaf: int) -> dict:
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }


def _lgbm_params(
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    n_estimators: int = N_ESTIMATORS_TUNE,
) -> dict:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
    }


def _fit_meta(oof_pds: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    meta = LogisticRegression(**_lr_params(C))
    meta.fit(oof_pds, y)
    return meta


def generate_oof_base_pds(
    X: pd.DataFrame,
    y: pd.Series,
    lr_params: dict,
    rf_params: dict,
    lgbm_params: dict,
) -> tuple[np.ndarray, list[int]]:
    """5-fold OOF PDs for the three bases. Never sees test."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X), 3), dtype=np.float64)
    lgbm_best_iterations: list[int] = []
    y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_arr), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        lr = LogisticRegression(**lr_params)
        lr.fit(X_tr, y_tr)
        oof[va_idx, 0] = lr.predict_proba(X_va)[:, 1]

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr, y_tr)
        oof[va_idx, 1] = rf.predict_proba(X_va)[:, 1]

        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[va_idx, 2] = lgbm.predict_proba(X_va)[:, 1]
        bi = getattr(lgbm, "best_iteration_", None)
        if bi is None or int(bi) <= 0:
            bi = int(lgbm_params.get("n_estimators", N_ESTIMATORS_TUNE))
        lgbm_best_iterations.append(int(bi))
        print(
            f"    fold {fold}/{CV_FOLDS}: lgbm_best_iteration={int(bi)}",
            flush=True,
        )

    return oof, lgbm_best_iterations


def _stack_auc_from_oof(oof_pds: np.ndarray, y: np.ndarray, meta_C: float) -> float:
    meta = _fit_meta(oof_pds, y, meta_C)
    pred = meta.predict_proba(oof_pds)[:, 1]
    return float(roc_auc_score(y, pred))


def main() -> None:
    _require_pipeline_files()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Frozen split (scripts/train.py load_and_split UNCHANGED)
    # ------------------------------------------------------------------
    train_df, test_df, feature_cols = load_and_split()
    train_n = len(train_df)
    test_n = len(test_df)
    train_bad_rate = round(float(train_df[TARGET].mean()), 4)
    test_bad_rate = round(float(test_df[TARGET].mean()), 4)

    if RAW_TARGET_COL in train_df.columns or RAW_TARGET_COL in feature_cols:
        raise RuntimeError(f"{RAW_TARGET_COL} leaked into the feature matrix / train frame")
    if TARGET in feature_cols:
        raise RuntimeError("target column present in feature_cols")
    if train_n != 14000 or train_bad_rate != 0.2001:
        raise RuntimeError(f"Train split mismatch: n={train_n} bad_rate={train_bad_rate}")
    if test_n != 6000 or test_bad_rate != 0.2002:
        raise RuntimeError(f"Test split mismatch: n={test_n} bad_rate={test_bad_rate}")

    # Split verified. Drop test frame so it cannot be used for fit/select.
    del test_df

    y_train = train_df[TARGET]
    y_np = y_train.to_numpy()

    # ------------------------------------------------------------------
    # 2. Diagnostic IV table (TRAIN only). Keep ALL features — do not drop.
    # ------------------------------------------------------------------
    iv_records = []
    for f in feature_cols:
        res = calculate_woe_iv(train_df, f, TARGET, bins=WOE_BINS)
        iv_records.append(
            {
                "feature": res["feature"],
                "iv": res["iv"],
                "predictive_power": res["predictive_power"],
            }
        )
    iv_table = pd.DataFrame(iv_records).sort_values("iv", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. WOE encoders fit on TRAIN only, applied to TRAIN only here
    # ------------------------------------------------------------------
    encoders = {f: fit_woe_encoder(train_df, f, TARGET, bins=WOE_BINS) for f in feature_cols}
    X_train = _encode_woe(train_df, feature_cols, encoders)
    if RAW_TARGET_COL in X_train.columns or TARGET in X_train.columns:
        raise RuntimeError("target leaked into WOE feature matrix")
    if list(X_train.columns) != list(feature_cols):
        raise RuntimeError("WOE feature matrix column order != feature_cols")

    print(
        f"[train_stack] Train n={train_n} bad_rate={train_bad_rate}; "
        f"Test n={test_n} bad_rate={test_bad_rate} (held out, unused for fit/select)",
        flush=True,
    )
    print(
        f"[train_stack] Features kept (no IV/VIF drop): {len(feature_cols)} -> {feature_cols}",
        flush=True,
    )
    print(f"[train_stack] loan_paid_back in features: {False}", flush=True)

    # ------------------------------------------------------------------
    # 4. Optuna TPE on TRAIN OOF stack AUC only
    # ------------------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        lr_params = _lr_params(trial.suggest_float("lr_C", 0.01, 100.0, log=True))
        rf_params = _rf_params(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 500),
            max_depth=trial.suggest_int("rf_max_depth", 4, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 20, 100),
        )
        lgbm_params = _lgbm_params(
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 15, 63),
            max_depth=trial.suggest_int("lgb_max_depth", 3, 8),
        )
        print(f"[train_stack] trial {trial.number + 1}/{N_TRIALS} starting", flush=True)
        oof, best_iters = generate_oof_base_pds(X_train, y_train, lr_params, rf_params, lgbm_params)
        score = _stack_auc_from_oof(oof, y_np, meta_C=1.0)
        trial.set_user_attr("lgbm_best_iterations", best_iters)
        trial.set_user_attr("per_base_oof_auc", {
            "lr": float(roc_auc_score(y_np, oof[:, 0])),
            "rf": float(roc_auc_score(y_np, oof[:, 1])),
            "lgbm": float(roc_auc_score(y_np, oof[:, 2])),
        })
        print(
            f"[train_stack] trial {trial.number + 1}/{N_TRIALS} OOF stack AUC={score:.6f}",
            flush=True,
        )
        return score

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = dict(study.best_params)
    print(
        f"[train_stack] Optuna done: {len(study.trials)} trials, "
        f"best TRAIN OOF stack AUC={study.best_value:.6f}",
        flush=True,
    )
    print(f"[train_stack] best_params={best_params}", flush=True)

    # ------------------------------------------------------------------
    # 5. Recompute OOF PDs with winning base hyperparams (TRAIN only)
    # ------------------------------------------------------------------
    win_lr = _lr_params(best_params["lr_C"])
    win_rf = _rf_params(
        best_params["rf_n_estimators"],
        best_params["rf_max_depth"],
        best_params["rf_min_samples_leaf"],
    )
    win_lgbm_tune = _lgbm_params(
        best_params["lgb_learning_rate"],
        best_params["lgb_num_leaves"],
        best_params["lgb_max_depth"],
        n_estimators=N_ESTIMATORS_TUNE,
    )
    print("[train_stack] recomputing OOF PDs with winning recipe", flush=True)
    oof_pds, lgbm_best_iterations = generate_oof_base_pds(
        X_train, y_train, win_lr, win_rf, win_lgbm_tune
    )
    mean_best_iteration = float(np.mean(lgbm_best_iterations))
    n_estimators_final = max(50, int(round(mean_best_iteration)))

    per_base_oof = {
        "lr": evaluate_discrimination_and_ks(y_np, oof_pds[:, 0]),
        "rf": evaluate_discrimination_and_ks(y_np, oof_pds[:, 1]),
        "lgbm": evaluate_discrimination_and_ks(y_np, oof_pds[:, 2]),
    }

    # Tiny meta-C grid on the already-built OOF PD matrix (TRAIN y only)
    meta_c_scores = {}
    best_meta_c = 1.0
    best_meta_auc = -1.0
    for c in META_C_GRID:
        auc_c = _stack_auc_from_oof(oof_pds, y_np, c)
        meta_c_scores[str(c)] = round(auc_c, 6)
        if auc_c > best_meta_auc:
            best_meta_auc = auc_c
            best_meta_c = c

    meta = _fit_meta(oof_pds, y_np, best_meta_c)
    train_stack_pd = meta.predict_proba(oof_pds)[:, 1]
    oof_stack_metrics = evaluate_discrimination_and_ks(y_np, train_stack_pd)
    oof_stack_auc = float(oof_stack_metrics["AUC"])
    beat_last_cv = bool(oof_stack_auc > LAST_RUN_CV_AUC)

    print(
        f"[train_stack] per-base OOF AUC lr={per_base_oof['lr']['AUC']} "
        f"rf={per_base_oof['rf']['AUC']} lgbm={per_base_oof['lgbm']['AUC']}",
        flush=True,
    )
    print(
        f"[train_stack] OOF stack AUC={oof_stack_metrics['AUC']} "
        f"(beat last-run CV {LAST_RUN_CV_AUC}? {'YES' if beat_last_cv else 'NO'}) "
        f"meta_C={best_meta_c} n_estimators_final={n_estimators_final}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 6. Refit three bases on ALL 14000 train rows. KEEP meta fitted on OOF.
    # ------------------------------------------------------------------
    win_lgbm_final = _lgbm_params(
        best_params["lgb_learning_rate"],
        best_params["lgb_num_leaves"],
        best_params["lgb_max_depth"],
        n_estimators=n_estimators_final,
    )
    lr_full = LogisticRegression(**win_lr)
    lr_full.fit(X_train, y_train)
    rf_full = RandomForestClassifier(**win_rf)
    rf_full.fit(X_train, y_train)
    lgbm_full = lgb.LGBMClassifier(**win_lgbm_final)
    lgbm_full.fit(X_train, y_train)

    oof_df = pd.DataFrame(
        {
            "pd_lr": oof_pds[:, 0],
            "pd_rf": oof_pds[:, 1],
            "pd_lgbm": oof_pds[:, 2],
            "y": y_np,
        }
    )

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    meta_record = {
        "model_type": "stack_lr_rf_lgbm",
        "freeze_timestamp_utc": freeze_timestamp,
        "test_looked_at": False,
        "test_metrics": None,
        "test_labels_used_to_fit_or_select": False,
        "split": {
            "source": "scripts/train.py::load_and_split",
            "test_size": 0.3,
            "stratify": TARGET,
            "random_state": RANDOM_STATE,
            "train_n": train_n,
            "train_bad_rate": train_bad_rate,
            "test_n": test_n,
            "test_bad_rate": test_bad_rate,
        },
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "loan_paid_back_in_features": False,
        "iv_drop": False,
        "vif_drop": False,
        "woe_bins": WOE_BINS,
        "cv_folds": CV_FOLDS,
        "n_trials_requested": N_TRIALS,
        "n_trials_completed": len(study.trials),
        "optuna_sampler": "TPESampler",
        "optuna_seed": RANDOM_STATE,
        "best_hyperparameters": {
            "lr": win_lr,
            "rf": {k: v for k, v in win_rf.items()},
            "lgbm_tune": win_lgbm_tune,
            "lgbm_final": win_lgbm_final,
            "meta": _lr_params(best_meta_c),
        },
        "lgbm_best_iterations_oof": lgbm_best_iterations,
        "lgbm_mean_best_iteration": mean_best_iteration,
        "n_estimators_final": n_estimators_final,
        "meta_c_grid": META_C_GRID,
        "meta_c_scores": meta_c_scores,
        "best_meta_C": best_meta_c,
        "oof_stack_metrics": oof_stack_metrics,
        "oof_stack_auc": oof_stack_auc,
        "oof_stack_auc_raw": float(best_meta_auc),
        "beat_last_run_cv_0.8862": beat_last_cv,
        "last_run_cv_auc": LAST_RUN_CV_AUC,
        "per_base_oof_metrics": per_base_oof,
        "optuna_best_value_during_search": float(study.best_value),
        "optuna_trial_values": [float(t.value) if t.value is not None else None for t in study.trials],
        "iv_table_diagnostic": iv_table.to_dict(orient="records"),
        "notes": (
            "Meta-learner is fitted on OOF base PDs only and is NOT refit on "
            "full-train in-sample base PDs. Train stacked PD for PSI must be "
            "meta.predict_proba on the OOF 14000x3 matrix."
        ),
    }

    # ------------------------------------------------------------------
    # 7. Persist freeze artifacts. Do not look at test.
    # ------------------------------------------------------------------
    for forbidden in FORBIDDEN_WRITE_PATHS:
        if os.path.abspath(forbidden) in {
            os.path.abspath(BASES_PATH),
            os.path.abspath(META_MODEL_PATH),
            os.path.abspath(ENCODERS_PATH),
            os.path.abspath(META_JSON_PATH),
            os.path.abspath(OOF_PDS_PATH),
        }:
            raise RuntimeError(f"refusing to overwrite forbidden path {forbidden}")

    joblib.dump({"lr": lr_full, "rf": rf_full, "lgbm": lgbm_full}, BASES_PATH)
    joblib.dump(meta, META_MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    oof_df.to_csv(OOF_PDS_PATH, index=False)
    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta_record), fh, ensure_ascii=False, indent=2)

    written = [BASES_PATH, META_MODEL_PATH, ENCODERS_PATH, META_JSON_PATH, OOF_PDS_PATH]
    for p in written:
        if os.path.basename(p) in {"lgbm_model.joblib"} or os.path.basename(p).startswith(
            ("lgbm_linear_tree_", "linear_tree_")
        ):
            raise RuntimeError(f"accidentally wrote forbidden artifact {p}")

    print(f"[train_stack] FREEZE {freeze_timestamp}", flush=True)
    print("[train_stack] test_looked_at=false test_metrics=null", flush=True)
    print(f"[train_stack] artifacts: {written}", flush=True)
    print("[train_stack] STOP — eval script computes test metrics after freeze.", flush=True)


if __name__ == "__main__":
    main()
