"""
OpenFE (Zhang et al., ICML 2023) + LightGBM gbdt — train / freeze only.

Frozen split via scripts/train.py::load_and_split UNCHANGED.
gender is dropped from X and from all FE. employment_status stays in as a
normal original feature. No IV/VIF drop. No monotone_constraints. No
interaction_constraints.

CRITICAL LEAKAGE RULE
- OpenFE.fit is called on TRAIN rows + TRAIN labels only.
- Selected feature formulas are frozen, then official openfe.transform is
  applied to produce train/test matrices. transform does not use labels.
  GroupByThen* / freq operators evaluate on concat(train, test) by OpenFE
  library design (no labels). Feature *selection* never sees test.
- LightGBM is tuned on TRAIN only (StratifiedKFold 5, shuffle=True,
  random_state=42, maximize OOF AUC). Test labels are never used to fit
  or select.
- This script FREEZES with test_looked_at=false and test_metrics=null.
  scripts/eval_openfe_lgbm.py computes Test metrics and never fits FE or
  the model.

Usage (from repo root): python3 scripts/train_openfe_lgbm.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from train import load_and_split
from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, PSI_WATCH, RANDOM_STATE, RAW_TARGET_COL, TARGET
from utils.risk_skills import evaluate_discrimination_and_ks

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

EXPLICIT_CAT_COLS = [
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
]

# OpenFE stage-2 ranked list: keep positive-gain formulas, cap at TOP_K.
OPENFE_TOP_K = 50
OPENFE_N_DATA_BLOCKS = 8
OPENFE_MIN_CANDIDATE_FEATURES = 2000
OPENFE_STAGE1_METRIC = "predictive"
OPENFE_STAGE2_METRIC = "gain_importance"
OPENFE_SEED = RANDOM_STATE

CV_FOLDS = 5
N_TRIALS_OPENFE = 25
N_TRIALS_BASELINE = 20
N_ESTIMATORS_TUNE = 1000
EARLY_STOPPING_ROUNDS = 50

PERM_SAMPLE_N = 4000
PERM_REPEATS = 3

REQUIRED_PATHS = [
    "data/loan_dataset_20000.csv",
    "scripts/train.py",
    "scripts/test.py",
    "utils/config.py",
    "utils/risk_skills.py",
]

FORBIDDEN_PREFIXES = (
    "lgbm_linear_tree_",
    "linear_tree_",
    "stack_lr_rf_lgbm_",
    "lgbm_no_gender_emp_overlay_",
    "lgbm_emp_overlay_",
    "lgbm_emp_in_",
    "lgbm_dart_monotone_",
    "dart_monotone_",
    "ensemble_no_gender_",
    "ensemble_fm_ft_",
    "danet_",
    "ag_realmlp_",
    "autogluon_",
    "realmlp_",
)

FORBIDDEN_EXACT = (
    os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib"),
)

PREFIX = "openfe_lgbm_"
FAILURE_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_FAILURE.md")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_model.joblib")
BASELINE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_baseline_model.joblib")
OPENFE_OBJ_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_openfe.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_features.joblib")
FORMULAS_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_formulas.json")
PREP_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_prep.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_y_test.csv")
X_TRAIN_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train.joblib")
X_TEST_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test.joblib")
X_TRAIN_BASE_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train_base.joblib")
X_TEST_BASE_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test_base.joblib")
X_TRAIN_RAW_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train_raw20.joblib")
X_TEST_RAW_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test_raw20.joblib")
SHA_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_sha256.txt")
REQS_COPY_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_requirements.txt")
VERSIONS_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_versions.json")

ALLOWED_WRITE_PATHS = {
    os.path.abspath(p)
    for p in [
        FAILURE_PATH,
        MODEL_PATH,
        BASELINE_MODEL_PATH,
        OPENFE_OBJ_PATH,
        FEATURES_PATH,
        FORMULAS_PATH,
        PREP_PATH,
        META_PATH,
        OOF_PDS_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        X_TRAIN_JOBLIB,
        X_TEST_JOBLIB,
        X_TRAIN_BASE_JOBLIB,
        X_TEST_BASE_JOBLIB,
        X_TRAIN_RAW_JOBLIB,
        X_TEST_RAW_JOBLIB,
        SHA_PATH,
        REQS_COPY_PATH,
        VERSIONS_PATH,
        os.path.join(ARTIFACTS_DIR, "openfe_lgbm_tmp_data.feather"),
    ]
}

TMP_FEATHER_CANDIDATES = (
    "./openfe_tmp_data.feather",
    "./openfe_tmp_data_xx.feather",
    os.path.join(ARTIFACTS_DIR, "openfe_lgbm_tmp_data.feather"),
)


def _require_pipeline_files() -> None:
    missing = [p for p in REQUIRED_PATHS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required pipeline files: {missing}")


def _assert_allowed_write(path: str) -> None:
    abs_path = os.path.abspath(path)
    name = os.path.basename(abs_path)
    if abs_path in {os.path.abspath(p) for p in FORBIDDEN_EXACT}:
        raise RuntimeError(f"Refusing to write forbidden path: {path}")
    for pref in FORBIDDEN_PREFIXES:
        if name.startswith(pref):
            raise RuntimeError(f"Refusing to write forbidden prefix {pref}: {path}")
    if name == "lgbm_model.joblib":
        raise RuntimeError(f"Refusing to overwrite artifacts/lgbm_model.joblib: {path}")
    if not name.startswith(PREFIX) and os.path.basename(abs_path) != "openfe_lgbm_tmp_data.feather":
        # Only openfe_lgbm_* artifacts (plus explicit tmp) may be written under artifacts/.
        if os.path.dirname(abs_path) == os.path.abspath(ARTIFACTS_DIR):
            raise RuntimeError(f"Refusing non-openfe_lgbm artifact write: {path}")
    if abs_path not in ALLOWED_WRITE_PATHS:
        raise RuntimeError(f"Write path not on allowlist: {path}")


def _safe_dump_json(path: str, obj) -> None:
    _assert_allowed_write(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def _safe_joblib_dump(path: str, obj) -> None:
    _assert_allowed_write(path)
    joblib.dump(obj, path)


def _safe_df_csv(path: str, df: pd.DataFrame) -> None:
    _assert_allowed_write(path)
    df.to_csv(path, index=False)


def _write_failure(title: str, detail: str) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    _assert_allowed_write(FAILURE_PATH)
    body = (
        f"# OpenFE + LightGBM failure — STOP\n\n"
        f"**{title}**\n\n"
        f"Chosen tool is OpenFE. This pack does **not** silently swap to "
        f"Featuretools, Autofeat, SAFE, or any other FE library.\n\n"
        f"```\n{detail}\n```\n"
    )
    with open(FAILURE_PATH, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"[train_openfe_lgbm] FAILURE written to {FAILURE_PATH}", flush=True)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def package_versions() -> dict:
    import sklearn

    versions = {
        "python": platform.python_version(),
        "openfe": None,
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "optuna": optuna.__version__,
    }
    try:
        import openfe as _openfe

        versions["openfe"] = getattr(_openfe, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        versions["openfe"] = f"IMPORT_FAILED: {exc}"
    return versions


def device_info() -> dict:
    return {
        "device": "cpu",
        "n_cpus": int(os.cpu_count() or 1),
        "note": "This VM has no GPU; OpenFE and LightGBM run on CPU.",
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_openfe_tmp() -> None:
    for path in TMP_FEATHER_CANDIDATES:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    for path in glob.glob("./openfe_tmp_data*.feather"):
        try:
            os.remove(path)
        except OSError:
            pass


def _assert_base_columns(columns: list[str], context: str) -> None:
    cols = list(columns)
    if "gender" in cols:
        raise RuntimeError(f"{context}: gender must be ABSENT. Found gender in {cols}")
    if "employment_status" not in cols:
        raise RuntimeError(
            f"{context}: employment_status must be PRESENT. columns={cols}"
        )
    if RAW_TARGET_COL in cols:
        raise RuntimeError(f"{context}: {RAW_TARGET_COL} must never be a feature.")
    if TARGET in cols:
        raise RuntimeError(f"{context}: target {TARGET} present in feature columns.")
    base = [c for c in cols if not str(c).startswith("autoFE_")]
    if base != EXPECTED_IN_MODEL_FEATURES:
        raise RuntimeError(
            f"{context}: base-20 order/set mismatch. got={base} expected={EXPECTED_IN_MODEL_FEATURES}"
        )


def _assert_no_gender_anywhere(columns, formulas, context: str) -> None:
    cols = [str(c) for c in columns]
    if any("gender" == c.lower() or c.lower().startswith("gender") for c in cols):
        raise RuntimeError(f"{context}: gender-like column in {cols}")
    for f in formulas:
        token = str(f).lower()
        if "gender" in token:
            raise RuntimeError(f"{context}: gender leaked into OpenFE formula: {f}")


def build_in_model_features(feature_cols: list[str]) -> list[str]:
    in_model = [c for c in feature_cols if c not in DROPPED_FROM_MODEL]
    if in_model != EXPECTED_IN_MODEL_FEATURES:
        raise RuntimeError(
            f"in_model_features order != expected original-order list. "
            f"got={in_model} expected={EXPECTED_IN_MODEL_FEATURES}"
        )
    return in_model


def verify_frozen_split(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    if len(train_df) != 14000 or len(test_df) != 6000:
        raise RuntimeError(f"Unexpected split sizes: train={len(train_df)} test={len(test_df)}")
    train_bad = int(train_df[TARGET].sum())
    test_bad = int(test_df[TARGET].sum())
    train_rate = round(float(train_df[TARGET].mean()), 4)
    test_rate = round(float(test_df[TARGET].mean()), 4)
    if train_bad != 2801 or abs(train_rate - 0.2001) > 1e-6:
        raise RuntimeError(f"Train bad count/rate mismatch: n_bads={train_bad} rate={train_rate}")
    if test_bad != 1201 or abs(test_rate - 0.2002) > 1e-6:
        raise RuntimeError(f"Test bad count/rate mismatch: n_bads={test_bad} rate={test_rate}")
    if RAW_TARGET_COL in train_df.columns or RAW_TARGET_COL in test_df.columns or RAW_TARGET_COL in feature_cols:
        raise RuntimeError(f"{RAW_TARGET_COL} leaked into the split frame or feature_cols")
    if TARGET in feature_cols:
        raise RuntimeError("target column present in feature_cols")
    if "gender" not in feature_cols or "gender" not in train_df.columns:
        raise RuntimeError("gender missing from original feature_cols; cannot drop it from X by name")
    if "employment_status" not in feature_cols or "employment_status" not in train_df.columns:
        raise RuntimeError("employment_status missing from original feature_cols")


def _ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def lgbm_params_template() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "subsample_freq": 1,
        "monotone_constraints": None,
        "interaction_constraints": None,
    }


def run_oof_cv(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    skf: StratifiedKFold,
    categorical_feature: list[str] | None = None,
) -> dict:
    oof = np.zeros(len(X), dtype=float)
    fold_aucs = []
    fold_ks = []
    best_iterations = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = lgb.LGBMClassifier(**params)
        fit_kwargs = {
            "eval_set": [(X_va, y_va)],
            "callbacks": [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        }
        if categorical_feature:
            fit_kwargs["categorical_feature"] = categorical_feature
        model.fit(X_tr, y_tr, **fit_kwargs)
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(float(roc_auc_score(y_va, pred)))
        fold_ks.append(_ks(y_va.to_numpy(), pred))
        best_it = int(getattr(model, "best_iteration_", params.get("n_estimators", N_ESTIMATORS_TUNE)) or 0)
        best_iterations.append(max(best_it, 1))
        print(
            f"    fold {fold}/{skf.n_splits} AUC={fold_aucs[-1]:.6f} KS={fold_ks[-1]:.6f} "
            f"best_iteration={best_iterations[-1]}",
            flush=True,
        )
    return {
        "oof_preds": oof,
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": _ks(y.to_numpy(), oof),
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
        "best_iterations": best_iterations,
    }


def _assert_numeric_frame(X: pd.DataFrame, context: str) -> None:
    bad = []
    for col, dtype in X.dtypes.items():
        if not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
            bad.append(f"{col}: {dtype}")
    if bad:
        raise TypeError(f"{context}: LightGBM requires int/float/bool dtypes. Bad columns: {bad}")


def tune_lgbm(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    categorical_feature: list[str] | None = None,
) -> dict:
    _assert_numeric_frame(X, name)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = lgbm_params_template()
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
        print(f"[{name}] trial {trial.number:03d}/{n_trials - 1} starting", flush=True)
        cv_res = run_oof_cv(params, X, y, skf, categorical_feature=categorical_feature)
        trial.set_user_attr("oof_ks", cv_res["oof_ks"])
        trial.set_user_attr("fold_best_iterations", cv_res["best_iterations"])
        print(
            f"[{name}] trial {trial.number:03d} OOF AUC={cv_res['oof_auc']:.6f} "
            f"OOF KS={cv_res['oof_ks']:.6f}",
            flush=True,
        )
        return cv_res["oof_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    winning = lgbm_params_template()
    winning.update(dict(study.best_params))
    winning["n_estimators"] = N_ESTIMATORS_TUNE
    print(f"[{name}] Re-running winning 5-fold OOF to freeze n_estimators_final...", flush=True)
    winner_cv = run_oof_cv(winning, X, y, skf, categorical_feature=categorical_feature)
    n_estimators_final = max(50, int(round(float(np.mean(winner_cv["best_iterations"])))))
    final_params = dict(winning)
    final_params["n_estimators"] = n_estimators_final
    final_params["n_jobs"] = int(os.cpu_count() or 1)
    print(
        f"[{name}] Refitting on ALL {len(X)} train rows with n_estimators={n_estimators_final} "
        f"(no early stopping, no test).",
        flush=True,
    )
    final_model = lgb.LGBMClassifier(**final_params)
    if categorical_feature:
        final_model.fit(X, y, categorical_feature=categorical_feature)
    else:
        final_model.fit(X, y)
    return {
        "study": study,
        "best_search": dict(study.best_params),
        "winning_tune_params": winning,
        "final_params": final_params,
        "n_estimators_final": n_estimators_final,
        "winner_cv": winner_cv,
        "final_model": final_model,
        "n_trials_completed": len(study.trials),
    }


class FeaturePrep:
    """Train-only category → integer codes. LightGBM 4.x + pandas 3 reject str/category."""

    def __init__(self, categorical_columns: list[str], category_levels: dict[str, list]):
        self.categorical_columns = list(categorical_columns)
        self.category_levels = {k: list(v) for k, v in category_levels.items()}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.categorical_columns:
            if col not in out.columns:
                continue
            mapping = {v: i for i, v in enumerate(self.category_levels[col])}
            codes = pd.Series(out[col].astype("string"), index=out.index).map(mapping)
            out[col] = pd.to_numeric(codes, errors="coerce")
        for col in out.columns:
            if col in self.categorical_columns:
                continue
            if not (
                pd.api.types.is_integer_dtype(out[col])
                or pd.api.types.is_float_dtype(out[col])
                or pd.api.types.is_bool_dtype(out[col])
            ):
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out


def fit_prep(df: pd.DataFrame) -> FeaturePrep:
    cat_cols = []
    levels = {}
    for col in df.columns:
        dtype = df[col].dtype
        if str(dtype) in ("object", "string", "str", "category") or isinstance(dtype, pd.CategoricalDtype):
            cat_cols.append(col)
            # Train levels only. Unseen test levels become NaN (LightGBM-safe).
            cats = pd.Index(pd.Series(df[col]).astype("string").dropna().unique())
            levels[col] = [str(x) for x in cats.tolist()]
    return FeaturePrep(cat_cols, levels)


def import_openfe():
    try:
        import openfe as openfe_mod
        from openfe import OpenFE, get_candidate_features, transform
        from openfe.utils import tree_to_formula
    except Exception as exc:  # noqa: BLE001
        _write_failure("openfe import failed", traceback.format_exc())
        raise RuntimeError(
            "openfe could not be imported. STOP — not swapping FE libraries."
        ) from exc
    return openfe_mod, OpenFE, get_candidate_features, transform, tree_to_formula


def select_openfe_features(ranked_nodes, ranked_scores, top_k: int):
    kept = []
    kept_scores = []
    for node, score in zip(ranked_nodes, ranked_scores):
        if float(score) > 0:
            kept.append(node)
            kept_scores.append(float(score))
        if len(kept) >= top_k:
            break
    if not kept and ranked_nodes:
        n = min(top_k, len(ranked_nodes))
        kept = list(ranked_nodes[:n])
        kept_scores = [float(s) for s in ranked_scores[:n]]
    return kept, kept_scores


def persistable_openfe(ofe):
    ofe.data = None
    ofe.label = None
    ofe.init_scores = None
    ofe.train_index = None
    ofe.val_index = None
    return ofe


def main() -> None:
    _require_pipeline_files()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if os.path.exists(os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")):
        print("[train_openfe_lgbm] Note: artifacts/lgbm_model.joblib exists and will NOT be overwritten.")

    versions = package_versions()
    device = device_info()
    n_jobs = int(device["n_cpus"])
    print(f"[train_openfe_lgbm] device={device} versions={versions}", flush=True)

    if versions.get("openfe") is None or str(versions.get("openfe", "")).startswith("IMPORT_FAILED"):
        _write_failure("openfe pip/import failed", str(versions.get("openfe")))
        raise RuntimeError("openfe unavailable. STOP — not swapping FE libraries.")

    openfe_mod, OpenFE, get_candidate_features, transform, tree_to_formula = import_openfe()

    print("[train_openfe_lgbm] Loading frozen split via scripts/train.py::load_and_split (unchanged)...", flush=True)
    train_df, test_df, feature_cols = load_and_split()
    verify_frozen_split(train_df, test_df, feature_cols)

    in_model_features = build_in_model_features(feature_cols)
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    X_train_base = train_df[in_model_features].copy()
    X_test_base = test_df[in_model_features].copy()
    # Unique indices so OpenFE concat(train, test) does not collide after reset_index.
    X_train_base.index = range(len(X_train_base))
    X_test_base.index = range(len(X_train_base), len(X_train_base) + len(X_test_base))
    y_train.index = X_train_base.index
    y_test.index = X_test_base.index

    _assert_base_columns(list(X_train_base.columns), "X_train_base")
    _assert_base_columns(list(X_test_base.columns), "X_test_base")
    if "gender" in X_train_base.columns or "gender" in X_test_base.columns:
        raise RuntimeError("gender present in base X")

    print(
        f"[train_openfe_lgbm] Train n={len(X_train_base)}, bad_rate={float(y_train.mean()):.4f} "
        f"n_bads={int(y_train.sum())} | Test n={len(X_test_base)}, bad_rate={float(y_test.mean()):.4f} "
        f"n_bads={int(y_test.sum())} (held out; unused for FE fit / tune / select)",
        flush=True,
    )
    print(f"[train_openfe_lgbm] in_model_features ({len(in_model_features)}): {in_model_features}", flush=True)
    print(
        "[train_openfe_lgbm] gender_in_model=false employment_status_in_model=true "
        "monotone_constraints=false interaction_constraints=false iv_drop=false vif_drop=false",
        flush=True,
    )

    # Detect ordinal vs numeric the same way OpenFE.fit does, so n_generated is exact.
    categorical_features = [c for c in EXPLICIT_CAT_COLS if c in in_model_features]
    ordinal_features = []
    numerical_features = []
    for feature in in_model_features:
        if feature in categorical_features:
            continue
        if X_train_base[feature].nunique(dropna=True) <= 100:
            ordinal_features.append(feature)
        else:
            numerical_features.append(feature)

    candidate_features_list = get_candidate_features(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        order=1,
    )
    n_generated = len(candidate_features_list)
    print(
        f"[train_openfe_lgbm] OpenFE candidate features generated (order=1): {n_generated} "
        f"(numeric={numerical_features} ordinal={ordinal_features} cat={categorical_features})",
        flush=True,
    )

    label_df = pd.DataFrame({TARGET: y_train.to_numpy()}, index=X_train_base.index)
    tmp_save_path = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_tmp_data.feather")
    _assert_allowed_write(tmp_save_path)

    print(
        f"[train_openfe_lgbm] Fitting OpenFE on TRAIN only (n_jobs={n_jobs}, seed={OPENFE_SEED}, "
        f"metric=auc, stage1={OPENFE_STAGE1_METRIC}, stage2={OPENFE_STAGE2_METRIC})...",
        flush=True,
    )
    ofe = OpenFE()
    try:
        new_features_list = ofe.fit(
            data=X_train_base.copy(),
            label=label_df.copy(),
            task="classification",
            metric="auc",
            categorical_features=categorical_features,
            candidate_features_list=candidate_features_list,
            n_data_blocks=OPENFE_N_DATA_BLOCKS,
            min_candidate_features=OPENFE_MIN_CANDIDATE_FEATURES,
            feature_boosting=False,
            stage1_metric=OPENFE_STAGE1_METRIC,
            stage2_metric=OPENFE_STAGE2_METRIC,
            is_stage1=True,
            n_repeats=1,
            tmp_save_path=tmp_save_path,
            n_jobs=n_jobs,
            seed=OPENFE_SEED,
            verbose=True,
        )
    except Exception:
        _write_failure("OpenFE.fit failed on TRAIN", traceback.format_exc())
        cleanup_openfe_tmp()
        raise

    ranked_nodes = list(new_features_list)
    ranked_scores = []
    if getattr(ofe, "new_features_scores_list", None):
        ranked_scores = [float(s) for _, s in ofe.new_features_scores_list]
    else:
        ranked_scores = [0.0] * len(ranked_nodes)

    selected_nodes, selected_scores = select_openfe_features(ranked_nodes, ranked_scores, OPENFE_TOP_K)
    selected_formulas = [tree_to_formula(n) for n in selected_nodes]
    all_formulas = [tree_to_formula(n) for n in ranked_nodes]
    _assert_no_gender_anywhere(in_model_features + selected_formulas, selected_formulas + all_formulas, "OpenFE formulas")

    print(
        f"[train_openfe_lgbm] OpenFE stage2 ranked {len(ranked_nodes)} features; "
        f"using {len(selected_nodes)} (gain>0, cap={OPENFE_TOP_K}). "
        f"Base 20 will be KEPT. Sample formulas: {selected_formulas[:8]}",
        flush=True,
    )

    print(
        "[train_openfe_lgbm] Transforming train+test with FROZEN OpenFE formulas "
        "(official transform; no refit, no test labels)...",
        flush=True,
    )
    try:
        X_train_fe, X_test_fe = transform(
            X_train_base.copy(),
            X_test_base.copy(),
            selected_nodes,
            n_jobs=n_jobs,
            name="",
        )
    except Exception:
        _write_failure("openfe.transform failed", traceback.format_exc())
        cleanup_openfe_tmp()
        raise
    finally:
        cleanup_openfe_tmp()

    _assert_base_columns(list(X_train_fe.columns), "X_train after OpenFE transform")
    _assert_base_columns(list(X_test_fe.columns), "X_test after OpenFE transform")
    _assert_no_gender_anywhere(list(X_train_fe.columns), selected_formulas, "transformed X")
    if list(X_train_fe.columns) != list(X_test_fe.columns):
        raise RuntimeError("Train/test feature names diverge after OpenFE transform")

    fe_names = [c for c in X_train_fe.columns if str(c).startswith("autoFE_")]
    if len(fe_names) != len(selected_nodes):
        raise RuntimeError(f"Expected {len(selected_nodes)} autoFE columns, got {len(fe_names)}")

    # Replace inf from division operators; LightGBM handles NaN.
    X_train_fe = X_train_fe.replace([np.inf, -np.inf], np.nan)
    X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan)

    prep = fit_prep(X_train_fe)
    X_train = prep.transform(X_train_fe)
    X_test = prep.transform(X_test_fe)
    # Baseline uses a separate prep fitted on base-20 train only.
    baseline_prep = fit_prep(X_train_base)
    X_train_raw = baseline_prep.transform(X_train_base)
    X_test_raw = baseline_prep.transform(X_test_base)

    in_model_all = list(X_train.columns)
    print(
        f"[train_openfe_lgbm] Final matrix: {len(in_model_all)} cols "
        f"(base 20 + {len(fe_names)} OpenFE). gender absent={('gender' not in in_model_all)}",
        flush=True,
    )

    print("[train_openfe_lgbm] Tuning LightGBM gbdt on OpenFE+base TRAIN features only...", flush=True)
    openfe_tune = tune_lgbm(
        "openfe_lgbm",
        X_train,
        y_train,
        N_TRIALS_OPENFE,
        categorical_feature=prep.categorical_columns,
    )
    print("[train_openfe_lgbm] Tuning baseline LightGBM gbdt on raw 20 TRAIN features only...", flush=True)
    baseline_tune = tune_lgbm(
        "baseline_raw20",
        X_train_raw,
        y_train,
        N_TRIALS_BASELINE,
        categorical_feature=baseline_prep.categorical_columns,
    )

    openfe_model = openfe_tune["final_model"]
    baseline_model = baseline_tune["final_model"]
    model_names = list(getattr(openfe_model, "feature_name_", in_model_all))
    _assert_no_gender_anywhere(model_names, selected_formulas, "fitted LGBM feature_name_")
    if "employment_status" not in model_names:
        raise RuntimeError("Fitted OpenFE+LGBM missing employment_status")

    gain = {
        f: int(v)
        for f, v in zip(list(X_train.columns), [int(x) for x in openfe_model.feature_importances_])
    }
    gain_ranked = sorted(gain.items(), key=lambda x: x[1], reverse=True)
    emp_gain_rank = next(i for i, (f, _) in enumerate(gain_ranked, start=1) if f == "employment_status")

    print(
        f"[train_openfe_lgbm] TRAIN permutation importance (sample={PERM_SAMPLE_N}, repeats={PERM_REPEATS})...",
        flush=True,
    )
    rng = np.random.RandomState(RANDOM_STATE)
    sample_n = min(PERM_SAMPLE_N, len(X_train))
    sample_idx = rng.choice(len(X_train), size=sample_n, replace=False)
    perm = permutation_importance(
        openfe_model,
        X_train.iloc[sample_idx],
        y_train.iloc[sample_idx],
        n_repeats=PERM_REPEATS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )
    perm_map = {f: float(v) for f, v in zip(X_train.columns, perm.importances_mean)}
    perm_ranked = sorted(perm_map.items(), key=lambda x: x[1], reverse=True)
    emp_perm_rank = next(i for i, (f, _) in enumerate(perm_ranked, start=1) if f == "employment_status")

    train_refit_prob = openfe_model.predict_proba(X_train)[:, 1]
    train_refit_metrics = evaluate_discrimination_and_ks(y_train.to_numpy(), train_refit_prob)
    baseline_refit_prob = baseline_model.predict_proba(X_train_raw)[:, 1]
    baseline_refit_metrics = evaluate_discrimination_and_ks(y_train.to_numpy(), baseline_refit_prob)

    oof_openfe = evaluate_discrimination_and_ks(y_train.to_numpy(), openfe_tune["winner_cv"]["oof_preds"])
    oof_base = evaluate_discrimination_and_ks(y_train.to_numpy(), baseline_tune["winner_cv"]["oof_preds"])

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_openfe_lgbm] FREEZE at {freeze_timestamp} (before any test metrics)", flush=True)

    ofe_persist = persistable_openfe(ofe)
    formulas_payload = {
        "selection_method": (
            "OpenFE two-stage selection on TRAIN only: stage1=predictive successive "
            f"feature-wise halving (n_data_blocks={OPENFE_N_DATA_BLOCKS}, "
            f"min_candidate_features={OPENFE_MIN_CANDIDATE_FEATURES}); "
            f"stage2={OPENFE_STAGE2_METRIC}. Keep formulas with stage2 gain>0, "
            f"capped at TOP_K={OPENFE_TOP_K}. Original base 20 columns are KEPT."
        ),
        "n_generated": int(n_generated),
        "n_stage2_ranked": int(len(ranked_nodes)),
        "n_selected": int(len(selected_nodes)),
        "top_k_cap": OPENFE_TOP_K,
        "base_20_kept": True,
        "formulas": selected_formulas,
        "scores": selected_scores,
        "all_stage2_formulas": all_formulas,
        "all_stage2_scores": ranked_scores,
        "autoFE_name_to_formula": {name: formula for name, formula in zip(fe_names, selected_formulas)},
        "leakage_note": (
            "OpenFE.fit used TRAIN rows and TRAIN labels only. "
            "openfe.transform applies the frozen formulas; it does not refit or use labels. "
            "OpenFE's official transform concatenates train+test solely so GroupByThen*/freq "
            "operators can evaluate (library design). Test labels were never used. "
            "Feature selection never included test rows or test labels."
        ),
    }

    oof_df = pd.DataFrame(
        {
            "oof_pd_openfe_lgbm": openfe_tune["winner_cv"]["oof_preds"],
            "oof_pd_baseline_raw20": baseline_tune["winner_cv"]["oof_preds"],
            TARGET: y_train.to_numpy(),
        }
    )

    meta = {
        "algorithm": "OpenFE + lightgbm.LGBMClassifier",
        "openfe": {
            "package": "openfe",
            "version": versions["openfe"],
            "paper": "Zhang et al., ICML 2023",
            "n_jobs": n_jobs,
            "n_generated": int(n_generated),
            "n_stage2_ranked": int(len(ranked_nodes)),
            "n_selected": int(len(selected_nodes)),
            "top_k_cap": OPENFE_TOP_K,
            "base_20_kept": True,
            "selection_method": formulas_payload["selection_method"],
            "categorical_features": categorical_features,
            "ordinal_features": ordinal_features,
            "numerical_features": numerical_features,
            "n_data_blocks": OPENFE_N_DATA_BLOCKS,
            "min_candidate_features": OPENFE_MIN_CANDIDATE_FEATURES,
            "stage1_metric": OPENFE_STAGE1_METRIC,
            "stage2_metric": OPENFE_STAGE2_METRIC,
            "feature_boosting": False,
            "metric": "auc",
            "task": "classification",
            "seed": OPENFE_SEED,
            "order": 1,
            "formulas": selected_formulas,
            "autoFE_columns": fe_names,
        },
        "lgbm_openfe": {
            "boosting_type": "gbdt",
            "best_hyperparameters": _json_ready(openfe_tune["best_search"]),
            "n_estimators_final": openfe_tune["n_estimators_final"],
            "final_params": _json_ready(openfe_tune["final_params"]),
            "n_trials_requested": N_TRIALS_OPENFE,
            "n_trials_completed": openfe_tune["n_trials_completed"],
            "winning_fold_best_iterations": openfe_tune["winner_cv"]["best_iterations"],
            "winning_fold_aucs": [round(x, 6) for x in openfe_tune["winner_cv"]["fold_aucs"]],
            "search_note": (
                f"Optuna TPESampler seed={RANDOM_STATE}, {N_TRIALS_OPENFE} trials, "
                f"StratifiedKFold({CV_FOLDS}, shuffle=True, random_state=42), maximize OOF AUC. "
                "n_estimators_final = mean fold best_iteration (early_stopping=50, n_estimators_tune=1000)."
            ),
        },
        "lgbm_baseline_raw20": {
            "boosting_type": "gbdt",
            "best_hyperparameters": _json_ready(baseline_tune["best_search"]),
            "n_estimators_final": baseline_tune["n_estimators_final"],
            "final_params": _json_ready(baseline_tune["final_params"]),
            "n_trials_requested": N_TRIALS_BASELINE,
            "n_trials_completed": baseline_tune["n_trials_completed"],
            "winning_fold_best_iterations": baseline_tune["winner_cv"]["best_iterations"],
            "winning_fold_aucs": [round(x, 6) for x in baseline_tune["winner_cv"]["fold_aucs"]],
            "search_note": (
                f"Same Optuna protocol on raw 20 only ({N_TRIALS_BASELINE} trials). "
                "Train-only. Not used to select the OpenFE candidate."
            ),
        },
        "oof_metrics": {
            "openfe_lgbm": oof_openfe,
            "baseline_raw20": oof_base,
        },
        "train_refit_metrics": {
            "openfe_lgbm": train_refit_metrics,
            "baseline_raw20": baseline_refit_metrics,
        },
        "categorical_columns_openfe": prep.categorical_columns,
        "categorical_columns_baseline": baseline_prep.categorical_columns,
        "in_model_features": in_model_all,
        "base_20_features": in_model_features,
        "openfe_feature_names": fe_names,
        "n_in_model_features": len(in_model_all),
        "gender_in_model": False,
        "employment_status_in_model": True,
        "monotone_constraints": False,
        "interaction_constraints": False,
        "iv_drop": False,
        "vif_drop": False,
        "loan_paid_back_in_features": False,
        "feature_importances_gain": [{"feature": f, "importance": int(v)} for f, v in gain_ranked],
        "employment_status_gain_rank": emp_gain_rank,
        "employment_status_permutation_rank_train": emp_perm_rank,
        "permutation_importance_train": {
            "n_sample": sample_n,
            "n_repeats": PERM_REPEATS,
            "scoring": "roc_auc",
            "ranking": [{"feature": f, "importance": float(v)} for f, v in perm_ranked],
        },
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
            "selection_rule": "OpenFE fit + LGBM Optuna maximize train OOF AUC (test not used)",
            "test_looked_at": False,
            "test_metrics": None,
            "test_labels_used_to_fit_or_select": False,
        },
        "gates": {"AUC_MIN": AUC_MIN, "KS_MIN": KS_MIN, "PSI_WATCH": PSI_WATCH},
        "package_versions": versions,
        "device": device,
        "artifact_paths": {
            "model": MODEL_PATH,
            "baseline_model": BASELINE_MODEL_PATH,
            "openfe_object": OPENFE_OBJ_PATH,
            "features": FEATURES_PATH,
            "formulas": FORMULAS_PATH,
            "prep": PREP_PATH,
            "meta": META_PATH,
            "oof_pds": OOF_PDS_PATH,
            "X_train_csv": X_TRAIN_PATH,
            "X_test_csv": X_TEST_PATH,
            "X_train_joblib": X_TRAIN_JOBLIB,
            "X_test_joblib": X_TEST_JOBLIB,
        },
        "did_not_write": [
            "artifacts/lgbm_model.joblib",
            "artifacts/lgbm_linear_tree_*",
            "artifacts/stack_lr_rf_lgbm_*",
            "artifacts/lgbm_emp_overlay_*",
            "artifacts/lgbm_emp_in_*",
            "artifacts/dart_monotone_*",
            "artifacts/ensemble_no_gender_*",
            "artifacts/ensemble_fm_ft_*",
            "artifacts/danet_*",
            "artifacts/ag_realmlp_*",
            "artifacts/autogluon_*",
            "artifacts/realmlp_*",
        ],
        "leakage_note": formulas_payload["leakage_note"],
        "computed_after_freeze": False,
    }

    _safe_joblib_dump(MODEL_PATH, openfe_model)
    _safe_joblib_dump(BASELINE_MODEL_PATH, baseline_model)
    _safe_joblib_dump(OPENFE_OBJ_PATH, ofe_persist)
    _safe_joblib_dump(
        FEATURES_PATH,
        {
            "selected_nodes": selected_nodes,
            "selected_scores": selected_scores,
            "selected_formulas": selected_formulas,
            "openfe_version": versions["openfe"],
        },
    )
    _safe_dump_json(FORMULAS_PATH, formulas_payload)
    _safe_df_csv(OOF_PDS_PATH, oof_df)
    _safe_df_csv(X_TRAIN_PATH, X_train)
    _safe_df_csv(X_TEST_PATH, X_test)
    _safe_df_csv(Y_TRAIN_PATH, pd.DataFrame({TARGET: y_train.to_numpy()}))
    _safe_df_csv(Y_TEST_PATH, pd.DataFrame({TARGET: y_test.to_numpy()}))
    _safe_joblib_dump(X_TRAIN_JOBLIB, X_train)
    _safe_joblib_dump(X_TEST_JOBLIB, X_test)
    _safe_joblib_dump(X_TRAIN_BASE_JOBLIB, X_train_base)
    _safe_joblib_dump(X_TEST_BASE_JOBLIB, X_test_base)
    _safe_joblib_dump(X_TRAIN_RAW_JOBLIB, X_train_raw)
    _safe_joblib_dump(X_TEST_RAW_JOBLIB, X_test_raw)
    _safe_joblib_dump(
        PREP_PATH,
        {
            "openfe": {
                "categorical_columns": prep.categorical_columns,
                "category_levels": prep.category_levels,
            },
            "baseline": {
                "categorical_columns": baseline_prep.categorical_columns,
                "category_levels": baseline_prep.category_levels,
            },
        },
    )
    _safe_dump_json(VERSIONS_PATH, {"package_versions": versions, "device": device})

    reqs_src = os.path.join("scripts", "requirements_openfe_lgbm.txt")
    if os.path.exists(reqs_src):
        _assert_allowed_write(REQS_COPY_PATH)
        with open(reqs_src, encoding="utf-8") as src, open(REQS_COPY_PATH, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    sha_lines = []
    for p in [
        MODEL_PATH,
        BASELINE_MODEL_PATH,
        OPENFE_OBJ_PATH,
        FEATURES_PATH,
        FORMULAS_PATH,
        PREP_PATH,
        X_TRAIN_JOBLIB,
        X_TEST_JOBLIB,
    ]:
        sha_lines.append(f"{sha256_file(p)}  {p}")
    _assert_allowed_write(SHA_PATH)
    with open(SHA_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sha_lines) + "\n")

    meta["sha256"] = {os.path.basename(line.split("  ", 1)[1]): line.split("  ", 1)[0] for line in sha_lines}
    _safe_dump_json(META_PATH, _json_ready(meta))
    if os.path.exists(FAILURE_PATH):
        os.remove(FAILURE_PATH)

    print(f"[train_openfe_lgbm] Wrote {MODEL_PATH}", flush=True)
    print(f"[train_openfe_lgbm] Wrote {OPENFE_OBJ_PATH} + {FORMULAS_PATH}", flush=True)
    print(f"[train_openfe_lgbm] Wrote {META_PATH} (freeze complete; no test metrics inside)", flush=True)
    print(
        f"[train_openfe_lgbm] OOF OpenFE+LGBM AUC={oof_openfe['AUC']} KS={oof_openfe['KS_Statistic']} | "
        f"OOF baseline raw20 AUC={oof_base['AUC']} KS={oof_base['KS_Statistic']}",
        flush=True,
    )
    print(
        f"[train_openfe_lgbm] employment_status gain_rank={emp_gain_rank} "
        f"perm_rank={emp_perm_rank} (TRAIN only)",
        flush=True,
    )
    print("[train_openfe_lgbm] STOPPING. Run python3 scripts/eval_openfe_lgbm.py for test metrics.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if not os.path.exists(FAILURE_PATH):
            try:
                _write_failure("train_openfe_lgbm crashed", traceback.format_exc())
            except Exception:
                pass
        raise
