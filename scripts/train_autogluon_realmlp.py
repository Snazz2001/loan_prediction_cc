"""
AutoGluon TabularPredictor + RealMLP (train / freeze only).

Frozen split via scripts/train.py::load_and_split UNCHANGED. Drops gender.
employment_status is a normal in-model feature. Exact 20 original columns
(same list/order as PR #4/#6/#7/#8). No IV drop, no VIF drop. No
monotone_constraints. No interaction_constraints.

Tune/fit on TRAIN only. This script never scores Test labels. Freeze with
test_looked_at=false / test_metrics=null BEFORE any Test discrimination
metric is computed.

If AutoGluon or RealMLP cannot install/run: STOP that one, write
artifacts/autogluon_FAILURE.md or artifacts/realmlp_FAILURE.md, and do NOT
silently swap a different architecture.

Does not write last-run artifacts/lgbm_model.joblib or PR #1–#8 prefixes.

Usage (from repo root): python3 scripts/train_autogluon_realmlp.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURE_AG_PATH = os.path.join("artifacts", "autogluon_FAILURE.md")
FAILURE_RM_PATH = os.path.join("artifacts", "realmlp_FAILURE.md")
FAILURE_PACK_PATH = os.path.join("artifacts", "ag_realmlp_FAILURE.md")


def _write_named_failure(path: str, title: str, detail: str) -> None:
    os.makedirs("artifacts", exist_ok=True)
    name = os.path.basename(path).replace("_FAILURE.md", "")
    body = (
        f"# {name} failure — STOP this model\n\n"
        f"**{title}**\n\n"
        f"This run stopped this model rather than silently swapping in a different "
        f"architecture and calling it {name}.\n\n"
        f"```\n{detail}\n```\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"[train_ag_realmlp] FAILURE written to {path}", flush=True)


import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from train import load_and_split
from utils.config import ARTIFACTS_DIR, RANDOM_STATE, RAW_TARGET_COL, TARGET
from utils.risk_skills import evaluate_discrimination_and_ks

CV_FOLDS = 5
EXPECTED_TRAIN_N = 14000
EXPECTED_TRAIN_BAD_RATE = 0.2001
EXPECTED_TRAIN_N_BADS = 2801
EXPECTED_TEST_N = 6000
EXPECTED_TEST_BAD_RATE = 0.2002
EXPECTED_TEST_N_BADS = 1201

AG_PRESET = "best_quality"
AG_TIME_LIMIT_S = 3600
AG_NUM_CPUS = max(1, int(os.cpu_count() or 1))
PERM_N = 500
PD_CLIP = 1e-6

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
    "delinquency_history",
    "public_records",
    "loan_term",
]

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
)

AG_PREDICTOR_DIR = os.path.join(ARTIFACTS_DIR, "autogluon_predictor")
AG_ARCHIVE_TAR = os.path.join(ARTIFACTS_DIR, "autogluon_predictor.tar.gz")
AG_ARCHIVE_PART_PREFIX = os.path.join(ARTIFACTS_DIR, "autogluon_predictor.tar.gz.part")
AG_LEADERBOARD_PATH = os.path.join(ARTIFACTS_DIR, "autogluon_leaderboard.csv")
AG_FI_PATH = os.path.join(ARTIFACTS_DIR, "autogluon_feature_importance.csv")
AG_MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "autogluon_predictor_manifest.json")

REALMLP_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "realmlp_model.joblib")
REALMLP_PREPROCESS_PATH = os.path.join(ARTIFACTS_DIR, "realmlp_preprocess.joblib")
REALMLP_PERM_PATH = os.path.join(ARTIFACTS_DIR, "realmlp_permutation_importance.json")
REALMLP_SHA_PATH = os.path.join(ARTIFACTS_DIR, "realmlp_sha256.txt")

STACK_META_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_stack_meta.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_y_test.csv")
REQS_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_requirements.txt")

ALLOWED_WRITE_PATHS = [
    FAILURE_AG_PATH,
    FAILURE_RM_PATH,
    FAILURE_PACK_PATH,
    AG_PREDICTOR_DIR,
    AG_LEADERBOARD_PATH,
    AG_FI_PATH,
    AG_MANIFEST_PATH,
    REALMLP_MODEL_PATH,
    REALMLP_PREPROCESS_PATH,
    REALMLP_PERM_PATH,
    REALMLP_SHA_PATH,
    STACK_META_PATH,
    META_JSON_PATH,
    OOF_PDS_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    REQS_PATH,
]

ALLOWED_DIR_PREFIXES = (
    os.path.abspath(AG_PREDICTOR_DIR) + os.sep,
    os.path.abspath(AG_ARCHIVE_PART_PREFIX),
)

REALMLP_FIT_KWARGS = {
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "n_cv": 1,
    "n_refit": 0,
    "verbosity": 1,
}


class FeatureFramePrep:
    """Column order + category dtypes. No learned statistics; never uses test labels."""

    def __init__(self, feature_order: list[str], cat_cols: list[str]):
        self.feature_order = list(feature_order)
        self.cat_cols = list(cat_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise RuntimeError(f"FeatureFramePrep missing columns: {missing}")
        out = df[self.feature_order].copy()
        extra = [c for c in out.columns if c not in self.feature_order]
        if extra:
            raise RuntimeError(f"FeatureFramePrep unexpected columns: {extra}")
        for c in self.cat_cols:
            out[c] = out[c].astype("category")
        return out


def seed_all(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass


def _pkg_version(dist: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(dist)
    except Exception:  # noqa: BLE001
        return None


def collect_package_versions() -> dict:
    torch_ver = None
    try:
        import torch

        torch_ver = getattr(torch, "__version__", _pkg_version("torch"))
    except Exception:  # noqa: BLE001
        torch_ver = _pkg_version("torch")
    return {
        "python": sys.version.split()[0],
        "autogluon": _pkg_version("autogluon.tabular"),
        "autogluon.tabular": _pkg_version("autogluon.tabular"),
        "torch": torch_ver,
        "pytabkit": _pkg_version("pytabkit"),
        "sklearn": _pkg_version("scikit-learn"),
        "pandas": _pkg_version("pandas"),
        "numpy": _pkg_version("numpy"),
        "joblib": _pkg_version("joblib"),
        "lightgbm": _pkg_version("lightgbm"),
        "xgboost": _pkg_version("xgboost"),
        "catboost": _pkg_version("catboost"),
    }


def _list_found_files() -> list[str]:
    found = []
    for root, _dirs, files in os.walk("."):
        if any(skip in root.split(os.sep) for skip in (".git", "__pycache__", ".cursor")):
            continue
        for fn in files:
            found.append(os.path.join(root, fn).lstrip("./"))
    return sorted(found)[:200]


def _require_pipeline_files() -> None:
    missing = [p for p in REQUIRED_PATHS if not os.path.exists(p)]
    if missing:
        detail = (
            "Required pipeline files missing: "
            + ", ".join(missing)
            + ". Files found: "
            + ", ".join(_list_found_files())
        )
        _write_named_failure(FAILURE_PACK_PATH, "Could not reconstruct pipeline files", detail)
        raise FileNotFoundError(detail)


def _path_is_forbidden(path: str) -> bool:
    base = os.path.basename(path)
    if base == "lgbm_model.joblib":
        return True
    rel = path.replace("\\", "/")
    if rel.endswith("/lgbm_model.joblib") or rel.endswith("artifacts/lgbm_model.joblib"):
        return True
    for pref in FORBIDDEN_PREFIXES:
        if base.startswith(pref):
            return True
        if f"/{pref}" in f"/{rel}":
            return True
    return False


def _assert_allowed_writes(paths: list[str]) -> None:
    allowed_set = {os.path.abspath(p) for p in ALLOWED_WRITE_PATHS}
    for path in paths:
        abs_path = os.path.abspath(path)
        if (
            abs_path not in allowed_set
            and not any(abs_path.startswith(p) for p in ALLOWED_DIR_PREFIXES)
            and not os.path.basename(abs_path).startswith("autogluon_predictor.tar.gz.part")
        ):
            raise RuntimeError(f"Refusing to write undeclared artifact: {path}")
        if _path_is_forbidden(path):
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
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _clip_pd(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), PD_CLIP, 1.0 - PD_CLIP)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_manifest(root: str) -> dict:
    files = []
    h_all = hashlib.sha256()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, root)
            digest = _file_sha256(fp)
            size = os.path.getsize(fp)
            files.append({"path": rel, "sha256": digest, "bytes": int(size)})
            h_all.update(rel.encode())
            h_all.update(digest.encode())
    return {
        "root": root,
        "n_files": len(files),
        "total_bytes": int(sum(f["bytes"] for f in files)),
        "aggregate_sha256": h_all.hexdigest(),
        "files": files,
    }


def write_ag_split_archive(part_size_bytes: int = 90 * 1024 * 1024) -> list[str]:
    """GitHub-safe split of the pruned predictor. Does not refit."""
    import glob
    import tarfile

    if not os.path.isdir(AG_PREDICTOR_DIR):
        raise FileNotFoundError(AG_PREDICTOR_DIR)
    for old in glob.glob(AG_ARCHIVE_PART_PREFIX + "*"):
        os.remove(old)
    if os.path.exists(AG_ARCHIVE_TAR):
        os.remove(AG_ARCHIVE_TAR)
    with tarfile.open(AG_ARCHIVE_TAR, "w:gz") as tar:
        tar.add(AG_PREDICTOR_DIR, arcname="autogluon_predictor")
    parts = []
    with open(AG_ARCHIVE_TAR, "rb") as src:
        idx = 0
        while True:
            chunk = src.read(part_size_bytes)
            if not chunk:
                break
            part = f"{AG_ARCHIVE_PART_PREFIX}{idx:02d}"
            with open(part, "wb") as dest:
                dest.write(chunk)
            parts.append(part)
            idx += 1
    os.remove(AG_ARCHIVE_TAR)
    return parts


def ensure_ag_predictor_dir() -> str:
    """Load path for TabularPredictor. Reconstruct from split archive if needed. Never fits."""
    import glob
    import tarfile
    import tempfile

    marker = os.path.join(AG_PREDICTOR_DIR, "metadata.json")
    models_dir = os.path.join(AG_PREDICTOR_DIR, "models")
    if os.path.isfile(marker) and os.path.isdir(models_dir):
        return AG_PREDICTOR_DIR
    parts = sorted(glob.glob(AG_ARCHIVE_PART_PREFIX + "*"))
    if not parts:
        raise FileNotFoundError(
            f"AutoGluon predictor missing at {AG_PREDICTOR_DIR} and no split archive "
            f"{AG_ARCHIVE_PART_PREFIX}* found"
        )
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        for part in parts:
            with open(part, "rb") as fh:
                tmp.write(fh.read())
    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(ARTIFACTS_DIR)
    finally:
        os.remove(tmp_path)
    if not os.path.isfile(marker):
        raise FileNotFoundError(f"Extracted AutoGluon predictor missing {marker}")
    return AG_PREDICTOR_DIR


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
            f"{context}: gender must be ABSENT from in-model X. Found gender in {cols}"
        )
    if "employment_status" not in cols:
        raise RuntimeError(
            f"{context}: employment_status must be PRESENT as a normal in-model feature. "
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


def split_num_cat(feature_order: list[str], raw_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols, cat_cols = [], []
    for col in feature_order:
        is_explicit_cat = col in EXPLICIT_CAT_COLS
        is_object = not pd.api.types.is_numeric_dtype(raw_df[col])
        if is_explicit_cat or is_object:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    if "employment_status" not in cat_cols:
        raise RuntimeError("employment_status must be a categorical in-model column")
    if "gender" in num_cols or "gender" in cat_cols:
        raise RuntimeError("gender leaked into num/cat lists")
    return num_cols, cat_cols


def verify_frozen_split(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    train_n = len(train_df)
    test_n = len(test_df)
    train_bads = int(train_df[TARGET].sum())
    test_bads = int(test_df[TARGET].sum())
    train_rate = round(float(train_df[TARGET].mean()), 4)
    test_rate = round(float(test_df[TARGET].mean()), 4)
    if train_n != EXPECTED_TRAIN_N or train_rate != EXPECTED_TRAIN_BAD_RATE or train_bads != EXPECTED_TRAIN_N_BADS:
        detail = (
            "Could not reconstruct frozen Train split. "
            f"got n={train_n} bad_rate={train_rate} bads={train_bads}; "
            f"expected n={EXPECTED_TRAIN_N} bad_rate={EXPECTED_TRAIN_BAD_RATE} bads={EXPECTED_TRAIN_N_BADS}."
        )
        _write_named_failure(FAILURE_PACK_PATH, "Split reconstruction failed", detail)
        raise RuntimeError(detail)
    if test_n != EXPECTED_TEST_N or test_rate != EXPECTED_TEST_BAD_RATE or test_bads != EXPECTED_TEST_N_BADS:
        detail = (
            "Could not reconstruct frozen Test split. "
            f"got n={test_n} bad_rate={test_rate} bads={test_bads}; "
            f"expected n={EXPECTED_TEST_N} bad_rate={EXPECTED_TEST_BAD_RATE} bads={EXPECTED_TEST_N_BADS}."
        )
        _write_named_failure(FAILURE_PACK_PATH, "Split reconstruction failed", detail)
        raise RuntimeError(detail)
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


def _rank_of(feature: str, ranking: list[dict]) -> int | None:
    for i, row in enumerate(ranking, start=1):
        if row["feature"] == feature:
            return i
    return None


def permutation_importance_auc(predict_fn, X: pd.DataFrame, y: pd.Series, *, n: int, seed: int, context: str) -> dict:
    _assert_in_model_columns(list(X.columns), f"{context} permutation X")
    rng = np.random.RandomState(seed)
    n_use = min(int(n), len(X))
    idx = rng.choice(len(X), size=n_use, replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].to_numpy()
    base_pred = np.asarray(predict_fn(Xs), dtype=float)
    base_auc = float(roc_auc_score(ys, base_pred))
    ranking = []
    for col in list(X.columns):
        Xperm = Xs.copy()
        Xperm[col] = rng.permutation(Xperm[col].to_numpy())
        perm_pred = np.asarray(predict_fn(Xperm), dtype=float)
        perm_auc = float(roc_auc_score(ys, perm_pred))
        ranking.append(
            {
                "feature": col,
                "auc_drop": round(base_auc - perm_auc, 6),
                "perm_auc": round(perm_auc, 6),
            }
        )
    ranking = sorted(ranking, key=lambda r: r["auc_drop"], reverse=True)
    names = [r["feature"] for r in ranking]
    _assert_in_model_columns(names, f"{context} permutation ranking", require_order=False)
    return {
        "method": (
            f"permutation importance on TRAIN sample n={n_use}, seed={seed}, "
            f"metric=AUC drop vs baseline AUC={round(base_auc, 6)} — {context}"
        ),
        "n_rows": int(n_use),
        "baseline_auc": round(base_auc, 6),
        "auc_drop_by_feature": ranking,
        "employment_status_rank": _rank_of("employment_status", ranking),
        "gender_present": False,
        "employment_status_present": True,
    }


def _device_info() -> dict:
    cuda = False
    torch_ver = None
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        torch_ver = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return {
        "device": "cpu",
        "torch_cuda_available": cuda,
        "torch_version": torch_ver,
        "n_cpus": AG_NUM_CPUS,
        "note": "This VM has no GPU; AutoGluon extreme preset was not run.",
    }


def _ag_positive_pd(predictor, proba) -> np.ndarray:
    pos = predictor.positive_class
    if isinstance(proba, pd.DataFrame):
        if pos in proba.columns:
            return proba[pos].to_numpy(dtype=float)
        if str(pos) in proba.columns:
            return proba[str(pos)].to_numpy(dtype=float)
        if 1 in proba.columns:
            return proba[1].to_numpy(dtype=float)
        if "1" in proba.columns:
            return proba["1"].to_numpy(dtype=float)
        return proba.iloc[:, -1].to_numpy(dtype=float)
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr[:, -1]


def _scan_name_seats(names: list[str]) -> dict:
    up = [str(n).upper() for n in names]
    def _hit(*needles: str) -> list[str]:
        hits = []
        for raw, u in zip(names, up):
            if any(nd in u for nd in needles):
                hits.append(str(raw))
        return hits

    tabpfn = _hit("TABPFN", "TABPFN")
    tabm = _hit("TABM")
    realmlp = _hit("REALMLP")
    return {
        "TabPFN_in_ensemble": bool(tabpfn),
        "TabM_in_ensemble": bool(tabm),
        "RealMLP_in_ensemble": bool(realmlp),
        "TabPFN_model_names": tabpfn,
        "TabM_model_names": tabm,
        "RealMLP_model_names": realmlp,
        "all_model_names": [str(n) for n in names],
    }


def train_autogluon(train_frame: pd.DataFrame, y_train: pd.Series, prep: FeatureFramePrep) -> dict:
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:  # noqa: BLE001
        _write_named_failure(
            FAILURE_AG_PATH,
            "autogluon.tabular.TabularPredictor import failed",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
        return {"ok": False, "failure_path": FAILURE_AG_PATH, "error": f"{type(exc).__name__}: {exc}"}

    try:
        import autogluon.tabular as ag_tab

        ag_version = getattr(ag_tab, "__version__", _pkg_version("autogluon.tabular"))
        if os.path.exists(AG_PREDICTOR_DIR):
            shutil.rmtree(AG_PREDICTOR_DIR)
        os.makedirs(AG_PREDICTOR_DIR, exist_ok=True)

        X = prep.transform(train_frame)
        _assert_in_model_columns(list(X.columns), "AutoGluon X_train")
        fit_df = X.copy()
        fit_df[TARGET] = y_train.to_numpy()
        if "gender" in fit_df.columns:
            raise RuntimeError("gender present in AutoGluon fit frame")
        if RAW_TARGET_COL in fit_df.columns:
            raise RuntimeError("loan_paid_back present in AutoGluon fit frame")

        print(
            f"[train_ag] TabularPredictor presets={AG_PRESET!r} time_limit={AG_TIME_LIMIT_S}s "
            f"num_gpus=0 num_cpus={AG_NUM_CPUS} (extreme skipped: no GPU)",
            flush=True,
        )
        predictor = TabularPredictor(
            label=TARGET,
            problem_type="binary",
            eval_metric="roc_auc",
            path=AG_PREDICTOR_DIR,
            verbosity=2,
            positive_class=1,
        )
        predictor.fit(
            fit_df,
            presets=AG_PRESET,
            time_limit=AG_TIME_LIMIT_S,
            num_cpus=AG_NUM_CPUS,
            num_gpus=0,
        )

        lb = predictor.leaderboard(silent=True)
        lb.to_csv(AG_LEADERBOARD_PATH, index=False)
        model_names = list(predictor.model_names())
        seats = _scan_name_seats(model_names)
        best_model = predictor.model_best
        top_rows = lb.head(15).to_dict(orient="records")

        oof_ok = False
        oof_pd = None
        oof_error = None
        try:
            oof_proba = predictor.predict_proba_oof(train_data=fit_df)
            oof_pd = _clip_pd(_ag_positive_pd(predictor, oof_proba))
            if len(oof_pd) != len(y_train):
                raise RuntimeError(f"AG OOF length {len(oof_pd)} != train {len(y_train)}")
            oof_ok = True
        except Exception as exc:  # noqa: BLE001
            oof_error = f"{type(exc).__name__}: {exc}"
            print(f"[train_ag] predict_proba_oof failed (stack/average OOF skipped): {oof_error}", flush=True)

        oof_metrics = None
        if oof_ok:
            oof_metrics = evaluate_discrimination_and_ks(y_train.to_numpy(), oof_pd)
            print(
                f"[train_ag] OOF AUC={oof_metrics['AUC']} KS={oof_metrics['KS_Statistic']} "
                f"Gini={oof_metrics['Gini']}",
                flush=True,
            )

        def _predict_raw(frame: pd.DataFrame) -> np.ndarray:
            xt = prep.transform(frame)
            return _clip_pd(_ag_positive_pd(predictor, predictor.predict_proba(xt)))

        print("[train_ag] TRAIN-only permutation importance n=500 seed=42...", flush=True)
        try:
            perm = permutation_importance_auc(
                _predict_raw, train_frame, y_train, n=PERM_N, seed=RANDOM_STATE, context="autogluon"
            )
            perm_error = None
        except Exception as exc:  # noqa: BLE001
            perm = {
                "method": "skipped",
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "gender_present": False,
                "employment_status_present": True,
                "employment_status_rank": None,
                "auc_drop_by_feature": [],
            }
            perm_error = perm["skip_reason"]
            print(f"[train_ag] permutation skipped: {perm_error}", flush=True)

        ag_fi = None
        try:
            fi = predictor.feature_importance(
                data=fit_df,
                silent=True,
                subsample_size=PERM_N,
                num_shuffle_sets=5,
            )
            fi.to_csv(AG_FI_PATH)
            ag_fi = {
                "method": "TabularPredictor.feature_importance on TRAIN only (subsample_size=500, num_shuffle_sets=5)",
                "columns": list(fi.columns),
                "ranking": [
                    {"feature": str(idx), **{str(c): _json_ready(val) for c, val in row.items()}}
                    for idx, row in fi.iterrows()
                ],
                "employment_status_rank": (
                    int(list(fi.index).index("employment_status") + 1)
                    if "employment_status" in list(fi.index)
                    else None
                ),
                "gender_present": bool("gender" in list(fi.index)),
            }
            if ag_fi["gender_present"]:
                raise RuntimeError("AutoGluon feature_importance ranking contains gender")
        except Exception as exc:  # noqa: BLE001
            print(f"[train_ag] feature_importance failed: {type(exc).__name__}: {exc}", flush=True)
            ag_fi = {"method": "failed", "error": f"{type(exc).__name__}: {exc}"}

        # Keep only models required to score the frozen best ensemble so the
        # predictor can be SHA-matched and pushed (GitHub 100MB file cap).
        # This does not refit and does not look at Test labels. OOF PDs are
        # already extracted from the full bagged fit.
        n_before = len(predictor.model_names())
        predictor.delete_models(models_to_keep="best", dry_run=False)
        n_after = len(predictor.model_names())
        print(
            f"[train_ag] pruned unused models for persistence: {n_before} -> {n_after} "
            f"(kept best={predictor.model_best})",
            flush=True,
        )
        archive_parts = write_ag_split_archive()
        print(f"[train_ag] wrote GitHub-safe split archive: {archive_parts}", flush=True)

        manifest = _dir_manifest(AG_PREDICTOR_DIR)
        with open(AG_MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        info = {
            "ok": True,
            "library": "autogluon.tabular.TabularPredictor",
            "class": "autogluon.tabular.TabularPredictor",
            "version": ag_version,
            "preset_requested": AG_PRESET,
            "preset_actually_run": AG_PRESET,
            "extreme_attempted": False,
            "extreme_skip_reason": "No GPU (torch.cuda.is_available() is false); extreme requires CUDA.",
            "time_limit_s": AG_TIME_LIMIT_S,
            "num_gpus": 0,
            "num_cpus": AG_NUM_CPUS,
            "problem_type": "binary",
            "eval_metric": "roc_auc",
            "positive_class": 1,
            "test_passed_to_fit": False,
            "best_model": str(best_model),
            "model_names": model_names,
            "persistence_pruned": True,
            "n_models_before_prune": int(n_before),
            "n_models_after_prune": int(n_after),
            "models_kept_for_inference": [str(n) for n in predictor.model_names()],
            "predictor_archive_parts": archive_parts,
            "leaderboard_top": top_rows,
            "tabpfn_tabm_realmlp_seats": seats,
            "oof_available": oof_ok,
            "oof_error": oof_error,
            "oof_metrics": oof_metrics,
            "permutation": perm,
            "feature_importance": ag_fi,
            "predictor_path": AG_PREDICTOR_DIR,
            "manifest_aggregate_sha256": manifest["aggregate_sha256"],
            "manifest_n_files": manifest["n_files"],
            "manifest_total_bytes": manifest["total_bytes"],
        }
        return {"ok": True, "predictor": predictor, "oof_pd": oof_pd, "info": info, "predict_fn": _predict_raw}
    except Exception as exc:  # noqa: BLE001
        _write_named_failure(
            FAILURE_AG_PATH,
            "AutoGluon TabularPredictor.fit failed",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
        return {"ok": False, "failure_path": FAILURE_AG_PATH, "error": f"{type(exc).__name__}: {exc}"}


def _make_realmlp():
    from pytabkit import RealMLP_TD_Classifier

    return RealMLP_TD_Classifier(**REALMLP_FIT_KWARGS)


def _realmlp_pd(model, frame: pd.DataFrame, prep: FeatureFramePrep) -> np.ndarray:
    xt = prep.transform(frame)
    proba = model.predict_proba(xt)
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        return _clip_pd(arr)
    return _clip_pd(arr[:, 1])


def train_realmlp(train_frame: pd.DataFrame, y_train: pd.Series, prep: FeatureFramePrep) -> dict:
    try:
        from pytabkit import RealMLP_TD_Classifier  # noqa: F401
        from pytabkit.models.sklearn.default_params import DefaultParams
    except Exception as orig_exc:  # noqa: BLE001
        # Do not silently swap a different architecture.
        _write_named_failure(
            FAILURE_RM_PATH,
            "pytabkit RealMLP_TD_Classifier import failed — STOP RealMLP (no architecture swap)",
            f"{type(orig_exc).__name__}: {orig_exc}\n\n{traceback.format_exc()}",
        )
        return {"ok": False, "failure_path": FAILURE_RM_PATH, "error": f"{type(orig_exc).__name__}: {orig_exc}"}

    try:
        td_defaults = dict(getattr(DefaultParams, "RealMLP_TD_CLASS", {}) or {})
        X = prep.transform(train_frame)
        y = y_train.to_numpy()
        _assert_in_model_columns(list(X.columns), "RealMLP X_train")
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_pd = np.zeros(len(y), dtype=float)
        fold_aucs, fold_ks = [], []
        print(
            f"[train_realmlp] RealMLP_TD_Classifier published default; "
            f"StratifiedKFold {CV_FOLDS} shuffle=True random_state={RANDOM_STATE} for OOF only. "
            f"HPO (RealMLP_HPO_Classifier) NOT run — paper-faithful TD default on CPU.",
            flush=True,
        )
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            model = _make_realmlp()
            X_tr = X.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            model.fit(X_tr, y_tr, cat_col_names=list(prep.cat_cols))
            pred_va = _clip_pd(np.asarray(model.predict_proba(X_va), dtype=float)[:, 1])
            oof_pd[va_idx] = pred_va
            fold_aucs.append(float(roc_auc_score(y[va_idx], pred_va)))
            fold_ks.append(_ks(y[va_idx], pred_va))
            print(
                f"[train_realmlp] fold {fold_i}/{CV_FOLDS} AUC={fold_aucs[-1]:.6f} KS={fold_ks[-1]:.6f}",
                flush=True,
            )

        oof_metrics = evaluate_discrimination_and_ks(y, oof_pd)
        print(
            f"[train_realmlp] OOF AUC={oof_metrics['AUC']} KS={oof_metrics['KS_Statistic']} "
            f"Gini={oof_metrics['Gini']}",
            flush=True,
        )

        print("[train_realmlp] Refit RealMLP-TD on ALL 14000 train rows (internal val split for early stopping).", flush=True)
        model_full = _make_realmlp()
        model_full.fit(X, y_train, cat_col_names=list(prep.cat_cols))
        if hasattr(model_full, "to"):
            try:
                model_full.to("cpu")
            except Exception:  # noqa: BLE001
                pass

        joblib.dump(model_full, REALMLP_MODEL_PATH)
        joblib.dump(prep, REALMLP_PREPROCESS_PATH)
        sha = _file_sha256(REALMLP_MODEL_PATH)
        with open(REALMLP_SHA_PATH, "w", encoding="utf-8") as fh:
            fh.write(f"{sha}  {REALMLP_MODEL_PATH}\n")
            fh.write(f"{_file_sha256(REALMLP_PREPROCESS_PATH)}  {REALMLP_PREPROCESS_PATH}\n")

        def _predict_raw(frame: pd.DataFrame) -> np.ndarray:
            return _realmlp_pd(model_full, frame, prep)

        print("[train_realmlp] TRAIN-only permutation importance n=500 seed=42...", flush=True)
        try:
            perm = permutation_importance_auc(
                _predict_raw, train_frame, y_train, n=PERM_N, seed=RANDOM_STATE, context="realmlp"
            )
            perm_error = None
        except Exception as exc:  # noqa: BLE001
            perm = {
                "method": "skipped",
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "gender_present": False,
                "employment_status_present": True,
                "employment_status_rank": None,
                "auc_drop_by_feature": [],
            }
            perm_error = perm["skip_reason"]
            print(f"[train_realmlp] permutation skipped: {perm_error}", flush=True)

        with open(REALMLP_PERM_PATH, "w", encoding="utf-8") as fh:
            json.dump(_json_ready(perm), fh, ensure_ascii=False, indent=2)

        info = {
            "ok": True,
            "library": "pytabkit",
            "class": "pytabkit.RealMLP_TD_Classifier",
            "paper": "Holzmüller, Grinsztajn, Steinwart — Better by Default (NeurIPS 2024)",
            "version": _pkg_version("pytabkit"),
            "config_name": "RealMLP-TD (published tuned default)",
            "hpo_run": False,
            "hpo_reason": (
                "Library exposes RealMLP_HPO_Classifier, but this pack uses the published "
                "RealMLP-TD default explicitly (paper-faithful). CPU-only 4-core budget cannot "
                "complete paper-scale HPO (n_hyperopt_steps=50 × n_cv=5) without cutting the "
                "search short. OOF AUC is reported from StratifiedKFold 5 on the TD default; "
                "it is not used to pick among HPO configs."
            ),
            "constructor_kwargs": dict(REALMLP_FIT_KWARGS),
            "published_td_defaults": td_defaults,
            "cat_col_names": list(prep.cat_cols),
            "cv_folds": CV_FOLDS,
            "cv_shuffle": True,
            "cv_random_state": RANDOM_STATE,
            "fold_aucs": [round(a, 6) for a in fold_aucs],
            "fold_ks": [round(a, 6) for a in fold_ks],
            "oof_metrics": oof_metrics,
            "permutation": perm,
            "model_path": REALMLP_MODEL_PATH,
            "preprocess_path": REALMLP_PREPROCESS_PATH,
            "model_sha256": sha,
            "device": "cpu",
        }
        return {"ok": True, "model": model_full, "oof_pd": oof_pd, "info": info, "predict_fn": _predict_raw}
    except Exception as exc:  # noqa: BLE001
        _write_named_failure(
            FAILURE_RM_PATH,
            "RealMLP_TD_Classifier training failed — STOP RealMLP (no architecture swap)",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
        return {"ok": False, "failure_path": FAILURE_RM_PATH, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    seed_all(RANDOM_STATE)
    _require_pipeline_files()
    _assert_load_and_split_unchanged()

    print("[train_ag_realmlp] load_and_split() unchanged...", flush=True)
    train_df, test_df, feature_cols = load_and_split()
    verify_frozen_split(train_df, test_df, feature_cols)

    leftover = [c for c in feature_cols if c not in DROPPED_FROM_MODEL]
    in_model_features = list(EXPECTED_IN_MODEL_FEATURES)
    _assert_in_model_columns(in_model_features, "in_model_features after dropping gender")
    if set(leftover) != set(EXPECTED_IN_MODEL_FEATURES):
        raise RuntimeError(f"load_and_split minus gender != expected 20. leftover={leftover}")
    for req in REQUIRED_IN_MODEL:
        if req not in in_model_features:
            raise RuntimeError(f"{req} missing from in-model features")

    raw_X_train = train_df[in_model_features].copy()
    raw_X_test = test_df[in_model_features].copy()
    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()
    _assert_in_model_columns(list(raw_X_train.columns), "raw_X_train")
    _assert_in_model_columns(list(raw_X_test.columns), "raw_X_test")
    if RAW_TARGET_COL in raw_X_train.columns:
        raise RuntimeError("loan_paid_back present as a feature")

    num_cols, cat_cols = split_num_cat(in_model_features, raw_X_train)
    prep = FeatureFramePrep(in_model_features, cat_cols)
    print(f"[train_ag_realmlp] num_cols={num_cols}", flush=True)
    print(f"[train_ag_realmlp] cat_cols={cat_cols}", flush=True)

    raw_X_train.to_csv(X_TRAIN_PATH, index=False)
    raw_X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(TARGET).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(TARGET).to_csv(Y_TEST_PATH, index=False)

    ag_result = train_autogluon(raw_X_train, y_train, prep)
    rm_result = train_realmlp(raw_X_train, y_train, prep)

    oof_df = pd.DataFrame({"y": y_train.to_numpy()})
    oof_ag = ag_result.get("oof_pd") if ag_result.get("ok") else None
    oof_rm = rm_result.get("oof_pd") if rm_result.get("ok") else None
    if oof_ag is not None:
        oof_df["pd_autogluon"] = oof_ag
    if oof_rm is not None:
        oof_df["pd_realmlp"] = oof_rm

    stack_info = {
        "built": False,
        "reason": None,
        "meta_coefs": None,
        "intercept": None,
        "oof_metrics": None,
    }
    avg_info = {"built": False, "reason": None, "oof_metrics": None}

    both_oof = oof_ag is not None and oof_rm is not None
    if both_oof:
        avg_pd = _clip_pd(0.5 * np.asarray(oof_ag) + 0.5 * np.asarray(oof_rm))
        oof_df["pd_equal_weight_avg"] = avg_pd
        avg_info = {
            "built": True,
            "reason": "equal-weight average of leak-free AutoGluon OOF PD and RealMLP OOF PD",
            "oof_metrics": evaluate_discrimination_and_ks(y_train.to_numpy(), avg_pd),
        }
        meta = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        )
        X_meta = np.column_stack([_clip_pd(oof_ag), _clip_pd(oof_rm)])
        meta.fit(X_meta, y_train.to_numpy())
        stack_oof = _clip_pd(meta.predict_proba(X_meta)[:, 1])
        oof_df["pd_stack_lr"] = stack_oof
        joblib.dump(meta, STACK_META_PATH)
        stack_info = {
            "built": True,
            "reason": "OOF logistic stack fitted on train OOFs only (columns: pd_autogluon, pd_realmlp)",
            "class": "sklearn.linear_model.LogisticRegression",
            "constructor": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
            "feature_order": ["pd_autogluon", "pd_realmlp"],
            "meta_coefs": {
                "pd_autogluon": float(meta.coef_[0][0]),
                "pd_realmlp": float(meta.coef_[0][1]),
            },
            "intercept": float(meta.intercept_[0]),
            "oof_metrics": evaluate_discrimination_and_ks(y_train.to_numpy(), stack_oof),
            "path": STACK_META_PATH,
        }
        print(
            f"[train_ag_realmlp] stack coefs ag={stack_info['meta_coefs']['pd_autogluon']:.6f} "
            f"realmlp={stack_info['meta_coefs']['pd_realmlp']:.6f} intercept={stack_info['intercept']:.6f}",
            flush=True,
        )
    else:
        missing = []
        if oof_ag is None:
            missing.append("AutoGluon OOF")
        if oof_rm is None:
            missing.append("RealMLP OOF")
        reason = "clean OOFs unavailable for: " + ", ".join(missing)
        stack_info["reason"] = reason
        avg_info["reason"] = reason
        print(f"[train_ag_realmlp] skip stack/average OOF: {reason}", flush=True)

    oof_df.to_csv(OOF_PDS_PATH, index=False)

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_ag_realmlp] FREEZE at {freeze_timestamp} (before any test metrics)", flush=True)

    pkg_versions = collect_package_versions()
    req_lines = [f"{k}=={v}" for k, v in pkg_versions.items() if v]
    with open(REQS_PATH, "w", encoding="utf-8") as fh:
        fh.write("# Versions actually used by scripts/train_autogluon_realmlp.py\n")
        fh.write("\n".join(req_lines) + "\n")

    written = [
        META_JSON_PATH,
        OOF_PDS_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        REQS_PATH,
    ]
    if ag_result.get("ok"):
        written.extend([AG_PREDICTOR_DIR, AG_LEADERBOARD_PATH, AG_MANIFEST_PATH])
        if os.path.exists(AG_FI_PATH):
            written.append(AG_FI_PATH)
    else:
        written.append(FAILURE_AG_PATH)
    if rm_result.get("ok"):
        written.extend(
            [REALMLP_MODEL_PATH, REALMLP_PREPROCESS_PATH, REALMLP_PERM_PATH, REALMLP_SHA_PATH]
        )
    else:
        written.append(FAILURE_RM_PATH)
    if stack_info.get("built"):
        written.append(STACK_META_PATH)
    _assert_allowed_writes(written)

    meta_record = {
        "model_type": "autogluon_realmlp",
        "freeze_timestamp_utc": freeze_timestamp,
        "test_looked_at": False,
        "test_metrics": None,
        "test_labels_used_to_fit_or_select": False,
        "computed_after_freeze": False,
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
        "cv_folds": CV_FOLDS,
        "device": _device_info(),
        "package_versions": pkg_versions,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
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
        "autogluon": ag_result.get("info") if ag_result.get("ok") else {"ok": False, "error": ag_result.get("error"), "failure_path": ag_result.get("failure_path")},
        "realmlp": rm_result.get("info") if rm_result.get("ok") else {"ok": False, "error": rm_result.get("error"), "failure_path": rm_result.get("failure_path")},
        "equal_weight_average": avg_info,
        "oof_stack": stack_info,
        "oof_metrics": {
            "autogluon": (ag_result.get("info") or {}).get("oof_metrics") if ag_result.get("ok") else None,
            "realmlp": (rm_result.get("info") or {}).get("oof_metrics") if rm_result.get("ok") else None,
            "equal_weight_average": avg_info.get("oof_metrics"),
            "stack_lr": stack_info.get("oof_metrics"),
        },
        "train_explains": {
            "autogluon_permutation": (ag_result.get("info") or {}).get("permutation") if ag_result.get("ok") else None,
            "autogluon_feature_importance": (ag_result.get("info") or {}).get("feature_importance") if ag_result.get("ok") else None,
            "realmlp_permutation": (rm_result.get("info") or {}).get("permutation") if rm_result.get("ok") else None,
        },
        "artifact_paths": {
            "autogluon_predictor": AG_PREDICTOR_DIR,
            "realmlp_model": REALMLP_MODEL_PATH,
            "realmlp_preprocess": REALMLP_PREPROCESS_PATH,
            "stack_meta": STACK_META_PATH if stack_info.get("built") else None,
            "meta": META_JSON_PATH,
            "oof_pds": OOF_PDS_PATH,
            "X_train": X_TRAIN_PATH,
            "y_train": Y_TRAIN_PATH,
            "X_test": X_TEST_PATH,
            "y_test": Y_TEST_PATH,
            "requirements": REQS_PATH,
        },
        "did_not_write": [
            "artifacts/lgbm_model.joblib",
            "artifacts/lgbm_linear_tree_*",
            "artifacts/linear_tree_*",
            "artifacts/stack_lr_rf_lgbm_*",
            "artifacts/lgbm_no_gender_emp_overlay_*",
            "artifacts/lgbm_emp_overlay_*",
            "artifacts/lgbm_emp_in_*",
            "artifacts/lgbm_dart_monotone_*",
            "artifacts/dart_monotone_*",
            "artifacts/ensemble_no_gender_*",
            "artifacts/ensemble_fm_ft_*",
            "artifacts/danet_*",
        ],
        "notes": (
            "Two models: AutoGluon TabularPredictor (best_quality, CPU) and pytabkit "
            "RealMLP_TD_Classifier (published TD default). gender dropped. employment_status "
            "is a normal in-model feature. No monotone_constraints. No interaction_constraints. "
            "This train script does not compute Test AUC/KS/Gini/PSI."
        ),
    }

    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta_record), fh, ensure_ascii=False, indent=2)

    print(f"[train_ag_realmlp] FREEZE {freeze_timestamp}", flush=True)
    print("[train_ag_realmlp] test_looked_at=false test_metrics=null", flush=True)
    print(f"[train_ag_realmlp] AutoGluon ok={ag_result.get('ok')} RealMLP ok={rm_result.get('ok')}", flush=True)
    print("[train_ag_realmlp] STOP — eval script computes test metrics after freeze.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(FAILURE_PACK_PATH):
            raise
        _write_named_failure(
            FAILURE_PACK_PATH,
            "Uncaught AutoGluon+RealMLP training failure",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
        raise
