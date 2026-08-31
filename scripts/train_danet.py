"""
DANet candidate (train / tune only) — no gender, frozen split.

ONE Deep Abstract Network (Huang/Chen et al., AAAI 2022). Official
WhatAShot/DANet AbstractLayer / ABSTLAY (Entmax15 k-masks), vendored under
third_party/danet/. No stack, no extra bases, no blend.

Frozen split via scripts/train.py::load_and_split UNCHANGED. Drops gender.
employment_status is a normal in-model feature. Exact 20 original columns
kept (no IV drop, no VIF drop). Train-only QuantileTransformer + category
maps. NO monotone_constraints. NO interaction_constraints.

Tune on TRAIN OOF AUC only (short CPU grid). Freeze with
test_looked_at=false / test_metrics=null BEFORE any Test discrimination
metric is computed. This script never scores Test labels.

Does not write last-run artifacts/lgbm_model.joblib or PR #1–#7 prefixes.

If DANet cannot be imported or trained: STOP, write artifacts/danet_FAILURE.md.

Usage (from repo root): python3 scripts/train_danet.py
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURE_NOTE_PATH = os.path.join("artifacts", "danet_FAILURE.md")


def _write_failure_and_stop(title: str, detail: str) -> None:
    os.makedirs("artifacts", exist_ok=True)
    body = (
        f"# DANet failure — STOP\n\n"
        f"**{title}**\n\n"
        f"This run stopped rather than silently swapping in a different architecture "
        f"and calling it DANet.\n\n"
        f"```\n{detail}\n```\n"
    )
    with open(FAILURE_NOTE_PATH, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"[train_danet] FAILURE written to {FAILURE_NOTE_PATH}", flush=True)
    raise RuntimeError(f"{title}\n{detail}")


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # noqa: BLE001
    _write_failure_and_stop(
        "torch import failed",
        f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
    )

try:
    from third_party.danet import (
        AbstractLayer,
        DANet,
        LearnableLocality,
        UPSTREAM_COMMIT,
        UPSTREAM_URL,
    )
    from third_party.danet.sparsemax import Entmax15
except Exception as exc:  # noqa: BLE001
    _write_failure_and_stop(
        "Official DANet (third_party.danet AbstractLayer / DANet) import failed; "
        "pytorch-tabular fallback was not used because the official vendored subset is required first",
        f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
    )

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

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

MAX_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 512
VIRTUAL_BATCH_SIZE = 256
PERM_N = 500
DEVICE = torch.device("cpu")
NUM_CLASSES = 2
CLIP_VALUE = 2.0
WEIGHT_DECAY = 1e-5
N_QUANTILES = 1000

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
    "lgbm_emp_in_no_gender_",
    "lgbm_dart_monotone_",
    "ensemble_no_gender_",
    "ensemble_fm_ft_",
)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "danet_model.pt")
PREPROCESS_PATH = os.path.join(ARTIFACTS_DIR, "danet_preprocess.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "danet_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "danet_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "danet_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "danet_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "danet_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "danet_y_test.csv")
REQS_PATH = os.path.join(ARTIFACTS_DIR, "danet_requirements.txt")
PERM_PATH = os.path.join(ARTIFACTS_DIR, "danet_permutation_importance.json")

ALLOWED_WRITE_PATHS = [
    MODEL_PATH,
    PREPROCESS_PATH,
    META_JSON_PATH,
    OOF_PDS_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    REQS_PATH,
    PERM_PATH,
    FAILURE_NOTE_PATH,
]

# CPU-shrunk short grid (not 50 Optuna trials). layer_num must be even and >= 2
# (official DANet stacks BasicBlocks via layer_num // 2).
DANET_GRID = [
    {"layer_num": 2, "k": 5, "base_outdim": 16, "drop_rate": 0.1, "lr": 1e-3},
    {"layer_num": 2, "k": 5, "base_outdim": 32, "drop_rate": 0.1, "lr": 1e-3},
    {"layer_num": 2, "k": 3, "base_outdim": 16, "drop_rate": 0.0, "lr": 3e-4},
    {"layer_num": 4, "k": 5, "base_outdim": 16, "drop_rate": 0.1, "lr": 1e-3},
    {"layer_num": 2, "k": 8, "base_outdim": 16, "drop_rate": 0.2, "lr": 3e-3},
    {"layer_num": 4, "k": 5, "base_outdim": 32, "drop_rate": 0.2, "lr": 3e-4},
]


def seed_all(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _pkg_version(dist: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(dist)
    except Exception:  # noqa: BLE001
        return None


def collect_package_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", _pkg_version("torch")),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "sklearn": _pkg_version("scikit-learn"),
        "joblib": _pkg_version("joblib"),
        "danet_library": "vendored WhatAShot/DANet (official)",
        "danet_upstream_commit": UPSTREAM_COMMIT,
        "danet_source_url": UPSTREAM_URL,
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
        os.makedirs("artifacts", exist_ok=True)
        detail = (
            "Required pipeline files missing: "
            + ", ".join(missing)
            + ". Files found: "
            + ", ".join(_list_found_files())
        )
        with open(FAILURE_NOTE_PATH, "w", encoding="utf-8") as fh:
            fh.write("# DANet failure — STOP\n\n**Could not reconstruct pipeline files**\n\n" + detail + "\n")
        raise FileNotFoundError(detail)


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
            f"feature_name_ / permutation. Found gender in {cols}"
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


def _assert_architecture() -> None:
    if AbstractLayer is None or LearnableLocality is None or Entmax15 is None:
        _write_failure_and_stop(
            "DANet architecture missing AbstractLayer / LearnableLocality / Entmax15",
            "Vendored third_party.danet did not expose ABSTLAY components.",
        )
    src = inspect.getsource(LearnableLocality)
    if "Entmax15" not in src and "sparsemax" not in src:
        _write_failure_and_stop(
            "LearnableLocality is not using Entmax15/sparsemax",
            src[:2000],
        )
    src_al = inspect.getsource(AbstractLayer)
    if "LearnableLocality" not in src_al:
        _write_failure_and_stop(
            "AbstractLayer does not use LearnableLocality (not paper-faithful ABSTLAY)",
            src_al[:2000],
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
        raise RuntimeError("gender leaked into neural num/cat lists")
    return num_cols, cat_cols


def verify_frozen_split(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    train_n = len(train_df)
    test_n = len(test_df)
    train_bads = int(train_df[TARGET].sum())
    test_bads = int(test_df[TARGET].sum())
    train_rate = round(float(train_df[TARGET].mean()), 4)
    test_rate = round(float(test_df[TARGET].mean()), 4)
    if train_n != EXPECTED_TRAIN_N or train_rate != EXPECTED_TRAIN_BAD_RATE or train_bads != EXPECTED_TRAIN_N_BADS:
        os.makedirs("artifacts", exist_ok=True)
        detail = (
            "Could not reconstruct frozen Train split. "
            f"got n={train_n} bad_rate={train_rate} bads={train_bads}; "
            f"expected n={EXPECTED_TRAIN_N} bad_rate={EXPECTED_TRAIN_BAD_RATE} bads={EXPECTED_TRAIN_N_BADS}. "
            f"Files found: {_list_found_files()}"
        )
        with open(FAILURE_NOTE_PATH, "w", encoding="utf-8") as fh:
            fh.write("# DANet failure — STOP\n\n**Split reconstruction failed**\n\n" + detail + "\n")
        raise RuntimeError(detail)
    if test_n != EXPECTED_TEST_N or test_rate != EXPECTED_TEST_BAD_RATE or test_bads != EXPECTED_TEST_N_BADS:
        os.makedirs("artifacts", exist_ok=True)
        detail = (
            "Could not reconstruct frozen Test split. "
            f"got n={test_n} bad_rate={test_rate} bads={test_bads}; "
            f"expected n={EXPECTED_TEST_N} bad_rate={EXPECTED_TEST_BAD_RATE} bads={EXPECTED_TEST_N_BADS}. "
            f"Files found: {_list_found_files()}"
        )
        with open(FAILURE_NOTE_PATH, "w", encoding="utf-8") as fh:
            fh.write("# DANet failure — STOP\n\n**Split reconstruction failed**\n\n" + detail + "\n")
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


def _rank_of(feature: str, ranking: list[dict], key: str) -> int | None:
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
        "employment_status_rank": _rank_of("employment_status", ranking, "auc_drop"),
        "gender_present": False,
        "employment_status_present": True,
    }


class DANetPreprocess:
    """Train-only category maps + QuantileTransformer (Gaussian). Never fit on test.

    Official DANet takes a continuous float matrix (no embeddings). Categorical
    columns are integer-mapped (0=unknown) then quantile-transformed with the
    numerics, preserving the exact 20-column order.
    """

    def __init__(
        self,
        feature_order: list[str],
        num_cols: list[str],
        cat_cols: list[str],
        n_quantiles: int = N_QUANTILES,
    ):
        self.feature_order = list(feature_order)
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.n_quantiles = int(n_quantiles)
        self.cat_maps_: dict[str, dict[str, int]] = {}
        self.qt_ = None

    def _encode(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        _assert_in_model_columns(self.feature_order, "DANetPreprocess feature_order")
        frame = df[self.feature_order]
        cols = []
        for col in self.feature_order:
            if col in self.cat_cols:
                as_str = frame[col].astype(str)
                if fit:
                    vals = sorted(pd.Series(as_str).unique().tolist())
                    self.cat_maps_[col] = {v: i + 1 for i, v in enumerate(vals)}
                mapping = self.cat_maps_[col]
                mapped = as_str.map(mapping).fillna(0).astype(float)
                cols.append(mapped.to_numpy())
            else:
                cols.append(frame[col].astype(float).to_numpy())
        return np.stack(cols, axis=1).astype(np.float64)

    def fit(self, df: pd.DataFrame) -> "DANetPreprocess":
        encoded = self._encode(df, fit=True)
        n_q = max(10, min(self.n_quantiles, int(encoded.shape[0])))
        self.qt_ = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            subsample=None,
            random_state=RANDOM_STATE,
        )
        self.qt_.fit(encoded)
        if "gender" in self.cat_maps_ or "gender" in self.feature_order:
            raise RuntimeError("gender leaked into DANetPreprocess")
        if "employment_status" not in self.cat_maps_:
            raise RuntimeError("employment_status missing from DANetPreprocess cat maps")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.qt_ is None:
            raise RuntimeError("DANetPreprocess.transform called before fit")
        encoded = self._encode(df, fit=False)
        out = self.qt_.transform(encoded).astype(np.float32)
        return out


def build_danet(input_dim: int, hparams: dict) -> DANet:
    seed_all(RANDOM_STATE)
    return DANet(
        input_dim=int(input_dim),
        num_classes=NUM_CLASSES,
        layer_num=int(hparams["layer_num"]),
        base_outdim=int(hparams["base_outdim"]),
        k=int(hparams["k"]),
        virtual_batch_size=int(hparams.get("virtual_batch_size", VIRTUAL_BATCH_SIZE)),
        drop_rate=float(hparams["drop_rate"]),
    )


def _predict_danet(model: nn.Module, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(x[start : start + batch]).float()
            logits = model(xb)
            pd_bad = torch.softmax(logits, dim=1)[:, 1]
            outs.append(pd_bad.cpu().numpy())
    pred = np.concatenate(outs, axis=0).astype(float)
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


def _train_danet_epochs(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray | None,
    y_va: np.ndarray | None,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    seed: int,
    n_epochs_fixed: int | None = None,
) -> tuple[nn.Module, int, float | None]:
    seed_all(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    rng = np.random.RandomState(seed)
    n = int(x_tr.shape[0])
    y_tr_i = y_tr.astype(np.int64)
    use_es = n_epochs_fixed is None and x_va is not None and y_va is not None
    epoch_limit = int(n_epochs_fixed) if n_epochs_fixed is not None else int(max_epochs)

    for epoch in range(1, epoch_limit + 1):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(x_tr[idx]).float()
            yb = torch.from_numpy(y_tr_i[idx]).long()
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            if not torch.isfinite(loss):
                _write_failure_and_stop(
                    "DANet training produced non-finite loss",
                    f"epoch={epoch} loss={loss}",
                )
            loss.backward()
            if CLIP_VALUE:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
            opt.step()

        if use_es:
            va_pd = _predict_danet(model, x_va)
            auc = float(roc_auc_score(y_va, va_pd))
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

    if use_es:
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch_limit
            best_auc = float(roc_auc_score(y_va, _predict_danet(model, x_va)))
        model.load_state_dict(best_state)
        model.eval()
        return model, int(best_epoch), float(best_auc)

    model.eval()
    return model, int(epoch_limit), None


def run_danet_oof(hparams: dict, raw_X: pd.DataFrame, y: pd.Series, folds, num_cols, cat_cols) -> dict:
    oof = np.zeros(len(raw_X), dtype=float)
    fold_aucs = []
    fold_ks = []
    fold_best_epochs = []
    fold_preprocess_notes = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = raw_X.iloc[tr_idx]
        X_va = raw_X.iloc[va_idx]
        y_tr = y.iloc[tr_idx].to_numpy()
        y_va = y.iloc[va_idx].to_numpy()
        prep = DANetPreprocess(EXPECTED_IN_MODEL_FEATURES, num_cols, cat_cols)
        prep.fit(X_tr)
        x_tr = prep.transform(X_tr)
        x_va = prep.transform(X_va)
        model = build_danet(x_tr.shape[1], hparams)
        model, best_epoch, best_auc = _train_danet_epochs(
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            lr=float(hparams["lr"]),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=RANDOM_STATE,
        )
        pred = _predict_danet(model, x_va)
        oof[va_idx] = pred
        fold_aucs.append(float(roc_auc_score(y_va, pred)))
        fold_ks.append(_ks(y_va, pred))
        fold_best_epochs.append(int(best_epoch))
        fold_preprocess_notes.append(
            {
                "fold": fold_i,
                "n_train": int(len(X_tr)),
                "n_valid": int(len(X_va)),
                "best_epoch": int(best_epoch),
                "fold_valid_auc_at_best": None if best_auc is None else round(float(best_auc), 6),
            }
        )
        print(
            f"  [danet {hparams}] fold {fold_i}/{CV_FOLDS}: "
            f"AUC={fold_aucs[-1]:.6f} KS={fold_ks[-1]:.6f} best_epoch={best_epoch}",
            flush=True,
        )
    return {
        "hparams": dict(hparams),
        "oof_preds": oof,
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ks": _ks(y.to_numpy(), oof),
        "fold_aucs": fold_aucs,
        "fold_ks": fold_ks,
        "fold_best_epochs": fold_best_epochs,
        "fold_notes": fold_preprocess_notes,
        "n_epochs_final": max(1, int(round(float(np.mean(fold_best_epochs))))),
    }


def tune_danet(raw_X: pd.DataFrame, y: pd.Series, folds, num_cols, cat_cols) -> dict:
    print(
        f"[train_danet] Tuning DANet on TRAIN OOF AUC; grid n={len(DANET_GRID)} "
        f"(not 50 Optuna trials)...",
        flush=True,
    )
    best = None
    grid_scores = []
    for i, cfg in enumerate(DANET_GRID, start=1):
        print(f"[tune_danet] config {i}/{len(DANET_GRID)} {cfg}", flush=True)
        try:
            res = run_danet_oof(cfg, raw_X, y, folds, num_cols, cat_cols)
        except Exception as exc:  # noqa: BLE001
            _write_failure_and_stop(
                f"DANet OOF training failed for config {cfg}",
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            )
        grid_scores.append(
            {
                **cfg,
                "oof_auc": round(res["oof_auc"], 6),
                "oof_ks": round(res["oof_ks"], 6),
                "fold_aucs": [round(a, 6) for a in res["fold_aucs"]],
                "fold_best_epochs": res["fold_best_epochs"],
                "n_epochs_final": res["n_epochs_final"],
            }
        )
        print(
            f"[tune_danet] config {i} OOF AUC={res['oof_auc']:.6f} OOF KS={res['oof_ks']:.6f} "
            f"n_epochs_final={res['n_epochs_final']}",
            flush=True,
        )
        if best is None or res["oof_auc"] > best["oof_auc"]:
            best = res
    return {"winner": best, "grid_scores": grid_scores}


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    seed_all(RANDOM_STATE)
    _require_pipeline_files()
    _assert_load_and_split_unchanged()
    _assert_architecture()

    print("[train_danet] load_and_split() unchanged...", flush=True)
    train_df, test_df, feature_cols = load_and_split()
    verify_frozen_split(train_df, test_df, feature_cols)

    in_model_features = [c for c in feature_cols if c not in DROPPED_FROM_MODEL]
    # Keep exact expected order, not the raw CSV leftover order after dropping gender.
    in_model_features = list(EXPECTED_IN_MODEL_FEATURES)
    _assert_in_model_columns(in_model_features, "in_model_features after dropping gender")
    leftover = [c for c in feature_cols if c not in DROPPED_FROM_MODEL]
    if set(leftover) != set(EXPECTED_IN_MODEL_FEATURES):
        raise RuntimeError(
            f"load_and_split minus gender != expected 20. leftover={leftover}"
        )
    for req in REQUIRED_IN_MODEL:
        if req not in in_model_features:
            raise RuntimeError(f"{req} missing from in-model features")

    raw_X_train = train_df[in_model_features].copy()
    raw_X_test = test_df[in_model_features].copy()
    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()
    _assert_in_model_columns(list(raw_X_train.columns), "raw_X_train")
    _assert_in_model_columns(list(raw_X_test.columns), "raw_X_test")
    if "gender" in raw_X_train.columns or "gender" in raw_X_test.columns:
        raise RuntimeError("gender present in persisted X")
    if RAW_TARGET_COL in raw_X_train.columns:
        raise RuntimeError("loan_paid_back present as a feature")

    num_cols, cat_cols = split_num_cat(in_model_features, raw_X_train)
    print(f"[train_danet] num_cols={num_cols}", flush=True)
    print(f"[train_danet] cat_cols={cat_cols}", flush=True)

    raw_X_train.to_csv(X_TRAIN_PATH, index=False)
    raw_X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(TARGET).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(TARGET).to_csv(Y_TEST_PATH, index=False)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(raw_X_train, y_train))

    try:
        tune = tune_danet(raw_X_train, y_train, folds, num_cols, cat_cols)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        _write_failure_and_stop(
            "DANet hyperparameter search failed",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )

    winner = tune["winner"]
    oof_preds = np.clip(winner["oof_preds"], 1e-6, 1.0 - 1e-6)
    oof_metrics = evaluate_discrimination_and_ks(y_train.to_numpy(), oof_preds)
    print(
        f"[train_danet] winner {winner['hparams']} OOF AUC={oof_metrics['AUC']} "
        f"KS={oof_metrics['KS_Statistic']} Gini={oof_metrics['Gini']}",
        flush=True,
    )

    oof_df = pd.DataFrame({"y": y_train.to_numpy(), "pd_danet": oof_preds})
    oof_df.to_csv(OOF_PDS_PATH, index=False)

    n_epochs_final = int(winner["n_epochs_final"])
    print(
        f"[train_danet] Refit preprocess + DANet on ALL {EXPECTED_TRAIN_N} train rows; "
        f"n_epochs_final={n_epochs_final} (mean fold best epoch, no test, no extra val)",
        flush=True,
    )
    prep_full = DANetPreprocess(in_model_features, num_cols, cat_cols)
    prep_full.fit(raw_X_train)
    x_full = prep_full.transform(raw_X_train)
    if x_full.shape[1] != 20:
        raise RuntimeError(f"preprocessed width {x_full.shape[1]} != 20")
    _assert_in_model_columns(list(prep_full.feature_order), "full-train preprocess")
    if "gender" in prep_full.cat_maps_:
        raise RuntimeError("gender in full-train cat maps")

    try:
        model_full = build_danet(x_full.shape[1], winner["hparams"])
        model_full, epochs_ran, _ = _train_danet_epochs(
            model_full,
            x_full,
            y_train.to_numpy(),
            None,
            None,
            lr=float(winner["hparams"]["lr"]),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=RANDOM_STATE,
            n_epochs_fixed=n_epochs_final,
        )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        _write_failure_and_stop(
            "DANet full-train refit failed",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )

    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[train_danet] FREEZE at {freeze_timestamp} (before any test metrics)", flush=True)

    def _predict_raw(frame: pd.DataFrame) -> np.ndarray:
        xt = prep_full.transform(frame)
        return _predict_danet(model_full, xt)

    print("[train_danet] TRAIN-only permutation importance n=500 seed=42...", flush=True)
    try:
        perm_explain = permutation_importance_auc(
            _predict_raw, raw_X_train, y_train, n=PERM_N, seed=RANDOM_STATE, context="danet"
        )
        perm_skip_reason = None
    except Exception as exc:  # noqa: BLE001
        perm_explain = {
            "method": "skipped",
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "gender_present": False,
            "employment_status_present": True,
            "employment_status_rank": None,
            "auc_drop_by_feature": [],
        }
        perm_skip_reason = perm_explain["skip_reason"]
        print(f"[train_danet] permutation skipped: {perm_skip_reason}", flush=True)

    if perm_explain.get("gender_present") is not False:
        raise RuntimeError("permutation table claims gender present")
    if perm_explain.get("employment_status_present") is not True:
        raise RuntimeError("permutation table missing employment_status")

    pkg_versions = collect_package_versions()
    constructor = {
        "input_dim": int(x_full.shape[1]),
        "num_classes": NUM_CLASSES,
        "layer_num": int(winner["hparams"]["layer_num"]),
        "base_outdim": int(winner["hparams"]["base_outdim"]),
        "k": int(winner["hparams"]["k"]),
        "virtual_batch_size": VIRTUAL_BATCH_SIZE,
        "drop_rate": float(winner["hparams"]["drop_rate"]),
    }
    neural_impl = {
        "library": "vendored WhatAShot/DANet (official)",
        "class": "third_party.danet.DANet.DANet",
        "version": UPSTREAM_COMMIT,
        "constructor": (
            "DANet(input_dim={input_dim}, num_classes={num_classes}, "
            "layer_num={layer_num}, base_outdim={base_outdim}, k={k}, "
            "virtual_batch_size={virtual_batch_size}, drop_rate={drop_rate})"
        ).format(**constructor),
        "constructor_kwargs": constructor,
        "source_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "abstlay": "AbstractLayer + LearnableLocality(Entmax15 k-masks)",
        "fallback": "not used; official vendored subset imported successfully",
        "paper": "Chen et al., DANets, AAAI 2022 (Huang et al. listing in task refers to this paper)",
        "objective": "CrossEntropyLoss over 2 classes; PD = softmax(logits)[:, 1]",
        "optimizer": "Adam (official repo uses QHAdam; Adam used here to avoid qhoptim on CPU)",
    }

    cpu_architecture_notes = {
        "shrunk_for_cpu": True,
        "notes": (
            f"Official default is layer=20, k=5, base_outdim=64, max_epochs=4000, "
            f"batch=8192. CPU grid: layer_num in {{2,4}}, k in {{3,5,8}}, "
            f"base_outdim in {{16,32}}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, "
            f"batch={BATCH_SIZE}, virtual_batch_size={VIRTUAL_BATCH_SIZE}, "
            f"grid n={len(DANET_GRID)} (not 50 Optuna trials). "
            f"n_epochs_final = mean fold best epoch."
        ),
        "winner_hyperparams": dict(winner["hparams"]),
        "n_epochs_final": n_epochs_final,
        "n_layers": int(winner["hparams"]["layer_num"]),
        "k": int(winner["hparams"]["k"]),
        "width": int(winner["hparams"]["base_outdim"]),
        "drop_rate": float(winner["hparams"]["drop_rate"]),
        "lr": float(winner["hparams"]["lr"]),
        "max_epochs_search": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "virtual_batch_size": VIRTUAL_BATCH_SIZE,
        "fold_best_epochs": winner["fold_best_epochs"],
        "epochs_ran_full_train": int(epochs_ran),
        "device": "cpu",
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "grid": DANET_GRID,
        "n_quantiles": N_QUANTILES,
        "preprocess": (
            "integer category maps (0=unknown) + sklearn QuantileTransformer "
            "output_distribution=normal on the 20-col matrix; fit on fold-train "
            "during OOF and on all 14000 train rows for the frozen model"
        ),
    }

    joblib.dump(prep_full, PREPROCESS_PATH)
    torch.save(
        {
            "state_dict": model_full.state_dict(),
            "constructor_kwargs": constructor,
            "hparams": {
                **winner["hparams"],
                "virtual_batch_size": VIRTUAL_BATCH_SIZE,
                "n_epochs_final": n_epochs_final,
                "lr": float(winner["hparams"]["lr"]),
            },
            "class": neural_impl["class"],
            "library": neural_impl["library"],
            "version": UPSTREAM_COMMIT,
            "source_url": UPSTREAM_URL,
        },
        MODEL_PATH,
    )

    with open(PERM_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(perm_explain), fh, ensure_ascii=False, indent=2)

    req_lines = [f"{k}=={v}" for k, v in pkg_versions.items() if v]
    with open(REQS_PATH, "w", encoding="utf-8") as fh:
        fh.write("# Versions actually used by scripts/train_danet.py\n")
        fh.write("\n".join(req_lines) + "\n")

    written = [
        MODEL_PATH,
        PREPROCESS_PATH,
        META_JSON_PATH,
        OOF_PDS_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        REQS_PATH,
        PERM_PATH,
    ]
    _assert_allowed_writes(written)
    for p in written:
        base = os.path.basename(p)
        if base == "lgbm_model.joblib" or base.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"accidentally wrote forbidden artifact {p}")

    meta_record = {
        "model_type": "danet",
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
        "cv_folds": CV_FOLDS,
        "neural_impl": neural_impl,
        "package_versions": pkg_versions,
        "cpu_architecture_notes": cpu_architecture_notes,
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
        "hyperparams": {
            "search": winner["hparams"],
            "grid_scores": tune["grid_scores"],
            "n_epochs_final": n_epochs_final,
            "fold_best_epochs": winner["fold_best_epochs"],
            "fold_aucs": winner["fold_aucs"],
            "fold_ks": winner["fold_ks"],
            "lr": float(winner["hparams"]["lr"]),
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "virtual_batch_size": VIRTUAL_BATCH_SIZE,
        },
        "oof_metrics": oof_metrics,
        "oof_auc": oof_metrics["AUC"],
        "oof_ks": oof_metrics["KS_Statistic"],
        "oof_gini": oof_metrics["Gini"],
        "train_explains": {"danet_permutation": perm_explain},
        "permutation_skip_reason": perm_skip_reason,
        "artifact_paths": {
            "model": MODEL_PATH,
            "preprocess": PREPROCESS_PATH,
            "meta": META_JSON_PATH,
            "oof_pds": OOF_PDS_PATH,
            "X_train": X_TRAIN_PATH,
            "y_train": Y_TRAIN_PATH,
            "X_test": X_TEST_PATH,
            "y_test": Y_TEST_PATH,
            "requirements": REQS_PATH,
            "permutation": PERM_PATH,
        },
        "did_not_write": [
            "artifacts/lgbm_model.joblib",
            "artifacts/lgbm_linear_tree_*",
            "artifacts/linear_tree_*",
            "artifacts/stack_lr_rf_lgbm_*",
            "artifacts/lgbm_no_gender_emp_overlay_*",
            "artifacts/lgbm_emp_in_no_gender_*",
            "artifacts/lgbm_dart_monotone_*",
            "artifacts/ensemble_no_gender_*",
            "artifacts/ensemble_fm_ft_*",
        ],
        "notes": (
            "ONE DANet model. No stack / extra bases / blend. gender dropped. "
            "employment_status is a normal in-model feature. No monotone_constraints. "
            "No interaction_constraints. Preprocess fit on fold-train during OOF and "
            "on all 14000 train rows for the frozen model. Architecture shrunk for CPU. "
            "This train script does not compute Test AUC/KS/Gini/PSI."
        ),
    }

    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(meta_record), fh, ensure_ascii=False, indent=2)

    print(f"[train_danet] FREEZE {freeze_timestamp}", flush=True)
    print("[train_danet] test_looked_at=false test_metrics=null", flush=True)
    print(f"[train_danet] artifacts: {written}", flush=True)
    print("[train_danet] STOP — eval script computes test metrics after freeze.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(FAILURE_NOTE_PATH):
            raise
        _write_failure_and_stop(
            "Uncaught DANet training failure",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
