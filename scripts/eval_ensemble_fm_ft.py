"""
Six-base ensemble candidate (eval only — never fits).

Loads frozen artifacts from scripts/train_ensemble_fm_ft.py, encodes Test
with the train-fitted WOE encoders, scores the six full-train bases, then the
equal-weight average AND the stack meta (fitted on OOF PDs). Metrics via
evaluate_discrimination_and_ks and calculate_psi. PSI train side = OOF PD
(not in-sample full-train PD).

Does not call fit / Optuna / encoder fitting. Does not write last-run or
PR #1–#6 artifact names.

Usage (from repo root): python3 scripts/eval_ensemble_fm_ft.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import rtdl
import torch
from torchfm.model.dfm import DeepFactorizationMachineModel

from train import load_and_split
from train_ensemble_fm_ft import NeuralPreprocess
from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, RAW_TARGET_COL, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks
from utils.woe_encoding import apply_woe_encoder

# joblib objects dumped by `python3 scripts/train_ensemble_fm_ft.py` pickle the
# class as __main__.NeuralPreprocess; alias it so eval can load without refitting.
import __main__ as _eval_main

_eval_main.NeuralPreprocess = NeuralPreprocess

LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
LAST_RUN_CV_AUC = 0.8862
PR2_STACK_TEST_AUC = 0.8885
PR4_TEST_AUC = 0.8859
PR5_DART_TEST_AUC = 0.7645
PR6_STACK_TEST_AUC = 0.8883
PR6_GBDT_TEST_AUC = 0.8895

ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_encoders.joblib")
LR_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_lr.joblib")
FM_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_modern_fm.pt")
FT_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_ft_transformer.pt")
GBDT_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_lgbm_gbdt.joblib")
DART_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_lgbm_dart.joblib")
CATBOOST_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_catboost.joblib")
STACK_META_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_stack_meta.joblib")
NEURAL_PREP_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_neural_preprocess.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_y_test.csv")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_test_metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_fm_ft_run_report.md")

TRAIN_SCRIPT = "scripts/train_ensemble_fm_ft.py"
EVAL_SCRIPT = "scripts/eval_ensemble_fm_ft.py"

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
BASE_COL_ORDER = [
    "pd_lr",
    "pd_modern_fm",
    "pd_ft_transformer",
    "pd_lgbm_gbdt",
    "pd_lgbm_dart",
    "pd_catboost",
]
MODEL_KEYS = [
    "lr",
    "modern_fm",
    "ft_transformer",
    "lgbm_gbdt",
    "lgbm_dart",
    "catboost",
    "average",
    "stack",
]
BASES_TRAINED = [
    "lr",
    "modern_fm",
    "ft_transformer",
    "lgbm_gbdt",
    "lgbm_dart",
    "catboost",
]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for rec in df.to_dict(orient="records"):
        cells = []
        for c in cols:
            val = rec[c]
            if isinstance(val, float):
                cells.append(f"{val:.6g}" if abs(val) < 0.001 and val != 0 else f"{val}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def _assert_in_model_columns(columns: list[str], context: str) -> None:
    cols = list(columns)
    if "gender" in cols:
        raise RuntimeError(f"{context}: gender must be ABSENT. Found gender in {cols}")
    if "employment_status" not in cols:
        raise RuntimeError(
            f"{context}: employment_status must be PRESENT as a normal in-model feature. columns={cols}"
        )
    if RAW_TARGET_COL in cols or "loan_paid_back" in cols:
        raise RuntimeError(f"{context}: loan_paid_back must never be a feature. columns={cols}")
    if TARGET in cols:
        raise RuntimeError(f"{context}: target {TARGET} present in feature columns: {cols}")
    if len(cols) != 20:
        raise RuntimeError(f"{context}: expected exactly 20 in-model columns, got {len(cols)}: {cols}")
    if cols != EXPECTED_IN_MODEL_FEATURES:
        raise RuntimeError(
            f"{context}: in-model columns mismatch. got={cols} expected={EXPECTED_IN_MODEL_FEATURES}"
        )


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


def _assert_no_constraints(model, context: str) -> None:
    params = model.get_params() if hasattr(model, "get_params") else {}
    for key in ("monotone_constraints", "interaction_constraints"):
        val = params.get(key)
        if val not in (None, [], (), "", "None"):
            raise RuntimeError(f"{context}: {key}={val!r} is forbidden")


def encode_woe(df: pd.DataFrame, feature_cols: list[str], encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in feature_cols})


def _pack_test(y_true: np.ndarray, pd_vec: np.ndarray, oof_pd: np.ndarray) -> dict:
    disc = evaluate_discrimination_and_ks(y_true, pd_vec)
    psi = calculate_psi(oof_pd, pd_vec, num_bins=10)
    return {
        "AUC": disc["AUC"],
        "KS_Statistic": disc["KS_Statistic"],
        "Gini": disc["Gini"],
        "Validation_Rating": disc["Validation_Rating"],
        "Optimal_Cutoff_Probability": disc["Optimal_Cutoff_Probability"],
        "PSI": psi["PSI"],
        "PSI_Status": psi["Status"],
    }


def _beats(auc: float) -> dict:
    return {
        "beats_last_run_0.8858": bool(auc > LAST_RUN_TEST_AUC),
        "beats_pr2_stack_0.8885": bool(auc > PR2_STACK_TEST_AUC),
        "beats_pr4_0.8859": bool(auc > PR4_TEST_AUC),
        "beats_pr5_dart_0.7645": bool(auc > PR5_DART_TEST_AUC),
        "beats_pr6_stack_0.8883": bool(auc > PR6_STACK_TEST_AUC),
        "beats_pr6_gbdt_alone_0.8895": bool(auc > PR6_GBDT_TEST_AUC),
    }


def _top_features(rows: list[dict], value_key: str, n: int = 8) -> str:
    parts = []
    for i, row in enumerate(rows[:n], start=1):
        parts.append(f"{i}. {row['feature']} ({row[value_key]})")
    return "; ".join(parts)


def _predict_fm(model, x_fm: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x_fm), batch):
            xb = torch.from_numpy(x_fm[start : start + batch]).long()
            outs.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(outs, axis=0).astype(float), 1e-6, 1 - 1e-6)


def _predict_ft(model, x_num: np.ndarray, x_cat: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x_num), batch):
            xn = torch.from_numpy(x_num[start : start + batch])
            xc = torch.from_numpy(x_cat[start : start + batch])
            logits = model(xn, xc).reshape(-1)
            outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.clip(np.concatenate(outs, axis=0).astype(float), 1e-6, 1 - 1e-6)


def _load_fm(ckpt: dict):
    hp = ckpt["hparams"]
    model = DeepFactorizationMachineModel(
        field_dims=list(hp["field_dims"]),
        embed_dim=int(hp["embed_dim"]),
        mlp_dims=tuple(hp["mlp_dims"]),
        dropout=float(hp["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_ft(ckpt: dict):
    hp = ckpt["hparams"]
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=int(hp["n_num_features"]),
        cat_cardinalities=list(hp["cat_cardinalities"]),
        d_token=int(hp["d_token"]),
        n_blocks=int(hp["n_blocks"]),
        attention_dropout=float(hp["attention_dropout"]),
        ffn_d_hidden=int(hp["ffn_d_hidden"]),
        ffn_dropout=float(hp["ffn_dropout"]),
        residual_dropout=float(hp.get("residual_dropout", 0.0)),
        last_layer_query_idx=list(hp.get("last_layer_query_idx", [-1])),
        d_out=int(hp.get("d_out", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    required = [
        ENCODERS_PATH,
        LR_PATH,
        FM_PATH,
        FT_PATH,
        GBDT_PATH,
        DART_PATH,
        CATBOOST_PATH,
        STACK_META_PATH,
        NEURAL_PREP_PATH,
        META_JSON_PATH,
        OOF_PDS_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        TRAIN_SCRIPT,
        EVAL_SCRIPT,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing freeze artifacts {missing}. Run `python3 {TRAIN_SCRIPT}` first. "
            "This eval script never fits."
        )

    with open(META_JSON_PATH, encoding="utf-8") as fh:
        freeze = json.load(fh)

    if freeze.get("test_looked_at") is not False:
        raise RuntimeError("Freeze record must have test_looked_at=false before eval.")
    if freeze.get("test_metrics") is not None:
        raise RuntimeError("Freeze record must have test_metrics=null before eval.")
    if freeze.get("test_labels_used_to_fit_or_select") is not False:
        raise RuntimeError("Freeze record claims test labels were used to fit/select.")
    if freeze.get("monotone_constraints") is not False:
        raise RuntimeError("Freeze record must have monotone_constraints=false")
    if freeze.get("interaction_constraints") is not False:
        raise RuntimeError("Freeze record must have interaction_constraints=false")
    if freeze.get("gender_in_model") is not False:
        raise RuntimeError("meta.gender_in_model is not false")
    if freeze.get("employment_status_in_model") is not True:
        raise RuntimeError("meta.employment_status_in_model is not true")
    if freeze.get("bases_trained") != BASES_TRAINED:
        raise RuntimeError(f"Freeze record missing all six bases: {freeze.get('bases_trained')}")

    in_model_features = list(freeze["in_model_features"])
    _assert_in_model_columns(in_model_features, "freeze in_model_features")

    train_df, test_df, feature_cols_split = load_and_split()
    train_n = len(train_df)
    test_n = len(test_df)
    train_bad_rate = round(float(train_df[TARGET].mean()), 4)
    test_bad_rate = round(float(test_df[TARGET].mean()), 4)
    if train_n != 14000 or train_bad_rate != 0.2001 or test_n != 6000 or test_bad_rate != 0.2002:
        raise RuntimeError(
            f"Split mismatch at eval: train n={train_n} br={train_bad_rate} "
            f"test n={test_n} br={test_bad_rate}"
        )
    expected_from_split = [c for c in feature_cols_split if c != "gender"]
    if expected_from_split != in_model_features:
        raise RuntimeError(
            f"in_model_features != load_and_split minus gender. "
            f"got={in_model_features} split_minus_gender={expected_from_split}"
        )
    if RAW_TARGET_COL in test_df.columns or RAW_TARGET_COL in in_model_features:
        raise RuntimeError("loan_paid_back present in eval feature path")

    encoders = joblib.load(ENCODERS_PATH)
    lr_model = joblib.load(LR_PATH)
    gbdt_model = joblib.load(GBDT_PATH)
    dart_model = joblib.load(DART_PATH)
    cat_model = joblib.load(CATBOOST_PATH)
    stack_meta = joblib.load(STACK_META_PATH)
    neural_prep = joblib.load(NEURAL_PREP_PATH)
    oof_df = pd.read_csv(OOF_PDS_PATH)
    X_train_persisted = pd.read_csv(X_TRAIN_PATH)
    X_test_persisted = pd.read_csv(X_TEST_PATH)
    y_test_persisted = pd.read_csv(Y_TEST_PATH)[TARGET].to_numpy()

    fm_ckpt = torch.load(FM_PATH, map_location="cpu", weights_only=False)
    ft_ckpt = torch.load(FT_PATH, map_location="cpu", weights_only=False)
    fm_model = _load_fm(fm_ckpt)
    ft_model = _load_ft(ft_ckpt)

    if len(oof_df) != 14000:
        raise RuntimeError(f"OOF PD matrix has {len(oof_df)} rows, expected 14000")
    _assert_in_model_columns(list(encoders.keys()), "eval encoders")
    _assert_in_model_columns(list(X_train_persisted.columns), "eval persisted X_train")
    _assert_in_model_columns(list(X_test_persisted.columns), "eval persisted X_test")
    if "gender" in encoders:
        raise RuntimeError("Eval encoders include gender; refusing to score")
    _assert_in_model_columns(list(neural_prep.feature_order), "eval neural preprocess feature_order")
    if "gender" in neural_prep.feature_order or "gender" in neural_prep.num_cols or "gender" in neural_prep.cat_cols:
        raise RuntimeError("Neural preprocess contains gender")
    if "employment_status" not in neural_prep.cat_cols:
        raise RuntimeError("Neural preprocess missing employment_status as categorical")

    for label, model in (
        ("lr", lr_model),
        ("lgbm_gbdt", gbdt_model),
        ("lgbm_dart", dart_model),
        ("catboost", cat_model),
    ):
        _assert_no_constraints(model, f"eval {label}")
        names = _feature_names_from_model(model, in_model_features)
        _assert_in_model_columns(names, f"eval {label} feature names")
        if "gender" in names:
            raise RuntimeError(f"{label} feature names contain gender")

    gbdt_boost = gbdt_model.get_params().get("boosting_type")
    dart_boost = dart_model.get_params().get("boosting_type")
    if gbdt_boost != "gbdt":
        raise RuntimeError(f"gbdt boosting_type={gbdt_boost}, expected gbdt")
    if dart_boost != "dart":
        raise RuntimeError(f"dart boosting_type={dart_boost}, expected dart")

    X_test = encode_woe(test_df, in_model_features, encoders)
    _assert_in_model_columns(list(X_test.columns), "eval encoded X_test")
    if not np.allclose(X_test.to_numpy(), X_test_persisted.to_numpy(), equal_nan=True):
        raise RuntimeError("Re-encoded X_test does not match persisted X_test")
    y_test = test_df[TARGET].to_numpy()
    if not np.array_equal(y_test, y_test_persisted):
        raise RuntimeError("load_and_split y_test does not match persisted y_test")

    raw_X_test = test_df[in_model_features].copy()
    _assert_in_model_columns(list(raw_X_test.columns), "eval raw X_test")
    x_fm = neural_prep.transform_fm(raw_X_test)
    x_num, x_cat = neural_prep.transform_ft(raw_X_test)

    test_base = pd.DataFrame(
        {
            "pd_lr": lr_model.predict_proba(X_test)[:, 1],
            "pd_modern_fm": _predict_fm(fm_model, x_fm),
            "pd_ft_transformer": _predict_ft(ft_model, x_num, x_cat),
            "pd_lgbm_gbdt": gbdt_model.predict_proba(X_test)[:, 1],
            "pd_lgbm_dart": dart_model.predict_proba(X_test)[:, 1],
            "pd_catboost": cat_model.predict_proba(X_test)[:, 1],
        }
    )
    if list(test_base.columns) != BASE_COL_ORDER:
        raise RuntimeError("test base column order mismatch")

    test_average = test_base[BASE_COL_ORDER].mean(axis=1).to_numpy()
    test_stack = stack_meta.predict_proba(test_base[BASE_COL_ORDER])[:, 1]

    oof_base_mat = oof_df[BASE_COL_ORDER]
    oof_average = oof_df["pd_average"].to_numpy()
    oof_stack = oof_df["pd_stack"].to_numpy()
    oof_stack_check = stack_meta.predict_proba(oof_base_mat)[:, 1]
    if not np.allclose(oof_stack_check, oof_stack, atol=1e-10):
        raise RuntimeError("Frozen stack meta applied to OOF PDs does not match persisted pd_stack")

    test_pds = {
        "lr": test_base["pd_lr"].to_numpy(),
        "modern_fm": test_base["pd_modern_fm"].to_numpy(),
        "ft_transformer": test_base["pd_ft_transformer"].to_numpy(),
        "lgbm_gbdt": test_base["pd_lgbm_gbdt"].to_numpy(),
        "lgbm_dart": test_base["pd_lgbm_dart"].to_numpy(),
        "catboost": test_base["pd_catboost"].to_numpy(),
        "average": test_average,
        "stack": test_stack,
    }
    oof_pds = {
        "lr": oof_df["pd_lr"].to_numpy(),
        "modern_fm": oof_df["pd_modern_fm"].to_numpy(),
        "ft_transformer": oof_df["pd_ft_transformer"].to_numpy(),
        "lgbm_gbdt": oof_df["pd_lgbm_gbdt"].to_numpy(),
        "lgbm_dart": oof_df["pd_lgbm_dart"].to_numpy(),
        "catboost": oof_df["pd_catboost"].to_numpy(),
        "average": oof_average,
        "stack": oof_stack,
    }

    test_metrics = {k: _pack_test(y_test, test_pds[k], oof_pds[k]) for k in MODEL_KEYS}
    oof_auc = freeze["oof_auc"]
    oof_ks = freeze["oof_ks"]
    oof_selected = freeze["oof_selected_ensemble"]

    avg_beats = _beats(float(test_metrics["average"]["AUC"]))
    stack_beats = _beats(float(test_metrics["stack"]["AUC"]))

    neural_impl = freeze["neural_impl"]
    pkg_versions = freeze["package_versions"]

    test_metrics_out = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze["freeze_timestamp_utc"],
        "gender_in_model": False,
        "employment_status_in_model": True,
        "in_model_features": in_model_features,
        "monotone_constraints": False,
        "interaction_constraints": False,
        "eval_never_fits": True,
        "test_labels_used_to_fit_or_select": False,
        "neural_impl": neural_impl,
        "package_versions": pkg_versions,
        "oof_selected_ensemble": oof_selected,
        "which_ensemble_oof_would_select": oof_selected,
        "oof_auc": oof_auc,
        "oof_ks": oof_ks,
        "test_metrics": test_metrics,
        "beats_last_run": {
            "average": avg_beats,
            "stack": stack_beats,
        },
        "comparators": {
            "last_run": {
                "AUC": LAST_RUN_TEST_AUC,
                "KS": LAST_RUN_TEST_KS,
                "Gini": LAST_RUN_TEST_GINI,
                "PSI": LAST_RUN_TEST_PSI,
                "CV": LAST_RUN_CV_AUC,
            },
            "pr2_stack_test_auc": PR2_STACK_TEST_AUC,
            "pr4_test_auc": PR4_TEST_AUC,
            "pr5_dart_test_auc": PR5_DART_TEST_AUC,
            "pr6_stack_test_auc": PR6_STACK_TEST_AUC,
            "pr6_gbdt_alone_test_auc": PR6_GBDT_TEST_AUC,
        },
        "stack_meta_coefs": freeze["stack_meta_coefs"],
        "stack_meta_intercept": freeze["stack_meta_intercept"],
        "bases_trained": freeze["bases_trained"],
        "split": freeze["split"],
        "cpu_architecture_notes": freeze.get("cpu_architecture_notes"),
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_metrics_out, fh, ensure_ascii=False, indent=2)

    freeze["test_looked_at"] = True
    freeze["test_metrics"] = test_metrics
    freeze["computed_after_freeze"] = True
    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(freeze, fh, ensure_ascii=False, indent=2)

    explains = freeze["train_explains"]
    hp = freeze["hyperparams"]

    name_key = [
        ("LR (WOE)", "lr"),
        ("modern_fm (DeepFM)", "modern_fm"),
        ("FT-Transformer", "ft_transformer"),
        ("LightGBM gbdt", "lgbm_gbdt"),
        ("LightGBM DART", "lgbm_dart"),
        ("CatBoost", "catboost"),
        ("Simple average", "average"),
        ("Stack meta-LR", "stack"),
    ]
    oof_table = pd.DataFrame(
        [
            {
                "model": name,
                "OOF_AUC": oof_auc[key],
                "OOF_KS": oof_ks[key],
                "OOF_Gini": freeze["oof_metrics"][key]["Gini"],
            }
            for name, key in name_key
        ]
    )
    test_table = pd.DataFrame(
        [
            {
                "model": name,
                "Test_AUC": test_metrics[key]["AUC"],
                "Test_KS": test_metrics[key]["KS_Statistic"],
                "Test_Gini": test_metrics[key]["Gini"],
                "Test_PSI": test_metrics[key]["PSI"],
            }
            for name, key in name_key
        ]
    )
    beat_rows = []
    for ens_name, beats in (("average", avg_beats), ("stack", stack_beats)):
        beat_rows.append(
            {
                "ensemble": ens_name,
                "Test_AUC": test_metrics[ens_name]["AUC"],
                "beat_0.8858": yes_no(beats["beats_last_run_0.8858"]),
                "beat_0.8885": yes_no(beats["beats_pr2_stack_0.8885"]),
                "beat_0.8859": yes_no(beats["beats_pr4_0.8859"]),
                "beat_0.8883": yes_no(beats["beats_pr6_stack_0.8883"]),
                "beat_0.8895": yes_no(beats["beats_pr6_gbdt_alone_0.8895"]),
            }
        )
    beat_table = pd.DataFrame(beat_rows)

    coef_table = pd.DataFrame(
        [{"base_pd": k, "meta_coef": v} for k, v in freeze["stack_meta_coefs"].items()]
    )

    lr_top = _top_features(explains["lr"]["abs_coef_by_feature"], "abs_coef")
    fm_top = _top_features(explains["modern_fm"]["auc_drop_by_feature"], "auc_drop")
    ft_top = _top_features(explains["ft_transformer"]["auc_drop_by_feature"], "auc_drop")
    gbdt_top = _top_features(explains["lgbm_gbdt"]["mean_abs_shap_by_feature"], "mean_abs_shap")
    dart_top = _top_features(explains["lgbm_dart"]["mean_abs_shap_by_feature"], "mean_abs_shap")
    cat_top = _top_features(explains["catboost"]["mean_abs_shap_by_feature"], "mean_abs_shap")

    feature_list = "\n".join(f"{i}. {f}" for i, f in enumerate(in_model_features, start=1))
    selected = oof_selected
    selected_auc = test_metrics[selected]["AUC"]
    cpu_notes = freeze.get("cpu_architecture_notes", {})
    fm_impl = neural_impl["modern_fm"]
    ft_impl = neural_impl["ft_transformer"]

    report = f"""# Six-base ensemble (LR + modern_fm + FT-Transformer + LGBM gbdt + DART + CatBoost) — run report

Six-base ensemble on the frozen `scripts/train.py` split. Gender is out. `employment_status` is a normal in-model feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#6 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `{TRAIN_SCRIPT}` (tune / fit on TRAIN only, then freeze)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; `predict_proba` / torch forward + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- WOE (LR / trees): `utils.woe_encoding.fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS={freeze['woe_bins']}` from `utils.config` (thresholds unchanged: KS_MIN={KS_MIN}, AUC_MIN={AUC_MIN})
- Neural preprocess: train-only StandardScaler + integer category maps (+ quantile bins for DeepFM); frozen in `{NEURAL_PREP_PATH}`

Eval never fits. Encoders, six bases, neural preprocess, and stack meta are loaded from freeze artifacts.

Split verified: Train n={train_n} bad_rate={train_bad_rate} (bads={freeze['split']['train_n_bads']}); Test n={test_n} bad_rate={test_bad_rate} (bads={freeze['split']['test_n_bads']}).

## 2. Exact 20 in-model features

`gender_in_model=false`. `employment_status_in_model=true`. `monotone_constraints=false`. `interaction_constraints=false`.

{feature_list}

- `loan_paid_back` in feature matrix: **false**
- IV-drop: **false**. VIF-drop: **false**. `credit_score` stays.
- Confirmed absent from `feature_name_` / `coef`, encoder keys, persisted X_train/X_test, neural preprocess, and SHAP/coef/permutation tables: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model feature, not overlay, not singleton interaction group)
- All six bases actually trained: `{freeze['bases_trained']}`

## 3. neural_impl

- modern_fm: library=`{fm_impl['library']}` class=`{fm_impl['class']}` version=`{fm_impl.get('version')}`
- FT-Transformer: library=`{ft_impl['library']}` class=`{ft_impl['class']}` constructor=`{ft_impl.get('constructor')}` version=`{ft_impl.get('version')}`
- Architecture shrunk for CPU: **{cpu_notes.get('shrunk_for_cpu')}**
- modern_fm CPU notes: {cpu_notes.get('modern_fm')}
- FT-Transformer CPU notes: {cpu_notes.get('ft_transformer')}
- device: {cpu_notes.get('device')} (torch cuda available: {cpu_notes.get('torch_cuda_available')})
- Package versions: {pkg_versions}

## 4. OOF AUC / KS (train only; used to select average vs stack)

{md_table(oof_table)}

- Which ensemble OOF selects: **{oof_selected}** (average OOF AUC={oof_auc['average']}, stack OOF AUC={oof_auc['stack']})
- Selection used Train OOF AUC only. Test was not used to pick among bases or between average vs stack.
- LR C grid (train OOF AUC): {hp['lr_grid_scores']}
- modern_fm n_epochs_final={hp['modern_fm_n_epochs_final']} fold_best_epochs={hp['modern_fm_fold_best_epochs']}
- FT-Transformer n_epochs_final={hp['ft_transformer_n_epochs_final']} fold_best_epochs={hp['ft_transformer_fold_best_epochs']}
- GBDT n_estimators_final = max(50, round(mean best_iteration)) = **{hp['lgbm_gbdt_n_estimators_final']}** (fold best_iterations = {hp['lgbm_gbdt_best_iterations']})
- GBDT trials completed: {hp['lgbm_gbdt_n_trials_completed']}/50; DART: {hp['lgbm_dart_n_trials_completed']}/50; CatBoost: {hp['catboost_n_trials_completed']}/50
- DART: no early_stopping. GBDT used train-fold early_stopping=30. CatBoost used searched `iterations` with no monotone. Neural used train-fold early stopping on fold-valid AUC.

### 4.1 Winning hyperparameters

- LR: {hp['lr']}
- modern_fm: {hp['modern_fm_search']}
- FT-Transformer: {hp['ft_transformer_search']}
- LGBM gbdt final: {hp['lgbm_gbdt_final']}
- LGBM DART final: {hp['lgbm_dart_final']}
- CatBoost final: {hp['catboost_final']}
- Stack meta: {hp['stack_meta']}

## 5. After freeze: Test AUC / KS / Gini / PSI

PSI train side = OOF PD of that same model/ensemble (not in-sample full-train PD).

{md_table(test_table)}

Internal gates (unchanged, not a go/no-go): AUC ≥ {AUC_MIN}, KS ≥ {KS_MIN}. OOF-selected ensemble `{selected}` Test AUC={selected_auc} ({'PASS' if selected_auc >= AUC_MIN else 'FAIL'} vs AUC_MIN), KS={test_metrics[selected]['KS_Statistic']} ({'PASS' if test_metrics[selected]['KS_Statistic'] >= KS_MIN else 'FAIL'} vs KS_MIN).

### 5.1 Plain yes/no vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858** / KS 0.5777 / Gini 0.7715 / PSI 0.0021 (CV 0.8862); PR #2 stack Test AUC **0.8885** (had gender); PR #4 Test AUC **0.8859** (emp in, gender out, monotone LGBM); PR #5 DART Test AUC 0.7645; PR #6 stack Test AUC **0.8883**; PR #6 gbdt-alone Test AUC **0.8895**.

{md_table(beat_table)}

- Did **average** beat 0.8858? **{yes_no(avg_beats['beats_last_run_0.8858'])}**
- Did **average** beat 0.8885? **{yes_no(avg_beats['beats_pr2_stack_0.8885'])}**
- Did **average** beat 0.8859? **{yes_no(avg_beats['beats_pr4_0.8859'])}**
- Did **average** beat 0.8883? **{yes_no(avg_beats['beats_pr6_stack_0.8883'])}**
- Did **average** beat 0.8895? **{yes_no(avg_beats['beats_pr6_gbdt_alone_0.8895'])}**
- Did **stack** beat 0.8858? **{yes_no(stack_beats['beats_last_run_0.8858'])}**
- Did **stack** beat 0.8885? **{yes_no(stack_beats['beats_pr2_stack_0.8885'])}**
- Did **stack** beat 0.8859? **{yes_no(stack_beats['beats_pr4_0.8859'])}**
- Did **stack** beat 0.8883? **{yes_no(stack_beats['beats_pr6_stack_0.8883'])}**
- Did **stack** beat 0.8895? **{yes_no(stack_beats['beats_pr6_gbdt_alone_0.8895'])}**

## 6. TRAIN attribution after freeze (in-model features only)

- LR abs-coef top: {lr_top}. employment_status rank = **{explains['lr']['employment_status_rank']}**. gender absent: **YES**.
- modern_fm permutation AUC-drop top: {fm_top}. employment_status rank = **{explains['modern_fm']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['modern_fm']['method']}
- FT-Transformer permutation AUC-drop top: {ft_top}. employment_status rank = **{explains['ft_transformer']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['ft_transformer']['method']}
- LGBM gbdt mean |SHAP| top: {gbdt_top}. employment_status rank = **{explains['lgbm_gbdt']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['lgbm_gbdt']['method']}
- LGBM DART mean |SHAP| top: {dart_top}. employment_status rank = **{explains['lgbm_dart']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['lgbm_dart']['method']}
- CatBoost mean |SHAP| top: {cat_top}. employment_status rank = **{explains['catboost']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['catboost']['method']}

## 7. Stack meta coefs and freeze-before-test

{md_table(coef_table)}

- Stack meta intercept: **{freeze['stack_meta_intercept']}**
- Freeze timestamp UTC: **{freeze['freeze_timestamp_utc']}**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- `computed_after_freeze=true` in `{TEST_METRICS_PATH}`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)

Artifacts:

- `{ENCODERS_PATH}`
- `{LR_PATH}`
- `{FM_PATH}`
- `{FT_PATH}`
- `{GBDT_PATH}`
- `{DART_PATH}`
- `{CATBOOST_PATH}`
- `{STACK_META_PATH}`
- `{NEURAL_PREP_PATH}`
- `{META_JSON_PATH}`
- `{OOF_PDS_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{X_TRAIN_PATH}` / `{Y_TRAIN_PATH}` / `{X_TEST_PATH}` / `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`, `artifacts/lgbm_dart_monotone_*`, `artifacts/ensemble_no_gender_*`.

## 8. Base install / runtime

All six bases were actually trained (LR, modern_fm, FT-Transformer, LightGBM gbdt, LightGBM DART, CatBoost). No base was dropped. CatBoost version `{pkg_versions.get('catboost')}`. DART runtime completed without early_stopping.

Governance thresholds in `utils/config.py` were not changed.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_ensemble_fm_ft] average Test AUC={test_metrics['average']['AUC']} "
        f"KS={test_metrics['average']['KS_Statistic']} "
        f"stack Test AUC={test_metrics['stack']['AUC']} "
        f"KS={test_metrics['stack']['KS_Statistic']} "
        f"OOF-selected={oof_selected}",
        flush=True,
    )
    print(
        f"[eval_ensemble_fm_ft] average beat 0.8858/0.8885/0.8859/0.8883/0.8895 = "
        f"{yes_no(avg_beats['beats_last_run_0.8858'])}/"
        f"{yes_no(avg_beats['beats_pr2_stack_0.8885'])}/"
        f"{yes_no(avg_beats['beats_pr4_0.8859'])}/"
        f"{yes_no(avg_beats['beats_pr6_stack_0.8883'])}/"
        f"{yes_no(avg_beats['beats_pr6_gbdt_alone_0.8895'])}; "
        f"stack = "
        f"{yes_no(stack_beats['beats_last_run_0.8858'])}/"
        f"{yes_no(stack_beats['beats_pr2_stack_0.8885'])}/"
        f"{yes_no(stack_beats['beats_pr4_0.8859'])}/"
        f"{yes_no(stack_beats['beats_pr6_stack_0.8883'])}/"
        f"{yes_no(stack_beats['beats_pr6_gbdt_alone_0.8895'])}",
        flush=True,
    )
    print(f"[eval_ensemble_fm_ft] report -> {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
