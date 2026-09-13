"""
OpenFE + LightGBM gbdt — eval only.

Loads frozen artifacts from scripts/train_openfe_lgbm.py.
Never fits OpenFE, never refits LightGBM, never re-runs Optuna.
Computes Test AUC / KS / Gini / PSI with evaluate_discrimination_and_ks
and calculate_psi, then writes artifacts/openfe_lgbm_test_metrics.json
and artifacts/openfe_lgbm_run_report.md.

Usage (from repo root): python3 scripts/eval_openfe_lgbm.py
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

from train import load_and_split
from train_openfe_lgbm import FeaturePrep  # noqa: F401  — required to unpickle artifacts/openfe_lgbm_prep.joblib
from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, PSI_WATCH, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_model.joblib")
BASELINE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_baseline_model.joblib")
OPENFE_OBJ_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_openfe.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_features.joblib")
FORMULAS_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_formulas.json")
PREP_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_prep.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_oof_pds.csv")
X_TRAIN_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train.joblib")
X_TEST_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test.joblib")
X_TRAIN_BASE_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train_base.joblib")
X_TEST_BASE_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test_base.joblib")
X_TRAIN_RAW_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_train_raw20.joblib")
X_TEST_RAW_JOBLIB = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_X_test_raw20.joblib")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_y_train.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_y_test.csv")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_run_report.md")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "openfe_lgbm_test_metrics.json")

TRAIN_SCRIPT = "scripts/train_openfe_lgbm.py"
EVAL_SCRIPT = "scripts/eval_openfe_lgbm.py"

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

COMPARATORS = {
    "last_run_0.8858": 0.8858,
    "pr2_0.8885": 0.8885,
    "pr4_0.8859": 0.8859,
    "pr6_stack_0.8883": 0.8883,
    "pr6_gbdt_0.8895": 0.8895,
    "pr7_stack_0.8951": 0.8951,
    "pr7_ft_0.8963": 0.8963,
    "pr8_danet_0.8655": 0.8655,
    "pr9_ag_0.8967": 0.8967,
    "pr9_avg_stack_0.8970": 0.8970,
}


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


def _assert_no_fit_hooks(meta: dict) -> None:
    freeze = meta["freeze"]
    if freeze.get("test_looked_at") is True or freeze.get("test_metrics") is not None:
        raise RuntimeError(
            "Freeze record already contains test metrics. Train script must freeze "
            "BEFORE the eval script looks at test."
        )
    if freeze.get("test_labels_used_to_fit_or_select") is not False:
        raise RuntimeError("Freeze record does not confirm test labels were unused for fit/select.")


def _assert_gender_emp(columns, context: str) -> None:
    cols = [str(c) for c in columns]
    if any(c.lower() == "gender" for c in cols):
        raise RuntimeError(f"{context}: gender must be ABSENT. columns={cols}")
    if "employment_status" not in cols:
        raise RuntimeError(f"{context}: employment_status must be PRESENT. columns={cols}")
    if "loan_paid_back" in cols or TARGET in cols:
        raise RuntimeError(f"{context}: target leaked into features")
    base = [c for c in cols if not str(c).startswith("autoFE_")]
    if base != EXPECTED_IN_MODEL_FEATURES:
        raise RuntimeError(f"{context}: base-20 mismatch. got={base}")


def beats_map(auc: float) -> dict:
    return {f"beats_{name}": bool(auc > value) for name, value in COMPARATORS.items()}


def score_model(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def metrics_block(y_true: np.ndarray, y_prob: np.ndarray, y_train_prob: np.ndarray) -> dict:
    disc = evaluate_discrimination_and_ks(y_true, y_prob)
    psi = calculate_psi(y_train_prob, y_prob)
    return {
        "AUC": disc["AUC"],
        "KS_Statistic": disc["KS_Statistic"],
        "Gini": disc["Gini"],
        "Validation_Rating": disc["Validation_Rating"],
        "Optimal_Cutoff_Probability": disc["Optimal_Cutoff_Probability"],
        "PSI": psi["PSI"],
        "PSI_Status": psi["Status"],
        "psi_above_0.10": bool(psi["PSI"] > PSI_WATCH),
    }


def main() -> None:
    required = [
        MODEL_PATH,
        BASELINE_MODEL_PATH,
        OPENFE_OBJ_PATH,
        FEATURES_PATH,
        FORMULAS_PATH,
        PREP_PATH,
        META_PATH,
        OOF_PDS_PATH,
        X_TRAIN_JOBLIB,
        X_TEST_JOBLIB,
        X_TRAIN_BASE_JOBLIB,
        X_TEST_BASE_JOBLIB,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
        TRAIN_SCRIPT,
        EVAL_SCRIPT,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python3 scripts/train_openfe_lgbm.py` first. "
            "This eval script never fits FE or the model."
        )

    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(FORMULAS_PATH, encoding="utf-8") as fh:
        formulas = json.load(fh)

    _assert_no_fit_hooks(meta)
    if meta.get("gender_in_model") is not False:
        raise RuntimeError("meta.gender_in_model is not false")
    if meta.get("employment_status_in_model") is not True:
        raise RuntimeError("meta.employment_status_in_model is not true")
    if meta.get("monotone_constraints") is not False:
        raise RuntimeError("meta.monotone_constraints is not false")
    if meta.get("interaction_constraints") is not False:
        raise RuntimeError("meta.interaction_constraints is not false")

    model = joblib.load(MODEL_PATH)
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    X_train = joblib.load(X_TRAIN_JOBLIB)
    X_test = joblib.load(X_TEST_JOBLIB)
    X_train_base = joblib.load(X_TRAIN_BASE_JOBLIB)
    X_test_base = joblib.load(X_TEST_BASE_JOBLIB)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET].astype(int)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET].astype(int)
    if os.path.exists(X_TRAIN_RAW_JOBLIB) and os.path.exists(X_TEST_RAW_JOBLIB):
        X_train_raw = joblib.load(X_TRAIN_RAW_JOBLIB)
        X_test_raw = joblib.load(X_TEST_RAW_JOBLIB)
    else:
        prep_bundle = joblib.load(PREP_PATH)
        baseline_spec = prep_bundle["baseline"]
        if isinstance(baseline_spec, FeaturePrep):
            baseline_prep = baseline_spec
        else:
            baseline_prep = FeaturePrep(
                baseline_spec["categorical_columns"],
                baseline_spec["category_levels"],
            )
        X_train_raw = baseline_prep.transform(X_train_base)
        X_test_raw = baseline_prep.transform(X_test_base)

    expected = list(meta["in_model_features"])
    if list(X_train.columns) != expected or list(X_test.columns) != expected:
        raise RuntimeError(
            f"Persisted matrix column mismatch. train={list(X_train.columns)} "
            f"test={list(X_test.columns)} expected={expected}"
        )
    _assert_gender_emp(list(X_train.columns), "eval X_train")
    _assert_gender_emp(list(X_test.columns), "eval X_test")
    _assert_gender_emp(list(X_train_base.columns), "eval X_train_base")
    model_names = list(getattr(model, "feature_name_", expected))
    _assert_gender_emp(model_names, "eval model.feature_name_")
    for formula in formulas.get("formulas", []):
        if "gender" in str(formula).lower():
            raise RuntimeError(f"OpenFE formula contains gender: {formula}")

    # Confirm we are not about to fit anything: objects already have fitted state.
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model cannot predict; refusing to fit.")
    if not hasattr(baseline_model, "predict_proba"):
        raise RuntimeError("Loaded baseline cannot predict; refusing to fit.")

    train_df, test_df, feature_cols = load_and_split()
    split = meta["split"]
    if len(train_df) != split["train_n"] or len(test_df) != split["test_n"]:
        raise RuntimeError("load_and_split sizes do not match freeze record")
    if round(float(train_df[TARGET].mean()), 4) != split["train_bad_rate"]:
        raise RuntimeError("Train bad_rate does not match freeze record")
    if round(float(test_df[TARGET].mean()), 4) != split["test_bad_rate"]:
        raise RuntimeError("Test bad_rate does not match freeze record")
    if int(train_df[TARGET].sum()) != split["train_n_bads"] or int(test_df[TARGET].sum()) != split["test_n_bads"]:
        raise RuntimeError("Bad counts do not match freeze record")
    if "gender" not in feature_cols:
        raise RuntimeError("gender missing from original load_and_split feature_cols")

    train_prob = score_model(model, X_train)
    test_prob = score_model(model, X_test)
    base_train_prob = score_model(baseline_model, X_train_raw)
    base_test_prob = score_model(baseline_model, X_test_raw)

    openfe_test = metrics_block(y_test.to_numpy(), test_prob, train_prob)
    baseline_test = metrics_block(y_test.to_numpy(), base_test_prob, base_train_prob)
    openfe_train = evaluate_discrimination_and_ks(y_train.to_numpy(), train_prob)
    baseline_train = evaluate_discrimination_and_ks(y_train.to_numpy(), base_train_prob)

    oof = meta["oof_metrics"]
    freeze_ts = meta["freeze"]["timestamp_utc"]
    openfe_meta = meta["openfe"]
    lgbm_meta = meta["lgbm_openfe"]
    versions = meta.get("package_versions", {})
    device = meta.get("device", {})

    beats_openfe = beats_map(float(openfe_test["AUC"]))
    beats_baseline = beats_map(float(baseline_test["AUC"]))

    payload = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze_ts,
        "eval_never_fits": True,
        "gender_in_model": False,
        "employment_status_in_model": True,
        "monotone_constraints": False,
        "interaction_constraints": False,
        "in_model_features": expected,
        "base_20_features": EXPECTED_IN_MODEL_FEATURES,
        "openfe_feature_names": list(meta.get("openfe_feature_names", [])),
        "n_in_model_features": int(len(expected)),
        "openfe": {
            "version": openfe_meta.get("version"),
            "n_jobs": openfe_meta.get("n_jobs"),
            "n_generated": openfe_meta.get("n_generated"),
            "n_selected": openfe_meta.get("n_selected"),
            "n_stage2_ranked": openfe_meta.get("n_stage2_ranked"),
            "base_20_kept": True,
            "selection_method": openfe_meta.get("selection_method"),
            "formulas": openfe_meta.get("formulas"),
        },
        "lgbm_hyperparameters": lgbm_meta.get("best_hyperparameters"),
        "n_estimators_final": lgbm_meta.get("n_estimators_final"),
        "optuna_note": lgbm_meta.get("search_note"),
        "oof_metrics": oof,
        "train_refit_metrics": {
            "openfe_lgbm": openfe_train,
            "baseline_raw20": baseline_train,
        },
        "test_metrics": {
            "openfe_lgbm": openfe_test,
            "baseline_raw20": baseline_test,
        },
        "baseline_delta_test_auc": round(float(openfe_test["AUC"]) - float(baseline_test["AUC"]), 4),
        "beats": {
            "openfe_lgbm": beats_openfe,
            "baseline_raw20": beats_baseline,
        },
        **{k: v for k, v in beats_openfe.items()},
        "psi_above_0.10": {
            "openfe_lgbm": openfe_test["psi_above_0.10"],
            "baseline_raw20": baseline_test["psi_above_0.10"],
        },
        "employment_status_importance_rank": {
            "lgbm_gain": meta.get("employment_status_gain_rank"),
            "train_permutation": meta.get("employment_status_permutation_rank_train"),
        },
        "gender_absent": True,
        "comparators": COMPARATORS,
        "split": split,
        "gates": {"AUC_MIN": AUC_MIN, "KS_MIN": KS_MIN, "PSI_WATCH": PSI_WATCH},
        "package_versions": versions,
        "device": device,
        "leakage_note": meta.get("leakage_note"),
        "used_to_select_model": False,
        "note": (
            "Test metrics are reported after freeze. They were not used to choose "
            "OpenFE formulas, LightGBM hyperparameters, or n_estimators_final. "
            "Eval never fits FE or the model. gender is absent from X, OpenFE "
            "formulas, and feature_name_. employment_status is a normal in-model feature."
        ),
    }

    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    gain_rows = meta.get("feature_importances_gain", [])[:20]
    perm_rows = (meta.get("permutation_importance_train") or {}).get("ranking", [])[:20]
    beats_rows = [
        {
            "comparator": name,
            "auc": auc,
            "openfe_lgbm_beats": yes_no(beats_openfe[f"beats_{name}"]),
            "baseline_beats": yes_no(beats_baseline[f"beats_{name}"]),
        }
        for name, auc in COMPARATORS.items()
    ]
    oof_tbl = md_table(
        pd.DataFrame(
            [
                {"model": "openfe_lgbm", **oof["openfe_lgbm"]},
                {"model": "baseline_raw20", **oof["baseline_raw20"]},
            ]
        )
    )
    train_tbl = md_table(
        pd.DataFrame(
            [
                {"model": "openfe_lgbm", **openfe_train},
                {"model": "baseline_raw20", **baseline_train},
            ]
        )
    )
    test_tbl = md_table(
        pd.DataFrame(
            [
                {"model": "openfe_lgbm", **openfe_test},
                {"model": "baseline_raw20", **baseline_test},
            ]
        )
    )
    formula_tbl = md_table(
        pd.DataFrame(
            {
                "autoFE": meta.get("openfe_feature_names", []),
                "formula": openfe_meta.get("formulas", []),
            }
        )
    )
    hp_tbl = md_table(
        pd.DataFrame(
            [{"param": k, "value": v} for k, v in (lgbm_meta.get("best_hyperparameters") or {}).items()]
        )
    )
    gain_tbl = md_table(pd.DataFrame(gain_rows)) if gain_rows else "_none_"
    perm_tbl = md_table(pd.DataFrame(perm_rows)) if perm_rows else "_none_"
    beats_tbl = md_table(pd.DataFrame(beats_rows))
    versions_tbl = md_table(pd.DataFrame([{"package": k, "version": v} for k, v in versions.items()]))
    perm_n = (meta.get("permutation_importance_train") or {}).get("n_sample")

    report = f"""# OpenFE + LightGBM validation report

## Executive Summary

OpenFE (Zhang et al., ICML 2023, pip `openfe=={openfe_meta.get("version")}`) was **fit on TRAIN only**, then LightGBM `boosting_type=gbdt` was tuned on the frozen OpenFE+base-20 matrix with StratifiedKFold 5 (`shuffle=True`, `random_state=42`) maximizing OOF AUC. Test labels were not used to fit or select. This eval script **never fits FE or the model**.

- Device: **{device.get("device", "cpu")}** ({device.get("n_cpus")} CPUs). {device.get("note", "")}
- gender in model / FE: **NO**
- employment_status in model: **YES** (LGBM gain rank **{meta.get("employment_status_gain_rank")}**, TRAIN permutation rank **{meta.get("employment_status_permutation_rank_train")}**)
- monotone_constraints: **false**; interaction_constraints: **false**; IV/VIF drop: **false**
- Freeze UTC: `{freeze_ts}`
- OpenFE+LGBM Test AUC **{openfe_test["AUC"]}** / KS **{openfe_test["KS_Statistic"]}** / Gini **{openfe_test["Gini"]}** / PSI **{openfe_test["PSI"]}**
- Baseline raw-20 LGBM Test AUC **{baseline_test["AUC"]}** (delta OpenFE−baseline = **{payload["baseline_delta_test_auc"]}**)
- Gates: AUC>={AUC_MIN} {yes_no(openfe_test["AUC"] >= AUC_MIN)}; KS>={KS_MIN} {yes_no(openfe_test["KS_Statistic"] >= KS_MIN)}; PSI_WATCH={PSI_WATCH} exceeded={yes_no(openfe_test["psi_above_0.10"])}

## Train / Test Split

Source: `scripts/train.py::load_and_split` **UNCHANGED** (`test_size=0.3`, `stratify=default`, `random_state=42`).

| split | n | bad_rate | n_bads |
| --- | --- | --- | --- |
| Train | {split["train_n"]} | {split["train_bad_rate"]} | {split["train_n_bads"]} |
| Test | {split["test_n"]} | {split["test_bad_rate"]} | {split["test_n_bads"]} |

Expected: Train n=14000 bad_rate≈0.2001 (2801 bads); Test n=6000 bad_rate≈0.2002 (1201 bads). Confirmed against a fresh `load_and_split` call in this eval.

## Feature Engineering & Selection

- Base features: the exact remaining **20** original columns (gender dropped; employment_status kept in native CSV order). No IV/VIF drop.
- OpenFE version: **{openfe_meta.get("version")}**; n_jobs: **{openfe_meta.get("n_jobs")}**
- Candidates generated (order=1): **{openfe_meta.get("n_generated")}**
- Stage-2 ranked: **{openfe_meta.get("n_stage2_ranked")}**; selected: **{openfe_meta.get("n_selected")}**
- Base 20 kept: **YES**
- Selection method: {openfe_meta.get("selection_method")}

Selected formulas:

{formula_tbl}

Leakage note: {meta.get("leakage_note")}

## Hyperparameter Tuning (Optuna)

{lgbm_meta.get("search_note")}

Best hyperparameters (OpenFE+LGBM):

{hp_tbl}

- n_estimators_final (OpenFE+LGBM): **{lgbm_meta.get("n_estimators_final")}**
- fold best_iterations: {lgbm_meta.get("winning_fold_best_iterations")}
- Baseline raw-20 n_estimators_final: **{meta.get("lgbm_baseline_raw20", {}).get("n_estimators_final")}**

## Model Performance & Discrimination

### OOF (train folds, pre-freeze)

{oof_tbl}

### After freeze — Train refit (not used for selection)

{train_tbl}

### After freeze — Test

{test_tbl}

## Population Stability Index (PSI)

Score PSI uses train refit PD as expected and test PD as actual (`calculate_psi`, 10 bins).

| model | PSI | status | psi_above_0.10 |
| --- | --- | --- | --- |
| openfe_lgbm | {openfe_test["PSI"]} | {openfe_test["PSI_Status"]} | {yes_no(openfe_test["psi_above_0.10"])} |
| baseline_raw20 | {baseline_test["PSI"]} | {baseline_test["PSI_Status"]} | {yes_no(baseline_test["psi_above_0.10"])} |

## Explainability & Governance

- gender absent from base X, OpenFE formulas, transformed columns, and `feature_name_`: **YES**
- employment_status in-model: **YES**
- TRAIN LGBM gain rank of employment_status: **{meta.get("employment_status_gain_rank")}**
- TRAIN permutation rank of employment_status: **{meta.get("employment_status_permutation_rank_train")}**

Top 20 TRAIN LGBM gain:

{gain_tbl}

Top 20 TRAIN permutation (roc_auc, sample={perm_n}):

{perm_tbl}

## beats_* vs comparators (Test AUC)

{beats_tbl}

## Package versions

{versions_tbl}

## Artifacts

Persisted OpenFE object / formulas / transformed matrices / LGBM models so a validator can SHA-match and rerun this eval **without refitting FE**.

- `{MODEL_PATH}`
- `{OPENFE_OBJ_PATH}`
- `{FEATURES_PATH}`
- `{FORMULAS_PATH}`
- `{X_TRAIN_JOBLIB}` / `{X_TEST_JOBLIB}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`

Did not write or overwrite prior-pack prefixes (`lgbm_linear_tree_`, `stack_lr_rf_lgbm_`, `lgbm_emp_overlay_`, `lgbm_emp_in_`, `dart_monotone_`, `ensemble_no_gender_`, `ensemble_fm_ft_`, `danet_`, `ag_realmlp_`, `autogluon_`, `realmlp_`) or `artifacts/lgbm_model.joblib`.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"[eval_openfe_lgbm] FREEZE {freeze_ts} computed_after_freeze=true")
    print(
        f"[eval_openfe_lgbm] OpenFE+LGBM Test AUC={openfe_test['AUC']} "
        f"KS={openfe_test['KS_Statistic']} Gini={openfe_test['Gini']} PSI={openfe_test['PSI']}"
    )
    print(
        f"[eval_openfe_lgbm] Baseline raw20 Test AUC={baseline_test['AUC']} "
        f"delta={payload['baseline_delta_test_auc']}"
    )
    print(f"[eval_openfe_lgbm] Wrote {TEST_METRICS_PATH}")
    print(f"[eval_openfe_lgbm] Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
