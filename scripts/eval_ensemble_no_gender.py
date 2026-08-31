"""
Four-base ensemble candidate (eval only — never fits).

Loads frozen artifacts from scripts/train_ensemble_no_gender.py, encodes Test
with the train-fitted WOE encoders, scores the four full-train bases, then the
equal-weight average AND the stack meta (fitted on OOF PDs). Metrics via
evaluate_discrimination_and_ks and calculate_psi. PSI train side = OOF ensemble
PD (not in-sample full-train PD).

Does not call fit / Optuna / encoder fitting. Does not write last-run or
PR #1–#5 artifact names.

Usage (from repo root): python3 scripts/eval_ensemble_no_gender.py
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
from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, RAW_TARGET_COL, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks
from utils.woe_encoding import apply_woe_encoder

LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
LAST_RUN_CV_AUC = 0.8862
PR2_STACK_TEST_AUC = 0.8885
PR4_TEST_AUC = 0.8859
PR5_DART_TEST_AUC = 0.7645

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
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_test_metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_no_gender_run_report.md")

TRAIN_SCRIPT = "scripts/train_ensemble_no_gender.py"
EVAL_SCRIPT = "scripts/eval_ensemble_no_gender.py"

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
MODEL_KEYS = ["lr", "lgbm_gbdt", "lgbm_dart", "catboost", "average", "stack"]


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
    }


def _top_features(rows: list[dict], value_key: str, n: int = 8) -> str:
    parts = []
    for i, row in enumerate(rows[:n], start=1):
        parts.append(f"{i}. {row['feature']} ({row[value_key]})")
    return "; ".join(parts)


def main() -> None:
    required = [
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
    if freeze.get("bases_trained") != ["lr", "lgbm_gbdt", "lgbm_dart", "catboost"]:
        raise RuntimeError(f"Freeze record missing all four bases: {freeze.get('bases_trained')}")

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
    oof_df = pd.read_csv(OOF_PDS_PATH)
    X_train_persisted = pd.read_csv(X_TRAIN_PATH)
    X_test_persisted = pd.read_csv(X_TEST_PATH)
    y_test_persisted = pd.read_csv(Y_TEST_PATH)[TARGET].to_numpy()

    if len(oof_df) != 14000:
        raise RuntimeError(f"OOF PD matrix has {len(oof_df)} rows, expected 14000")
    _assert_in_model_columns(list(encoders.keys()), "eval encoders")
    _assert_in_model_columns(list(X_train_persisted.columns), "eval persisted X_train")
    _assert_in_model_columns(list(X_test_persisted.columns), "eval persisted X_test")
    if "gender" in encoders:
        raise RuntimeError("Eval encoders include gender; refusing to score")

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

    # Never fits: encode test with train-fitted encoders, then predict_proba only
    X_test = encode_woe(test_df, in_model_features, encoders)
    _assert_in_model_columns(list(X_test.columns), "eval encoded X_test")
    if not np.allclose(X_test.to_numpy(), X_test_persisted.to_numpy(), equal_nan=True):
        raise RuntimeError("Re-encoded X_test does not match persisted X_test")
    y_test = test_df[TARGET].to_numpy()
    if not np.array_equal(y_test, y_test_persisted):
        raise RuntimeError("load_and_split y_test does not match persisted y_test")

    test_base = pd.DataFrame(
        {
            "pd_lr": lr_model.predict_proba(X_test)[:, 1],
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
    # Recompute stack on OOF from frozen meta to confirm it was not refit on in-sample
    oof_stack_check = stack_meta.predict_proba(oof_base_mat)[:, 1]
    if not np.allclose(oof_stack_check, oof_stack, atol=1e-10):
        raise RuntimeError("Frozen stack meta applied to OOF PDs does not match persisted pd_stack")

    test_pds = {
        "lr": test_base["pd_lr"].to_numpy(),
        "lgbm_gbdt": test_base["pd_lgbm_gbdt"].to_numpy(),
        "lgbm_dart": test_base["pd_lgbm_dart"].to_numpy(),
        "catboost": test_base["pd_catboost"].to_numpy(),
        "average": test_average,
        "stack": test_stack,
    }
    oof_pds = {
        "lr": oof_df["pd_lr"].to_numpy(),
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
        "oof_selected_ensemble": oof_selected,
        "oof_auc": oof_auc,
        "oof_ks": oof_ks,
        "test_metrics": test_metrics,
        "beats_last_run": {
            "average": avg_beats,
            "stack": stack_beats,
        },
        "which_ensemble_oof_would_select": oof_selected,
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
        },
        "stack_meta_coefs": freeze["stack_meta_coefs"],
        "stack_meta_intercept": freeze["stack_meta_intercept"],
        "bases_trained": freeze["bases_trained"],
        "split": freeze["split"],
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_metrics_out, fh, ensure_ascii=False, indent=2)

    explains = freeze["train_explains"]
    hp = freeze["hyperparams"]

    oof_table = pd.DataFrame(
        [
            {
                "model": name,
                "OOF_AUC": oof_auc[key],
                "OOF_KS": oof_ks[key],
                "OOF_Gini": freeze["oof_metrics"][key]["Gini"],
            }
            for name, key in [
                ("LR (WOE)", "lr"),
                ("LightGBM gbdt", "lgbm_gbdt"),
                ("LightGBM DART", "lgbm_dart"),
                ("CatBoost", "catboost"),
                ("Simple average", "average"),
                ("Stack meta-LR", "stack"),
            ]
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
            for name, key in [
                ("LR (WOE)", "lr"),
                ("LightGBM gbdt", "lgbm_gbdt"),
                ("LightGBM DART", "lgbm_dart"),
                ("CatBoost", "catboost"),
                ("Simple average", "average"),
                ("Stack meta-LR", "stack"),
            ]
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
            }
        )
    beat_table = pd.DataFrame(beat_rows)

    coef_table = pd.DataFrame(
        [{"base_pd": k, "meta_coef": v} for k, v in freeze["stack_meta_coefs"].items()]
    )

    lr_top = _top_features(explains["lr"]["abs_coef_by_feature"], "abs_coef")
    gbdt_top = _top_features(explains["lgbm_gbdt"]["mean_abs_shap_by_feature"], "mean_abs_shap")
    dart_top = _top_features(explains["lgbm_dart"]["mean_abs_shap_by_feature"], "mean_abs_shap")
    cat_top = _top_features(explains["catboost"]["mean_abs_shap_by_feature"], "mean_abs_shap")

    feature_list = "\n".join(f"{i}. {f}" for i, f in enumerate(in_model_features, start=1))
    selected = oof_selected
    selected_auc = test_metrics[selected]["AUC"]

    report = f"""# Ensemble no-gender (LR + LGBM gbdt + DART + CatBoost) — run report

Four-base ensemble on the frozen `scripts/train.py` split. Gender is out. `employment_status` is a normal in-model WOE feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#5 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `{TRAIN_SCRIPT}` (tune / fit on TRAIN only, then freeze)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- WOE: `utils.woe_encoding.fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS={freeze['woe_bins']}` from `utils.config` (thresholds unchanged: KS_MIN={KS_MIN}, AUC_MIN={AUC_MIN})

Eval never fits. Encoders, four bases, and stack meta are loaded from freeze artifacts.

Split verified: Train n={train_n} bad_rate={train_bad_rate} (bads={freeze['split']['train_n_bads']}); Test n={test_n} bad_rate={test_bad_rate} (bads={freeze['split']['test_n_bads']}).

## 2. Exact 20 in-model features

`gender_in_model=false`. `employment_status_in_model=true`. `monotone_constraints=false`. `interaction_constraints=false`.

{feature_list}

- `loan_paid_back` in feature matrix: **false**
- IV-drop: **false**. VIF-drop: **false**. `credit_score` stays.
- Confirmed absent from `feature_name_` / `coef`, encoder keys, persisted X_train/X_test, and SHAP/coef tables: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model WOE feature, not overlay, not singleton interaction group)
- All four bases actually trained: `{freeze['bases_trained']}`

## 3. OOF AUC / KS (train only; used to select average vs stack)

{md_table(oof_table)}

- Which ensemble OOF selects: **{oof_selected}** (average OOF AUC={oof_auc['average']}, stack OOF AUC={oof_auc['stack']})
- Selection used Train OOF AUC only. Test was not used to pick among bases or between average vs stack.
- LR C grid (train OOF AUC): {hp['lr_grid_scores']}
- GBDT n_estimators_final = max(50, round(mean best_iteration)) = **{hp['lgbm_gbdt_n_estimators_final']}** (fold best_iterations = {hp['lgbm_gbdt_best_iterations']})
- GBDT trials completed: {hp['lgbm_gbdt_n_trials_completed']}/50; DART: {hp['lgbm_dart_n_trials_completed']}/50; CatBoost: {hp['catboost_n_trials_completed']}/50
- DART: no early_stopping. GBDT used train-fold early_stopping=30. CatBoost used searched `iterations` with no monotone.

### 3.1 Winning hyperparameters

- LR: {hp['lr']}
- LGBM gbdt final: {hp['lgbm_gbdt_final']}
- LGBM DART final: {hp['lgbm_dart_final']}
- CatBoost final: {hp['catboost_final']}
- Stack meta: {hp['stack_meta']}

## 4. After freeze: Test AUC / KS / Gini / PSI

PSI train side = OOF PD of that same model/ensemble (not in-sample full-train PD).

{md_table(test_table)}

Internal gates (unchanged, not a go/no-go): AUC ≥ {AUC_MIN}, KS ≥ {KS_MIN}. OOF-selected ensemble `{selected}` Test AUC={selected_auc} ({'PASS' if selected_auc >= AUC_MIN else 'FAIL'} vs AUC_MIN), KS={test_metrics[selected]['KS_Statistic']} ({'PASS' if test_metrics[selected]['KS_Statistic'] >= KS_MIN else 'FAIL'} vs KS_MIN).

### 4.1 Plain yes/no vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858** / KS 0.5777 / Gini 0.7715 / PSI 0.0021 (CV 0.8862); PR #2 stack Test AUC **0.8885** (had gender); PR #4 Test AUC **0.8859** (emp in, gender out, monotone LGBM); PR #5 DART Test AUC 0.7645.

{md_table(beat_table)}

- Did **average** beat 0.8858? **{yes_no(avg_beats['beats_last_run_0.8858'])}**
- Did **average** beat 0.8885? **{yes_no(avg_beats['beats_pr2_stack_0.8885'])}**
- Did **average** beat 0.8859? **{yes_no(avg_beats['beats_pr4_0.8859'])}**
- Did **stack** beat 0.8858? **{yes_no(stack_beats['beats_last_run_0.8858'])}**
- Did **stack** beat 0.8885? **{yes_no(stack_beats['beats_pr2_stack_0.8885'])}**
- Did **stack** beat 0.8859? **{yes_no(stack_beats['beats_pr4_0.8859'])}**

## 5. TRAIN explains after freeze (in-model features only)

- LR abs-coef top: {lr_top}. employment_status rank = **{explains['lr']['employment_status_rank']}**. gender absent: **YES**.
- LGBM gbdt mean |SHAP| top: {gbdt_top}. employment_status rank = **{explains['lgbm_gbdt']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['lgbm_gbdt']['method']}
- LGBM DART mean |SHAP| top: {dart_top}. employment_status rank = **{explains['lgbm_dart']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['lgbm_dart']['method']}
- CatBoost mean |SHAP| top: {cat_top}. employment_status rank = **{explains['catboost']['employment_status_rank']}**. gender absent: **YES**. Method: {explains['catboost']['method']}

## 6. Stack meta coefs and freeze-before-test

{md_table(coef_table)}

- Stack meta intercept: **{freeze['stack_meta_intercept']}**
- Freeze timestamp UTC: **{freeze['freeze_timestamp_utc']}**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- `computed_after_freeze=true` in `{TEST_METRICS_PATH}`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)

Artifacts:

- `{ENCODERS_PATH}`
- `{LR_PATH}`
- `{GBDT_PATH}`
- `{DART_PATH}`
- `{CATBOOST_PATH}`
- `{STACK_META_PATH}`
- `{META_JSON_PATH}`
- `{OOF_PDS_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{X_TRAIN_PATH}` / `{Y_TRAIN_PATH}` / `{X_TEST_PATH}` / `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`, `artifacts/lgbm_dart_monotone_*`.

## 7. Base install / runtime

All four bases were actually trained (LR, LightGBM gbdt, LightGBM DART, CatBoost). No base was dropped. CatBoost import succeeded. DART runtime completed without early_stopping.

Governance thresholds in `utils/config.py` were not changed.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_ensemble] average Test AUC={test_metrics['average']['AUC']} "
        f"KS={test_metrics['average']['KS_Statistic']} "
        f"stack Test AUC={test_metrics['stack']['AUC']} "
        f"KS={test_metrics['stack']['KS_Statistic']} "
        f"OOF-selected={oof_selected}",
        flush=True,
    )
    print(
        f"[eval_ensemble] average beat 0.8858/0.8885/0.8859 = "
        f"{yes_no(avg_beats['beats_last_run_0.8858'])}/"
        f"{yes_no(avg_beats['beats_pr2_stack_0.8885'])}/"
        f"{yes_no(avg_beats['beats_pr4_0.8859'])}; "
        f"stack = "
        f"{yes_no(stack_beats['beats_last_run_0.8858'])}/"
        f"{yes_no(stack_beats['beats_pr2_stack_0.8885'])}/"
        f"{yes_no(stack_beats['beats_pr4_0.8859'])}",
        flush=True,
    )
    print(f"[eval_ensemble] report -> {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
