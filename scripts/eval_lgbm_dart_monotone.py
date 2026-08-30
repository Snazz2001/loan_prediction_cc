"""
LightGBM DART monotone candidate (eval only).

Loads frozen artifacts from scripts/train_lgbm_dart_monotone.py.
Never fits a model, never re-tunes, never re-runs Optuna. Applies the frozen
TRAIN-ONLY WOE encoder to Test, then computes Test AUC / KS / Gini / PSI with
evaluate_discrimination_and_ks and calculate_psi from utils/risk_skills.py.
Writes artifacts/lgbm_dart_monotone_run_report.md and
artifacts/lgbm_dart_monotone_test_metrics.json.

gender is absent. employment_status is in-model (additive-only singleton).
boosting_type=dart.

Usage (from repo root): python3 scripts/eval_lgbm_dart_monotone.py
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
from utils.config import ARTIFACTS_DIR, TARGET, WOE_BINS
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks
from utils.woe_encoding import apply_woe_encoder

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_model.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_encoders.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_y_test.csv")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_run_report.md")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_dart_monotone_test_metrics.json")

LAST_RUN_CV_AUC = 0.8862
LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
STACK_TEST_AUC = 0.8885
PR4_TEST_AUC = 0.8859

TRAIN_SCRIPT = "scripts/train_lgbm_dart_monotone.py"
EVAL_SCRIPT = "scripts/eval_lgbm_dart_monotone.py"

IN_MODEL_FEATURES = [
    "employment_status",
    "debt_to_income_ratio",
    "interest_rate",
    "grade_subgrade",
    "delinquency_history",
    "num_of_delinquencies",
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


def rows_to_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _assert_in_model_columns(columns: list[str], context: str) -> None:
    cols = list(columns)
    if cols != IN_MODEL_FEATURES:
        raise RuntimeError(
            f"{context}: expected exact in-model order {IN_MODEL_FEATURES}, got {cols}"
        )
    if "gender" in cols:
        raise RuntimeError(f"{context}: gender must be ABSENT. Found gender in {cols}")
    if "employment_status" not in cols:
        raise RuntimeError(
            f"{context}: employment_status must be PRESENT as an in-model feature. columns={cols}"
        )
    if "loan_paid_back" in cols:
        raise RuntimeError(f"{context}: loan_paid_back must never be a feature. columns={cols}")
    if TARGET in cols:
        raise RuntimeError(f"{context}: target {TARGET} present in feature columns: {cols}")


def yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def encode_woe(df: pd.DataFrame, features: list[str], encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in features})


def main() -> None:
    missing = [
        p
        for p in [
            MODEL_PATH,
            ENCODERS_PATH,
            META_PATH,
            OOF_PDS_PATH,
            X_TRAIN_PATH,
            Y_TRAIN_PATH,
            X_TEST_PATH,
            Y_TEST_PATH,
            TRAIN_SCRIPT,
            EVAL_SCRIPT,
        ]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python3 scripts/train_lgbm_dart_monotone.py` first. "
            "This eval script never fits."
        )

    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)

    freeze = meta["freeze"]
    if freeze.get("test_looked_at") is True or freeze.get("test_metrics") is not None:
        raise RuntimeError(
            "Freeze record already contains test metrics. Train script must freeze "
            "BEFORE the eval script looks at test."
        )
    if freeze.get("test_labels_used_to_fit_or_select") is not False:
        raise RuntimeError("Freeze record does not confirm test labels were unused for fit/select.")
    if meta.get("boosting_type") != "dart" and meta.get("fixed", {}).get("boosting_type") != "dart":
        raise RuntimeError("Frozen meta boosting_type is not dart")
    if meta.get("early_stopping_used") is True or meta.get("fixed", {}).get("early_stopping") is True:
        raise RuntimeError("Freeze record claims early_stopping was used; DART recipe forbids it")

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET]
    X_test_persisted = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET]

    expected = list(meta["in_model_features"])
    if expected != IN_MODEL_FEATURES:
        raise RuntimeError(
            f"Meta in_model_features mismatch. got={expected} expected={IN_MODEL_FEATURES}"
        )
    if list(X_train.columns) != expected or list(X_test_persisted.columns) != expected:
        raise RuntimeError(
            f"Encoded matrix column order mismatch. train={list(X_train.columns)} "
            f"test={list(X_test_persisted.columns)} expected={expected}"
        )
    _assert_in_model_columns(list(X_train.columns), "eval X_train")
    _assert_in_model_columns(list(X_test_persisted.columns), "eval persisted X_test")
    _assert_in_model_columns(list(encoders.keys()), "eval encoders")
    if "gender" in encoders:
        raise RuntimeError("Eval encoders include gender; refusing to score")
    if "employment_status" not in encoders:
        raise RuntimeError("Eval encoders missing employment_status; refusing to score")

    model_params = model.get_params()
    if model_params.get("boosting_type") != "dart":
        raise RuntimeError(f"Loaded model boosting_type={model_params.get('boosting_type')}, expected dart")
    model_names = list(getattr(model, "feature_name_", expected))
    _assert_in_model_columns(model_names, "eval model.feature_name_")
    if meta.get("gender_in_model") is not False:
        raise RuntimeError("meta.gender_in_model is not false")
    if meta.get("employment_status_in_model") is not True:
        raise RuntimeError("meta.employment_status_in_model is not true")

    train_df, test_df, feature_cols = load_and_split()
    if len(train_df) != meta["split"]["train_n"] or len(test_df) != meta["split"]["test_n"]:
        raise RuntimeError("load_and_split sizes do not match freeze record")
    if round(float(train_df[TARGET].mean()), 4) != meta["split"]["train_bad_rate"]:
        raise RuntimeError("Train bad_rate does not match freeze record")
    if round(float(test_df[TARGET].mean()), 4) != meta["split"]["test_bad_rate"]:
        raise RuntimeError("Test bad_rate does not match freeze record")
    if "loan_paid_back" in feature_cols or "loan_paid_back" in train_df.columns:
        raise RuntimeError("loan_paid_back leaked into load_and_split output")

    # Apply frozen TRAIN-ONLY encoder to Test here (eval never fits).
    X_test = encode_woe(test_df, expected, encoders)
    _assert_in_model_columns(list(X_test.columns), "eval encoded X_test")
    if not np.allclose(X_test.values, X_test_persisted.values, equal_nan=True):
        raise RuntimeError("Eval WOE transform of Test does not match persisted X_test")
    if WOE_BINS != 5:
        raise RuntimeError(f"WOE_BINS from utils.config is {WOE_BINS}, expected 5")

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_discrimination_and_ks(y_train.values, train_prob)
    test_metrics = evaluate_discrimination_and_ks(y_test.values, test_prob)
    score_psi = calculate_psi(train_prob, test_prob, num_bins=10)

    beats_last_test_auc = bool(test_metrics["AUC"] > LAST_RUN_TEST_AUC)
    beats_stack_test_auc = bool(test_metrics["AUC"] > STACK_TEST_AUC)
    beats_pr4_test_auc = bool(test_metrics["AUC"] > PR4_TEST_AUC)
    beats_last_cv = bool(freeze["beats_last_cv"])

    shap_rows = meta["train_shap"]["mean_abs_shap_by_feature"]
    shap_features = [r["feature"] for r in shap_rows]
    if set(shap_features) != set(IN_MODEL_FEATURES):
        raise RuntimeError(f"SHAP table feature set mismatch: {shap_features}")
    if "gender" in shap_features:
        raise RuntimeError("SHAP table contains gender")
    if "employment_status" not in shap_features:
        raise RuntimeError("SHAP table missing employment_status")
    emp_rank = meta["train_shap"].get("employment_status_rank")
    collapsed = bool(meta["train_shap"].get("collapsed_to_employment_lookup", False))

    test_payload = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze["timestamp_utc"],
        "gender_in_model": False,
        "employment_status_in_model": True,
        "in_model_features": expected,
        "beats_last_run_test_auc_0_8858": beats_last_test_auc,
        "boosting_type": "dart",
        "early_stopping_used": False,
        "beats_stack_test_auc_0_8885": beats_stack_test_auc,
        "beats_pr4_test_auc_0_8859": beats_pr4_test_auc,
        "train_refit_metrics": train_metrics,
        "test_metrics": test_metrics,
        "score_psi": score_psi,
        "used_to_select_model": False,
        "collapsed_to_employment_lookup": collapsed,
        "note": (
            "Test metrics are reported after freeze. They were not used to choose "
            "hyperparameters, n_estimators, or whether to keep the candidate. "
            "Eval never fits. employment_status is in-model (additive-only singleton "
            "interaction group). gender is absent from X, encoders, feature_name_, and SHAP. "
            "boosting_type=dart. No early_stopping."
        ),
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_payload, fh, ensure_ascii=False, indent=2)

    params_table = rows_to_df(
        [{"hyperparameter": k, "value": v} for k, v in meta["best_hyperparameters"].items()]
    )
    shap_table = rows_to_df(shap_rows)
    feature_table = rows_to_df(
        [{"order": i + 1, "feature": f} for i, f in enumerate(expected)]
    )
    perf_table = rows_to_df(
        [
            {
                "model": "LightGBM DART monotone (this candidate)",
                "sample": "Train (refit, not OOF)",
                "AUC": train_metrics["AUC"],
                "Gini": train_metrics["Gini"],
                "KS": train_metrics["KS_Statistic"],
                "PSI": "n/a",
            },
            {
                "model": "LightGBM DART monotone (this candidate)",
                "sample": "Test (after freeze)",
                "AUC": test_metrics["AUC"],
                "Gini": test_metrics["Gini"],
                "KS": test_metrics["KS_Statistic"],
                "PSI": score_psi["PSI"],
            },
            {
                "model": "Last-run champion LightGBM (compare only, not retrained)",
                "sample": "Test",
                "AUC": LAST_RUN_TEST_AUC,
                "Gini": LAST_RUN_TEST_GINI,
                "KS": LAST_RUN_TEST_KS,
                "PSI": LAST_RUN_TEST_PSI,
            },
            {
                "model": "PR #2 stack LR+RF+LGBM (compare only, not retrained)",
                "sample": "Test",
                "AUC": STACK_TEST_AUC,
                "Gini": "n/a",
                "KS": "n/a",
                "PSI": "n/a",
            },
            {
                "model": "PR #4 emp-in no-gender LightGBM (compare only, not a go champion)",
                "sample": "Test",
                "AUC": PR4_TEST_AUC,
                "Gini": "n/a",
                "KS": "n/a",
                "PSI": "n/a",
            },
        ]
    )

    beat_cv_plain = yes_no(beats_last_cv)
    beat_test_plain = yes_no(beats_last_test_auc)
    beat_stack_test_plain = yes_no(beats_stack_test_auc)
    beat_pr4_test_plain = yes_no(beats_pr4_test_auc)

    if beats_last_test_auc:
        beat_8858 = f"YES — Test AUC {test_metrics['AUC']} beat last-run Test AUC {LAST_RUN_TEST_AUC}."
    else:
        beat_8858 = f"NO — Test AUC {test_metrics['AUC']} did not beat last-run Test AUC {LAST_RUN_TEST_AUC}."
    if beats_stack_test_auc:
        beat_8885 = f"YES — Test AUC {test_metrics['AUC']} beat stack Test AUC {STACK_TEST_AUC}."
    else:
        beat_8885 = f"NO — Test AUC {test_metrics['AUC']} did not beat stack Test AUC {STACK_TEST_AUC}."
    if beats_pr4_test_auc:
        beat_8859 = f"YES — Test AUC {test_metrics['AUC']} beat PR #4 Test AUC {PR4_TEST_AUC}."
    else:
        beat_8859 = f"NO — Test AUC {test_metrics['AUC']} did not beat PR #4 Test AUC {PR4_TEST_AUC}."

    in_model_csv = ", ".join(expected)
    shap_txt = ", ".join(f"{r['feature']} ({r['mean_abs_shap']})" for r in shap_rows)
    if collapsed:
        collapse_plain = (
            "YES — the model collapsed to an employment lookup (other 5 features ~0 SHAP). "
            "This is reported, not fixed (interaction_constraints / emp / monotone left as written)."
        )
    else:
        collapse_plain = (
            "NO — remaining features retain non-trivial mean |SHAP|; not an employment-only lookup."
        )

    did_not_beat_note = ""
    if not beats_last_test_auc:
        did_not_beat_note = (
            f"\nTest AUC {test_metrics['AUC']} did not beat last-run champion Test AUC 0.8858.\n"
        )

    report = f"""# LightGBM DART monotone (6-feature, emp singleton) — run report

One model: `lightgbm.LGBMClassifier` (`objective=cross_entropy`, `boosting_type=dart`, `random_state=42`).
Not a stack. LogisticRegression and RandomForest were not trained. Not `linear_tree`.
`employment_status` is in-model, additive-only (singleton interaction group). No overlay.
No policy table as a model input. Frozen split reused from `scripts/train.py::load_and_split`
(unchanged). Last-run `artifacts/lgbm_model.joblib` and PR #1–#4 artifact names were not written.
Early stopping was **not** used (incompatible with DART). `n_estimators` was searched.

## 1. Script paths

- Train script: `{TRAIN_SCRIPT}` (tune + freeze on TRAIN only)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; loads artifacts; applies frozen WOE to Test; writes this report)
- Eval confirmation: no `.fit(`, no Optuna study, no `LGBMClassifier(` constructor in this file.

## 2. Exact 6 in-model features in order

`gender` is **ABSENT** from the in-model feature list, the WOE encoder keys, the encoded
matrices, the LightGBM `feature_name_`, and the SHAP table.
`employment_status` is **IN-MODEL**, additive-only singleton interaction group.
`loan_paid_back` is not a feature (`default = 1 - loan_paid_back` is the target only).
No IV/VIF re-run. No extra fields. WOE 5-bin TRAIN-ONLY via `fit_woe_encoder` /
`apply_woe_encoder` (`WOE_BINS` from `utils.config`).

In-model features ({len(expected)}, asserted exact order): `{in_model_csv}`

{md_table(feature_table)}

- `gender_in_model` = **false**
- `employment_status_in_model` = **true**
- `boosting_type` = **dart**
- `loan_paid_back` in model? **NO**
- monotone_constraints = `[-1, -1, -1, -1, -1, -1]` (aligned to the 6-column order; signs not searched)
- interaction_constraints = `{meta['fixed']['interaction_constraints']}` (not relaxed)

## 3. Train OOF AUC/KS vs last CV 0.8862

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **{meta['train_oof_auc']}** (raw {meta['train_oof_auc_raw']})
- Train OOF KS (monitoring only, not used to select): {meta['train_oof_ks_monitoring']}
- Last-run CV AUC (compare only, not retrained): {LAST_RUN_CV_AUC}
- Beat last-run CV 0.8862? **{beat_cv_plain}**
- n_trials completed: {meta['n_trials_completed']} / {meta['n_trials_requested']}
- n_estimators (searched; used for final refit): **{meta['n_estimators_final']}**
- Fold AUCs (winning params re-run): {meta['winning_fold_aucs']}
- Fold KS (monitoring): {meta['winning_fold_ks']}
- OOF PDs persisted: `{OOF_PDS_PATH}`
- early_stopping used? **NO**

Best hyperparameters (search params; fixed `objective=cross_entropy`, `boosting_type=dart`,
`random_state=42`, monotone all −1, emp singleton interaction):

{md_table(params_table)}

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 4. After freeze: Test metrics vs last-run / stack / PR #4

Plain yes/no on beat 0.8858 / 0.8885 / 0.8859 (no go/no-go declaration):

- Beat last-run Test AUC 0.8858? **{beat_test_plain}**. {beat_8858}
- Beat stack Test AUC 0.8885? **{beat_stack_test_plain}**. {beat_8885}
- Beat PR #4 Test AUC 0.8859? **{beat_pr4_test_plain}**. {beat_8859}
{did_not_beat_note}
| metric | this candidate (Test) | last-run champion (Test) | beat 0.8858? | PR #2 stack (Test) | beat 0.8885? | PR #4 emp-in no-gender (Test) | beat 0.8859? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUC | **{test_metrics['AUC']}** | {LAST_RUN_TEST_AUC} | **{beat_test_plain}** | {STACK_TEST_AUC} | **{beat_stack_test_plain}** | {PR4_TEST_AUC} | **{beat_pr4_test_plain}** |
| KS | **{test_metrics['KS_Statistic']}** | {LAST_RUN_TEST_KS} | n/a | n/a | n/a | n/a | n/a |
| Gini | **{test_metrics['Gini']}** | {LAST_RUN_TEST_GINI} | n/a | n/a | n/a | n/a | n/a |
| PSI (train PD vs test PD) | **{score_psi['PSI']}** | {LAST_RUN_TEST_PSI} | n/a | n/a | n/a | n/a | n/a |

- PSI status: {score_psi['Status']}
- This candidate Train refit (not OOF) AUC/KS: {train_metrics['AUC']} / {train_metrics['KS_Statistic']}
- Last-run, stack, and PR #4 numbers are documented comparators; they were not retrained in this run.
- PR #4 is not a go champion.

{md_table(perf_table)}

## 5. Train mean |SHAP| ranking

Computed **after freeze**, on **TRAIN**, in-model features only. Method: {meta['train_shap']['method']}
n_rows={meta['train_shap']['n_rows']}.

Ranking by mean |SHAP|: {shap_txt}

{md_table(shap_table)}

- `employment_status` present in SHAP table? **YES**
- `employment_status` rank (1 = highest mean |SHAP|): **{emp_rank}**
- `gender` present in SHAP table? **NO**
- Collapsed to an employment lookup (other 5 features ~0 SHAP)? **{yes_no(collapsed)}**. {collapse_plain}

`utils.risk_skills.generate_shap_summary` on TRAIN (same frozen model, no refit):
`{meta['train_shap'].get('generate_shap_summary_train')}`

## 6. Artifact paths + freeze-before-test confirmation

- `{MODEL_PATH}`
- `{ENCODERS_PATH}`
- `{META_PATH}`
- `{OOF_PDS_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{X_TRAIN_PATH}`
- `{Y_TRAIN_PATH}`
- `{X_TEST_PATH}`
- `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`.

- Freeze timestamp (UTC), written by the train script before this eval script ran: **{freeze['timestamp_utc']}**
- `freeze.test_looked_at` at freeze time: `{freeze['test_looked_at']}`
- `freeze.test_metrics` at freeze time: `{freeze['test_metrics']}`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select={freeze['test_labels_used_to_fit_or_select']}`)
- Freeze before test look? **YES**
- Eval script never calls `fit` / Optuna / early stopping. It only applies the frozen encoder, then `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split: `scripts/train.py::load_and_split` unchanged (`test_size=0.3`, `stratify=default`, `random_state=42`).
  Train n={meta['split']['train_n']} bad_rate={meta['split']['train_bad_rate']} (bads={meta['split'].get('train_n_bads', 'n/a')});
  Test n={meta['split']['test_n']} bad_rate={meta['split']['test_bad_rate']} (bads={meta['split'].get('test_n_bads', 'n/a')}).
- WOE 5-bin encoders (`fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS` from `utils.config`) fit on TRAIN only; applied to Test in this eval script.
- HEAD of this PR branch is the commit that contains these scripts/artifacts.

## 7. Outcome (plain)

- Beat last-run Test AUC 0.8858? **{beat_test_plain}**
- Beat stack Test AUC 0.8885? **{beat_stack_test_plain}**
- Beat PR #4 Test AUC 0.8859? **{beat_pr4_test_plain}**
- Employment-lookup collapse? **{yes_no(collapsed)}**
"""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_lgbm_dart] Test AUC={test_metrics['AUC']} KS={test_metrics['KS_Statistic']} "
        f"Gini={test_metrics['Gini']} PSI={score_psi['PSI']}"
    )
    print(f"[eval_lgbm_dart] Beat last-run Test AUC {LAST_RUN_TEST_AUC}? {beat_test_plain}")
    print(f"[eval_lgbm_dart] Beat stack Test AUC {STACK_TEST_AUC}? {beat_stack_test_plain}")
    print(f"[eval_lgbm_dart] Beat PR #4 Test AUC {PR4_TEST_AUC}? {beat_pr4_test_plain}")
    print(f"[eval_lgbm_dart] collapsed_to_employment_lookup={collapsed}")
    print(f"[eval_lgbm_dart] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
