"""
LightGBM-only candidate (eval only).

Loads frozen artifacts from scripts/train_lgbm_no_gender_emp_overlay.py.
Never fits a model, never re-tunes, never re-runs Optuna. Computes Test AUC /
KS / Gini / PSI with the same functions as the last run
(evaluate_discrimination_and_ks and calculate_psi from utils/risk_skills.py)
and writes artifacts/lgbm_no_gender_emp_overlay_run_report.md.

Does not consume gender or employment_status as model features. Overlay JSON
is rendered in the report only.

Usage (from repo root): python3 scripts/eval_lgbm_no_gender_emp_overlay.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

from train import load_and_split
from utils.config import ARTIFACTS_DIR, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_model.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_encoders.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_meta.json")
OVERLAY_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_overlay.json")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_y_test.csv")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_run_report.md")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_no_gender_emp_overlay_test_metrics.json")

LAST_RUN_CV_AUC = 0.8862
LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
STACK_OOF_AUC = 0.8891
STACK_TEST_AUC = 0.8885
STACK_TEST_KS = 0.5795
STACK_TEST_GINI = 0.7769
STACK_TEST_PSI = 0.0028

TRAIN_SCRIPT = "scripts/train_lgbm_no_gender_emp_overlay.py"
EVAL_SCRIPT = "scripts/eval_lgbm_no_gender_emp_overlay.py"
DROPPED_FROM_MODEL = ("gender", "employment_status")


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
    forbidden_present = [c for c in DROPPED_FROM_MODEL if c in cols]
    if forbidden_present:
        raise RuntimeError(
            f"{context}: gender and employment_status must be ABSENT from in-model X. "
            f"Found: {forbidden_present} in {cols}"
        )
    if "loan_paid_back" in cols:
        raise RuntimeError(f"{context}: loan_paid_back must never be a feature. columns={cols}")
    if TARGET in cols:
        raise RuntimeError(f"{context}: target {TARGET} present in feature columns: {cols}")


def yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def main() -> None:
    missing = [
        p
        for p in [
            MODEL_PATH,
            ENCODERS_PATH,
            META_PATH,
            OVERLAY_PATH,
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
            f"Missing {missing}. Run `python3 scripts/train_lgbm_no_gender_emp_overlay.py` first. "
            "This eval script never fits."
        )

    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(OVERLAY_PATH, encoding="utf-8") as fh:
        overlay = json.load(fh)

    freeze = meta["freeze"]
    if freeze.get("test_looked_at") is True or freeze.get("test_metrics") is not None:
        raise RuntimeError(
            "Freeze record already contains test metrics. Train script must freeze "
            "BEFORE the eval script looks at test."
        )
    if freeze.get("test_labels_used_to_fit_or_select") is not False:
        raise RuntimeError("Freeze record does not confirm test labels were unused for fit/select.")

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET]
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET]

    expected = list(meta["in_model_features"])
    if list(X_train.columns) != expected or list(X_test.columns) != expected:
        raise RuntimeError(
            f"Encoded matrix column order mismatch. train={list(X_train.columns)} "
            f"test={list(X_test.columns)} expected={expected}"
        )
    _assert_in_model_columns(list(X_train.columns), "eval X_train")
    _assert_in_model_columns(list(X_test.columns), "eval X_test")
    _assert_in_model_columns(list(encoders.keys()), "eval encoders")
    if any(name in encoders for name in DROPPED_FROM_MODEL):
        raise RuntimeError("Eval encoders include gender or employment_status; refusing to score")

    model_names = list(getattr(model, "feature_name_", expected))
    _assert_in_model_columns(model_names, "eval model.feature_name_")

    train_df, test_df, feature_cols = load_and_split()
    if len(train_df) != meta["split"]["train_n"] or len(test_df) != meta["split"]["test_n"]:
        raise RuntimeError("load_and_split sizes do not match freeze record")
    if round(float(train_df[TARGET].mean()), 4) != meta["split"]["train_bad_rate"]:
        raise RuntimeError("Train bad_rate does not match freeze record")
    if round(float(test_df[TARGET].mean()), 4) != meta["split"]["test_bad_rate"]:
        raise RuntimeError("Test bad_rate does not match freeze record")
    if "loan_paid_back" in feature_cols or "loan_paid_back" in train_df.columns:
        raise RuntimeError("loan_paid_back leaked into load_and_split output")

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_discrimination_and_ks(y_train.values, train_prob)
    test_metrics = evaluate_discrimination_and_ks(y_test.values, test_prob)
    score_psi = calculate_psi(train_prob, test_prob, num_bins=10)

    beats_last_test_auc = bool(test_metrics["AUC"] > LAST_RUN_TEST_AUC)
    beats_stack_test_auc = bool(test_metrics["AUC"] > STACK_TEST_AUC)
    beats_last_cv = bool(freeze["beats_last_cv"])
    beats_stack_oof = bool(freeze["beats_stack_oof"])

    shap_rows = meta["train_shap"]["mean_abs_shap_by_feature"]
    shap_features = [r["feature"] for r in shap_rows]
    _assert_in_model_columns(shap_features, "eval SHAP table")
    if overlay.get("used_as_model_input") is True:
        raise RuntimeError("Overlay JSON claims employment_status was used as a model input")

    test_payload = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze["timestamp_utc"],
        "train_refit_metrics": train_metrics,
        "test_metrics": test_metrics,
        "score_psi": score_psi,
        "beats_last_run_test_auc_0_8858": beats_last_test_auc,
        "beats_stack_test_auc_0_8885": beats_stack_test_auc,
        "used_to_select_model": False,
        "in_model_features": expected,
        "gender_in_model": False,
        "employment_status_in_model": False,
        "note": (
            "Test metrics are reported after freeze. They were not used to choose "
            "hyperparameters, n_estimators_final, or whether to keep the candidate. "
            "Eval never fits. employment_status is overlay-only and was not scored."
        ),
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_payload, fh, ensure_ascii=False, indent=2)

    params_table = rows_to_df(
        [{"hyperparameter": k, "value": v} for k, v in meta["best_hyperparameters"].items()]
        + [{"hyperparameter": "n_estimators_final", "value": meta["n_estimators_final"]}]
    )
    shap_table = rows_to_df(shap_rows)
    woe_table = rows_to_df(overlay["woe_table"])
    share = overlay["share_of_train_defaults"]
    share_table = rows_to_df(
        [
            {
                "employment_status": status,
                "n": share[status]["n"],
                "n_default": share[status]["n_default"],
                "share_of_all_train_defaults": share[status]["share_of_all_train_defaults"],
                "share_of_all_train_defaults_pct": share[status]["share_of_all_train_defaults_pct"],
            }
            for status in ("Unemployed", "Retired")
        ]
    )
    feature_table = rows_to_df(
        [{"order": i + 1, "feature": f} for i, f in enumerate(expected)]
    )
    perf_table = rows_to_df(
        [
            {
                "model": "LightGBM (this candidate; no gender / no employment_status)",
                "sample": "Train (refit, not OOF)",
                "AUC": train_metrics["AUC"],
                "Gini": train_metrics["Gini"],
                "KS": train_metrics["KS_Statistic"],
                "PSI": "n/a",
            },
            {
                "model": "LightGBM (this candidate; no gender / no employment_status)",
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
                "Gini": STACK_TEST_GINI,
                "KS": STACK_TEST_KS,
                "PSI": STACK_TEST_PSI,
            },
        ]
    )

    beat_cv_plain = yes_no(beats_last_cv)
    beat_stack_oof_plain = yes_no(beats_stack_oof)
    beat_test_plain = yes_no(beats_last_test_auc)
    beat_stack_test_plain = yes_no(beats_stack_test_auc)
    if beats_last_test_auc:
        test_auc_plain = (
            f"Test AUC {test_metrics['AUC']} DID beat last-run Test AUC {LAST_RUN_TEST_AUC}."
        )
    else:
        test_auc_plain = (
            f"Test AUC {test_metrics['AUC']} did NOT beat last-run Test AUC {LAST_RUN_TEST_AUC}."
        )

    in_model_csv = ", ".join(expected)
    unemployed = share["Unemployed"]
    retired = share["Retired"]

    report = f"""# LightGBM (no gender; employment_status overlay-only) — run report

One model: `lightgbm.LGBMClassifier` (`objective=binary`, `boosting_type=gbdt`, `random_state=42`).
Not a stack. LogisticRegression and RandomForest were not trained as competing models.
Frozen split reused from `scripts/train.py::load_and_split` (unchanged). Last-run
`artifacts/lgbm_model.joblib`, PR #1 linear-tree artifacts, and PR #2 stack artifacts
were not written.

## 1. Script paths

- Train script: `{TRAIN_SCRIPT}` (tune + freeze on TRAIN only)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; loads artifacts; scores Test; writes this report)
- Eval confirmation: no `.fit(`, no Optuna study, no `LGBMClassifier(` constructor in this file.

## 2. Exact in-model feature list

`gender` and `employment_status` are **ABSENT** from the in-model feature list, the WOE
encoder keys, the encoded matrices, the LightGBM `feature_name_`, and the SHAP table.
`loan_paid_back` is not a feature (`default = 1 - loan_paid_back` is the target only).
No IV drop. No VIF drop. `credit_score` and all other original columns except the two
policy drops remain.

In-model features ({len(expected)}): `{in_model_csv}`

{md_table(feature_table)}

- `gender` in model? **NO**
- `employment_status` in model? **NO** (policy overlay only; see §6)
- `loan_paid_back` in model? **NO**

## 3. Train OOF AUC vs last-run CV 0.8862 and vs stack OOF 0.8891

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **{meta['train_oof_auc']}** (raw {meta['train_oof_auc_raw']})
- Train OOF KS (monitoring only, not used to select): {meta['train_oof_ks_monitoring']}
- Last-run CV AUC (compare only, not retrained): {LAST_RUN_CV_AUC}
- Beat last-run CV 0.8862? **{beat_cv_plain}**
- PR #2 stack Train OOF AUC (compare only, not retrained): {STACK_OOF_AUC}
- Beat stack OOF 0.8891? **{beat_stack_oof_plain}**
- A drop vs the stack OOF is expected (this is a single LightGBM without gender / employment_status) and is not a reason to restore those features.
- n_trials completed: {meta['n_trials_completed']} / {meta['n_trials_requested']}
- n_estimators_final = max(50, round(mean best_iteration)) = **{meta['n_estimators_final']}**
- Fold best_iterations: {meta['winning_fold_best_iterations']}
- Fold AUCs (winning params re-run): {meta['winning_fold_aucs']}
- Fold KS (monitoring): {meta['winning_fold_ks']}

Best hyperparameters (search params; fixed `objective=binary`, `boosting_type=gbdt`, `random_state=42`, monotone_constraints all −1 on WOE columns):

{md_table(params_table)}

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 4. After freeze: Test metrics vs last-run champion and vs PR #2 stack

{test_auc_plain}

| metric | this candidate (Test) | last-run champion (Test, compare only) | beat last-run 0.8858? | PR #2 stack (Test, compare only) | beat stack 0.8885? |
| --- | --- | --- | --- | --- | --- |
| AUC | **{test_metrics['AUC']}** | {LAST_RUN_TEST_AUC} | **{beat_test_plain}** | {STACK_TEST_AUC} | {beat_stack_test_plain} |
| KS | **{test_metrics['KS_Statistic']}** | {LAST_RUN_TEST_KS} | n/a (selection was OOF AUC) | {STACK_TEST_KS} | n/a |
| Gini | **{test_metrics['Gini']}** | {LAST_RUN_TEST_GINI} | n/a | {STACK_TEST_GINI} | n/a |
| PSI (train PD vs test PD) | **{score_psi['PSI']}** | {LAST_RUN_TEST_PSI} | n/a | {STACK_TEST_PSI} | n/a |

- PSI status: {score_psi['Status']}
- This candidate Train refit (not OOF) AUC/KS: {train_metrics['AUC']} / {train_metrics['KS_Statistic']}
- Last-run and stack numbers are documented comparators; they were not retrained in this run.
- A drop vs PR #2 stack Test AUC 0.8885 is expected and is **not** a reason to put `gender` or `employment_status` back into the model.

{md_table(perf_table)}

## 5. Train mean |SHAP| by in-model feature

Computed **after freeze**, on **TRAIN**, in-model features only. Method: {meta['train_shap']['method']}
n_rows={meta['train_shap']['n_rows']}.

{md_table(shap_table)}

- `gender` present in SHAP table? **NO**
- `employment_status` present in SHAP table? **NO**

`utils.risk_skills.generate_shap_summary` on TRAIN (same frozen model, no refit):
`{meta['train_shap'].get('generate_shap_summary_train')}`

## 6. Overlay exhibit (TRAIN only; NOT a model input)

**Explicit statement:** `employment_status` is a human-review / policy overlay. It is **not** a
model input. It was not WOE-encoded into the LightGBM feature matrix, not passed to
`LGBMClassifier`, and not consumed on the eval model path. Do not restore it to chase AUC.

Overlay IV (`calculate_woe_iv` on TRAIN): **{overlay['iv']}** ({overlay['predictive_power']})

{md_table(woe_table)}

Share of TRAIN defaults in Unemployed vs Retired (counts and % of all train defaults; train defaults total = {overlay['train_defaults_total']}):

{md_table(share_table)}

- Unemployed: n={unemployed['n']}, n_default={unemployed['n_default']}, share of all train defaults = {unemployed['share_of_all_train_defaults_pct']}%
- Retired: n={retired['n']}, n_default={retired['n_default']}, share of all train defaults = {retired['share_of_all_train_defaults_pct']}%

{overlay['statement']}

## 7. Artifact paths + freeze-before-test confirmation

- `{MODEL_PATH}`
- `{ENCODERS_PATH}`
- `{META_PATH}`
- `{OVERLAY_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{X_TRAIN_PATH}`
- `{Y_TRAIN_PATH}`
- `{X_TEST_PATH}`
- `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`.

- Freeze timestamp (UTC), written by the train script before this eval script ran: **{freeze['timestamp_utc']}**
- `freeze.test_looked_at` at freeze time: `{freeze['test_looked_at']}`
- `freeze.test_metrics` at freeze time: `{freeze['test_metrics']}`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select={freeze['test_labels_used_to_fit_or_select']}`)
- Eval script never calls `fit` / Optuna / early stopping. It only `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split: `scripts/train.py::load_and_split` unchanged (`test_size=0.3`, `stratify=default`, `random_state=42`).
  Train n={meta['split']['train_n']} bad_rate={meta['split']['train_bad_rate']};
  Test n={meta['split']['test_n']} bad_rate={meta['split']['test_bad_rate']}.
- WOE 5-bin encoders (`fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS` from `utils.config`) fit on TRAIN only.

## 8. Plain statement on Test AUC vs 0.8858

{test_auc_plain}

Beat last-run Test AUC 0.8858? **{beat_test_plain}**
"""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_lgbm_overlay] Test AUC={test_metrics['AUC']} KS={test_metrics['KS_Statistic']} "
        f"Gini={test_metrics['Gini']} PSI={score_psi['PSI']}"
    )
    print(f"[eval_lgbm_overlay] Beat last-run Test AUC {LAST_RUN_TEST_AUC}? {beat_test_plain}")
    print(f"[eval_lgbm_overlay] Beat stack Test AUC {STACK_TEST_AUC}? {beat_stack_test_plain}")
    print(f"[eval_lgbm_overlay] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
