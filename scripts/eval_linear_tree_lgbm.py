"""
Linear-tree LightGBM candidate (eval only).

Loads frozen artifacts from scripts/train_linear_tree_lgbm.py. Never fits a
model, never re-tunes, never re-runs Optuna. Computes Test AUC / KS / Gini /
PSI with the same functions as the last run (evaluate_discrimination_and_ks
and calculate_psi from utils/risk_skills.py) and writes
artifacts/linear_tree_run_report.md.

Usage (from repo root): python3 scripts/eval_linear_tree_lgbm.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd

from utils.config import ARTIFACTS_DIR, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_model.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_linear_tree_meta.json")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_y_test.csv")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_run_report.md")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "linear_tree_test_metrics.json")

LAST_RUN_CV_AUC = 0.8862
LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
LAST_RUN_TRAIN_AUC = 0.8908
LAST_RUN_TRAIN_KS = 0.5819
LAST_RUN_LR_TEST_AUC = 0.8812
LAST_RUN_LR_TEST_KS = 0.5717

TRAIN_SCRIPT = "scripts/train_linear_tree_lgbm.py"
EVAL_SCRIPT = "scripts/eval_linear_tree_lgbm.py"


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load_source(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def rows_to_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main() -> None:
    missing = [
        p
        for p in [MODEL_PATH, META_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, TRAIN_SCRIPT, EVAL_SCRIPT]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python3 scripts/train_linear_tree_lgbm.py` first. "
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

    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET]
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET]

    expected = meta["fixed"]["final_features"]
    if list(X_train.columns) != expected or list(X_test.columns) != expected:
        raise RuntimeError(
            f"Encoded matrix column order mismatch. train={list(X_train.columns)} "
            f"test={list(X_test.columns)} expected={expected}"
        )

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_discrimination_and_ks(y_train.values, train_prob)
    test_metrics = evaluate_discrimination_and_ks(y_test.values, test_prob)
    score_psi = calculate_psi(train_prob, test_prob, num_bins=10)

    beats_last_test_auc = bool(test_metrics["AUC"] > LAST_RUN_TEST_AUC)
    beats_last_cv = bool(freeze["beats_last_cv"])

    test_payload = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze["timestamp_utc"],
        "train_refit_metrics": train_metrics,
        "test_metrics": test_metrics,
        "score_psi": score_psi,
        "beats_last_run_test_auc_0_8858": beats_last_test_auc,
        "used_to_select_model": False,
        "note": (
            "Test metrics are reported after freeze. They were not used to choose "
            "hyperparameters, n_estimators_final, or whether to keep the candidate."
        ),
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_payload, fh, ensure_ascii=False, indent=2)

    exhibits = meta["train_governance_exhibits"]
    shap_table = rows_to_df(exhibits["mean_abs_shap_by_feature"])
    inter_table = rows_to_df(exhibits["employment_status_pairwise_shap_interactions"])
    woe_table = rows_to_df(exhibits["employment_status_woe_bin_table"])
    params_table = rows_to_df(
        [{"hyperparameter": k, "value": v} for k, v in meta["best_hyperparameters"].items()]
        + [{"hyperparameter": "n_estimators_final", "value": meta["n_estimators_final"]}]
    )
    perf_table = rows_to_df(
        [
            {
                "model": "Linear-tree LightGBM (this candidate)",
                "sample": "Train (refit, not OOF)",
                "AUC": train_metrics["AUC"],
                "Gini": train_metrics["Gini"],
                "KS": train_metrics["KS_Statistic"],
                "rating": train_metrics["Validation_Rating"],
            },
            {
                "model": "Linear-tree LightGBM (this candidate)",
                "sample": "Test (after freeze)",
                "AUC": test_metrics["AUC"],
                "Gini": test_metrics["Gini"],
                "KS": test_metrics["KS_Statistic"],
                "rating": test_metrics["Validation_Rating"],
            },
            {
                "model": "Last-run champion LightGBM (compare only, not retrained)",
                "sample": "Test",
                "AUC": LAST_RUN_TEST_AUC,
                "Gini": LAST_RUN_TEST_GINI,
                "KS": LAST_RUN_TEST_KS,
                "rating": "reported, not recomputed",
            },
        ]
    )

    unemployed = next((r for r in exhibits["employment_status_woe_bin_table"] if r["employment_status"] == "Unemployed"), None)
    retired = next((r for r in exhibits["employment_status_woe_bin_table"] if r["employment_status"] == "Retired"), None)

    beat_cv_plain = "YES" if beats_last_cv else "NO"
    beat_test_plain = "YES" if beats_last_test_auc else "NO"
    test_auc_plain = (
        f"Test AUC {test_metrics['AUC']} DID beat last-run Test AUC {LAST_RUN_TEST_AUC}."
        if beats_last_test_auc
        else f"Test AUC {test_metrics['AUC']} did NOT beat last-run Test AUC {LAST_RUN_TEST_AUC}."
    )

    train_src = load_source(TRAIN_SCRIPT)
    eval_src = load_source(EVAL_SCRIPT)

    report = f"""# Linear-tree LightGBM candidate — run report

Frozen protocol candidate. Dataset, target, split, and preprocessing were reused from
existing `scripts/train.py` / `utils/*` code (unchanged). Last-run champion
`artifacts/lgbm_model.joblib` was not written or overwritten. No extra models.
No CatBoost / EBM / AutoML. `employment_status` was kept. Interaction constraints
and monotonicity were not relaxed.

## 1. Paths + full source

- Train script path: `{TRAIN_SCRIPT}`
- Eval script path: `{EVAL_SCRIPT}`
- Model artifact: `{MODEL_PATH}`
- Freeze meta (written BEFORE test metrics): `{META_PATH}`
- Test metrics json (eval only): `{TEST_METRICS_PATH}`
- This report: `{REPORT_PATH}`

Full source of both scripts is at the bottom of this file.

## 2. Train OOF AUC vs last-run CV 0.8862

- Train OOF AUC (selection metric, 5-fold StratifiedKFold, seed=42): **{meta['train_oof_auc']}** (raw {meta['train_oof_auc_raw']})
- Train OOF KS (monitoring only, not used to select): {meta['train_oof_ks_monitoring']}
- Last-run CV AUC (compare only): {LAST_RUN_CV_AUC}
- Beat last-run CV 0.8862? **{beat_cv_plain}**
- n_trials completed: {meta['n_trials_completed']}
- n_estimators_final = max(50, round(mean best_iteration)) = **{meta['n_estimators_final']}**
- Fold best_iterations: {meta['winning_fold_best_iterations']}
- Fold AUCs (winning params re-run): {meta['winning_fold_aucs']}
- Fold KS (monitoring): {meta['winning_fold_ks']}

Best hyperparameters (search params only; fixed `linear_tree=True`, `objective=binary`,
`boosting_type=gbdt`, `random_state=42`, monotone and interaction constraints):

{md_table(params_table)}

Selection happened on Train OOF AUC only. Test was not used to choose this candidate.

## 3. After freeze: Test metrics vs last-run champion

{test_auc_plain}

| metric | this candidate (Test) | last-run champion (Test, compare only) | beat last run? |
| --- | --- | --- | --- |
| AUC | **{test_metrics['AUC']}** | {LAST_RUN_TEST_AUC} | {beat_test_plain} |
| KS | **{test_metrics['KS_Statistic']}** | {LAST_RUN_TEST_KS} | n/a (selection was OOF AUC) |
| Gini | **{test_metrics['Gini']}** | {LAST_RUN_TEST_GINI} | n/a |
| PSI (train PD vs test PD) | **{score_psi['PSI']}** | {LAST_RUN_TEST_PSI} | n/a |

- PSI status: {score_psi['Status']}
- Last-run Train AUC/KS (compare only, not recomputed): {LAST_RUN_TRAIN_AUC} / {LAST_RUN_TRAIN_KS}
- Last-run LR scorecard Test AUC/KS (compare only, not retuned): {LAST_RUN_LR_TEST_AUC} / {LAST_RUN_LR_TEST_KS}
- This candidate Train refit (not OOF) AUC/KS: {train_metrics['AUC']} / {train_metrics['KS_Statistic']}

{md_table(perf_table)}

## 4. Train SHAP / governance exhibits (computed on TRAIN after freeze, not test)

### (i) mean |SHAP| by feature on train

Computed with Exact SHAP (`shap.explainers.Exact` on `predict_proba`, Independent train background n=200, `interactions=True`) on all {len(X_train)} train rows. Native LightGBM `pred_contrib` is not implemented for `linear_tree`.

{md_table(shap_table)}

`utils.risk_skills.generate_shap_summary` on train (same model, no refit):
`{exhibits.get('generate_shap_summary_train')}`

### (ii) pairwise SHAP interaction of employment_status vs each other feature

Must be ~0 by construction because `interaction_constraints` isolate `employment_status`
from the other five features. Method: `{exhibits['interaction_method']}`

{md_table(inter_table)}

### (iii) WOE bin table for employment_status on TRAIN

Especially Unemployed vs Retired. WOE mapping came from the frozen train-only encoder
(`fit_woe_encoder` / `calculate_woe_iv`), not from test.

{md_table(woe_table)}

- Unemployed: {unemployed}
- Retired: {retired}
- IV (train, `calculate_woe_iv`): {exhibits['employment_status_iv']['iv']} ({exhibits['employment_status_iv']['predictive_power']})

`employment_status` remains flagged for governance / fair-lending review and is **not
cleared for production use**.

## 5. Artifact paths (new names only)

- `{MODEL_PATH}`
- `{META_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{os.path.join(ARTIFACTS_DIR, 'linear_tree_woe_encoders.joblib')}`
- `{os.path.join(ARTIFACTS_DIR, 'linear_tree_feature_selection.json')}`
- `{X_TRAIN_PATH}`
- `{Y_TRAIN_PATH}`
- `{X_TEST_PATH}`
- `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`.

## 6. Freeze-before-test confirmation

- Freeze timestamp (UTC), written by the train script before this eval script ran: **{freeze['timestamp_utc']}**
- `freeze.test_looked_at` at freeze time: `{freeze['test_looked_at']}`
- `freeze.test_metrics` at freeze time: `{freeze['test_metrics']}`
- Test labels used to fit or select? **NO** (`test_labels_used_to_fit_or_select={freeze['test_labels_used_to_fit_or_select']}`)
- Eval script never calls `fit` / Optuna / early stopping. It only `predict_proba` +
  `evaluate_discrimination_and_ks` + `calculate_psi`.
- Split / preprocessing: `scripts/train.py` `load_and_split` + `screen_features` +
  `iterative_vif_filter` and `utils/woe_encoding.py` fit-on-train / transform-on-test.
  Train n={meta['split']['train_n']} bad_rate={meta['split']['train_bad_rate']};
  Test n={meta['split']['test_n']} bad_rate={meta['split']['test_bad_rate']}.
- final_features order: {expected}
- monotone_constraints: {meta['fixed']['monotone_constraints']}
- interaction_constraints: {meta['fixed']['interaction_constraints']}
- linear_tree: {meta['fixed']['linear_tree']}

## 7. Plain statement on Test AUC vs 0.8858

{test_auc_plain}

Beat last-run Test AUC 0.8858? **{beat_test_plain}**

---

## Appendix A — full source of `{TRAIN_SCRIPT}`

```python
{train_src}
```

## Appendix B — full source of `{EVAL_SCRIPT}`

```python
{eval_src}
```
"""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_linear_tree] Test AUC={test_metrics['AUC']} KS={test_metrics['KS_Statistic']} "
        f"Gini={test_metrics['Gini']} PSI={score_psi['PSI']}"
    )
    print(f"[eval_linear_tree] Beat last-run Test AUC {LAST_RUN_TEST_AUC}? {beat_test_plain}")
    print(f"[eval_linear_tree] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
