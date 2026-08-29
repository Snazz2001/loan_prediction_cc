"""
Leak-free stacking candidate (eval only — never fits).

Loads frozen artifacts from scripts/train_stack_lr_rf_lgbm.py, encodes Test with
the train-fitted WOE encoders, scores the three full-train bases, then the meta
learner (fitted on OOF PDs). Metrics via evaluate_discrimination_and_ks and
calculate_psi. Train stacked PD = meta.predict_proba on the frozen OOF 14000x3
matrix (not in-sample full-train base PDs).

Does not call fit / Optuna / encoder fitting. Does not write last-run or PR #1
artifact names.

Usage (from repo root): python3 scripts/eval_stack_lr_rf_lgbm.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from train import load_and_split
from utils.config import (
    ARTIFACTS_DIR,
    AUC_MIN,
    KS_MIN,
    PSI_ACTION,
    PSI_WATCH,
    RANDOM_STATE,
    RAW_TARGET_COL,
    TARGET,
)
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks
from utils.woe_encoding import apply_woe_encoder

LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
LAST_RUN_CV_AUC = 0.8862

BASES_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_bases.joblib")
META_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_meta.joblib")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_encoders.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_oof_pds.csv")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_test_metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "stack_lr_rf_lgbm_run_report.md")
TRAIN_SCRIPT = "scripts/train_stack_lr_rf_lgbm.py"
EVAL_SCRIPT = "scripts/eval_stack_lr_rf_lgbm.py"


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def _encode_woe(df: pd.DataFrame, feature_cols: list, encoders: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe_encoder(df, f, encoders[f]) for f in feature_cols})


def main() -> None:
    required = [BASES_PATH, META_MODEL_PATH, ENCODERS_PATH, META_JSON_PATH, OOF_PDS_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing freeze artifacts {missing}. Run `python3 {TRAIN_SCRIPT}` first."
        )

    with open(META_JSON_PATH, encoding="utf-8") as fh:
        freeze = json.load(fh)

    if freeze.get("test_looked_at") is not False:
        raise RuntimeError("Freeze record must have test_looked_at=false before eval.")
    if freeze.get("test_metrics") is not None:
        raise RuntimeError("Freeze record must have test_metrics=null before eval.")
    if freeze.get("test_labels_used_to_fit_or_select") is not False:
        raise RuntimeError("Freeze record claims test labels were used to fit/select.")

    train_df, test_df, feature_cols_split = load_and_split()
    feature_cols = freeze["feature_cols"]
    if feature_cols != feature_cols_split:
        raise RuntimeError("Frozen feature_cols do not match load_and_split feature_cols")
    if RAW_TARGET_COL in feature_cols or RAW_TARGET_COL in test_df.columns:
        raise RuntimeError(f"{RAW_TARGET_COL} present in eval feature path")
    if TARGET in feature_cols:
        raise RuntimeError("target in feature_cols")

    train_n = len(train_df)
    test_n = len(test_df)
    train_bad_rate = round(float(train_df[TARGET].mean()), 4)
    test_bad_rate = round(float(test_df[TARGET].mean()), 4)
    if train_n != 14000 or train_bad_rate != 0.2001 or test_n != 6000 or test_bad_rate != 0.2002:
        raise RuntimeError(
            f"Split mismatch at eval: train n={train_n} br={train_bad_rate} "
            f"test n={test_n} br={test_bad_rate}"
        )

    encoders = joblib.load(ENCODERS_PATH)
    bases = joblib.load(BASES_PATH)
    meta = joblib.load(META_MODEL_PATH)
    oof_df = pd.read_csv(OOF_PDS_PATH)
    if len(oof_df) != 14000:
        raise RuntimeError(f"OOF PD matrix has {len(oof_df)} rows, expected 14000")

    X_test = _encode_woe(test_df, feature_cols, encoders)
    y_test = test_df[TARGET].to_numpy()

    # Never fits: predict_proba only
    test_base = pd.DataFrame(
        {
            "pd_lr": bases["lr"].predict_proba(X_test)[:, 1],
            "pd_rf": bases["rf"].predict_proba(X_test)[:, 1],
            "pd_lgbm": bases["lgbm"].predict_proba(X_test)[:, 1],
        }
    )
    test_stack_pd = meta.predict_proba(test_base[["pd_lr", "pd_rf", "pd_lgbm"]].to_numpy())[:, 1]

    oof_base_mat = oof_df[["pd_lr", "pd_rf", "pd_lgbm"]].to_numpy()
    train_stack_pd = meta.predict_proba(oof_base_mat)[:, 1]

    test_metrics = evaluate_discrimination_and_ks(y_test, test_stack_pd)
    train_oof_metrics = evaluate_discrimination_and_ks(oof_df["y"].to_numpy(), train_stack_pd)
    score_psi = calculate_psi(train_stack_pd, test_stack_pd, num_bins=10)

    test_auc = float(test_metrics["AUC"])
    beat_last_test = bool(test_auc > LAST_RUN_TEST_AUC)
    oof_stack_auc = float(freeze["oof_stack_auc"])
    beat_last_cv = bool(oof_stack_auc > LAST_RUN_CV_AUC)

    cutoff = test_metrics["Optimal_Cutoff_Probability"]
    y_pred = (test_stack_pd >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_test)
    confusion_summary = {
        "cutoff": round(float(cutoff), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
    }

    per_base_oof = freeze["per_base_oof_metrics"]
    test_metrics_out = {
        "AUC": test_metrics["AUC"],
        "KS_Statistic": test_metrics["KS_Statistic"],
        "Gini": test_metrics["Gini"],
        "PSI": score_psi["PSI"],
        "PSI_Status": score_psi["Status"],
        "Validation_Rating": test_metrics["Validation_Rating"],
        "Optimal_Cutoff_Probability": test_metrics["Optimal_Cutoff_Probability"],
        "beat_last_run_test_auc_0.8858": beat_last_test,
        "last_run_champion": {
            "AUC": LAST_RUN_TEST_AUC,
            "KS": LAST_RUN_TEST_KS,
            "Gini": LAST_RUN_TEST_GINI,
            "PSI": LAST_RUN_TEST_PSI,
        },
        "train_oof_stack": train_oof_metrics,
        "confusion": confusion_summary,
        "eval_never_fits": True,
        "freeze_timestamp_utc": freeze["freeze_timestamp_utc"],
        "test_labels_used_to_fit_or_select": False,
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_metrics_out, fh, ensure_ascii=False, indent=2)

    iv_table = pd.DataFrame(freeze["iv_table_diagnostic"])
    hp = freeze["best_hyperparameters"]
    hp_rows = []
    for group, params in hp.items():
        for k, v in params.items():
            hp_rows.append({"component": group, "hyperparameter": k, "value": v})
    hp_table = pd.DataFrame(hp_rows)

    base_oof_table = pd.DataFrame(
        [
            {
                "model": "LR (base)",
                "OOF_AUC": per_base_oof["lr"]["AUC"],
                "OOF_KS": per_base_oof["lr"]["KS_Statistic"],
                "OOF_Gini": per_base_oof["lr"]["Gini"],
            },
            {
                "model": "RandomForest (base)",
                "OOF_AUC": per_base_oof["rf"]["AUC"],
                "OOF_KS": per_base_oof["rf"]["KS_Statistic"],
                "OOF_Gini": per_base_oof["rf"]["Gini"],
            },
            {
                "model": "LightGBM (base)",
                "OOF_AUC": per_base_oof["lgbm"]["AUC"],
                "OOF_KS": per_base_oof["lgbm"]["KS_Statistic"],
                "OOF_Gini": per_base_oof["lgbm"]["Gini"],
            },
            {
                "model": "Stack meta-LR (this candidate)",
                "OOF_AUC": freeze["oof_stack_metrics"]["AUC"],
                "OOF_KS": freeze["oof_stack_metrics"]["KS_Statistic"],
                "OOF_Gini": freeze["oof_stack_metrics"]["Gini"],
            },
        ]
    )
    compare_table = pd.DataFrame(
        [
            {
                "metric": "AUC",
                "this_candidate_test": test_metrics["AUC"],
                "last_run_champion": LAST_RUN_TEST_AUC,
                "beat_last_run": "YES" if beat_last_test else "NO",
            },
            {
                "metric": "KS",
                "this_candidate_test": test_metrics["KS_Statistic"],
                "last_run_champion": LAST_RUN_TEST_KS,
                "beat_last_run": "n/a",
            },
            {
                "metric": "Gini",
                "this_candidate_test": test_metrics["Gini"],
                "last_run_champion": LAST_RUN_TEST_GINI,
                "beat_last_run": "n/a",
            },
            {
                "metric": "PSI (train OOF stacked PD vs test stacked PD)",
                "this_candidate_test": score_psi["PSI"],
                "last_run_champion": LAST_RUN_TEST_PSI,
                "beat_last_run": "n/a",
            },
        ]
    )

    beat_cv_text = "YES" if beat_last_cv else "NO"
    beat_test_text = "YES" if beat_last_test else "NO"
    plain_test = (
        f"Test AUC {test_metrics['AUC']} DID beat last-run Test AUC {LAST_RUN_TEST_AUC}."
        if beat_last_test
        else f"Test AUC {test_metrics['AUC']} did NOT beat last-run Test AUC {LAST_RUN_TEST_AUC}."
    )
    psi_note = (
        "below watch threshold, no action needed"
        if score_psi["PSI"] < PSI_WATCH
        else (
            "in 0.10–0.25 watch band"
            if score_psi["PSI"] <= PSI_ACTION
            else "above 0.25, recalibration required"
        )
    )

    report = f"""# Stack LR + RF + LightGBM — run report

Leak-free stacking candidate on the frozen `scripts/train.py` split. One stack only (three bases + one meta-learner). No CatBoost / EBM / AutoML. Last-run `artifacts/lgbm_model.joblib` and PR #1 linear-tree artifacts were not written.

## 1. Paths

- Train script: `{TRAIN_SCRIPT}` (tune / fit on TRAIN only, then freeze)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; `predict_proba` + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- WOE: `utils.woe_encoding.fit_woe_encoder` / `apply_woe_encoder`, `WOE_BINS={freeze['woe_bins']}` from `utils.config` (thresholds unchanged)

Eval never fits. Encoders, bases, and meta are loaded from freeze artifacts.

## 2. Train OOF stack AUC vs last-run CV 0.8862

- **Train OOF stack AUC = {freeze['oof_stack_metrics']['AUC']}** (raw {freeze['oof_stack_auc_raw']})
- Train OOF KS = {freeze['oof_stack_metrics']['KS_Statistic']}; Gini = {freeze['oof_stack_metrics']['Gini']}
- Last-run CV AUC = {LAST_RUN_CV_AUC}
- **Beat last-run CV 0.8862? {beat_cv_text}**
- Optuna TPESampler seed=42, {freeze['n_trials_completed']}/{freeze['n_trials_requested']} trials, {freeze['cv_folds']}-fold StratifiedKFold
- Objective = TRAIN OOF stack AUC (meta fitted on OOF base PDs vs train y, scored on those OOF PDs)
- LGBM `n_estimators_final` = max(50, round(mean OOF best_iteration)) = **{freeze['n_estimators_final']}** (fold best_iterations = {freeze['lgbm_best_iterations_oof']})
- Meta LR C selected on the frozen OOF PD matrix only: **{freeze['best_meta_C']}** (grid scores: {freeze['meta_c_scores']})

Selection used Train OOF stack AUC only. Test was not used to choose this candidate.

### 2.1 Winning hyperparameters

{md_table(hp_table)}

## 3. After freeze: Test metrics vs last-run champion

**{plain_test}**

{md_table(compare_table)}

- Test Validation_Rating: {test_metrics['Validation_Rating']}
- Internal gates (unchanged): AUC ≥ {AUC_MIN}, KS ≥ {KS_MIN}. This Test AUC {'PASS' if test_auc >= AUC_MIN else 'FAIL'}, KS {'PASS' if test_metrics['KS_Statistic'] >= KS_MIN else 'FAIL'}.
- PSI status: {score_psi['Status']} ({psi_note})
- Confusion @ KS cutoff {confusion_summary['cutoff']}: TN={confusion_summary['TN']} FP={confusion_summary['FP']} FN={confusion_summary['FN']} TP={confusion_summary['TP']}; accuracy={confusion_summary['accuracy']} precision={confusion_summary['precision']} recall={confusion_summary['recall_sensitivity']} specificity={confusion_summary['specificity']}

Train OOF stacked PD (meta on OOF 14000×3, not full-train in-sample base PDs): AUC={train_oof_metrics['AUC']} KS={train_oof_metrics['KS_Statistic']} Gini={train_oof_metrics['Gini']}.

## 4. Per-base OOF AUC (did the stack add anything?)

{md_table(base_oof_table)}

Last-run CV AUC = {LAST_RUN_CV_AUC}. Compare each base and the stack against that number and against each other. Stack OOF AUC minus best base OOF AUC = {round(freeze['oof_stack_metrics']['AUC'] - max(per_base_oof['lr']['AUC'], per_base_oof['rf']['AUC'], per_base_oof['lgbm']['AUC']), 4)}.

## 5. Artifact paths, freeze-before-test, no test-label leakage

- `{BASES_PATH}` — dict of lr / rf / lgbm fitted on all 14000 train rows
- `{META_MODEL_PATH}` — meta LR fitted on OOF PDs (not refit on in-sample full-train base PDs)
- `{ENCODERS_PATH}` — WOE encoders fit on TRAIN only
- `{META_JSON_PATH}` — freeze record
- `{OOF_PDS_PATH}` — 14000 × 3 base PDs + y
- `{TEST_METRICS_PATH}` — test metrics written by this eval script only
- `{REPORT_PATH}` — this report

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`.

- Freeze timestamp UTC: **{freeze['freeze_timestamp_utc']}**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- Frozen split verified again at eval: Train n={train_n} bad_rate={train_bad_rate}; Test n={test_n} bad_rate={test_bad_rate}
- `loan_paid_back` in feature matrix: **false** (dropped in `load_and_split`; target is `default = 1 - loan_paid_back`)
- IV-drop: **false**. VIF-drop: **false**. All {freeze['n_features']} features kept: {', '.join(feature_cols)}
- Test labels were never used to fit or select. Eval does not call `fit`.

## 6. Plain statement on Test AUC vs 0.8858

{plain_test}

## 7. Split / features (diagnostics)

| sample | n | bad_rate |
| --- | --- | --- |
| Train | {train_n} | {train_bad_rate} |
| Test | {test_n} | {test_bad_rate} |

IV table computed on TRAIN as a diagnostic only (no screening drop):

{md_table(iv_table)}

`employment_status` IV is in the Suspicious / Overfitting band and is **kept** for evaluation; it is **not cleared for production** pending independent validation and fair-lending review.

Governance thresholds in `utils/config.py` were not changed.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_stack] Test AUC={test_metrics['AUC']} KS={test_metrics['KS_Statistic']} "
        f"Gini={test_metrics['Gini']} PSI={score_psi['PSI']} "
        f"beat_0.8858={'YES' if beat_last_test else 'NO'}",
        flush=True,
    )
    print(f"[eval_stack] report -> {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
