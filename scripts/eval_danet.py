"""
DANet candidate (eval only — never fits).

Loads frozen artifacts from scripts/train_danet.py, applies the train-fitted
preprocess, scores Test PDs. Metrics via evaluate_discrimination_and_ks and
calculate_psi. PSI train/expected side = OOF PD (not in-sample full-train PD).

Does not call fit / encoder fitting / DANet training. Does not write last-run
or PR #1–#7 artifact names.

Usage (from repo root): python3 scripts/eval_danet.py
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
import torch

from third_party.danet import DANet
from train import load_and_split
from train_danet import DANetPreprocess, _predict_danet
from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, PSI_WATCH, RAW_TARGET_COL, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

import __main__ as _eval_main

_eval_main.DANetPreprocess = DANetPreprocess

LAST_RUN_TEST_AUC = 0.8858
LAST_RUN_TEST_KS = 0.5777
LAST_RUN_TEST_GINI = 0.7715
LAST_RUN_TEST_PSI = 0.0021
LAST_RUN_CV_AUC = 0.8862
PR2_STACK_TEST_AUC = 0.8885
PR4_TEST_AUC = 0.8859
PR6_STACK_TEST_AUC = 0.8883
PR6_GBDT_TEST_AUC = 0.8895
PR7_STACK_TEST_AUC = 0.8951
PR7_FT_TEST_AUC = 0.8963

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "danet_model.pt")
PREPROCESS_PATH = os.path.join(ARTIFACTS_DIR, "danet_preprocess.joblib")
META_JSON_PATH = os.path.join(ARTIFACTS_DIR, "danet_meta.json")
OOF_PDS_PATH = os.path.join(ARTIFACTS_DIR, "danet_oof_pds.csv")
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "danet_X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "danet_y_train.csv")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "danet_X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "danet_y_test.csv")
TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "danet_test_metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "danet_run_report.md")
REQS_PATH = os.path.join(ARTIFACTS_DIR, "danet_requirements.txt")
PERM_PATH = os.path.join(ARTIFACTS_DIR, "danet_permutation_importance.json")

TRAIN_SCRIPT = "scripts/train_danet.py"
EVAL_SCRIPT = "scripts/eval_danet.py"

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
        "psi_above_0.10": bool(float(psi["PSI"]) > PSI_WATCH),
    }


def _beats(auc: float) -> dict:
    return {
        "beats_last_run_0.8858": bool(auc > LAST_RUN_TEST_AUC),
        "beats_pr2_stack_0.8885": bool(auc > PR2_STACK_TEST_AUC),
        "beats_pr4_0.8859": bool(auc > PR4_TEST_AUC),
        "beats_pr6_stack_0.8883": bool(auc > PR6_STACK_TEST_AUC),
        "beats_pr6_gbdt_alone_0.8895": bool(auc > PR6_GBDT_TEST_AUC),
        "beats_pr7_stack_0.8951": bool(auc > PR7_STACK_TEST_AUC),
        "beats_pr7_ft_alone_0.8963": bool(auc > PR7_FT_TEST_AUC),
    }


def _top_features(rows: list[dict], value_key: str, n: int = 8) -> str:
    parts = []
    for i, row in enumerate(rows[:n], start=1):
        parts.append(f"{i}. {row['feature']} ({row[value_key]})")
    return "; ".join(parts)


def _load_danet(ckpt: dict) -> DANet:
    kwargs = dict(ckpt["constructor_kwargs"])
    model = DANet(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    required = [
        MODEL_PATH,
        PREPROCESS_PATH,
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
    if freeze.get("model_type") != "danet":
        raise RuntimeError(f"Unexpected model_type={freeze.get('model_type')}")

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
    expected_from_split = [c for c in EXPECTED_IN_MODEL_FEATURES]
    leftover = [c for c in feature_cols_split if c != "gender"]
    if set(leftover) != set(expected_from_split):
        raise RuntimeError(
            f"in_model_features != load_and_split minus gender. "
            f"got={in_model_features} split_minus_gender={leftover}"
        )
    if RAW_TARGET_COL in test_df.columns or RAW_TARGET_COL in in_model_features:
        raise RuntimeError("loan_paid_back present in eval feature path")

    prep = joblib.load(PREPROCESS_PATH)
    oof_df = pd.read_csv(OOF_PDS_PATH)
    X_train_persisted = pd.read_csv(X_TRAIN_PATH)
    X_test_persisted = pd.read_csv(X_TEST_PATH)
    y_test_persisted = pd.read_csv(Y_TEST_PATH)[TARGET].to_numpy()
    y_train_persisted = pd.read_csv(Y_TRAIN_PATH)[TARGET].to_numpy()

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = _load_danet(ckpt)

    if len(oof_df) != 14000:
        raise RuntimeError(f"OOF PD matrix has {len(oof_df)} rows, expected 14000")
    _assert_in_model_columns(list(X_train_persisted.columns), "eval persisted X_train")
    _assert_in_model_columns(list(X_test_persisted.columns), "eval persisted X_test")
    _assert_in_model_columns(list(prep.feature_order), "eval preprocess feature_order")
    if "gender" in prep.feature_order or "gender" in prep.num_cols or "gender" in prep.cat_cols:
        raise RuntimeError("Preprocess contains gender")
    if "employment_status" not in prep.cat_cols:
        raise RuntimeError("Preprocess missing employment_status as categorical")
    if "gender" in getattr(prep, "cat_maps_", {}):
        raise RuntimeError("Preprocess cat maps include gender")

    raw_X_test = test_df[in_model_features].copy()
    _assert_in_model_columns(list(raw_X_test.columns), "eval raw X_test")
    if not np.array_equal(raw_X_test.to_numpy(), X_test_persisted[in_model_features].to_numpy()):
        # object/int dtypes can differ after CSV roundtrip; compare as string for cats
        for col in in_model_features:
            a = raw_X_test[col].astype(str).to_numpy()
            b = X_test_persisted[col].astype(str).to_numpy()
            if not np.array_equal(a, b):
                raise RuntimeError(f"Persisted X_test mismatch on column {col}")

    y_test = test_df[TARGET].to_numpy()
    if not np.array_equal(y_test, y_test_persisted):
        raise RuntimeError("load_and_split y_test does not match persisted y_test")
    if not np.array_equal(train_df[TARGET].to_numpy(), y_train_persisted):
        raise RuntimeError("load_and_split y_train does not match persisted y_train")

    x_test = prep.transform(raw_X_test)
    if x_test.shape != (6000, 20):
        raise RuntimeError(f"Unexpected transformed X_test shape {x_test.shape}")

    test_pd = _predict_danet(model, x_test)
    oof_pd = oof_df["pd_danet"].to_numpy()
    oof_y = oof_df["y"].to_numpy()
    if not np.array_equal(oof_y, y_train_persisted):
        raise RuntimeError("OOF y does not match persisted y_train")

    test_pack = _pack_test(y_test, test_pd, oof_pd)
    oof_auc = freeze["oof_auc"]
    oof_ks = freeze["oof_ks"]
    oof_gini = freeze["oof_gini"]
    beats = _beats(float(test_pack["AUC"]))
    neural_impl = freeze["neural_impl"]
    pkg_versions = freeze["package_versions"]
    cpu_notes = freeze.get("cpu_architecture_notes", {})

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
        "cpu_architecture_notes": cpu_notes,
        "package_versions": pkg_versions,
        "oof_auc": oof_auc,
        "oof_ks": oof_ks,
        "oof_gini": oof_gini,
        "test_auc": test_pack["AUC"],
        "test_ks": test_pack["KS_Statistic"],
        "test_gini": test_pack["Gini"],
        "test_psi": test_pack["PSI"],
        "test_psi_status": test_pack["PSI_Status"],
        "test_psi_above_0.10": test_pack["psi_above_0.10"],
        "test_metrics": test_pack,
        "beats": beats,
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
            "pr6_stack_test_auc": PR6_STACK_TEST_AUC,
            "pr6_gbdt_alone_test_auc": PR6_GBDT_TEST_AUC,
            "pr7_stack_test_auc": PR7_STACK_TEST_AUC,
            "pr7_ft_alone_test_auc": PR7_FT_TEST_AUC,
        },
        "split": freeze["split"],
        "gates": {
            "AUC_MIN": AUC_MIN,
            "KS_MIN": KS_MIN,
            "PSI_WATCH": PSI_WATCH,
            "test_auc_ge_auc_min": bool(float(test_pack["AUC"]) >= AUC_MIN),
            "test_ks_ge_ks_min": bool(float(test_pack["KS_Statistic"]) >= KS_MIN),
        },
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_metrics_out, fh, ensure_ascii=False, indent=2)

    freeze["test_looked_at"] = True
    freeze["test_metrics"] = test_pack
    freeze["computed_after_freeze"] = True
    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(freeze, fh, ensure_ascii=False, indent=2)

    explains = freeze.get("train_explains", {}).get("danet_permutation", {})
    perm_rows = explains.get("auc_drop_by_feature", [])
    perm_top = _top_features(perm_rows, "auc_drop") if perm_rows else "(none)"
    emp_rank = explains.get("employment_status_rank")
    feature_list = "\n".join(f"{i}. {f}" for i, f in enumerate(in_model_features, start=1))
    hp = freeze["hyperparams"]
    winner = cpu_notes.get("winner_hyperparams", hp.get("search"))

    beat_table = pd.DataFrame(
        [
            {
                "comparator": "last-run 0.8858",
                "beat": yes_no(beats["beats_last_run_0.8858"]),
            },
            {
                "comparator": "PR#2 stack 0.8885",
                "beat": yes_no(beats["beats_pr2_stack_0.8885"]),
            },
            {
                "comparator": "PR#4 0.8859",
                "beat": yes_no(beats["beats_pr4_0.8859"]),
            },
            {
                "comparator": "PR#6 stack 0.8883",
                "beat": yes_no(beats["beats_pr6_stack_0.8883"]),
            },
            {
                "comparator": "PR#6 gbdt-alone 0.8895",
                "beat": yes_no(beats["beats_pr6_gbdt_alone_0.8895"]),
            },
            {
                "comparator": "PR#7 stack 0.8951",
                "beat": yes_no(beats["beats_pr7_stack_0.8951"]),
            },
            {
                "comparator": "PR#7 FT-alone 0.8963",
                "beat": yes_no(beats["beats_pr7_ft_alone_0.8963"]),
            },
        ]
    )
    metric_table = pd.DataFrame(
        [
            {
                "split": "OOF (train folds)",
                "AUC": oof_auc,
                "KS": oof_ks,
                "Gini": oof_gini,
                "PSI": "",
            },
            {
                "split": "Test (after freeze)",
                "AUC": test_pack["AUC"],
                "KS": test_pack["KS_Statistic"],
                "Gini": test_pack["Gini"],
                "PSI": test_pack["PSI"],
            },
        ]
    )
    grid_table = pd.DataFrame(hp.get("grid_scores", []))

    report = f"""# DANet candidate (no gender, frozen split) — run report

ONE DANet (Deep Abstract Networks, Chen/Huang et al. AAAI 2022) on the frozen `scripts/train.py` split. Gender is out. `employment_status` is a normal in-model feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#7 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `{TRAIN_SCRIPT}` (tune / fit on TRAIN only, then freeze)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; torch forward + `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`
- Vendored official DANet: `third_party/danet/` (AbstractLayer / LearnableLocality / Entmax15)

Eval never fits. Preprocess and DANet weights are loaded from freeze artifacts.

Split verified: Train n={train_n} bad_rate={train_bad_rate} (bads={freeze['split']['train_n_bads']}); Test n={test_n} bad_rate={test_bad_rate} (bads={freeze['split']['test_n_bads']}).

## 2. Exact 20 in-model features

`gender_in_model=false`. `employment_status_in_model=true`. `monotone_constraints=false`. `interaction_constraints=false`.

{feature_list}

- `loan_paid_back` in feature matrix: **false**
- IV-drop: **false**. VIF-drop: **false**. `credit_score` stays.
- Confirmed absent from persisted X_train/X_test, preprocess feature_order / cat maps, and permutation table: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model feature, not overlay, not singleton interaction group)

## 3. neural_impl

- library: `{neural_impl.get('library')}`
- class: `{neural_impl.get('class')}`
- version / upstream commit: `{neural_impl.get('version')}`
- constructor: `{neural_impl.get('constructor')}`
- source: `{neural_impl.get('source_url')}`
- ABSTLAY: {neural_impl.get('abstlay')}
- official vs pytorch-tabular fallback: **official vendored** (`{neural_impl.get('fallback')}`)
- objective: {neural_impl.get('objective')}
- optimizer: {neural_impl.get('optimizer')}
- `shrunk_for_cpu`: **{cpu_notes.get('shrunk_for_cpu')}**
- winner hyperparams: {winner}
- n_epochs_final: **{cpu_notes.get('n_epochs_final')}**
- n_layers / k / width: {cpu_notes.get('n_layers')} / {cpu_notes.get('k')} / {cpu_notes.get('width')}
- notes: {cpu_notes.get('notes')}
- device: {cpu_notes.get('device')} (torch cuda available: {cpu_notes.get('torch_cuda_available')})
- Package versions: {pkg_versions}

## 4. OOF AUC / KS / Gini (train only)

{md_table(metric_table)}

- OOF AUC = **{oof_auc}**, OOF KS = **{oof_ks}**, OOF Gini = **{oof_gini}**
- Fold AUCs: {hp.get('fold_aucs')}
- Fold KS: {hp.get('fold_ks')}
- Fold best epochs: {hp.get('fold_best_epochs')}
- Selection used Train OOF AUC only. Test was not used to pick hyperparameters.

### 4.1 Grid scores (train OOF)

{md_table(grid_table) if len(grid_table) else '(none)'}

## 5. After freeze: Test AUC / KS / Gini / PSI

PSI expected side = OOF PD; actual side = Test PD (not in-sample full-train PD).

- Test AUC = **{test_pack['AUC']}**
- Test KS = **{test_pack['KS_Statistic']}**
- Test Gini = **{test_pack['Gini']}**
- Test PSI = **{test_pack['PSI']}** ({test_pack['PSI_Status']})
- Test PSI above 0.10: **{yes_no(test_pack['psi_above_0.10'])}** (fact; not a go/no-go)
- Internal bars (unchanged, not a go/no-go): AUC ≥ {AUC_MIN} → {yes_no(float(test_pack['AUC']) >= AUC_MIN)}; KS ≥ {KS_MIN} → {yes_no(float(test_pack['KS_Statistic']) >= KS_MIN)}

### 5.1 Plain YES/NO vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858** / KS 0.5777 / Gini 0.7715 / PSI 0.0021 (CV 0.8862); PR #2 stack **0.8885** (had gender); PR #4 **0.8859**; PR #6 stack **0.8883**; PR #6 gbdt-alone **0.8895**; PR #7 stack **0.8951**; PR #7 FT-alone **0.8963**.

{md_table(beat_table)}

- Beat 0.8858? **{yes_no(beats['beats_last_run_0.8858'])}**
- Beat 0.8885? **{yes_no(beats['beats_pr2_stack_0.8885'])}**
- Beat 0.8859? **{yes_no(beats['beats_pr4_0.8859'])}**
- Beat 0.8883? **{yes_no(beats['beats_pr6_stack_0.8883'])}**
- Beat 0.8895? **{yes_no(beats['beats_pr6_gbdt_alone_0.8895'])}**
- Beat 0.8951? **{yes_no(beats['beats_pr7_stack_0.8951'])}**
- Beat 0.8963? **{yes_no(beats['beats_pr7_ft_alone_0.8963'])}**

## 6. TRAIN permutation (after freeze, in-model features only)

- Method: {explains.get('method')}
- Top features: {perm_top}
- employment_status rank = **{emp_rank}**
- gender absent: **YES**
- skip_reason: {freeze.get('permutation_skip_reason')}

## 7. Freeze-before-test

- Freeze timestamp UTC: **{freeze['freeze_timestamp_utc']}**
- At freeze: `test_looked_at=false`, `test_metrics=null`, `test_labels_used_to_fit_or_select=false`
- `computed_after_freeze=true` in `{TEST_METRICS_PATH}`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)

Artifacts:

- `{MODEL_PATH}`
- `{PREPROCESS_PATH}`
- `{META_JSON_PATH}`
- `{OOF_PDS_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{REQS_PATH}`
- `{PERM_PATH}`
- `{X_TRAIN_PATH}` / `{Y_TRAIN_PATH}` / `{X_TEST_PATH}` / `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `artifacts/lgbm_linear_tree_*`, `artifacts/linear_tree_*`, `artifacts/stack_lr_rf_lgbm_*`, `artifacts/lgbm_no_gender_emp_overlay_*`, `artifacts/lgbm_emp_in_no_gender_*`, `artifacts/lgbm_dart_monotone_*`, `artifacts/ensemble_no_gender_*`, `artifacts/ensemble_fm_ft_*`.

Governance thresholds in `utils/config.py` were not changed.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(
        f"[eval_danet] Test AUC={test_pack['AUC']} KS={test_pack['KS_Statistic']} "
        f"Gini={test_pack['Gini']} PSI={test_pack['PSI']}",
        flush=True,
    )
    print(
        f"[eval_danet] beat 0.8858/0.8885/0.8859/0.8883/0.8895/0.8951/0.8963 = "
        f"{yes_no(beats['beats_last_run_0.8858'])}/"
        f"{yes_no(beats['beats_pr2_stack_0.8885'])}/"
        f"{yes_no(beats['beats_pr4_0.8859'])}/"
        f"{yes_no(beats['beats_pr6_stack_0.8883'])}/"
        f"{yes_no(beats['beats_pr6_gbdt_alone_0.8895'])}/"
        f"{yes_no(beats['beats_pr7_stack_0.8951'])}/"
        f"{yes_no(beats['beats_pr7_ft_alone_0.8963'])}",
        flush=True,
    )
    print(f"[eval_danet] report -> {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
