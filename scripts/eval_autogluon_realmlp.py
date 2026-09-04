"""
AutoGluon + RealMLP (eval only — never fits).

Loads frozen artifacts from scripts/train_autogluon_realmlp.py, scores Test PDs.
Metrics via evaluate_discrimination_and_ks and calculate_psi.
PSI train/expected side = OOF PD (not in-sample full-train PD).

Does not call fit on AutoGluon, RealMLP, or the stack meta-learner.
Does not write last-run or PR #1–#8 artifact names.

Usage (from repo root): python3 scripts/eval_autogluon_realmlp.py
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
from train_autogluon_realmlp import (
    AG_FI_PATH,
    AG_LEADERBOARD_PATH,
    AG_PREDICTOR_DIR,
    EXPECTED_IN_MODEL_FEATURES,
    FeatureFramePrep,
    META_JSON_PATH,
    OOF_PDS_PATH,
    REALMLP_MODEL_PATH,
    REALMLP_PERM_PATH,
    REALMLP_PREPROCESS_PATH,
    REQS_PATH,
    STACK_META_PATH,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
    _ag_positive_pd,
    _assert_in_model_columns,
    _clip_pd,
    _realmlp_pd,
    ensure_ag_predictor_dir,
)
from utils.config import ARTIFACTS_DIR, AUC_MIN, KS_MIN, PSI_WATCH, RAW_TARGET_COL, TARGET
from utils.risk_skills import calculate_psi, evaluate_discrimination_and_ks

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
PR8_DANET_TEST_AUC = 0.8655

TEST_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_test_metrics.json")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "ag_realmlp_run_report.md")

TRAIN_SCRIPT = "scripts/train_autogluon_realmlp.py"
EVAL_SCRIPT = "scripts/eval_autogluon_realmlp.py"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for rec in df.to_dict(orient="records"):
        cells = []
        for c in cols:
            val = rec[c]
            if val is None:
                cells.append("")
            elif isinstance(val, float):
                cells.append(f"{val:.6g}" if abs(val) < 0.001 and val != 0 else f"{val}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def yes_no(flag: bool | None) -> str:
    if flag is None:
        return "n/a"
    return "YES" if flag else "NO"


def _pack_test(y_true: np.ndarray, pd_vec: np.ndarray, oof_pd: np.ndarray | None) -> dict:
    disc = evaluate_discrimination_and_ks(y_true, pd_vec)
    out = {
        "AUC": disc["AUC"],
        "KS_Statistic": disc["KS_Statistic"],
        "Gini": disc["Gini"],
        "Validation_Rating": disc["Validation_Rating"],
        "Optimal_Cutoff_Probability": disc["Optimal_Cutoff_Probability"],
    }
    if oof_pd is not None:
        psi = calculate_psi(oof_pd, pd_vec, num_bins=10)
        out["PSI"] = psi["PSI"]
        out["PSI_Status"] = psi["Status"]
        out["psi_above_0.10"] = bool(float(psi["PSI"]) > PSI_WATCH)
    else:
        out["PSI"] = None
        out["PSI_Status"] = "PSI skipped — no leak-free OOF PD for expected side"
        out["psi_above_0.10"] = None
    return out


def _beats(auc: float) -> dict:
    return {
        "beats_last_run_0.8858": bool(auc > LAST_RUN_TEST_AUC),
        "beats_pr2_stack_0.8885": bool(auc > PR2_STACK_TEST_AUC),
        "beats_pr4_0.8859": bool(auc > PR4_TEST_AUC),
        "beats_pr6_stack_0.8883": bool(auc > PR6_STACK_TEST_AUC),
        "beats_pr6_gbdt_alone_0.8895": bool(auc > PR6_GBDT_TEST_AUC),
        "beats_pr7_stack_0.8951": bool(auc > PR7_STACK_TEST_AUC),
        "beats_pr7_ft_alone_0.8963": bool(auc > PR7_FT_TEST_AUC),
        "beats_pr8_danet_0.8655": bool(auc > PR8_DANET_TEST_AUC),
    }


def _top_features(rows: list[dict], value_key: str, n: int = 8) -> str:
    parts = []
    for i, row in enumerate(rows[:n], start=1):
        parts.append(f"{i}. {row['feature']} ({row.get(value_key)})")
    return "; ".join(parts) if parts else "(none)"


def _oof_col(oof_df: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in oof_df.columns:
        return None
    return oof_df[name].to_numpy(dtype=float)


def main() -> None:
    required = [
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
    if freeze.get("model_type") != "autogluon_realmlp":
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
    leftover = [c for c in feature_cols_split if c != "gender"]
    if set(leftover) != set(EXPECTED_IN_MODEL_FEATURES):
        raise RuntimeError(
            f"in_model_features != load_and_split minus gender. "
            f"got={in_model_features} split_minus_gender={leftover}"
        )
    if RAW_TARGET_COL in test_df.columns or RAW_TARGET_COL in in_model_features:
        raise RuntimeError("loan_paid_back present in eval feature path")

    oof_df = pd.read_csv(OOF_PDS_PATH)
    X_train_persisted = pd.read_csv(X_TRAIN_PATH)
    X_test_persisted = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET].to_numpy()
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET].to_numpy()
    if not np.array_equal(y_test, test_df[TARGET].to_numpy()):
        raise RuntimeError("Persisted y_test does not match load_and_split Test labels")
    if not np.array_equal(y_train, train_df[TARGET].to_numpy()):
        raise RuntimeError("Persisted y_train does not match load_and_split Train labels")

    _assert_in_model_columns(list(X_train_persisted.columns), "eval persisted X_train")
    _assert_in_model_columns(list(X_test_persisted.columns), "eval persisted X_test")
    if len(oof_df) != 14000:
        raise RuntimeError(f"OOF PD matrix has {len(oof_df)} rows, expected 14000")

    raw_X_test = test_df[in_model_features].copy()
    _assert_in_model_columns(list(raw_X_test.columns), "eval raw_X_test")

    cat_cols = list(freeze.get("cat_cols") or [])
    prep = FeatureFramePrep(in_model_features, cat_cols)

    ag_ok = bool((freeze.get("autogluon") or {}).get("ok"))
    rm_ok = bool((freeze.get("realmlp") or {}).get("ok"))
    test_pds: dict[str, np.ndarray] = {}
    failures: dict[str, str | None] = {
        "autogluon": None if ag_ok else str((freeze.get("autogluon") or {}).get("error")),
        "realmlp": None if rm_ok else str((freeze.get("realmlp") or {}).get("error")),
    }

    if ag_ok:
        predictor_path = ensure_ag_predictor_dir()
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor.load(predictor_path)
        try:
            ag_feat_list = list(predictor.features())
        except Exception:  # noqa: BLE001
            ag_feat_list = list(in_model_features)
        if "gender" in ag_feat_list:
            raise RuntimeError(f"Loaded AutoGluon predictor contains gender: {ag_feat_list}")
        if "employment_status" not in ag_feat_list:
            raise RuntimeError(f"Loaded AutoGluon predictor missing employment_status: {ag_feat_list}")
        xt = prep.transform(raw_X_test)
        test_pds["autogluon"] = _clip_pd(_ag_positive_pd(predictor, predictor.predict_proba(xt)))

    if rm_ok:
        if not os.path.exists(REALMLP_MODEL_PATH) or not os.path.exists(REALMLP_PREPROCESS_PATH):
            raise FileNotFoundError("Frozen RealMLP model/preprocess missing")
        rm_model = joblib.load(REALMLP_MODEL_PATH)
        rm_prep = joblib.load(REALMLP_PREPROCESS_PATH)
        _assert_in_model_columns(list(rm_prep.feature_order), "eval realmlp preprocess")
        if "gender" in rm_prep.feature_order or "gender" in rm_prep.cat_cols:
            raise RuntimeError("RealMLP preprocess contains gender")
        if "employment_status" not in rm_prep.cat_cols:
            raise RuntimeError("RealMLP preprocess missing employment_status as categorical")
        test_pds["realmlp"] = _realmlp_pd(rm_model, raw_X_test, rm_prep)

    oof_ag = _oof_col(oof_df, "pd_autogluon")
    oof_rm = _oof_col(oof_df, "pd_realmlp")
    oof_avg = _oof_col(oof_df, "pd_equal_weight_avg")
    oof_stack = _oof_col(oof_df, "pd_stack_lr")

    if "autogluon" in test_pds and "realmlp" in test_pds:
        test_pds["equal_weight_average"] = _clip_pd(0.5 * test_pds["autogluon"] + 0.5 * test_pds["realmlp"])
        if os.path.exists(STACK_META_PATH) and bool((freeze.get("oof_stack") or {}).get("built")):
            meta = joblib.load(STACK_META_PATH)
            X_meta = np.column_stack([test_pds["autogluon"], test_pds["realmlp"]])
            test_pds["stack_lr"] = _clip_pd(meta.predict_proba(X_meta)[:, 1])

    oof_map = {
        "autogluon": oof_ag,
        "realmlp": oof_rm,
        "equal_weight_average": oof_avg,
        "stack_lr": oof_stack,
    }

    test_metrics: dict[str, dict] = {}
    beats: dict[str, dict] = {}
    for name, pd_vec in test_pds.items():
        pack = _pack_test(y_test, pd_vec, oof_map.get(name))
        test_metrics[name] = pack
        beats[name] = _beats(float(pack["AUC"]))
        print(
            f"[eval_ag_realmlp] {name} Test AUC={pack['AUC']} KS={pack['KS_Statistic']} "
            f"Gini={pack['Gini']} PSI={pack['PSI']}",
            flush=True,
        )

    pkg_versions = freeze.get("package_versions") or {}
    ag_info = freeze.get("autogluon") or {}
    rm_info = freeze.get("realmlp") or {}
    stack_info = freeze.get("oof_stack") or {}
    avg_info = freeze.get("equal_weight_average") or {}
    oof_metrics = freeze.get("oof_metrics") or {}
    explains = freeze.get("train_explains") or {}
    device = freeze.get("device") or {}
    seats = (ag_info.get("tabpfn_tabm_realmlp_seats") or {}) if ag_ok else {}

    def _emp_rank() -> dict:
        out = {}
        ag_perm = (explains.get("autogluon_permutation") or {}) if ag_ok else {}
        ag_fi = (explains.get("autogluon_feature_importance") or {}) if ag_ok else {}
        rm_perm = (explains.get("realmlp_permutation") or {}) if rm_ok else {}
        out["autogluon_permutation"] = ag_perm.get("employment_status_rank")
        out["autogluon_feature_importance"] = ag_fi.get("employment_status_rank")
        out["realmlp_permutation"] = rm_perm.get("employment_status_rank")
        return out

    emp_ranks = _emp_rank()
    gender_absent = True
    if ag_ok and (explains.get("autogluon_permutation") or {}).get("gender_present"):
        gender_absent = False
    if rm_ok and (explains.get("realmlp_permutation") or {}).get("gender_present"):
        gender_absent = False

    test_metrics_out = {
        "computed_after_freeze": True,
        "freeze_timestamp_utc": freeze["freeze_timestamp_utc"],
        "eval_never_fits": True,
        "gender_in_model": False,
        "employment_status_in_model": True,
        "monotone_constraints": False,
        "interaction_constraints": False,
        "in_model_features": in_model_features,
        "package_versions": pkg_versions,
        "device": device,
        "autogluon": {
            "ok": ag_ok,
            "version": ag_info.get("version"),
            "preset": ag_info.get("preset_actually_run"),
            "time_limit_s": ag_info.get("time_limit_s"),
            "leaderboard_top": ag_info.get("leaderboard_top"),
            "tabpfn_tabm_realmlp_seats": seats,
            "failure": failures["autogluon"],
        },
        "realmlp": {
            "ok": rm_ok,
            "library": rm_info.get("library"),
            "class": rm_info.get("class"),
            "version": rm_info.get("version"),
            "config_name": rm_info.get("config_name"),
            "hpo_run": rm_info.get("hpo_run"),
            "hpo_reason": rm_info.get("hpo_reason"),
            "constructor_kwargs": rm_info.get("constructor_kwargs"),
            "failure": failures["realmlp"],
        },
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "beats": beats,
        "psi_above_0.10": {k: v.get("psi_above_0.10") for k, v in test_metrics.items()},
        "employment_status_importance_rank": emp_ranks,
        "gender_absent": gender_absent,
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
            "pr8_danet_test_auc": PR8_DANET_TEST_AUC,
        },
        "split": freeze["split"],
        "gates": {
            "AUC_MIN": AUC_MIN,
            "KS_MIN": KS_MIN,
            "PSI_WATCH": PSI_WATCH,
        },
        "oof_stack": {
            "built": bool(stack_info.get("built")),
            "meta_coefs": stack_info.get("meta_coefs"),
            "intercept": stack_info.get("intercept"),
            "reason": stack_info.get("reason"),
        },
        "equal_weight_average": {
            "built": bool(avg_info.get("built")) or ("equal_weight_average" in test_pds),
            "reason": avg_info.get("reason"),
        },
    }
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_metrics_out, fh, ensure_ascii=False, indent=2)

    freeze["test_looked_at"] = True
    freeze["test_metrics"] = test_metrics
    freeze["computed_after_freeze"] = True
    with open(META_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(freeze, fh, ensure_ascii=False, indent=2)

    feature_list = "\n".join(f"{i}. {f}" for i, f in enumerate(in_model_features, start=1))

    def _oof_row(label: str, metrics: dict | None) -> dict:
        if not metrics:
            return {"model": label, "OOF_AUC": "", "OOF_KS": "", "OOF_Gini": ""}
        return {
            "model": label,
            "OOF_AUC": metrics.get("AUC"),
            "OOF_KS": metrics.get("KS_Statistic"),
            "OOF_Gini": metrics.get("Gini"),
        }

    def _test_row(label: str, name: str) -> dict:
        pack = test_metrics.get(name)
        if not pack:
            return {
                "model": label,
                "Test_AUC": "",
                "Test_KS": "",
                "Test_Gini": "",
                "Test_PSI": "",
                "psi_above_0.10": "",
            }
        return {
            "model": label,
            "Test_AUC": pack.get("AUC"),
            "Test_KS": pack.get("KS_Statistic"),
            "Test_Gini": pack.get("Gini"),
            "Test_PSI": pack.get("PSI"),
            "psi_above_0.10": yes_no(pack.get("psi_above_0.10")),
        }

    oof_table = pd.DataFrame(
        [
            _oof_row("AutoGluon", oof_metrics.get("autogluon")),
            _oof_row("RealMLP-TD", oof_metrics.get("realmlp")),
            _oof_row("Equal-weight average", oof_metrics.get("equal_weight_average")),
            _oof_row("OOF logistic stack", oof_metrics.get("stack_lr")),
        ]
    )
    test_table = pd.DataFrame(
        [
            _test_row("AutoGluon", "autogluon"),
            _test_row("RealMLP-TD", "realmlp"),
            _test_row("Equal-weight average", "equal_weight_average"),
            _test_row("OOF logistic stack", "stack_lr"),
        ]
    )

    beat_rows = []
    comparators = [
        ("last-run 0.8858", "beats_last_run_0.8858"),
        ("PR#2 stack 0.8885", "beats_pr2_stack_0.8885"),
        ("PR#4 0.8859", "beats_pr4_0.8859"),
        ("PR#6 stack 0.8883", "beats_pr6_stack_0.8883"),
        ("PR#6 gbdt-alone 0.8895", "beats_pr6_gbdt_alone_0.8895"),
        ("PR#7 stack 0.8951", "beats_pr7_stack_0.8951"),
        ("PR#7 FT-alone 0.8963", "beats_pr7_ft_alone_0.8963"),
        ("PR#8 DANet 0.8655", "beats_pr8_danet_0.8655"),
    ]
    for label, key in comparators:
        row = {"comparator": label}
        for model_name in ["autogluon", "realmlp", "equal_weight_average", "stack_lr"]:
            b = beats.get(model_name)
            row[model_name] = yes_no(b.get(key) if b else None)
        beat_rows.append(row)
    beat_table = pd.DataFrame(beat_rows)

    ag_perm = explains.get("autogluon_permutation") or {}
    rm_perm = explains.get("realmlp_permutation") or {}
    ag_fi = explains.get("autogluon_feature_importance") or {}
    ag_perm_top = _top_features(ag_perm.get("auc_drop_by_feature") or [], "auc_drop")
    rm_perm_top = _top_features(rm_perm.get("auc_drop_by_feature") or [], "auc_drop")

    lb_top = ag_info.get("leaderboard_top") or []
    lb_df = pd.DataFrame(lb_top[:10]) if lb_top else pd.DataFrame()
    keep_cols = [c for c in ["model", "score_val", "pred_time_val", "fit_time", "stack_level", "can_infer"] if c in lb_df.columns]
    if keep_cols:
        lb_df = lb_df[keep_cols]

    coefs = stack_info.get("meta_coefs") or {}
    seats_line = (
        f"TabPFN={yes_no(seats.get('TabPFN_in_ensemble'))}; "
        f"TabM={yes_no(seats.get('TabM_in_ensemble'))}; "
        f"RealMLP={yes_no(seats.get('RealMLP_in_ensemble'))}"
        if ag_ok
        else "n/a (AutoGluon did not run)"
    )

    def _auc(name: str) -> str:
        pack = test_metrics.get(name) or {}
        return str(pack.get("AUC", ""))

    report = f"""# AutoGluon + RealMLP pack (no gender, frozen split) — run report

Two models on the frozen `scripts/train.py::load_and_split` split, reported separately, plus an equal-weight average and an OOF logistic stack when leak-free OOFs exist. Gender is out. `employment_status` is a normal in-model feature. No monotone_constraints. No interaction_constraints. No IV drop, no VIF drop. Last-run `artifacts/lgbm_model.joblib` and PR #1–#8 artifact prefixes were not written.

This document reports numbers. It does **not** declare go / no-go.

## 1. Paths

- Train script: `{TRAIN_SCRIPT}` (fit on TRAIN only, then freeze)
- Eval script: `{EVAL_SCRIPT}` (**never fits**; loads frozen AutoGluon predictor + RealMLP weights + stack meta; `evaluate_discrimination_and_ks` + `calculate_psi` only)
- Split function: `scripts/train.py::load_and_split` imported **unchanged** (`test_size=0.3`, `stratify=default`, `random_state=42`)
- Metrics: `utils.risk_skills.evaluate_discrimination_and_ks` and `utils.risk_skills.calculate_psi`

Eval never fits. Split verified: Train n={train_n} bad_rate={train_bad_rate} (bads={freeze['split']['train_n_bads']}); Test n={test_n} bad_rate={test_bad_rate} (bads={freeze['split']['test_n_bads']}).

## 2. Exact 20 in-model features

`gender_in_model=false`. `employment_status_in_model=true`. `monotone_constraints=false`. `interaction_constraints=false`.

{feature_list}

- `loan_paid_back` in feature matrix: **false**
- IV-drop: **false**. VIF-drop: **false**. `credit_score` stays.
- Confirmed absent from persisted X_train/X_test and importance tables: **gender**
- Confirmed present in all of those: **employment_status** (normal in-model feature)

## 3. AutoGluon

- ran: **{yes_no(ag_ok)}**
- failure: {failures['autogluon'] or 'none'}
- library/class: `{ag_info.get('library')}` / `{ag_info.get('class')}`
- version: **{ag_info.get('version')}**
- preset requested: `{ag_info.get('preset_requested')}`
- preset actually run: `{ag_info.get('preset_actually_run')}`
- extreme attempted: **{yes_no(bool(ag_info.get('extreme_attempted')))}** — {ag_info.get('extreme_skip_reason')}
- time_limit: **{ag_info.get('time_limit_s')}** seconds
- problem_type=`binary`, eval_metric=`roc_auc`, test passed to fit: **false**
- best_model: `{ag_info.get('best_model')}`
- TabPFN / TabM / RealMLP seats in the AG ensemble: **{seats_line}**
- TabPFN names: {seats.get('TabPFN_model_names')}
- TabM names: {seats.get('TabM_model_names')}
- RealMLP names: {seats.get('RealMLP_model_names')}
- predictor path: `{AG_PREDICTOR_DIR}`
- leaderboard (validation, no test):

{md_table(lb_df) if len(lb_df) else '(no leaderboard — AutoGluon did not run or leaderboard empty)'}

## 4. RealMLP

- ran: **{yes_no(rm_ok)}**
- failure: {failures['realmlp'] or 'none'}
- library: `{rm_info.get('library')}`
- class: `{rm_info.get('class')}`
- version: **{rm_info.get('version')}**
- paper: {rm_info.get('paper')}
- config: **{rm_info.get('config_name')}**
- HPO run: **{yes_no(bool(rm_info.get('hpo_run')))}**
- HPO note: {rm_info.get('hpo_reason')}
- constructor kwargs: `{rm_info.get('constructor_kwargs')}`
- published TD defaults: `{rm_info.get('published_td_defaults')}`
- device: cpu
- model SHA-256: `{rm_info.get('model_sha256')}`

## 5. OOF AUC / KS / Gini (train only)

{md_table(oof_table)}

- AutoGluon OOF available: **{yes_no(oof_ag is not None)}**
- RealMLP OOF available: **{yes_no(oof_rm is not None)}**
- RealMLP fold AUCs: {rm_info.get('fold_aucs')}
- RealMLP fold KS: {rm_info.get('fold_ks')}
- Selection used Train OOF only. Test was not used to pick hyperparameters or to fit the stack.

## 6. After freeze: Test AUC / KS / Gini / PSI

PSI expected side = OOF PD; actual side = Test PD (not in-sample full-train PD).

{md_table(test_table)}

- Freeze timestamp UTC: **{freeze['freeze_timestamp_utc']}**
- `computed_after_freeze=true`
- Confirm freeze before test look: **YES** (eval refused to run unless freeze flags were clean)
- Internal bars (unchanged, not a go/no-go): AUC_MIN={AUC_MIN}, KS_MIN={KS_MIN}, PSI_WATCH={PSI_WATCH}

### 6.1 Stack meta coefs

- built: **{yes_no(bool(stack_info.get('built')))}**
- pd_autogluon coef: **{coefs.get('pd_autogluon')}**
- pd_realmlp coef: **{coefs.get('pd_realmlp')}**
- intercept: **{stack_info.get('intercept')}**
- reason: {stack_info.get('reason')}

### 6.2 Plain YES/NO vs comparators (not retrained)

Comparators: last-run Test AUC **0.8858**; PR #2 stack **0.8885**; PR #4 **0.8859**; PR #6 stack **0.8883**; PR #6 gbdt-alone **0.8895**; PR #7 stack **0.8951**; PR #7 FT-alone **0.8963**; PR #8 DANet **0.8655**.

{md_table(beat_table)}

Best reported Test AUC among scorers that ran: AutoGluon={_auc('autogluon')} RealMLP={_auc('realmlp')} average={_auc('equal_weight_average')} stack={_auc('stack_lr')}

## 7. TRAIN explainability (after freeze, in-model features only)

- AutoGluon permutation top: {ag_perm_top}
- AutoGluon permutation employment_status rank: **{emp_ranks.get('autogluon_permutation')}**
- AutoGluon feature_importance employment_status rank: **{emp_ranks.get('autogluon_feature_importance')}**
- RealMLP permutation top: {rm_perm_top}
- RealMLP permutation employment_status rank: **{emp_ranks.get('realmlp_permutation')}**
- gender absent: **{yes_no(gender_absent)}**

## 8. Package versions / device

- python: {pkg_versions.get('python')}
- autogluon.tabular: {pkg_versions.get('autogluon.tabular') or pkg_versions.get('autogluon')}
- torch: {pkg_versions.get('torch')}
- pytabkit: {pkg_versions.get('pytabkit')}
- sklearn: {pkg_versions.get('sklearn')}
- pandas: {pkg_versions.get('pandas')}
- numpy: {pkg_versions.get('numpy')}
- joblib: {pkg_versions.get('joblib')}
- device: {device}

## 9. Artifacts

- `{AG_PREDICTOR_DIR}`
- `{AG_LEADERBOARD_PATH}`
- `{AG_FI_PATH}`
- `{REALMLP_MODEL_PATH}`
- `{REALMLP_PREPROCESS_PATH}`
- `{REALMLP_PERM_PATH}`
- `{STACK_META_PATH}`
- `{META_JSON_PATH}`
- `{OOF_PDS_PATH}`
- `{TEST_METRICS_PATH}`
- `{REPORT_PATH}`
- `{REQS_PATH}`
- `{X_TRAIN_PATH}` / `{Y_TRAIN_PATH}` / `{X_TEST_PATH}` / `{Y_TEST_PATH}`

Explicitly **not** written: `artifacts/lgbm_model.joblib`, `lgbm_linear_tree_*`, `linear_tree_*`, `stack_lr_rf_lgbm_*`, `lgbm_no_gender_emp_overlay_*`, `lgbm_emp_overlay_*`, `lgbm_emp_in_*`, `lgbm_dart_monotone_*`, `dart_monotone_*`, `ensemble_no_gender_*`, `ensemble_fm_ft_*`, `danet_*`.

Governance thresholds in `utils/config.py` were not changed.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"[eval_ag_realmlp] report -> {REPORT_PATH}", flush=True)
    print(f"[eval_ag_realmlp] metrics -> {TEST_METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
