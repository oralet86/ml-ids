# CICIDS_dl.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from cicids2017 import (
    load_cicids2017,
    preprocess_cicids2017,
    metrics_binary,
    numeric_preprocess_fit_transform,
    save_results,
    numeric_preprocess_transform,
    stratified_holdout,
)

from params import (
    N_TRIALS,
    RANDOM_STATE,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    CICIDS_DATA_PCT,
    VAL_FRAC,
)
from utils import CICIDS2017_PATH, RESULTS_DIR, DL_MODEL_LIST, logger


# CUDA speed knobs
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# DL knobs
EVAL_EVERY = 1
ACCUM_STEPS = 2

if not torch.cuda.is_available():
    AMP_DTYPE = "fp32"
elif torch.cuda.is_bf16_supported():
    AMP_DTYPE = "bf16"
else:
    AMP_DTYPE = "fp16"

SEQ_LEN = 8
SEQ_STRIDE = 1
WINDOW_LABEL_MODE = "max"


@dataclass(frozen=True)
class DLKnobs:
    batch_size: int
    epochs: int
    patience: int
    eval_every: int
    accum_steps: int
    amp_dtype: str  # "fp16" or "bf16"
    seq_len: int
    seq_stride: int
    window_label_mode: str  # "last" or "max"


def is_sequence_model(model_class: Type[nn.Module]) -> bool:
    return model_class.__name__ in {"LSTMModel", "CNNLSTMModel", "FlowTransformerModel"}


_TIME_COL_CANDIDATES = ["Timestamp", "time", "Time", "Flow Timestamp"]
_FLOW_KEY_COLS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Protocol",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
]


def extract_time_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in _TIME_COL_CANDIDATES:
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce", utc=False)
    return None


def extract_flow_groups(df: pd.DataFrame) -> Optional[pd.Series]:
    if "Flow ID" in df.columns:
        return df["Flow ID"].astype("string")

    present = [c for c in _FLOW_KEY_COLS if c in df.columns]
    if not present:
        return None

    parts = [df[c].astype("string") for c in present]
    g = parts[0]
    for p in parts[1:]:
        g = g + "|" + p
    return g


def make_temporal_windows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    times: Optional[np.ndarray],
    seq_len: int,
    stride: int,
    label_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(X.shape[0])
    if n < seq_len:
        raise ValueError(f"Not enough rows for windowing: N={n} < seq_len={seq_len}")

    if groups is None:
        groups = np.zeros(n, dtype=np.int64)

    if times is None:
        times = np.arange(n, dtype=np.int64)
    else:
        if np.issubdtype(times.dtype, np.datetime64):
            t_int = times.astype("datetime64[ns]").astype(np.int64)
            bad = t_int == np.iinfo(np.int64).min
            if bad.any():
                t_int[bad] = np.iinfo(np.int64).max
            times = t_int
        else:
            times = times.astype(np.int64, copy=False)

    Xw_list: List[np.ndarray] = []
    yw_list: List[np.ndarray] = []

    order = np.lexsort((times, groups))
    groups_sorted = groups[order]
    boundaries = np.flatnonzero(
        np.r_[True, groups_sorted[1:] != groups_sorted[:-1], True]
    )

    for s, e in zip(boundaries[:-1], boundaries[1:]):
        idx = order[s:e]
        m = idx.size
        if m < seq_len:
            continue

        for start in range(0, m - seq_len + 1, stride):
            w_idx = idx[start : start + seq_len]
            Xw_list.append(X[w_idx][None, ...])
            if label_mode == "last":
                yw_list.append(np.asarray([y[w_idx[-1]]], dtype=np.float32))
            elif label_mode == "max":
                yw_list.append(np.asarray([y[w_idx].max()], dtype=np.float32))
            else:
                raise ValueError(
                    f"Unknown label_mode='{label_mode}' (use 'last' or 'max')"
                )

    if not Xw_list:
        raise ValueError(
            f"No windows produced (seq_len={seq_len}). "
            "Likely your flow groups are too short after filtering."
        )

    Xw = np.concatenate(Xw_list, axis=0)
    yw = np.concatenate(yw_list, axis=0)

    Xw = np.ascontiguousarray(Xw, dtype=np.float32)
    yw = np.ascontiguousarray(yw, dtype=np.float32)
    return Xw, yw


def process_holdout_data_dl(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    t_tr = extract_time_series(X_train_raw)
    t_te = extract_time_series(X_test_raw)
    g_tr = extract_flow_groups(X_train_raw)
    g_te = extract_flow_groups(X_test_raw)

    X_tr, X_te, _, _, _ = numeric_preprocess_fit_transform(X_train_raw, X_test_raw)

    y_tr = y_train.astype(np.float32, copy=False)
    y_te = y_test.astype(np.float32, copy=False)

    t_tr_np = None if t_tr is None else t_tr.to_numpy()
    t_te_np = None if t_te is None else t_te.to_numpy()
    g_tr_np = None if g_tr is None else g_tr.to_numpy()
    g_te_np = None if g_te is None else g_te.to_numpy()

    return X_tr, y_tr, X_te, y_te, g_tr_np, t_tr_np, g_te_np, t_te_np


def process_split_dl(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    *,
    names: List[str],
    imp: SimpleImputer,
    sc: MinMaxScaler,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    t = extract_time_series(X_raw)
    g = extract_flow_groups(X_raw)

    X_np = numeric_preprocess_transform(X_raw, names=names, imp=imp, sc=sc)
    y_np = y.astype(np.float32, copy=False)

    t_np = None if t is None else t.to_numpy()
    g_np = None if g is None else g.to_numpy()

    return X_np, y_np, g_np, t_np


def build_dl_views(
    *,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    knobs: "DLKnobs",
    logger: Any,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_tr_np, X_val_np, names, imp, sc = numeric_preprocess_fit_transform(
        X_train_raw, X_val_raw
    )
    X_te_np = numeric_preprocess_transform(X_test_raw, names=names, imp=imp, sc=sc)

    y_tr = y_train.astype(np.float32, copy=False)
    y_val_out = y_val.astype(np.float32, copy=False)
    y_te = y_test.astype(np.float32, copy=False)

    g_tr = extract_flow_groups(X_train_raw)
    t_tr = extract_time_series(X_train_raw)
    g_val = extract_flow_groups(X_val_raw)
    t_val = extract_time_series(X_val_raw)
    g_te = extract_flow_groups(X_test_raw)
    t_te = extract_time_series(X_test_raw)

    g_tr_np = None if g_tr is None else g_tr.to_numpy()
    t_tr_np = None if t_tr is None else t_tr.to_numpy()
    g_val_np = None if g_val is None else g_val.to_numpy()
    t_val_np = None if t_val is None else t_val.to_numpy()
    g_te_np = None if g_te is None else g_te.to_numpy()
    t_te_np = None if t_te is None else t_te.to_numpy()

    X_tr_seq, y_tr_seq = make_temporal_windows(
        X_tr_np,
        y_tr,
        groups=g_tr_np,
        times=t_tr_np,
        seq_len=knobs.seq_len,
        stride=knobs.seq_stride,
        label_mode=knobs.window_label_mode,
    )
    X_val_seq, y_val_seq = make_temporal_windows(
        X_val_np,
        y_val_out,
        groups=g_val_np,
        times=t_val_np,
        seq_len=knobs.seq_len,
        stride=knobs.seq_stride,
        label_mode=knobs.window_label_mode,
    )
    X_te_seq, y_te_seq = make_temporal_windows(
        X_te_np,
        y_te,
        groups=g_te_np,
        times=t_te_np,
        seq_len=knobs.seq_len,
        stride=knobs.seq_stride,
        label_mode=knobs.window_label_mode,
    )

    logger.info(
        f"DL windows: train={X_tr_seq.shape} val={X_val_seq.shape} test={X_te_seq.shape} "
        f"(seq_len={knobs.seq_len}, stride={knobs.seq_stride}, label_mode={knobs.window_label_mode})"
    )

    return (
        X_tr_np,
        y_tr,
        X_val_np,
        y_val_out,
        X_te_np,
        y_te,
        X_tr_seq,
        y_tr_seq,
        X_val_seq,
        y_val_seq,
        X_te_seq,
        y_te_seq,
    )


def _best_threshold_f1(
    y_true: np.ndarray, probs: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    y_true_i = y_true.astype(np.int64, copy=False)
    probs = probs.astype(np.float64, copy=False)

    uniq = np.unique(probs)
    if uniq.size == 1:
        thr = float(uniq[0])
        pred = (probs >= thr).astype(np.int64)
        return thr, metrics_binary(y_true_i, pred)

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([uniq[0] - 1e-12], mids, [uniq[-1] + 1e-12]))

    best_thr = float(candidates[0])
    best_m = {"f1": -1.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    for thr in candidates:
        pred = (probs >= thr).astype(np.int64)
        m = metrics_binary(y_true_i, pred)
        if m["f1"] > best_m["f1"]:
            best_m = m
            best_thr = float(thr)

    return best_thr, best_m


def run_cicids2017_dl(
    *,
    model_class: Type[nn.Module],
    X_tr_flat: np.ndarray,
    y_tr_flat: np.ndarray,
    X_val_flat: np.ndarray,
    y_val_flat: np.ndarray,
    X_te_flat: np.ndarray,
    y_te_flat: np.ndarray,
    X_tr_seq: np.ndarray,
    y_tr_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    X_te_seq: np.ndarray,
    y_te_seq: np.ndarray,
    knobs: DLKnobs,
    n_trials: int,
    results_dir,
    logger: Any,
) -> Dict[str, Any]:
    t0 = time.time()

    if is_sequence_model(model_class):
        X_tr, y_tr = X_tr_seq, y_tr_seq
        X_val, y_val = X_val_seq, y_val_seq
        X_te, y_te = X_te_seq, y_te_seq
    else:
        X_tr, y_tr = X_tr_flat, y_tr_flat
        X_val, y_val = X_val_flat, y_val_flat
        X_te, y_te = X_te_flat, y_te_flat

    if len(np.unique(y_tr)) < 2:
        logger.warning("Training has only 1 class after preprocessing. Skipping.")
        return save_results(
            model_name=model_class.__name__,
            best_value=0.0,
            best_metrics={"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0},
            best_params={},
            n_trials=n_trials,
            total_time=time.time() - t0,
            results_dir=results_dir,
            logger=logger,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    input_dim = X_tr.shape[-1]
    model = model_class(input_dim=input_dim).to(device)

    pos = float(np.sum(y_tr))
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(
        f"pos_weight={float(pos_weight.item()):.4f} (pos={pos:.0f}, neg={neg:.0f})"
    )

    try:
        opt = optim.AdamW(
            model.parameters(),
            lr=4e-3,
            weight_decay=1e-4,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # NEW: scheduler (monitor val F1)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, threshold=1e-4
    )

    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if knobs.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    X_tr = np.ascontiguousarray(X_tr)
    y_tr = np.ascontiguousarray(y_tr)
    X_val = np.ascontiguousarray(X_val)
    y_val = np.ascontiguousarray(y_val)
    X_te = np.ascontiguousarray(X_te)
    y_te = np.ascontiguousarray(y_te)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=knobs.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=knobs.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=knobs.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    best_f1 = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_thr: float = 0.5
    best_val_at_thr: Dict[str, float] | None = None
    bad = 0

    def _collect_probs(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        yt: List[np.ndarray] = []
        pr: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    logits = model(xb).view(-1).float()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                pr.append(probs.astype(np.float64, copy=False))
                yt.append(yb.to(torch.int64).cpu().numpy())
        return np.concatenate(yt), np.concatenate(pr)

    def eval_fixed(loader: DataLoader, thr: float) -> Dict[str, float]:
        y_true, probs = _collect_probs(loader)
        pred = (probs >= float(thr)).astype(np.int64)
        return metrics_binary(y_true, pred)

    # NEW: grad clipping knob
    max_grad_norm = 1.0

    pbar = tqdm(range(knobs.epochs), desc=f"{model_class.__name__}", unit="epoch")
    for epoch in pbar:
        model.train()
        opt.zero_grad(set_to_none=True)

        accum = max(1, int(knobs.accum_steps))

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1, 1)

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=amp_dtype
            ):
                out = model(xb).view(-1, 1)
                loss = crit(out, yb) / accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((i + 1) % accum == 0) or (i + 1 == len(train_loader))
            if step_now:
                # NEW: grad clipping (after unscale if using AMP)
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

        do_eval = (
            (epoch == 0)
            or ((epoch + 1) % max(1, int(knobs.eval_every)) == 0)
            or (epoch + 1 == knobs.epochs)
        )
        if not do_eval:
            pbar.set_postfix(val_f1=f"{best_f1:.4f}", bad=bad, eval="skip")
            continue

        yv_true, yv_probs = _collect_probs(val_loader)
        thr_now, m_val = _best_threshold_f1(yv_true, yv_probs)

        # NEW: scheduler step on current val F1
        sched.step(m_val["f1"])

        pbar.set_postfix(
            val_f1=f"{m_val['f1']:.4f}",
            val_acc=f"{m_val['accuracy']:.4f}",
            thr=f"{thr_now:.4f}",
            lr=f"{opt.param_groups[0]['lr']:.2e}",
            bad=bad,
        )

        if m_val["f1"] > best_f1:
            best_f1 = m_val["f1"]
            best_thr = float(thr_now)
            best_val_at_thr = dict(m_val)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= knobs.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    m_test = eval_fixed(test_loader, best_thr)

    return save_results(
        model_name=model_class.__name__,
        best_value=m_test["f1"],
        best_metrics=m_test,
        best_params={
            "amp": use_amp,
            "amp_dtype": knobs.amp_dtype,
            "eval_every": knobs.eval_every,
            "accum_steps": knobs.accum_steps,
            "seq_len": knobs.seq_len,
            "seq_stride": knobs.seq_stride,
            "window_label_mode": knobs.window_label_mode,
            "threshold": float(best_thr),
            "val_metrics_at_threshold": (best_val_at_thr or {}),
            "scheduler": "ReduceLROnPlateau(monitor=val_f1,factor=0.5,patience=2)",
            "grad_clip_max_norm": float(max_grad_norm),
            "dl_no_smote": True,
            "dl_no_topk": True,
            "eval_split": "thursday",
        },
        n_trials=n_trials,
        total_time=time.time() - t0,
        results_dir=results_dir,
        logger=logger,
    )


if __name__ == "__main__":
    df = preprocess_cicids2017(
        load_cicids2017(
            cicids_dir=CICIDS2017_PATH,
            label_col="Label",
            pct=CICIDS_DATA_PCT,
            random_state=RANDOM_STATE,
            logger=logger,
        ),
        logger=logger,
    )

    day = df["__day"].astype("string").str.strip().str.lower()

    train_pool_df = df.loc[day.isin(("monday", "tuesday", "wednesday"))].copy()
    test_df = df.loc[day.isin(("thursday", "friday"))].copy()

    if train_pool_df.empty:
        raise ValueError("Train pool (Mon-Wed) is empty.")
    if test_df.empty:
        raise ValueError("Test (Thu-Fri) is empty.")

    # Binary labels
    y_pool = (
        (train_pool_df["Label"].astype("string").str.strip() != "BENIGN")
        .astype(np.int64)
        .to_numpy()
    )
    y_test = (
        (test_df["Label"].astype("string").str.strip() != "BENIGN")
        .astype(np.int64)
        .to_numpy()
    )

    X_pool = train_pool_df.drop(columns=["Label", "__day"])
    X_test_raw = test_df.drop(columns=["Label", "__day"])

    X_train_raw, y_train, X_val_raw, y_val = stratified_holdout(
        X_pool,
        y_pool,
        val_frac=VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Custom split: "
        f"train_pool(Mon-Wed)={len(train_pool_df)} -> train={len(X_train_raw)} val={len(X_val_raw)} "
        f"| test(Thu-Fri)={len(X_test_raw)} "
        f"| pool_pos={float(y_pool.mean()):.4f} train_pos={float(y_train.mean()):.4f} "
        f"val_pos={float(y_val.mean()):.4f} test_pos={float(y_test.mean()):.4f}"
    )

    knobs = DLKnobs(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        eval_every=EVAL_EVERY,
        accum_steps=ACCUM_STEPS,
        amp_dtype=AMP_DTYPE,
        seq_len=SEQ_LEN,
        seq_stride=SEQ_STRIDE,
        window_label_mode=WINDOW_LABEL_MODE,
    )

    (
        X_tr_flat,
        y_tr_flat,
        X_val_flat,
        y_val_flat,
        X_te_flat,
        y_te_flat,
        X_tr_seq,
        y_tr_seq,
        X_val_seq,
        y_val_seq,
        X_te_seq,
        y_te_seq,
    ) = build_dl_views(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        knobs=knobs,
        logger=logger,
    )

    logger.info(
        "pos_rate: "
        f"train_flat={float(y_tr_flat.mean()):.6f} val_flat={float(y_val_flat.mean()):.6f} test_flat={float(y_te_flat.mean()):.6f} | "
        f"train_seq={float(y_tr_seq.mean()):.6f} val_seq={float(y_val_seq.mean()):.6f} test_seq={float(y_te_seq.mean()):.6f}"
    )

    for m in DL_MODEL_LIST:
        run_cicids2017_dl(
            model_class=m,
            X_tr_flat=X_tr_flat,
            y_tr_flat=y_tr_flat,
            X_val_flat=X_val_flat,
            y_val_flat=y_val_flat,
            X_te_flat=X_te_flat,
            y_te_flat=y_te_flat,
            X_tr_seq=X_tr_seq,
            y_tr_seq=y_tr_seq,
            X_val_seq=X_val_seq,
            y_val_seq=y_val_seq,
            X_te_seq=X_te_seq,
            y_te_seq=y_te_seq,
            knobs=knobs,
            n_trials=N_TRIALS,
            results_dir=RESULTS_DIR,
            logger=logger,
        )
