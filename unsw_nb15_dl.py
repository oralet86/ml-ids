# unsw_dl.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from unsw_nb15 import (
    get_unsw_nb15,
    preprocess_unsw_nb15,
    preprocess_fit_transform,
    preprocess_transform,
    metrics_binary,
    stratified_holdout,
    save_results,
)

from params import (
    N_TRIALS,
    RANDOM_STATE,
    VAL_FRAC,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    UNSW_DATA_PCT,
)
from utils import UNSW_NB15_PATH, RESULTS_DIR, DL_MODEL_LIST, logger


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# DL knobs (same pattern as CICIDS)
EVAL_EVERY = 1
ACCUM_STEPS = 2

if not torch.cuda.is_available():
    AMP_DTYPE = "fp32"
elif torch.cuda.is_bf16_supported():
    AMP_DTYPE = "bf16"
else:
    AMP_DTYPE = "fp16"


@dataclass(frozen=True)
class DLKnobs:
    batch_size: int
    epochs: int
    patience: int
    eval_every: int
    accum_steps: int
    amp_dtype: str  # "fp16" or "bf16" or "fp32"


def is_sequence_model(model_class: Type[nn.Module]) -> bool:
    # kept aligned with your CICIDS gate
    return model_class.__name__ in {"LSTMModel", "CNNLSTMModel", "FlowTransformerModel"}


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
        pred = (probs >= float(thr)).astype(np.int64)
        m = metrics_binary(y_true_i, pred)
        if m["f1"] > best_m["f1"]:
            best_m = m
            best_thr = float(thr)

    return best_thr, best_m


def build_dl_views(
    *,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # UNSW-specific: categorical-aware preprocessing (fit on train only)
    X_tr_np, X_val_np, _, pre = preprocess_fit_transform(X_train_raw, X_val_raw)

    cat_cols = X_train_raw.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    num_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()

    X_te_np = preprocess_transform(
        X_test_raw,
        fitted_preprocessor=pre,
        train_num_cols=num_cols,
        train_cat_cols=cat_cols,
    )

    y_tr = y_train.astype(np.float32, copy=False)
    y_val_out = y_val.astype(np.float32, copy=False)
    y_te = y_test.astype(np.float32, copy=False)

    return (
        X_tr_np.astype(np.float32, copy=False),
        y_tr,
        X_val_np.astype(np.float32, copy=False),
        y_val_out,
        X_te_np.astype(np.float32, copy=False),
        y_te,
    )


def run_unsw_nb15_dl(
    *,
    model_class: Type[nn.Module],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    knobs: DLKnobs,
    n_trials: int,
    results_dir,
    logger: Any,
) -> Dict[str, Any]:
    t0 = time.time()

    # CICIDS-style skip if 1-class
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

    # Device (same as CICIDS)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # If this is a sequence model, we will feed seq_len=1.
    seq_mode: Optional[str] = None  # None => flat; otherwise "bsf" or "sbf"

    if is_sequence_model(model_class):
        logger.info(
            f"{model_class.__name__} detected as sequence model. Feeding seq_len=1 windows."
        )

    input_dim = int(X_tr.shape[-1])
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

    # CICIDS-style scheduler
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, threshold=1e-4
    )

    use_amp = device.type == "cuda" and knobs.amp_dtype in {"fp16", "bf16"}
    amp_dtype = torch.float16 if knobs.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )

    # Dataloaders
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

    def _shape_xb(xb: torch.Tensor) -> torch.Tensor:
        """
        Shape xb for model input.
        - Flat models: (B, F)
        - Sequence models (seq_len=1):
            * "bsf": (B, 1, F)  (batch_first=True style)
            * "sbf": (1, B, F)  (batch_first=False style)
        """
        if seq_mode is None:
            return xb
        if seq_mode == "bsf":
            return xb.unsqueeze(1)
        if seq_mode == "sbf":
            return xb.unsqueeze(0)
        raise RuntimeError(f"Unknown seq_mode={seq_mode}")

    def _infer_seq_mode() -> str:
        """
        Infer whether the sequence model expects (B, 1, F) or (1, B, F)
        by checking which produces per-sample outputs for a single batch.
        """
        model.eval()
        xb0, _ = next(iter(train_loader))
        xb0 = xb0.to(device, non_blocking=True)
        b = int(xb0.shape[0])

        # Try (B, 1, F) first
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=amp_dtype
            ):
                out_bsf = model(xb0.unsqueeze(1))

        if out_bsf.numel() == b or (out_bsf.ndim >= 1 and int(out_bsf.shape[0]) == b):
            return "bsf"

        # Try (1, B, F)
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=amp_dtype
            ):
                out_sbf = model(xb0.unsqueeze(0))

        if out_sbf.numel() == b or (out_sbf.ndim >= 1 and int(out_sbf.shape[0]) == b):
            return "sbf"

        raise ValueError(
            f"{model_class.__name__}: could not infer sequence layout for seq_len=1. "
            "Model did not produce batch-sized outputs for either (B,1,F) or (1,B,F)."
        )

    # If sequence model, infer how to feed seq_len=1
    if is_sequence_model(model_class):
        seq_mode = _infer_seq_mode()
        logger.info(f"{model_class.__name__}: seq_mode={seq_mode} (seq_len=1)")

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
                    logits = model(_shape_xb(xb)).view(-1).float()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                pr.append(probs.astype(np.float64, copy=False))
                yt.append(yb.to(torch.int64).cpu().numpy())
        return np.concatenate(yt), np.concatenate(pr)

    def eval_fixed(loader: DataLoader, thr: float) -> Dict[str, float]:
        y_true, probs = _collect_probs(loader)
        pred = (probs >= float(thr)).astype(np.int64)
        return metrics_binary(y_true, pred)

    max_grad_norm = 1.0  # CICIDS-style grad clipping knob

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
                out = model(_shape_xb(xb)).view(-1, 1)
                loss = crit(out, yb) / accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((i + 1) % accum == 0) or (i + 1 == len(train_loader))
            if step_now:
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
            "amp": bool(use_amp),
            "amp_dtype": knobs.amp_dtype,
            "eval_every": knobs.eval_every,
            "accum_steps": knobs.accum_steps,
            "threshold": float(best_thr),
            "val_metrics_at_threshold": (best_val_at_thr or {}),
            "scheduler": "ReduceLROnPlateau(monitor=val_f1,factor=0.5,patience=2)",
            "grad_clip_max_norm": float(max_grad_norm),
            "dl_no_smote": True,
            "dl_no_topk": True,
        },
        n_trials=n_trials,
        total_time=time.time() - t0,
        results_dir=results_dir,
        logger=logger,
    )


if __name__ == "__main__":
    df_train, df_test = get_unsw_nb15(
        unsw_dir=UNSW_NB15_PATH,
        pct=UNSW_DATA_PCT,
        random_state=RANDOM_STATE,
        logger=logger,
    )

    df_train = preprocess_unsw_nb15(df_train, logger=logger)
    df_test = preprocess_unsw_nb15(df_test, logger=logger)

    # Build X/y from TRAIN pool
    drop_cols_train = ["label"]
    if "attack_cat" in df_train.columns:
        drop_cols_train.append("attack_cat")

    X_pool = df_train.drop(columns=drop_cols_train)
    y_pool = df_train["label"].to_numpy(dtype=np.int64, copy=False)

    # Build X/y from TEST set
    drop_cols_test = ["label"]
    if "attack_cat" in df_test.columns:
        drop_cols_test.append("attack_cat")

    X_test_raw = df_test.drop(columns=drop_cols_test)
    y_test = df_test["label"].to_numpy(dtype=np.int64, copy=False)

    # Split TRAIN pool -> TRAIN / VAL (stratified)
    X_train_raw, y_train, X_val_raw, y_val = stratified_holdout(
        X_pool, y_pool, val_frac=VAL_FRAC, random_state=RANDOM_STATE
    )

    logger.info(
        "Custom split (DL): "
        f"train_pool={len(X_pool)} -> train={len(X_train_raw)} val={len(X_val_raw)} "
        f"| test={len(X_test_raw)} "
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
    )

    X_tr, y_tr, X_val, y_val_out, X_te, y_te_out = build_dl_views(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
    )

    logger.info(
        "pos_rate: "
        f"train={float(y_tr.mean()):.6f} val={float(y_val_out.mean()):.6f} test={float(y_te_out.mean()):.6f}"
    )

    for m in DL_MODEL_LIST:
        run_unsw_nb15_dl(
            model_class=m,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val_out,
            X_te=X_te,
            y_te=y_te_out,
            knobs=knobs,
            n_trials=N_TRIALS,
            results_dir=RESULTS_DIR,
            logger=logger,
        )
