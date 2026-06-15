"""Training + along-time evaluation for the temporal attention probe (Workstream B).

Loss is ``1 - Pearson`` over the batch (matches the along-time scoring objective and MIRAGE),
not MSE. Evaluation concatenates predictions over time on held-out runs and correlates along
time per parcel (``pearson_along_time``) — the *same* metric Method A reports, so the numbers
sit side by side.

Readout level is selected purely by which subjects you pass to ``build_probe_system``:
``["group"]`` -> one head (Kadir group); the 19 subject ids -> one head each (Kadir individual).
The shared ``LatentAttentionTrunk`` is unchanged from the vision probe; the lookback window is
fed straight through ``TokenAdapter``'s 3-D path as a token sequence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mbs.evaluation.evaluate_features_mtrf import pearson_along_time
from .dataset_temporal import build_windowed_design, sampled_windowed_design
from .model import ProbeConfig, SingleRoiProbeSystem


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def corr_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over channels of ``1 - Pearson`` along the batch axis (axis 0).

    Differentiable; scale/shift invariant in ``y_hat`` (so EEG amplitude need not be
    normalised). Numerically matches ``pearson_along_time`` on the same arrays.
    """
    yh = y_hat - y_hat.mean(dim=0, keepdim=True)
    yt = y - y.mean(dim=0, keepdim=True)
    num = (yh * yt).sum(dim=0)
    den = torch.sqrt((yh ** 2).sum(dim=0) * (yt ** 2).sum(dim=0) + eps)
    r = num / (den + eps)
    return (1.0 - r).mean()


# ---------------------------------------------------------------------------
# Model factory + config
# ---------------------------------------------------------------------------

def build_probe_system(in_dim: int, n_parcel: int, subjects: List[str],
                       cfg: Optional[ProbeConfig] = None) -> SingleRoiProbeSystem:
    """Construct the probe. ``subjects=["group"]`` -> group head; many subjects -> per-subject
    heads. ``pos_mode`` defaults to a learned temporal position encoding over lag positions."""
    if cfg is None:
        cfg = ProbeConfig(in_dim=in_dim, pos_mode="learned")
    else:
        cfg.in_dim = in_dim
    neuroid_dims = {s: int(n_parcel) for s in subjects}
    return SingleRoiProbeSystem(cfg=cfg, subjects=subjects, neuroid_dims=neuroid_dims)


@dataclass
class TemporalTrainConfig:
    device: str = "cuda"
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    amp: bool = True
    n_train_time_samples: int = 200
    eval_every: int = 0          # 0 = no mid-training eval
    lr_schedule: str = "cosine"
    seed: int = 42


# ---------------------------------------------------------------------------
# Inference / scoring (along time, Kadir-style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_concat(model: SingleRoiProbeSystem, feats: np.ndarray, lookback: int,
                   subject: str, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """Predict every valid output time (>= lookback-1) for all stimuli, concatenated over time
    in stimulus-major order -> ``[n_stim*(T-(L-1)), P]`` (aligns with ``eeg[:, L-1:, :]``)."""
    model.eval()
    model.to(device)
    T = feats.shape[1]
    L = int(lookback)
    time_idx = np.arange(L - 1, T)
    dummy = np.zeros((feats.shape[0], T, 1), np.float32)
    X, _ = build_windowed_design(feats, dummy, L, time_idx)     # [N, L, d]
    out = []
    for s in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).float().to(device)
        out.append(model(xb, subject=subject).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def score_heldout(model: SingleRoiProbeSystem, feats: np.ndarray, eeg: np.ndarray,
                  lookback: int, subject: str, device: str = "cpu") -> np.ndarray:
    """Per-parcel out-of-sample Pearson r along time (the Method-A-comparable number)."""
    L = int(lookback)
    Yhat = predict_concat(model, feats, L, subject, device=device)
    Y = eeg[:, L - 1:, :].reshape(-1, eeg.shape[2])
    return pearson_along_time(Y, Yhat)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_temporal_probe(
    *,
    feats_train: Dict[str, np.ndarray],
    eeg_train: Dict[str, np.ndarray],
    feats_val: Optional[Dict[str, np.ndarray]],
    eeg_val: Optional[Dict[str, np.ndarray]],
    subjects: List[str],
    in_dim: int,
    n_parcel: int,
    lookback: int,
    train_cfg: TemporalTrainConfig,
    probe_cfg: Optional[ProbeConfig] = None,
) -> Tuple[SingleRoiProbeSystem, List[dict]]:
    """Train one probe over the given subjects (group = single ``"group"`` subject).

    Each epoch: per subject, redraw ``n_train_time_samples`` random output times, build windowed
    samples, and step ``corr_loss`` over shuffled minibatches. Returns (model, history).
    """
    _set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    model = build_probe_system(in_dim, n_parcel, subjects, cfg=probe_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    sched = None
    if train_cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.epochs)
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rngs = {s: np.random.default_rng(train_cfg.seed + i) for i, s in enumerate(subjects)}
    history: List[dict] = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running, n_running = 0.0, 0
        for s in subjects:
            X, Y = sampled_windowed_design(
                feats_train[s], eeg_train[s], lookback,
                train_cfg.n_train_time_samples, rngs[s],
            )
            Xt = torch.from_numpy(X).float()
            Yt = torch.from_numpy(Y).float()
            perm = torch.randperm(Xt.shape[0])
            for b in range(0, Xt.shape[0], train_cfg.batch_size):
                sel = perm[b:b + train_cfg.batch_size]
                if sel.numel() < 2:            # corr needs >=2 rows of variance
                    continue
                xb = Xt[sel].to(device, non_blocking=True)
                yb = Yt[sel].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = corr_loss(model(xb, subject=s), yb)
                scaler.scale(loss).backward()
                if train_cfg.grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                running += float(loss.item()) * sel.numel()
                n_running += int(sel.numel())
        if sched is not None:
            sched.step()

        rec = {"epoch": epoch, "train_loss": running / max(1, n_running)}
        if (train_cfg.eval_every and epoch % train_cfg.eval_every == 0
                and feats_val is not None and eeg_val is not None):
            rs = [score_heldout(model, feats_val[s], eeg_val[s], lookback, s, device.type).mean()
                  for s in subjects]
            rec["val_r"] = float(np.mean(rs))
        history.append(rec)

    return model, history
