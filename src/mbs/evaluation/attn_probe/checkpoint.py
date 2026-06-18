"""Save / load a trained temporal attention probe as a self-contained, reusable mapping.

Workstream B's deliverable is not a score table but the *trained network* that maps a model
layer (e.g. whisper-small blocks.10) to parcel EEG. The mTRF (Workstream A) is a closed-form
ridge that ``insilico_mmn.py`` re-fits on the spot; the attention encoder is gradient-trained,
so to apply it to ARBITRARY new stimuli (MMN tones, held-out speech, or whatever Sophie brings)
its weights + preprocessing must be persisted.

A checkpoint bundles everything ``predict_parcels`` needs: the ``state_dict``, the ``ProbeConfig``,
the per-feature standardization (``mu``/``sd``), the lookback, the parcel definitions, and the
high-pass / fs used at fit time. ``load_probe_checkpoint`` reconstructs an eval-ready model.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from mbs.evaluation.evaluate_features_mtrf import highpass_along_time
from .model import ProbeConfig, SingleRoiProbeSystem
from .engine_temporal import predict_concat

FORMAT_VERSION = 1


def save_probe_checkpoint(
    path,
    *,
    model: SingleRoiProbeSystem,
    cfg: ProbeConfig,
    mu: np.ndarray,
    sd: np.ndarray,
    lookback: int,
    parcel_names: Sequence[str],
    parcel_members: Sequence[str],
    parcel_nc: Sequence[float],
    subjects: Sequence[str],
    highpass_hz: float,
    fs: float,
    layer: str,
    eeg_mu: Optional[Dict[str, np.ndarray]] = None,
    eeg_sd: Optional[Dict[str, np.ndarray]] = None,
    meta: Optional[dict] = None,
) -> Path:
    """Persist a trained probe + the preprocessing needed to reproduce its predictions.

    ``parcel_members`` are the '+'-joined channel lists (one string per parcel), so a reload
    documents exactly which electrodes each output column averaged.

    ``eeg_mu`` / ``eeg_sd`` are the per-subject, per-target z-score stats applied to the EEG
    *targets* at fit time (model trained on MSE → outputs are in z-units). They are stored so
    ``predictions_to_units`` can invert predictions back to the EEG's real amplitude units —
    required for a magnitude-based in-silico MMN criterion (plan §16).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_parcel = len(parcel_names)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "subjects": list(subjects),
            "neuroid_dims": {s: int(n_parcel) for s in subjects},
            "mu": np.asarray(mu, np.float32),
            "sd": np.asarray(sd, np.float32),
            "lookback": int(lookback),
            "parcels": {
                "names": list(parcel_names),
                "members": list(parcel_members),
                "nc_r": np.asarray(parcel_nc, np.float32),
            },
            "highpass_hz": float(highpass_hz),
            "fs": float(fs),
            "layer": str(layer),
            "in_dim": int(cfg.in_dim),
            "eeg_mu": {s: np.asarray(v, np.float32) for s, v in (eeg_mu or {}).items()},
            "eeg_sd": {s: np.asarray(v, np.float32) for s, v in (eeg_sd or {}).items()},
            "meta": dict(meta or {}),
        },
        path,
    )
    return path


def load_probe_checkpoint(path, device: str = "cpu"):
    """Reconstruct an eval-ready probe. Returns ``(model, ckpt_dict)``.

    ``ckpt_dict`` carries ``mu``/``sd``/``lookback``/``parcels``/``highpass_hz``/``fs`` so the
    caller has everything ``predict_parcels`` / ``predict_timecourse`` need.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = ProbeConfig(**ckpt["cfg"])
    model = SingleRoiProbeSystem(cfg=cfg, subjects=ckpt["subjects"],
                                 neuroid_dims=ckpt["neuroid_dims"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _preprocess(feats_raw: np.ndarray, ckpt: dict) -> np.ndarray:
    """Raw model features [.., T, d] -> high-passed, train-stat z-scored (the fit-time pipeline)."""
    f = highpass_along_time(np.asarray(feats_raw, np.float32), ckpt["fs"], ckpt["highpass_hz"])
    return ((f - ckpt["mu"]) / ckpt["sd"]).astype(np.float32)


def predict_parcels(model, ckpt: dict, feats_raw: np.ndarray, subject: str = "group",
                    device: str = "cpu") -> np.ndarray:
    """Predict parcel EEG for a stack of stimuli. ``feats_raw`` [n_stim, T, d] (RAW model feats)
    -> [n_stim*(T-(L-1)), P], concatenated stimulus-major (aligns with ``eeg[:, L-1:, :]``)."""
    f = _preprocess(feats_raw, ckpt)
    return predict_concat(model, f, ckpt["lookback"], subject, device=device)


def predict_timecourse(model, ckpt: dict, feat_1stim: np.ndarray, subject: str = "group",
                       device: str = "cpu"):
    """Single stimulus [T, d] -> (t_idx, predicted EEG [n_t, P]) over valid output bins.

    Mirrors ``insilico_mmn.predict_timecourse`` so the same time-locking / MMN machinery can
    drive the attention encoder by swapping the predictor.
    """
    f = _preprocess(feat_1stim[None], ckpt)
    L = int(ckpt["lookback"])
    T = f.shape[1]
    t_idx = np.arange(L - 1, T)
    pred = predict_concat(model, f, L, subject, device=device)   # [(T-(L-1)), P]
    return t_idx, pred


def predictions_to_units(pred_z: np.ndarray, ckpt: dict, subject: str = "group") -> np.ndarray:
    """Invert the EEG z-scoring: z-unit predictions ``[.., P]`` -> real EEG amplitude units.

    ``real = z * sd + mu`` per target, using the stored fit-time stats. Needed for a
    magnitude-based MMN criterion (the additive ``mu`` cancels in a deviant−standard difference,
    so a difference of two ``predictions_to_units`` outputs is exactly ``z_diff * sd``). Raises
    if the checkpoint predates EEG-stat storage (legacy 1-Pearson checkpoints have none)."""
    mu = ckpt.get("eeg_mu", {}).get(subject)
    sd = ckpt.get("eeg_sd", {}).get(subject)
    if mu is None or sd is None:
        raise KeyError(
            f"checkpoint has no eeg_mu/eeg_sd for subject={subject!r}; it was not trained with "
            "EEG target standardisation, so predictions cannot be put in real units.")
    return np.asarray(pred_z, np.float32) * np.asarray(sd, np.float32) + np.asarray(mu, np.float32)
