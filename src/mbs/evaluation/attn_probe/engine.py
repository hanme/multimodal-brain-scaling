"""
Training + evaluation for ONE ROI and ONE layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .h5io import load_layer_features
from .dataset import (
    build_subject_loaders,
    infer_in_dim_from_layer_feats,
)
from .metrics import RunningPearson
from .model import ProbeConfig, SingleRoiProbeSystem
from mbs.evaluation.utils.evaluation_helpers import compute_metrics, compute_rsa_cka


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    device: str = "cuda"
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    amp: bool = True
    grad_accum_steps: int = 1
    eval_every: int = 1
    lr_schedule: str = "cosine"  # {"constant", "linear", "cosine"}

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    num_epochs: int,
):
    schedule = schedule.lower()

    if schedule == "constant":
        return None

    elif schedule == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 - epoch / float(max(1, num_epochs)),
        )

    elif schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=0.0,
        )

    else:
        raise ValueError(
            f"Unknown lr_schedule='{schedule}'. "
            f"Choose from ['constant', 'linear', 'cosine']."
        )

@torch.no_grad()
def evaluate(
    model: SingleRoiProbeSystem,
    *,
    loaders: Dict[Tuple[str, str], torch.utils.data.DataLoader],
    noise_ceilings: Dict[Tuple[str, str], np.ndarray],
    subjects: List[str],
    split: str,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()

    mse_all = []
    r_all = []
    rnc_all = []
    
    all_subject_results = []

    for s in subjects:
        loader = loaders[(split, s)]
        N = int(noise_ceilings[(split, s)].shape[0])

        rp = RunningPearson(dim=N, device=device)
        mse_sum = 0.0
        n_sum = 0
        
        y_hat_list = []
        y_list = []
        
        # Check wheter `y` has non-zero second dimension in the first batch
        first_batch = next(iter(loader))
        if first_batch["y"].shape[1] == 0:
            print(f"Skipping subject {s} in {split} split due to zero-dimensional target.")
            all_subject_results.append({
                'subject': s,
            })
            continue

        for batch in loader:
            feats = batch["feats"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                y_hat = model(feats, subject=s)
                loss = F.mse_loss(y_hat, y, reduction="mean")

            bs = int(y.shape[0])
            mse_sum += float(loss.item()) * bs
            n_sum += bs
            rp.update(y_hat, y)
            
            y_hat_list.append(y_hat.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())

        mse = mse_sum / max(1, n_sum)
        r = rp.corr()  # (N,)
        r_mean = float(r.mean().item())

        nc = torch.from_numpy(noise_ceilings[(split, s)]).to(device)
        rnc_mean = float((r / (nc + 1e-8)).mean().item())
        

        mse_all.append(mse)
        r_all.append(r_mean)
        rnc_all.append(rnc_mean)
        
        y_hat_list = np.concatenate(y_hat_list, axis=0)
        y_list = np.concatenate(y_list, axis=0)
        
        r2, evs, mae, mse, pearsonr, app_evs, pearsonr_nc, app_evs_nc = compute_metrics(
            y_true=y_list,
            y_pred=y_hat_list,
            noise_ceiling=noise_ceilings[(split, s)],
            verbose=True,
        )
        
        # Create a dummy X_test for RSA/CKA computation
        X_test_dummy = np.random.randn(y_list.shape[0], 128)  # Dummy
        
        ( rsa_c_train, rsa_ve_train, rsa_c_test, rsa_ve_test,
            cka_c_train, cka_ve_train, cka_c_test, cka_ve_test 
        ) = compute_rsa_cka(X_test=X_test_dummy, y_test=y_list, y_test_pred=y_hat_list, X_train=None, y_train=None, y_train_pred=None, verbose=False, use_gpu=(device.type=="cuda"))
        
        # Remove dummy results
        rsa_c_test = float('nan')
        cka_c_test = float('nan')
        
        subject_results = {
            # 'benchmark_name': benchmark_name,
            # 'model_id': model_name,
            # 'layer_name': layer_name,
            # 'layer_position': layer_pos,
            # 'layer_position_normalized': layer_pos_norm,
            'subject': s,
            # 'roi': roi,
            'cv_score': None,
            'r2': float(r2),
            'evs': float(evs),
            'mae': float(mae),
            'mse': float(mse),
            'pearsonr': float(pearsonr),
            'approx_exp_var': float(app_evs),
            'noise_ceiling': float(noise_ceilings[(split, s)].mean()),
            'pearsonr_nc': float(pearsonr_nc),
            'approx_exp_var_nc': float(app_evs_nc),
            'rsa_c_train': None,
            'rsa_ve_train': None,
            'cka_c_train': None,
            'cka_ve_train': None,
            'rsa_c_test': None,
            'rsa_ve_test': float(rsa_ve_test),
            'cka_c_test': None,
            'cka_ve_test': float(cka_ve_test),
        }
        all_subject_results.append(subject_results)
        

    return {
        f"{split}_mse_mean_over_subjects": float(np.mean(mse_all)) if mse_all else float("nan"),
        f"{split}_pearson_mean_over_subjects": float(np.mean(r_all)) if r_all else float("nan"),
        f"{split}_pearson_nc_mean_over_subjects": float(np.mean(rnc_all)) if rnc_all else float("nan"),
        f"{split}_all_subject_results": all_subject_results,
    }


def train_single_roi_single_layer(
    *,
    neural_h5_path: Path,
    roi: str,
    layer_name: str,
    features_folder: Optional[Path],
    features_file_path: Optional[Path],
    train_split: str,
    val_split: str,
    probe_cfg: ProbeConfig,
    train_cfg: TrainConfig,
    subjects_allowlist: Optional[List[str]] = None,
    seed: int = 42,
    data_pct: Optional[float] = None,
) -> SingleRoiProbeSystem:
    """
    End-to-end trainer.

    This implementation is deliberately simple:
    - one loader per subject (no padding)
    - iterate subjects sequentially each epoch
    - shared trunk, subject heads
    """
    set_seed(seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    # 1) Load features for exactly one layer
    layer_feats, feature_id_map = load_layer_features(
        layer_name=layer_name,
        features_folder=features_folder,
        features_file_path=features_file_path,
    )

    # 2) Verify in_dim matches stored features
    inferred_in = infer_in_dim_from_layer_feats(layer_feats)
    if probe_cfg.in_dim != inferred_in:
        raise ValueError(
            f"probe_cfg.in_dim={probe_cfg.in_dim} does not match inferred_in_dim={inferred_in} from features."
        )

    # 3) Build subject loaders (train/val) + noise ceilings
    subjects, loaders, noise_ceilings = build_subject_loaders(
        neural_h5_path=neural_h5_path,
        roi=roi,
        layer_feats=layer_feats,
        feature_id_map=feature_id_map,
        train_split=train_split,
        val_split=val_split,
        batch_size=train_cfg.batch_size,
        subjects_allowlist=subjects_allowlist,
        seed=seed,
        data_pct=data_pct,
    )

    # 4) Build model with subject-specific output heads
    neuroid_dims = {s: int(noise_ceilings[(train_split, s)].shape[0]) for s in subjects}
    model = SingleRoiProbeSystem(cfg=probe_cfg, subjects=subjects, neuroid_dims=neuroid_dims).to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f} M")
    
    total_input_dim = int(np.prod(layer_feats.shape[1:]))
    total_neuroid_dims = sum(neuroid_dims.values())
    linear_probe_dim = (total_input_dim + 1) * total_neuroid_dims
    print(f"Equivalent linear probe parameters: {linear_probe_dim/1e6:.2f} M")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    scheduler = build_lr_scheduler(
        optimizer=opt,
        schedule=train_cfg.lr_schedule,
        num_epochs=train_cfg.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(train_cfg.amp and device.type == "cuda")
    )

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()

        subj_order = subjects[:]
        random.shuffle(subj_order)

        opt.zero_grad(set_to_none=True)
        step = 0

        running_sum = 0.0
        running_n = 0

        for s in subj_order:
            loader = loaders[(train_split, s)]
            for batch in loader:
                feats = batch["feats"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(train_cfg.amp and device.type == "cuda")):
                    y_hat = model(feats, subject=s)
                    loss = F.mse_loss(y_hat, y, reduction="mean") / train_cfg.grad_accum_steps
                    # 1 - cosine similarity loss
                    # loss = (1.0 - F.cosine_similarity(y_hat, y, dim=-1).mean()) / train_cfg.grad_accum_steps

                scaler.scale(loss).backward()

                bs = int(y.shape[0])
                running_sum += float(loss.item()) * bs * train_cfg.grad_accum_steps
                running_n += bs

                step += 1
                if step % train_cfg.grad_accum_steps == 0:
                    if train_cfg.grad_clip is not None:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

        train_mse = running_sum / max(1, running_n)

        if epoch % train_cfg.eval_every == 0:
            val_metrics = evaluate(
                model,
                loaders=loaders,
                noise_ceilings=noise_ceilings,
                subjects=subjects,
                split=val_split,
                device=device,
                # amp=train_cfg.amp,
                amp=False,  # Disable AMP during eval for stability
            )
            print(
                f"[epoch {epoch:03d}] "
                f"train_mse={train_mse:.6f} "
                f"{val_split}_mse={val_metrics[f'{val_split}_mse_mean_over_subjects']:.6f} "
                f"{val_split}_r={val_metrics[f'{val_split}_pearson_mean_over_subjects']:.4f} "
                f"{val_split}_rNC={val_metrics[f'{val_split}_pearson_nc_mean_over_subjects']:.4f}"
            )
        else:
            print(f"[epoch {epoch:03d}] train_mse={train_mse:.6f}")
            
        
        if scheduler is not None:
            scheduler.step()

    return model, loaders, noise_ceilings
