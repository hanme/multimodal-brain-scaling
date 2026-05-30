#!/usr/bin/env python
"""
CLI wrapper around the simplified pipeline.
"""

import argparse
from pathlib import Path
import json

from .model import ProbeConfig
from .engine import TrainConfig, train_single_roi_single_layer


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--neural_h5_path", type=Path, required=True)
    parser.add_argument("--roi", type=str, required=True)
    # parser.add_argument("--layer_name", type=str, required=True)
    parser.add_argument("--layer_commitments_file", type=Path, required=True)

    # Features input (choose one)
    parser.add_argument("--features_folder", type=Path, default=None)
    parser.add_argument("--features_file_path", type=Path, default=None)

    # Splits
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="test")

    # Subjects, comma-separated list
    parser.add_argument("--subjects", type=str, default=None)

    # Probe config
    parser.add_argument("--in_dim", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_encoder_layers", type=int, default=0)
    parser.add_argument("--num_latents", type=int, default=64)
    parser.add_argument("--cross_attn_layers", type=int, default=2)
    parser.add_argument("--query_self_attn", action="store_true")
    parser.add_argument("--pos_mode", type=str, choices=["none", "mlp_coords", "sin", "learned"], default="none")
    parser.add_argument("--head_type", type=str, choices=["linear", "lowrank", "shallow_mlp"], default="linear")
    parser.add_argument("--head_rank", type=int, default=256)
    parser.add_argument("--head_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--head_mlp_dropout", type=float, default=0.0)

    # Train config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_pct", type=float, default=None)

    args = parser.parse_args()
    
    return args



def main(args: argparse.Namespace):
    
    if (args.features_folder is None) == (args.features_file_path is None):
        raise ValueError("Provide exactly one of --features_folder or --features_file_path")
    
    
    ############ Load layer commitments ############
    layer_commitments_file = Path(args.layer_commitments_file)
    assert layer_commitments_file.exists(), f"Layer commitments file does not exist: {layer_commitments_file}"

    with open(layer_commitments_file, 'r') as f:
        layer_commitments = json.load(f)
        
    layer_commitments_model_id = args.model_id
    assert layer_commitments_model_id in layer_commitments, f"Model ID {layer_commitments_model_id} not found in layer commitments."
    model_layer_commitments = layer_commitments[layer_commitments_model_id]
    benchmark_name = args.neural_h5_path.stem
    model_benchmark__roi_layer = model_layer_commitments[benchmark_name][args.roi]["layer_name"]
    #################################################
    
    args.subjects = args.subjects.split(",") if args.subjects is not None else None
    

    probe_cfg = ProbeConfig(
        in_dim=args.in_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        token_encoder_layers=args.token_encoder_layers,
        num_latents=args.num_latents,
        cross_attn_layers=args.cross_attn_layers,
        query_self_attn=args.query_self_attn,
        pos_mode=args.pos_mode,
        head_type=args.head_type,
        head_rank=args.head_rank,
        head_mlp_hidden_dim=args.head_mlp_hidden_dim,
        head_mlp_dropout=args.head_mlp_dropout,
    )

    train_cfg = TrainConfig(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=(not args.no_amp),
        grad_accum_steps=args.grad_accum_steps,
        eval_every=args.eval_every,
        lr_schedule=args.lr_schedule,
    )

    _ = train_single_roi_single_layer(
        neural_h5_path=args.neural_h5_path,
        roi=args.roi,
        layer_name=model_benchmark__roi_layer,
        features_folder=args.features_folder,
        features_file_path=args.features_file_path,
        train_split=args.train_split,
        val_split=args.val_split,
        probe_cfg=probe_cfg,
        train_cfg=train_cfg,
        subjects_allowlist=args.subjects,
        seed=args.seed,
        data_pct=args.data_pct,
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    main(args)
