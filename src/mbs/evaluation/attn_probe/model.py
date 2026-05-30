"""
Model definition for the single ROI + single layer attention probe.

- Shared trunk across subjects
- Subject-specific heads (variable output dim per subject)
"""

from dataclasses import dataclass
from typing import Dict, List, Literal

import torch
import torch.nn as nn


@dataclass
class ProbeConfig:
    in_dim: int
    d_model: int = 384
    nhead: int = 16
    dim_ff: int = 5
    dropout: float = 0.1

    token_encoder_layers: int = 0

    num_latents: int = 16
    cross_attn_layers: int = 1
    query_self_attn: bool = False

    # pos_mode: Literal["none", "mlp_coords"] = "none"
    pos_mode: Literal["none", "mlp_coords", "sin", "learned"] = "none"
    coord_dim: int = 2
    max_tokens: int = 4096
    sin_base: float = 10000.0 

    head_type: Literal["linear", "lowrank", "shallow_mlp"] = "linear"
    head_rank: int = 256
    head_mlp_hidden_dim: int = 256
    head_mlp_dropout: float = 0.0


class TokenAdapter(nn.Module):
    """
    Convert feature tensors into tokens.

    Inputs supported:
      feats.ndim == 2 : (B, C)       -> tokens (B, 1, C)
      feats.ndim == 3 : (B, N, C)    -> tokens (B, N, C)
      feats.ndim == 4 : (B, C, H, W) -> tokens (B, HW, C)

    For BCHW and pos_mode="mlp_coords", also returns (x,y) coords normalized to [0,1].
    """
    def __init__(self, pos_mode: Literal["none", "mlp_coords"], coord_dim: int):
        super().__init__()
        self.pos_mode = pos_mode
        self.coord_dim = coord_dim

    def forward(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if feats.ndim == 2:
            return feats[:, None, :], None
        if feats.ndim == 3:
            # For transformer tokens (B, N, C), coords are optional and only used for mlp_coords
            return feats, coords
        if feats.ndim == 4:
            B, C, H, W = feats.shape
            tokens = feats.flatten(2).transpose(1, 2)  # (B, HW, C)

            coords = None
            if self.pos_mode == "mlp_coords":
                if self.coord_dim != 2:
                    raise ValueError("For BCHW coords we provide (x,y) so coord_dim must be 2.")
                yy, xx = torch.meshgrid(
                    torch.linspace(0, 1, H, device=feats.device),
                    torch.linspace(0, 1, W, device=feats.device),
                    indexing="ij",
                )
                c = torch.stack([xx, yy], dim=-1).reshape(H * W, 2)
                coords = c[None, :, :].expand(B, H * W, 2).contiguous()

            return tokens, coords

        raise ValueError(f"Unsupported feats.ndim={feats.ndim}")


class MLPPosEnc(nn.Module):
    def __init__(self, coord_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
    
class SinusoidalPosEnc(nn.Module):
    """
    Returns (1, N, d_model) sinusoidal positional encodings for token indices [0..N-1].
    """
    def __init__(self, d_model: int, max_tokens: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.base = base

        # Precompute up to max_tokens as a non-trainable buffer
        pe = torch.zeros(max_tokens, d_model)
        position = torch.arange(0, max_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(base)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, N: int, device: torch.device) -> torch.Tensor:
        if N > self.max_tokens:
            raise ValueError(f"N={N} exceeds max_tokens={self.max_tokens}. Increase cfg.max_tokens.")
        return self.pe[:N, :].to(device)[None, :, :]  # (1, N, d_model)


class LearnedPosEnc(nn.Module):
    """
    Learnable absolute positional embedding: (max_tokens, d_model) -> (1, N, d_model)
    """
    def __init__(self, d_model: int, max_tokens: int = 4096):
        super().__init__()
        self.pe = nn.Embedding(max_tokens, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, N: int, device: torch.device) -> torch.Tensor:
        if N > self.pe.num_embeddings:
            raise ValueError(f"N={N} exceeds max_tokens={self.pe.num_embeddings}. Increase cfg.max_tokens.")
        idx = torch.arange(N, device=device)
        return self.pe(idx)[None, :, :]  # (1, N, d_model)



class TokenSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        if dim_ff > 0:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.ffn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.ln(x)
        x2, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + self.drop(x2)
        x = x + self.ffn(x)
        return x


class QueryCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, query_self_attn: bool):
        super().__init__()
        self.query_self_attn = query_self_attn
        if query_self_attn:
            self.q_ln = nn.LayerNorm(d_model)
            self.q_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.q_drop = nn.Dropout(dropout)

        self.ln = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.cross = nn.MultiheadAttention(d_model, nhead, kdim=384, vdim=384, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        if dim_ff > 0:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.ffn = nn.Identity()

    def forward(self, q: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if self.query_self_attn:
            qn = self.q_ln(q)
            q2, _ = self.q_attn(qn, qn, qn, need_weights=False)
            q = q + self.q_drop(q2)

        qn = self.ln(q)
        q2, _ = self.cross(qn, tokens, tokens, need_weights=False)
        q = q + self.drop(q2)

        q = q + self.ffn(q)
        return q


class LatentAttentionTrunk(nn.Module):
    """
    Shared trunk.

    Produces a fixed-length representation:
      feats -> tokens -> latents -> flatten => (B, num_latents * d_model)
    """
    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg

        self.adapter = TokenAdapter(cfg.pos_mode, cfg.coord_dim)

        self.token_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.token_ln = nn.LayerNorm(cfg.d_model)
        
        # self.token_proj = nn.Identity()
        # self.token_ln = nn.LayerNorm(cfg.in_dim)

        self.pos_mlp = MLPPosEnc(cfg.coord_dim, cfg.d_model) if cfg.pos_mode == "mlp_coords" else None
        self.pos_sin = SinusoidalPosEnc(cfg.d_model, cfg.max_tokens, cfg.sin_base) if cfg.pos_mode == "sin" else None
        self.pos_learned = LearnedPosEnc(cfg.d_model, cfg.max_tokens) if cfg.pos_mode == "learned" else None

        self.token_encoder = nn.ModuleList([
            TokenSelfAttentionBlock(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.token_encoder_layers)
        ])

        self.latents = None
        self.decoder = nn.ModuleList()
        self.shallow_mlp = None

        if cfg.head_type == "shallow_mlp":
            self.shallow_mlp = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.head_mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.head_mlp_dropout),
                nn.Linear(cfg.head_mlp_hidden_dim, cfg.num_latents * cfg.d_model),
            )
        elif cfg.head_type == "lowrank":
            pass
        else:
            self.latents = nn.Embedding(cfg.num_latents, cfg.d_model)
            self.decoder = nn.ModuleList([
                QueryCrossAttentionBlock(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout, cfg.query_self_attn)
                for _ in range(cfg.cross_attn_layers)
            ])
        self.final_ln = nn.LayerNorm(cfg.d_model)

        self._reset_params()

    def _reset_params(self):
        if self.latents is not None:
            nn.init.normal_(self.latents.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        tokens, coords = self.adapter(feats, coords=coords)
        x = self.token_ln(self.token_proj(tokens))

        if self.pos_mlp is not None:
            assert coords is not None, "pos_mode=mlp_coords for BNC requires coords (B, N, coord_dim)."
            x = x + self.pos_mlp(coords)
        elif self.pos_sin is not None:
            N = x.shape[1]
            x = x + self.pos_sin(N, x.device)
        elif self.pos_learned is not None:
            N = x.shape[1]
            x = x + self.pos_learned(N, x.device)

        for blk in self.token_encoder:
            x = blk(x)

        if self.cfg.head_type == "lowrank":
            return x.mean(dim=1)

        if self.shallow_mlp is not None:
            pooled = x.mean(dim=1)
            return self.shallow_mlp(pooled)

        B = x.shape[0]
        q = self.latents.weight[None, :, :].expand(B, self.cfg.num_latents, self.cfg.d_model)

        for blk in self.decoder:
            q = blk(q, x)

        q = self.final_ln(q)
        return q.reshape(B, -1)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LowRankHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class SubjectHeadBank(nn.Module):
    """
    One head per subject (output dim differs across subjects).
    """
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict()

    def add(self, subject: str, head: nn.Module):
        self.heads[subject] = head

    def forward(self, subject: str, x: torch.Tensor) -> torch.Tensor:
        return self.heads[subject](x)


class SingleRoiProbeSystem(nn.Module):
    """
    Full system:
      y_hat = head_subject(trunk(feats))

    - trunk is shared across subjects
    - head is subject-specific to handle variable voxel/unit count
    """
    def __init__(self, cfg: ProbeConfig, subjects: List[str], neuroid_dims: Dict[str, int]):
        super().__init__()
        self.cfg = cfg
        self.trunk = LatentAttentionTrunk(cfg)

        head_in = cfg.d_model if cfg.head_type == "lowrank" else cfg.num_latents * cfg.d_model
        self.heads = SubjectHeadBank()

        for s in subjects:
            N = int(neuroid_dims[s])
            if cfg.head_type == "linear":
                head = LinearHead(head_in, N)
            elif cfg.head_type == "lowrank":
                head = LowRankHead(head_in, N, cfg.head_rank)
            elif cfg.head_type == "shallow_mlp":
                head = LinearHead(head_in, N)
            else:
                raise ValueError(f"Unsupported head_type={cfg.head_type}")
            self.heads.add(s, head)

    def forward(self, feats: torch.Tensor, subject: str, coords: torch.Tensor | None = None) -> torch.Tensor:
        z = self.trunk(feats, coords=coords)
        return self.heads(subject, z)
