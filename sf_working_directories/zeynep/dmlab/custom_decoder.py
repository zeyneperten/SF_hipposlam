import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------
# Fixed positional bases
# ---------------------------


def fixed_smoothed_time_basis(T: int, R: int, normalize: bool = True, device=None, dtype=None) -> Tensor:
    """
    R-smoothed canonical time basis: returns (T, T).
    Row t is a boxcar over indices [t, t+R-1] clipped to T.

    With your convention (newest token at index 0), position t corresponds
    to "t steps into the past". This basis gives each token a smoothed
    indicator of nearby temporal positions.

    If normalize=True, each row sums to 1.
    """
    R = max(int(R), 1)
    P = torch.zeros((T, T), device=device, dtype=dtype)
    for t in range(T):
        j0 = t
        j1 = min(T, t + R)
        P[t, j0:j1] = 1.0
    if normalize:
        P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
    return P


def fixed_sinusoidal_positions(T: int, d: int, device=None, dtype=None) -> Tensor:
    """
    Standard sinusoidal PE with dimension d (must be even for sin/cos pairs).
    Returns (T, d).
    """
    if d % 2 != 0:
        raise ValueError(f"Sinusoidal PE requires even d, got {d}")
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
    i = torch.arange(d // 2, device=device, dtype=dtype).unsqueeze(0)  # (1, d/2)
    div = torch.exp(-math.log(10000.0) * (2 * i) / d)  # (1, d/2)
    angles = pos * div  # (T, d/2)
    pe = torch.zeros((T, d), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def fixed_fourier_basis(T: int, d_p: int, max_freq: float = 1.0, device=None, dtype=None) -> Tensor:
    """
    Canonical Fourier-like bases over discrete positions 0..T-1.
    Returns (T, d_p) with sin/cos pairs. d_p must be even.

    This is a simple canonical basis (sin/cos) at multiple frequencies. You can
    change spacing (linear/log) without affecting interface.
    """
    if d_p % 2 != 0:
        raise ValueError(f"Fourier basis requires even d_p, got {d_p}")

    t = torch.arange(T, device=device, dtype=dtype) / float(T)  # (T,)
    t = t.unsqueeze(1)  # (T,1)

    n_freq = d_p // 2
    freqs = torch.linspace(1.0, float(n_freq), n_freq, device=device, dtype=dtype)
    freqs = freqs / float(n_freq) * max_freq  # (n_freq,)

    angles = 2.0 * math.pi * t * freqs.unsqueeze(0)  # (T, n_freq)
    pe = torch.zeros((T, d_p), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


# ---------------------------
# RoPE (rotary) helpers
# ---------------------------


def rope_build_cache(T: int, d: int, base: float = 100.0, device=None, dtype=None):
    """
    Build cos/sin caches for RoPE. Returns cos, sin of shape (T, d/2).
    d must be even.
    """
    if d % 2 != 0:
        raise ValueError(f"RoPE requires even d, got {d}")
    half = d // 2
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
    i = torch.arange(half, device=device, dtype=dtype).unsqueeze(0)  # (1,half)
    inv_freq = 1.0 / (base ** (i / half))  # (1,half)
    angles = pos * inv_freq  # (T,half)
    return torch.cos(angles), torch.sin(angles)


def rope_apply_btd(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply RoPE to x in shape (B, T, d). cos/sin are (T, d/2).
    """
    B, T, d = x.shape
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos.unsqueeze(0)  # (1,T,half)
    sin_ = sin.unsqueeze(0)  # (1,T,half)
    out1 = x1 * cos_ - x2 * sin_
    out2 = x1 * sin_ + x2 * cos_
    return torch.cat([out1, out2], dim=-1)


# ---------------------------
# Transformer components
# ---------------------------


class CausalSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, rope: bool = False, rope_base: float = 10000.0, dropout: float = 0.0
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.rope = rope
        self.rope_base = rope_base
        if self.rope and (self.d_head % 2 != 0):
            raise ValueError(
                f"RoPE requires even per-head dim. Got d_model={d_model}, n_heads={n_heads} => d_head={self.d_head}."
            )

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("_mask", torch.empty(0), persistent=False)
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)

    def _get_causal_mask(self, T: int, device) -> Tensor:
        if self._mask.numel() == 0 or self._mask.size(0) < T:
            # self._mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)) # lower triangular, works for oldest to newest ordering
            self._mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool))
        return self._mask[:T, :T]

    def _get_rope_cache(self, T: int, device, dtype):
        if self._rope_cos.numel() == 0 or self._rope_cos.size(0) < T:
            cos, sin = rope_build_cache(T, self.d_head, base=self.rope_base, device=device, dtype=dtype)
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:T], self._rope_sin[:T]

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B,T,D)

        # (B, nH, T, dH)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.rope:
            cos, sin = self._get_rope_cache(T, x.device, x.dtype)  # (T, dH/2)
            # Apply RoPE per head by flattening head dimension
            q_ = q.reshape(B * self.n_heads, T, self.d_head)
            k_ = k.reshape(B * self.n_heads, T, self.d_head)
            q_ = rope_apply_btd(q_, cos, sin)
            k_ = rope_apply_btd(k_, cos, sin)
            q = q_.view(B, self.n_heads, T, self.d_head)
            k = k_.view(B, self.n_heads, T, self.d_head)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))  # (B,nH,T,T)
        mask = self._get_causal_mask(T, x.device)  # (T,T)
        att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = F.softmax(att, dim=-1)
        w = self.dropout(w)

        y = w @ v  # (B,nH,T,dH)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        y = self.proj(y)
        return self.dropout(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rope: bool, rope_base: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope=rope, rope_base=rope_base, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------
# Main module: 2-layer transformer on shift-register output
# ---------------------------

PosMode = Literal["none", "sin_add", "learned_add", "concat_sin", "concat_fourier", "concat_smoothed", "rope"]


@dataclass
class SRTransformerConfig:
    T: int = 64
    d_f: int = 16
    bypass_size: int = 0

    d_model: int = 16
    n_heads: int = 2
    d_ff: Optional[int] = None
    dropout: float = 0.0

    pos_mode: PosMode = "rope"

    # concat_* only
    d_p: int = 16
    fourier_max_freq: float = 1.0

    # smoothed time basis only (concat_smoothed)
    time_basis_R: int = 8
    time_basis_normalize: bool = True

    # rope only
    rope_base: float = 10000.0

    # output
    out_dim: int = 128
    include_bypass_in_output: bool = True

    # readout
    readout_mode: Literal["last", "weighted_sum"] = "last"
    readout_attn_hidden: int = 0  # 0 => linear score; >0 => 2-layer scorer with hidden dim


class TwoLayerCausalTransformerFromShiftRegisterNoPacked(nn.Module):
    """
    Expects input of shape (B, T*d_f + bypass_size), where the T tokens are ordered
    newest-to-oldest in the flattened layout (newest token at index 0).

    The module reshapes the first T*d_f dims into (B, T, d_f) and runs a 2-layer
    causal transformer. Readout uses token index 0 (newest).
    """

    def __init__(self, cfg: SRTransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.d_ff is None:
            cfg.d_ff = 4 * cfg.d_model

        self.pos_mode = cfg.pos_mode.lower()
        if self.pos_mode == "rope":
            d_head = cfg.d_model // cfg.n_heads
            if cfg.d_model % cfg.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            if d_head % 2 != 0:
                raise ValueError(
                    f"RoPE requires even head dim. Got d_model={cfg.d_model}, n_heads={cfg.n_heads} => d_head={d_head}."
                )

        # buffers for fixed PEs
        self.register_buffer("_pe_add", torch.empty(0), persistent=False)
        self.register_buffer("_pe_concat", torch.empty(0), persistent=False)

        # learned absolute pos embedding (additive)
        if self.pos_mode == "learned_add":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.T, cfg.d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        # input projection
        if self.pos_mode in ("concat_sin", "concat_fourier"):
            in_dim = cfg.d_f + cfg.d_p
            self.in_proj = nn.Linear(in_dim, cfg.d_model, bias=True)
        elif self.pos_mode == "concat_smoothed":
            in_dim = cfg.d_f + cfg.T  # smoothed basis is (T,T)
            self.in_proj = nn.Linear(in_dim, cfg.d_model, bias=True)
        else:
            self.in_proj = nn.Identity() if cfg.d_f == cfg.d_model else nn.Linear(cfg.d_f, cfg.d_model, bias=True)

        rope_flag = self.pos_mode == "rope"
        self.block1 = TransformerBlock(
            cfg.d_model, cfg.n_heads, cfg.d_ff, rope=rope_flag, rope_base=cfg.rope_base, dropout=cfg.dropout
        )
        self.block2 = TransformerBlock(
            cfg.d_model, cfg.n_heads, cfg.d_ff, rope=rope_flag, rope_base=cfg.rope_base, dropout=cfg.dropout
        )

        self.out = nn.Linear(cfg.d_model, cfg.out_dim, bias=True)

        # Optional token pooling for readout
        self.readout_mode = getattr(cfg, "readout_mode", "last")
        self.readout_attn_hidden = int(getattr(cfg, "readout_attn_hidden", 0))

        if self.readout_mode == "weighted_sum":
            # score each token -> softmax over time -> weighted sum
            if self.readout_attn_hidden and self.readout_attn_hidden > 0:
                self.readout_score = nn.Sequential(
                    nn.Linear(cfg.d_model, self.readout_attn_hidden),
                    nn.Tanh(),
                    nn.Linear(self.readout_attn_hidden, 1),
                )
            else:
                self.readout_score = nn.Linear(cfg.d_model, 1, bias=True)
        else:
            self.readout_score = None

    def _ensure_pe_add(self, device, dtype):
        if self.pos_mode == "sin_add":
            if self._pe_add.numel() == 0:
                self._pe_add = fixed_sinusoidal_positions(self.cfg.T, self.cfg.d_model, device=device, dtype=dtype)

    def _ensure_pe_concat(self, device, dtype):
        if self.pos_mode == "concat_sin":
            if self._pe_concat.numel() == 0:
                self._pe_concat = fixed_sinusoidal_positions(self.cfg.T, self.cfg.d_p, device=device, dtype=dtype)

        elif self.pos_mode == "concat_fourier":
            if self._pe_concat.numel() == 0:
                self._pe_concat = fixed_fourier_basis(
                    self.cfg.T, self.cfg.d_p, max_freq=self.cfg.fourier_max_freq, device=device, dtype=dtype
                )

        elif self.pos_mode == "concat_smoothed":
            if self._pe_concat.numel() == 0:
                self._pe_concat = fixed_smoothed_time_basis(
                    self.cfg.T,
                    self.cfg.time_basis_R,
                    normalize=self.cfg.time_basis_normalize,
                    device=device,
                    dtype=dtype,
                )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T*d_f + bypass_size)
        returns:
          - if include_bypass_in_output: (B, out_dim + bypass_size)
          - else: (B, out_dim)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected (B, T*d_f+bypass), got {x.shape}")

        B, D = x.shape
        expected_core = self.cfg.T * self.cfg.d_f
        expected_total = expected_core + self.cfg.bypass_size
        if D != expected_total:
            raise ValueError(f"Expected input dim {expected_total} (=T*d_f+bypass), got {D}")

        core_flat = x[:, :expected_core]  # (B, T*d_f)
        bypass = x[:, expected_core:] if self.cfg.bypass_size > 0 else None

        # tokens: newest token at index 0
        tokens = core_flat.view(B, self.cfg.T, self.cfg.d_f)  # (B,T,d_f)

        device, dtype = tokens.device, tokens.dtype

        # Positional encoding integration
        if self.pos_mode == "none":
            h = self.in_proj(tokens)

        elif self.pos_mode == "sin_add":
            h = self.in_proj(tokens)
            self._ensure_pe_add(device, dtype)
            h = h + self._pe_add.unsqueeze(0)  # (1,T,d_model)

        elif self.pos_mode == "learned_add":
            h = self.in_proj(tokens)
            h = h + self.pos_emb.unsqueeze(0)  # (1,T,d_model)

        elif self.pos_mode in ("concat_sin", "concat_fourier", "concat_smoothed"):
            self._ensure_pe_concat(device, dtype)
            pe = self._pe_concat.unsqueeze(0).expand(B, -1, -1)  # (B,T,d_p) or (B,T,T)
            h = torch.cat([tokens, pe], dim=-1)  # (B,T,d_f+d_p) or (B,T,d_f+T)
            h = self.in_proj(h)  # (B,T,d_model)

        elif self.pos_mode == "rope":
            # RoPE is applied inside attention to Q/K, so do not add/concat here.
            h = self.in_proj(tokens)

        else:
            raise ValueError(f"Unknown pos_mode: {self.pos_mode}")

        # 2-layer causal transformer
        h = self.block1(h)
        h = self.block2(h)

        # Readout: newest token at index 0
        # Readout
        if self.readout_mode == "last":
            # newest token at index 0
            readout = h[:, 0, :]  # (B, d_model)

        elif self.readout_mode == "weighted_sum":
            # learned content-dependent pooling over tokens
            # scores: (B, T, 1) -> weights: (B, T, 1)
            scores = self.readout_score(h)
            weights = torch.softmax(scores, dim=1)
            readout = (weights * h).sum(dim=1)  # (B, d_model)

        else:
            raise ValueError(f"Unknown readout_mode: {self.readout_mode}")

        y = self.out(readout)  # (B, out_dim)

        if self.cfg.include_bypass_in_output and bypass is not None:
            return torch.cat([y, bypass], dim=1)
        return y


from typing import Optional

import torch
import torch.nn as nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule
from sample_factory.utils.typing import Config

# import your module (adjust import path)
# from your_module import SRTransformerConfig, TwoLayerCausalTransformerFromShiftRegisterNoPacked


class ShiftRegisterTransformerDecoder(ModelModule):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)

        # --- required parse sizes ---
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)

        T = self.R + self.L - 1
        d_f = Hippo_n_feature

        # bypass size can be explicit or inferred
        bypass_size: Optional[int] = getattr(cfg, "decoder_sr_bypass_size", None)
        if bypass_size is None:
            core = T * Hippo_n_feature
            if decoder_input_size < core:
                raise ValueError(f"decoder_input_size={decoder_input_size} < T*Hippo_n_feature={core}")
            bypass_size = decoder_input_size - core
        else:
            bypass_size = int(bypass_size)

        include_bypass = bool(getattr(cfg, "decoder_sr_include_bypass_in_output", True))

        # --- transformer hyperparams ---
        d_model = int(getattr(cfg, "decoder_attn_d_model", d_f))
        n_heads = int(getattr(cfg, "decoder_attn_n_heads", 1))
        d_ff = getattr(cfg, "decoder_attn_d_ff", None)
        d_ff = int(d_ff) if d_ff is not None else 4 * d_model
        dropout = float(getattr(cfg, "decoder_attn_dropout", 0.0))

        pos_mode = str(getattr(cfg, "decoder_attn_pos_mode", "rope"))

        # --- pos-encoding extras ---
        d_p = int(getattr(cfg, "decoder_attn_d_p", d_f))
        fourier_max_freq = float(getattr(cfg, "decoder_attn_fourier_max_freq", 1.0))
        rope_base = float(getattr(cfg, "decoder_attn_rope_base", 10000.0))

        out_dim = int(getattr(cfg, "decoder_attn_out_dim", 128))

        time_basis_normalize = bool(getattr(cfg, "decoder_attn_time_basis_normalize", True))

        readout_mode = str(getattr(cfg, "decoder_attn_readout_mode", "last"))
        readout_attn_hidden = int(getattr(cfg, "decoder_attn_readout_attn_hidden", 0))

        sr_cfg = SRTransformerConfig(
            T=T,
            d_f=d_f,
            bypass_size=bypass_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_mode=pos_mode,
            d_p=d_p,
            fourier_max_freq=fourier_max_freq,
            rope_base=rope_base,
            out_dim=out_dim,
            include_bypass_in_output=include_bypass,
            time_basis_R=self.R,
            time_basis_normalize=time_basis_normalize,
            readout_mode=readout_mode,
            readout_attn_hidden=readout_attn_hidden,
        )

        self.decoder = TwoLayerCausalTransformerFromShiftRegisterNoPacked(sr_cfg)

        # output size (includes bypass if enabled)
        self.decoder_out_size = out_dim + (bypass_size if include_bypass else 0)

        # Optional: script for speed parity with MlpDecoder.
        # TorchScript can be finicky with dataclasses and Literal;
        # I recommend leaving it unscripted unless you specifically need it.
        # self.decoder = torch.jit.script(self.decoder)

    def forward(self, core_output: torch.Tensor) -> torch.Tensor:
        return self.decoder(core_output)

    def get_out_size(self) -> int:
        return self.decoder_out_size


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    cfg = SRTransformerConfig(
        T=64,
        d_f=16,
        bypass_size=0,
        d_model=16,
        n_heads=2,
        d_ff=64,
        pos_mode="rope",  # try: "none", "sin_add", "learned_add", "concat_sin", "concat_fourier", "rope"
        out_dim=128,
        include_bypass_in_output=True,
    )
    m = TwoLayerCausalTransformerFromShiftRegisterNoPacked(cfg)
    x = torch.zeros(32, 64 * 16 + 0)
    y = m(x)
    print(y.shape)  # (32, 128)
