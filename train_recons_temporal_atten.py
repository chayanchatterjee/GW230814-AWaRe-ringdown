"""
GW230814 Reconstruction with Temporal Self-Attention + Calibrated Uncertainty
=============================================================================

Key changes from the original Restormer1D:

1. ARCHITECTURE: Replaces channel-wise MDTA with standard temporal 
   self-attention (T×T attention maps). This gives a [B, heads, T, T]
   attention matrix that directly tells you "how much does time step i
   attend to time step j" — the standard way to do Fisher information
   localization via attention maps.

2. ROTARY POSITIONAL ENCODING (RoPE): Encodes relative temporal
   position directly into Q and K via rotation, so attention scores
   naturally decay with temporal distance. This produces structured,
   localized attention maps instead of flat uniform profiles.

3. UNCERTAINTY CALIBRATION: During training, random contiguous segments
   of the input are masked (zeroed out). The model must reconstruct those
   masked regions from context alone. The Gaussian NLL loss naturally
   forces the predicted variance to be higher in masked regions (since
   the squared error is larger there and the loss penalizes
   overconfidence). This produces well-calibrated, input-dependent
   uncertainty.

4. ATTENTION MAP EXTRACTION: The forward pass can optionally return
   all attention weight matrices for post-hoc analysis and plotting.

The Ml4gwReconstructionModel wrapper is preserved with minimal changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import fnmatch
import math

from ml4gw import augmentations, distributions, gw, transforms, waveforms
import torchaudio.transforms as T
from typing import Tuple


# =====================================================================
# BUILDING BLOCKS
# =====================================================================

class LayerNorm1d(nn.Module):
    """Channel-last LayerNorm for [B, C, T] tensors."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):  # [B, C, T]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class FeedForward1D(nn.Module):
    """Conv1d-based gated feedforward: project up -> GELU gate -> project down."""
    def __init__(self, channels, expansion=4.0, dropout=0.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden * 2, 1),
            GatedActivation(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatedActivation(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * F.gelu(b)


# =====================================================================
# ROTARY POSITIONAL ENCODING (RoPE)
# =====================================================================

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) for 1-D temporal sequences.

    Instead of adding positional vectors to embeddings, RoPE rotates
    pairs of dimensions in Q and K by angles proportional to their
    position index. When computing Q_i . K_j, the dot product depends
    on (i - j) — the relative distance — not the absolute positions.

    This gives the attention two key properties:
      1. LOCALITY BIAS: nearby time steps naturally get higher attention
         scores because the rotation angle difference is small.
      2. SMOOTH DECAY: attention falls off smoothly with distance,
         producing structured, physically interpretable attention maps
         rather than flat uniform profiles.

    For GW signals this is ideal: the model should attend more strongly
    to nearby oscillation cycles than to distant noise.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding" (2021), arXiv:2104.09864
    """
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        # Precompute inverse frequencies for dimension pairs
        # theta_i = 1 / base^(2i/dim) for i = 0, 1, ..., dim/2 - 1
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin tables for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """Build and cache cos/sin rotation matrices."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype,
                         device=self.inv_freq.device)
        # Outer product: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        # Repeat each frequency for the pair: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        # Cache as [1, 1, seq_len, dim] for broadcasting with [B, h, T, d]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :],
                             persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :],
                             persistent=False)

    @staticmethod
    def _rotate_half(x):
        """Rotate pairs of dimensions: [x0,x1,x2,x3,...] -> [-x1,x0,-x3,x2,...]"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """
        Apply rotary encoding to Q and K.

        Parameters
        ----------
        q, k : [B, heads, T, head_dim]

        Returns
        -------
        q_rot, k_rot : same shape, with rotary encoding applied
        """
        T_seq = q.shape[2]

        # Extend cache if needed
        if T_seq > self.cos_cached.shape[2]:
            self._build_cache(T_seq)

        cos = self.cos_cached[:, :, :T_seq, :q.shape[-1]]
        sin = self.sin_cached[:, :, :T_seq, :q.shape[-1]]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# =====================================================================
# TEMPORAL SELF-ATTENTION WITH RoPE
# =====================================================================

class TemporalSelfAttention(nn.Module):
    """
    Standard multi-head self-attention over the TIME axis, with
    Rotary Positional Encoding (RoPE).

    Input:  [B, C, T]
    Output: [B, C, T], optionally also [B, heads, T, T] attention weights

    The attention matrix A[i,j] tells you how much time step i attends
    to time step j. With RoPE, the scores depend on relative distance
    |i-j|, producing localized attention patterns that are directly
    comparable to the Fisher information integrand.
    """
    def __init__(self, channels, num_heads=4, dropout=0.0, max_seq_len=8192,
                 rope_base=10000.0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Rotary positional encoding
        self.rope = RotaryPositionalEncoding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        # Storage for attention weights (set by forward when requested)
        self._attn_weights = None

    def forward(self, x, return_attn=False):
        """
        Parameters
        ----------
        x : [B, C, T]
        return_attn : bool
            If True, store attention weights in self._attn_weights

        Returns
        -------
        out : [B, C, T]
        """
        B, C, T = x.shape
        h, d = self.num_heads, self.head_dim

        qkv = self.qkv(x)                           # [B, 3C, T]
        qkv = qkv.view(B, 3, h, d, T)               # [B, 3, h, d, T]
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # each [B, h, d, T]

        # Reshape to [B, h, T, d] for RoPE (operates on last dim = head_dim)
        q = q.permute(0, 1, 3, 2)  # [B, h, T, d]
        k = k.permute(0, 1, 3, 2)  # [B, h, T, d]

        # Apply rotary positional encoding
        q, k = self.rope(q, k)

        # Attention: [B, h, T, d] @ [B, h, d, T] -> [B, h, T, T]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_logits, dim=-1)         # [B, h, T, T]
        attn = self.attn_drop(attn)

        if return_attn:
            self._attn_weights = attn.detach()

        # Apply attention to values
        # v is [B, h, d, T], reshape to [B, h, T, d]
        v = v.permute(0, 1, 3, 2)                    # [B, h, T, d]
        out = torch.matmul(attn, v)                   # [B, h, T, d]
        out = out.permute(0, 1, 3, 2)                 # [B, h, d, T]
        out = out.contiguous().view(B, C, T)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock1D(nn.Module):
    """
    Pre-norm transformer block:  LN -> TemporalSelfAttn -> +res -> LN -> FFN -> +res
    """
    def __init__(self, channels, num_heads=4, expansion=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = LayerNorm1d(channels)
        self.attn  = TemporalSelfAttention(channels, num_heads, dropout)
        self.norm2 = LayerNorm1d(channels)
        self.ffn   = FeedForward1D(channels, expansion, dropout)

    def forward(self, x, return_attn=False):
        x = x + self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.ffn(self.norm2(x))
        return x


# =====================================================================
# U-NET WITH TEMPORAL SELF-ATTENTION
# =====================================================================

class Downsample1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4,
                                          stride=2, padding=1)

    def forward(self, x):
        return self.deconv(x)


class TemporalAttnUNet1D(nn.Module):
    """
    U-Net with temporal self-attention (+ RoPE) at every level.

    Returns (mu, logvar) for heteroscedastic Gaussian NLL training.

    The attention matrices at every level are accessible after a forward
    pass via get_attention_maps(), enabling direct Fisher information
    localization.
    """
    def __init__(
        self,
        in_channels: int = 1,
        dims=(32, 64, 128, 256),
        num_blocks=(2, 2, 2, 2),
        num_heads=(2, 4, 4, 8),
        expansion: float = 4.0,
        dropout: float = 0.0,
        out_channels: int = 2,     # (mu, logvar)
        min_var: float = 1e-6,
    ):
        super().__init__()
        assert len(dims) == len(num_blocks) == len(num_heads)
        self.levels  = len(dims)
        self.min_var = min_var
        self.out_ch  = out_channels

        # Shallow feature embedding
        self.embed = nn.Conv1d(in_channels, dims[0], kernel_size=7, padding=3)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs      = nn.ModuleList()
        for i in range(self.levels):
            blocks = nn.ModuleList([
                TransformerBlock1D(dims[i], num_heads[i], expansion, dropout)
                for _ in range(num_blocks[i])
            ])
            self.enc_blocks.append(blocks)
            if i < self.levels - 1:
                self.downs.append(Downsample1D(dims[i], dims[i + 1]))

        # Decoder
        self.ups        = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(self.levels - 1, 0, -1):
            self.ups.append(Upsample1D(dims[i], dims[i - 1]))
            self.fuse_convs.append(
                nn.Conv1d(dims[i - 1] * 2, dims[i - 1], kernel_size=1)
            )
            blocks = nn.ModuleList([
                TransformerBlock1D(dims[i - 1], num_heads[i - 1],
                                   expansion, dropout)
                for _ in range(num_blocks[i - 1])
            ])
            self.dec_blocks.append(blocks)

        # Output head
        self.head = nn.Conv1d(dims[0], out_channels, kernel_size=1)

        # For attention map storage
        self._return_attn = False

    def forward(self, x, return_attn=False):
        """
        Parameters
        ----------
        x : [B, 1, T]
        return_attn : bool
            If True, collect attention maps from all layers.

        Returns
        -------
        mu, logvar : each [B, 1, T]
        """
        self._return_attn = return_attn

        x = self.embed(x)  # [B, C0, T]

        skips = []
        for i in range(self.levels):
            for block in self.enc_blocks[i]:
                x = block(x, return_attn=return_attn)
            skips.append(x)
            if i < self.levels - 1:
                x = self.downs[i](x)

        for up, fuse, dec_blocks, skip in zip(
            self.ups, self.fuse_convs, self.dec_blocks,
            reversed(skips[:-1])
        ):
            x = up(x)
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                x = F.pad(x, (0, diff))
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            for block in dec_blocks:
                x = block(x, return_attn=return_attn)

        out = self.head(x)  # [B, out_channels, T]
        mu, s = torch.chunk(out, 2, dim=1)
        var    = F.softplus(s) + self.min_var
        logvar = torch.log(var)
        return mu, logvar

    def get_attention_maps(self):
        """
        Collect all stored attention maps from the last forward pass.

        Returns
        -------
        maps : dict
            Keys are like "enc_0_block_1", values are [B, heads, T_level, T_level].
            T_level is the temporal resolution at that U-Net level
            (T, T/2, T/4, T/8 for 4 levels).
        """
        maps = {}
        for i, blocks in enumerate(self.enc_blocks):
            for j, block in enumerate(blocks):
                w = block.attn._attn_weights
                if w is not None:
                    maps[f"enc_{i}_block_{j}"] = w
        for i, blocks in enumerate(self.dec_blocks):
            for j, block in enumerate(blocks):
                w = block.attn._attn_weights
                if w is not None:
                    maps[f"dec_{i}_block_{j}"] = w
        return maps

    def clear_attention_maps(self):
        """Reset stored attention weights."""
        for mod in self.modules():
            if isinstance(mod, TemporalSelfAttention):
                mod._attn_weights = None


# =====================================================================
# LOSS: Gaussian NLL with overconfidence penalty
# =====================================================================

def gaussian_nll(mu, logvar, target, beta=1e-3):
    var = torch.exp(logvar).clamp(min=1e-6, max=1e3)
    nll = 0.5 * (logvar + (target - mu) ** 2 / var + math.log(2 * math.pi))
    reg = beta * torch.mean(torch.exp(-logvar))
    return nll.mean() + reg


# =====================================================================
# MASKING AUGMENTATION FOR UNCERTAINTY CALIBRATION
# =====================================================================

def apply_random_mask(x, mask_prob=0.3, min_mask_len=16, max_mask_len=128):
    """
    Zero out a random contiguous segment of the input to train
    calibrated uncertainty.
    """
    B, C, T_len = x.shape
    x_masked = x.clone()
    mask_indicator = torch.zeros_like(x)

    for i in range(B):
        if torch.rand(1).item() < mask_prob:
            L = torch.randint(min_mask_len, max_mask_len + 1, (1,)).item()
            L = min(L, T_len)
            start = torch.randint(0, T_len - L + 1, (1,)).item()
            x_masked[i, :, start:start + L] = 0.0
            mask_indicator[i, :, start:start + L] = 1.0

    return x_masked, mask_indicator


# =====================================================================
# MULTI-STFT LOSS (kept from original)
# =====================================================================

class MultiSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[128, 256, 512], hop_sizes=[32, 64, 128]):
        super().__init__()
        self.fft_sizes, self.hop_sizes = fft_sizes, hop_sizes

    def forward(self, pred, targ):
        loss = 0
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            P = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop,
                           return_complex=True)
            T_stft = torch.stft(targ.squeeze(1), n_fft=n_fft, hop_length=hop,
                                return_complex=True)
            loss += F.l1_loss(P.abs(), T_stft.abs())
        return loss / len(self.fft_sizes)


# =====================================================================
# ATTENTION MAP ANALYSIS UTILITIES
# =====================================================================

def extract_temporal_attention_profile(attn_maps, target_T, sample_rate):
    """
    Aggregate attention maps from all layers into a single
    per-time-step importance profile.

    For each T*T attention map A, the "importance" of time step j
    is how much all other time steps attend to it:  sum_i A[i, j]
    (column sum). This is the standard attention rollout approach.
    """
    aggregated = np.zeros(target_T, dtype=np.float64)
    per_layer  = {}

    for name, attn in attn_maps.items():
        # attn: [B, heads, T_level, T_level]
        # Column-sum = how much each position is attended to
        col_sum = attn[0].mean(dim=0).sum(dim=0).cpu().numpy()  # [T_level]
        T_level = len(col_sum)

        per_layer[name] = col_sum

        # Interpolate to input resolution
        if T_level == target_T:
            upsampled = col_sum
        else:
            t = torch.tensor(col_sum, dtype=torch.float32)
            upsampled = F.interpolate(
                t.unsqueeze(0).unsqueeze(0), size=target_T,
                mode='linear', align_corners=False
            ).squeeze().numpy()

        aggregated += upsampled

    aggregated /= max(len(attn_maps), 1)
    time_axis = np.arange(target_T) / sample_rate

    return aggregated, per_layer, time_axis


def compute_fisher_integrand(time_axis, f_qnm=250.0, tau_qnm=3.5e-3,
                              t_start=0.0):
    """
    Analytical Fisher information integrand for delta_tau of a damped sinusoid,
    following Eq. 6 of the GW230814 paper.
    """
    t = time_axis - t_start
    mask = t >= 0
    t_pos = np.where(mask, t, 0.0)
    dh_dtau = (t_pos / tau_qnm ** 2) * np.exp(-t_pos / tau_qnm) * \
              np.cos(2 * np.pi * f_qnm * t_pos)
    fisher = dh_dtau ** 2
    fisher[~mask] = 0
    return fisher


def plot_attention_vs_fisher(time_axis, signal, reconstruction,
                              attn_profile, fisher, save_path=None):
    """
    Three-panel comparison figure for the paper.
    """
    t_ms = time_axis * 1000

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'hspace': 0.06})

    # Panel 1: signal + reconstruction
    ax = axes[0]
    ax.plot(t_ms, signal, color='gray', alpha=0.5, lw=0.8,
            label='Whitened data')
    ax.plot(t_ms, reconstruction, color='tab:orange', lw=1.5,
            label='AWaRe reconstruction')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.set_title('GW230814 — Temporal Attention (with RoPE) vs Fisher Information')

    # Panel 2: attention profile
    ax = axes[1]
    norm_a = attn_profile / (attn_profile.max() + 1e-12)
    ax.fill_between(t_ms, 0, norm_a, color='tab:blue', alpha=0.3)
    ax.plot(t_ms, norm_a, color='tab:blue', lw=1.2,
            label='Attention importance (column sum)')
    ax.set_ylabel('Normalized\nattention')
    ax.legend(loc='upper right')

    # Panel 3: Fisher comparison
    ax = axes[2]
    norm_f = fisher / (fisher.max() + 1e-12)
    ax.plot(t_ms, norm_f, color='tab:red', lw=1.5,
            label=r'Fisher $I_\tau(t)$  ($f=250$ Hz, $\tau=3.5$ ms)')
    ax.plot(t_ms, norm_a, color='tab:blue', lw=1.0, ls='--', alpha=0.7,
            label='Attention (overlay)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Normalized')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_uncertainty_calibration(time_axis, signal, mu, std,
                                  mask_indicator=None, save_path=None):
    """
    Show that uncertainty grows in masked / low-information regions.
    """
    t_ms = time_axis * 1000
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={'hspace': 0.06})

    ax = axes[0]
    ax.plot(t_ms, signal, color='gray', alpha=0.5, lw=0.8,
            label='Clean signal')
    ax.plot(t_ms, mu, color='tab:orange', lw=1.2, label='mu (reconstruction)')
    ax.fill_between(t_ms, mu - 1.645 * std, mu + 1.645 * std,
                    color='gold', alpha=0.4, label='90% CI')
    if mask_indicator is not None and mask_indicator.any():
        ax.fill_between(t_ms, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=mask_indicator > 0.5,
                        color='red', alpha=0.15, label='Masked region')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.set_title('Uncertainty calibration: masked vs unmasked regions')

    ax = axes[1]
    ax.plot(t_ms, std, color='tab:purple', lw=1.2, label='sigma (predicted std)')
    if mask_indicator is not None and mask_indicator.any():
        ax.fill_between(t_ms, 0, std.max() * 1.1,
                        where=mask_indicator > 0.5,
                        color='red', alpha=0.15, label='Masked region')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Predicted sigma')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# =====================================================================
# Ml4gwReconstructionModel WRAPPER
# (preserves your full training infrastructure, adds masking)
# =====================================================================

class Ml4gwReconstructionModel(torch.nn.Module):
    def __init__(
        self,
        architecture: nn.Module,
        ifos: list = ["L1"],
        kernel_length: float = 1.0,
        fduration: float = 2,
        psd_length: float = 16,
        sample_rate: float = 1024,
        fftlength: float = 2,
        highpass: float = 20,
        chunk_length: float = 128,
        reads_per_chunk: int = 40,
        learning_rate: float = 1e-4,
        batch_size: int = 128,
        waveform_prob: float = 1.0,
        approximant: callable = None,
        param_dict: dict = None,
        waveform_duration: float = 8,
        f_min: float = 20,
        f_max: float = None,
        f_ref: float = 20,
        min_snr: float = 10,
        max_snr: float = 35,
        inversion_prob: float = 0.5,
        reversal_prob: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_epochs: int = 200,
        checkpoint_dir: str = "checkpoints_temporal_attn_rope_ringdown",
        log_dir: str = "logs",
        # masking parameters
        mask_prob: float = 0.3,
        min_mask_len: int = 16,
        max_mask_len: int = 128,
    ) -> None:
        super().__init__()
        self.nn = architecture
        self.device = device

        # Save hyperparameters
        self.ifos = ifos
        self.kernel_length = kernel_length
        self.fduration = fduration
        self.psd_length = psd_length
        self.sample_rate = sample_rate
        self.fftlength = fftlength
        self.highpass = highpass
        self.chunk_length = chunk_length
        self.reads_per_chunk = reads_per_chunk
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.waveform_prob = waveform_prob
        self.waveform_duration = waveform_duration
        self.inversion_prob = inversion_prob
        self.reversal_prob = reversal_prob
        self.f_min = f_min
        self.f_max = f_max or (sample_rate / 2)
        self.f_ref = f_ref
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Masking parameters
        self.mask_prob = mask_prob
        self.min_mask_len = min_mask_len
        self.max_mask_len = max_mask_len

        self.use_presaved = False

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Augmentations
        self.inverter = augmentations.SignalInverter(prob=inversion_prob)
        self.reverser = augmentations.SignalReverser(prob=reversal_prob)

        # PSD / whitening
        from ml4gw import transforms
        self.spectral_density = transforms.SpectralDensity(
            sample_rate, fftlength, average="median", fast=False
        ).to(device)
        self.whitener = transforms.Whiten(
            fduration, sample_rate, highpass=highpass
        ).to(device)

        # Interferometer geometry
        from ml4gw import gw
        detector_tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("detector_tensors", detector_tensors.to(device))
        self.register_buffer("detector_vertices", vertices.to(device))

        # Frequency setup
        nyquist = sample_rate / 2
        num_samples = int(waveform_duration * sample_rate)
        num_freqs = num_samples // 2 + 1
        frequencies = torch.linspace(0, nyquist, num_freqs)
        freq_mask = (frequencies >= f_min) * (frequencies < self.f_max)
        self.register_buffer("frequencies", frequencies.to(device))
        self.register_buffer("freq_mask", freq_mask.to(device))

        # Parameter distributions
        if param_dict is None:
            from ml4gw.distributions import PowerLaw, Sine, DeltaFunction
            from torch.distributions import Uniform
            param_dict = {
                "chirp_mass": Uniform(10.0, 100.0),
                "mass_ratio": Uniform(0.25, 0.999),
                "chi1": Uniform(-0.999, 0.999),
                "chi2": Uniform(-0.999, 0.999),
                "distance": PowerLaw(100, 1000, 2),
                "phic": DeltaFunction(0),
                "inclination": Sine(),
            }
        self.param_dict = param_dict

        # Waveform generator
        from ml4gw import waveforms
        if approximant is None:
            approximant = waveforms.cbc.IMRPhenomD
        self.approximant = approximant().to(device)
        self.psi = torch.distributions.Uniform(0, torch.pi)
        self.phi = torch.distributions.Uniform(-torch.pi, torch.pi)

        from ml4gw import distributions
        self.snr = torch.distributions.Uniform(min_snr, max_snr)

        self.kernel_size = int(kernel_length * sample_rate)
        self.window_size = self.kernel_size + int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)

        self.optimizer = torch.optim.AdamW(
            self.nn.parameters(), self.learning_rate
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.train_losses = []
        self.valid_metrics = []
        self.steps = []

        self.responses_file = None

    def forward(self, X, return_attn=False):
        return self.nn(X, return_attn=return_attn)

    # ─────────────────────────────────────────────────────────────────
    # AUGMENTATION (preserved from original, adds masking)
    # ─────────────────────────────────────────────────────────────────

    def augment_noise2noise(self, X: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        background, X = torch.split(
            X, [self.psd_size, self.window_size], dim=-1
        )
        psd = self.spectral_density(background.double())
        batch_size = X.size(0)

        if hasattr(self, "responses_file") and self.responses_file is not None:
            with h5py.File(self.responses_file, "r") as f:
                if ("injection_parameters" in f and
                        "l1_signal_whitened" in f["injection_parameters"]):
                    responses_data = f[
                        "injection_parameters/l1_signal_whitened"][()]
                else:
                    responses_data = f["data"][:]

            if responses_data.ndim == 1:
                responses = torch.tensor(
                    responses_data, device=self.device, dtype=torch.float32
                )
                L = responses.shape[0]
                window = 2048
                if L > window:
                    responses = responses[-window:]
                elif L < window:
                    responses = F.pad(responses, (0, window - L))
                responses = responses.unsqueeze(0).unsqueeze(0)
                responses = responses.repeat(batch_size, 1, 1)
                mask = torch.ones(
                    batch_size, dtype=torch.bool, device=self.device
                )

            elif responses_data.ndim == 2:
                num_samples, total_timesteps = responses_data.shape
                window = 2048
                segments = []
                sample_idxs = torch.randint(
                    0, num_samples, (batch_size,), device=self.device
                )
                if total_timesteps >= window:
                    max_start = total_timesteps - window
                    start_idxs = torch.randint(
                        0, max_start + 1, (batch_size,), device=self.device
                    )
                    for s_idx, start in zip(sample_idxs, start_idxs):
                        raw = responses_data[
                            s_idx.item(), start.item():start.item() + window
                        ]
                        segments.append(
                            torch.from_numpy(raw).to(device=self.device)
                        )
                else:
                    pad_size = window - total_timesteps
                    for s_idx in sample_idxs:
                        raw = responses_data[s_idx.item(), :]
                        padded = np.pad(raw, (0, pad_size), mode="constant")
                        segments.append(
                            torch.from_numpy(padded).to(device=self.device)
                        )
                responses = torch.stack(segments, dim=0).unsqueeze(1)
                mask = torch.ones(
                    batch_size, dtype=torch.bool, device=self.device
                )

                X1 = X.clone()
                indices = torch.randperm(batch_size)
                X2 = X[indices].clone()
                X1 = X1[:, 0:1, :]
                X2 = X2[:, 0:1, :]
                X1 = self.whitener(X1, psd)
                X2 = self.whitener(X2, psd)
                responses = responses / (2048 ** 0.5)
                X1[mask] += responses.float()
                X2[mask] += responses.float()
                labels = torch.zeros_like(X1)
                labels[mask] = responses.float()
            else:
                raise ValueError(
                    f"Unsupported responses_data.ndim = {responses_data.ndim}"
                )
        else:
            hc, hp, mask = self.generate_waveforms(batch_size)
            responses = self.project_waveforms(hc, hp)
            responses = self.rescale_snrs(responses, psd[mask])
            responses = self.sample_waveforms(responses)

            X1 = X.clone()
            indices = torch.randperm(batch_size)
            X2 = X[indices].clone()
            X1[mask] += responses.float()
            X2[mask] += responses.float()
            X1 = X1[:, 0:1, :]
            X1 = self.whitener(X1, psd)
            X2 = X2[:, 0:1, :]
            X2 = self.whitener(X2, psd)
            labels = torch.zeros_like(X1)
            labels[mask] = self.whitener(responses, psd[mask])

        return X1, X2, labels

    def augment_for_test(self, X: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:
        background, X_signal = torch.split(
            X, [self.psd_size, self.window_size], dim=-1
        )
        psd = self.spectral_density(background.double())
        batch_size = X_signal.size(0)

        if hasattr(self, "responses_file") and self.responses_file is not None:
            with h5py.File(self.responses_file, "r") as f:
                if ("injection_parameters" in f and
                        "l1_signal_whitened" in f["injection_parameters"]):
                    responses_data = f[
                        "injection_parameters/l1_signal_whitened"][()]
                else:
                    responses_data = f["data"][:]

            if responses_data.ndim == 1:
                responses = torch.tensor(
                    responses_data, device=self.device, dtype=torch.float32
                )
                L = responses.shape[0]
                window = 2048
                if L > window:
                    responses = responses[-window:]
                elif L < window:
                    responses = F.pad(responses, (0, window - L))
                responses = responses.unsqueeze(0).unsqueeze(0)
                responses = responses.repeat(batch_size, 1, 1)
                mask = torch.ones(
                    batch_size, dtype=torch.bool, device=self.device
                )
            elif responses_data.ndim == 2:
                num_samples, total_timesteps = responses_data.shape
                window = 2048
                segments = []
                sample_idxs = torch.randint(
                    0, num_samples, (batch_size,), device=self.device
                )
                if total_timesteps >= window:
                    max_start = total_timesteps - window
                    start_idxs = torch.randint(
                        0, max_start + 1, (batch_size,), device=self.device
                    )
                    for s_idx, start in zip(sample_idxs, start_idxs):
                        raw = responses_data[
                            s_idx.item(), start.item():start.item() + window
                        ]
                        segments.append(
                            torch.from_numpy(raw).to(device=self.device)
                        )
                else:
                    pad_size = window - total_timesteps
                    for s_idx in sample_idxs:
                        raw = responses_data[s_idx.item(), :]
                        padded = np.pad(raw, (0, pad_size), mode="constant")
                        segments.append(
                            torch.from_numpy(padded).to(device=self.device)
                        )
                responses = torch.stack(segments, dim=0).unsqueeze(1)
                mask = torch.ones(
                    batch_size, dtype=torch.bool, device=self.device
                )
            else:
                raise ValueError(
                    f"Unsupported responses_data.ndim = {responses_data.ndim}"
                )

            responses = responses / (2048 ** 0.5)
            X_signal_aug = X.clone()[:, 0:1, :]
            X_whiten = self.whitener(X_signal_aug, psd)
            X_whiten[mask] += responses.float()
            labels = torch.zeros_like(X_whiten)
            labels[mask] = responses.float()

        else:
            hc, hp, mask = self.generate_waveforms(batch_size)
            responses = self.project_waveforms(hc, hp)
            responses_scaled = self.rescale_snrs(responses, psd[mask])
            responses_sampled = self.sample_waveforms(responses_scaled)
            X_signal_aug = X_signal.clone()
            X_signal_aug[mask] += responses_sampled.float()
            X_signal_aug = X_signal_aug[:, 0:1, :]
            X_whiten = self.whitener(X_signal_aug, psd)
            labels = torch.zeros_like(X_whiten)
            labels[mask] = self.whitener(responses_sampled, psd[mask])

        return X_whiten, labels

    # ─────────────────────────────────────────────────────────────────
    # TRAINING / VALIDATION STEPS
    # ─────────────────────────────────────────────────────────────────

    def training_step(self, batch):
        if isinstance(batch, (list, tuple)):
            X, y = batch
        else:
            X, y = batch, None

        if self.use_presaved:
            X1 = X.to(self.device)
            labels = y.to(self.device)
        else:
            X1, _, labels = self.augment_noise2noise(X.to(self.device))

        # Apply random masking for uncertainty calibration
        X1_masked, mask_ind = apply_random_mask(
            X1,
            mask_prob=self.mask_prob,
            min_mask_len=self.min_mask_len,
            max_mask_len=self.max_mask_len,
        )

        self.optimizer.zero_grad()
        mu, logvar = self(X1_masked)
        loss = gaussian_nll(mu, logvar, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def validation_step(self, batch):
        if isinstance(batch, (list, tuple)):
            X, y = batch
        else:
            X, y = batch, None

        if self.use_presaved:
            X1 = X.to(self.device)
            labels = y.to(self.device)
        else:
            X1, _, labels = self.augment_noise2noise(X.to(self.device))

        # Validate WITHOUT masking to measure clean performance
        with torch.no_grad():
            mu, logvar = self(X1)
            loss = gaussian_nll(mu, logvar, labels)
        return loss.item()

    # ─────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────

    def log_metrics(self):
        with open(os.path.join(self.log_dir, 'metrics.csv'), 'a') as f:
            for step, loss in zip(self.steps, self.train_losses):
                f.write(f"{self.current_epoch},{step},{loss},,\n")
            for metric_val in self.valid_metrics:
                f.write(
                    f"{self.current_epoch},{self.global_step},,,"
                    f"{metric_val}\n"
                )
        self.steps = []
        self.train_losses = []
        self.valid_metrics = []

    def fit(self):
        if not os.path.exists(os.path.join(self.log_dir, 'metrics.csv')):
            with open(os.path.join(self.log_dir, 'metrics.csv'), 'w') as f:
                f.write("epoch,step,train_loss, ,valid_loss\n")

        train_dataloader = self.create_train_dataloader()
        val_dataloader = self.create_train_dataloader()
        total_steps = self.max_epochs * len(train_dataloader)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.learning_rate,
            pct_start=0.1, total_steps=total_steps
        )

        self.to(self.device)
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch + 1}/{self.max_epochs}")

            self.nn.train()
            train_loop = tqdm(train_dataloader, desc="Training")
            for batch in train_loop:
                loss = self.training_step(batch)
                scheduler.step()
                self.global_step += 1
                self.steps.append(self.global_step)
                self.train_losses.append(loss)
                if self.global_step % 5 == 0:
                    train_loop.set_postfix(loss=f"{loss:.4f}")

            self.nn.eval()
            val_losses = []
            val_loop = tqdm(val_dataloader, desc="Validation")
            for batch in val_loop:
                loss = self.validation_step(batch)
                val_losses.append(loss)
            avg_val_loss = np.mean(val_losses)
            self.valid_metrics.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_metric:
                self.best_metric = avg_val_loss
                self.save_checkpoint(os.path.join(
                    self.checkpoint_dir,
                    'best_model_temporal_attn_rope_ringdown.pt'
                ))
                print(f"New best model saved with Loss: {avg_val_loss:.4f}")
            self.log_metrics()

    # ─────────────────────────────────────────────────────────────────
    # TEST + ATTENTION MAP EXTRACTION
    # ─────────────────────────────────────────────────────────────────

    def test_and_save_on_test_set(self):
        test_dataloader = self.create_test_dataloader()
        all_noisy, all_clean, all_recon, all_std = [], [], [], []
        all_lower90, all_upper90 = [], []
        self.nn.eval()

        for noisy, clean in tqdm(test_dataloader, desc="Testing"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            with torch.no_grad():
                mu, logvar = self.nn(noisy)
                var = torch.exp(logvar)
                std = torch.sqrt(var)
            all_noisy.append(noisy.cpu())
            all_clean.append(clean.cpu())
            all_recon.append(mu.cpu())
            all_std.append(std.cpu())
            all_lower90.append((mu - 1.645 * std).cpu())
            all_upper90.append((mu + 1.645 * std).cpu())

        noisy_all = torch.cat(all_noisy, dim=0)
        clean_all = torch.cat(all_clean, dim=0)
        recon_all = torch.cat(all_recon, dim=0)
        std_all   = torch.cat(all_std, dim=0)
        l90_all   = torch.cat(all_lower90, dim=0)
        u90_all   = torch.cat(all_upper90, dim=0)

        torch.save({
            'noisy': noisy_all, 'clean': clean_all, 'recon': recon_all,
            'std': std_all, 'lower90': l90_all, 'upper90': u90_all,
        }, "test_results_temporal_attn_rope_ringdown.pt")
        print("Test results saved.")

        # Plot first 10 samples
        os.makedirs("test_plots_temporal_attn_rope_ringdown", exist_ok=True)
        for idx in range(min(10, noisy_all.size(0))):
            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(recon_all.shape[-1])
            ax.plot(clean_all[idx, 0].numpy(), label="Clean", ls="--", lw=1)
            ax.plot(recon_all[idx, 0].numpy(), label="Recon mu", ls=":", lw=1)
            ax.fill_between(x, l90_all[idx, 0].numpy(),
                            u90_all[idx, 0].numpy(),
                            alpha=0.4, label="90% CI")
            ax.legend()
            ax.set_title(f"Test Sample {idx}")
            plt.tight_layout()
            fig.savefig(os.path.join("test_plots_temporal_attn_rope_ringdown",
                                     f"sample_{idx}.png"))
            plt.close(fig)
        print("Test plots saved.")

    def test_uncertainty_calibration(self, num_samples=5):
        """
        Demonstrate that masking input regions increases predicted sigma.
        """
        test_dataloader = self.create_test_dataloader()
        self.nn.eval()
        os.makedirs("calibration_plots_rope_ringdown", exist_ok=True)

        count = 0
        for noisy, clean in test_dataloader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            for i in range(min(num_samples - count, noisy.size(0))):
                x_single = noisy[i:i + 1]
                clean_single = clean[i, 0].cpu().numpy()
                T_len = x_single.shape[-1]
                time_axis = np.arange(T_len) / self.sample_rate

                # Unmasked forward pass
                with torch.no_grad():
                    mu_clean, logvar_clean = self.nn(x_single)
                std_clean = torch.exp(0.5 * logvar_clean)[0, 0].cpu().numpy()
                mu_clean_np = mu_clean[0, 0].cpu().numpy()

                # Masked forward pass: mask the middle third
                x_masked = x_single.clone()
                start = T_len // 3
                end = 2 * T_len // 3
                x_masked[:, :, start:end] = 0.0
                mask_ind = np.zeros(T_len)
                mask_ind[start:end] = 1.0

                with torch.no_grad():
                    mu_masked, logvar_masked = self.nn(x_masked)
                std_masked = torch.exp(
                    0.5 * logvar_masked
                )[0, 0].cpu().numpy()
                mu_masked_np = mu_masked[0, 0].cpu().numpy()

                # Plot comparison
                plot_uncertainty_calibration(
                    time_axis, clean_single, mu_masked_np, std_masked,
                    mask_indicator=mask_ind,
                    save_path=os.path.join(
                        "calibration_plots_rope_ringdown", f"calibration_{count}.png"
                    )
                )

                # Print quantitative ratio
                sigma_in_mask = np.mean(std_masked[start:end])
                sigma_outside = np.mean(
                    np.concatenate([std_masked[:start], std_masked[end:]])
                )
                print(f"Sample {count}: sigma_masked={sigma_in_mask:.4f}, "
                      f"sigma_unmasked={sigma_outside:.4f}, "
                      f"ratio={sigma_in_mask / (sigma_outside + 1e-12):.2f}x")

                count += 1
                if count >= num_samples:
                    return

    def extract_attention_maps_for_sample(self, x_input):
        """
        Run a single forward pass with attention map collection.
        """
        self.nn.eval()
        self.nn.clear_attention_maps()
        with torch.no_grad():
            mu, logvar = self.nn(x_input, return_attn=True)
        attn_maps = self.nn.get_attention_maps()
        return mu, logvar, attn_maps

    # ─────────────────────────────────────────────────────────────────
    # WAVEFORM GENERATION (preserved from original)
    # ─────────────────────────────────────────────────────────────────

    def generate_waveforms(self, batch_size):
        rvs = torch.rand(size=(batch_size,), device=self.device)
        mask = rvs < self.waveform_prob
        num_injections = mask.sum().item()
        params = {
            k: v.sample((num_injections,)).to(self.device)
            for k, v in self.param_dict.items()
        }
        hc, hp = self.approximant(
            f=self.frequencies[self.freq_mask], f_ref=self.f_ref, **params
        )
        shape = (hc.shape[0], len(self.frequencies))
        hc_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)
        hp_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)
        hc_spectrum[:, self.freq_mask] = hc
        hp_spectrum[:, self.freq_mask] = hp
        hc = torch.fft.irfft(hc_spectrum) * self.sample_rate
        hp = torch.fft.irfft(hp_spectrum) * self.sample_rate
        ringdown_duration = 0.5
        ringdown_size = int(ringdown_duration * self.sample_rate)
        hc = torch.roll(hc, -ringdown_size, dims=-1)
        hp = torch.roll(hp, -ringdown_size, dims=-1)
        return hc, hp, mask

    def project_waveforms(self, hc, hp):
        N = len(hc)
        dec = torch.distributions.Uniform(-1, 1).sample((N,)).to(hc.device)
        psi = self.psi.sample((N,)).to(hc.device)
        phi = self.phi.sample((N,)).to(hc.device)
        from ml4gw import gw
        return gw.compute_observed_strain(
            dec=dec, psi=psi, phi=phi,
            detector_tensors=self.detector_tensors,
            detector_vertices=self.detector_vertices,
            sample_rate=self.sample_rate, cross=hc, plus=hp
        )

    def rescale_snrs(self, responses, psd):
        num_freqs = int(responses.size(-1) // 2) + 1
        added_channel = False
        if psd.dim() == 2:
            psd = psd.unsqueeze(1)
            added_channel = True
        if psd.size(-1) != num_freqs:
            psd = F.interpolate(
                psd, size=num_freqs, mode="linear", align_corners=False
            )
        if added_channel:
            psd = psd.squeeze(1)
        N = len(responses)
        target_snrs = self.snr.sample((N,)).to(responses.device)
        from ml4gw import gw
        return gw.reweight_snrs(
            responses=responses.double(), target_snrs=target_snrs,
            psd=psd, sample_rate=self.sample_rate, highpass=self.highpass,
        )

    def sample_waveforms(self, responses):
        responses = responses[:, :, -self.window_size:]
        pad = [0, int(self.window_size // 2)]
        responses = F.pad(responses, pad)
        from ml4gw.utils.slicing import sample_kernels
        return sample_kernels(responses, self.window_size, coincident=True)

    # ─────────────────────────────────────────────────────────────────
    # DATA LOADERS (preserved from original)
    # ─────────────────────────────────────────────────────────────────

    def create_train_dataloader(self):
        if hasattr(self, "train_hdf") and self.train_hdf is not None:
            with h5py.File(self.train_hdf, "r") as f:
                X = torch.Tensor(f["strain"][:])
                y = torch.Tensor(f["signal"][:])
            X = X[:, None, :]
            y = y[:, None, :]
            dataset = TensorDataset(X, y)
            return DataLoader(
                dataset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True
            )

        from ml4gw.dataloading import ChunkedTimeSeriesDataset
        from ml4gw.dataloading import Hdf5TimeSeriesDataset
        samples_per_epoch = 3000
        batches_per_epoch = (samples_per_epoch - 1) // self.batch_size + 1
        batches_per_chunk = batches_per_epoch // 10
        chunks_per_epoch = batches_per_epoch // batches_per_chunk + 1
        fnames = list(Path(
            "/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/"
            "ligo_data/ml4gw_data"
        ).iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames, channels=self.ifos,
            kernel_size=int(self.chunk_length * self.sample_rate),
            batch_size=self.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch, coincident=False,
        )
        return ChunkedTimeSeriesDataset(
            dataset, kernel_size=self.window_size + self.psd_size,
            batch_size=self.batch_size,
            batches_per_chunk=batches_per_chunk, coincident=False
        )

    def create_val_dataloader(self):
        return self.create_test_dataloader()

    def create_test_dataloader(self):
        if hasattr(self, "test_hdf") and self.test_hdf is not None:
            with h5py.File(self.test_hdf, "r") as f:
                X = torch.Tensor(f["strain"][:])
                y = torch.Tensor(f["signal"][:])
            X = X[:, None, :]
            y = y[:, None, :]
            dataset = TensorDataset(X, y)
            return DataLoader(
                dataset, batch_size=self.batch_size * 4,
                shuffle=False, pin_memory=True
            )

        if not os.path.exists("test_dataset_temporal_attn_rope_ringdown.hdf5"):
            print("Creating test dataset...")
            self._create_test_dataset(num_samples=100)

        with h5py.File("test_dataset_temporal_attn_rope_ringdown.hdf5", "r") as f:
            X = torch.Tensor(f["X"][:])
            y = torch.Tensor(f["y"][:])
        X = X[:, 0:1, :]
        y = y[:, 0:1, :]
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset, batch_size=self.batch_size * 4,
            shuffle=False, pin_memory=True
        )

    def _create_test_dataset(self, num_samples=50):
        new_dataloader = self.create_new_dataloader()
        X_list, y_list = [], []
        total_samples = 0
        for batch in new_dataloader:
            batch = batch.to(self.device)
            X_whiten, labels = self.augment_for_test(batch)
            X_list.append(X_whiten.cpu())
            y_list.append(labels.cpu())
            total_samples += X_whiten.size(0)
            if total_samples >= num_samples:
                break
        X_test = torch.cat(X_list, dim=0)[:num_samples]
        y_test = torch.cat(y_list, dim=0)[:num_samples]
        with h5py.File("test_dataset_temporal_attn_rope_ringdown.hdf5", "w") as f:
            f.create_dataset("X", data=X_test.numpy())
            f.create_dataset("y", data=y_test.numpy())
        print("Test dataset saved.")

    def create_new_dataloader(self):
        from ml4gw.dataloading import ChunkedTimeSeriesDataset
        from ml4gw.dataloading import Hdf5TimeSeriesDataset
        samples_per_epoch = 3000
        batches_per_epoch = (samples_per_epoch - 1) // self.batch_size + 1
        batches_per_chunk = batches_per_epoch // 10
        chunks_per_epoch = batches_per_epoch // batches_per_chunk + 1
        fnames = list(Path(
            "/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/"
            "ligo_data/ml4gw_data_test"
        ).iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames, channels=self.ifos,
            kernel_size=int(self.chunk_length * self.sample_rate),
            batch_size=self.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch, coincident=False,
        )
        return ChunkedTimeSeriesDataset(
            dataset, kernel_size=self.window_size + self.psd_size,
            batch_size=self.batch_size,
            batches_per_chunk=batches_per_chunk, coincident=False
        )

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hdf", type=str, default=None)
    parser.add_argument("--test_hdf", type=str, default=None)
    parser.add_argument("--responses_file", type=str, default=None)
    parser.add_argument("--polarizations_file", type=str, default=None)
    parser.add_argument("--extract_attention", action="store_true",
                        help="Run attention extraction after training")
    parser.add_argument("--test_calibration", action="store_true",
                        help="Run uncertainty calibration test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    architecture = TemporalAttnUNet1D(
        in_channels=1,
        dims=(32, 64, 128, 256),
        num_blocks=(2, 2, 2, 2),
        num_heads=(2, 4, 4, 8),
        expansion=4.0,
        dropout=0.0,
        out_channels=2,
    ).to(device)

    model = Ml4gwReconstructionModel(
        architecture=architecture,
        device=device,
        kernel_length=0.06,       # 0.06 for ringdown
        fduration=2,
        psd_length=16,
        sample_rate=16384,        # 16384 for ringdown
        highpass=20,
        min_snr=6,
        max_snr=45,
        # Masking parameters for uncertainty calibration
        mask_prob=0.3,
        min_mask_len=16,     # ~1 ms at 16384 Hz
        max_mask_len=128,    # ~8 ms at 16384 Hz
#        min_mask_len=10,         # ~10 ms at 1024 Hz
#        max_mask_len=100,        # ~100 ms at 1024 Hz
    ).to(device)

    if args.train_hdf is not None:
        model.train_hdf = args.train_hdf
        model.use_presaved = True
        print(f"Using pre-saved training dataset: {args.train_hdf}")

    if args.test_hdf is not None:
        model.test_hdf = args.test_hdf
        model.use_presaved = True
        print(f"Using pre-saved test dataset: {args.test_hdf}")

    if args.responses_file is not None:
        model.responses_file = args.responses_file

    if args.polarizations_file is not None:
        model.polarizations_file = args.polarizations_file

    # Train
    model.fit()
    model.test_and_save_on_test_set()

    # Uncertainty calibration test
    if args.test_calibration:
        print("\n" + "=" * 60)
        print("UNCERTAINTY CALIBRATION TEST")
        print("=" * 60)
        model.test_uncertainty_calibration(num_samples=5)

    # Attention map extraction (with auto-detected merger time for Fisher)
    if args.extract_attention:
        print("\n" + "=" * 60)
        print("ATTENTION MAP EXTRACTION (with RoPE)")
        print("=" * 60)
        os.makedirs("attention_results_rope_ringdown", exist_ok=True)

        test_dl = model.create_test_dataloader()
        sample_noisy, sample_clean = next(iter(test_dl))
        x_in = sample_noisy[0:1].to(device)
        T_input = x_in.shape[-1]

        mu, logvar, attn_maps = model.extract_attention_maps_for_sample(x_in)

        print(f"Collected attention maps from {len(attn_maps)} layers:")
        for name, attn in attn_maps.items():
            print(f"  {name}: {attn.shape}")

        # Aggregate into temporal profile
        profile, per_layer, time_axis = extract_temporal_attention_profile(
            attn_maps, target_T=T_input, sample_rate=model.sample_rate
        )

        # Auto-detect merger time from reconstruction peak
        recon_np = mu[0, 0].cpu().numpy()
        peak_time = time_axis[np.argmax(np.abs(recon_np))]
        print(f"Auto-detected merger peak at {peak_time*1000:.1f} ms")

        # Compute Fisher comparison aligned to actual merger
        fisher = compute_fisher_integrand(
            time_axis, f_qnm=250.0, tau_qnm=3.5e-3, t_start=peak_time
        )

        # Plot
        plot_attention_vs_fisher(
            time_axis,
            signal=sample_clean[0, 0].numpy(),
            reconstruction=recon_np,
            attn_profile=profile,
            fisher=fisher,
            save_path="attention_results_rope_ringdown/attn_vs_fisher.png"
        )

        # Save raw data
        np.savez(
            "attention_results_rope_ringdown/attention_data.npz",
            time_axis=time_axis,
            attention_profile=profile,
            fisher_integrand=fisher,
            reconstruction=recon_np,
            peak_time=peak_time,
            **{f"attn_{k}": v[0].mean(dim=0).cpu().numpy()
               for k, v in attn_maps.items()}
        )
        print("Attention analysis complete.")