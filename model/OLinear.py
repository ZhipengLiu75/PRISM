# models/OLinear_CycleRefine.py
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import numpy as np

from layers.RevIN import RevIN
from layers.Transformer_EncDec import (
    Encoder_ori,
    EncoderLayer,
    LinearEncoder,
    LinearEncoder_Multihead
)
from layers.SelfAttention_Family import AttentionLayer, EnhancedAttention


# =========================================================
# Adaptive L Predictor and Soft Cropping Mask
# Default disabled. Kept for compatibility.
# =========================================================
class AdaptiveLengthPredictor(nn.Module):
    """
    Predict variable-wise effective input length.

    Input:
        x: [B, T, N]

    Output:
        boundary: [B, N], value in [min_len, T]
    """
    def __init__(self, seq_len, channels, hidden_dim=64, dropout=0.1, min_len=24):
        super().__init__()

        self.seq_len = seq_len
        self.channels = channels
        self.min_len = min_len

        self.local_encoder = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=15,
                padding=7,
                groups=channels
            ),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.length_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x:
            [B, T, N]

        return:
            boundary: [B, N]
        """
        B, T, N = x.shape

        assert T == self.seq_len, \
            f"Expected seq_len={self.seq_len}, got T={T}"

        assert N == self.channels, \
            f"Expected channels={self.channels}, got N={N}"

        xc = x.transpose(1, 2).contiguous()  # [B, N, T]

        z = self.local_encoder(xc)

        mean_feat = z.mean(dim=-1)
        std_feat = z.std(dim=-1, unbiased=False)
        last_feat = z[:, :, -1]
        trend_feat = z[:, :, -1] - z[:, :, 0]
        energy_feat = torch.mean(torch.abs(z), dim=-1)

        feat = torch.stack(
            [mean_feat, std_feat, last_feat, trend_feat, energy_feat],
            dim=-1
        )  # [B, N, 5]

        ratio = torch.sigmoid(self.length_mlp(feat)).squeeze(-1)  # [B, N]

        min_len = float(self.min_len)
        max_len = float(T)

        boundary = min_len + (max_len - min_len) * ratio

        return boundary


class AdaptiveLengthMask(nn.Module):
    """
    Differentiable soft cropping mask according to predicted L.

    Input:
        boundary: [B, N]

    Output:
        mask: [B, T, N]
    """
    def __init__(self, init_tau=8.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    def forward(self, boundary, length):
        """
        boundary:
            [B, N]

        length:
            int
        """
        B, N = boundary.shape
        device = boundary.device
        dtype = boundary.dtype

        tau = torch.clamp(torch.exp(self.log_tau), min=1.0, max=100.0)

        position = torch.arange(length, device=device, dtype=dtype)

        # newest step distance = 1
        # oldest step distance = length
        distance = length - position
        distance = distance.view(1, length, 1)  # [1, T, 1]

        boundary = boundary.unsqueeze(1)  # [B, 1, N]

        mask = torch.sigmoid((boundary - distance) / tau)

        return mask


# =========================================================
# Cycle-Aware Refinement
# =========================================================
class CycleAwareRefinement(nn.Module):
    """
    Lightweight cycle-aware output refinement.

    It does not change Q matrix or OLinear backbone.

    Input:
        out:
            [B, pred_len, N], normalized prediction before RevIN denorm

        x_ori:
            [B, seq_len, N], normalized input after RevIN norm

    Output:
        refined:
            [B, pred_len, N]

    Main idea:
        Use the most recent cycle as a periodic anchor, then predict a
        small correction for the OLinear output.

        cycle_base = repeat(x_ori[:, -cycle_len:, :])
        correction = MLP([out, cycle_base])
        refined = out + gate * correction
    """
    def __init__(
        self,
        pred_len,
        channels,
        cycle_len=24,
        hidden_mult=2,
        dropout=0.1,
        init_ratio=0.03
    ):
        super().__init__()

        self.pred_len = pred_len
        self.channels = channels
        self.cycle_len = cycle_len

        hidden_dim = hidden_mult * channels

        self.refine_net = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels)
        )

        init_ratio = float(init_ratio)
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_ratio / (1.0 - init_ratio))

        self.gate_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

    def forward(self, out, x_ori):
        """
        out:
            [B, pred_len, N]

        x_ori:
            [B, seq_len, N]
        """
        B, pred_len, N = out.shape
        T = x_ori.shape[1]

        assert pred_len == self.pred_len, \
            f"Expected pred_len={self.pred_len}, got {pred_len}"

        assert N == self.channels, \
            f"Expected channels={self.channels}, got {N}"

        cycle_len = min(int(self.cycle_len), T)

        # [B, cycle_len, N]
        recent_cycle = x_ori[:, -cycle_len:, :]

        # Repeat cycle anchor to prediction length
        repeat_times = math.ceil(pred_len / cycle_len)

        # [B, pred_len, N]
        cycle_base = recent_cycle.repeat(1, repeat_times, 1)[:, :pred_len, :]

        # [B, pred_len, 2N]
        refine_input = torch.cat([out, cycle_base], dim=-1)

        # [B, pred_len, N]
        correction = self.refine_net(refine_input)

        gate = torch.sigmoid(self.gate_logit)

        refined = out + gate * correction

        return refined

    def get_gate(self):
        return torch.sigmoid(self.gate_logit).detach()


class HorizonConditionedMultiAnchor(nn.Module):
    """Content- and horizon-conditioned residual calibration.

    The module keeps the OLinear forecast as its primary path and builds three
    conservative references from the observed window: a periodic repeat, a
    local-level continuation, and a bounded local-trend continuation.  A
    content gate and a low-rank horizon gate decide which reference is useful
    at every sample, channel, and forecast position.  The final correction is
    initialized with a small residual strength, so enabling the branch does
    not destroy a well-behaved OLinear solution at the beginning of training.
    """
    def __init__(
        self,
        pred_len,
        channels,
        cycle_len=24,
        hidden_dim=32,
        horizon_rank=8,
        dropout=0.1,
        init_ratio=0.01,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.cycle_len = cycle_len

        self.content_gate = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.horizon_code = nn.Parameter(
            torch.randn(pred_len, horizon_rank) * 0.02
        )
        self.horizon_to_anchor = nn.Linear(horizon_rank, 3, bias=False)
        nn.init.zeros_(self.horizon_to_anchor.weight)

        init_ratio = min(max(float(init_ratio), 1e-4), 1.0 - 1e-4)
        self.strength_logit = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1.0 - init_ratio)))
        )

    def forward(self, out, x_ori):
        B, H, N = out.shape
        if H != self.pred_len or N != self.channels:
            raise ValueError(
                f"Expected output [B,{self.pred_len},{self.channels}], "
                f"got {tuple(out.shape)}"
            )

        T = x_ori.shape[1]
        period = min(int(self.cycle_len), T)
        recent = x_ori[:, -period:, :]
        previous = x_ori[:, -min(2 * period, T):-period, :]
        if previous.shape[1] == 0:
            previous = recent

        recent_mean = recent.mean(dim=1)
        previous_mean = previous.mean(dim=1)
        last = x_ori[:, -1, :]
        trend = torch.tanh(recent_mean - previous_mean)
        volatility = recent.std(dim=1, unbiased=False)

        features = torch.stack(
            [last, recent_mean, trend, volatility], dim=-1
        )  # [B, N, 4]
        content_logits = self.content_gate(features)  # [B, N, 3]
        horizon_logits = self.horizon_to_anchor(self.horizon_code)  # [H, 3]
        weights = torch.softmax(
            content_logits[:, :, None, :] + horizon_logits[None, None, :, :],
            dim=-1,
        )

        repeats = math.ceil(H / period)
        periodic = recent.repeat(1, repeats, 1)[:, :H, :]
        level = last[:, None, :].expand(-1, H, -1)
        horizon = torch.linspace(
            1.0 / H, 1.0, H, device=out.device, dtype=out.dtype
        )[None, :, None]
        trend_anchor = level + horizon * trend[:, None, :]
        anchors = torch.stack([periodic, level, trend_anchor], dim=-1)

        correction = ((anchors - out.unsqueeze(-1)) * weights.transpose(1, 2)).sum(-1)
        strength = torch.sigmoid(self.strength_logit)
        return out + strength * correction

    def get_strength(self):
        return torch.sigmoid(self.strength_logit).detach()


# =========================================================
# OLinear + Cycle-Aware Refinement
# =========================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len

        self.hidden_size = self.d_model = configs.d_model
        self.d_ff = configs.d_ff

        self.Q_chan_indep = getattr(configs, "Q_chan_indep", False)

        # =====================================================
        # Adaptive L cropping
        # Default OFF.
        # =====================================================
        self.use_adaptive_l = getattr(configs, "use_adaptive_l", True)
        self.min_context_len = getattr(configs, "min_context_len", 12)
        self.lambda_context = getattr(configs, "lambda_context", 0.00001)

        crop_init_ratio = getattr(configs, "crop_init_ratio", 0.02)
        crop_init_ratio = min(max(float(crop_init_ratio), 1e-4), 1.0 - 1e-4)
        crop_init_logit = math.log(crop_init_ratio / (1.0 - crop_init_ratio))

        self.crop_gate_logit = nn.Parameter(
            torch.tensor(crop_init_logit, dtype=torch.float32)
        )

        self.normalize_adaptive_mask = getattr(configs, "normalize_adaptive_mask", False)
        self.mask_norm_eps = getattr(configs, "mask_norm_eps", 1e-4)
        self.mask_norm_clip = getattr(configs, "mask_norm_clip", 3.0)

        boundary_hidden = getattr(configs, "boundary_hidden", 64)
        context_tau = getattr(configs, "context_tau", 12.0)

        if self.use_adaptive_l:
            self.length_predictor = AdaptiveLengthPredictor(
                seq_len=self.seq_len,
                channels=self.enc_in,
                hidden_dim=boundary_hidden,
                dropout=configs.dropout,
                min_len=self.min_context_len
            )

            self.length_mask = AdaptiveLengthMask(
                init_tau=context_tau
            )

        # =====================================================
        # Cycle refinement configs
        # =====================================================
        self.use_cycle_refine = getattr(configs, "use_cycle_refine", True)
        self.cycle_len = getattr(configs, "cycle_len", 24)
        self.cycle_refine_ratio = getattr(configs, "cycle_refine_ratio", 0.03)
        self.cycle_hidden_mult = getattr(configs, "cycle_hidden_mult", 2)

        # =====================================================
        # Load Q input matrix
        # =====================================================
        q_mat_dir = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file

        if not os.path.isfile(q_mat_dir):
            q_mat_dir = os.path.join(configs.root_path, q_mat_dir)

        assert os.path.isfile(q_mat_dir), f"Q matrix file not found: {q_mat_dir}"

        Q_mat_np = np.load(q_mat_dir)

        if Q_mat_np.ndim == 1:
            raise ValueError(
                f"Loaded Q_mat is 1D: shape={Q_mat_np.shape}. "
                f"You probably loaded an eigenvalue file such as *_eig_ratio*.npy. "
                f"Please use Q matrix file like ETTh1_{self.seq_len}_ratio0.6.npy."
            )

        Q_mat = torch.from_numpy(Q_mat_np).to(torch.float32)

        if self.Q_chan_indep:
            assert Q_mat.ndim == 3, \
                f"Q_chan_indep=True requires Q_mat shape [N, L, L], but got {Q_mat.shape}"

            assert (
                Q_mat.shape[0] == self.enc_in
                and Q_mat.shape[1] == self.seq_len
                and Q_mat.shape[2] == self.seq_len
            ), f"Invalid Q_mat shape: {Q_mat.shape}, expected [{self.enc_in}, {self.seq_len}, {self.seq_len}]"
        else:
            assert Q_mat.ndim == 2, \
                f"Q_chan_indep=False requires Q_mat shape [L, L], but got {Q_mat.shape}"

            assert (
                Q_mat.shape[0] == self.seq_len
                and Q_mat.shape[1] == self.seq_len
            ), f"Invalid Q_mat shape: {Q_mat.shape}, expected [{self.seq_len}, {self.seq_len}]"

        # =====================================================
        # Load Q output matrix
        # =====================================================
        q_out_mat_dir = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file

        if not os.path.isfile(q_out_mat_dir):
            q_out_mat_dir = os.path.join(configs.root_path, q_out_mat_dir)

        assert os.path.isfile(q_out_mat_dir), f"Q output matrix file not found: {q_out_mat_dir}"

        Q_out_mat_np = np.load(q_out_mat_dir)

        if Q_out_mat_np.ndim == 1:
            raise ValueError(
                f"Loaded Q_out_mat is 1D: shape={Q_out_mat_np.shape}. "
                f"You probably loaded an eigenvalue file such as *_eig_ratio*.npy. "
                f"Please use Q matrix file like ETTh1_{self.pred_len}_ratio0.6.npy."
            )

        Q_out_mat = torch.from_numpy(Q_out_mat_np).to(torch.float32)

        if self.Q_chan_indep:
            assert Q_out_mat.ndim == 3, \
                f"Q_chan_indep=True requires Q_out_mat shape [N, pred_len, pred_len], but got {Q_out_mat.shape}"

            assert (
                Q_out_mat.shape[0] == self.enc_in
                and Q_out_mat.shape[1] == self.pred_len
                and Q_out_mat.shape[2] == self.pred_len
            ), f"Invalid Q_out_mat shape: {Q_out_mat.shape}, expected [{self.enc_in}, {self.pred_len}, {self.pred_len}]"
        else:
            assert Q_out_mat.ndim == 2, \
                f"Q_chan_indep=False requires Q_out_mat shape [pred_len, pred_len], but got {Q_out_mat.shape}"

            assert (
                Q_out_mat.shape[0] == self.pred_len
                and Q_out_mat.shape[1] == self.pred_len
            ), f"Invalid Q_out_mat shape: {Q_out_mat.shape}, expected [{self.pred_len}, {self.pred_len}]"

        self.register_buffer("Q_mat", Q_mat)
        self.register_buffer("Q_out_mat", Q_out_mat)

        # =====================================================
        # Dimension extension
        # =====================================================
        self.embed_size = configs.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        # =====================================================
        # Original OLinear representation learner
        # =====================================================
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    CovMat=None,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    token_num=self.enc_in,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )

        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size)
        )

        # =====================================================
        # Final projection
        # =====================================================
        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # =====================================================
        # Cycle-aware refinement
        # =====================================================
        if self.use_cycle_refine:
            self.cycle_refiner = CycleAwareRefinement(
                pred_len=self.pred_len,
                channels=self.enc_in,
                cycle_len=self.cycle_len,
                hidden_mult=self.cycle_hidden_mult,
                dropout=configs.dropout,
                init_ratio=self.cycle_refine_ratio
            )
        else:
            self.cycle_refiner = None

        # =====================================================
        # Horizon-conditioned multi-anchor calibration (HMAC)
        # =====================================================
        self.use_horizon_calibration = bool(
            getattr(configs, "use_horizon_calibration", 0)
        )
        if self.use_horizon_calibration:
            self.horizon_calibrator = HorizonConditionedMultiAnchor(
                pred_len=self.pred_len,
                channels=self.enc_in,
                cycle_len=getattr(configs, "horizon_calibration_cycle", 24),
                hidden_dim=getattr(configs, "horizon_calibration_hidden", 32),
                horizon_rank=getattr(configs, "horizon_calibration_rank", 8),
                dropout=configs.dropout,
                init_ratio=getattr(configs, "horizon_calibration_ratio", 0.01),
            )
        else:
            self.horizon_calibrator = None

        # =====================================================
        # RevIN and dropout
        # =====================================================
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # =====================================================
        # Learnable delta, original design retained
        # =====================================================
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

        # cache for logging
        self.last_context_boundary = None
        self.last_context_mask = None

    # =====================================================
    # Adaptive L Cropping
    # =====================================================
    def apply_adaptive_l_crop(self, x):
        """
        x:
            [B, T, N]

        return:
            x_eff:    [B, T, N]
            boundary: [B, N]
            mask:     [B, T, N]
        """
        B, T, N = x.shape

        if not self.use_adaptive_l:
            boundary = torch.full(
                (B, N),
                float(T),
                device=x.device,
                dtype=x.dtype
            )
            mask = torch.ones_like(x)
            return x, boundary, mask

        boundary = self.length_predictor(x)
        mask = self.length_mask(boundary, T)

        if self.normalize_adaptive_mask:
            mask_mean = mask.mean(dim=1, keepdim=True)
            mask_norm = mask / (mask_mean + self.mask_norm_eps)
            mask_norm = torch.clamp(mask_norm, max=self.mask_norm_clip)
        else:
            mask_norm = mask

        x_crop = x * mask_norm

        crop_gate = torch.sigmoid(self.crop_gate_logit)
        x_eff = x + crop_gate * (x_crop - x)

        return x_eff, boundary, mask

    # =====================================================
    # Dimension extension
    # =====================================================
    def tokenEmb(self, x, embeddings):
        """
        Input:
            x: [B, T, N]

        Output:
            [B, N, T, D]
        """
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)

        x = x.transpose(-1, -2)  # [B, N, T]
        x = x.unsqueeze(-1)      # [B, N, T, 1]

        return x * embeddings    # [B, N, T, D]

    # =====================================================
    # Fixed OrthoTrans
    # =====================================================
    def Fre_Trans(self, x):
        """
        Input:
            x: [B, N, T, D]

        Output:
            x: [B, N, pred_len, D]
        """
        B, N, T, D = x.shape

        assert T == self.seq_len, \
            f"Expected seq_len={self.seq_len}, got T={T}"

        # [B, N, T, D] -> [B, N, D, T]
        x = x.transpose(-1, -2).contiguous()

        # =================================================
        # Fixed input OrthoTrans
        # =================================================
        if self.Q_chan_indep:
            x_trans = torch.einsum(
                "bndt,ntv->bndv",
                x,
                self.Q_mat.transpose(-1, -2)
            )
        else:
            x_trans = torch.einsum(
                "bndt,tv->bndv",
                x,
                self.Q_mat.transpose(-1, -2)
            )

        x_trans = x_trans + self.delta1

        assert x_trans.shape[-1] == self.seq_len

        # =================================================
        # Representation learner in transformed domain
        # [B, N, D, T] -> [B, N, D*T]
        # =================================================
        x_trans = self.ortho_trans(
            x_trans.flatten(-2)
        ).reshape(B, N, D, self.pred_len)

        # =================================================
        # Fixed output OrthoTrans
        # =================================================
        if self.Q_chan_indep:
            x_out = torch.einsum(
                "bndt,ntv->bndv",
                x_trans,
                self.Q_out_mat
            )
        else:
            x_out = torch.einsum(
                "bndt,tv->bndv",
                x_trans,
                self.Q_out_mat
            )

        x_out = x_out + self.delta2

        # [B, N, D, pred_len] -> [B, N, pred_len, D]
        x_out = x_out.transpose(-1, -2).contiguous()

        return x_out

    # =====================================================
    # Aux loss
    # =====================================================
    def get_aux_loss(self):
        """
        Only Adaptive L auxiliary loss is kept.

        Default:
            lambda_context = 0.0
        """
        total_loss = torch.tensor(0.0, device=self.delta1.device)

        if (
            self.use_adaptive_l
            and self.last_context_boundary is not None
            and self.lambda_context != 0
        ):
            context_loss = self.last_context_boundary.mean() / float(self.seq_len)
            total_loss = total_loss + self.lambda_context * context_loss

        return total_loss

    # =====================================================
    # Logging helpers
    # =====================================================
    def get_context_boundary(self):
        if self.last_context_boundary is None:
            return None
        return self.last_context_boundary.detach()

    def get_context_mask(self):
        return self.last_context_mask

    def get_cycle_gate(self):
        if self.cycle_refiner is None:
            return None
        return self.cycle_refiner.get_gate()

    def get_horizon_calibration_strength(self):
        if self.horizon_calibrator is None:
            return None
        return self.horizon_calibrator.get_strength()

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Input:
            x: [B, T, N]

        Output:
            out: [B, pred_len, N]
        """
        B, T, N = x.shape

        assert T == self.seq_len, \
            f"Expected seq_len={self.seq_len}, but got T={T}"

        assert N == self.enc_in, \
            f"Expected enc_in={self.enc_in}, but got N={N}"

        # =================================================
        # RevIN norm
        # =================================================
        x = self.revin_layer(x, mode="norm")
        x_ori = x

        # =================================================
        # Optional Adaptive L cropping
        # Default disabled.
        # =================================================
        x_crop, boundary, l_mask = self.apply_adaptive_l_crop(x_ori)

        self.last_context_boundary = boundary
        self.last_context_mask = l_mask.detach()

        # =================================================
        # Token embedding
        # =================================================
        x = self.tokenEmb(x_crop, self.embeddings)

        # =================================================
        # Fixed OLinear OrthoTrans
        # =================================================
        x = self.Fre_Trans(x)

        # =================================================
        # Final linear head
        # [B, N, pred_len, D] -> [B, N, pred_len] -> [B, pred_len, N]
        # =================================================
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # =================================================
        # Cycle-aware refinement before RevIN denorm
        # x_ori and out are both in normalized space.
        # =================================================
        if self.cycle_refiner is not None:
            out = self.cycle_refiner(out, x_ori)

        if self.horizon_calibrator is not None:
            out = self.horizon_calibrator(out, x_ori)

        out = self.dropout(out)

        # =================================================
        # RevIN denorm
        # =================================================
        out = self.revin_layer(out, mode="denorm")

        return out
