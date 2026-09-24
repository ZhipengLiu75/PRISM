"""PRISM: Progressive Re-encoding with Innovation-State Memory.

Every revealed target block is appended to a rolling input window.  The shared
backbone then forecasts again from the new time origin, while a recurrent
innovation state decides how much of the re-forecast should replace the
aligned previous trajectory.  A block is always emitted before its truth is
used, so the rollout is strictly causal.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SHIFT import MultiScaleDecompositionBackbone


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.block_size = int(getattr(configs, 'prism_block_size', 96))
        self.hidden = int(getattr(configs, 'prism_hidden', 64))
        self.lambda_progressive = float(
            getattr(configs, 'prism_lambda_progressive', 1.0)
        )
        self.modulation_scale = float(
            getattr(configs, 'prism_modulation_scale', 0.1)
        )

        backbone_configs = copy.copy(configs)
        backbone_configs.pred_len = self.pred_len
        self.backbone = MultiScaleDecompositionBackbone(backbone_configs)

        # Per-channel feedback token: normalized revealed values, innovations,
        # and four innovation statistics.  Padding makes the representation
        # valid when the last feedback block is shorter than block_size.
        feedback_dim = 2 * self.block_size + 4
        self.innovation_encoder = nn.Sequential(
            nn.Linear(feedback_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.memory_cell = nn.GRUCell(self.hidden, self.hidden)

        # The transition gate compares a fresh forecast from the progressively
        # updated input with the aligned tail of the previous forecast.
        transition_dim = self.hidden + 5
        self.transition_gate = nn.Sequential(
            nn.Linear(transition_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.transition_gate[-1].weight)
        nn.init.constant_(self.transition_gate[-1].bias, 1.5)

        # Horizon-aware FiLM modulation reads the innovation memory.  Zero
        # initialization makes the initial model exactly the re-forecasting
        # path and lets feedback modulation enter only when useful.
        self.memory_decoder = nn.Sequential(
            nn.Linear(self.hidden + 4, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 2),
        )
        nn.init.zeros_(self.memory_decoder[-1].weight)
        nn.init.zeros_(self.memory_decoder[-1].bias)

        self._last_aux_loss = None
        self._last_stats = {}

    @staticmethod
    def _channel_scale(history):
        return history.std(1, unbiased=False).clamp_min(1e-4)

    def _update_memory(self, revealed, innovation, history, state):
        scale = self._channel_scale(history)[:, :, None]
        revealed_n = revealed.permute(0, 2, 1) / scale
        innovation_n = innovation.permute(0, 2, 1) / scale
        padding = self.block_size - revealed_n.size(-1)
        if padding > 0:
            revealed_n = F.pad(revealed_n, (0, padding))
            innovation_n = F.pad(innovation_n, (0, padding))

        mean = innovation_n.mean(-1, keepdim=True)
        last = innovation_n[..., -1:]
        slope = innovation_n[..., -1:] - innovation_n[..., :1]
        spread = innovation_n.std(-1, unbiased=False, keepdim=True)
        token = torch.cat([
            revealed_n.clamp(-8.0, 8.0),
            innovation_n.clamp(-8.0, 8.0),
            mean, last, slope, spread,
        ], dim=-1)
        encoded = self.innovation_encoder(token)
        batch, channels, _ = encoded.shape
        return self.memory_cell(
            encoded.reshape(-1, self.hidden),
            state.reshape(-1, self.hidden),
        ).reshape(batch, channels, self.hidden)

    @staticmethod
    def _position_features(length, progress, reference):
        rel = torch.linspace(
            0.0, 1.0, length, device=reference.device,
            dtype=reference.dtype,
        )
        return torch.stack([
            rel,
            torch.sin(2.0 * math.pi * rel),
            torch.cos(2.0 * math.pi * rel),
            torch.full_like(rel, float(progress)),
        ], dim=-1)

    def _transition(self, fresh, previous_tail, memory, history, progress):
        length = fresh.size(1)
        position = self._position_features(length, progress, fresh)
        batch, _, channels = fresh.shape
        state_f = memory[:, None].expand(-1, length, -1, -1)
        pos_f = position[None, :, None].expand(batch, -1, channels, -1)
        scale = self._channel_scale(history)[:, None, :]
        disagreement = ((fresh - previous_tail) / scale).clamp(-8.0, 8.0)

        gate_features = torch.cat([
            state_f,
            pos_f,
            disagreement[..., None],
        ], dim=-1)
        gate = torch.sigmoid(self.transition_gate(gate_features)).squeeze(-1)
        fused = gate * fresh + (1.0 - gate) * previous_tail

        film = self.memory_decoder(torch.cat([state_f, pos_f], dim=-1))
        gain = 1.0 + self.modulation_scale * torch.tanh(film[..., 0])
        bias = self.modulation_scale * scale * torch.tanh(film[..., 1])
        return gain * fused + bias, gate

    def rollout(self, history, target):
        batch = history.size(0)
        target = target[:, -self.pred_len:]
        rolling_history = history
        memory = history.new_zeros(batch, self.enc_in, self.hidden)

        forecast = self.backbone(rolling_history)
        initial = forecast
        deployed = torch.empty_like(target)
        future_losses = []
        gate_means = []
        input_updates = 0

        boundary = 0
        while boundary < self.pred_len:
            block = min(self.block_size, self.pred_len - boundary)
            # Release first; only then reveal the matching target block.
            deployed[:, boundary:boundary + block] = forecast[:, :block]
            next_boundary = boundary + block
            if next_boundary >= self.pred_len:
                break

            revealed = target[:, boundary:next_boundary]
            innovation = revealed - forecast[:, :block]
            rolling_history = torch.cat([rolling_history, revealed], dim=1)
            rolling_history = rolling_history[:, -self.seq_len:]
            memory = self._update_memory(
                revealed, innovation, rolling_history, memory
            )

            # The fresh output is indexed from the new time origin.  Its prefix
            # aligns with the unreleased tail of the previous stage forecast.
            fresh_full = self.backbone(rolling_history)
            remaining = self.pred_len - next_boundary
            fresh = fresh_full[:, :remaining]
            previous_tail = forecast[:, block:block + remaining]
            forecast, gate = self._transition(
                fresh, previous_tail, memory, rolling_history,
                progress=float(next_boundary) / float(self.pred_len),
            )
            future_target = target[:, next_boundary:]
            future_losses.append(F.mse_loss(forecast, future_target))
            gate_means.append(gate.mean())
            input_updates += 1
            boundary = next_boundary

        zero = deployed.sum() * 0.0
        progressive = (
            torch.stack(future_losses).mean() if future_losses else zero
        )
        mean_gate = torch.stack(gate_means).mean() if gate_means else zero
        return initial, deployed, progressive, mean_gate, input_updates

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, target=None, progressive=False):
        self._last_aux_loss = None
        if target is None:
            return self.backbone(x)

        initial, deployed, prog, gate, input_updates = self.rollout(x, target)
        target = target[:, -self.pred_len:]
        self._last_stats = {
            'initial_mse': F.mse_loss(initial, target).detach(),
            'deployed_mse': F.mse_loss(deployed, target).detach(),
            'progressive': prog.detach(),
            'transition_gate': gate.detach(),
            'input_updates': target.new_tensor(float(input_updates)),
        }
        if self.training:
            self._last_aux_loss = self.lambda_progressive * prog
        return deployed if progressive else initial

    def get_aux_loss(self):
        if self._last_aux_loss is None:
            return next(self.parameters()).sum() * 0.0
        return self._last_aux_loss

    def get_prism_stats(self):
        return self._last_stats
