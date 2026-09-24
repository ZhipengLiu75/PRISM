"""SHIFT: hindsight-guided causal hypothesis filtering.

SHIFT generates a bank of complete future hypotheses.  During training a
hindsight teacher observes the full target and defines a soft posterior over
the hypotheses.  Causal students only consume revealed target blocks and are
distilled toward that teacher.  At inference, each block is emitted before its
truth is used to update the posterior for later blocks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiScaleDecompositionBackbone(nn.Module):
    """Independent multi-resolution forecaster used by SHIFT.

    The backbone combines a smooth trend projection, several shared causal
    scale filters, a periodic anchor and a linear-complexity cross-channel
    factor mixer.  It deliberately has no dependency on OLinear.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.cycle_len = min(
            int(getattr(configs, 'shift_backbone_cycle', 24)), self.seq_len
        )
        self.factor_rank = int(
            getattr(configs, 'shift_backbone_factor_rank', 8)
        )
        hidden = int(getattr(configs, 'shift_backbone_hidden', 64))
        dropout = float(getattr(configs, 'shift_backbone_dropout', 0.1))

        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)
        self.scale_kernels = (3, 7, 15, 25)
        self.scale_filters = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in self.scale_kernels
        ])
        self.scale_projections = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len)
            for _ in self.scale_kernels
        ])
        self.branch_gate = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 + len(self.scale_kernels)),
        )

        # Dynamic low-rank channel interaction is O(CR), not O(C^2), so the
        # same architecture remains practical on Traffic with 862 variables.
        self.factor_score = nn.Linear(6, self.factor_rank)
        self.factor_loading = nn.Linear(6, self.factor_rank)
        self.channel_gate = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.channel_mix_logit = nn.Parameter(torch.tensor(-2.0))
        self.output_refine = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.refine_logit = nn.Parameter(torch.tensor(-2.0))

        self._initialize_filters()

    def _initialize_filters(self):
        # Start each convolution as a stable moving average at its own scale.
        with torch.no_grad():
            for conv, kernel in zip(self.scale_filters, self.scale_kernels):
                conv.weight.fill_(1.0 / float(kernel))

    @staticmethod
    def _stats(x):
        mean = x.mean(1)
        std = x.std(1, unbiased=False)
        last = x[:, -1]
        slope = x[:, -1] - x[:, 0]
        energy = x.square().mean(1).sqrt()
        if x.size(1) > 1:
            roughness = (x[:, 1:] - x[:, :-1]).abs().mean(1)
        else:
            roughness = torch.zeros_like(last)
        return torch.stack([mean, std, last, slope, energy, roughness], dim=-1)

    def _moving_trend(self, x):
        kernel = min(25, self.seq_len)
        if kernel % 2 == 0:
            kernel -= 1
        pad = kernel // 2
        xc = x.transpose(1, 2)
        xc = F.pad(xc, (pad, pad), mode='replicate')
        return F.avg_pool1d(xc, kernel_size=kernel, stride=1).transpose(1, 2)

    def forward(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        xn = (x - mean) / std
        stats = self._stats(xn)  # [B,C,6]

        trend = self._moving_trend(xn)
        seasonal = xn - trend
        branches = [
            self.trend_projection(trend.transpose(1, 2)).transpose(1, 2)
        ]

        seasonal_ci = seasonal.transpose(1, 2).reshape(-1, 1, self.seq_len)
        for conv, projection in zip(self.scale_filters, self.scale_projections):
            filtered = conv(seasonal_ci).reshape(
                x.size(0), self.channels, self.seq_len
            )
            branches.append(projection(filtered).transpose(1, 2))

        recent = xn[:, -self.cycle_len:]
        repeats = math.ceil(self.pred_len / self.cycle_len)
        cycle = recent.repeat(1, repeats, 1)[:, :self.pred_len]
        branches.append(cycle)

        branch_stack = torch.stack(branches, dim=-1)  # [B,H,C,S]
        weights = torch.softmax(self.branch_gate(stats), dim=-1)
        forecast = (branch_stack * weights[:, None]).sum(-1)

        score = torch.softmax(self.factor_score(stats), dim=1)
        loading = torch.tanh(self.factor_loading(stats))
        factors = torch.einsum('bcr,bhc->bhr', score, forecast)
        channel_correction = torch.einsum(
            'bhr,bcr->bhc', factors, loading
        ) / math.sqrt(float(self.factor_rank))
        channel_gate = torch.sigmoid(self.channel_gate(stats)).transpose(1, 2)
        forecast = forecast + (
            torch.sigmoid(self.channel_mix_logit)
            * channel_gate
            * channel_correction
        )

        refine_input = torch.stack([forecast, cycle, forecast - cycle], dim=-1)
        refinement = self.output_refine(refine_input).squeeze(-1)
        forecast = forecast + torch.sigmoid(self.refine_logit) * refinement
        return forecast * std + mean


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.block_size = int(getattr(configs, 'shift_block_size', 96))
        self.num_scenarios = int(getattr(configs, 'shift_scenarios', 8))
        self.rank = int(getattr(configs, 'shift_rank', 8))
        self.hidden = int(getattr(configs, 'shift_hidden', 64))
        self.embedding_dim = int(getattr(configs, 'shift_embedding_dim', 16))
        self.teacher_temperature = float(
            getattr(configs, 'shift_teacher_temperature', 0.1))
        self.posterior_temperature = float(
            getattr(configs, 'shift_posterior_temperature', 1.0))
        self.posterior_gamma = float(
            getattr(configs, 'shift_posterior_gamma', 1.0))

        self.lambda_coverage = float(
            getattr(configs, 'shift_lambda_coverage', 0.2))
        self.lambda_bridge = float(
            getattr(configs, 'shift_lambda_bridge', 1.0))
        self.lambda_contract = float(
            getattr(configs, 'shift_lambda_contract', 0.1))
        self.lambda_balance = float(
            getattr(configs, 'shift_lambda_balance', 0.01))
        self.lambda_diversity = float(
            getattr(configs, 'shift_lambda_diversity', 0.01))
        self.lambda_progressive = float(
            getattr(configs, 'shift_lambda_progressive', 1.0))
        self.lambda_monotonic = float(
            getattr(configs, 'shift_lambda_monotonic', 0.1))

        self.backbone = MultiScaleDecompositionBackbone(configs)

        self.scenario_embedding = nn.Parameter(
            torch.randn(self.num_scenarios, self.embedding_dim) * 0.02
        )
        self.register_buffer(
            'position_features', self._make_position_features(), persistent=False
        )

        position_dim = self.position_features.size(-1)
        self.shape_net = nn.Sequential(
            nn.Linear(position_dim + self.embedding_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.rank),
        )
        self.coefficient_net = nn.Sequential(
            nn.Linear(5 + self.embedding_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.rank),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(7, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.num_scenarios),
        )

        evidence_dim = 6 + self.embedding_dim
        self.evidence_net = nn.Sequential(
            nn.Linear(evidence_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.evidence_net[-1].weight)
        nn.init.zeros_(self.evidence_net[-1].bias)

        # A causal state assimilator changes the hypotheses themselves after a
        # block is revealed.  This is deliberately different from merely
        # reweighting a fixed scenario bank: new sample-specific innovations
        # can alter every still-unreleased point of every hypothesis.
        self.feedback_encoder = nn.Sequential(
            nn.Linear(4 + self.embedding_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.feedback_cell = nn.GRUCell(self.hidden, self.hidden)
        self.feedback_decoder = nn.Sequential(
            nn.Linear(self.hidden + self.embedding_dim + 3, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.feedback_decoder[-1].weight)
        nn.init.zeros_(self.feedback_decoder[-1].bias)
        self.assimilation_logit = nn.Parameter(torch.tensor(-1.0))
        self.log_observation_scale = nn.Parameter(torch.tensor(-2.0))

        init_scale = float(getattr(configs, 'shift_scenario_scale', 0.1))
        init_scale = min(max(init_scale, 1e-4), 1.0 - 1e-4)
        self.scenario_scale_logit = nn.Parameter(torch.tensor(
            math.log(init_scale / (1.0 - init_scale)), dtype=torch.float32
        ))

        self._last_aux_loss = None
        self._last_stats = {}

    def _make_position_features(self):
        t = torch.linspace(0.0, 1.0, self.pred_len)
        features = [2.0 * t - 1.0]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            features.extend([
                torch.sin(2.0 * math.pi * frequency * t),
                torch.cos(2.0 * math.pi * frequency * t),
            ])
        return torch.stack(features, dim=-1)

    @staticmethod
    def _history_features(history):
        mean = history.mean(1)
        std = history.std(1, unbiased=False).clamp_min(1e-4)
        last = history[:, -1]
        trend = torch.tanh(history[:, -1] - history[:, 0])
        if history.size(1) > 1:
            velocity = torch.tanh(
                (history[:, 1:] - history[:, :-1]).mean(1)
            )
        else:
            velocity = torch.zeros_like(last)
        return torch.stack([mean, std, last, trend, velocity], dim=-1)

    def build_scenarios(self, history, base):
        """Return scenarios [B,M,H,C] and the initial posterior [B,M]."""
        batch = history.size(0)
        channel_features = self._history_features(history)  # [B,C,5]

        pos = self.position_features.to(base)
        pos = pos[None, :, :].expand(self.num_scenarios, -1, -1)
        emb_h = self.scenario_embedding[:, None, :].expand(
            -1, self.pred_len, -1
        )
        shapes = self.shape_net(torch.cat([pos, emb_h], dim=-1))
        shapes = shapes / shapes.square().mean(1, keepdim=True).sqrt().clamp_min(1e-4)

        feat = channel_features[:, :, None, :].expand(
            -1, -1, self.num_scenarios, -1
        )
        emb_c = self.scenario_embedding[None, None, :, :].expand(
            batch, self.enc_in, -1, -1
        )
        coefficients = torch.tanh(
            self.coefficient_net(torch.cat([feat, emb_c], dim=-1))
        )
        residual = torch.einsum('mhr,bcmr->bmhc', shapes, coefficients)
        channel_scale = channel_features[..., 1][:, None, None, :]
        strength = torch.sigmoid(self.scenario_scale_logit)
        scenarios = base[:, None, :, :] + strength * channel_scale * residual

        base_mean = base.mean((1, 2))
        base_std = base.std((1, 2), unbiased=False)
        global_history = channel_features.mean(1)
        prior_features = torch.cat([
            global_history, base_mean[:, None], base_std[:, None]
        ], dim=-1)
        prior = torch.softmax(self.prior_net(prior_features), dim=-1)
        return scenarios, prior

    def teacher_posterior(self, scenarios, target):
        energy = (scenarios - target[:, None]).square().mean((2, 3))
        posterior = torch.softmax(
            -energy / max(self.teacher_temperature, 1e-4), dim=-1
        )
        return posterior, energy

    def evidence_features(self, residual, progress, posterior):
        """Summarize a newly revealed block without using later truth."""
        abs_error = residual.abs().mean((2, 3))
        mse = residual.square().mean((2, 3))
        bias = residual.mean((2, 3))
        channel_consistency = residual.mean(2).std(-1, unbiased=False)
        if residual.size(2) > 1:
            roughness = (
                residual[:, :, 1:] - residual[:, :, :-1]
            ).abs().mean((2, 3))
        else:
            roughness = torch.zeros_like(abs_error)
        progress_feature = torch.full_like(abs_error, float(progress))
        return torch.stack([
            abs_error, mse, bias, channel_consistency, roughness,
            progress_feature,
        ], dim=-1)

    def _assimilate(self, scenarios, state, residual, start, end):
        """Update latent hypothesis states and redraw only unrevealed points."""
        batch, num_scenarios, _, channels = scenarios.shape
        block_mean = residual.mean(2)
        block_last = residual[:, :, -1]
        block_slope = residual[:, :, -1] - residual[:, :, 0]
        block_std = residual.std(2, unbiased=False)
        summary = torch.stack(
            [block_mean, block_last, block_slope, block_std], dim=-1
        )  # [B,M,C,4]
        emb = self.scenario_embedding[None, :, None, :].expand(
            batch, -1, channels, -1
        )
        encoded = self.feedback_encoder(torch.cat([summary, emb], dim=-1))
        state = self.feedback_cell(
            encoded.reshape(-1, self.hidden), state.reshape(-1, self.hidden)
        ).reshape(batch, num_scenarios, channels, self.hidden)

        future_length = self.pred_len - end
        if future_length <= 0:
            return scenarios, state
        rel = torch.linspace(
            0.0, 1.0, future_length,
            device=scenarios.device, dtype=scenarios.dtype
        )
        rel_features = torch.stack([
            rel,
            torch.sin(2.0 * math.pi * rel),
            torch.cos(2.0 * math.pi * rel),
        ], dim=-1)
        state_f = state[:, :, None, :, :].expand(
            -1, -1, future_length, -1, -1
        )
        emb_f = self.scenario_embedding[None, :, None, None, :].expand(
            batch, -1, future_length, channels, -1
        )
        rel_f = rel_features[None, None, :, None, :].expand(
            batch, num_scenarios, -1, channels, -1
        )
        correction = self.feedback_decoder(
            torch.cat([state_f, emb_f, rel_f], dim=-1)
        ).squeeze(-1)
        correction = torch.sigmoid(self.assimilation_logit) * correction
        scenarios = torch.cat([
            scenarios[:, :, :end],
            scenarios[:, :, end:] + correction,
        ], dim=2)
        return scenarios, state

    def rollout(self, scenarios, prior, target):
        """Causal Bayesian filtering plus latent-state hypothesis assimilation."""
        batch, _, horizon, _ = scenarios.shape
        deployed = torch.empty(
            batch, horizon, self.enc_in,
            device=scenarios.device, dtype=scenarios.dtype
        )
        posterior = prior
        posteriors = [posterior]
        logits = torch.log(posterior.clamp_min(1e-8))
        scenario_state = scenarios.new_zeros(
            batch, self.num_scenarios, self.enc_in, self.hidden
        )
        working_scenarios = scenarios
        bridge_terms = []
        future_terms = []
        monotonic_terms = []

        for start in range(0, horizon, self.block_size):
            end = min(start + self.block_size, horizon)
            current = torch.einsum(
                'bm,bmhc->bhc', posterior, working_scenarios
            )
            deployed[:, start:end] = current[:, start:end]
            if end >= horizon:
                break

            residual = (
                target[:, None, start:end, :]
                - working_scenarios[:, :, start:end, :]
            )
            evidence = self.evidence_features(
                residual, progress=float(end) / float(horizon), posterior=posterior
            )
            emb = self.scenario_embedding[None, :, :].expand(batch, -1, -1)
            learned_delta = self.evidence_net(
                torch.cat([evidence, emb], dim=-1)
            ).squeeze(-1)
            block_energy = residual.square().mean((2, 3))
            observation_scale = F.softplus(self.log_observation_scale) + 1e-4
            # The likelihood term guarantees that revealed observations affect
            # the posterior even before the learned calibrator is well trained.
            delta = learned_delta - block_energy / observation_scale
            logits = self.posterior_gamma * logits + (
                delta / max(self.posterior_temperature, 1e-4)
            )
            posterior = torch.softmax(logits, dim=-1)
            posteriors.append(posterior)

            before = torch.einsum(
                'bm,bmhc->bhc', posterior, working_scenarios
            )[:, end:]
            working_scenarios, scenario_state = self._assimilate(
                working_scenarios, scenario_state, residual, start, end
            )
            after = torch.einsum(
                'bm,bmhc->bhc', posterior, working_scenarios
            )[:, end:]
            future_terms.append(F.mse_loss(after, target[:, end:]))
            monotonic_terms.append(F.relu(
                F.mse_loss(after, target[:, end:])
                - F.mse_loss(before, target[:, end:])
            ))

            # Stage-specific oracle: after the same feedback, which hypothesis
            # best explains only the still-unseen future?
            oracle_energy = (
                working_scenarios[:, :, end:] - target[:, None, end:]
            ).square().mean((2, 3))
            oracle = torch.softmax(
                -oracle_energy / max(self.teacher_temperature, 1e-4), dim=-1
            ).detach().clamp_min(1e-8)
            bridge_terms.append((oracle * (
                oracle.log() - posterior.clamp_min(1e-8).log()
            )).sum(-1).mean())

        zero = deployed.sum() * 0.0
        progressive = torch.stack(future_terms).mean() if future_terms else zero
        monotonic = torch.stack(monotonic_terms).mean() if monotonic_terms else zero
        bridge = torch.stack(bridge_terms).mean() if bridge_terms else zero
        return deployed, posteriors, progressive, monotonic, bridge

    def loss_components(self, scenarios, prior, target):
        teacher, energy = self.teacher_posterior(scenarios, target)
        deployed, posteriors, progressive, monotonic, stage_bridge = self.rollout(
            scenarios, prior, target
        )
        teacher_target = teacher.detach().clamp_min(1e-8)

        bridge_terms = []
        # The history-only prior cannot identify unpredictable future errors;
        # distillation begins only after causal evidence has arrived.
        for posterior in posteriors[1:]:
            bridge_terms.append((
                teacher_target * (
                    teacher_target.log() - posterior.clamp_min(1e-8).log()
                )
            ).sum(-1).mean())
        bridge = (
            0.5 * torch.stack(bridge_terms).mean() + 0.5 * stage_bridge
            if bridge_terms else stage_bridge
        )

        contraction_terms = []
        for previous, current in zip(posteriors[:-1], posteriors[1:]):
            previous_entropy = -(
                previous * previous.clamp_min(1e-8).log()
            ).sum(-1)
            current_entropy = -(
                current * current.clamp_min(1e-8).log()
            ).sum(-1)
            contraction_terms.append(
                F.relu(current_entropy - previous_entropy).mean()
            )
        contraction = (
            torch.stack(contraction_terms).mean()
            if contraction_terms else bridge * 0.0
        )

        coverage = energy.min(-1).values.mean()
        mean_teacher = teacher.mean(0).clamp_min(1e-8)
        balance = (mean_teacher * (
            mean_teacher.log() + math.log(self.num_scenarios)
        )).sum()

        deviations = scenarios - scenarios.mean(1, keepdim=True)
        deviations = F.normalize(deviations.flatten(2), dim=-1, eps=1e-6)
        gram = torch.bmm(deviations, deviations.transpose(1, 2))
        eye = torch.eye(
            self.num_scenarios, device=gram.device, dtype=torch.bool
        )[None]
        diversity = gram.masked_select(~eye).square().mean()

        initial = torch.einsum('bm,bmhc->bhc', prior, scenarios)
        initial_mse = F.mse_loss(initial, target)
        deployed_mse = F.mse_loss(deployed, target)
        aux = (
            self.lambda_progressive * progressive
            + self.lambda_monotonic * monotonic
            + self.lambda_coverage * coverage
            + self.lambda_bridge * bridge
            + self.lambda_contract * contraction
            + self.lambda_balance * balance
            + self.lambda_diversity * diversity
        )
        stats = {
            'initial_mse': initial_mse,
            'deployed_mse': deployed_mse,
            'progressive': progressive,
            'monotonic': monotonic,
            'coverage': coverage,
            'bridge': bridge,
            'contraction': contraction,
            'balance': balance,
            'diversity': diversity,
            'teacher_entropy': -(
                teacher * teacher.clamp_min(1e-8).log()
            ).sum(-1).mean(),
            'student_entropy': -(
                posteriors[-1] * posteriors[-1].clamp_min(1e-8).log()
            ).sum(-1).mean(),
        }
        return initial, deployed, aux, stats

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,
                target=None, progressive=False):
        base = self.backbone(x)
        scenarios, prior = self.build_scenarios(x, base)
        initial = torch.einsum('bm,bmhc->bhc', prior, scenarios)
        self._last_aux_loss = None

        if target is None:
            return initial

        target = target[:, -self.pred_len:, :]
        initial, deployed, aux, stats = self.loss_components(
            scenarios, prior, target
        )
        self._last_stats = {key: value.detach() for key, value in stats.items()}
        if self.training:
            self._last_aux_loss = aux
        return deployed if progressive else initial

    def get_aux_loss(self):
        if self._last_aux_loss is None:
            return self.scenario_scale_logit.sum() * 0.0
        return self._last_aux_loss

    def get_shift_stats(self):
        return self._last_stats
