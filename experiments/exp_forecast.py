from torch.optim import lr_scheduler
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping, adjust_learning_rate, visual, write_into_xls,
    compute_gradient_norm, find_most_recently_modified_subfolder,
    compare_prefix_before_third_underscore, compute_weights
)
from utils.metrics import metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim

import os
import time
import warnings
import numpy as np
from typing import List
import random
import copy
import csv
from contextlib import contextmanager

from fvcore.nn import FlopCountAnalysis
import logging
import shutil

warnings.filterwarnings('ignore')


def compute_model_stats(model, args, num_iterations=50):
    assert num_iterations > 10, 'num_iterations should be greater than 10'
    if not args.model_stats_mode:
        print('No compute_model_stats because model_stats_mode is False!')
        return False

    logging.getLogger('fvcore').setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot measure GPU memory and timings.")
        return False

    device = torch.device("cuda")

    input_size = (1, args.seq_len, args.enc_in)
    inputs = torch.randn(input_size).to(device)

    model = model.to(device).eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters(M): {params:.3f}")

    flops = FlopCountAnalysis(model, inputs)
    flops = flops.total() / 1e6

    print(f"FLOPS(M): {flops:.3f}")

    if 'PEMS' in args.data:
        num_iterations = 15

    inputs = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)

    torch.cuda.reset_peak_memory_stats()
    inference_times = []

    for i in range(num_iterations):
        start_time = time.time()
        if args.use_amp:
            with torch.cuda.amp.autocast():
                _ = model(inputs)
        else:
            with torch.no_grad():
                _ = model(inputs)

        inference_times.append(time.time() - start_time)

    avg_inference_time = np.mean(inference_times[-10:])
    inference_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Inference Time / iter: {avg_inference_time * 1000:.3f} ms")
    print(f"Inference Memory Usage: {inference_memory:.3f} MB")

    criterion = WeightedL1Loss(args.lossfun_alpha, args.loss_mode)
    targets = torch.randn(args.batch_size, args.pred_len, args.enc_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    training_times = []
    torch.cuda.reset_peak_memory_stats()

    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for _ in range(num_iterations):
        model.train()
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        training_times.append(time.time() - start_time)

    avg_training_time = np.mean(training_times[-10:])
    training_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    with open('model_stats.txt', 'a') as f:
        f.write(f'============================ stats {args.model_id_ori}============================= \n')
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            f.write(f'{k}: {v}, ')
        f.write('\n\n')
        f.write(f"\tParameters(M): {params:.3f}\n")
        f.write(f"\tFLOPS(M): {flops:.3f}\n")
        f.write(f"\tTraining Time / iter: {avg_training_time * 1000:.3f} ms\n")
        f.write(f"\tTraining Memory Usage: {training_memory:.3f} MB\n")
        f.write(f"\tInference Time / iter: {avg_inference_time * 1000:.3f} ms\n")
        f.write(f"\tInference Memory Usage: {inference_memory:.2f} MB\n\n\n")

    best_log_dataset_path = 'best_results'
    if not os.path.exists(best_log_dataset_path):
        os.makedirs(best_log_dataset_path, exist_ok=True)

    best_log_dataset_txt = os.path.join(best_log_dataset_path, args.model_id_ori + '_stats.txt')

    with open(best_log_dataset_txt, 'a') as f:
        f.write(f'============================ stats {args.model_id_ori}============================= \n')
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            f.write(f'{k}: {v}, ')
        f.write('\n\n')
        f.write(f"\tParameters(M): {params:.3f}\n")
        f.write(f"\tFLOPS(M): {flops:.3f}\n")
        f.write(f"\tTraining Time / iter: {avg_training_time * 1000:.3f} ms\n")
        f.write(f"\tTraining Memory Usage: {training_memory:.3f} MB\n")
        f.write(f"\tInference Time / iter: {avg_inference_time * 1000:.3f} ms\n")
        f.write(f"\tInference Memory Usage: {inference_memory:.2f} MB\n\n\n")

    print(f"Training Time / iter: {avg_training_time * 1000:.3f} ms")
    print(f"Training Memory Usage: {training_memory:.3f} MB")

    return True


class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        self.alpha = alpha
        self.loss_mode = loss_mode

        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f"Unsupported loss_mode: {self.loss_mode}")

    def __call__(self, pred, gt):
        if pred.ndim == 1:
            mask = torch.isnan(gt)
            if torch.any(mask):
                pred, gt = pred[~mask], gt[~mask]

            loss_fun = nn.L1Loss(reduction='mean')
            weighted_loss = loss_fun(pred, gt)

        else:
            L = pred.shape[1]
            weights = torch.tensor(
                [(i + 1) ** (-self.alpha) for i in range(L)],
                dtype=pred.dtype,
                device=pred.device
            ).unsqueeze(dim=0).unsqueeze(dim=-1)

            if self.loss_mode in ['L1', 'L2']:
                loss_vec = self.loss_fun(pred, gt)
                weighted_loss = torch.mean(loss_vec * weights)

            elif self.loss_mode == 'L1L2':
                loss_vec1 = self.loss_fun1(pred, gt)
                loss_vec2 = self.loss_fun2(pred, gt)
                weighted_loss = torch.mean(loss_vec1 * weights + loss_vec2 * weights)

            else:
                raise NotImplementedError

        return weighted_loss


class ModelEMA:
    """
    Exponential Moving Average for model parameters.

    EMA:
        theta_ema = decay * theta_ema + (1 - decay) * theta

    This class supports normal model and DataParallel model because it copies
    the whole self.model object and updates by state_dict keys.
    """
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()

            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.ema.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        self.ema.to(device)
        return self


def _select_criterion():
    criterion = nn.L1Loss(reduction='mean')
    return criterion


def _select_mse_criterion():
    criterion = nn.MSELoss()
    return criterion


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.imp_mode = args.task_name == 'imputation'
        self.eval_flag = args.eval_flag
        self.resume_training = args.resume_training
        self.resume_epoch = args.resume_epoch
        self.folder_path = args.folder_path

        self.use_ema = bool(getattr(args, "use_ema", 0))
        self.ema_decay = float(getattr(args, "ema_decay", 0.999))
        self.ema_model = None

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path, exist_ok=True)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def compute_model_stats(self):
        if self.args.model_stats_mode:
            compute_model_stats(self.model, self.args)

    def save_linear_weight_2npy(self, setting=None):
        assert self.args.save_linear_weight

        full_folder, new_setting = find_most_recently_modified_subfolder(
            self.args.checkpoints,
            file_name='checkpoint.pth',
            contain_str=self.args.model_id_ori
        )

        if full_folder is not None and compare_prefix_before_third_underscore(setting, new_setting):
            print(f'loading model from {full_folder}')
            self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
        else:
            raise ValueError('No checkpoints are found...')

        self.model.eval()

        raw_model = self._get_raw_model()
        weight_mat = raw_model.encoder.attn_layers[0].weight_mat[0, ...]

        weight_mat = F.normalize(F.softplus(weight_mat), p=1, dim=-1)

        weight_numpy = weight_mat.detach().cpu().numpy()

        if not os.path.exists(self.args.save_linear_weight_path):
            os.makedirs(self.args.save_linear_weight_path)

        output_filename = os.path.join(
            self.args.save_linear_weight_path,
            f'{self.args.save_linear_weight_tag}_normlin_weight_after_norm.npy'
        )

        np.save(output_filename, weight_numpy)

        print(f"The weight is saved to '{output_filename}'")
        print(f"Weight shape: {weight_numpy.shape}")

        loaded_weights = np.load(output_filename)
        print("Weight (first 5 entries):", loaded_weights[0, :5])

    def _get_data(self, flag=None, test_batch_size=None):
        data_set, data_loader = data_provider(self.args, flag, test_batch_size=test_batch_size)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _get_raw_model(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _init_ema(self):
        if not self.use_ema:
            self.ema_model = None
            return

        self.ema_model = ModelEMA(self.model, decay=self.ema_decay)
        self.ema_model.to(self.device)
        print(f"Using EMA | decay={self.ema_decay}")

    def _update_ema(self):
        if self.use_ema and self.ema_model is not None:
            self.ema_model.update(self.model)

    @contextmanager
    def _ema_scope(self):
        """
        Temporarily use EMA weights for validation, testing, and checkpoint saving.
        """
        if (not self.use_ema) or self.ema_model is None:
            yield
            return

        backup_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

        self.model.load_state_dict(self.ema_model.state_dict(), strict=True)

        try:
            yield
        finally:
            self.model.load_state_dict(backup_state, strict=True)

    def _get_aux_loss(self):
        raw_model = self._get_raw_model()

        if hasattr(raw_model, 'get_aux_loss'):
            aux_loss = raw_model.get_aux_loss()

            if aux_loss is None:
                aux_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if not torch.is_tensor(aux_loss):
                aux_loss = torch.tensor(float(aux_loss), dtype=torch.float32, device=self.device)

            return aux_loss

        return torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def _get_selected_l_stats(self):
        raw_model = self._get_raw_model()

        if not hasattr(raw_model, 'get_context_boundary'):
            return None

        boundary = raw_model.get_context_boundary()

        if boundary is None:
            return None

        boundary = boundary.detach()

        return {
            'mean': boundary.mean().item(),
            'min': boundary.min().item(),
            'max': boundary.max().item()
        }

    def vali(self, vali_data=None, vali_loader=None, criterion=None, return_l_stats=False):
        total_loss = []

        l_mean_list = []
        l_min_list = []
        l_max_list = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                mask_input = None

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = (self.model(
                            batch_x,
                            target=batch_y[:, -self.args.pred_len:, :],
                            progressive=True
                        ) if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST') and not self.imp_mode
                          else self.model(batch_x))
                else:
                    outputs = (self.model(
                        batch_x,
                        target=batch_y[:, -self.args.pred_len:, :],
                        progressive=True
                    ) if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST') and not self.imp_mode
                      else self.model(batch_x))

                if return_l_stats:
                    l_stats = self._get_selected_l_stats()

                    if l_stats is not None:
                        l_mean_list.append(l_stats['mean'])
                        l_min_list.append(l_stats['min'])
                        l_max_list.append(l_stats['max'])

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = (F.mse_loss(outputs, batch_y)
                        if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST')
                        else criterion(outputs, batch_y))
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)

        self.model.train()

        if return_l_stats:
            if len(l_mean_list) > 0:
                l_epoch_stats = {
                    'mean': float(np.mean(l_mean_list)),
                    'min': float(np.min(l_min_list)),
                    'max': float(np.max(l_max_list)),
                }
            else:
                l_epoch_stats = {
                    'mean': -1.0,
                    'min': -1.0,
                    'max': -1.0,
                }

            return total_loss, l_epoch_stats

        return total_loss

    def train(self, setting=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid', test_batch_size=self.args.batch_size)
        test_data, test_loader = self._get_data(flag='test', test_batch_size=self.args.batch_size)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            save_every_epoch=self.args.save_every_epoch
        )

        model_optim = self._select_optimizer()

        criterion = WeightedL1Loss(self.args.lossfun_alpha, self.args.loss_mode)

        criterion_no_decay = None
        if self.args.Q_loss:
            criterion_no_decay = WeightedL1Loss(0, 'L1')

        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.resume_training and self.resume_epoch > 0:
            full_folder, new_setting = find_most_recently_modified_subfolder(
                self.args.checkpoints,
                file_name='checkpoint.pth',
                contain_str=[self.args.model_id_ori, self.args.model]
            )

            if compare_prefix_before_third_underscore(setting, new_setting, num=3):
                print(f'loading model from {full_folder}')
                self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
                shutil.copy(os.path.join(full_folder, 'checkpoint.pth'), path)
            else:
                raise ValueError('No checkpoint folder found. Please check...')

            current_val_loss = self.vali(vali_data, vali_loader, criterion)
            early_stopping.best_score = -current_val_loss
            early_stopping.val_loss_min = current_val_loss

        start_epoch = self.resume_epoch if self.resume_training else 0

        if self.resume_training:
            print('Restoring the learning rate...')
            adjust_learning_rate(model_optim, start_epoch, self.args)

        self._init_ema()

        Q_out_mat = None

        if self.args.Q_loss:
            q_out_mat_dir = os.path.join(self.args.root_path, self.args.q_out_mat_file)
            assert os.path.isfile(q_out_mat_dir)
            Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(self.device)

        scheduler = None

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                if self.args.efficient_training:
                    _, _, N = batch_x.shape
                    if N > self.args.enc_in:
                        index = np.stack(random.sample(range(N), self.args.enc_in))
                        batch_x = batch_x[:, :, index]
                        batch_y = batch_y[:, :, index]

                mask_input = None

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs0 = (self.model(
                            batch_x,
                            target=batch_y[:, -self.args.pred_len:, :],
                            progressive=False
                        ) if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST') and not self.imp_mode
                          else self.model(batch_x))

                        if isinstance(outputs0, tuple):
                            outputs = outputs0[0]
                        else:
                            outputs = outputs0

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.Q_loss and self.args.Q_loss_alpha >= 1:
                            loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        else:
                            loss = criterion(outputs, batch_y)

                        if self.args.Q_loss:
                            outputs_trans = torch.einsum(
                                'btn,tv->bvn',
                                outputs,
                                Q_out_mat.transpose(-1, -2)
                            )
                            batch_y_trans = torch.einsum(
                                'btn,tv->bvn',
                                batch_y,
                                Q_out_mat.transpose(-1, -2)
                            )
                            loss_q = criterion_no_decay(outputs_trans, batch_y_trans)
                            loss = (1 - self.args.Q_loss_alpha) * loss + self.args.Q_loss_alpha * loss_q

                        if isinstance(outputs0, tuple) and len(outputs0) >= 3:
                            dec_out_inter = outputs0[2]

                            if dec_out_inter:
                                if isinstance(dec_out_inter, List):
                                    loss1_vec = torch.stack([
                                        criterion(seq, batch_y)
                                        for seq in dec_out_inter
                                    ])

                                    weights = compute_weights(
                                        self.args.alpha,
                                        len(loss1_vec),
                                        self.args.git_multi_stage + 1,
                                        multiple_flag=self.imp_mode
                                    ).to(self.device)

                                    loss1 = (loss1_vec * weights).sum()

                                else:
                                    loss1 = criterion(dec_out_inter, batch_y)

                                loss = loss + self.args.lamda1 * loss1

                        aux_loss = self._get_aux_loss()
                        loss = loss + aux_loss

                        train_loss.append(loss.item())

                else:
                    outputs0 = (self.model(
                        batch_x,
                        target=batch_y[:, -self.args.pred_len:, :],
                        progressive=False
                    ) if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST') and not self.imp_mode
                      else self.model(batch_x))

                    if isinstance(outputs0, tuple):
                        outputs = outputs0[0]
                    else:
                        outputs = outputs0

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.Q_loss and self.args.Q_loss_alpha >= 1:
                        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    else:
                        loss = criterion(outputs, batch_y)

                    if self.args.Q_loss:
                        outputs_trans = torch.einsum(
                            'btn,tv->bvn',
                            outputs,
                            Q_out_mat.transpose(-1, -2)
                        )
                        batch_y_trans = torch.einsum(
                            'btn,tv->bvn',
                            batch_y,
                            Q_out_mat.transpose(-1, -2)
                        )
                        loss_q = criterion_no_decay(outputs_trans, batch_y_trans)
                        loss = (1 - self.args.Q_loss_alpha) * loss + self.args.Q_loss_alpha * loss_q

                    elif self.args.FFT_loss:
                        outputs_fft = torch.fft.rfft(outputs, dim=1)
                        batch_y_fft = torch.fft.rfft(batch_y, dim=1)
                        loss_fft = torch.mean(torch.abs(outputs_fft - batch_y_fft))
                        loss = (1 - self.args.Q_loss_alpha) * loss + self.args.Q_loss_alpha * loss_fft

                    if isinstance(outputs0, tuple) and len(outputs0) >= 3:
                        dec_out_inter = outputs0[2]

                        if dec_out_inter:
                            if isinstance(dec_out_inter, List):
                                loss1_vec = torch.stack([
                                    criterion(seq, batch_y)
                                    for seq in dec_out_inter
                                ])

                                weights = compute_weights(
                                    self.args.alpha,
                                    len(loss1_vec),
                                    self.args.git_multi_stage + 1,
                                    multiple_flag=self.imp_mode
                                ).to(self.device)

                                loss1 = (loss1_vec * weights).sum()

                            else:
                                loss1 = criterion(dec_out_inter, batch_y)

                            loss = loss + self.args.lamda1 * loss1

                    aux_loss = self._get_aux_loss()
                    loss = loss + aux_loss

                    if torch.isnan(loss):
                        print('\tloss is nan. please check...')
                        print("\toutputs.shape: ", outputs.shape, '\tbatch_y.shape: ', batch_y.shape)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0 or i == 0:
                    l_stats = self._get_selected_l_stats()

                    aux_loss_value = 0.0
                    if 'aux_loss' in locals():
                        aux_loss_value = aux_loss.item() if torch.is_tensor(aux_loss) else float(aux_loss)

                    if l_stats is not None:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f} | "
                            "aux_loss: {3:.7f} | selected_L mean/min/max: "
                            "{4:.2f}/{5:.2f}/{6:.2f}".format(
                                i + 1,
                                epoch + 1,
                                loss.item(),
                                aux_loss_value,
                                l_stats['mean'],
                                l_stats['min'],
                                l_stats['max']
                            )
                        )
                    else:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f} | aux_loss: {3:.7f}".format(
                                i + 1,
                                epoch + 1,
                                loss.item(),
                                aux_loss_value
                            )
                        )

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)

                    print('\tspeed: {:.4f}s/iter; left time: {:d} min, {:.2f} s'.format(
                        speed,
                        int(left_time // 60),
                        left_time % 60
                    ))

                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    model_optim.zero_grad()
                    scaler.scale(loss).backward()

                    if (i + 1) % 100 == 0 or i == 0:
                        grd_norm = compute_gradient_norm(self.model)

                        if self.args.grad_clip:
                            print(
                                f"\t\tTotal norm of gradients: {grd_norm:.2f}, "
                                f"{'clipped!' if grd_norm > self.args.max_norm else ''}"
                            )
                        else:
                            print(f"\t\tTotal norm of gradients: {grd_norm:.2f}")

                    if self.args.grad_clip:
                        scaler.unscale_(model_optim)
                        clip_grad_norm_(self.model.parameters(), self.args.max_norm)

                    scaler.step(model_optim)
                    scaler.update()

                    self._update_ema()

                else:
                    model_optim.zero_grad()
                    loss.backward()

                    if self.args.grad_clip:
                        clip_grad_norm_(self.model.parameters(), self.args.max_norm)

                    model_optim.step()

                    self._update_ema()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, False)
                    scheduler.step()

            t2 = time.time() - epoch_time

            print("Epoch: {} cost time: {}min {:.1f}s".format(epoch + 1, t2 // 60, t2 % 60))

            train_loss = np.average(train_loss)

            with self._ema_scope():
                vali_loss, vali_l_stats = self.vali(
                    vali_data,
                    vali_loader,
                    criterion,
                    return_l_stats=True
                )

                test_loss, test_l_stats = self.vali(
                    test_data,
                    test_loader,
                    criterion,
                    return_l_stats=True
                )

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} "
                f"Test Loss: {test_loss:.7f}"
            )

            print(
                f"Selected L | "
                f"Val mean/min/max: "
                f"{vali_l_stats['mean']:.2f}/"
                f"{vali_l_stats['min']:.2f}/"
                f"{vali_l_stats['max']:.2f} | "
                f"Test mean/min/max: "
                f"{test_l_stats['mean']:.2f}/"
                f"{test_l_stats['min']:.2f}/"
                f"{test_l_stats['max']:.2f}"
            )

            with self._ema_scope():
                early_stopping(vali_loss, self.model, path, epoch=epoch + 1)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler)

        best_model_path = os.path.join(path, 'checkpoint.pth')

        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting=None, test=0, test_batch_size=None):
        test_data, test_loader = self._get_data(flag='test', test_batch_size=test_batch_size)

        if test < 1:
            print('loading model...')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device)
            )
        else:
            full_folder, new_setting = find_most_recently_modified_subfolder(
                self.args.checkpoints,
                file_name='checkpoint.pth',
                contain_str=self.args.model_id_ori
            )

            if full_folder is not None and compare_prefix_before_third_underscore(setting, new_setting):
                print(f'loading model from {full_folder}')
                self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth'), map_location=self.device))
            else:
                raise ValueError('check most_recently_modified_subfolder')

        preds = []
        trues = []
        inters = []
        eval_imp = []
        mask_mat_list = []

        folder_path = self.args.folder_path

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_period = max(len(test_loader) // 5, 1)
        eval_stages = self.args.git_multi_stage + 1

        self.model.eval()

        if hasattr(self.model, 'alpha'):
            print(f'self.model.alpha: {self.model.alpha}')

        outputs_list = None

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    if not self.imp_mode:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                B, T, N = batch_x.shape

                if self.imp_mode:
                    if self.args.data.lower() == 'air':
                        obs_mat = batch_y.to(self.device)
                        batch_y = batch_x

                        batch_y[~obs_mat] = torch.nan

                        mask = torch.rand_like(batch_x) * obs_mat
                        mask[mask <= self.args.mask_rate] = 0
                        mask[mask > self.args.mask_rate] = 1

                        mask_mat = mask == 0

                        assert torch.all(mask_mat.float() + obs_mat.float())

                        batch_x = batch_x.masked_fill(mask_mat, 0)

                    elif self.args.data.lower() == 'physio':
                        obs_mat = (batch_y > 0).to(self.device)

                        batch_y = batch_x.to(self.device)

                        batch_y[~obs_mat] = torch.nan

                        mask = torch.rand_like(batch_x) * obs_mat
                        mask[mask <= self.args.mask_rate] = 0
                        mask[mask > self.args.mask_rate] = 1

                        mask_mat = mask == 0

                        batch_x = batch_x.masked_fill(mask_mat, 0)

                    else:
                        batch_y = batch_x
                        batch_y_mark = batch_x_mark

                        mask = torch.rand((B, T, N)).to(self.device)
                        mask[mask <= self.args.mask_rate] = 0
                        mask[mask > self.args.mask_rate] = 1

                        mask_mat = mask == 0

                        batch_x = batch_x.masked_fill(mask_mat, 0)

                else:
                    batch_y = batch_y.float().to(self.device)
                    mask_mat = torch.ones((B, T, N)).to(self.device) != 0

                mask_input = mask_mat if self.imp_mode else None

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_list = (self.model(
                            batch_x,
                            target=batch_y[:, -self.args.pred_len:, :],
                            progressive=True
                        ) if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST') and not self.imp_mode
                          else self.model(batch_x))
                else:
                    outputs_list = (self.model(
                        batch_x,
                        target=batch_y[:, -self.args.pred_len:, :],
                        progressive=True
                    ) if self.args.model in ('FORT', 'SHIFT', 'TRACE', 'PRISM', 'RECAST') and not self.imp_mode
                      else self.model(batch_x))

                if isinstance(outputs_list, tuple):
                    outputs = outputs_list[0]
                else:
                    outputs = outputs_list

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()

                dec_seq_inter = []

                if isinstance(outputs_list, tuple):
                    dec_seq_inter_raw = outputs_list[-1]

                    if isinstance(dec_seq_inter_raw, list):
                        dec_seq_inter = [
                            seq[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                            for seq in dec_seq_inter_raw
                        ]

                    if self.imp_mode and self.eval_flag:
                        eval_imp_batch = outputs_list[1].detach().cpu().numpy()
                        eval_stages = eval_imp_batch.shape[-1]

                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape

                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    if len(dec_seq_inter) > 0:
                        dec_seq_inter = [
                            test_data.inverse_transform(seq.squeeze(0)).reshape(shape)
                            for seq in dec_seq_inter
                        ]

                pred = outputs
                true = batch_y

                mask_mat_np = mask_mat.cpu().numpy()

                if self.imp_mode:
                    pred[~mask_mat_np] = true[~mask_mat_np]

                preds.append(pred)
                trues.append(true)
                inters.append(dec_seq_inter)

                if self.imp_mode and self.eval_flag:
                    eval_imp.append(eval_imp_batch)

                mask_mat_list.append(mask_mat_np)

                if i % save_period == 0:
                    print(f'processing batch{i} at test phase...')

                    input_ = batch_x.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        shape = input_.shape
                        input_ = test_data.inverse_transform(input_.squeeze(0)).reshape(shape)

                    if self.imp_mode:
                        if self.args.data == 'physio':
                            nan_mask = np.isnan(true)
                            nan_counts = np.sum(nan_mask, axis=1)
                            min_ind = np.argmin(nan_counts)

                            it, jt = min_ind // true.shape[-1], min_ind % true.shape[-1]

                            gt = true[it, :, jt]
                            pd = pred[it, :, jt]

                        else:
                            gt = true[0, :, -1]
                            pd = pred[0, :, -1]

                    else:
                        gt = np.concatenate((input_[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_[0, :, -1], pred[0, :, -1]), axis=0)

                    if self.args.save_pdf:
                        save_folder = os.path.join(folder_path, 'pred-gt-pdf-npy')

                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder, exist_ok=True)

                        visual(
                            gt,
                            pd,
                            os.path.join(
                                save_folder,
                                f'{self.args.model_id}_batch_{i}_imp-gt.pdf'
                                if self.imp_mode
                                else f'{self.args.model_id}_batch_{i}_pred-gt.pdf'
                            ),
                            imp=self.imp_mode
                        )

                        np.save(os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_pred.npy'), pd)
                        np.save(os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_gt.npy'), gt)

                        write_into_xls(
                            excel_name=os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_gt.xlsx'),
                            mat=gt,
                            columns=None
                        )

                        write_into_xls(
                            excel_name=os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_pred.xlsx'),
                            mat=pd,
                            columns=None
                        )

                        for ii, seq_inter in enumerate(dec_seq_inter):
                            if self.imp_mode:
                                seq_inter[~mask_mat_np] = true[~mask_mat_np]

                                if self.args.data == 'physio':
                                    pdi = seq_inter[it, :, jt]
                                else:
                                    pdi = seq_inter[0, :, -1]

                            else:
                                pdi = np.concatenate((input_[0, :, -1], seq_inter[0, :, -1]), axis=0)

                            visual(
                                gt,
                                pdi,
                                os.path.join(
                                    save_folder,
                                    f'{self.args.model_id}_batch_{i}_imp-gt_stage_{ii}.pdf'
                                    if self.imp_mode
                                    else f'{self.args.model_id}_batch_{i}_pred-gt_stage_{ii}.pdf'
                                ),
                                imp=self.imp_mode
                            )

                            np.save(
                                os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_pred_stage_{ii}.npy'),
                                pdi
                            )

                            write_into_xls(
                                excel_name=os.path.join(
                                    save_folder,
                                    f'{self.args.model_id}_batch_{i}_pred_stage_{ii}.xlsx'
                                ),
                                mat=pdi,
                                columns=None
                            )

        preds_array = np.concatenate(preds, axis=0)
        trues_array = np.concatenate(trues, axis=0)
        mask_mat = np.concatenate(mask_mat_list, axis=0)

        if self.imp_mode and self.eval_flag:
            eval_imp = np.array(eval_imp).reshape((-1, eval_stages)).mean(axis=0)

        print('test shape:', preds_array.shape, trues_array.shape)

        mse_list = []
        mae_list = []
        self.last_segment_metrics = []

        if self.imp_mode:
            mae, mse, rmse, mape, mspe = metric(preds_array[mask_mat], trues_array[mask_mat])
            r2, pear, mase = 0.0, 0.0, 0.0
            f = open("result_imputation.txt", 'a')

        else:
            mae, mse, rmse, mape, mspe, r2, pear, mase = metric(preds_array, trues_array)

            print(f'preds_array.shape: {preds_array.shape}')
            print(
                f'final output mse: {mse:.5f}, mae: {mae:.5f}, '
                f'r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}'
            )

            f = open("result_long_term_forecast.txt", 'a')

            if self.args.pred_len == 720:
                segment_bounds = ((0, 96), (96, 192), (192, 336), (336, 720))
                for start, end in segment_bounds:
                    seg_mae, seg_mse, _, _, _, _, _, _ = metric(
                        preds_array[:, start:end, :],
                        trues_array[:, start:end, :]
                    )
                    self.last_segment_metrics.append({
                        'start': int(start),
                        'end': int(end),
                        'mse': float(seg_mse),
                        'mae': float(seg_mae),
                    })

                segment_table_lines = [
                    '[Segment Metrics] H=720',
                    '+-----------+------------+------------+',
                    '| Interval  | MSE        | MAE        |',
                    '+-----------+------------+------------+',
                ]
                segment_table_lines.extend(
                    f"| {item['start']:>3}~{item['end']:<3}   | "
                    f"{item['mse']:>10.6f} | {item['mae']:>10.6f} |"
                    for item in self.last_segment_metrics
                )
                segment_table_lines.append(
                    '+-----------+------------+------------+'
                )
                segment_table = '\n'.join(segment_table_lines)
                print(segment_table)
                f.write(segment_table + '\n')

                segment_csv_path = os.path.join(
                    folder_path, 'segment_metrics.csv'
                )
                with open(
                    segment_csv_path, 'w', newline='', encoding='utf-8'
                ) as segment_file:
                    writer = csv.DictWriter(
                        segment_file,
                        fieldnames=['start', 'end', 'mse', 'mae']
                    )
                    writer.writeheader()
                    writer.writerows(self.last_segment_metrics)
                print(
                    f'[Segment Metrics] CSV saved to: {segment_csv_path}'
                )

        if isinstance(outputs_list, tuple) and self.args.seq_inter:
            if self.imp_mode:
                true_list = []
                pred_list = []

                for inter_seq_list, true, mask_mat_item in zip(inters, trues, mask_mat_list):
                    true_list.append(true[mask_mat_item])
                    pred_list.append([inter_seq[mask_mat_item] for inter_seq in inter_seq_list])

                all_pred_list = list(zip(*pred_list))

                true_arr = np.concatenate([arr.flatten() for arr in true_list])
                all_pred_arr = [
                    np.concatenate([arr.flatten() for arr in pred_stage])
                    for pred_stage in all_pred_list
                ]

                all_stages = len(all_pred_arr) + 1
                stage_num = max(all_stages // (self.args.git_multi_stage + 1), 1)

                for i, pred_stage in enumerate(all_pred_arr):
                    if i == 0:
                        print('stage (i) seq shape:', pred_stage.shape)
                    elif i % stage_num == 0:
                        print('----------------------------------')

                    mae_, mse_, _, _, _ = metric(pred_stage, true_arr)

                    mse_list.append(mse_)
                    mae_list.append(mae_)

                    if self.eval_flag:
                        print(
                            f'stage{i}: mse:{mse_:.5f}, mae:{mae_:.5f}, '
                            f'cos_dist:{eval_imp[min(i, len(eval_imp) - 1)]:.5f}'
                        )
                    else:
                        print(f'stage{i}: mse:{mse_:.5f}, mae:{mae_:.5f}')

            else:
                stage_list = list(zip(*inters))
                stage_list = [np.concatenate(stage_item, axis=0) for stage_item in stage_list]

                for i, pred_stage in enumerate(stage_list):
                    if i == 0:
                        print('stage (i) seq shape:', pred_stage.shape)

                    mae_, mse_, _, _, _, r2_, pear_, mase_ = metric(pred_stage, trues_array)

                    mse_list.append(mse_)
                    mae_list.append(mae_)

                    print(
                        f'stage{i}: mse:{mse_:.5f}, mae:{mae_:.5f}, '
                        f'r2: {r2_:.5f}, pear: {pear_:.5f}, mase: {mase_:.5f}'
                    )

        if self.imp_mode and self.eval_flag:
            print(
                f'final stage: mse:{mse:.5f}, mae:{mae:.5f}, '
                f'cos_dist:{eval_imp[-1]:.5f}'
            )
        else:
            if self.imp_mode:
                print(f'final stage: mse:{mse:.5f}, mae:{mae:.5f}')
            else:
                print(
                    f'final stage: mse:{mse:.5f}, mae:{mae:.5f}, '
                    f'r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}'
                )

        final_mae = mae
        final_mse = mse
        final_rmse = rmse
        final_mape = mape
        final_mspe = mspe

        f.write(setting + "  \n")

        if self.imp_mode:
            f.write('mse:{:.5f}, mae:{:.5f}'.format(final_mse, final_mae))
        else:
            f.write(
                f'mae: {final_mae:.5f}, mse: {final_mse:.5f}, '
                f'r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}'
            )

        f.write('\n')
        f.write('\n')
        f.close()

        if self.imp_mode:
            print(f"Final Test Output | mse:{final_mse:.5f}, mae:{final_mae:.5f}")
        else:
            print(
                f"Final Test Output | mse:{final_mse:.5f}, mae:{final_mae:.5f}, "
                f"r2:{r2:.5f}, pear:{pear:.5f}, mase:{mase:.5f}"
            )

        np.save(
            os.path.join(folder_path, 'metrics.npy'),
            np.array([final_mae, final_mse, final_rmse, final_mape, final_mspe])
        )

        write_into_xls(
            os.path.join(folder_path, 'metrics.xlsx'),
            [final_mae, final_mse, final_rmse, final_mape, final_mspe]
        )

        file_name = f"MSE_{final_mse:.5f}_MAE_{final_mae:.5f}_" + setting
        new_folder_path = os.path.join('results', file_name[:254])

        if os.path.exists(new_folder_path):
            new_folder_path = os.path.join(
                'results',
                f"MSE_{final_mse:.5f}_MAE_{final_mae:.5f}_{int(time.time())}_" + setting
            )
            new_folder_path = new_folder_path[:254]

        os.rename(folder_path, new_folder_path)

        return final_mse, final_mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()

                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)

                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = os.path.join('results', setting)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return
