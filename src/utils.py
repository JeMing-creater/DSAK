import math
import warnings
from collections import OrderedDict
from typing import List, Tuple
import random
import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
import numpy as np
import math

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                    1
                    + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
            )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash,
                                                        map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(download_path, map_location=torch.device('cpu'))
    return state_dict


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict, strict=False)
        accelerator.print(f'加载预训练模型成功！')
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'加载预训练模型失败！')
        return model


def give_model(config) -> Tuple[torch.nn.Module, torch.nn.Module]:
    from src import model_dic
    model_name = config.finetune.model
    if 'AlexNet' in model_name or 'alexnet' in model_name:
        target_model = model_dic.AlexNet(**config.model)
        shadow_model = model_dic.AlexNet(**config.model)
        return target_model, shadow_model
    elif 'ResNet' in model_name or 'resnet' in model_name:
        target_model = model_dic.ResNet(**config.model)
        shadow_model = model_dic.ResNet(**config.model)
        return target_model, shadow_model
    elif 'DesNet' in model_name or 'desnet' in model_name:
        target_model = model_dic.DesNet(**config.model)
        shadow_model = model_dic.DesNet(**config.model)
        return target_model, shadow_model
    elif 'VGG' in model_name or 'vgg' in model_name:
        target_model = model_dic.VGG(**config.model)
        shadow_model = model_dic.VGG(**config.model)
        return target_model, shadow_model
    elif 'Linear' in model_name and 'Linear_5' not in model_name:
        target_model = model_dic.NN(**config.model)
        shadow_model = model_dic.NN(**config.model)
        return target_model, shadow_model
    elif 'Linear_5' in model_name:
        target_model = model_dic.NN_5(**config.model)
        shadow_model = model_dic.NN_5(**config.model)
        return target_model, shadow_model
    else:
        raise ValueError('model name error!')


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def common_params(student_model: nn.Module, teacher_model: nn.Module, accelerator: Accelerator):
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in accelerator.unwrap_model(student_model).named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in accelerator.unwrap_model(teacher_model).named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
    return params_q, params_k


def give_mask(config, input):
    if 'CIFAR' in config.preprocess.dataset:
        mask_radio = config.defence.mask_radio
        masks = []
        keep_flag = False
        for batch in input:
            c, w, h = batch.shape[0], batch.shape[1], batch.shape[2]
            mask_c, mask_w, mask_h, = math.ceil(mask_radio * c), math.ceil(mask_radio * w), math.ceil(mask_radio * h)
            c_end, w_end, h_end = c - mask_c, w - mask_w, h - mask_h  # 矩形掩码终点
            c_start, w_start, h_start = random.randint(0, c_end), random.randint(0, w_end), random.randint(0, h_end)
            end_c, end_w, end_h = c_start + mask_c, w_start + mask_w, h_start + mask_h
            mask = np.zeros((c, w, h), dtype=bool)
            for i in range(0, c):
                for j in range(0, w):
                    for z in range(0, h):
                        if i >= c_start and j >= w_start and z >= h_start:
                            if i< end_c and j<end_w and z<end_h:
                                mask[i, j, z] = 1
            mask = torch.from_numpy(mask)
            if keep_flag == False:
                masks = mask.unsqueeze(0)
                keep_flag = True
            else:
                masks = torch.cat([masks, mask.unsqueeze(0)], dim=0)
    return masks


if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 32)
    give_mask(0, x)
