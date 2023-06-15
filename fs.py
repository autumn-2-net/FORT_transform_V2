from typing import Union

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
class V2LSGDRLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,

            last_epoch: int = -1,T_0=15000,eta_min=0.00004,eta_max=0.00006,tmctx=0.99
    ):
        # assert check_argument_types()

        self.eta_min = eta_min
        self.T_0 = T_0
        self.eta_max = eta_max
        self.tmctx=tmctx

        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, lr={self.base_lr}, min_lr={self.min_lr}, last_epoch={self.last_epoch})"

    def mmmctxadjust_lr(self,):
        step_num = self.last_epoch+1

        T_cur = (step_num ) % self.T_0
        T_i = self.T_0
        T_curX = (step_num)// self.T_0


        cur_lr = self.eta_min*(self.tmctx**T_curX) + 0.5 * (self.eta_max *(self.tmctx**T_curX)- self.eta_min*(self.tmctx**T_curX)) * (1 + np.cos(np.pi * T_cur / T_i))


        return cur_lr


    def get_lr(self):
        # step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            lrs = []
            for lr in self.base_lrs:
                lr = self.ctxadjust_lr()
                lrs.append(lr)
            return lrs
        else:
            lrs = []
            for lr in self.base_lrs:
                lr = self.ctxadjust_lr()
                lrs.append(lr)
            return lrs

    def set_step(self, step: int):
        self.last_epoch = step