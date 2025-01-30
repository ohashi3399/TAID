"""
Curriculum Temperature for Knowledge Distillation
https://arxiv.org/abs/2211.16231

This implementation is based on https://github.com/zhengli97/CTKD
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from .base import DistilLoss


# copied from https://github.com/zhengli97/CTKD/blob/master/models/temp_global.py#L21
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class CTKD(DistilLoss):
    """
    Implementation of CTKD for Language Modeling
    """

    def __init__(
        self,
        lambda_max: float = 1,
        lambda_min: float = 0,
        num_loops: int = 10,
        temp_start: float = 1,
        temp_end: float = 20,
    ):
        super().__init__()
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.num_loops = num_loops
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.global_temperature = nn.Parameter(torch.ones([]))
        # In their experiments, Global-T is used as default
        self.grl = GradientReversal()

    def get_value(self, epoch):
        if epoch < 0:
            epoch = 0
        if epoch >= self.num_loops:
            epoch = self.num_loops
        value = (math.cos(epoch * math.pi / self.num_loops) + 1.0) * 0.5
        value = value * (self.lambda_max - self.lambda_min) + self.lambda_min
        return value

    def forward(
        self,
        lightning_module: LightningModule,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch,
        **kwargs,
    ) -> torch.Tensor:
        epoch = lightning_module.trainer.current_epoch + 1
        lambda_ = self.get_value(epoch)
        temp = self.grl(self.global_temperature, lambda_)
        temp = self.temp_start + self.temp_end * torch.sigmoid(temp)
        # forward kl
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(logits / temp, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(
            mask.view(-1), dim=0
        )
        distil_loss = distil_loss * temp * temp
        return distil_loss
