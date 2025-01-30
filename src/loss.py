from typing import Dict, Optional
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from lightning import LightningModule
from src.utils import default
from src.distil_losses import *


@dataclass
class LossOutput:
    loss: Tensor
    loss_dict: Dict[str, Tensor]


class KDLoss(nn.Module):
    def __init__(
        self,
        distil_loss_fn: Optional[DistilLoss] = None,
        distil_ratio: float = 1.0,
    ):
        super().__init__()
        self.distil_loss_fn = distil_loss_fn
        self.lm_weight = 1.0 - distil_ratio
        self.distil_weight = distil_ratio

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Dict[str, Tensor],
        distill_loss_kwargs: Optional[dict] = None,
    ) -> LossOutput:
        distill_loss_kwargs = default(distill_loss_kwargs, {})
        loss_dict = {}
        model_inputs = batch.get("model_inputs")

        # 1. Compute causal lm loss
        outputs = lightning_module.student_model(**model_inputs, use_cache=False)
        logits = outputs.logits
        lm_loss = outputs.loss
        loss_dict["lm_loss"] = lm_loss
        # 2. Compute distil loss
        if self.distil_loss_fn is None:
            distil_loss = None
            loss = lm_loss
        else:
            mask = (model_inputs["labels"] != -100).int()
            with torch.no_grad():
                teacher_outputs = lightning_module.teacher_model(
                    **model_inputs, use_cache=False
                )
                teacher_logits = teacher_outputs.logits
            distill_loss_kwargs["teacher_logits"] = teacher_logits[
                ..., :-1, :
            ].contiguous()
            distil_loss = self.distil_loss_fn(
                lightning_module=lightning_module,
                logits=logits[..., :-1, :].contiguous(),
                mask=mask[..., 1:].contiguous(),
                batch=batch,
                lm_loss=lm_loss,
                **distill_loss_kwargs,
            )

            if isinstance(distil_loss, dict):
                assert "distil_loss" in distil_loss
                loss_dict.update(distil_loss)
                distil_loss = loss_dict["distil_loss"]
            else:
                loss_dict["distil_loss"] = distil_loss
            loss = self.lm_weight * lm_loss + self.distil_weight * distil_loss

        loss_dict["loss"] = loss
        return LossOutput(
            loss=loss,
            loss_dict=loss_dict,
        )


def get_loss_fn(args):
    if args.loss_type == "sft":
        distil_loss_fn = None
    elif args.loss_type == "fkl":
        distil_loss_fn = ForwardKL()
    elif args.loss_type == "rkl":
        distil_loss_fn = ReverseKL()
    elif args.loss_type == "tvd":
        distil_loss_fn = TVD()
    elif args.loss_type == "js":
        distil_loss_fn = JS(args.js_beta)
    elif args.loss_type == "adaptive_kl":
        distil_loss_fn = AdaptiveKL(args.adaptive_kl_threshold)
    elif args.loss_type == "sfkl":
        distil_loss_fn = SkewForwardKL(args.skew_beta)
    elif args.loss_type == "srkl":
        distil_loss_fn = SkewReverseKL(args.skew_beta)
    elif args.loss_type == "ctkd":
        distil_loss_fn = CTKD()
    elif args.loss_type == "dkd":
        distil_loss_fn = DKD()
    elif args.loss_type == "taid":
        distil_loss_fn = TAID(
            t_start=args.taid_t_start,
            t_end=args.taid_t_end,
            alpha=args.taid_alpha,
            beta=args.taid_beta,
            disable_adaptive=args.taid_disable_adaptive,
        )
    else:
        raise NotImplementedError(args.loss_type)

    return KDLoss(distil_loss_fn=distil_loss_fn, distil_ratio=args.distil_ratio)
