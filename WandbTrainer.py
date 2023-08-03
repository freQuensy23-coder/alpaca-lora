from typing import Any, Dict, Union

import torch
import transformers
from torch import nn


class WandbTrainer(transformers.Trainer):
    def __init__(self, wandb_logger, **kwargs):
        super().__init__(**kwargs)
        self.wandb_logger = wandb_logger

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        log = {"train_loss": loss.item(), "epoch": int(self.state.epoch), "step": int(self.state.global_step)}
        self.wandb_logger.log(log)
        print(log)
        # Возвращаем loss
        return loss.detach() / self.args.gradient_accumulation_steps
