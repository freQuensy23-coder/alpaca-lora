from typing import Any, Dict, Union

import torch
import transformers
from torch import nn


class WandbTrainer(transformers.Trainer):
    def __init__(self, wandb_logger, **kwargs):
        super().__init__(**kwargs)
        self.wandb_logger = wandb_logger

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        print("training_step")
        loss = super().training_step(model, inputs)
        self.wandb_logger.log({"train_loss": loss.item()})

        # Возвращаем loss
        return loss.detach() / self.args.gradient_accumulation_steps
