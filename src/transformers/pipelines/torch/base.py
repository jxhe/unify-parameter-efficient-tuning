from abc import ABC

import torch

from transformers import Pipeline, PreTrainedModel
from transformers.pipelines.base import ConfigType, InputType, IntermediateType, OutputType


class TorchPipeline(Pipeline[ConfigType, PreTrainedModel, InputType, IntermediateType, OutputType], ABC):
    @property
    def device(self) -> torch.device:
        return self._model.device

    def cpu(self) -> "TorchPipeline":
        self._model.cpu()
        return self

    def gpu(self) -> "TorchPipeline":
        self._model.gpu()
        return self
