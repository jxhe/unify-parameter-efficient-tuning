from abc import ABC
from typing import Optional, Union

import torch

from transformers import PreTrainedModel, BatchEncoding, TensorType
from transformers.pipelines import MaybeBatch, Pipeline
from transformers.pipelines.base import PipelineConfigType, PipelineInputType, PipelineOutputType, \
    PipelineIntermediateType


class PreTrainedPipeline(Pipeline[PipelineConfigType, PreTrainedModel, PipelineInputType, PipelineIntermediateType, PipelineOutputType], ABC):

    @property
    def device(self) -> torch.device:
        return self._model.device

    def cpu(self) -> "PreTrainedPipeline":
        self._model.cpu()
        return self

    def gpu(self) -> "PreTrainedPipeline":
        self._model.gpu()
        return self

    def preprocess(self, inputs: MaybeBatch[PipelineInputType], config: PipelineConfigType) -> BatchEncoding:
        return self._tokenizer(
            inputs,
            return_tensors=TensorType.PYTORCH,
            **(config.tokenizer_kwargs or {}),
        )

    def __call__(self, inputs: MaybeBatch[PipelineInputType], config: Optional[Union[str, PipelineConfigType]], **kwargs) -> MaybeBatch[PipelineIntermediateType]:
        # Retrieve the appropriate config if identifier is provided or None
        if isinstance(config, str):
            config = self.get_config(config)
        elif not config:
            config = self.default_config

        # Preprocess the input
        encodings = self.preprocess(inputs, config)

        # Forward the encoding through the model
        model_output = self.forward(encodings, config)

        # Apply any postprocessing steps required
        return self.postprocess(encodings, model_output, config)

    def forward(self, encodings, config) -> PipelineIntermediateType:
        return self._model(**encodings, **config.model_kwargs)





