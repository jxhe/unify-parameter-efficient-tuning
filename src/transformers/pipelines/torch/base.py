from abc import ABC
from typing import Optional, Union

import torch

from transformers import PreTrainedModel, BatchEncoding
from transformers.pipelines import MaybeBatch, Pipeline
from transformers.pipelines.base import PipelineConfigType, PipelineInputType, IntermediateType, PipelineOutputType, \
    PipelineIntermediateType


class PreTrainedPipeline(Pipeline[PipelineConfigType, PreTrainedModel, PipelineInputType, IntermediateType, PipelineOutputType], ABC):

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
        return self._tokenizer.batch_encode_plus(
            inputs,
            return_tensors=torch.TensorType.PYTORCH,
            **config.tokenizer_kwargs,
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
        return self.postprocess(model_output, config)

    def forward(self, encodings, config):
        return self._model(**encodings, **config.model_kwargs)





