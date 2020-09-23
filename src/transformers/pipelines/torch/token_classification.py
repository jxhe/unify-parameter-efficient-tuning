from transformers import BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.pipelines.base import ConfigType, InputType, IntermediateType, MaybeBatch, OutputType, PipelineConfig
from transformers.pipelines.torch import TorchPipeline


# Token Classification

TokenClassificationInput = str
TokenClassificationConfig = PipelineConfig


class TokenClassificationOutput:
    pass


class TokenClassificationPipeline(
    TorchPipeline[
        TokenClassificationConfig, TokenClassificationInput, TokenClassifierOutput, TokenClassificationOutput
    ]
):
    task = "token-classification"
    default_config = TokenClassificationConfig()

    def preprocess(self, inputs: MaybeBatch[InputType], config: ConfigType) -> BatchEncoding:
        pass

    def forward(self, encodings: BatchEncoding, config: ConfigType) -> IntermediateType:
        pass

    def postprocess(self, model_output: IntermediateType, config: ConfigType) -> MaybeBatch[OutputType]:
        pass
