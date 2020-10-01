from transformers.modeling_outputs import TokenClassifierOutput
from transformers.pipelines.base import IntermediateType, MaybeBatch, PipelineOutputType, PipelineConfig
from transformers.pipelines.torch import PreTrainedPipeline


# Token Classification

TokenClassificationInput = str


class TokenClassificationConfig(PipelineConfig):
    group_entities: bool = False


class TokenClassificationOutput:
    pass


class TokenClassificationPipeline(
    PreTrainedPipeline[
        TokenClassificationConfig, TokenClassificationInput, TokenClassifierOutput, TokenClassificationOutput
    ]
):
    task = "token-classification"
    default_config = TokenClassificationConfig()

    def postprocess(self, model_output: TokenClassifierOutput, config: TokenClassificationConfig) -> MaybeBatch[TokenClassificationOutput]:

        pass
