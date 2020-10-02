from pathlib import Path
from typing import Any, Union

from transformers import PreTrainedTokenizer

from .base import PipelineConfigType, PipelineInputType, PipelineIntermediateType, MaybeBatch, ModelType, \
    PipelineOutputType, Pipeline, PipelineConfig
from .configs import TokenClassificationConfig
from .outputs import TokenClassificationOutput


def pipeline(task: str, model: Union[str, Path, Any], tokenizer: Union[str, Path, PreTrainedTokenizer]):
    pass
