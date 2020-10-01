from pathlib import Path
from typing import Any, Union

from transformers import PreTrainedTokenizer

from .base import PipelineConfigType, PipelineInputType, IntermediateType, MaybeBatch, ModelType, PipelineOutputType, Pipeline, PipelineConfig


def pipeline(task: str, model: Union[str, Path, Any], tokenizer: Union[str, Path, PreTrainedTokenizer]):
    pass
