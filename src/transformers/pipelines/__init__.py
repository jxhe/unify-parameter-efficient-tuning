from pathlib import Path
from typing import Any, Union

from transformers import PreTrainedTokenizer

from .base import ConfigType, InputType, IntermediateType, MaybeBatch, ModelType, OutputType, Pipeline, PipelineConfig


def pipeline(task: str, model: Union[str, Path, Any], tokenizer: Union[str, Path, PreTrainedTokenizer]):
    pass
