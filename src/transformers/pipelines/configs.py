from dataclasses import dataclass, field
from typing import List

from transformers.pipelines import PipelineConfig


@dataclass
class TokenClassificationConfig(PipelineConfig):
    group_entities: bool = False
    ignore_labels: List[int] = field(default_factory=list)

