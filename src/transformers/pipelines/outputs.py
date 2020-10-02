from dataclasses import dataclass
from typing import Union, List, NamedTuple

from transformers import BatchEncoding

Entity = NamedTuple("Entity", [("token", str), ("index", int), ("label", str), ("score", float)])


@dataclass
class TokenClassificationOutput:
    input: Union[str, List[str]]
    encodings: BatchEncoding
    entities: List[Entity]
