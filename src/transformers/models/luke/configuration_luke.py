# coding=utf-8
# Copyright Studio Ousia and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LUKE configuration """

from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "luke-large": "https://huggingface.co/luke-large/resolve/main/config.json",
}


class LukeConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LukeModel`. It is used to
    instantiate a LUKE model according to the specified arguments, defining the model architecture. Configuration
    objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model outputs. Read the
    documentation from :class:`~transformers.PretrainedConfig` for more information. The
    :class:`~transformers.LukeConfig` class directly inherits :class:`~transformers.RobertaConfig`. It reuses the same
    defaults. Please check the parent class for more information.

    Examples::
        >>> from transformers import LukeConfig, LukeModel
        >>> # Initializing a LUKE configuration
        >>> configuration = LukeConfig()
        >>> # Initializing a model from the configuration
        >>> model = LukeModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "luke"

    def __init__(self, entity_vocab_size: int = 500000, entity_emb_size: int = None, **kwargs):
        """Constructs LukeConfig."""
        super(LukeConfig, self).__init__(**kwargs)

        self.entity_vocab_size = entity_vocab_size
        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size
