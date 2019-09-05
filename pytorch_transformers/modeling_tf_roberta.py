# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" TF 2.0 BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import numpy as np
import tensorflow as tf

from pytorch_transformers import RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_tf_bert import TFBertEmbeddings, TFBertModel
from .configuration_bert import BertConfig
from .modeling_tf_utils import TFPreTrainedModel
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}


class TFRobertaEmbeddings(TFBertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, **kwargs):
        super(TFRobertaEmbeddings, self).__init__(config, **kwargs)
        self.padding_idx = 1

    @tf.function
    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """

        input_ids, position_ids, token_type_ids = inputs

        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = tf.range(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=tf.int32)[tf.newaxis, :]

        inputs = [input_ids, position_ids, token_type_ids]

        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))


class TFRobertaModel(TFBertModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(TFRobertaModel, self).__init__(config)

        self.embeddings = TFRobertaEmbeddings(config, name="embeddings")

    @tf.function
    def __call__(self, inputs, training=False):
        if not isinstance(inputs, (dict, tuple, list)):
            input_ids = inputs
        else:
            if isinstance(inputs, (tuple, list)):
                input_ids = inputs[0]
            else:
                input_ids = inputs["input_ids"]

        if tf.reduce_sum(input_ids[:, 0]) == 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")

        return super(TFRobertaModel, self).call(inputs)




















