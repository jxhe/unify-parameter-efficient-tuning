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

from pytorch_transformers import RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, TFSharedEmbeddings
from pytorch_transformers.modeling_tf_bert import TFBertEmbeddings, TFBertModel, gelu, TFBertPreTrainedModel
from .configuration_bert import BertConfig
from .modeling_tf_utils import TFPreTrainedModel
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

class TFRobertaModel(TFBertModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(TFRobertaModel, self).__init__(config)

        self.embeddings = TFSharedEmbeddings(config.vocab_size, config.hidden_size, name="embeddings")

    def call(self, inputs, training=False):
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

        return super(TFRobertaModel, self).call(inputs, training=training)


class TFRobertaForMaskedLM(TFBertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(TFRobertaForMaskedLM, self).__init__(config)

        self.roberta = TFRobertaModel(config)
        self.lm_head = TFRobertaLMHead(config)

    def call(self, inputs, training=False):

        outputs = self.roberta(inputs, training=training)
        sequence_output = outputs[0]
        predictions = self.lm_head(sequence_output)
        prediction_scores = self.roberta.embeddings(predictions, mode='linear')

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        return outputs  # prediction_scores, (hidden_states), (attentions)


class TFRobertaLMHead(tf.keras.layers.Layer):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(TFRobertaLMHead, self).__init__()
        self.dense = tf.keras.layers.Dense(config.hidden_size)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')

    def call(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        return x


class TFRobertaForSequenceClassification(TFBertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(TFRobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaModel(config)
        self.classifier = TFRobertaClassificationHead(config)


    def call(self, inputs, training=False):
        outputs = self.roberta(inputs, training=training)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, training=training)
        outputs = (logits,) + outputs[2:]
        return outputs  # (loss), logits, (hidden_states), (attentions)


class TFRobertaClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(TFRobertaClassificationHead, self).__init__()
        self.dense = tf.keras.layers.Dense(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = tf.keras.layers.Dense(config.num_labels)

    def call(self, features, training,  **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = tf.tanh(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x














