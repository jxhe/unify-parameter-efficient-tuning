# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch LUKE model. """

import unittest

from tests.test_modeling_common import floats_tensor
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        LukeConfig,
        LukeModel,
        LukeEntityAwareAttentionModel,
        LukeTokenizer,
        LukeTokenizer
    )
    from transformers.models.luke.modeling_luke import (
        LUKE_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class LukeModelTester:
    def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            use_token_type_ids=True,
            use_labels=True,
            vocab_size=99,
            entity_vocab_size=10,
            entity_emb_size=6,
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=16,
            type_sequence_label_size=2,
            initializer_range=0.02,
            num_labels=3,
            num_choices=4,
            scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.entity_emb_size = entity_emb_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = LukeConfig(
            vocab_size=self.vocab_size,
            entity_vocab_size=self.entity_vocab_size,
            entity_emb_size=self.entity_emb_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LukeModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class LukeModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            LukeModel,
            LukeEntityAwareAttentionModel,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = LukeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LukeConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in LUKE_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = LukeModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


def prepare_luke_batch_inputs(tokenizer):
        # Taken from Open Entity dev set
        text = """Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."""
        span = (39,42)
        
        ENTITY_TOKEN = '<ent>'
        max_mention_length = 30
        
#         conv_tables = (
#             ("-LRB-", "("),
#             ("-LCB-", "("),
#             ("-LSB-", "("),
#             ("-RRB-", ")"),
#             ("-RCB-", ")"),
#             ("-RSB-", ")"),
#         )
        
#         def preprocess_and_tokenize(text, start, end=None):
#                 target_text = text[start:end]
#                 for a, b in conv_tables:
#                     target_text = target_text.replace(a, b)

                return tokenizer.tokenize(target_text.strip(), add_prefix_space=True)

#         tokens = [tokenizer.cls_token]
#         tokens += preprocess_and_tokenize(text, 0, span[0])
#         mention_start = len(tokens)
#         tokens.append(ENTITY_TOKEN)
#         tokens += preprocess_and_tokenize(text, span[0], span[1])
#         tokens.append(ENTITY_TOKEN)
#         mention_end = len(tokens)

#         tokens += preprocess_and_tokenize(text, span[1])
#         tokens.append(tokenizer.sep_token)

#         encoding = {}
#         encoding['input_ids'] = tokenizer.convert_tokens_to_ids(tokens)
#         encoding['attention_mask'] = [1] * len(tokens)
#         encoding['token_type_ids'] = [0] * len(tokens)

#         encoding['entity_ids'] = [1, 0]
#         encoding['entity_attention_mask'] = [1, 0]
#         encoding['entity_token_type_ids'] = [0, 0]
#         entity_position_ids = list(range(mention_start, mention_end))[:max_mention_length]
#         entity_position_ids += [-1] * (max_mention_length - mention_end + mention_start)
#         entity_position_ids = [entity_position_ids, [-1] * max_mention_length]
#         encoding['entity_position_ids'] = entity_position_ids

#         return encoding


def prepare_luke_batch_inputs():

    text = """Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."""
    span = (39,42)

    tokenizer = LukeTokenizer.from_pretrained("nielsr/luke-large")
    
    encoding = tokenizer(text, task="entity_typing", additional_info=span, return_tensors="pt")

    return encoding
    

@require_torch
class LukeModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = LukeEntityAwareAttentionModel.from_pretrained("nielsr/luke-large").eval()
        model.to(torch_device)
        
        tokenizer = LukeTokenizer.from_pretrained("nielsr/luke-large")
        encoding = prepare_luke_batch_inputs(tokenizer)
        # convert all values to PyTorch tensors
        for key, value in encoding.items():
            encoding[key] = encoding[key].to(torch_device)
        
        outputs = model(**encoding)
        
        # Verify word hidden states
        expected_shape = torch.Size((1, 42, 1024))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)
        
        expected_slice = torch.tensor([[ 0.0301,  0.0980,  0.0092],
            [ 0.2718, -0.2413, -0.9446],
            [-0.1382, -0.2608, -0.3927]])
            
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

        # Verify entity hidden states
        expected_shape = torch.Size((1, 2, 1024))
        
        self.assertTrue(outputs.entity_last_hidden_state.shape == expected_shape)
        expected_slice = torch.tensor([[ 0.3251,  0.3981, -0.0689],
            [-0.0098,  0.1215,  0.3544]])

        self.assertTrue(torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))
