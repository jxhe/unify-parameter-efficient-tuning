# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import json
import os
import unittest

from transformers import AddedToken, LukeTokenizer
from transformers.models.luke.tokenization_luke import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, slow

from .test_tokenization_common import TokenizerTesterMixin


class Luke(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LukeTokenizer
    from_pretrained_kwargs = {"cls_token": "<s>"}

    def setUp(self):
        super().setUp()

        # to be updated once files are on the hub
        self.vocab_file = os.path.join(r"C:\Users\niels.rogge\Documents\LUKE\tokenizer_files\vocab.json")
        self.merges_file = os.path.join(r"C:\Users\niels.rogge\Documents\LUKE\tokenizer_files\merges.txt")

    def get_tokenizer(self):
        return self.tokenizer_class(vocab_file=self.vocab_file, merges_file=self.merges_file)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)  # , add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def luke_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2])
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        )

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("roberta-base")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_text_from_decode = tokenizer.encode(
            "sequence builders", add_special_tokens=True, add_prefix_space=False
        )
        encoded_pair_from_decode = tokenizer.encode(
            "sequence builders", "multi-sequence build", add_special_tokens=True, add_prefix_space=False
        )

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == encoded_text_from_decode
        assert encoded_pair == encoded_pair_from_decode

    def test_space_encoding(self):
        tokenizer = self.get_tokenizer()

        sequence = "Encode this sequence."
        space_encoding = tokenizer.byte_encoder[" ".encode("utf-8")[0]]

        # Testing encoder arguments
        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=False)
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertNotEqual(first_char, space_encoding)

        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertEqual(first_char, space_encoding)

        tokenizer.add_special_tokens({"bos_token": "<s>"})
        encoded = tokenizer.encode(sequence, add_special_tokens=True)
        first_char = tokenizer.convert_ids_to_tokens(encoded[1])[0]
        self.assertNotEqual(first_char, space_encoding)

        # Testing spaces after special tokens
        mask = "<mask>"
        tokenizer.add_special_tokens(
            {"mask_token": AddedToken(mask, lstrip=True, rstrip=False)}
        )  # mask token has a left space
        mask_ind = tokenizer.convert_tokens_to_ids(mask)

        sequence = "Encode <mask> sequence"
        sequence_nospace = "Encode <mask>sequence"

        encoded = tokenizer.encode(sequence)
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertEqual(first_char, space_encoding)

        encoded = tokenizer.encode(sequence_nospace)
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertNotEqual(first_char, space_encoding)

    def test_pretokenized_inputs(self):
        pass

    def test_embeded_special_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest("{} ({})".format(tokenizer.__class__.__name__, pretrained_name)):
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = "A, <mask> AllenNLP sentence."
                tokens_p = tokenizer_p.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)

                # token_type_ids should put 0 everywhere
                self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

                # attention_mask should put 1 everywhere, so sum over length should be 1
                self.assertEqual(
                    sum(tokens_p["attention_mask"]) / len(tokens_p["attention_mask"]),
                )

                tokens_p_str = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])

                # Rust correctly handles the space before the mask while python doesnt
                self.assertSequenceEqual(tokens_p["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])

                self.assertSequenceEqual(
                    tokens_p_str, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"]
                )

    def test_entity_linking(self):
        tokenizer = self.get_tokenizer()
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."
        span = (39,42)
        
        encoding = tokenizer(sentence, task="entity_typing", additional_info=span)

        self.assertEqual(tokenizer.decode(encoding["input_ids"]), "<s>Top seed Ana Ivanovic said on Thursday [ENT]she[ENT] could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon.</s>")
        self.assertEqual(encoding["entity_ids"], [1])
        self.assertEqual(encoding["entity_attention_mask"], [1])
        self.assertEqual(encoding["entity_token_type_ids"], [0])

