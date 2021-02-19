# coding=utf-8
# Copyright Studio-Ouisa and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for LUKE."""
from ...utils import logging
from transformers import RobertaTokenizer

from typing import Optional, Union, Tuple, List, Dict

from ...file_utils import add_end_docstrings
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "studio-ouisa/luke-base": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "studio-ouisa/luke-large": "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    },
    "merges_file": {
        "studio-ouisa/luke-base": "https://huggingface.co/roberta-base/resolve/main/merges.txt",
        "studio-ouisa/luke-large": "https://huggingface.co/roberta-large/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "studio-ouisa/luke-base": 512,
    "studio-ouisa/luke-large": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "studio-ouisa/luke-base": {"do_lower_case": False},
    "studio-ouisa/luke-large": {"do_lower_case": False},
}


class LukeTokenizer(RobertaTokenizer):
    r"""
    Construct a LUKE tokenizer.  

    This tokenizer inherits from :class:`~transformers.RobertaTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods. Compared to :class:`~transformers.RobertaTokenizer`,
     :class:`~transformers.LukeTokenizer` also creates :obj:`entity_ids`, :obj:`entity_attention_mask`, :obj:`entity_token_type_ids` 
     and :obj:`entity_position_ids` to be used by the entity-aware attention mechanism of LUKE. 
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file,
        merges_file,
        model_max_length=512,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs
    ):

        # we add 2 special tokens for downstream tasks 
        # for more information about lstrip and rstrip, see https://github.com/huggingface/transformers/pull/2778       
        entity_token_1 = AddedToken(entity_token_1, lstrip=False, rstrip=False) if isinstance(entity_token_1, str) else entity_token_1
        entity_token_2 = AddedToken(entity_token_2, lstrip=False, rstrip=False) if isinstance(entity_token_2, str) else entity_token_2
        additional_special_tokens = [entity_token_1, entity_token_2]

        super().__init__(vocab_file=vocab_file,
                        merges_file=merges_file,
                        model_max_length=model_max_length,
                        additional_special_tokens=additional_special_tokens,
                        **kwargs)

        self.max_entity_length = max_entity_length
        self.max_mention_length = max_mention_length
    
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        task: Optional[str] = None,
        additional_info: Optional[Union[Tuple, List[Tuple], List[List[Tuple]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_entity_position_ids: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences, depending on the task you want to prepare them for.
        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            task (:obj:`str`, `optional`):
                Task for which you want to prepare sequences for. One of 'entity typing' or 'relation classification'.
                If no task is provided, then no :obj:`entity_ids`, :obj:`entity_attention_mask`, :obj:`entity_token_type_ids` 
                and :obj:`entity_position_ids` will be created.
            additional_info (:obj:`Tuple`, :obj:`List[Tuple]`, :obj:`List[List[Tuple]]`, `optional`):
                Additional information regarding entities to provide to prepare sequences. 
        """
        # Input type checking for clearer error
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple))
            and (
                len(text) == 0
                or (
                    isinstance(text[0], str)
                    or (isinstance(text[0], (list, tuple)) and (len(text[0]) == 0 or isinstance(text[0][0], str)))
                )
            )
        ), (
            "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples)."
        )

        assert (
            text_pair is None
            or isinstance(text_pair, str)
            or (
                isinstance(text_pair, (list, tuple))
                and (
                    len(text_pair) == 0
                    or (
                        isinstance(text_pair[0], str)
                        or (
                            isinstance(text_pair[0], (list, tuple))
                            and (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str))
                        )
                    )
                )
            )
        ), (
            "text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples)."
        )

        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple)))
            or (
                is_split_into_words and isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
            )
        )

        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                task=task,
                additional_info=additional_info,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_entity_position_ids=return_entity_position_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                task=task,
                additional_info=additional_info,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_entity_position_ids=return_entity_position_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        task: Optional[str] = None,
        additional_info: Optional[Union[Tuple, List[Tuple], List[List[Tuple]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_entity_position_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.
        .. warning::
            This method is deprecated, ``__call__`` should be used instead.
        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                ``tokenize`` method) or a list of integers (tokenized string ids using the ``convert_tokens_to_ids``
                method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            task=task,
            additional_info=additional_info,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_entity_position_ids=return_entity_position_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        task: Optional[str] = None,
        additional_info: Optional[Tuple] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_entity_position_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                    )
        
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids, second_ids, entity_ids, entity_position_ids = None, None, None, None
        if task is None:
            first_ids = get_input_ids(text)
            second_ids = get_input_ids(text_pair) if text_pair is not None else None
        
        elif task == "entity_typing":
            self.max_entity_length = 1
            # additional information: span, which is a tuple (begin, end)
            assert isinstance(additional_info, tuple), "Additional info should be provided as a tuple containing the start and end character indices of an entity"
            span = additional_info
            
            ENTITY_TOKEN = self.additional_special_tokens[0]
            
            conv_tables = (
                ("-LRB-", "("),
                ("-LCB-", "("),
                ("-LSB-", "("),
                ("-RRB-", ")"),
                ("-RCB-", ")"),
                ("-RSB-", ")"),
            )
            
            def preprocess_and_tokenize(text, start, end=None):
                target_text = text[start:end]
                for a, b in conv_tables:
                    target_text = target_text.replace(a, b)

                return self.tokenize(target_text.strip(), add_prefix_space=True)

            tokens = []
            tokens += preprocess_and_tokenize(text, 0, span[0])
            mention_start = len(tokens)
            tokens.append(ENTITY_TOKEN)
            tokens += preprocess_and_tokenize(text, span[0], span[1])
            tokens.append(ENTITY_TOKEN)
            mention_end = len(tokens)
 
            tokens += preprocess_and_tokenize(text, span[1])

            first_ids, second_ids = self.convert_tokens_to_ids(tokens), None

            entity_ids = [1]
            entity_position_ids = list(range(mention_start, mention_end))[:self.max_mention_length]
            entity_position_ids += [-1] * (self.max_mention_length - mention_end + mention_start)
            entity_position_ids = [entity_position_ids]
            #entity_position_ids = [entity_position_ids, [-1] * self.max_mention_length]

        elif task == "relation_classification":
            self.max_entity_length = 2
            # additional information: span_a and span_b (both should be tuples of begin and end)
            assert isinstance(additional_info, list) and isinstance(additional_info[0], tuple) and isinstance(additional_info[1], tuple), "Additional info should be provided as a list of tuples, each tuple containing the start and end character indices of an entity"
            span_a = additional_info[0]
            span_b = additional_info[1]

            HEAD_TOKEN = self.additional_special_tokens[0]
            TAIL_TOKEN = self.additional_special_tokens[1]
            
            if span_a[1] < span_b[1]:
                span_order = [("span_a", span_a), ("span_b", span_b)]
            else:
                span_order = [("span_b", span_b), ("span_a", span_a)]

            tokens = []
            cur = 0
            token_spans = {}
            for span_tuple in span_order:
                span_name = span_tuple[0]
                span = span_tuple[1]
                tokens += self.tokenize(text[cur : span[0]])
                start = len(tokens)
                tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
                tokens += self.tokenize(text[span[0] : span[1]])
                tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
                token_spans[span_name] = (start, len(tokens))
                cur = span[1]

            tokens += self.tokenize(text[cur:])

            first_ids, second_ids = self.convert_tokens_to_ids(tokens), None

            entity_ids = [1, 2]
            entity_position_ids = []
            for span_name in ("span_a", "span_b"):
                span = token_spans[span_name]
                position_ids = list(range(span[0], span[1]))[:self.max_mention_length]
                position_ids += [-1] * (self.max_mention_length - span[1] + span[0])
                entity_position_ids.append(position_ids)

        else:
            raise ValueError(f"Task {task} not supported")

        # prepare_for_model will create the attention_mask and token_type_ids
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_entity_position_ids=return_entity_position_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        task: str = None,
        additional_info: Optional[Tuple] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_entity_position_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # input_ids is a list of tuples (one for each example in the batch)
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        # entity_ids should become a list of lists
        entity_ids = []

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            entity_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_entity_position_ids=return_entity_position_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        batch_entity_ids: List[List[int]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_entity_position_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens
        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for input_ids, entity_ids in zip(batch_ids_pairs, batch_entity_ids):
            first_ids, second_ids = input_ids
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                entity_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_entity_position_ids=return_entity_position_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs


    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        entity_ids: Optional[List[int]] = None,
        entity_position_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_entity_position_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids together with entity ids so that it can be used 
        by the model. It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens
        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            pair_ids (:obj:`List[int]`, `optional`):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            entity_ids (:obj:`List[int]`, `optional`):
                Tokenized entity ids.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # Compute lengths
        pair = bool(pair_ids is not None)
        entities_provided = bool(entity_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0
        len_entity_ids = len(entity_ids) if entities_provided else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned word encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0) 

        # Set max entity length
        if not max_entity_length:
            max_entity_length = self.max_entity_length
        
        # Truncation: Handle max sequence length and max_entity_length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            # truncate words up to max_length
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

            # truncate entities up to max_entity_length
            if entities_provided:
               entity_ids, overflowing_entity_tokens = self.truncate_sequences(
                    entity_ids,
                    pair_ids=None,
                    num_tokens_to_remove=len_entity_ids - max_entity_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
               )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length
            if entities_provided:
                encoded_inputs["overflowing_entity_tokens"] = overflowing_entity_tokens
                encoded_inputs["num_truncated_tokens"] = len_entity_ids - max_entity_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            if entities_provided:
                entity_token_type_ids = [0] * len(entity_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            if entities_provided:
                entity_token_type_ids = [0] * len(entity_ids)

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if entities_provided:
            encoded_inputs["entity_ids"] = entity_ids
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
            if entities_provided:
                encoded_inputs["entity_token_type_ids"] = entity_token_type_ids
        if return_entity_position_ids and entities_provided:
            encoded_inputs["entity_position_ids"] = entity_position_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        # To do: add padding of entities
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
        
        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.
                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs["input_ids"])
            max_entity_length = len(encoded_inputs["entity_ids"])

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # Set max entity length
        if not max_entity_length:
            max_entity_length = self.max_entity_length
        
        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD and 
                (len(encoded_inputs["input_ids"]) != max_length or len(encoded_inputs["entity_ids"]) != max_entity_length)
        )

        entities_provided = bool("entity_ids" in encoded_inputs)        
        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs["input_ids"]) 
            if entities_provided:
                entity_difference = max_entity_length - len(encoded_inputs["entity_ids"])
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = [1] * len(encoded_inputs["entity_ids"]) + [0] * entity_difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"] + [0] * difference
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = encoded_inputs["entity_token_type_ids"] + [0] * entity_difference
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [0] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
                if entities_provided:
                   encoded_inputs["entity_attention_mask"] = [1] * len(encoded_inputs["entity_ids"])

        return encoded_inputs
