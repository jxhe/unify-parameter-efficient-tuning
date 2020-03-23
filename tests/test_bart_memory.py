import unittest

import torch

from tests.utils import require_torch, slow
from transformers import BartTokenizer, BartModel
from transformers.modeling_bart import shift_tokens_right

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@require_torch
class TestHface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        source_path = "test.source"
        cls.lns = [" " + x.rstrip() for x in open(source_path).readlines()][:6]
        tokenizer = BartTokenizer.from_pretrained('bart-large')
        dct = tokenizer.batch_encode_plus(cls.lns, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        cls.ids = dct['input_ids'].to(DEFAULT_DEVICE)
        cls.enc_mask = dct['attention_mask'].to(DEFAULT_DEVICE)
        cls.prev_output_tokens = shift_tokens_right(cls.ids, 1).to(DEFAULT_DEVICE)
        cls.model = BartForConditionalGeneration.from_pretrained('bart-large-cnn').half().to(DEFAULT_DEVICE).half()
        return cls



    @classmethod
    def setUpClass(cls):
        source_path = "test.source"
        cls.lns = [" " + x.rstrip() for x in open(source_path).readlines()][:6]
        tokenizer = BartTokenizer.from_pretrained('bart-large')
        dct = tokenizer.batch_encode_plus(cls.lns, max_length=100, return_tensors="pt", pad_to_max_length=True)
        cls.ids = dct['input_ids'].to(DEFAULT_DEVICE)
        cls.prev_output_tokens = shift_tokens_right(cls.ids, 1).to(DEFAULT_DEVICE)
        cls.model = BartModel.from_pretrained('bart-large').to(DEFAULT_DEVICE)
        #cls.lns = pickle_load('/Users/shleifer/transformers_fork/lns.pkl')
        return cls

    def test_hf_fwd_batch(self):
        bart = self.model
        bart.reset_logs()
        with torch.no_grad():
            bart(self.ids)
        try:
            log_df = bart.combine_logs()
            #log_df.to_csv('hf_batch_fwd_logs.csv')
            bart.save_logs('hf_batch_fwd_logs.txt')
            print(bart.summary)
        except AttributeError as e:
            print(e)

