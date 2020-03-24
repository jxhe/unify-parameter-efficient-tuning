import unittest

import torch
import os
from tests.utils import require_torch, slow
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "small_test.source"
SAVE_PREFIX =  os.getenv('SAVE_PREFIX', '')
from durbango.logging_utils import collect_log_data
from durbango import patch_module_with_memory_mixin

def save_logs_print_mem(bart, save_path):
    pth = SAVE_PREFIX + save_path
    print(f'*** {pth} ***')
    bart.save_logs(pth+'.txt')
    bart.save_log_csv(pth+'.csv')
    print(bart.summary)
    print(f'*** DONE ***')

class Memtest(unittest.TestCase):
    def setUp(self):
        if hasattr(self.model, 'reset_logs'): self.model.reset_logs()
        self.model.log_mem('start')
        torch.cuda.empty_cache()

class TestHface(Memtest):



    @classmethod
    def setUpClass(cls):
        cls.lns = [" " + x.rstrip() for x in open(DATA_PATH).readlines()][:6]
        tokenizer = BartTokenizer.from_pretrained('bart-large')
        dct = tokenizer.batch_encode_plus(cls.lns, max_length=12, return_tensors="pt", pad_to_max_length=True)
        cls.ids = dct['input_ids'].to(DEFAULT_DEVICE)
        cls.enc_mask = dct['attention_mask'].to(DEFAULT_DEVICE)
        cls.prev_output_tokens = shift_tokens_right(cls.ids, 1).to(DEFAULT_DEVICE)
        cls.model = BartForConditionalGeneration.from_pretrained('bart-large-cnn').to(DEFAULT_DEVICE)
        patch_module_with_memory_mixin(cls.model)
        return cls

    def test_hf_fwd(self):
        bart = self.model
        with torch.no_grad():
            self.model(self.ids, attention_mask=self.enc_mask, generation_mode=False)
        save_logs_print_mem(self.model, 'hf_fwd')


    def test_hf_short_generate(self):
        self.model.generate(self.ids, attention_mask=self.enc_mask, num_beams=4,
                            max_length=9, min_length=6,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            decoder_start_token_id=2,
                            )
        save_logs_print_mem(self.model, 'hf_short_generate')


    @slow
    def test_hf_generate(self):
        self.model.generate(self.ids, attention_mask=self.enc_mask, num_beams=4, max_length=140, min_length=56,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            decoder_start_token_id=2,
                            )
        save_logs_print_mem(self.model, 'hf_generate')


class TestFairseq(Memtest):

    @classmethod
    def setUpClass(cls):
        import fairseq
        source_path = "test.source"
        cls.lns = [" " + x.rstrip() for x in open(source_path).readlines()][:6]
        tokenizer = BartTokenizer.from_pretrained('bart-large')
        dct = tokenizer.batch_encode_plus(cls.lns, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        cls.ids = dct['input_ids'].to(DEFAULT_DEVICE)
        cls.prev_output_tokens = shift_tokens_right(cls.ids, 1).to(DEFAULT_DEVICE)
        cls.model = torch.hub.load('pytorch/fairseq', 'bart.large.cnn').eval().to(DEFAULT_DEVICE)
        patch_module_with_memory_mixin(cls.model)
        return cls

    def test_fs_fwd(self):
        bart = self.model
        with torch.no_grad():
            bart.model(self.ids, None, self.prev_output_tokens)
        save_logs_print_mem(bart, 'fs_fwd')

    def test_fs_short_gen(self):
        bart = self.model
        bart.sample(self.lns, beam=4, lenpen=2.0, max_len_b=7, min_len=5, no_repeat_ngram_size=3)
        save_logs_print_mem(bart, 'fs_short_generate')

    def test_fs_gen(self):
        bart = self.model
        bart.sample(self.lns, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        save_logs_print_mem(bart, 'fs_generate')
