from collections import namedtuple
import unittest

import numpy as np
import torch

from transformers import BeamSearch


StubTokenizer = namedtuple("Tokenizer", ["start_token_id", "end_token_id", "pad_token_id"])
StubTransformer = namedtuple("Transformer", ["encoder", "decoder"])


class BeamSearchtest(unittest.TestCase):
    def test_beam_search_min_length(self):
        """ We keep predicting the end_token for the first beam
        and check that it is only marked as finished once the beam
        has reached the minimum length.
        """
        vocab_size = 100
        min_length = 5
        batch_size = 3
        beam_size = 2
        eos_idx = 23
        model = StubTransformer('encoder', 'decoder')
        tokenizer = StubTokenizer(1, eos_idx, 2)
        beam = BeamSearch(model, tokenizer, batch_size, beam_size, min_length, 10, 0, False)

        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))

        valid_score_dist = torch.log_softmax(
            torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0
        )
        log_probabilities[0::beam_size, eos_idx] = valid_score_dist[0]
        non_eos_idxs = [47, 51, 13, 88, 99]
        for idx, score in zip(non_eos_idxs, valid_score_dist[1:]):
            log_probabilities[0::beam_size, idx] = score

        for step in range(1, min_length + 2):
            log_probabilities[::beam_size, eos_idx] = valid_score_dist[0]
            for k, (j, score) in enumerate(zip(non_eos_idxs, valid_score_dist[1:])):
                beam_idx = min(beam_size - 1, k)
                log_probabilities[beam_idx::beam_size, j] = score

            surviving_beams_rows = beam.grow(log_probabilities)
            print(surviving_beams_rows)
            if step < min_length:
                np.testing.assert_array_equal(
                    surviving_beams_rows.numpy(), torch.tensor([0, 0, 2, 2, 4, 4])
                )
            if step == min_length:
                np.testing.assert_array_equal(
                    surviving_beams_rows.numpy(), torch.tensor([3, 3, 3])
                )


if __name__ == "__name__":
    unittest.main()
