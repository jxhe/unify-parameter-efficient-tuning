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
        vocab_size = 10
        min_length = 5
        batch_size = 3
        beam_size = 2
        eos_idx = 3

        beam = BeamSearch(
            model=StubTransformer("encoder", "decoder"),
            tokenizer=StubTokenizer(start_token_id=0, end_token_id=eos_idx, pad_token_id=2),
            batch_size=batch_size,
            beam_size=beam_size,
            min_length=5,
            max_length=10,
            alpha=0,
            block_repeating_trigrams=False,
        )

        # To test that the minimum length is correctly enforced we constantly
        # assign the highest probability to the [EOS] token (and assign lower
        # probabilities to some other tokens.
        # Since BeamSearch will reset its probability to 1e-20 as long as
        # min_length has not been reached, we need to reset the value between
        # steps.
        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))
        non_eos_idxs = [4, 5, 1, 8, 9]
        valid_score_dist = torch.log_softmax(
            torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0
        )
        log_probabilities[0, eos_idx] = valid_score_dist[0]
        for idx, score in zip(non_eos_idxs, valid_score_dist[1:]):
            log_probabilities[0, idx] = score

        for step in range(1, min_length + 2):
            log_probabilities[0, eos_idx] = valid_score_dist[0]

            # Beam #2 and #3 will finish at the first step since the probability
            # of the [EOS] token is still > -\infty
            surviving_beams_rows = beam.grow(log_probabilities)
            if step < min_length:
                np.testing.assert_array_equal(
                    beam.growing_beam.numpy(),
                    np.repeat(np.array([[0] + [4] * step]), 2, axis=0),
                )
            elif step == min_length:  # Now [EOS] is the most probable token
                np.testing.assert_array_equal(surviving_beams_rows.numpy(), np.array([]))
                self.assertTrue(beam.is_done)
                break

            log_probabilities = log_probabilities.index_select(0, surviving_beams_rows)

    def test_beam_search_max_length(self):
        """ We keep predicting the same non-EOS token until we reach the
        maximum permitted length """
        beam_size = 2
        batch_size = 3
        vocab_size = 10
        max_length = 5

        beam = BeamSearch(
            model=StubTransformer("encoder", "decoder"),
            tokenizer=StubTokenizer(start_token_id=0, end_token_id=1, pad_token_id=2),
            batch_size=batch_size,
            beam_size=beam_size,
            min_length=2,
            max_length=max_length,
            alpha=0,
            block_repeating_trigrams=False,
        )

        log_probabilities = torch.full((batch_size * beam_size, vocab_size), float("-inf"))

        # To test that beam search enforces the max length constraint we
        # keep giving the highest probability to a token that is not the
        # [EOS] token.
        # The beam search will stop at max_length-1, assuming that one would
        # add the [EOS] token at the end of the returned sequence.
        token_idxs = [3, 4, 5]
        valid_score_dist = torch.log_softmax(torch.tensor([10.0, 6.0, 4.0]), dim=0)
        for idx, score in zip(token_idxs, valid_score_dist[1:]):
            log_probabilities[:, idx] = score

        for step in range(1, max_length + 2):
            surviving_beams_rows = beam.grow(log_probabilities)
            if step + 1 < max_length:
                self.assertFalse(beam.is_done)
            elif step + 1 == max_length:  # Now [EOS] is the most probable token
                np.testing.assert_array_equal(surviving_beams_rows.numpy(), np.array([]))
                self.assertTrue(beam.is_done)
                break

            log_probabilities = log_probabilities.index_select(0, surviving_beams_rows)


if __name__ == "__name__":
    unittest.main()
