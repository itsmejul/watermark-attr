import random
import unittest

from src.experiments.main.oracle_watermark_eval import select_indices, summarize


class OracleTests(unittest.TestCase):
    def test_grid_selection(self):
        for n in (100, 500, 1000, 5000, 10000, 50000):
            training, evaluation = select_indices(64000, n)
            expected = sorted(random.Random(48).sample(list(range(63800)), 63800)[:n])
            self.assertEqual(training, expected)
            local = sorted(random.Random(1234).sample(range(n), 1000)) if n > 1000 else range(n)
            self.assertEqual(evaluation, [expected[i] for i in local])
            self.assertEqual(len(evaluation), min(n, 1000))
            self.assertTrue(set(evaluation) <= set(training))

    def test_summary_and_ties(self):
        result = dict(correct_ranks=[1, 2, 1], n_candidates=100,
                      correct_k_ps=[1, 2, 3], top_k_p=[[1, 4], [4, 2], [4, 3]],
                      correct_scores=[.8, .3, .5],
                      top_scores=[[.8, .2], [.7, .3], [.5, .5]])
        summary = summarize(result)
        self.assertEqual(summary['acc_at_1'], 2 / 3)
        self.assertEqual(summary['selected_top1_accuracy'], 1 / 3)
        self.assertEqual(summary['acc_at_5'], 1)
        self.assertAlmostEqual(summary['mean_margin'], (.6 - .4) / 3)
        with_empty = summarize(result, ['', 'text', 'text'])
        self.assertEqual(with_empty['acc_at_1'], 1 / 3)
        self.assertEqual(with_empty['grid_rank_acc_at_1'], 2 / 3)
        self.assertEqual(with_empty['n_empty_texts'], 1)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            select_indices(64000, 64000)


if __name__ == '__main__':
    unittest.main()
