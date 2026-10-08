import unittest

from src.eval.open_metrics import evaluate_open_set


def fixture(pos, neg, ranks, predicted=None):
    return ({'n_candidates': 10, 'top_scores': [[x] for x in pos],
             'correct_ranks': ranks, 'correct_k_ps': [1] * len(pos),
             'top_k_p': [[x] for x in (predicted or [1] * len(pos))]},
            {'n_candidates': 10, 'top_scores': [[x] for x in neg]})


class OpenMetricsTests(unittest.TestCase):
    def test_standardization_uses_each_outputs_background(self):
        c, o = fixture([.5], [.8], [1])
        c.update(score_mean=[.1], score_std=[.1])
        o.update(score_mean=[.2], score_std=[.3])
        self.assertEqual(evaluate_open_set(c, o)['openacc@0.01'], 0)
        m = evaluate_open_set(c, o, score_type='standardized_maxq')
        self.assertAlmostEqual(m['threshold@0.01'], 2)
        self.assertEqual(m['openacc@0.01'], 1)
        o['score_std'] = [0]
        with self.assertRaises(ValueError):
            evaluate_open_set(c, o, score_type='standardized_maxq')

    def test_detection_and_attribution_share_threshold(self):
        c, o = fixture([.95, .8, .4, .2], [.9, .7, .3, .1], [1, 2, 1, 1])
        m = evaluate_open_set(c, o, [.25])
        self.assertEqual(m['threshold@0.25'], .7)
        self.assertEqual(m['fpr@0.25'], .25)
        self.assertEqual(m['tpr@0.25'], .5)
        self.assertEqual(m['openacc@0.25'], .25)

    def test_negative_ties_are_rejected_conservatively(self):
        c, o = fixture([.9, .8], [.8, .8, .2, .1], [1, 1])
        m = evaluate_open_set(c, o, [.25])
        self.assertEqual(m['fpr@0.25'], 0)
        self.assertEqual(m['openacc@0.25'], .5)

    def test_threshold_independent_of_positive_scores(self):
        c, o = fixture([.9], [.8, .6], [1])
        before = evaluate_open_set(c, o)
        c['top_scores'] = [[.1]]
        after = evaluate_open_set(c, o)
        self.assertEqual(before['threshold@0.01'], after['threshold@0.01'])

    def test_rank_tie_requires_actual_top_key(self):
        c, o = fixture([.9], [.2], [1], predicted=[2])
        m = evaluate_open_set(c, o)
        self.assertEqual(m['tpr@0.01'], 1)
        self.assertEqual(m['openacc@0.01'], 0)


if __name__ == '__main__':
    unittest.main()
