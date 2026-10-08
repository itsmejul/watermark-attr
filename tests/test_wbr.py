import unittest
from src.util.wbr import bigram_set, wbr_result


class WBRTests(unittest.TestCase):
    def test_excludes_original_and_counts_unique_bigrams(self):
        eligible = bigram_set([1, 2, 3, 4]) - bigram_set([1, 2, 9])
        r = wbr_result([eligible], [[2, 3, 2, 3]], [7], 'test')
        self.assertEqual(r['n_watermark_only_bigrams'], [2])
        self.assertEqual(r['wbr'], [.5])

    def test_empty_sets_excluded_and_macro_not_micro(self):
        r = wbr_result([set(), {(1, 2)}, {(1, 2), (2, 3)}],
                       [[], [1, 2], []], [1, 2, 3], 'test')
        self.assertEqual(r['wbr'], [None, 1, 0])
        self.assertEqual(r['mean_wbr'], .5)
        self.assertEqual(r['micro_wbr'], 1/3)
        self.assertEqual(r['n_empty_target'], 1)

    def test_alignment_and_all_empty(self):
        with self.assertRaises(ValueError):
            wbr_result([set()], [], [1], 'test')
        self.assertIsNone(wbr_result([set()], [[]], [1], 'test')['mean_wbr'])
