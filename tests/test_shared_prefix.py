import unittest
from types import SimpleNamespace

from src.util.shared_prefix import PrefixWatermarkProcessor, join_continuation, prefix_ids
from src.data_creation.create_qwen_prefix_variants import check_texts, variant_manifest


class CharacterTokenizer:
    def encode(self, text, max_length=None, truncation=False):
        return list(text)[:max_length] if truncation else list(text)

    def decode(self, ids, **kwargs):
        return ''.join(ids)


class FakeProcessor:
    def reset(self, n_gram=2):
        self.init_token_count = None
        self.n_gram = n_gram

    def __call__(self, input_ids, scores):
        return self.init_token_count + self.n_gram - 1 <= input_ids.shape[1]


class SharedPrefixTests(unittest.TestCase):
    def test_join_preserves_space_and_prefix(self):
        text = join_continuation(CharacterTokenizer(), '0123456789', list(' next word'))
        self.assertEqual(text, '0123456789 next word')

    def test_no_space_is_invented_inside_word(self):
        self.assertEqual(join_continuation(CharacterTokenizer(), '0123456789', list('suffix')),
                         '0123456789suffix')

    def test_invalid_prefix_and_empty_suffix(self):
        with self.assertRaises(ValueError):
            prefix_ids(CharacterTokenizer(), '0123456789 rest', 'wrong')
        with self.assertRaises(ValueError):
            join_continuation(CharacterTokenizer(), '0123456789', [])

    def test_first_continuation_token_is_watermarked_after_each_reset(self):
        original = FakeProcessor()
        wrapped = PrefixWatermarkProcessor(original, 10)
        for length in (50, 80):
            wrapped.reset(2)
            self.assertTrue(wrapped(SimpleNamespace(shape=(1, length)), None))
            self.assertEqual(original.init_token_count, length - 10)

    def test_requested_bases_and_reproducible_document_seeds(self):
        rows = [dict(document_index=i) for i in range(1000)]
        seeds = []
        for v in (2, 3, 4, 5):
            m = variant_manifest({'generation_seed': 20260920}, rows, v)
            self.assertEqual(m, variant_manifest({'generation_seed': 20260920}, rows, v))
            self.assertEqual(m['generation_seed_base'], 199 + v)
            self.assertEqual(m['generation_seeds'], [199 + v + i for i in range(1000)])
            seeds.append(m['generation_seeds'])
        for per_document in zip(*seeds):
            self.assertEqual(len(set(per_document)), 4)

    def test_partial_and_complete_validation(self):
        rows = [dict(prefix='prefix', document_index=i) for i in (1, 2)]
        check_texts(['prefix continuation'], rows)
        with self.assertRaises(ValueError):
            check_texts(['prefix continuation'], rows, complete=True)
        with self.assertRaises(ValueError):
            check_texts(['wrong continuation'], rows)

    def test_llama_manifest_is_separate_and_preserves_legacy_length(self):
        rows = [dict(document_index=3)]
        manifest = variant_manifest({'kappa': 6}, rows, 2, source='llama')
        self.assertEqual(manifest['source'], 'llama')
        self.assertEqual(manifest['generation_seeds'], [201])
        self.assertIn('legacy prompt-character', manifest['length_policy'])
        self.assertIn('BOS', manifest['prefix_policy'])

    def test_llama_prefix_includes_bos_in_ten_token_budget(self):
        class BosTokenizer(CharacterTokenizer):
            def encode(self, text, max_length=None, truncation=False, add_special_tokens=True):
                ids = (['BOS'] if add_special_tokens else []) + list(text)
                return ids[:max_length] if truncation else ids

            def decode(self, ids, **kwargs):
                return ''.join(i for i in ids if i != 'BOS')

        tok = BosTokenizer()
        self.assertEqual(len(prefix_ids(tok, '123456789 rest', '123456789')), 10)
        self.assertEqual(len(tok.encode('123456789', add_special_tokens=False)), 9)
        self.assertEqual(join_continuation(tok, '123456789', list(' rest')), '123456789 rest')


if __name__ == '__main__':
    unittest.main()
