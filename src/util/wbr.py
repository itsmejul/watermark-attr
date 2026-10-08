"""Watermark Bigram Recall using unique tokenizer bigrams, no special tokens."""


def bigram_set(tokens):
    return set(zip(tokens, tokens[1:]))


def wbr_result(eligible_sets, answer_tokens, correct_k_ps, tokenizer_name):
    if not (len(eligible_sets) == len(answer_tokens) == len(correct_k_ps)):
        raise ValueError('WBR inputs must be document-aligned')
    denominators = [len(s) for s in eligible_sets]
    hits = [len(s & bigram_set(a)) for s, a in zip(eligible_sets, answer_tokens)]
    recalls = [h / n if n else None for h, n in zip(hits, denominators)]
    valid = [r for r in recalls if r is not None]
    return {
        'tokenizer': tokenizer_name, 'definition': 'unique B(watermarked) minus B(original)',
        'aggregation': 'macro mean over documents with nonempty eligible bigram sets',
        'empty_target_policy': 'null per document; excluded from macro mean',
        'n': len(recalls), 'n_valid': len(valid), 'n_empty_target': len(recalls) - len(valid),
        'correct_k_ps': list(map(int, correct_k_ps)),
        'n_watermark_only_bigrams': denominators,
        'overlap_watermark_only_bigrams': hits,
        'wbr': recalls, 'mean_wbr': sum(valid) / len(valid) if valid else None,
        'micro_wbr': sum(hits) / sum(denominators) if sum(denominators) else None,
    }
