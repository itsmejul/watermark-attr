"""Token-preserving helpers for fixed-prefix watermark augmentation."""


class PrefixWatermarkProcessor:
    """Let Waterfall use the prefilled tokens as n-gram context after reset."""

    def __init__(self, processor, prefix_tokens):
        self.processor = processor
        self.prefix_tokens = prefix_tokens

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def reset(self, *args, **kwargs):
        self.processor.reset(*args, **kwargs)

    def __call__(self, input_ids, scores):
        if self.processor.init_token_count is None:
            self.processor.init_token_count = input_ids.shape[1] - self.prefix_tokens
        return self.processor(input_ids, scores)


def prefix_ids(tokenizer, text, prefix):
    ids = tokenizer.encode(text, max_length=10, truncation=True)
    if len(ids) != 10 or tokenizer.decode(ids, skip_special_tokens=True) != prefix:
        raise ValueError('Saved prefix does not match the first ten source tokens')
    if not text.startswith(prefix):
        raise ValueError('Saved prefix is not a literal prefix of the source text')
    return ids


def join_continuation(tokenizer, prefix, generated_ids):
    # Do not use Waterfall's stripped text: the first continuation token may
    # contain the space separating the prefix from the next word.
    suffix = tokenizer.decode(generated_ids, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
    if not suffix.strip():
        raise ValueError('Empty continuation (EOS or discarded incomplete generation)')
    result = prefix + suffix
    prefix_ids(tokenizer, result, prefix)
    return result
