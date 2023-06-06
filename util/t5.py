import numpy as np

def create_sentinel_ids(mask_indices, tokenizer, num_bins):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, (len(tokenizer) - num_bins - sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids


def filter_input_ids(input_ids, sentinel_ids, tokenizer):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
        [input_ids, np.full((batch_size, 1), tokenizer.eos_token_id, dtype=np.int32)], axis=-1
    )
    return input_ids


def random_spans_noise_mask(length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]