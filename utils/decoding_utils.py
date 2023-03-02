import torch
import torch.nn.functional as F

def batch_top_k_top_p_sampling(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    We modified the code from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    Batched top-{k or p} sampling is available 

    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)  
    """
    assert logits.dim() == 2 
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = sorted_indices_to_remove.gather(-1, sorted_indices.argsort(-1))
        logits = logits.masked_fill(mask, filter_value)
    return logits


def batch_ngram_blocking(logits, enc_input_ids, dec_input_ids, ngram_size=0, filter_value=-float('Inf'), special_token_ids=(0, 100, 101, 102, 103)):
    """
    Code for N-gram repetition blocking in sequence generation
    The generated sequences does NOT overlap any N-grams in encoder input sequence
    """
    assert logits.dim() == 2
    if ngram_size > 0:
        batch_size = enc_input_ids.shape[0]

        generated_ngrams = [{} for _ in range(batch_size)]
        for idx in range(batch_size):
            gen_tokens = enc_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                # if ngram combination includes the special tokens, exclude the ngram
                if set(ngram) & set(special_token_ids):
                    continue
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        cur_len = dec_input_ids.shape[-1]
        banned_batch_tokens = [
            _get_generated_ngrams(
                generated_ngrams[idx], dec_input_ids[idx], ngram_size, cur_len
            )
            for idx in range(batch_size)
        ]
        for i, banned_tokens in enumerate(banned_batch_tokens):
            logits[i, banned_tokens] = filter_value
    return logits


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Code from https://github.com/huggingface/transformers/blob/96ac7549cbb1f1f39f624ad5d52fc07f1f9a2f51/src/transformers/generation_logits_process.py
    N-gram repetition blocking in sequence generation 
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])