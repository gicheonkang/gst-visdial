import torch
import torch.nn.functional as F
from torch import nn
from transformers import EncoderDecoderConfig
from transformers.modeling_utils import PreTrainedModel
from utils.decoding_utils import batch_top_k_top_p_sampling, batch_ngram_blocking

class EncoderDecoderModel(PreTrainedModel):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.
    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, params, encoder, decoder):
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config) 
        super().__init__(config)
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.vlfusion = VLFusion(encoder.config)

    def forward(
        self,
        enc_image_features=None,
        enc_image_spatials=None,
        enc_image_mask=None,
        enc_image_target=None,
        enc_image_label=None,
        enc_next_sentence_labels=None,
        enc_input_ids=None,
        enc_segments=None,
        enc_sep_indices=None,
        enc_mlm_labels=None,
        enc_attention_mask=None,
        dec_input_ids=None,
        dec_attention_mask=None,
        dec_labels=None,
        loss_reduction=True,
        **decoding_kwargs
    ):

        # GST model
        # encoder-decoder; generative setting
        _, _, _, _, _, enc_hidden_t, enc_hidden_v = self.encoder(
            enc_input_ids, 
            enc_image_features, 
            enc_image_spatials, 
            sep_indices=enc_sep_indices, 
            token_type_ids=enc_segments, 
            masked_lm_labels=enc_mlm_labels,
            attention_mask=enc_attention_mask,
            next_sentence_label=enc_next_sentence_labels, 
            image_attention_mask=enc_image_mask, 
            image_label=enc_image_label, 
            image_target=enc_image_target
        )

        enc_hidden_states, enc_attention_mask = \
            self.vlfusion(enc_hidden_t, enc_hidden_v, enc_attention_mask, enc_image_mask) 

        if 'train' in self.params['mode'] or 'eval' in self.params['mode']: 
            decoder_outputs = self.decoder(
                decoder_input_ids=dec_input_ids,
                labels=dec_labels,
                attention_mask=dec_attention_mask,
                encoder_hidden_states=enc_hidden_states,
                encoder_attention_mask=enc_attention_mask,
                loss_reduction=loss_reduction
            )
            return decoder_outputs.loss, decoder_outputs.logits     

        else:
            # decode mode
            # we use top-p sampling considering computational cost and trustworthy results
            max_seq_len = 18
            batch_size = enc_input_ids.shape[0]
            sequence = []

            temperature = decoding_kwargs['temperature']             
            top_k = decoding_kwargs['top_k']
            top_p = decoding_kwargs['top_p']
            ngram_blocking_size = decoding_kwargs['ngram_blocking_size']

            for i in range(max_seq_len): 
                decoder_outputs = self.decoder(
                    decoder_input_ids=dec_input_ids,
                    attention_mask=None,
                    encoder_hidden_states=enc_hidden_states,
                    encoder_attention_mask=enc_attention_mask
                )

                logits_with_temp = decoder_outputs.logits[:, -1, :] / temperature

                # blocking N-gram combinations in encoder input sequences to avoid repetitive generation
                # zero-valued entry in segments denote previously generated questions
                hist_ques_indices = (enc_segments == 0).long()
                hist_ques = enc_input_ids * hist_ques_indices 
                filtered_logits = batch_ngram_blocking(logits_with_temp, hist_ques, dec_input_ids, ngram_size=ngram_blocking_size)

                # decoding strategy: top-k or top-p sampling
                filtered_logits = batch_top_k_top_p_sampling(filtered_logits, top_k=top_k, top_p=top_p)

                # compute word probabilities and sample
                prob = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(prob, 1)

                dec_input_ids = torch.cat((dec_input_ids, next_token), dim=-1)
                sequence.append(next_token)
            sequence = torch.cat(sequence, 1)

            # fill [PAD] token after end-of-sentence (i.e., [SEP] token in here)
            seq_after_eos = (sequence == self.decoder.config.eos_token_id).nonzero(as_tuple=False)
            mask = torch.zeros_like(sequence)
            for eos in seq_after_eos:
                mask[eos[0], eos[1]+1:] = True    

            sequence.masked_fill_(mask.bool(), self.decoder.config.pad_token_id)
            return sequence
            

class VLFusion(nn.Module):
    def __init__(self, config):
        super(VLFusion, self).__init__()
        self.config = config
        self.fc_l = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_v = nn.Linear(config.v_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)    

    def forward(self, enc_hidden_t, enc_hidden_v, enc_attention_mask, enc_image_mask):
        enc_hidden = torch.cat((self.fc_v(enc_hidden_v), self.fc_l(enc_hidden_t)), dim=1)
        enc_hidden = self.dropout(enc_hidden) # [b, 293, 768]
        enc_attention_mask = torch.cat((enc_image_mask, enc_attention_mask), dim=1)
        return enc_hidden, enc_attention_mask








        
