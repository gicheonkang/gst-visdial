import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils import logging
from transformers import BertGenerationDecoder, BertGenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput
from transformers.models.bert.modeling_bert import BertEncoder
from models.vilbert_dialog import BertEmbeddingsDialog

class VisualDialogDecoder(nn.Module):
    def __init__(self, params):
        super(VisualDialogDecoder, self).__init__()
        self.params = params
        self.config = BertGenerationConfig.from_json_file(params['model_dec_config'])
        self.config.__dict__['cur_device'] = params["gpu_ids"][0]
        self.decoder = BertForSequenceGeneration.from_pretrained(
            "bert-base-uncased",
            config=self.config
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def forward(
        self,
        decoder_input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        loss_reduction=True
    ):

        # input_ids: [b, seq_len] -- torch.LongTensor
        # attention_mask: [b, seq_len] (0=masked or 1=not masked) -- torch.FloatTensor             
        # encoder_hidden_states: [b, seq_len, num_hid] -- torch.FloatTensor
        # encoder_attention_mask: [b, seq_len] (0=masked or 1=not masked) -- torch.FloatTensor
        # labels: [b, seq_len] (ignore_index=0)

        if (labels is None) and (decoder_input_ids is not None):
            # shifting tokens to left for labels
            labels = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            labels[:, :-1] = decoder_input_ids[:, 1:].clone()
            decoder_input_ids.masked_fill_(decoder_input_ids == self.config.eos_token_id, self.config.pad_token_id)

        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        loss = None
        if 'train' in self.params['mode'] or 'eval' in self.params['mode']: 
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            if loss_reduction:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')    
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )


class BertGenerationPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertGenerationConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BertForSequenceGeneration(BertGenerationPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertForSequenceGeneration` as a standalone, add `is_decoder=True.`")

        self.bert = BertGenerationEncoder(config)
        self.lm_head = BertGenerationOnlyLMHead(self.bert.embeddings.word_embeddings.weight)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        decoder_input_ids=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        lm_loss = None

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class BertGenerationEncoder(BertGenerationPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    This model should be used when leveraging Bert or Roberta checkpoints for the [`EncoderDecoderModel`] class as
    described in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
    by Sascha Rothe, Shashi Narayan, and Aliaksei Severyn.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsDialog(config)
        self.encoder = BertEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        decoder_input_ids=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`: `1` for
            tokens that are NOT MASKED, `0` for MASKED tokens.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        input_shape = decoder_input_ids.size()
        batch_size, seq_length = input_shape
        device = decoder_input_ids.device 

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = None
        if not use_cache:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=decoder_input_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertGenerationOnlyLMHead(nn.Module):
    def __init__(self, bert_model_embedding_weights):
        super().__init__()
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0)
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias



