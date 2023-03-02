import copy
import json
import torch
from torch import nn
from models.vilbert_dialog import BertForMultiModalPreTraining, BertConfig

class VisualDialogEncoder(nn.Module):

    def __init__(self, params):
        super(VisualDialogEncoder, self).__init__()
        self.params = params
        self.config = BertConfig.from_json_file(params['model_enc_config'])
        self.config.__dict__['cur_device'] = params["gpu_ids"][0]
        self.config.__dict__['model_arch'] = params['model']
        self.config.__dict__['mode'] = params['mode']
        self.model_arch = params['model']
        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased', self.config)
        
    def forward(
        self, 
        input_ids, 
        image_feat, 
        image_loc, 
        sep_indices=None, 
        token_type_ids=None,
        attention_mask=None, 
        masked_lm_labels=None, 
        next_sentence_label=None, 
        image_attention_mask=None,
        image_label=None, 
        image_target=None
    ):                      

        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        prediction_scores_t = None
        seq_relationship_score = None
        enc_hidden_t = None
        enc_hidden_v = None

        if 'train' in self.params['mode']:
            if 'enc_dec' in self.model_arch:
                # train mode, enc_dec model, output hidden states
                enc_hidden_t, enc_hidden_v = \
                    self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, token_type_ids=token_type_ids, \
                        attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                        next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                        image_label=image_label, image_target=image_target)

            else: 
                # train mode, enc_only model, output losses
                masked_lm_loss, masked_img_loss, nsp_loss, _, prediction_scores_t, seq_relationship_score  = \
                    self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, token_type_ids=token_type_ids, \
                        attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                        next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                        image_label=image_label, image_target=image_target)

        else:
            if 'enc_dec' in self.model_arch:
                # eval or inference mode, enc_dec model, output hidden states
                enc_hidden_t, enc_hidden_v = \
                    self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, token_type_ids=token_type_ids, \
                        attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                        next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                        image_label=image_label, image_target=image_target)
                    
            else:
                # eval or inference mode, enc_only model, output
                prediction_scores_t, _, seq_relationship_score, _, _ = \
                    self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, token_type_ids=token_type_ids, \
                        attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                        next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                        image_label=image_label, image_target=image_target)

        return (masked_lm_loss, masked_img_loss, nsp_loss, seq_relationship_score, prediction_scores_t, enc_hidden_t, enc_hidden_v)

