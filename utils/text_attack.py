import warnings
import os

import torch
import torch.nn as nn
import json
from transformers import BertConfig, BertForMaskedLM
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
from utils.data_utils import list2tensorpad


class TextAttack(object):
    """
    The Dialog Must Go On: Improving Visual Dialog via Generative Self-Training
    Gi-Cheon Kang, Sungdong Kim, Jin-Hwa Kim, Donghyun Kwak, Byoung-Tak Zhang
    https://arxiv.org/abs/2205.12502  
    """
    def __init__(self, mlm_path, device, cos_sim, cos_sim_idx2word, cos_sim_word2idx):
        # initialization of masked language model
        self.tokenizer = BertTokenizer.from_pretrained(mlm_path)
        config_atk = BertConfig.from_pretrained(mlm_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk)
        self.mlm_model.to(device)
        self.device = device
        self.cos_sim = cos_sim
        self.cos_sim_idx2word = cos_sim_idx2word
        self.cos_sim_word2idx = cos_sim_word2idx

    def random_token_attack(self, input_ids, token_type_ids, attention_mask):
        batch_size, seq_len = input_ids.size() # [100, 256]
        """
        We process 100 input sequences in the text-attack because 
        all 100 input sequences are the same except for the random mask.

        Why 100 input sequences are the same? 
        100 denotes the number of answer candidates for each question.
        The dialog history and the given question are the same for each dialog round.
        """ 
        masked_input = input_ids[:1, :]
        with torch.no_grad():
            logits = self.mlm_model(
                input_ids=masked_input, 
                attention_mask=attention_mask[:1, :], 
                token_type_ids=token_type_ids[:1, :]).logits

            mask_token_index = (masked_input == self.tokenizer.mask_token_id)
            try:
                # error occurred if all tokens are not masked
                predicted_token_id = logits[mask_token_index].argmax(axis=-1)
                masked_input[mask_token_index] = predicted_token_id
            except:
                pass
                
            perturbed_input_ids = masked_input.repeat(batch_size, 1)
            return perturbed_input_ids

    def coreference_attack(self, input_ids, sep_indices, coref_dependency):
        """
        substitute noun phrases or pronouns in the dialog history to fool the VisDial models.
        """
        batch_size, seq_len = input_ids.size() # [100, 256]
        unit_input_ids = input_ids[:1, :]

        num_coref = len(coref_dependency)
        if num_coref == 0: return input_ids
        else:
            for k, v in coref_dependency.items():
                target_round = int(k)
                target_word = v

                if target_word in self.cos_sim_word2idx:
                    synonym_words, _ = self.pick_most_similar_words_batch(
                        [self.cos_sim_word2idx[target_word]], 
                        self.cos_sim, 
                        self.cos_sim_idx2word
                    )
                    # we greedily perform synonym substitution to guarantee the semantic consistency of the original inputs.
                    synonym_word = synonym_words[0][0]
                else: continue

                if target_round == 0:
                    # substitution of image caption 
                    unit_input_ids = self.substitute_word(unit_input_ids, target_word, synonym_word, target_round)
                else:
                    # substitution of QA pairs
                    unit_input_ids = self.substitute_word(unit_input_ids, target_word, synonym_word, target_round*2-1)
                    unit_input_ids = self.substitute_word(unit_input_ids, target_word, synonym_word, target_round*2)

        perturbed_input_ids = unit_input_ids.repeat(batch_size, 1)
        return perturbed_input_ids

    def substitute_word(self, unit_input_ids, target_word, synonym_word, sep_indice, max_seq_len=256):
        input_str_split = self.tokenizer.decode(unit_input_ids[0].tolist())      
        input_str_split[sep_indice] = input_str_split[sep_indice].replace(target_word, synonym_word)
        input_str_split = self.tokenizer.tokenize('[SEP]'.join(input_str_split))
        input_str_split.insert(0, '[CLS]')
        unit_input_ids = self.tokenizer.convert_tokens_to_ids(input_str_split)
        unit_input_ids = list2tensorpad(unit_input_ids, max_seq_len)
        return unit_input_ids


    def pick_most_similar_words_batch(self, src_words, sim_mat, idx2word, ret_count=10, threshold=0.5):
        """
        We modify the source code from https://github.com/jind11/TextFooler/blob/master/attack_classification.py
        """
        sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(src_words):
            sim_value = sim_mat[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values    


