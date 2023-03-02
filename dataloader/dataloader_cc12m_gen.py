import torch
from torch.utils import data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import random
from utils.data_utils import list2tensorpad, encode_input, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader


class CC12mDataset(data.Dataset):
    def __init__(self, params):
        self.params = params
        assert params['cc12m_image_feats'] != ""
        self._image_features_reader = ImageFeaturesH5Reader(params['cc12m_image_feats'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = ['[CLS]','[MASK]','[SEP]', '[PAD]', '[UNK]']
        self.CLS, self.MASK, self.SEP, self.PAD, self.UNK = self.tokenizer.convert_tokens_to_ids(tokens)
        self.cc12m_caption = None
        self.cc12m_data_train = None

        assert params['cc12m_processed_train'] != "" or params['cc12m_caption'] != ""
        if params['cc12m_caption']:
            with open(params['cc12m_caption']) as f:
                self.cc12m_caption = json.load(f)
                self.numDataPoints = len(self.cc12m_caption)
                assert self.numDataPoints == len(self._image_features_reader._image_ids) # for sanity check
        else:
            with open(params['cc12m_processed_train']) as f:
                self.cc12m_data_train = json.load(f)
                self.numDataPoints = len(self.cc12m_data_train)
            
        self._mode = 'cc12m_gen'
        self.subsets = ['cc12m_gen', 'cc12m_train']

        self._max_region_num = 37
        self._max_seq_len = self.params['max_seq_len']
        self._max_utt_len = self.params['max_utt_len']


    def __len__(self):
        return self.numDataPoints

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        assert mode in self.subsets
        self._mode = mode


    def __getitem__(self, index):
        if self._mode == 'cc12m_gen':
        	# questioner-answerer dialog generation mode on CC12M
            # returns image feature and caption
            cur_data = self.cc12m_caption[index]
            # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[cur_data['image_id']]
            features, spatials, image_mask, _, _ = encode_image_input(
                features, 
                num_boxes, 
                boxes,
                image_target, 
                max_regions=self._max_region_num,
                mask_prob=0
            )
            item = {}
            item['enc_image_feat'] = features
            item['enc_image_loc'] = spatials
            item['enc_image_mask'] = image_mask
            item['image_id'] = torch.LongTensor([cur_data['image_id']])

            max_cap_len = 38
            context_utterance = []
            tokenized_caption = self.tokenizer.encode(cur_data['caption'])
            if len(tokenized_caption) > max_cap_len:
                tokenized_caption = tokenized_caption[:max_cap_len]
            context_utterance.append(tokenized_caption)

            start_segment = 1    
            enc_input_ids, enc_segments, enc_sep_indices, _, enc_att_masks = encode_input(
                context_utterance, 
                start_segment, 
                self.CLS,
                self.SEP, 
                self.MASK,
                self.PAD, 
                max_seq_len=self._max_seq_len, 
                mask_prob=0,
            )
            dec_input_ids = torch.ones((1), dtype=torch.long) * self.CLS
            dec_att_masks = (dec_input_ids!=0).float()

            item['enc_input_ids'] = enc_input_ids
            item['enc_segments'] = enc_segments
            item['enc_sep_indices'] = enc_sep_indices
            item['enc_att_mask'] = enc_att_masks
            item['dec_input_ids'] = dec_input_ids
            item['dec_att_mask'] = dec_att_masks
            return item
        else:
            # answerer train mode on CC12M
            # returns image feature, dialog history including current question, and target answer
            full_utterances = []
            context_utterances = []
            target_utterances = []
            answer_ppls = []
            max_cap_len = 38

            cur_data = self.cc12m_data_train[index]
            tokenized_caption = self.tokenizer.encode(cur_data['caption'])
            if len(tokenized_caption) > max_cap_len:
                tokenized_caption = tokenized_caption[:max_cap_len]
            full_utterances.append([tokenized_caption]) 
            context_utterances.append([tokenized_caption])
        
            # -------------------------------------------------------------------
            # Structuring dialog data
            # -------------------------------------------------------------------
            for rnd,utterance in enumerate(cur_data['dialog']):
                full_utterance = full_utterances[-1].copy()
                context_utterance = full_utterances[-1].copy()
                target_utterance = []
                
                tokenized_question = self.tokenizer.encode(utterance['question'])
                full_utterance.append(tokenized_question) 
                context_utterance.append(tokenized_question)
                
                tokenized_answer = self.tokenizer.encode(utterance['answer'])
                full_utterance.append(tokenized_answer)
                
                if len(tokenized_answer) > self._max_utt_len -2:
                    tokenized_answer = tokenized_answer[:self._max_utt_len -2]
                target_utterance.append(tokenized_answer)

                full_utterances.append(full_utterance)
                context_utterances.append(context_utterance)
                target_utterances.append(target_utterance)
                answer_ppls.append(utterance['answer_ppl'])
                
            # removing the caption in the beginning
            context_utterances = context_utterances[1:]
            assert len(context_utterances) == len(target_utterances) == 10
            
            enc_input_ids_all_rnd = []
            enc_segments_all_rnd = []
            enc_sep_indices_all_rnd = []
            enc_mlm_labels_all_rnd = []
            enc_next_labels_all_rnd = []
            enc_hist_len_all_rnd = []
            enc_att_masks_all_rnd = []
            dec_input_ids_all_rnd = []
            dec_att_masks_all_rnd = []
            dec_labels_all_rnd = []

            for j in range(10):
                enc_input_ids_rnd = []
                enc_segments_rnd = []
                enc_sep_indices_rnd = []
                enc_mlm_labels_rnd = []
                enc_next_labels_rnd = []
                enc_hist_len_rnd = []
                enc_att_masks_rnd = []
                dec_input_ids_rnd = []
                dec_att_masks_rnd = []
                dec_labels_rnd = []

                start_segment = 1    
                enc_input_ids, enc_segments, enc_sep_indices, enc_mlm_labels, enc_att_masks = encode_input(
                    context_utterances[j], 
                    start_segment, 
                    self.CLS,
                    self.SEP, 
                    self.MASK,
                    self.PAD, 
                    max_seq_len=self._max_seq_len, 
                    mask_prob=self.params['mask_prob'],
                )

                dec_input_ids, _, _, _, dec_att_masks = encode_input(
                    target_utterances[j], 
                    start_segment, 
                    self.CLS,
                    self.SEP, 
                    self.MASK,
                    self.PAD, 
                    max_seq_len=self._max_utt_len,
                    mask_prob=0,
                )

                # label masking whose ppl score is over than threshold
                # the lower the better for ppl
                if self.params['select_data'] and answer_ppls[j] >= self.params['threshold']:
                    dec_labels = dec_input_ids.new_zeros(dec_input_ids.shape)
                else:
                    # delete CLS (SOS) token
                    dec_labels = dec_input_ids.new_zeros(dec_input_ids.shape)
                    dec_labels[:, :-1] = dec_input_ids[:, 1:].clone()

                # delete SEP (EOS) token
                dec_input_ids.masked_fill_(dec_input_ids==self.SEP, self.PAD)
                
                enc_input_ids_rnd.append(enc_input_ids)
                enc_segments_rnd.append(enc_segments)
                enc_sep_indices_rnd.append(enc_sep_indices) 
                enc_mlm_labels_rnd.append(enc_mlm_labels)
                enc_next_labels_rnd.append(torch.LongTensor([-1])) # do not perform next sentence prediction in gen mode, it will be ignored
                enc_hist_len_rnd.append(torch.LongTensor([len(context_utterances[j])-1]))
                enc_att_masks_rnd.append(enc_att_masks)
                dec_input_ids_rnd.append(dec_input_ids)
                dec_att_masks_rnd.append(dec_att_masks)
                dec_labels_rnd.append(dec_labels)

                enc_input_ids_all_rnd.append(torch.cat(enc_input_ids_rnd, 0).unsqueeze(0))
                enc_segments_all_rnd.append(torch.cat(enc_segments_rnd, 0).unsqueeze(0))
                enc_sep_indices_all_rnd.append(torch.cat(enc_sep_indices_rnd, 0).unsqueeze(0))
                enc_mlm_labels_all_rnd.append(torch.cat(enc_mlm_labels_rnd, 0).unsqueeze(0))
                enc_next_labels_all_rnd.append(torch.cat(enc_next_labels_rnd, 0).unsqueeze(0)) 
                enc_hist_len_all_rnd.append(torch.cat(enc_hist_len_rnd, 0).unsqueeze(0))
                enc_att_masks_all_rnd.append(torch.cat(enc_att_masks_rnd, 0).unsqueeze(0))
                dec_input_ids_all_rnd.append(torch.cat(dec_input_ids_rnd, 0).unsqueeze(0))
                dec_att_masks_all_rnd.append(torch.cat(dec_att_masks_rnd, 0).unsqueeze(0))
                dec_labels_all_rnd.append(torch.cat(dec_labels_rnd, 0).unsqueeze(0)) 

            enc_input_ids_all_rnd = torch.cat(enc_input_ids_all_rnd, 0)
            enc_segments_all_rnd = torch.cat(enc_segments_all_rnd, 0)
            enc_sep_indices_all_rnd = torch.cat(enc_sep_indices_all_rnd, 0)
            enc_mlm_labels_all_rnd = torch.cat(enc_mlm_labels_all_rnd, 0)
            enc_next_labels_all_rnd = torch.cat(enc_next_labels_all_rnd, 0)
            enc_hist_len_all_rnd = torch.cat(enc_hist_len_all_rnd, 0)
            enc_att_masks_all_rnd = torch.cat(enc_att_masks_all_rnd, 0)
            dec_input_ids_all_rnd = torch.cat(dec_input_ids_all_rnd, 0)
            dec_att_masks_all_rnd = torch.cat(dec_att_masks_all_rnd, 0)
            dec_labels_all_rnd = torch.cat(dec_labels_all_rnd, 0)

            item = {}
            item['enc_input_ids'] = enc_input_ids_all_rnd
            item['enc_segments'] = enc_segments_all_rnd
            item['enc_sep_indices'] = enc_sep_indices_all_rnd
            item['enc_mlm_labels'] = enc_mlm_labels_all_rnd
            item['enc_next_sentence_labels'] = enc_next_labels_all_rnd
            item['enc_hist_len'] = enc_hist_len_all_rnd
            item['enc_att_mask'] = enc_att_masks_all_rnd
            item['dec_input_ids'] = dec_input_ids_all_rnd
            item['dec_att_mask'] = dec_att_masks_all_rnd 
            item['dec_labels'] = dec_labels_all_rnd

            # get image features
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[cur_data['image_id']]
            features, spatials, image_mask, image_target, image_label = encode_image_input(
                features, 
                num_boxes, 
                boxes, 
                image_target, 
                max_regions=self._max_region_num, 
                mask_prob=self.params['mask_prob']
            )

            item['enc_image_feat'] = features
            item['enc_image_loc'] = spatials
            item['enc_image_mask'] = image_mask
            item['enc_image_target'] = image_target
            item['enc_image_label'] = image_label
            return item


    	
