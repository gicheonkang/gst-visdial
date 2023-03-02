import torch
from torch.utils import data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import random
from utils.data_utils import list2tensorpad, encode_input, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader

class VisdialDataset(data.Dataset):
    def __init__(self, params):
        self.numDataPoints = {}
        num_samples_train = params['num_train_samples']
        num_samples_val = params['num_val_samples']
        self._image_features_reader = ImageFeaturesH5Reader(params['visdial_image_feats'])
        data_key_train = 'visdial_processed_train'
        data_key_val = 'visdial_processed_val'
        if params['vd_version'] == '0.9':
            data_key_train += '_0.9'
            data_key_val += '_0.9'        

        with open(params[data_key_train]) as f:
            self.visdial_data_train = json.load(f)
            if params['overfit']:
                if num_samples_train:
                    self.numDataPoints['vd_train'] = num_samples_train
                else:                
                    self.numDataPoints['vd_train'] = 5
            else:
                if num_samples_train:
                    self.numDataPoints['vd_train'] = num_samples_train
                else:
                    self.numDataPoints['vd_train'] = len(self.visdial_data_train['data']['dialogs'])

        with open(params[data_key_val]) as f:
            self.visdial_data_val = json.load(f)
            if params['overfit']:
                if num_samples_val:
                    self.numDataPoints['vd_eval_val'] = num_samples_val
                else:
                    self.numDataPoints['vd_eval_val'] = 5
            else:
                if num_samples_val:
                    self.numDataPoints['vd_eval_val'] = num_samples_val
                else:
                    self.numDataPoints['vd_eval_val'] = len(self.visdial_data_val['data']['dialogs'])
            self.numDataPoints['vd_gen_val'] = self.numDataPoints['vd_eval_val']
                    
        with open(params['visdial_processed_test']) as f:
            self.visdial_data_test = json.load(f)
            self.numDataPoints['vd_eval_test'] = len(self.visdial_data_test['data']['dialogs'])

        self.overfit = params['overfit']
        with open(params['visdial_processed_val_dense_annotations']) as f:
            self.visdial_data_val_dense = json.load(f)

        self.num_options = params['num_options']
        self._mode = 'vd_train'
        self.subsets = ['vd_train', 'vd_eval_val', 'vd_eval_test', 'vd_gen_val']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ['[CLS]','[MASK]','[SEP]', '[PAD]', '[UNK]']
        self.CLS, self.MASK, self.SEP, self.PAD, self.UNK = self.tokenizer.convert_tokens_to_ids(tokens)
        self.params = params
        self._max_region_num = 37
        self._max_seq_len = self.params['max_seq_len']
        self._max_utt_len = self.params['max_utt_len']

    def __len__(self):
        return self.numDataPoints[self._mode]
    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        assert mode in self.subsets
        self._mode = mode

    def __getitem__(self, index):

        def tokens2str(seq):
            dialog_sequence = ''
            for sentence in seq:
                for word in sentence:
                    dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
                dialog_sequence += ' </end> '
            dialog_sequence = dialog_sequence.encode('utf8')
            return dialog_sequence

        def pruneRounds(context, num_rounds):
            start_segment = 1
            len_context = len(context)
            cur_rounds = (len(context) // 2) + 1
            l_index = 0
            if cur_rounds > num_rounds:
                # caption is not part of the final input
                l_index = len_context - (2 * num_rounds)
                start_segment = 0   
            return context[l_index:], start_segment

        cur_data = None
        if self._mode == 'vd_train':
            cur_data = self.visdial_data_train['data']
        elif self._mode == 'vd_eval_val' or self._mode == 'vd_gen_val':
            if self.overfit:
                cur_data = self.visdial_data_train['data']
            else:
                cur_data = self.visdial_data_val['data']
        else:
            cur_data = self.visdial_data_test['data']
        
        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']

        if self._mode == 'vd_train':
            # train split, train mode, enc_dec model
            full_utterances = []
            context_utterances = []
            target_utterances = []

            tokenized_caption = self.tokenizer.encode(dialog['caption'])
            full_utterances.append([tokenized_caption]) 
            context_utterances.append([tokenized_caption])
        
            # -------------------------------------------------------------------
            # Structuring positive dialog data
            # -------------------------------------------------------------------
            for rnd,utterance in enumerate(dialog['dialog']):
                if self.params['model'] == 'enc_dec_q':
                    full_utterance = full_utterances[-1].copy()
                    context_utterance = full_utterances[-1].copy()
                    target_utterance = []

                    tokenized_question = self.tokenizer.encode(cur_questions[utterance['question']])
                    full_utterance.append(tokenized_question)

                    if len(tokenized_question) > self._max_utt_len -2:
                        tokenized_question = tokenized_question[:self._max_utt_len -2]
                    target_utterance.append(tokenized_question)

                    tokenized_answer = self.tokenizer.encode(cur_answers[utterance['answer']])
                    full_utterance.append(tokenized_answer)

                    full_utterances.append(full_utterance)
                    context_utterances.append(context_utterance)
                    target_utterances.append(target_utterance)
                else:
                    full_utterance = full_utterances[-1].copy()
                    context_utterance = full_utterances[-1].copy()
                    target_utterance = []
                    
                    tokenized_question = self.tokenizer.encode(cur_questions[utterance['question']])
                    full_utterance.append(tokenized_question) 
                    context_utterance.append(tokenized_question)
                    
                    tokenized_answer = self.tokenizer.encode(cur_answers[utterance['answer']])
                    full_utterance.append(tokenized_answer)
                    
                    if len(tokenized_answer) > self._max_utt_len -2:
                        tokenized_answer = tokenized_answer[:self._max_utt_len -2]
                    target_utterance.append(tokenized_answer)

                    full_utterances.append(full_utterance)
                    context_utterances.append(context_utterance)
                    target_utterances.append(target_utterance)
                
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
                    mask_prob=0,
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
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(
                features, 
                num_boxes, 
                boxes, 
                image_target, 
                max_regions=self._max_region_num, 
                mask_prob=0
            )
            item['enc_image_feat'] = features
            item['enc_image_loc'] = spatials
            item['enc_image_mask'] = image_mask
            item['enc_image_target'] = image_target
            item['enc_image_label'] = image_label
            return item 
        
        elif self._mode == 'vd_eval_val':               
            # val split, eval mode, enc_dec model
            full_utterances = []
            all_context_utterances = []
            all_option_utterances = []

            tokenized_caption = self.tokenizer.encode(dialog['caption'])
            full_utterances.append([tokenized_caption])
            
            gt_option_inds = []
            gt_relevance = None 

            # -------------------------------------------------------------------
            # Structuring dialog data for evaluation
            # -------------------------------------------------------------------
            for rnd,utterance in enumerate(dialog['dialog']):
                full_utterance = full_utterances[-1].copy()
                context_utterances = []
                option_utterances = []
                context_utterance = full_utterances[-1].copy()

                tokenized_question = self.tokenizer.encode(cur_questions[utterance['question']])
                full_utterance.append(tokenized_question)
                context_utterance.append(tokenized_question)

                tokenized_answer = self.tokenizer.encode(cur_answers[utterance['answer']])    
                full_utterance.append(tokenized_answer)

                # current round
                gt_option_ind = utterance['gt_index']
                option_inds = []
                option_inds.append(gt_option_ind) 
                all_inds = list(range(100))
                all_inds.remove(gt_option_ind)
                all_inds = all_inds[:(num_options-1)]
                option_inds.extend(all_inds)
                gt_option_inds.append(0)
                cur_rnd_options = []
                answer_options = [utterance['answer_options'][k] for k in option_inds]
                assert len(answer_options) == len(option_inds) == num_options
                assert answer_options[0] == utterance['answer']

                if self.params['vd_version'] == "1.0":
                    if rnd == self.visdial_data_val_dense[index]['round_id'] - 1:
                        gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['gt_relevance'])
                        # shuffle based on new indices
                        gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
                
                for answer_option in answer_options:
                    context_utterances.append(context_utterance)
                    option_utterance = []
                    tokenized_option = self.tokenizer.encode(cur_answers[answer_option])
                    if len(tokenized_option) > self._max_utt_len -2:
                        tokenized_option = tokenized_option[:self._max_utt_len -2]

                    option_utterance.append(tokenized_option)
                    option_utterances.append(option_utterance)

                full_utterances.append(full_utterance)
                all_context_utterances.append(context_utterances)
                all_option_utterances.append(option_utterances)

            # encode the input and create batch x 10 x 100 * max_len arrays (batch x num_rounds x num_options) 
            enc_input_ids_all_rnd = []
            enc_segments_all_rnd = []
            enc_sep_indices_all_rnd = []
            enc_mlm_labels_all_rnd = []
            enc_hist_len_all_rnd = []
            enc_att_masks_all_rnd = []
            dec_input_ids_all_rnd = []
            dec_att_masks_all_rnd = []
            
            for j in range(10):
                enc_input_ids_rnd = []
                enc_segments_rnd = []
                enc_sep_indices_rnd = []
                enc_mlm_labels_rnd = []
                enc_hist_len_rnd = []
                enc_att_masks_rnd = []
                dec_input_ids_rnd = []
                dec_att_masks_rnd = []

                for k in range(100):
                    start_segment = 1
                    enc_input_ids, enc_segments, enc_sep_indices, enc_mlm_labels, enc_att_masks = encode_input(
                        all_context_utterances[j][k], 
                        start_segment, 
                        self.CLS,
                        self.SEP, 
                        self.MASK,
                        self.PAD, 
                        max_seq_len=self._max_seq_len, 
                        mask_prob=self.params['mask_prob'] if self.params['attack'] == 'random_token' else 0,
                    )

                    dec_input_ids, _, _, _, dec_att_masks = encode_input(
                        all_option_utterances[j][k], 
                        start_segment, 
                        self.CLS,
                        self.SEP, 
                        self.MASK,
                        self.PAD, 
                        max_seq_len=self._max_utt_len,
                        mask_prob=0,
                    )

                    enc_input_ids_rnd.append(enc_input_ids)
                    enc_segments_rnd.append(enc_segments)
                    enc_sep_indices_rnd.append(enc_sep_indices) 
                    enc_mlm_labels_rnd.append(enc_mlm_labels)
                    enc_hist_len_rnd.append(torch.LongTensor([len(all_context_utterances[j][k])-1]))
                    enc_att_masks_rnd.append(enc_att_masks)
                    dec_input_ids_rnd.append(dec_input_ids)
                    dec_att_masks_rnd.append(dec_att_masks)

                enc_input_ids_all_rnd.append(torch.cat(enc_input_ids_rnd, 0).unsqueeze(0))
                enc_segments_all_rnd.append(torch.cat(enc_segments_rnd, 0).unsqueeze(0))
                enc_sep_indices_all_rnd.append(torch.cat(enc_sep_indices_rnd, 0).unsqueeze(0))
                enc_mlm_labels_all_rnd.append(torch.cat(enc_mlm_labels_rnd, 0).unsqueeze(0))
                enc_hist_len_all_rnd.append(torch.cat(enc_hist_len_rnd, 0).unsqueeze(0))
                enc_att_masks_all_rnd.append(torch.cat(enc_att_masks_rnd, 0).unsqueeze(0))
                dec_input_ids_all_rnd.append(torch.cat(dec_input_ids_rnd, 0).unsqueeze(0))
                dec_att_masks_all_rnd.append(torch.cat(dec_att_masks_rnd, 0).unsqueeze(0))

            enc_input_ids_all_rnd = torch.cat(enc_input_ids_all_rnd, 0)
            enc_segments_all_rnd = torch.cat(enc_segments_all_rnd, 0)
            enc_sep_indices_all_rnd = torch.cat(enc_sep_indices_all_rnd, 0)
            enc_mlm_labels_all_rnd = torch.cat(enc_mlm_labels_all_rnd, 0)
            enc_hist_len_all_rnd = torch.cat(enc_hist_len_all_rnd, 0)
            enc_att_masks_all_rnd = torch.cat(enc_att_masks_all_rnd, 0)
            dec_input_ids_all_rnd = torch.cat(dec_input_ids_all_rnd, 0)
            dec_att_masks_all_rnd = torch.cat(dec_att_masks_all_rnd, 0)

            item = {}
            item['enc_input_ids'] = enc_input_ids_all_rnd
            item['enc_segments'] = enc_segments_all_rnd
            item['enc_sep_indices'] = enc_sep_indices_all_rnd
            item['enc_mlm_labels'] = enc_mlm_labels_all_rnd
            item['enc_hist_len'] = enc_hist_len_all_rnd
            item['enc_att_mask'] = enc_att_masks_all_rnd
            item['dec_input_ids'] = dec_input_ids_all_rnd
            item['dec_att_mask'] = dec_att_masks_all_rnd
            item['gt_option_inds'] = torch.LongTensor(gt_option_inds)

            # return dense annotation data as well
            item['image_id'] = torch.LongTensor([img_id])
            if self.params['vd_version'] == "1.0":
                item['round_id'] = torch.LongTensor([self.visdial_data_val_dense[index]['round_id']])
                item['gt_relevance'] = gt_relevance

            # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(
                features, 
                num_boxes, 
                boxes,
                image_target, 
                max_regions=self._max_region_num,
                mask_prob=0
            )
            item['enc_image_feat'] = features
            item['enc_image_loc'] = spatials
            item['enc_image_mask'] = image_mask
            return item

        elif self._mode == 'vd_gen_val':
            # val split, generation mode, enc_dec model                           
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
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
            item['image_id'] = torch.LongTensor([img_id])

            context_utterance = []
            tokenized_caption = self.tokenizer.encode(dialog['caption'])
            context_utterance.append(tokenized_caption)

            #tokenized_question = self.tokenizer.encode(cur_questions[dialog['dialog'][0]['question']])
            #context_utterance.append(tokenized_question)


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
            # test split, evaluation mode, enc_dec model
            assert num_options == 100
            cur_rnd_utterance = [self.tokenizer.encode(dialog['caption'])]
            context_utterances = []
            option_utterances = []

            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                if rnd != len(dialog['dialog'])-1:
                    cur_rnd_utterance.append(self.tokenizer.encode(cur_answers[utterance['answer']]))

            for answer_option in dialog['dialog'][-1]['answer_options']:
                context_utterances.append(cur_rnd_utterance)
                tokenized_option = self.tokenizer.encode(cur_answers[answer_option])
                if len(tokenized_option) > self._max_utt_len -2:
                    tokenized_option = tokenized_option[:self._max_utt_len -2]
                option_utterances.append([tokenized_option])

            enc_input_ids_rnd = []
            enc_segments_rnd = []
            enc_sep_indices_rnd = []
            enc_mlm_labels_rnd = []
            enc_hist_len_rnd = []
            enc_att_masks_rnd = []
            dec_input_ids_rnd = []
            dec_att_masks_rnd = []

            for j in range(100):
                start_segment = 1
                enc_input_ids, enc_segments, enc_sep_indices, enc_mlm_labels, enc_att_masks = encode_input(
                    context_utterances[j], 
                    start_segment, 
                    self.CLS,
                    self.SEP, 
                    self.MASK,
                    self.PAD, 
                    max_seq_len=self._max_seq_len, 
                    mask_prob=0,
                )

                dec_input_ids, _, _, _, dec_att_masks = encode_input(
                    option_utterances[j], 
                    start_segment, 
                    self.CLS,
                    self.SEP, 
                    self.MASK,
                    self.PAD, 
                    max_seq_len=self._max_utt_len,
                    mask_prob=0,
                )

                enc_input_ids_rnd.append(enc_input_ids)
                enc_segments_rnd.append(enc_segments)
                enc_sep_indices_rnd.append(enc_sep_indices) 
                enc_mlm_labels_rnd.append(enc_mlm_labels)
                enc_hist_len_rnd.append(torch.LongTensor([len(context_utterances[j])-1]))
                enc_att_masks_rnd.append(enc_att_masks)
                dec_input_ids_rnd.append(dec_input_ids)
                dec_att_masks_rnd.append(dec_att_masks)

            enc_input_ids_rnd = torch.cat(enc_input_ids_rnd, 0)
            enc_segments_rnd = torch.cat(enc_segments_rnd, 0)
            enc_sep_indices_rnd = torch.cat(enc_sep_indices_rnd, 0)
            enc_mlm_labels_rnd = torch.cat(enc_mlm_labels_rnd, 0)
            enc_hist_len_rnd = torch.cat(enc_hist_len_rnd, 0)
            enc_att_masks_rnd = torch.cat(enc_att_masks_rnd, 0)
            dec_input_ids_rnd = torch.cat(dec_input_ids_rnd, 0)
            dec_att_masks_rnd = torch.cat(dec_att_masks_rnd, 0)

            item = {}
            item['enc_input_ids'] = enc_input_ids_rnd.unsqueeze(0)
            item['enc_segments'] = enc_segments_rnd.unsqueeze(0)
            item['enc_sep_indices'] = enc_sep_indices_rnd.unsqueeze(0)
            item['enc_mlm_labels'] = enc_mlm_labels_rnd.unsqueeze(0)
            item['enc_hist_len'] = enc_hist_len_rnd.unsqueeze(0)
            item['enc_att_mask'] = enc_att_masks_rnd.unsqueeze(0)
            item['dec_input_ids'] = dec_input_ids_rnd.unsqueeze(0)
            item['dec_att_mask'] = dec_att_masks_rnd.unsqueeze(0)             

            item['image_id'] = torch.LongTensor([img_id])
            item['round_id'] = torch.LongTensor([dialog['round_id']])

            # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(
                features, 
                num_boxes, 
                boxes,
                image_target, 
                max_regions=self._max_region_num,
                mask_prob=0
            )
            item['enc_image_feat'] = features
            item['enc_image_loc'] = spatials
            item['enc_image_mask'] = image_mask
            return item
