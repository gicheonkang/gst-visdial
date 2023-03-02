import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pytorch_transformers.optimization import AdamW
from dataloader.dataloader_visdial_gen import VisdialDataset

from models.visual_dialog_encoder import VisualDialogEncoder
from models.visual_dialog_decoder import VisualDialogDecoder
from models.visual_dialog_model import EncoderDecoderModel
from utils.logger import Logger

import numpy as np
import pickle
import options
import pprint
import os
import json
import random
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter
from utils.text_attack import TextAttack


def forward(model, batch, params, textattack, epsilon=1.0):
    enc_next_sentence_labels = None
    enc_image_target = None
    enc_image_label = None
    dec_labels = None

    # language stuff
    enc_input_ids = batch['enc_input_ids']
    enc_segments = batch['enc_segments']
    enc_sep_indices = batch['enc_sep_indices']
    enc_mlm_labels = batch['enc_mlm_labels']
    enc_hist_len = batch['enc_hist_len']
    enc_att_mask = batch['enc_att_mask']
    dec_input_ids = batch['dec_input_ids']
    dec_att_mask = batch['dec_att_mask']
    gt_relevance = batch['gt_relevance']
    if params['attack'] == 'coreference':
        coref_dependency = batch['coref_dependency']

    enc_input_ids = enc_input_ids.view(-1,enc_input_ids.shape[-1])
    enc_segments = enc_segments.view(-1, enc_segments.shape[-1])
    enc_sep_indices = enc_sep_indices.view(-1,enc_sep_indices.shape[-1])
    enc_mlm_labels = enc_mlm_labels.view(-1, enc_mlm_labels.shape[-1])
    enc_hist_len = enc_hist_len.view(-1)
    enc_att_mask = enc_att_mask.view(-1, enc_att_mask.shape[-1])
    dec_input_ids = dec_input_ids.view(-1,dec_input_ids.shape[-1])
    dec_att_mask = dec_att_mask.view(-1, dec_att_mask.shape[-1])
          
    # image stuff
    orig_features = batch['enc_image_feat'] 
    orig_spatials = batch['enc_image_loc'] 
    orig_image_mask = batch['enc_image_mask']
    
    enc_image_features = orig_features.view(-1, orig_features.shape[-2], orig_features.shape[-1])       
    enc_image_spatials = orig_spatials.view(-1, orig_spatials.shape[-2], orig_spatials.shape[-1])
    enc_image_mask = orig_image_mask.view(-1, orig_image_mask.shape[-1])

    sample_indices = torch.arange(enc_hist_len.shape[0])   
    enc_input_ids = enc_input_ids[sample_indices, :]
    enc_segments = enc_segments[sample_indices, :]
    enc_sep_indices = enc_sep_indices[sample_indices, :]
    enc_mlm_labels = enc_mlm_labels[sample_indices, :]
    enc_hist_len = enc_hist_len[sample_indices]
    enc_att_mask = enc_att_mask[sample_indices, :]
    dec_input_ids = dec_input_ids[sample_indices, :]
    enc_image_features = enc_image_features[sample_indices, : , :]
    enc_image_spatials = enc_image_spatials[sample_indices, :, :]
    enc_image_mask =  enc_image_mask[sample_indices, :]

    enc_input_ids = enc_input_ids.to(params['device'])
    enc_segments = enc_segments.to(params['device'])
    enc_sep_indices = enc_sep_indices.to(params['device'])
    enc_mlm_labels = enc_mlm_labels.to(params['device'])
    enc_hist_len = enc_hist_len.to(params['device'])
    enc_att_mask = enc_att_mask.to(params['device'])
    dec_input_ids = dec_input_ids.to(params['device'])
    dec_att_mask = dec_att_mask.to(params['device'])

    enc_image_features = enc_image_features.to(params['device'])
    enc_image_spatials = enc_image_spatials.to(params['device'])
    enc_image_mask = enc_image_mask.to(params['device'])
    gt_relevance = gt_relevance.to(params['device'])
    batch_size, seq_len = dec_input_ids.size()
    
    if params['attack'] == 'fgsm':
        # get the current dialog round (1~10)
        unit_sep_indices = enc_sep_indices[0, :]
        dialog_round = torch.sum(unit_sep_indices != 0).item() / 2
        dialog_round = int(dialog_round) 
 
        # In VisDial v1.0 validation dataset, only one dialog round out of ten rounds contains human-annotated relevance scores.
        # So we only need to apply the fgsm attack for the round that has the relevance scores (just for efficient computation).  
        if dialog_round == batch['round_id'].item(): 
            enc_image_variables = Variable(enc_image_features.data, requires_grad=True)
            opt = optim.AdamW([enc_image_variables], lr=1e-5)
            opt.zero_grad()

            with torch.enable_grad():
                lm_loss, lm_scores = model(
                    enc_image_features=enc_image_variables,
                    enc_image_spatials=enc_image_spatials,
                    enc_image_mask=enc_image_mask,
                    enc_image_target=enc_image_target,
                    enc_image_label=enc_image_label,
                    enc_next_sentence_labels=enc_next_sentence_labels,
                    enc_input_ids=enc_input_ids,
                    enc_segments=enc_segments,
                    enc_sep_indices=enc_sep_indices,
                    enc_mlm_labels=enc_mlm_labels,
                    enc_attention_mask=enc_att_mask,
                    dec_input_ids=dec_input_ids,
                    dec_attention_mask=dec_att_mask,
                    dec_labels=dec_labels,
                    loss_reduction=False
                )

                # Extact the loss for each sample and weighted sum with the relevance scores.
                # So the loss values for zero-relevance candidate answers are not considered. 
                lm_loss = lm_loss.view(batch_size, seq_len)
                lm_loss = lm_loss.mean(dim=1)
                lm_loss = torch.sum(lm_loss * gt_relevance)

            lm_loss.backward()
            enc_image_variables = Variable(enc_image_variables + epsilon * torch.sign(enc_image_variables.grad.data), requires_grad=True) 

            _, lm_scores = model(
                enc_image_features=enc_image_variables,
                enc_image_spatials=enc_image_spatials,
                enc_image_mask=enc_image_mask,
                enc_image_target=enc_image_target,
                enc_image_label=enc_image_label,
                enc_next_sentence_labels=enc_next_sentence_labels,
                enc_input_ids=enc_input_ids,
                enc_segments=enc_segments,
                enc_sep_indices=enc_sep_indices,
                enc_mlm_labels=enc_mlm_labels,
                enc_attention_mask=enc_att_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_att_mask,
                dec_labels=dec_labels
            )
        else:
            _, lm_scores = model(
                enc_image_features=enc_image_features,
                enc_image_spatials=enc_image_spatials,
                enc_image_mask=enc_image_mask,
                enc_image_target=enc_image_target,
                enc_image_label=enc_image_label,
                enc_next_sentence_labels=enc_next_sentence_labels,
                enc_input_ids=enc_input_ids,
                enc_segments=enc_segments,
                enc_sep_indices=enc_sep_indices,
                enc_mlm_labels=enc_mlm_labels,
                enc_attention_mask=enc_att_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_att_mask,
                dec_labels=dec_labels
            )

    elif params['attack'] == 'coreference':
        # get the current dialog round (1~10)
        unit_sep_indices = enc_sep_indices[0, :]
        dialog_round = torch.sum(unit_sep_indices != 0).item() / 2
        dialog_round = int(dialog_round)

        if dialog_round == batch['round_id'].item():
            attacked_enc_input_ids = textattack.coreference_attack(enc_input_ids, enc_sep_indices, coref_dependency)
            _, lm_scores = model(
                enc_image_features=enc_image_features,
                enc_image_spatials=enc_image_spatials,
                enc_image_mask=enc_image_mask,
                enc_image_target=enc_image_target,
                enc_image_label=enc_image_label,
                enc_next_sentence_labels=enc_next_sentence_labels,
                enc_input_ids=attacked_enc_input_ids,
                enc_segments=enc_segments,
                enc_sep_indices=enc_sep_indices,
                enc_mlm_labels=enc_mlm_labels,
                enc_attention_mask=enc_att_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_att_mask,
                dec_labels=dec_labels
            )
        else:
            _, lm_scores = model(
                enc_image_features=enc_image_features,
                enc_image_spatials=enc_image_spatials,
                enc_image_mask=enc_image_mask,
                enc_image_target=enc_image_target,
                enc_image_label=enc_image_label,
                enc_next_sentence_labels=enc_next_sentence_labels,
                enc_input_ids=enc_input_ids,
                enc_segments=enc_segments,
                enc_sep_indices=enc_sep_indices,
                enc_mlm_labels=enc_mlm_labels,
                enc_attention_mask=enc_att_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_att_mask,
                dec_labels=dec_labels
            )    

    elif params['attack'] == 'random_token':
        attacked_enc_input_ids = textattack.random_token_attack(enc_input_ids, enc_segments, enc_att_mask)
        _, lm_scores = model(
            enc_image_features=enc_image_features,
            enc_image_spatials=enc_image_spatials,
            enc_image_mask=enc_image_mask,
            enc_image_target=enc_image_target,
            enc_image_label=enc_image_label,
            enc_next_sentence_labels=enc_next_sentence_labels,
            enc_input_ids=attacked_enc_input_ids,
            enc_segments=enc_segments,
            enc_sep_indices=enc_sep_indices,
            enc_mlm_labels=enc_mlm_labels,
            enc_attention_mask=enc_att_mask,
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_att_mask,
            dec_labels=dec_labels
        )
    else:
        print('please check the name of the attack')
        exit(-1)    

    return lm_scores 

def evaluate(model, dataloader, params, eval_batch_size, mode='vd_eval_val'):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()
    ranks_json = []
    model.eval()
    batch_idx = 0
    with torch.no_grad():
        textattack = None
        batch_size = 100
        if params['attack'] == 'random_token' or params['attack'] == 'coreference':
            coref_dependency = json.load(open(params['visdial_processed_val_coref_dependency'], 'r'))
            cos_sim = np.load(params['cos_sim_counter_fitting'])
            cos_sim_idx2word = pickle.load(open(params['cos_sim_idx2word'], 'rb'))
            cos_sim_word2idx = pickle.load(open(params['cos_sim_word2idx'], 'rb'))
            textattack = TextAttack('bert-base-uncased', params['device'], cos_sim, cos_sim_idx2word, cos_sim_word2idx)

        print("batch size for evaluation", batch_size)
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break

            # language stuff
            enc_input_ids = batch['enc_input_ids']
            enc_segments = batch['enc_segments']
            enc_sep_indices = batch['enc_sep_indices']
            enc_mlm_labels = batch['enc_mlm_labels']
            enc_hist_len = batch['enc_hist_len']
            enc_att_mask = batch['enc_att_mask']
            dec_input_ids = batch['dec_input_ids']
            dec_att_mask = batch['dec_att_mask']

            num_rounds = enc_input_ids.shape[1]
            num_options = enc_input_ids.shape[2] 

            enc_input_ids = enc_input_ids.view(-1, enc_input_ids.shape[-1])
            enc_segments = enc_segments.view(-1, enc_segments.shape[-1])
            enc_sep_indices = enc_sep_indices.view(-1, enc_sep_indices.shape[-1])
            enc_mlm_labels = enc_mlm_labels.view(-1, enc_mlm_labels.shape[-1])
            enc_hist_len = enc_hist_len.view(-1)
            enc_att_mask = enc_att_mask.view(-1, enc_att_mask.shape[-1])
            dec_input_ids = dec_input_ids.view(-1,dec_input_ids.shape[-1])
            dec_att_mask = dec_att_mask.view(-1, dec_att_mask.shape[-1])
                
            # image stuff
            enc_image_features = batch['enc_image_feat'] 
            enc_image_spatials = batch['enc_image_loc'] 
            enc_image_mask = batch['enc_image_mask']

            # round id and gt relevance
            round_id = batch['round_id']
            gt_relevance = batch['gt_relevance']

            # expand the image features to match those of tokens etc.
            max_num_regions = enc_image_features.shape[-2]
            enc_image_features = enc_image_features.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions, 2048).contiguous()
            enc_image_spatials = enc_image_spatials.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions, 5).contiguous()
            enc_image_mask = enc_image_mask.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions).contiguous()

            enc_image_features = enc_image_features.view(-1, max_num_regions, 2048)
            enc_image_spatials = enc_image_spatials.view(-1, max_num_regions, 5)
            enc_image_mask = enc_image_mask.view(-1, max_num_regions)

            assert enc_input_ids.shape[0] == enc_segments.shape[0] == enc_sep_indices.shape[0] == enc_mlm_labels.shape[0] == \
                enc_hist_len.shape[0] == enc_att_mask.shape[0] == dec_input_ids.shape[0] == dec_att_mask.shape[0] == \
                enc_image_features.shape[0] == enc_image_spatials.shape[0] == enc_image_mask.shape[0] == num_rounds * num_options * eval_batch_size 

            output = []
            assert (eval_batch_size * num_rounds * num_options)//batch_size == (eval_batch_size * num_rounds * num_options)/batch_size
            for j in range((eval_batch_size * num_rounds * num_options)//batch_size):
                # create chunks of the original batch
                item = {}
                item['enc_input_ids'] = enc_input_ids[j*batch_size:(j+1)*batch_size,:]
                item['enc_segments'] = enc_segments[j*batch_size:(j+1)*batch_size,:]
                item['enc_sep_indices'] = enc_sep_indices[j*batch_size:(j+1)*batch_size,:]
                item['enc_mlm_labels'] = enc_mlm_labels[j*batch_size:(j+1)*batch_size,:]
                item['enc_hist_len'] = enc_hist_len[j*batch_size:(j+1)*batch_size]
                item['enc_att_mask'] = enc_att_mask[j*batch_size:(j+1)*batch_size,:]
                item['dec_input_ids'] = dec_input_ids[j*batch_size:(j+1)*batch_size,:]
                item['dec_att_mask'] = dec_att_mask[j*batch_size:(j+1)*batch_size,:]

                item['enc_image_feat'] = enc_image_features[j*batch_size:(j+1)*batch_size, : , :]
                item['enc_image_loc'] = enc_image_spatials[j*batch_size:(j+1)*batch_size, : , :]
                item['enc_image_mask'] = enc_image_mask[j*batch_size:(j+1)*batch_size, :]
                item['round_id'] = round_id
                item['gt_relevance'] = gt_relevance
                if params['attack'] == 'coreference':
                    item['coref_dependency'] = coref_dependency[batch_idx]['coreference'][j]

                lm_scores = forward(model, item, params, textattack) # (batch_size, seq_len, vocab_size)
                lm_scores = F.log_softmax(lm_scores, dim=-1)

                # remove CLS tokens in the target answers
                target_ans_ids = item['dec_input_ids'].to(params['device'])
                shifted_target_ans_ids = target_ans_ids.new_zeros(target_ans_ids.shape)
                shifted_target_ans_ids[:, :-1] = target_ans_ids[:, 1:].clone()
                target_ans_ids = shifted_target_ans_ids

                ans_scores = torch.gather(lm_scores, -1, target_ans_ids.unsqueeze(-1)).squeeze(-1)
                # exclude zero-padded tokens when computing likelihood scores 
                ans_scores = ans_scores * (target_ans_ids != 0).float()  
                ans_scores = ans_scores.sum(-1)
                output.append(ans_scores)

            output = torch.cat(output,0).view(eval_batch_size, num_rounds, num_options)
            if mode == 'vd_eval_val':
                gt_option_inds = batch['gt_option_inds']
                sparse_metrics.observe(output, gt_option_inds)
                if params['vd_version'] == "1.0":
                    gt_relevance = batch['gt_relevance']
                    gt_relevance_round_id = batch['round_id'].squeeze(1)
                    output = output[torch.arange(output.size(0)), gt_relevance_round_id - 1, :]
                    ndcg.observe(output, gt_relevance)
            else:
                ranks = scores_to_ranks(output)
                ranks = ranks.squeeze(1)
                for i in range(eval_batch_size):
                    ranks_json.append(
                        {
                            "image_id": batch["image_id"][i].item(),
                            "round_id": int(batch["round_id"][i].item()),
                            "ranks": [
                                rank.item()
                                for rank in ranks[i][:]
                            ],
                        }
                    )
            batch_idx += 1

        if mode == 'vd_eval_val':
            all_metrics = {}
            all_metrics.update(sparse_metrics.retrieve(reset=True))
            if params['vd_version'] == "1.0":
                all_metrics.update(ndcg.retrieve(reset=True))
            for metric_name, metric_value in all_metrics.items():
                logger.write(f"{metric_name}: {metric_value}")
   
    return ranks_json

if __name__ == '__main__':
    params = options.read_command_line()
    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'], exist_ok=True)

    pprint.pprint(params)
    dataset = VisdialDataset(params)
    eval_batch_size = 1 if params['vd_version'] == '1.0' else 25
    save_name = params['save_name'] if params['save_name'] else 'performance_log.txt'
    logger = Logger(os.path.join(params['save_path'], save_name))

    mode = params['mode']
    assert mode == 'vd_eval_val' or mode == 'vd_eval_test'
    dataset.mode = mode
    params['model'] == 'enc_dec_a'

    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=params['num_workers'],
        drop_last=False,
        pin_memory=False
    )

    device = (
        torch.device("cuda", params["gpu_ids"][0]) 
        if params["gpu_ids"][0] >= 0
        else torch.device("cpu")
    )
    params['device'] = device

    dialog_encoder = VisualDialogEncoder(params)
    dialog_decoder = VisualDialogDecoder(params)
    # share embedding layers
    dialog_decoder.decoder.bert.embeddings = dialog_encoder.bert_pretrained.bert.embeddings
    model = EncoderDecoderModel(params, dialog_encoder, dialog_decoder).to(device)
    #for p in model.parameters():
    #    p.requires_grad = False
    model = nn.DataParallel(model, params["gpu_ids"]) 

    # load pretrained model
    assert params['start_path'] != ''
    model_state_dict = torch.load(params['start_path'], map_location=device)
    model.module.load_state_dict(model_state_dict['model_state_dict'])
    print("model succesfully loaded from {}".format(params['start_path']))
    ranks_json = evaluate(model, dataloader, params, eval_batch_size, mode=mode)

    if mode == 'vd_eval_test':
        json.dump(ranks_json, open(os.path.join(parsed['save_path'], 'predictions.txt'), "w"))
