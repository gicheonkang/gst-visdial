import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.dataloader_visdial_disc import VisdialDataset
from dataloader.dataloader_visdial_gen import VisdialDataset

from models.visual_dialog_encoder import VisualDialogEncoder
from models.visual_dialog_decoder import VisualDialogDecoder
from models.visual_dialog_model import EncoderDecoderModel
from utils.logger import Logger

import options
import pprint
import os
import json
from train_gen import forward
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter

def evaluate(model, dataloader, params, eval_batch_size, mode='vd_eval_val'):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()
    ranks_json = []
    model.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 500
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

                _, lm_scores = forward(model, item, params) # (batch_size, seq_len, vocab_size)
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

            # print("output shape",torch.cat(output,0).shape)
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
    eval_batch_size = 20 if params['vd_version'] == '1.0' else 25
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
    model = nn.DataParallel(model, params["gpu_ids"]) 

    # load pretrained model
    assert params['start_path'] != ''
    model_state_dict = torch.load(params['start_path'], map_location=device)
    model.module.load_state_dict(model_state_dict['model_state_dict'])
    print("model succesfully loaded from {}".format(params['start_path']))
    ranks_json = evaluate(model, dataloader, params, eval_batch_size, mode=mode)

    if mode == 'vd_eval_test':
        json.dump(ranks_json, open(os.path.join(parsed['save_path'], 'predictions.txt'), "w"))
