import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.dataloader_visdial_disc import VisdialDataset
from models.visual_dialog_encoder import VisualDialogEncoder

import options
import pprint
import os
import json
from train_disc import forward
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter
from utils.logger import Logger

def evaluate(dataloader, params, eval_batch_size, mode='vd_eval_val'):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()
    ranks_json = []
    dialog_encoder.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 200
        print("batch size for evaluation", batch_size)
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break

            tokens = batch['tokens']
            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]
            tokens = tokens.view(-1, tokens.shape[-1])                       
            segments = batch['segments']
            segments = segments.view(-1, segments.shape[-1])
            sep_indices = batch['sep_indices']
            sep_indices = sep_indices.view(-1, sep_indices.shape[-1])

            mask = batch['mask']
            mask = mask.view(-1, mask.shape[-1])
            hist_len = batch['hist_len']
            hist_len = hist_len.view(-1)
            
            # get image features
            features = batch['image_feat'] 
            spatials = batch['image_loc'] 
            image_mask = batch['image_mask']

            # expand the image features to match those of tokens etc.
            max_num_regions = features.shape[-2]
            features = features.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions, 2048).contiguous()
            spatials = spatials.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions, 5).contiguous()
            image_mask = image_mask.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions).contiguous()

            features = features.view(-1, max_num_regions, 2048)
            spatials = spatials.view(-1, max_num_regions, 5)
            image_mask = image_mask.view(-1, max_num_regions)

            assert tokens.shape[0] == segments.shape[0] == sep_indices.shape[0] == mask.shape[0] == \
                hist_len.shape[0] == features.shape[0] == spatials.shape[0] == \
                    image_mask.shape[0] == num_rounds * num_options * eval_batch_size

            output = []
            assert (eval_batch_size * num_rounds * num_options)//batch_size == (eval_batch_size * num_rounds * num_options)/batch_size
            for j in range((eval_batch_size * num_rounds * num_options)//batch_size):
                # create chunks of the original batch
                item = {}
                item['tokens'] = tokens[j*batch_size:(j+1)*batch_size,:]
                item['segments'] = segments[j*batch_size:(j+1)*batch_size,:]
                item['sep_indices'] = sep_indices[j*batch_size:(j+1)*batch_size,:]
                item['mask'] = mask[j*batch_size:(j+1)*batch_size,:]
                item['hist_len'] = hist_len[j*batch_size:(j+1)*batch_size]

                item['image_feat'] = features[j*batch_size:(j+1)*batch_size, : , :]
                item['image_loc'] = spatials[j*batch_size:(j+1)*batch_size, : , :]
                item['image_mask'] = image_mask[j*batch_size:(j+1)*batch_size, :]

                _, _, _, _, nsp_scores, _ = forward(dialog_encoder, item, params)
                # normalize nsp scores
                nsp_probs = F.softmax(nsp_scores, dim=1)
                assert nsp_probs.shape[-1] == 2
                output.append(nsp_probs[:,0])    


            # print("output shape",torch.cat(output,0).shape)
            output = torch.cat(output,0).view(eval_batch_size, num_rounds, num_options)
            if mode == 'vd_eval_val':
                gt_option_inds = batch['gt_option_inds']
                gt_relevance = batch['gt_relevance']
                gt_relevance_round_id = batch['round_id'].squeeze(1)

                sparse_metrics.observe(output, gt_option_inds)
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
    eval_batch_size = 20
    save_name = params['save_name'] if params['save_name'] else 'performance_log.txt'
    logger = Logger(os.path.join(params['save_path'], save_name))

    mode = params['mode']
    assert mode == 'vd_eval_val' or mode == 'vd_eval_test'
    dataset.mode = mode
    params['model'] = 'enc_only_a'

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
    if params['start_path']:
        pretrained_dict = torch.load(params['start_path'], map_location=device)

        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']

        model_dict = dialog_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("number of keys transferred", len(pretrained_dict))
        assert len(pretrained_dict.keys()) > 0
        model_dict.update(pretrained_dict)
        dialog_encoder.load_state_dict(model_dict)

    dialog_encoder = dialog_encoder.to(device)
    dialog_encoder = nn.DataParallel(dialog_encoder, params["gpu_ids"])
    ranks_json = evaluate(dataloader, params, eval_batch_size, mode=mode)

    if mode == 'vd_eval_test':
        json.dump(ranks_json, open(params['save_name'] + '_predictions.txt', "w"))
