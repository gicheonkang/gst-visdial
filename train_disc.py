import os
import json
import options
import pprint
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from pytorch_transformers.optimization import AdamW

from dataloader.dataloader_visdial_disc import VisdialDataset
from models.visual_dialog_encoder import VisualDialogEncoder

from utils.data_utils import sequence_mask, batch_iter
from utils.logger import Logger
from utils.optim_utils import WarmupLinearScheduleNonZero

from time import gmtime, strftime
from timeit import default_timer as timer


def forward(dialog_encoder, batch, params):
    next_sentence_labels = None
    image_target = None
    image_label = None

    tokens = batch['tokens']
    segments = batch['segments']
    sep_indices = batch['sep_indices']
    mask = batch['mask']
    hist_len = batch['hist_len']

    # image stuff
    orig_features = batch['image_feat'] 
    orig_spatials = batch['image_loc'] 
    orig_image_mask = batch['image_mask']

    tokens = tokens.view(-1,tokens.shape[-1])
    segments = segments.view(-1, segments.shape[-1])
    sep_indices = sep_indices.view(-1,sep_indices.shape[-1])
    mask = mask.view(-1, mask.shape[-1])
    hist_len = hist_len.view(-1)

    features = orig_features.view(-1, orig_features.shape[-2], orig_features.shape[-1])
    spatials = orig_spatials.view(-1, orig_spatials.shape[-2], orig_spatials.shape[-1])
    image_mask = orig_image_mask.view(-1, orig_image_mask.shape[-1])

    if 'train' in params['mode']:
        sample_indices = torch.randperm(hist_len.shape[0])
        sample_indices = sample_indices[:params['batch_size']]
    else:
        sample_indices = torch.arange(hist_len.shape[0])

    tokens = tokens[sample_indices, :]
    segments = segments[sample_indices, :]
    sep_indices = sep_indices[sample_indices, :]
    mask = mask[sample_indices, :]
    hist_len = hist_len[sample_indices]

    features = features[sample_indices, : , :]
    spatials = spatials[sample_indices, :, :]
    image_mask =  image_mask[sample_indices, :]

    if 'train' in params['mode']:
        next_sentence_labels = batch['next_sentence_labels']
        next_sentence_labels = next_sentence_labels.view(-1, next_sentence_labels.shape[-1])
        next_sentence_labels = next_sentence_labels[sample_indices, :]
        next_sentence_labels = next_sentence_labels.to(params['device'])

        orig_image_target = batch['image_target'] 
        orig_image_label = batch['image_label']

        image_target = orig_image_target.view(-1, orig_image_target.shape[-2], orig_image_target.shape[-1])
        image_label = orig_image_label.view(-1, orig_image_label.shape[-1])

        image_target = image_target[sample_indices, : , :]
        image_label = image_label[sample_indices, :]        

        image_target = image_target.to(params['device'])
        image_label = image_label.to(params['device'])

    tokens = tokens.to(params['device'])
    segments = segments.to(params['device'])
    sep_indices = sep_indices.to(params['device'])
    mask = mask.to(params['device'])
    hist_len = hist_len.to(params['device'])

    features = features.to(params['device'])
    spatials = spatials.to(params['device'])
    image_mask = image_mask.to(params['device'])

    sequence_lengths = torch.gather(sep_indices,1,hist_len.view(-1,1)) + 1
    sequence_lengths = sequence_lengths.squeeze(1)
    attention_mask_lm_nsp = sequence_mask(sequence_lengths, params, max_len=tokens.shape[1])
    
    lm_loss, img_loss, nsp_loss, nsp_scores, lm_scores, _, _ = dialog_encoder(
        tokens, 
        features, 
        spatials, 
        sep_indices=sep_indices, 
        token_type_ids=segments, 
        masked_lm_labels=mask,
        attention_mask=attention_mask_lm_nsp,
        next_sentence_label=next_sentence_labels, 
        image_attention_mask=image_mask, 
        image_label=image_label, 
        image_target=image_target
    )
    
    loss = None    
    if 'train' in params['mode']:
        lm_loss = lm_loss.mean()
        nsp_loss = nsp_loss.mean()
        img_loss = img_loss.mean()
        lm_loss = params['lm_loss_coeff'] * lm_loss
        nsp_loss = params['nsp_loss_coeff'] * nsp_loss
        img_loss = params['img_loss_coeff'] * img_loss
        loss = lm_loss + nsp_loss + img_loss
    return loss, lm_loss, nsp_loss, img_loss, nsp_scores, lm_scores

if __name__ == '__main__':

    params = options.read_command_line()
    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'], exist_ok=True)
    pprint.pprint(params)

    logger = Logger(os.path.join(params['save_path'], 'log.txt'))
    logger.write(str(params))

    # select mode (train vd or cc12m)
    mode = params['mode']
    assert mode == 'vd_train'
    assert params['model'] == 'enc_only_a'

    datasets = VisdialDataset(params)
    datasets.mode = 'vd_train'
    num_iter_epoch = datasets.numDataPoints[mode] // params['batch_size']
    
    step_total = num_iter_epoch * 100
    warmup_steps = 10000

    dataloader = DataLoader(
        datasets,
        batch_size= params['batch_size'],
        shuffle=True,
        num_workers=params['num_workers'],
        drop_last=True,
        pin_memory=False
    )

    if isinstance(params["gpu_ids"], int):
        params["gpu_ids"] = [params["gpu_ids"]]
        
    device = (
        torch.device("cuda", params["gpu_ids"][0]) 
        if params["gpu_ids"][0] >= 0
        else torch.device("cpu")
    )
    params['device'] = device
    dialog_encoder = VisualDialogEncoder(params)
    named_params = dict(dialog_encoder.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    langauge_weights = None
    with open('config/language_weights.json') as f:
        langauge_weights = json.load(f)

    optimizer_grouped_parameters = []
    for key, value in named_params.items():
        if value.requires_grad:
            if key in langauge_weights:
                lr = params['lr'] 
            else:
                lr = params['image_lr']

            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0}
                ]

            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    logger.write('\n%d iter per epoch.' % num_iter_epoch)
    logger.write('\n%d total step.' % step_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=warmup_steps, t_total=step_total)
    start_iter_id = 0
    start_epoch_id = 0

    if params['start_path']:
        pretrained_dict = torch.load(params['start_path'], map_location=device)
        if params['continue']:
            if 'start' in params['start_path']:
                model_dict = dialog_encoder.state_dict()
                pretrained_dict_model = pretrained_dict['model_state_dict']
                # extract pretrained weights of the encoder-decoder model for generative VisDial!
                pretrained_dict_model = {k.split('.', 1)[1]: v for k, v in pretrained_dict_model.items() if k.split('.', 1)[1] in model_dict}
                model_dict.update(pretrained_dict_model)
                dialog_encoder.load_state_dict(model_dict)
                del pretrained_dict, model_dict, pretrained_dict_model
            else:
                model_dict = dialog_encoder.state_dict()
                optimizer_dict = optimizer.state_dict()
                pretrained_dict_model = pretrained_dict['model_state_dict']
                pretrained_dict_optimizer = pretrained_dict['optimizer_state_dict']
                pretrained_dict_scheduler = pretrained_dict['scheduler_state_dict']
                pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
                pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if k in optimizer_dict}
                model_dict.update(pretrained_dict_model)
                optimizer_dict.update(pretrained_dict_optimizer)
                dialog_encoder.load_state_dict(model_dict)
                optimizer.load_state_dict(optimizer_dict)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                if mode in params['start_path']:
                    # load the scheduler when start checkpoint and mode are the same
                    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=warmup_steps, \
                        t_total=step_total, last_epoch=pretrained_dict["iter_id"])
                    scheduler.load_state_dict(pretrained_dict_scheduler)
                    start_iter_id = pretrained_dict['iter_id']
                    start_epoch_id = start_iter_id // num_iter_epoch
                del pretrained_dict, model_dict, optimizer_dict, pretrained_dict_model, pretrained_dict_optimizer, pretrained_dict_scheduler 
            with torch.cuda.device("cuda:%s" % params["gpu_ids"][0]):
                torch.cuda.empty_cache()
        else:
            if 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = dialog_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("number of keys transferred", len(pretrained_dict))
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            dialog_encoder.load_state_dict(model_dict)
            del pretrained_dict, model_dict

    start_t = timer()
    dialog_encoder = dialog_encoder.to(device)
    dialog_encoder = nn.DataParallel(dialog_encoder, params["gpu_ids"])    

    for epoch_id, idx, batch in batch_iter(dataloader, params, start_epoch_id):
        iter_id = idx + (epoch_id * num_iter_epoch)
        dialog_encoder.train()
        # expand image features, 
        orig_features = batch['image_feat'] 
        orig_spatials = batch['image_loc'] 
        orig_image_mask = batch['image_mask']
        orig_image_target = batch['image_target'] 
        orig_image_label = batch['image_label']

        num_rounds = batch["tokens"].shape[1]
        num_samples = batch["tokens"].shape[2]

        features = orig_features.unsqueeze(1).unsqueeze(1).expand(orig_features.shape[0], num_rounds, num_samples, orig_features.shape[1], orig_features.shape[2]).contiguous()
        spatials = orig_spatials.unsqueeze(1).unsqueeze(1).expand(orig_spatials.shape[0], num_rounds, num_samples, orig_spatials.shape[1], orig_spatials.shape[2]).contiguous()
        image_label = orig_image_label.unsqueeze(1).unsqueeze(1).expand(orig_image_label.shape[0], num_rounds, num_samples, orig_image_label.shape[1]).contiguous()
        image_mask = orig_image_mask.unsqueeze(1).unsqueeze(1).expand(orig_image_mask.shape[0], num_rounds, num_samples, orig_image_mask.shape[1]).contiguous()
        image_target = orig_image_target.unsqueeze(1).unsqueeze(1).expand(orig_image_target.shape[0], num_rounds, num_samples, orig_image_target.shape[1], orig_image_target.shape[2]).contiguous()

        batch['image_feat'] = features.contiguous()
        batch['image_loc'] = spatials.contiguous()
        batch['image_mask'] = image_mask.contiguous()
        batch['image_target'] = image_target.contiguous()
        batch['image_label'] = image_label.contiguous()

        loss = None
        lm_loss = None
        nsp_loss = None
        img_loss = None
        nsp_loss = None
        nsp_scores = None
        loss, lm_loss, nsp_loss, img_loss, _, _ = forward(dialog_encoder, batch, params)

        lm_nsp_loss = None
        if lm_loss is not None and nsp_loss is not None:
            lm_nsp_loss = lm_loss + nsp_loss
        loss.backward()            

        if iter_id > 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        if iter_id % 10 == 0:
            end_t = timer()
            cur_lr = optimizer.param_groups[0]['lr']
            cur_epoch = float(iter_id) / num_iter_epoch
            timestamp = strftime('%a %d %b %y %X', gmtime())
            print_lm_loss = 0
            print_nsp_loss = 0
            print_lm_nsp_loss = 0
            print_img_loss = 0

            if lm_loss is not None:
                print_lm_loss = lm_loss.item()
            if nsp_loss is not None:
                print_nsp_loss = nsp_loss.item()
            if lm_nsp_loss is not None:
                print_lm_nsp_loss = lm_nsp_loss.item()
            if img_loss is not None:
                print_img_loss = img_loss.item()

            print_format = '[%s][LR: %.7f][Ep: %.2f][Iter: %d][Time: %5.2fs][NSP + LM Loss: %.3g][LM Loss: %.3g][NSP Loss: %.3g][IMG Loss: %.3g]'
            print_info = [
                timestamp, cur_lr, cur_epoch, iter_id, end_t - start_t, print_lm_nsp_loss, print_lm_loss, print_nsp_loss, print_img_loss
            ]
            logger.write(print_format % tuple(print_info))
            start_t = end_t


        if iter_id % num_iter_epoch == 0 and iter_id != start_iter_id:
            torch.save(
                {
                    'model_state_dict' : dialog_encoder.module.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'iter_id':iter_id
                }, 
                os.path.join(
                    params['save_path'], 
                    '%s_%s_%d.ckpt'%(mode, params['chunk'], epoch_id)
                )
            )
            logger.write('\n%d epoch ended.' % epoch_id)

                
