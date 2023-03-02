import torch
import torch.nn as nn
import torch.nn.functional as F
import options
import pprint
import json
import os
from dataloader.dataloader_cc12m_gen import CC12mDataset
from dataloader.dataloader_visdial_gen import VisdialDataset
from torch.utils.data import DataLoader

from models.visual_dialog_encoder import VisualDialogEncoder
from models.visual_dialog_decoder import VisualDialogDecoder
from models.visual_dialog_model import EncoderDecoderModel
from utils.data_utils import batch_iter


def decode_data(tokenizer, seq):
    seq = seq.tolist()
    return [
        tokenizer.decode(item, skip_special_tokens=True)    
        for item in seq
    ]

if __name__ == '__main__':

    params = options.read_command_line()
    pprint.pprint(params)

    mode = params['mode']
    assert mode == 'vd_gen_val' or mode == 'cc12m_gen'
    assert params['save_name'] != ''

    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'], exist_ok=True)

    if mode == 'vd_gen_val':
        dataset = VisdialDataset(params)
        dataset.mode = 'vd_gen_val'
    else:
        dataset = CC12mDataset(params)
        dataset.mode = 'cc12m_gen'

    dataloader = DataLoader(
        dataset,
        batch_size= params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
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

    assert params['start_path_q'] != "" and params['start_path_a'] != ""
    q_encoder = VisualDialogEncoder(params)
    q_decoder = VisualDialogDecoder(params)
    q_decoder.decoder.bert.embeddings = q_encoder.bert_pretrained.bert.embeddings
    q_model = EncoderDecoderModel(params, q_encoder, q_decoder).to(device)
    q_model = nn.DataParallel(q_model, params["gpu_ids"])
    model_state_dict_q = torch.load(params['start_path_q'], map_location=device)
    q_model.module.load_state_dict(model_state_dict_q['model_state_dict'])
    print("model succesfully loaded from {}".format(params['start_path_q']))
    q_model.eval()

    a_encoder = VisualDialogEncoder(params)
    a_decoder = VisualDialogDecoder(params)
    a_decoder.decoder.bert.embeddings = a_encoder.bert_pretrained.bert.embeddings
    a_model = EncoderDecoderModel(params, a_encoder, a_decoder).to(device)
    a_model = nn.DataParallel(a_model, params["gpu_ids"])
    model_state_dict_a = torch.load(params['start_path_a'], map_location=device)
    a_model.module.load_state_dict(model_state_dict_a['model_state_dict'])
    print("model succesfully loaded from {}".format(params['start_path_a']))
    a_model.eval()

    num_round = 10
    max_seq_len = params['max_seq_len']
    gen_data_json = []

    url_to_cap = json.load(open("./data/url_to_cap.json", "r"))
    image_id_to_url = json.load(open("./data/image_id_to_url.json", "r"))

    with torch.no_grad():
        for epoch_id, idx, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break

            # language stuff
            enc_input_ids = batch['enc_input_ids']
            enc_segments = batch['enc_segments']
            enc_att_mask = batch['enc_att_mask']
            dec_input_ids = batch['dec_input_ids']
            dec_att_mask = batch['dec_att_mask']

            enc_input_ids = enc_input_ids.view(-1,enc_input_ids.shape[-1]).to(params['device'])
            enc_segments = enc_segments.view(-1, enc_segments.shape[-1]).to(params['device'])
            enc_att_mask = enc_att_mask.view(-1, enc_att_mask.shape[-1]).to(params['device'])
            dec_input_ids = dec_input_ids.view(-1,dec_input_ids.shape[-1]).to(params['device'])
            dec_att_mask = dec_att_mask.view(-1, dec_att_mask.shape[-1]).to(params['device'])

            # image stuff
            enc_image_features = batch['enc_image_feat'].to(params['device']) 
            enc_image_spatials = batch['enc_image_loc'].to(params['device'])
            enc_image_mask = batch['enc_image_mask'].to(params['device'])

            # auto-regressive generation
            # question generation
            abnormal_sample = []
            ques_list = []
            ans_list = []
            ppl_list = []
            batch_size = enc_input_ids.size(0) 
            enc_input_len = torch.sum((enc_input_ids!=0), dim=-1)

            for rnd in range(num_round):
                # question generation
                ques_ids = q_model(
                    enc_image_features=enc_image_features,
                    enc_image_spatials=enc_image_spatials,
                    enc_image_mask=enc_image_mask,
                    enc_image_target=None,
                    enc_image_label=None,
                    enc_next_sentence_labels=None,
                    enc_input_ids=enc_input_ids,
                    enc_segments=enc_segments,
                    enc_sep_indices=None,
                    enc_mlm_labels=None,
                    enc_attention_mask=enc_att_mask,
                    dec_input_ids=dec_input_ids,
                    dec_attention_mask=dec_att_mask,
                    temperature=0.7,
                    top_k=7,
                    top_p=0.0,
                    ngram_blocking_size=4,
                )

                # get the length of question sequence
                ques_len = torch.sum((ques_ids!=0), dim=-1)

                # add generated question to the context
                for iidx in range(batch_size):
                    start = enc_input_len[iidx]
                    end = start + ques_len[iidx]
                    try:
                        # if length exceeds max_seq_len, exception occurs  
                        enc_input_ids[iidx, start:end] = ques_ids[iidx, :ques_len[iidx]].clone() # error occur?
                    except RuntimeError:
                        enc_input_ids[iidx, start:start+1] = torch.LongTensor([dataset.SEP])
                        ques_len[iidx] = 1
                        abnormal_sample.append(iidx)

                enc_att_mask = (enc_input_ids!=0).float()
                enc_input_len += ques_len        

                # answer generation
                ans_ids = a_model(
                    enc_image_features=enc_image_features,
                    enc_image_spatials=enc_image_spatials,
                    enc_image_mask=enc_image_mask,
                    enc_image_target=None,
                    enc_image_label=None,
                    enc_next_sentence_labels=None,
                    enc_input_ids=enc_input_ids,
                    enc_segments=enc_segments,
                    enc_sep_indices=None,
                    enc_mlm_labels=None,
                    enc_attention_mask=enc_att_mask,
                    dec_input_ids=dec_input_ids,
                    dec_attention_mask=dec_att_mask,
                    temperature=0.7,
                    top_k=7,
                    top_p=0.0,
                    ngram_blocking_size=0,
                )

                # kind of trick to get perplexity score
                # ppl = torch.exp(negative mean of log likelihood)
                a_model.module.params['mode'] = 'train'
                ans_ids_att_mask = (ans_ids!=0).float()
                loss, _ = a_model(
                    enc_image_features=enc_image_features,
                    enc_image_spatials=enc_image_spatials,
                    enc_image_mask=enc_image_mask,
                    enc_image_target=None,
                    enc_image_label=None,
                    enc_next_sentence_labels=None,
                    enc_input_ids=enc_input_ids,
                    enc_segments=enc_segments,
                    enc_sep_indices=None,
                    enc_mlm_labels=None,
                    enc_attention_mask=enc_att_mask,
                    dec_input_ids=ans_ids,
                    dec_attention_mask=ans_ids_att_mask,
                    loss_reduction=False
                )

                # get the length of answer sequence
                ans_len = torch.sum((ans_ids!=0), dim=-1)
                loss = loss.reshape(batch_size, 18)
                loss = loss.sum(-1) / ans_len 

                ppl = torch.exp(loss)
                ppl_list.append(ppl)
                a_model.module.params['mode'] = mode

                # add generated answer to the context
                for iidx in range(batch_size):
                    start = enc_input_len[iidx]
                    end = start + ans_len[iidx]
                    try: 
                        # if length exceeds max_seq_len, exception occurs  
                        enc_input_ids[iidx, start:end] = ans_ids[iidx, :ans_len[iidx]].clone() 
                    except RuntimeError:
                        enc_input_ids[iidx, start:start+1] = torch.LongTensor([dataset.SEP])
                        ans_len[iidx] = 1
                        end = start + 1
                        abnormal_sample.append(iidx) 

                    enc_segments[iidx, start:end] = torch.ones(ans_len[iidx], dtype=torch.long).to(params['device'])
                enc_att_mask = (enc_input_ids!=0).float()
                enc_input_len += ans_len

                txt_ques = decode_data(dataset.tokenizer, ques_ids)
                txt_ans = decode_data(dataset.tokenizer, ans_ids)
                ques_list.append(txt_ques)
                ans_list.append(txt_ans)

            for j in range(batch_size):
                if j in abnormal_sample:
                    continue

                imgid = batch["image_id"][j].item()
                url = image_id_to_url[str(imgid)]
                cap = url_to_cap[url]
                        
                gen_data_json.append(
                    {
                        "image_id": imgid,
                        "url": url,
                        "caption": cap,     
                        "dialog": [
                            {
                                'question': ques_list[k][j],
                                'answer': ans_list[k][j],
                                'answer_ppl': ppl_list[k][j].item()
                            }
                            for k in range(num_round)
                        ]
                    }    
                )
    json.dump(gen_data_json, open(os.path.join(params['save_path'], params['save_name']), "w"))




