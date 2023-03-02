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
 
    params['mode'] = 'vd_gen_val'
    params['start_path_a'] = 'checkpoints/ckpt_done/student_v1.0.ckpt' 
    params['save_name'] = "cc12m_dialogs_0.txt"
    params['save_path'] = "data/gen_dialog"
    mode = params['mode']

    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'], exist_ok=True)

    if mode == 'vd_gen_val':
        dataset = VisdialDataset(params)
        dataset.mode = 'vd_gen_val'

    dataloader = DataLoader(
        dataset,
        batch_size=1,
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
    
    iid2idx = {}
    val_data = dataset.visdial_data_val['data']['dialogs']
    for idx, dial_data in enumerate(val_data):
        iid2idx[dial_data['image_id']] = idx    
    os.system('clear')

    with torch.no_grad():
        while True:
            iid = input('please enter the image id: ')
            if iid == 'quit':
                print('-------------------------------------------')
                break
            try:
                idx = iid2idx[int(iid)]
                batch = dataset[idx]
            except:
                print('image id {} does not exist in the queue'.format(iid))
                continue

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
            enc_image_features = batch['enc_image_feat'].unsqueeze(0).to(params['device'])
            enc_image_spatials = batch['enc_image_loc'].unsqueeze(0).to(params['device'])
            enc_image_mask = batch['enc_image_mask'].unsqueeze(0).to(params['device'])

            # auto-regressive generation 
            abnormal_sample = []
            ques_list = []
            ans_list = []
            batch_size = enc_input_ids.size(0)
            enc_input_len = torch.sum((enc_input_ids!=0), dim=-1)

            for rnd in range(num_round):
                q = input("please ask question to AI #{}: ".format(rnd+1))
                if q == 'quit':
                    break    
                ques_ids = dataset.tokenizer.encode(q)
                ques_ids.append(102)
                ques_ids = torch.LongTensor(ques_ids).unsqueeze(0).to(params['device'])
            
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

                # get the length of answer sequence
                ans_len = torch.sum((ans_ids!=0), dim=-1)

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

                txt_ans = decode_data(dataset.tokenizer, ans_ids)
                print("answer for the question   #{}: {}".format(rnd+1, txt_ans[0]))
            print('-------------------------------------------')


