import os
import concurrent.futures
import json
import argparse
import glob
import importlib  
import sys
from pytorch_transformers.tokenization_bert import BertTokenizer

import torch

def read_options(argv=None):
    parser = argparse.ArgumentParser(description='Options')    
    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-visdial_train', default='./visdial_0.9_train.json', help='json file containing train split of visdial data')
    parser.add_argument('-visdial_val', default='./visdial_0.9_val.json',
                            help='json file containing val split of visdial data')
    parser.add_argument('-max_seq_len', default=256, type=int,
                            help='the max len of the input representation of the dialog encoder')
    #-------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-save_path_train', default='/data/dialgen/data/visdial/visdial_0.9_train_processed.json',
                            help='Path to save processed train json')
    parser.add_argument('-save_path_val', default='/data/dialgen/data/visdial/visdial_0.9_val_processed.json',
                            help='Path to save val json')

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg)) 
    return parsed

if __name__ == "__main__":
    params = read_options() 
    # read all the three splits 

    f = open(params['visdial_train'])
    input_train = json.load(f)
    input_train_data = input_train['data']['dialogs']
    train_questions = input_train['data']['questions']
    train_answers = input_train['data']['answers']
    f.close()

    f = open(params['visdial_val'])
    input_val = json.load(f)
    input_val_data = input_val['data']['dialogs']
    val_questions = input_val['data']['questions']
    val_answers = input_val['data']['answers'] 
    f.close()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_seq_len = params["max_seq_len"]
    num_illegal_train = 0
    num_illegal_val = 0
    num_illegal_test = 0
    # process train
    i = 0
    while i < len(input_train_data):
        cur_dialog = input_train_data[i]['dialog']
        caption = input_train_data[i]['caption']
        tot_len = 22 + len(tokenizer.encode(caption)) # account for 21 sep tokens, CLS token and caption
        for rnd in range(len(cur_dialog)):
            tot_len += len(tokenizer.encode(train_answers[cur_dialog[rnd]['answer']]))
            tot_len += len(tokenizer.encode(train_questions[cur_dialog[rnd]['question']]))
        if tot_len <= max_seq_len:
            i += 1
        else:
            input_train_data.pop(i)
            num_illegal_train += 1

    # process val
    i = 0
    while i <  len(input_val_data):
        remove = False
        cur_dialog = input_val_data[i]['dialog']
        caption = input_val_data[i]['caption']
        tot_len = 1 # CLS token
        tot_len += len(tokenizer.encode(caption)) + 1
        for rnd in range(len(cur_dialog)):
            tot_len += len(tokenizer.encode(val_questions[cur_dialog[rnd]['question']])) + 1
            for option in cur_dialog[rnd]['answer_options']:
                cur_len = len(tokenizer.encode(val_answers[option])) + 1 + tot_len
                if cur_len > max_seq_len:
                    input_val_data.pop(i)
                    num_illegal_val += 1
                    remove = True
                    break
            if not remove:
                tot_len += len(tokenizer.encode(val_answers[cur_dialog[rnd]['answer']])) + 1
            else:
                break
        if not remove:
            i += 1
    
    '''
    # store processed files
    '''
    with open(params['save_path_train'],'w') as train_out_file:
        json.dump(input_train, train_out_file)

    with open(params['save_path_val'],'w') as val_out_file:
        json.dump(input_val, val_out_file)
    
    # spit stats
    print("number of illegal train samples", num_illegal_train)
    print("number of illegal val samples", num_illegal_val)
