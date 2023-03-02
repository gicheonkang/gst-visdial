import h5py
import os
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import csv
import base64
import lmdb # install lmdb by "pip install lmdb"
import pickle
import argparse
csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('-file_num', default=1000000, type=int)
parsed = vars(parser.parse_args(args=None))

assert parsed['file_num'] != 1000000
count = 0
infiles = []
file_num = parsed['file_num']
name = '/data/cc12m/dialgen/filter/features/cc12m_filtered_%d.tsv.0' % file_num
infiles.append(name)

save_path = '/data/dialgen/data/cc12m/cc12m_img_feat_%d.lmdb' % file_num
if not os.path.exists(save_path):
    os.mkdir(save_path)


id_list = []
env = lmdb.open(save_path, map_size=1099511627776)
with env.begin(write=True) as txn:
    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                img_id = str(item['image_id']).encode()
                id_list.append(img_id)
                txn.put(img_id, pickle.dumps(item))
                if count % 1000 == 0:
                    print(count) 
                count += 1
    txn.put('keys'.encode(), pickle.dumps(id_list))
print(count)


