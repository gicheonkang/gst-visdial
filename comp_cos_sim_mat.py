# This code is modified from https://github.com/jind11/TextFooler/blob/master/attack_nli.py
import numpy as np
import sys
import pickle

embedding_path = sys.argv[1]

embeddings = []
with open(embedding_path, 'r') as ifile:
    for line in ifile:
        embedding = [float(num) for num in line.strip().split()[1:]]
        embeddings.append(embedding)
embeddings = np.array(embeddings)
print(embeddings.T.shape)
norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = np.asarray(embeddings / norm, "float32")
product = np.dot(embeddings, embeddings.T)
np.save(('./data/visdial/cos_sim_counter_fitting.npy'), product)

idx2word = {}
word2idx = {}

with open(embedding_path, 'r') as ifile:
    for line in ifile:
        word = line.split()[0]
        if word not in idx2word:
            idx2word[len(idx2word)] = word
            word2idx[word] = len(idx2word)-1

with open('./data/visdial/cos_sim_idx2word.pickle', 'wb') as f:
    pickle.dump(idx2word, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/visdial/cos_sim_word2idx.pickle', 'wb') as f:
    pickle.dump(word2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
