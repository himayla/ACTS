# data.py: Loads data and builds vocabulary
import json
from nltk.tokenize import word_tokenize
import numpy as np
import os
import torch

label_index = {'entailment': 0,  'neutral': 1, 'contradiction': 2, '<unk>': '<unk>'}

def get_batch(batch, word_vec, params_model):

    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), params_model['word_emb_dim']))

    for i in range(len(batch)):
        if batch[i] in word_vec:
            embed[i, :] = word_vec[batch[i]]/len(batch)
        else:
            embed[i, :] = np.zeros((params_model['word_emb_dim']))

    return torch.from_numpy(embed).float(), lengths


def load_data(path_to_dataset):
    data = []

    for type in ['train', 'dev', 'test']:
        for file in os.listdir(path_to_dataset):

            if type in file and file.endswith(".jsonl"):
                data.append(extract_sentences(os.path.join(path_to_dataset, file)))

    train, dev, test = [i for i in data]

    return train, dev, test

def extract_sentences(file_name):
    counter = 0##

    with open(file_name) as f:
        s1, s2, target = [], [], []
        
        for line in f:
            if counter == 3: ##
                break ##
            line = json.loads(line)


            s1.append(line['sentence1'])
            s2.append(line['sentence2'])

            if line['gold_label'] == '-':
                line['gold_label'] = '<unk>' # replcae unknown labels with unk
            
            target.append(label_index[line['gold_label']])

            counter += 1 ##
    
    return {'s1': s1, 's2':s2, 'target': target}
    
def get_vocab(sentences):
    word_dict = {}
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['</>'] = ''

    print(word_dict)

    return word_dict

def get_glove(glove_path, vocab):
    word_vec = {}

    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in vocab: # dataset words
                word_vec[word] = np.array(list(map(float, vec.split())))
   
    return word_vec

def build_vocab(sentences, glove_path):
    vocab =  get_vocab(sentences)
    word_vec = get_glove(glove_path, vocab)

    return vocab, word_vec