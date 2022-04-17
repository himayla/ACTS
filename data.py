# data.py: Loads data and builds vocabulary
import json
from nltk.tokenize import word_tokenize
import numpy as np
import os
import torch

label_index = {'entailment': 0,  'neutral': 1, 'contradiction': 2, '<unk>': '<unk>'}

def get_batch(batch, word_vec, model_params):
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), model_params['word_emb_dim']))
    
    for i in range(len(batch)):
        for j in range(len(batch[i])):
                embed[j, i, :] = word_vec[batch[i][j]]

    return torch.FloatTensor(embed), lengths

def load_data(path_to_dataset):
    data = []

    for type in ['train', 'dev', 'test']:
        for file in os.listdir(path_to_dataset):

            if type in file and file.endswith(".jsonl"):
                data.append(extract_sentences(os.path.join(path_to_dataset, file)))

    train, dev, test = [i for i in data]

    assert len(train) == len(dev) == len(test)
    return train, dev, test

def extract_sentences(file_name):
    # counter = 0##

    with open(file_name) as f:
        s1, s2, target = [], [], []
        
        for line in f:
            # if counter == 100: ##
            #     break ##
            line = json.loads(line)

            s1.append(line['sentence1'])
            s2.append(line['sentence2'])

            if line['gold_label'] == '-':
                line['gold_label'] = '<unk>' # replcae unknown labels with unk
            
            target.append(label_index[line['gold_label']])
        target = np.array(target)
        print(target)
            # counter += 1 ##

    return {'s1': s1, 's2': s2, 'target': target}
    
def get_vocab(sentences):
    word_dict = {}
    sentences = [word_tokenize(s) for s in sentences]

    for sent in sentences:
        for word in sent:            
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''

    return word_dict

def get_glove(glove_path, vocab):
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in vocab: # dataset words
                word_vec[word] = np.fromstring(vec, sep=' ')

    return word_vec

def build_vocab(sentences, glove_path):
    vocab =  get_vocab(sentences)
    word_vec = get_glove(glove_path, vocab)

    return vocab, word_vec
