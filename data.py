# data.py: Loads data and builds vocabulary
import json
from nltk.tokenize import word_tokenize
import numpy as np
import os
import torch

label_index = {'entailment': 0,  'neutral': 1, 'contradiction': 2, '<unk>': '<unk>'}


# def prepare_samples(sentences, params):


    # n_w = np.sum([len(x) for x in sentences])

    # # filters words without w2v vectors
    # for i in range(len(sentences)):
    #     s_f = [word for word in sentences[i] if word in self.word_vec]
    #     if not s_f:
    #         import warnings
    #         warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
    #                         Replacing by "</s>"..' % (sentences[i], i))
    #         s_f = [self.eos]
    #     sentences[i] = s_f

    # lengths = np.array([len(s) for s in sentences])
    # n_wk = np.sum(lengths)
    # if verbose:
    #     print('Nb words kept : %s/%s (%.1f%s)' % (
    #                 n_wk, n_w, 100.0 * n_wk / n_w, '%'))

    # # sort by decreasing length
    # lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
    # sentences = np.array(sentences)[idx_sort]

    # return sentences, lengths, idx_sort


def get_batch(batch, word_vec, params_model):

    # sentences, lengths, idx_sort = prepare_samples(batch, params_model)
    batch = [['<s>'] + word_tokenize(s) + ['</s>'] for s in batch]

    embed = np.zeros((len(batch[0]), len(batch), params_model['word_emb_dim']))


    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.FloatTensor(embed)

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
    counter = 0##

    with open(file_name) as f:
        s1, s2, target = [], [], []
        
        for line in f:
            if counter == 4: ##
                break ##
            line = json.loads(line)

            s1.append(line['sentence1'])
            s2.append(line['sentence2'])

            if line['gold_label'] == '-':
                line['gold_label'] = '<unk>' # replcae unknown labels with unk
            
            target.append(label_index[line['gold_label']])

            counter += 1 ##

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
    # word_dict['<p>'] = '' # padding token

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
