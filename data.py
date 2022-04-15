# data.py: Loads data and builds vocabulary
import json
from nltk.tokenize import word_tokenize
import numpy as np
import os

word_dict = {}

def load_data(dir_name):
    data = []

    for type in ['train', 'dev', 'test']:
        for file in os.listdir(dir_name):

            if type in file and file.endswith(".jsonl"):
                data.append(extract_sentences(os.path.join(dir_name, file)))

    train, dev, test = [i for i in data]

    return train, dev, test

def extract_sentences(file):
    counter = 0##

    with open(file) as f:

        s1, s2, target = [], [], []
        label_index = {'entailment': 0,  'neutral': 1, 'contradiction': 2, '<unk>': '<unk>'}
        
        for line in f:
            if counter == 200: ##
                break ##
            line = json.loads(line)

            s1.append(word_tokenize(line['sentence1']))
            s2.append(word_tokenize(line['sentence2']))

            if line['gold_label'] == '-':
                line['gold_label'] = '<unk>' # replcae unknown with unk
            
            target.append(label_index[line['gold_label']])

            counter += 1##

    return {'s1': s1, 's2':s2, 'target': target}
    
def get_vocab(sentences):
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['</>'] = ''

def get_glove(glove_path):
    word_vec = {}

    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict: # dataset words
                word_vec[word] = np.array(list(map(float, vec.split())))
   
    print('Found {0}(from the {1}) words with glove vectors'.format(len(word_vec), len(word_dict)))

    return word_vec

def build_vocab(sentences, glove_path):
    get_vocab(sentences)
    word_vec = get_glove(glove_path)

    print('Vocabulary size: {0}'.format(len(word_vec)))

    return word_vec