# train.py:  Load data, choose model, define training loop

from data import load_data, build_vocab, get_batch
import numpy as np
import torch.nn as nn
import torch
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from models import SNLI #, LSTM, BiLSTM, BiLSTM_pooling

PATH_TO_DATA = 'dataset/'
GLOVE_PATH = 'pretrained/glove.840B.300d.txt'

params_model = {'model_name': 'base', 'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0,\
                'fc_dim': 512, 'n_classes': 3, 'learning_rate': 0.1, 'weight_decay': 1e-4, 'momentum': 0.08, 'max_epoch': 50}

def evaluate_model(model):
    # TO DO
    print("*EVALUATING MODEL*")
    score = 0
    model.eval()
    with torch.no_grad():
        for i, (premise, hyp, label) in enumerate(test_batch):
            out = model(premise[0], premise[1], hyp[0], hyp[1])
            preds = torch.argmax(out, dim=1)
            accuracy = torch.sum(preds == label, dtype=torch.float32) / out.shape[0]
            score += accuracy
        score /= i
    return score

if __name__ == "__main__":
    print("*PREPARING FOR TRAINING*")

    demo = {'s1': ['A person on a horse jumps over a broken down airplane.'], 's2': [['A person is training his horse for a competition.']], 'target': [1]}

    print("*BUILDING THE TEST VOCAB*")
    vocab, embeddings = build_vocab(demo['s1'] + demo['s2'], GLOVE_PATH)

    print(f'demo:{vocab}')

 
    print("*BUILDING THE VOCAB*")
    train, dev, test = load_data(PATH_TO_DATA)
    vocab, embedings = build_vocab(train['s1'] + train['s2'] +
                            dev['s1'] + dev['s2'] +
                            test['s1'] + test['s2'], GLOVE_PATH)
    print(f'real:{vocab}')

    print('Found {0} words from the {1}) words with glove vectors'.format(len(embeddings), len(vocab)))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # s1 = train['s1']
    # s2 = train['s2']
    # target = train['target']

    # for stidx in range(0, len(s1), params_model['bsize']):
    #     s1_batch, s1_len = get_batch(s1[stidx:stidx + params_model['bsize']], embedings, params_model)
    #     s2_batch, s2_len = get_batch(s2[stidx:stidx + params_model['bsize']], embedings, params_model)



    # print("*SETTING UP SNLI MODEL*")
    # model = SNLI(params_model)

    # criterion = nn.CrossEntropyLoss() # Loss function
    # optimizer = torch.optim.SGD(model.parameters(), params_model['learning_rate'], params_model['weight_decay'], params_model['momentum'])

    # print("*TRAINING*")
    # writer = SummaryWriter(os.path.join('logs', params_model['model_name']))
    # model.train()