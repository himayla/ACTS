# train.py:  Load data, choose model, define training loop
import numpy as np
from data import load_data, build_vocab#, get_batch,
import pandas as pd

# from models import Baseline, LSTM, BiLSTM, BiLSTM_pooling

PATH_TO_DATA = 'data/snli_1.0/'

GLOVE_PATH = 'pretrained/glove.840B.300d.txt' # path to glove embeddings


if __name__ == "__main__":
    train, dev, test = load_data(PATH_TO_DATA)

    # train = pd.DataFrame(train)
    # print(train.head(1))

    word_vec = build_vocab(train['s1'] + train['s2'] +
                            dev['s1'] + dev['s2'] +
                            test['s1'] + test['s2'], GLOVE_PATH)


    # parameters = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
    #             'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    # baseline = Baseline(parameters)