# train.py:  Load data, choose model, define training loop
import argparse
import time
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import os 
from models import NLINet #, LSTM, BiLSTM, BiLSTM_pooling
from data import load_data, build_vocab, get_batch#, prepare_samples

PATH_TO_DATA = 'dataset/'
GLOVE_PATH = 'pretrained/glove.840B.300d.txt'

parser = argparse.ArgumentParser(description="Natural Language Inference training")

# paths
parser.add_argument('--model_name', type=str, default='base')
parser.add_argument("--nlipath", type=str, default=PATH_TO_DATA, help="SNLI data path")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default=GLOVE_PATH, help="Word embeddings file")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--bsize", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="Encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="Classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="Use nonlinearity in fc") # fc/fullyclosed = hidden layers

# parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument('--lr', type = float, default = 0.1, help='learning rate for training')
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'weight decay for optimizer')
parser.add_argument('--momentum', type = float, default = 0.8, help = 'momentum for optimizer')

parser.add_argument("--lrshrink", type=float, default=5, help="hrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
# model
parser.add_argument("--encoder_type", type=str, default='ConvNetEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu 
parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######### DATA #########
train, dev, test = load_data(PATH_TO_DATA)
vocab, word_vec = build_vocab(train['s1'] + train['s2'] +
                                dev['s1'] + dev['s2'] +
                                test['s1'] + test['s2'], GLOVE_PATH)
print('Found {0} words from the {1} words with glove vectors'.format(len(word_vec), len(vocab)))

for split in ['s1', 's2']:
    for data_type in ['train', 'dev', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])


######### MODEL #########
print("*SETTING UP MODEL*")
params_model = {
    'bsize'          :  params.bsize     ,
    'n_words'        :  len(word_vec)         ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'dpout_model'    :  params.dpout_model    ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'pool_type'      :  params.pool_type      ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'n_classes'      :  params.n_classes      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  False                 ,
    'lr'             :  params.lr             ,
    'weight_decay'   :  params.weight_decay   ,
    'momentum'       :  params.momentum        ,

}

print("*TRAINING*")
model = NLINet(params_model)
#print(model)

# loss
weight = torch.FloatTensor(params_model['n_classes']).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# # optimizer
optimizer = torch.optim.SGD(model.parameters(), params_model['lr'], params_model['weight_decay'], params_model['momentum'])

# ######### TRAIN #########
val_acc_best = -1e10
adam_stop = False
stop_training = False

def train_model(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    
    model.train()

    all_costs = []
    logs = []

    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))


    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['target'][permutation]

    for stidx in range(0, len(s1), params_model['bsize']):

        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params_model['bsize']],
                                     word_vec, params_model)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params_model['bsize']],
                                     word_vec, params_model)

        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params_model['bsize']]))
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = model((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params_model['bsize']])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data)
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params_model['word_emb_dim']

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm

        # optimizer step
        optimizer.step()

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params_model['bsize'] / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = [np.round(100 * correct/len(s1), 2)]
    print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))

    return train_acc

def evaluate(epoch, eval_type='dev', final_eval=False):
    model.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'dev':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = dev['s1'] if eval_type == 'dev' else test['s1']
    s2 = dev['s2'] if eval_type == 'dev' else test['s2']
    target = dev['target'] if eval_type == 'dev' else test['target']

    for i in range(0, len(s1), params_model['bsize']):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params_model['bsize']], word_vec, params_model)
        s2_batch, s2_len = get_batch(s2[i:i + params_model['bsize']], word_vec, params_model)

        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[i:i + params_model['bsize']]))


        # model forward
        output = model((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)

    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:

        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(model.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 1e-5:
                    terminate_training = True
                    break
                param_group['lr'] /= 5
                print("Learning rate changed to :  {}\n".format(param_group['lr']))

    return eval_acc


epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = train_model(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1


# Run best model on test set.
model.load_state_dict(torch.load(os.path.join(params_model.outputdir, params_model.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save encoder instead of full model
torch.save(model.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
