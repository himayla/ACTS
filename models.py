import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4 * 2 * self.enc_lstm_dim
        self.inputdim = 4 * self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes))

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

class ConvNetEncoder(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        self.convnet1 = nn.Sequential(
            nn.Conv1d(self.word_emb_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        sent = self.convnet1(sent)
        u1 = torch.max(sent, 2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb

#NLI
# class Encoder(nn.Module):
#     def __init__(self, config):
#         super(Encoder, self).__init__()
#         self.bsize = config['bsize'] # Size of mini batches
#         self.n_words = config['n_words']
#         self.word_emb_dim = config['word_emb_dim'] # Hidden layer of 512 units
#         self.dpout_model = config['dpout_model']
#         self.enc_lstm_dim = config['enc_lstm_dim']
#         self.pool_type = config['pool_type']
#         self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
#                                 bidirectional=True, dropout=self.dpout_model)
#         # Optimizer
#         self.fc_dim = config['fc_dim']
#         self.n_classes = config['n_classes'] # Categories
#         self.lr = config['lr'] 
#         self.weight_decay = config['weight_decay']
#         self.momentum = config['momentum']

#         # self.model_name = config['model_name']
#         # self.enc_lstm_dim = config['enc_lstm_dim']
#         # self.embedding = nn.Embedding(self.bsize, self.word_emb_dim)
#         # self.embedding.requires_grad = False
#         # self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1, bidirectional=True, dropout=self.dpout_model)
#         # if config['model_name'] == 'base':
#         #     self.input_dim = config['word_emb_dim'] * 4

#         self.net = nn.Sequential(nn.Linear(self.input_dim, self.fc_dim), 
#                                 nn.Linear(self.fc_dim, self.n_classes))


#     def forward(self, sent_tuple):
#         sent, sent_len = sent_tuple

#         sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
#         sent_len_sorted = sent_len_sorted.copy()
#         idx_unsort = np.argsort(idx_sort)

#         idx_sort = torch.from_numpy(idx_sort)
#         sent = sent.index_select(1, idx_sort)

#         # Handling padding in Recurrent Networks
#         sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
#         sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
#         sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

#         # Un-sort by length
#         idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
#             else torch.from_numpy(idx_unsort)
#         sent_output = sent_output.index_select(1, idx_unsort)

#         # Pooling
#         if self.pool_type == "mean":
#             sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
#             emb = torch.sum(sent_output, 0).squeeze(0)
#             emb = emb / sent_len.expand_as(emb)
#         elif self.pool_type == "max":
#             if not self.max_pad:
#                 sent_output[sent_output == 0] = -1e9
#             emb = torch.max(sent_output, 0)[0]
#             if emb.ndimension() == 3:
#                 emb = emb.squeeze(0)
#                 assert emb.ndimension() == 2

#         return emb


#     def prepare_samples(sentences, word_vec):
#         sentences = [['<s>'] + word_tokenize(s) + ['</s>'] for s in sentences] # Add begin/end mark
#         n_w = np.sum([len(x) for x in sentences]) # Before filter

#         # # Filter out words without vectors
#         for i in range(len(sentences)):
#             s_f = [word for word in sentences[i] if word in word_vec]
#             if not s_f:
#                 import warnings
#                 warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
#                             Replacing by "</s>"..' % (sentences[i], i))
#                 s_f = '</s>'
#             sentences[i] = s_f

#         lengths = np.array([len(s) for s in sentences])
#         n_wk = np.sum(lengths) # After filter

#         print('Nb words kept : %s/%s (%.1f%s)' % (
#                     n_wk, n_w, 100.0 * n_wk / n_w, '%'))

#         lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths) # Mark longest
#         sentences = np.array(sentences)[idx_sort]

#         return(sentences, lengths)

#     def get_batch(batch, word_vec, model_params):
#         sen_lens = np.array([len(x) for x in batch])
#         max_len = np.max(sen_lens)
#         embed = np.zeros((max_len, len(batch), model_params['word_emb_dim']))
        
        
#         for i in range(len(batch)):
#             for j in range(len(batch[i])):
#                 if model_params['model_name'] == 'base':
#                     embed[j, i, :] = word_vec[batch[i][j]] / sen_lens[i]
#                 else:
#                     embed[j, i, :] = word_vec[batch[i][j]]

#         return torch.FloatTensor(embed), sen_lens

#         def encode(self, sentences):
#             tic = time.time()
#             sentences, lengths, idx_sort = self.prepare_samples(
#                             sentences, self.bsize)

#             embeddings = []
#             for stidx in range(0, len(sentences), bsize):
#                 batch = self.get_batch(sentences[stidx:stidx + bsize])
#                 if model.is_cuda():
#                     batch = batch.cuda()
#                 with torch.no_grad():
#                     batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
#                 embeddings.append(batch)
#             embeddings = np.vstack(embeddings)

#             # unsort
#             idx_unsort = np.argsort(idx_sort)
#             embeddings = embeddings[idx_unsort]

#             if verbose:
#                 print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
#                         len(embeddings)/(time.time()-tic),
#                         'gpu' if self.is_cuda() else 'cpu', bsize))
#             return embeddings


# class ClassificationNet(nn.Module):
#     def __init__(self, config):
#         super(ClassificationNet, self).__init__()

#         # classifier
#         self.nonlinear_fc = config['nonlinear_fc']
#         self.fc_dim = config['fc_dim']
#         self.n_classes = config['n_classes']
#         self.enc_lstm_dim = config['enc_lstm_dim']
#         self.encoder_type = config['encoder_type']
#         self.dpout_fc = config['dpout_fc']

#         self.encoder = eval(self.encoder_type)(config)
#         self.inputdim = 2*self.enc_lstm_dim
#         self.inputdim = 4*self.inputdim if self.encoder_type == "ConvNetEncoder" else self.inputdim
#         self.inputdim = self.enc_lstm_dim if self.encoder_type =="LSTMEncoder" else self.inputdim
#         self.classifier = nn.Sequential(
#             nn.Linear(self.inputdim, 512),
#             nn.Linear(512, self.n_classes),
#         )

#     def forward(self, s1):
#         # s1 : (s1, s1_len)
#         u = self.encoder(s1)

#         output = self.classifier(u)
#         return output

#     def encode(self, s1):
#         emb = self.encoder(s1)
#         return emb