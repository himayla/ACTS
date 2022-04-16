import torch
import torch.nn as nn


class SNLI(nn.Module):

    def __init__(self, config):
        super(SNLI, self).__init__()
        self.model_name = config['model_name']
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['pool_type']

        self.embedding = nn.Embedding(self.bsize, self.word_emb_dim)
        self.embedding.requires_grad = False
        self.fc_dim = config['fc_dim']
        self.num_classes = config['n_classes']

        # Needed for optimizer
        self.net = nn.Sequential(nn.Linear(4*self.enc_lstm_dim, self.fc_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.fc_dim, self.num_classes))


    def forward(self, s1, s1_len, s2, s2_len):
        s1 = self.embedding(s1)
        s2 = self.embedding(s2)

        u = self.encoder(s1, s1_len)
        v = self.encoder(s2, s2_len)
        feat = torch.cat((u, v, torch.abs(u - v), u * v), dim=1)

        out = self.net(feat)
        return out
 
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, embed, length):
        out = torch.sum(embed, dim=0) / length.unsqueeze(1).float()
        return out


#class MLP(nn.Moduke):
#     def __init__(self, n_inputs):
#         super(MLP, self).__init__()
#         self.layer = nn.Linear(n_inputs, 1)
#         self.activation = nn.Sigmoid()

#     def forward(self, x):
#         x = self.layer
#         x = self.activation

#         return x


# class CBOW(nn.Module):
#     def __init__(self, vocab_size, context_size, embedding_dim):
#         super(CBOW, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
#         self.linear2 = nn.Linear(128, self.vocab_size)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.view(1, -1)
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         x = F.log_softmax(x)
#         return x