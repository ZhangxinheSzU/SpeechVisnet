import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.common_modules import Transformer

#from noise import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextEncoder(nn.Module):
    def __init__(self, vocab, args):
        super(TextEncoder, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.num_layers,
                dropout=args.dropout if args.num_layers > 1 else 0, bidirectional=True, batch_first = True)
                                                            # cfg.TRANSFORMER.n_heads, cfg.TRANSFORMER.dim_feedforward, args.dropout)
        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.__init_weights__()

    def __init_weights__(self):
        if self.vocab.embedding is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            print('load embedding from vocab')
            self.embed.weight.data.copy_(torch.from_numpy(self.vocab.embedding))
            #self.embed.weight.requires_grad = False

    def forward(self, input, lens):
        input = self.embed(input)
        input = nn.utils.rnn.pack_padded_sequence(input, lens, batch_first = True, enforce_sorted=False)
        x, (hx, cx) = self.E(input)
        sentence_states, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        h1 = torch.cat([hx[-2], hx[-1]], 1)
        h2 = torch.cat([hx[-4], hx[-3]], 1)
        c1 = torch.cat([cx[-2], cx[-1]], 1)
        c2 = torch.cat([cx[-4], cx[-3]], 1)
        h = torch.stack([h1, h2])
        c = torch.stack([c1, c2])
        hidden = (h, c)

        mu = self.h2mu(h1)
        logvar = self.h2logvar(h1)
        z_d = reparameterize(mu, logvar)
        return hidden, z_d, sentence_states 



    def loss_rec(self, logits, targets):
        #print('logits', logits.size(), targets.size())
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return torch.mean(torch.mean(loss, 1))

    def loss_rec_vae(self, logits, targets):
        loss_rec = self.loss_rec(logits, targets).mean()
        loss_kl = loss_kl(mu, logvar)
        return loss_rec + loss_kl



class TextDecoder(nn.Module):
    def __init__(self, vocab, args):
        super(TextDecoder, self).__init__()
        self.drop = nn.Dropout(args.dropout)
        self.G = nn.LSTM(args.dim_emb, 2*args.dim_h, args.num_layers,
            dropout=args.dropout if args.num_layers > 1 else 0, batch_first = True)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.proj = nn.Linear(2*args.dim_h, vocab.size)


    def __init_weights__(self):
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, z, hidden, steps):
        input_z = self.z2emb(z)
        input_z = torch.unsqueeze(input_z, 1)
        inputs = input_z.repeat([1, steps, 1])
        output, hidden = self.G(inputs, hidden)
        output = self.drop(output)
        output = output.reshape(-1, output.size(-1))
        logits = self.proj(output)
        return logits
