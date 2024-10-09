import pickle
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

class Vocab(object):
    def __init__(self, path, embedding_path = None):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        print('pad token', self.pad)
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']


        if embedding_path is None:
            #print('------- vocab embedding is None')
            #self.embedding = np.random.randn(self.size, 300)
            self.embedding = None
        else:
            print('--------- load vocab embedding from embedding_path: {}'.format(embedding_path))
            with open(embedding_path, 'rb') as f:
                self.embedding = pickle.load(f)

    @staticmethod
    def build(sents, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [wordnet_lemmatizer.lemmatize(w).lower() for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))

    def sen2indexes(self, sentences):
        '''sentence: a string or list of string
           return: a numpy array of word indices
        '''
        def convert_sent(sent, maxlen):
            idxes = np.zeros(maxlen, dtype=np.int64)
            idxes.fill(self.pad)
            tokens = nltk.word_tokenize(sent.strip())
            idx_len = min(len(tokens), maxlen)
            for i in range(idx_len): idxes[i] = self.word2idx.get(tokens[i], self.unk)
            return idxes, idx_len

        if type(sentence) is list:
            inds, lens = [], []
            for sent in sentence:
                idxes, idx_len = convert_sent(sent, maxlen)
                inds.append(idxes)
                lens.append(idx_len)
            return np.vstack(inds), np.vstack(lens)
        else:
            inds, lens = self.sent2indexes([sentence], maxlen)
            return inds[0], lens[0]
