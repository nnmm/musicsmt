from gensim.models.word2vec import *
import matplotlib.pyplot as plt


class Music2Vec(Word2Vec):
    """Extension of Word2Vec"""
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5, seed=1, workers=1, min_alpha=0.0001, sg=1):
        # super(Music2Vec, self).__init__(sentences, size, alpha, window, min_count, seed, workers, min_alpha, sg)
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.min_count = min_count
        self.workers = workers
        self.min_alpha = min_alpha
        # from music2vec_inner import train_sentence_sg, train_sentence_cbow, FAST_VERSION
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)


    def copy_weights(self, other):
        self.syn0 = array(other.syn0)


    def plot_words(self, words, subplot=None):
        from numpy import histogram2d
        if not subplot:
            plt.clf()
            subplot = plt
        if not words:
            words = []
            for i in xrange(7):
                words.append(self.index2word[random.randint(len(self.vocab))])

        self.init_sims()

        x, y = self.project_data()
        heatmap, xedges, yedges = histogram2d(x, y, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        wordx, wordy = [x[self.vocab[w].index] for w in words], [y[self.vocab[w].index] for w in words]

        subplot.imshow(heatmap.T, extent=extent, origin='lower', cmap='binary')
        subplot.scatter(wordx, wordy, marker="s", color='red')
        for i, label in enumerate(words):
            subplot.annotate(label, xy=(wordx[i], wordy[i]), color='red')
        return subplot

    def plot_frequencies(self, number=1000, subplot=None):
        from numpy import log
        if not subplot:
            plt.clf()
            subplot = plt
        self.init_sims()
        x, y = self.project_data()
        counts = [log(w.count) for w in self.vocab.values()]
        # http://matplotlib.org/examples/color/colormaps_reference.html - OrRd
        subplot.scatter(x[:number], y[:number], lw=0, c=counts[:number], cmap=plt.cm.get_cmap('winter'))
        return subplot

    def project_data(self):
        from matplotlib.mlab import PCA
        self.init_sims()
        myPCA = PCA(self.syn0norm)
        x = myPCA.Y[:, 0]
        y = myPCA.Y[:, 1]
        return x, y

    def max_count(self):
        maxcount = 0
        for w in self.vocab.values():
            if w.count > maxcount:
                maxcount = w.count
        return maxcount
    




class EmailCorpus(object):
    def __init__(self, filename):
        self.filename = filename
        self.stoplist = ['Am ', '>', '=']

    def __iter__(self):
        for line in open(self.filename):
            ignore = False
            for s in self.stoplist:
                #print type(line)
                #print(line)
                if line.startswith(s):
                    ignore = True
            if not ignore:
                tokenized_line = list(utils.tokenize(line))
                yield tokenized_line


class Translator(object):
    """Given that we have two vector spaces whose vector identities are known, how can we translate new vectors?"""
    def __init__(self, model1, model2):
        self.mnames = [model1, model2]
        self.m = [Music2Vec.load(model1), Music2Vec.load(model2)]
        # list of words in ascending order of index
        # words_ordered = sorted(model1.vocab, key=lambda x: model1.vocab[x].index)
        # # Indices of words_ordered in model2
        # self.mapping = [model2.vocab[w].index for w in words_ordered]
        # model1.init_sims()
        # self.vec1 = model1.syn0norm
        # model2.init_sims()
        # self.vec2 = model2.syn0norm[self.mapping, :]

    def simple_translate(self, vector):
        topn = 10
        # calculate distances of vector to vectors of model1
        dists = dot(self.vec1, vector)
        best = argsort(dists)[::-1][:topn]
        # ignore (don't return) words from the input
        result = [self.mapping[sim] for sim in best]
        return result[:topn]

    def plot_test(self, number=1000):
        from numpy import arange, log
        import datetime
        fig = plt.figure()
        data = arange(900).reshape((30,30))
        for i in range(0,2):
            occ = [log(w.count) for w in self.m[i].vocab.values()]
            px, py = self.m[i].project_data()
            ax = fig.add_subplot(1,2,i+1)
            ax.set_aspect('equal')
            plt.title('PCA of ' + self.mnames[i])
            # ax = self.m[i].plot_frequencies(subplot=ax)
            ax = self.m[i].plot_words(words=None, subplot=ax)
        plt.suptitle('Plot from ' + datetime.today())
        plt.show()   


if __name__ == '__main__':
    t = Translator('models/de_filtered', 'models/en_filtered')
    t.plot_test()