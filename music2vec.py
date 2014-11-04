from gensim.models.word2vec import *
import matplotlib.pyplot as plt

class Music2Vec(Word2Vec):
    def __init__(self, *args, **kwargs):
        Word2Vec.__init__(*args, **kwargs)

    def print_vocab(self):
        return list(self.vocab.keys())

    def random_vectors(size=100):
        r = Music2Vec(size=size)
        r.reset_weights()
        r.init_sims(replace=True)
        return r

    def reduce_vocabulary(self, language="all"):
        allowed_langs = ["american-english", "french", "british-english", "ngerman"]
        if language == "all":
            words = set()
            for lang in allowed_langs:
                langset = set(open('/usr/share/dict/' + lang).read().splitlines())
                words = words.union(langset)
        elif language in allowed_langs:
            words = set(open('/usr/share/dict/' + language).read().splitlines())
        else:
            return
        words = words & set(self.vocab.keys())
        good_vocab = {utils.to_utf8(w): self.vocab[w] for w in words}
        self.vocab = good_vocab


    # --- helper functions ---


    def intersect_vocab(self, wordlist):
        return list(set(self.vocab.keys()) & set(wordlist))


    def random_words(self, number=5):
        words = []
        for i in xrange(number):
            words.append(self.index2word[random.randint(len(self.vocab))])
        return words


    def max_count(self):
        """
        Helper function to find the most frequent word.
        """
        maxcount = 0
        for w in self.vocab.values():
            if w.count > maxcount:
                maxcount = w.count
        return maxcount


    def replace_with_PCA(self):
        from matplotlib.mlab import PCA
        myPCA = PCA(self.syn0norm)
        self.syn0norm = myPCA

    # --- end helper functions ---


    # --- plotting functions ---

    def get_plot_data(self, do_PCA=True):
        """
        Return the first two PCA dimensions or just the first two dimensions of syn0norm.
        """
        self.init_sims()
        if do_PCA:
            from matplotlib.mlab import PCA
            myPCA = PCA(self.syn0norm)
            return myPCA.Y[:, 0:2]
        else:
            return self.syn0norm[:, 0:2]


    def plot_heatmap(self, subplot, points):
        """
        Plots a heatmap in the given plot/subplot.
        """
        from numpy import histogram2d
        heatmap, xedges, yedges = histogram2d(points[:,0], points[:,1], bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        subplot.imshow(heatmap.T, extent=extent, origin='lower', cmap='binary')

    @staticmethod
    def plot_points(subplot, points, labels, colors=None):
        """
        More general plotting method, for internal use or advanced use.
        Plot points with associated labels in the given plot/subplot.
        """
        if colors is None:
            colors = [0]*len(labels)
        # lw is the linewidth, s is size
        subplot.scatter(points[:,0], points[:,1], lw=0, s=50, c=colors, cmap=plt.cm.get_cmap('winter'))
        if labels:
            for i, w in enumerate(labels):
                subplot.annotate(w, (points[i,0], points[i,1]), textcoords='offset points', xytext=(5, 5), color='darkblue')


    def plot_words(self, subplot, words, do_PCA=True):
        """
        Plot words from the vocabulary in the given plot/subplot, colored by frequency.
        """
        from numpy import log
        
        counts = [log(self.vocab[w].count) for w in words]
        word_indices = [self.vocab[w].index for w in words]
        all_points = self.get_plot_data(do_PCA)
        Music2Vec.plot_points(subplot, all_points[word_indices, :], words, counts)
        
        
    def visualize(self, words=None, do_PCA=True):
        """
        Plot a heatmap or, if the words attribute is set, the respective words, and show it.
        """
        plt.clf()
        points = self.get_plot_data(do_PCA)
        if words is not None:
            self.plot_words(plt, words, do_PCA)
        self.plot_heatmap(plt, points)
        plt.show()


    def frequency_histogram(self):
        """
        Plot a histogram of relative word frequency.
        """
        from numpy import log, logspace

        plt.clf()
        counts = [w.count for w in self.vocab.values()]
        total_count = float(sum(counts))
        freqs = [c/total_count for c in counts]
        # alternatively, logspace with end log(self.max_count())/log(10)
        n, bins, patches = plt.hist(freqs, logspace(-10, 0, 20), facecolor='green', normed=False, log=True, alpha=0.5)
        plt.gca().set_xscale("log")
        plt.xlabel('Word frequency')
        plt.ylabel('Words in frequency bin')
        plt.title('Frequency histogram')
        plt.show()


    def plot_linear_demo(self, positive=['king', 'woman'], negative=['queen'], solution='man'):

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        pos_weighted = [(word, 1.0) for word in positive]
        neg_weighted = [(word, -1.0) for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in pos_weighted + neg_weighted:
            if word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        best = argsort(dists)[::-1][:1 + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]

        labels = positive + negative + [solution, 'X']
        points = array([self.syn0norm[self.vocab[word].index] for word in labels[:-1]] + [mean])
        from matplotlib.mlab import PCA
        myPCA = PCA(self.syn0norm)
        points = [myPCA.project(point) for point in points]

        plt.clf()
        Music2Vec.plot_points(plt, points, labels)
        plt.axhline(0)
        plt.axvline(0)
        plt.title('Linear demo')
        plt.xlabel('Word frequency')
        plt.ylabel('Words in frequency bin')
        plt.show()
        import code
        code.interact(local=locals())


    # --- end plotting functions ---




    def least_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = -matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]



class OANCCorpus(object):
    """Iterate over sentences from the OANC corpus"""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            print(fname)
            continue
            for line in open(fname):
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words

    def test(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            continue
            for line in open(fname):
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                #yield words


class Translator(object):
    """Given that we have two vector spaces whose vector identities are known, how can we translate new vectors?"""
    def __init__(self, model1, model2):
        self.mnames = [model1, model2]
        # self.m = [Music2Vec.load(model1), Music2Vec.load(model2)]
        # list of words in ascending order of index
        # words_ordered = sorted(model1.vocab, key=lambda x: model1.vocab[x].index)
        # # Indices of words_ordered in model2
        # self.mapping = [model2.vocab[w].index for w in words_ordered]
        # model1.init_sims()
        # self.vec1 = model1.syn0norm
        # model2.init_sims()
        # self.vec2 = model2.syn0norm[self.mapping, :]

    def get_random_words(self):
        # http://stackoverflow.com/questions/3540288/how-do-i-read-a-random-line-from-one-file-in-python
        # http://stackoverflow.com/questions/10819911/read-random-lines-from-huge-csv-file-in-python
        filesize = 938848
        offset = random.randint(filesize)
        dictfile = '/usr/share/dict/american-english'
        with open(dictfile) as f:
            f.seek(offset)
            f.readline()
            random_line = f.readline()      # bingo!

            # extra to handle last/first line edge cases
            if len(random_line) == 0:       # we have hit the end
                f.seek(0)
                random_line = f.readline()
        print(random_line)

    def simple_translate(self, vector):
        topn = 10
        # calculate distances of vector to vectors of model1
        dists = dot(self.vec1, vector)
        best = argsort(dists)[::-1][:topn]
        # ignore (don't return) words from the input
        result = [self.mapping[sim] for sim in best]
        return result[:topn]

    def plot_freq_comparison(self):
        from numpy import log
        import datetime
        fig = plt.figure()
        plt.suptitle('Plot from ' + str(datetime.date.today()))
        for i in range(0,2):
            ax = fig.add_subplot(1,2,i+1)
            ax.set_aspect('equal')
            plt.title('PCA of ' + self.mnames[i])
            ax = self.m[i].plot_frequencies(subplot=ax)
        plt.show()

    def plot_word_comparison(self):
        from numpy import log
        import datetime
        words = []
        for i in xrange(7):
            words.append(self.m[0].index2word[random.randint(len(self.m[0].vocab))])
        words = ['resource', 'tree', 'school', 'dangerous', 'views', 'never', 'sell']
        fig = plt.figure()
        plt.suptitle('Data projected by PCA, plot from ' + str(datetime.date.today()))
        for i in range(0,2):
            ax = fig.add_subplot(1,2,i+1)
            ax.set_aspect('equal')
            plt.title(self.mnames[i])
            # ax = self.m[i].plot_frequencies(subplot=ax)
            self.m = Music2Vec.load(self.mnames[i])
            ax = self.m.plot_words(words=words, subplot=ax)
            del self.m
        plt.show()


if __name__ == '__main__':
    #t = Translator('models/enwiki300M', 'models/frwiki300M')
    #t.get_random_words()
    m = Music2Vec.load('/home/internet/Dropbox/Bachelorarbeit/Code/my/models/enwiki300M')
    m.slim_down()
    m.print_vocab()
    # c = OANCCorpus('/home/internet/Downloads/OANC-GrAF/data')
    # c.test()