from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from collections import defaultdict

class Loader:

    def __init__(self, modelpath):
        with open(modelpath + 'vocab_list.txt', 'r') as f:
            self.model_vocab = f.read().splitlines()

    def read_word2vec_file(self):
        tmp_file = get_tmpfile("/mnt/Data/zero_shot_reg/gloVe/test_word2vec.txt")
        self.model = KeyedVectors.load_word2vec_format(tmp_file)

    def test(self):
        print self.model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
        # print model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'], topn=1)  ## only for analogies
        print self.model.similar_by_vector(self.model['horse'], topn=3)

    def get_vector(self, word):
        if word in self.model.vocab:
            return self.model[word]
        else:
            return []

    def embeddings_for_vocab(self):
        self.embeddings  = defaultdict()
        for word in self.model_vocab:
            vector = self.get_vector(word)
            if not len(vector) == 0:
                self.embeddings[word] = vector
            else:
                print word

if __name__ == '__main__':
    loader = Loader('./../eval/model/with_reduced_cats_2/')
    loader.read_word2vec_file()
    loader.embeddings_for_vocab()
  #  loader.test()