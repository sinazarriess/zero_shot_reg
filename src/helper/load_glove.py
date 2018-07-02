import pandas as pd
import csv
import numpy as np
import gensim

glove_data = '/mnt/Data/zero_shot_reg/gloVe/glove.6B.300d.txt'
gensim_file = 'glove_model.txt'

class Embeddings:


    def load_raw_glove_file(self):
        self.glove_data_file = glove_data
        self.words = pd.read_table(self.glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        self.words_matrix = self.words.as_matrix()
        print "loaded embeddings file"

    def vec_for_word(self, w):
        return self.words.loc[w].as_matrix()

    def find_closed_word_for_vec(self, v):
        diff = self.words_matrix - v
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        return self.words.iloc[i].name

    # code von sina
    def words2embedding_weighted(word_predictions, word_probs, word_vectors):
        vecs = []
        for i in range(len(word_predictions)):
            wlist = word_predictions[i]
            plist = word_probs[i]
            new_vec = np.sum([word_vectors[w].astype(float) * plist[x] for x, w in enumerate(wlist)], axis=0)
            vecs.append(new_vec)

        vecs = np.array(vecs)
        # print "Embedding matrix",vecs.shape
        return vecs

   # def find_nearest_neighbor(self, candidates):

if __name__ == "__main__":
    e = Embeddings()
    e.load_raw_glove_file()
    vec = e.vec_for_word("shirt")
    word = e.find_closed_word_for_vec(vec)
   # print vec
    print word
   # e.load_glove_gensim()
