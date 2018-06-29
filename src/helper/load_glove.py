import pandas as pd
import csv
import numpy as np


class Embeddings:

    def __init__(self, glove_data_file = '/mnt/Data/zero_shot_reg/gloVe/glove.42B.300d_2.txt'):
        self.words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        self.words_matrix = self.words.as_matrix()

        # glove = np.loadtxt(glove_data_file, dtype='str', comments=None)
        # words = glove[:, 0]
        # vectors = glove[:, 1:].astype('float')

        print "loaded embeddings file"

    def vec_for_word(self, w):
        return self.words.loc[w].as_matrix()

    def find_closed_word_for_vec(self, v):
        diff = self.words_matrix - v
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        return self.words.iloc[i].name


if __name__ == "__main__":
    e = Embeddings()
    vec = e.vec_for_word("horse")
    word = e.find_closed_word_for_vec(vec)
    print vec
    print word

