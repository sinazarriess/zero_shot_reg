from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from collections import defaultdict
import numpy as np
import time

glove_files_path = "/mnt/Data/zero_shot_reg/gloVe/"

class Embeddings:

    def __init__(self, modelpath, use_only_names):
        self.model_initialized = False
        self.global_model_initialized = False
        self.modelpath = modelpath
        self.use_names_only = use_only_names

    def init_reduced_embeddings(self):
        if not self.model_initialized:
            if self.use_names_only:
                filepath = self.modelpath + "tmp_word2vec_only_names.txt"
            else:
                filepath = self.modelpath + "tmp_word2vec.txt"
            self.w2v_file = get_tmpfile(filepath)
        self.model = KeyedVectors.load_word2vec_format(self.w2v_file)
        self.model_initialized = True
        return self.model

    def test(self):
        print self.model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
        # print model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'], topn=1)  ## only for analogies
        print self.model.similar_by_vector(self.model['horse'], topn=3)

    def get_global_model(self):
        if not self.global_model_initialized:
            global_tmp = get_tmpfile(glove_files_path + "word2vec.txt")
            self.global_model = KeyedVectors.load_word2vec_format(global_tmp)
            #print self.global_model.similar_by_vector(self.global_model['horse'], topn=3)
            self.global_model_initialized = True
        return self.global_model

    def embeddings_for_vocab(self):
        with open(self.modelpath + 'vocab_list.txt', 'r') as f:
            self.model_vocab = f.read().splitlines()
        with open(self.modelpath + 'additional_vocab.txt', 'r') as f:
            self.additional_words = f.read().splitlines()

        self.word_that_are_names = list()
        with open("./noun_list_long.txt", 'r') as f:
            for row in f.readlines():
                self.word_that_are_names.append(row.strip())

        complete_vocab = self.model_vocab + self.additional_words
        print complete_vocab
        print len(complete_vocab)
        print self.additional_words

        self.embeddings  = defaultdict()
        for word in complete_vocab:
            if self.use_names_only:
                if word in self.word_that_are_names:
                    vector = self.get_global_vector(word)
                else:
                    vector = []
            else:
                vector = self.get_global_vector(word)
            if not len(vector) == 0:
                self.embeddings[word] = vector

    def write_new_glove_file(self):
        if self.use_names_only:
            filepath = self.modelpath + "reduced_vocab_glove_only_names.txt"
        else:
            filepath = self.modelpath + "reduced_vocab_glove.txt"
        with open(filepath, "w") as f:
            for vocab_word in self.embeddings:
                f.write(vocab_word + " ")
                for dim in self.embeddings[vocab_word]:
                    f.write(str(dim) + " ")
                f.write("\n")

    def convert_glove_to_w2v(self):
        if self.use_names_only:
            filepath = self.modelpath + "reduced_vocab_glove_only_names.txt"
        else:
            filepath = self.modelpath + "reduced_vocab_glove.txt"
        glove_file = datapath(filepath)
        if self.use_names_only:
            filepath2 = self.modelpath + "tmp_word2vec_only_names.txt"
        else:
            filepath2 = self.modelpath + "tmp_word2vec.txt"
        self.w2v_file = get_tmpfile(filepath2)
        # call glove2word2vec script
        # default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
        glove2word2vec(glove_file, self.w2v_file)
        #self.model_initialized = True

    def generate_reduced_w2v_file(self):
        self.get_global_model()
        self.embeddings_for_vocab()
        self.write_new_glove_file()
        self.convert_glove_to_w2v()

    def get_global_vector(self, word):
        if self.global_model_initialized:
            if word in self.global_model.vocab:
                return self.global_model[word]
        else:
            print "global model not initialized (1)"
        return []

    def get_vector_for_word(self, word):
        if self.model_initialized:
            if word in self.model.vocab:
                return self.model[word]
        elif self.global_model_initialized:
            if word in self.global_model.vocab:
                return self.global_model[word]
        else:
            print "model not initialized (2) "
        return []

    def get_words_for_vector(self, vec, n):
        if self.model_initialized:
            return self.model.similar_by_vector(vec, topn=n)
        elif self.global_model_initialized:                             #todo clean up class
            return self.global_model.similar_by_vector(vec, topn=n)
        else:
            print "model not initialized (3)"
        return ""

    def get_mean_vec(self, vecs):
        self.get_words_for_vector()

    def vectors2embedding_weighted(self, word_probs, word_vectors):
        new_vec = np.sum([word_vectors[x].astype(float) * word_probs[x] for x in range(len(word_probs))], axis=0)
        vecs = np.array(new_vec)
        return new_vec

    def words2embedding_weighted(self, predictions, word_probs):
        vectors = list()
        probs = list()
        valid_words_counter = 0
        for word in predictions:
            if not (word == 'EDGE' or word == 'UNKNOWN'):
                vec = self.get_vector_for_word(word)
                if len(vec) > 0:
                    vectors.append(vec)
                    probs.append(predictions.index(word))
                    valid_words_counter += 1
        if valid_words_counter > 0:
            new_vec = np.sum([vectors[x].astype(float) * word_probs[x] for x in range(valid_words_counter)], axis=0)
            vecs = np.array(new_vec)
            return new_vec
        return None

    # def words2embedding_weighted(self, word_predictions, word_probs, word_vectors):
    #     vecs = []
    #     for i in range(len(word_predictions)):
    #         wlist = word_predictions[i]
    #         plist = word_probs[i]
    #         new_vec = np.sum([word_vectors[w].astype(float) * plist[x] for x, w in enumerate(wlist)], axis=0)
    #         vecs.append(new_vec)
    #     vecs = np.array(vecs)
    #     # print "Embedding matrix",vecs.shape
    #     return vecs

if __name__ == '__main__':

    use_only_nouns = True
    embeddings = Embeddings('/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats_horse/', use_only_nouns)
    ## generate custom embeddings for a model (with reduced vocabulary)
    embeddings.generate_reduced_w2v_file()

    start = time.time()

    ## initialize custom embeddings
    word_model = embeddings.init_reduced_embeddings()
    #word_model = embeddings.get_global_model()
    end = time.time()
    print (end - start)

    start = time.time()
    ## how to use
    horse_vec = embeddings.get_vector_for_word('horse')
    #print embeddings.get_words_for_vector(horse_vec, 1)

    end = time.time()
    print (end - start)
    print "Vocabs: ", len(word_model.vocab)

    # a = embeddings.get_vector_for_word('car')
    # print a
    # b = embeddings.get_vector_for_word('area')
    # c = embeddings.get_vector_for_word('thing')
    # d = embeddings.get_vector_for_word('truck')
    #
    # probs = [0.05730285, 0.057440747, 0.07579447, 0.08411062]
    # words = ['car', 'area', 'thing', 'truck']
    # print embeddings.get_words_for_vector(embeddings.words2embedding_weighted(words, probs), 10)

    #'black', 'bus', 'train', 'big'
    #'bus', 'car', 'train'
    #'truck', 'car', 'train'