from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from collections import defaultdict
import numpy as np
import time

## Class to encapsulate the access to word embeddings.
## Includes methods to generate a costum embedding space and for converting GloVe files to word2vec files.

glove_files_path = "/mnt/Data/zero_shot_reg/gloVe/"

class Embeddings:

    def __init__(self, modelpath, use_only_names):
        self.model_initialized = False
        self.global_model_initialized = False
        self.modelpath = modelpath
        self.use_names_only = use_only_names

    # load embeddings
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

    # load whole original GloVe space
    def get_global_model(self):
        if not self.global_model_initialized:
            global_tmp = get_tmpfile(glove_files_path + "word2vec.txt")
            self.global_model = KeyedVectors.load_word2vec_format(global_tmp)
            #print self.global_model.similar_by_vector(self.global_model['horse'], topn=3)
            self.global_model_initialized = True
        return self.global_model

    # generate space that contains only of words from a given vocabulary (+ the zero-shot category)
    def embeddings_for_vocab(self):
        with open(self.modelpath + 'vocab_list.txt', 'r') as f:
            self.model_vocab = f.read().splitlines()
        # the category's name itself has to be added separately since it is not necessarily in the training vocabulary
        with open(self.modelpath + 'additional_vocab.txt', 'r') as f:
            self.additional_words = [x.strip() for x in f.read().splitlines()]

        self.word_that_are_names = list()
        with open("./noun_list_long.txt", 'r') as f:
            for row in f.readlines():
                self.word_that_are_names.append(row.strip())

        self.embeddings  = defaultdict()
        for word in self.model_vocab:
            if self.use_names_only:
                if word in self.word_that_are_names:
                    vector = self.get_global_vector(word)
                else:
                    vector = []
            else:
                vector = self.get_global_vector(word)
            if not len(vector) == 0:
                self.embeddings[word] = vector

        for word in self.additional_words:
            if word not in self.embeddings.keys(): # sometimes it is already known
                vector = self.get_global_vector(word)
                if not len(vector) == 0:
                    self.embeddings[word] = vector

    # store space in a file
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

    # convert a glove file to a word2vec formatted file for easy gensim use
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

    # calls all necessary methods for the generation of a costum space
    def generate_reduced_w2v_file(self):
        self.get_global_model()
        self.embeddings_for_vocab()
        self.write_new_glove_file()
        self.convert_glove_to_w2v()

    # interface for zero-shot code: returns vector for a word from the global space
    def get_global_vector(self, word):
        if self.global_model_initialized:
            if word in self.global_model.vocab:
                return self.global_model[word]
        else:
            print "global model not initialized (1)"
        return []

    # interface for zero-shot code: returns vector for a word from the given space
    def get_vector_for_word(self, word, from_small_model):
        if from_small_model:
            if self.model_initialized:
                if word in self.model.vocab:
                 return self.model[word]
        else:
            if self.global_model_initialized:
                if word in self.global_model.vocab:
                    return self.global_model[word]
            else:
                print "not initialized"
        return []

    # interface for zero-shot code: returns most similar words for a given vector from given space
    def get_words_for_vector(self, vec, n, from_small_model):

        if from_small_model:
            if self.model_initialized:
                return self.model.similar_by_vector(vec, topn=n)
        else:
            if self.global_model_initialized:
                return self.global_model.similar_by_vector(vec, topn=n)
            else:
                print "not initialized"
        return ""


    #def get_mean_vec(self, vecs):
     #   self.get_words_for_vector()

    def vectors2embedding_weighted(self, word_probs, word_vectors):
        new_vec = np.sum([word_vectors[x].astype(float) * word_probs[x] for x in range(len(word_probs))], axis=0)
        vecs = np.array(new_vec)
        return new_vec

    # implements ConSE method: weighted sum of word vectors, weighted by probability
    def words2embedding_weighted(self, predictions, word_probs, from_small_model):
        vectors = list()
        probs = list()
        valid_words_counter = 0
        for word in predictions:
            if not (word == 'EDGE' or word == 'UNKNOWN'):
                vec = self.get_vector_for_word(word, from_small_model)
                if len(vec) > 0:
                    vectors.append(vec)
                    probs.append(predictions.index(word))
                    valid_words_counter += 1
        if valid_words_counter > 0:
            new_vec = np.sum([vectors[x].astype(float) * word_probs[x] for x in range(valid_words_counter)], axis=0)
            vecs = np.array(new_vec)
            return new_vec
        print 'no valid predictions for combination found'
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

    use_only_nouns = False
    embeddings = Embeddings('/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_all/', use_only_nouns)

    ## generate custom embeddings for a model (with reduced vocabulary)
    #embeddings.generate_reduced_w2v_file()

    start = time.time()

    ## initialize custom embeddings
    word_model = embeddings.init_reduced_embeddings()
    #word_model = embeddings.get_global_model()
    end = time.time()
    print (end - start)

    start = time.time()
    ## how to use
    horse_vec = embeddings.get_vector_for_word('white',1)
    print horse_vec
    #print embeddings.get_words_for_vector(horse_vec, 1)

    end = time.time()
    print (end - start)
    print "Vocabs: ", len(word_model.vocab)
