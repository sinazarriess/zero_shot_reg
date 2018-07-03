from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from collections import defaultdict

glove_files_path = "/mnt/Data/zero_shot_reg/gloVe/"

class Embeddings:

    def __init__(self, modelpath):
        self.model_initialized = False
        self.global_model_initialized = False
        self.modelpath = modelpath
      #  self.additional_words = additional_words  # words that were removed during training!

    def init_reduced_embeddings(self):
        if not self.model_initialized:
            self.w2v_file = get_tmpfile(self.modelpath + "tmp_word2vec.txt")
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
            print self.global_model.similar_by_vector(self.global_model['horse'], topn=3)
            self.global_model_initialized = True
        return self.global_model

    def embeddings_for_vocab(self):
        with open(self.modelpath + 'vocab_list.txt', 'r') as f:
            self.model_vocab = f.read().splitlines()
        with open(self.modelpath + 'additional_vocab.txt', 'r') as f:
            self.additional_words = f.read().splitlines()

        complete_vocab = self.model_vocab + self.additional_words
        print complete_vocab
        print len(complete_vocab)
        print self.additional_words

        self.embeddings  = defaultdict()
        for word in self.model_vocab:
            vector = self.get_global_vector(word)
            if not len(vector) == 0:
                self.embeddings[word] = vector
            else:
                print word

    def write_new_glove_file(self):
        with open(self.modelpath + "reduced_vocab_glove.txt", "w") as f:
            for vocab_word in self.embeddings:
                f.write(vocab_word + " ")
                for dim in self.embeddings[vocab_word]:
                    f.write(str(dim) + " ")
                f.write("\n")

    def convert_glove_to_w2v(self):
        glove_file = datapath(self.modelpath + 'reduced_vocab_glove.txt')
        self.w2v_file = get_tmpfile(self.modelpath + 'tmp_word2vec.txt')
        # call glove2word2vec script
        # default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
        glove2word2vec(glove_file, self.w2v_file)
        self.model_initialized = True

    def generate_reduced_wv2_file(self):
        self.get_global_model()
        self.embeddings_for_vocab()
        self.write_new_glove_file()
        self.convert_glove_to_w2v()

    def get_global_vector(self, word):
        if self.global_model_initialized:
            if word in self.global_model.vocab:
                return self.global_model[word]
        else:
            print "model not initialized"
        return []

    def get_vector_for_word(self, word):
        if self.model_initialized:
            if word in self.model.vocab:
                return self.model[word]
        else:
            print "model not initialized"
        return []

    def get_words_for_vector(self, vec, n):
        if self.model_initialized:
            return self.model.similar_by_vector(vec, topn=n)
        else:
            print "model not initialized"
        return ""

    def get_mean_vec(self, vecs):
        self.get_words_for_vector()

    # def words2embedding_weighted(word_predictions, word_probs, word_vectors):
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
    embeddings = Embeddings('/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats/')
    ## generate custom embeddings for a model (with reduced vocabulary)
    # embeddings.generate_reduced_wv2_file()

    ## initialize custom embeddings
    word_model = embeddings.init_reduced_embeddings()

    ## how to use
    horse_vec = embeddings.get_vector_for_word('horse')
    print embeddings.get_words_for_vector(horse_vec, 1)

    a = embeddings.get_vector_for_word('cow')
    b = embeddings.get_vector_for_word('pony')
    c = embeddings.get_vector_for_word('man')
    print embeddings.get_words_for_vector( a+b+c, 5)