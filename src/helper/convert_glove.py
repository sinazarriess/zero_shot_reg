from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/mnt/Data/zero_shot_reg/gloVe/glove.6B.300d.txt')
tmp_file = get_tmpfile("/mnt/Data/zero_shot_reg/gloVe/test_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
#glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'], topn=1)  ## only for analogies
print model.similar_by_vector(model['horse'], topn=3)
#similar_by_vector ??

print model['horse']
