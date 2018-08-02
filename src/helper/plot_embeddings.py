import numpy as np
import matplotlib.pyplot as plt
import gensim
from sklearn.manifold import TSNE
from helper.word_embeddings import Embeddings

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word, 25)

    arr = np.append(arr, np.array([model[word]]), axis = 0)

    # add the vector for each of the closest words to the array
    for i, wrd_score in enumerate(close_words):
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis = 0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.scatter(x_coords, y_coords, c = "blue")

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

def display_closestwords_tsnescatterplot_compare(model, complete_model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]
    words_in_small_embed = [word]

    # get close words
    close_words = complete_model.similar_by_word(word, 25)
    for w in close_words:
        if w[0] in model.vocab:
            words_in_small_embed.append(w[0])
            print w [0],

    arr = np.append(arr, np.array([model[word]]), axis = 0)

    # add the vector for each of the closest words to the array
    for i, wrd_score in enumerate(close_words):
        wrd_vector = complete_model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis = 0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    colors = []

    for i, y in enumerate(Y):
        if word_labels[i] in words_in_small_embed:
            colors.append('red')
        else:
            colors.append('blue')

    plt.scatter(x_coords, y_coords, c = colors)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


if __name__ == "__main__":
    embeddings = Embeddings('/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_74/', True)
    global_model = embeddings.get_global_model()
    word_model = embeddings.init_reduced_embeddings()
    #display_closestwords_tsnescatterplot(global_model, 'mouse')
    #print global_model.similar_by_word("mouse", 3)
    print  global_model.similar_by_word('mouse', 3)

    embeddings = Embeddings('/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_82/', True)
    word_model = embeddings.init_reduced_embeddings()
    print  global_model.similar_by_word('refridgerator', 3)

    embeddings = Embeddings('/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_42/', True)
    word_model = embeddings.init_reduced_embeddings()
    print  word_model.similar_by_word('surfboard', 3)