import lstm.beam
import lstm.lstm
import tensorflow as tf
import json
import numpy as np
import lstm.params as p
from collections import defaultdict


model_dir = './model/with_unknown/'
edge_index = 0

def final_prediction(self, tensor):
    return tensor[:, -1]

if __name__ == "__main__":

    filenames = np.genfromtxt(model_dir + 'raw_dataset_filenames.txt', delimiter=',', dtype=None, encoding=None)

    with open(model_dir + 'index2token.json', "r") as f:
        index2token = json.load(f)
    new_index2token= {int(key): value for key, value in index2token.iteritems()}

    # keep unknown tokens
    new_index2token[1] = "UNKNOWN"

    images = np.genfromtxt(model_dir + 'raw_dataset.txt', delimiter=',', dtype=None)

    imported_meta = tf.train.import_meta_graph(model_dir + 'inject_refcoco_refrnn_compositional_3_512_1/model.meta')
    graph = tf.get_default_graph()
    seq_in = graph.get_tensor_by_name('seq_in:0')
    seq_len = graph.get_tensor_by_name('seq_len:0')
    image = graph.get_tensor_by_name('image:0')
    predictions = graph.get_tensor_by_name('softmax/prediction:0')
    last_prediction = predictions[:, -1]

    with tf.Session() as sess:

        imported_meta.restore(sess, tf.train.latest_checkpoint(model_dir + 'inject_refcoco_refrnn_compositional_3_512_1'))

        oids = list()
        captions = list()
        captions_greedy = list()
      #  searcher = lstm.beam.Search(new_index2token, True)
      #  for (i, image_input) in enumerate(images):
      #      caption = searcher.generate_sequence_beamsearch(lambda prefixes: sess.run(last_prediction, feed_dict={
      #          seq_in: prefixes,
      #          seq_len: [len(p) for p in prefixes],
      #          image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
      #      }))
      #      captions.append(caption)

        for (i, image_input) in enumerate(images):
            predictions_function = (lambda prefix: sess.run(last_prediction, feed_dict={
                seq_in: [prefix],
                seq_len: [len(prefix)],
                image: image_input.reshape([1, -1]).repeat(1, axis=0)
            }))
            gen_prefix = list()
            isComplete = False
            #while not isComplete:
            indexes_distributions = predictions_function(gen_prefix)

         #   for (next_index, next_prob) in enumerate(indexes_distribution):
            #if next_word == edge_index:
            #    isComplete = True
            print gen_prefix

        for (i, item) in enumerate(filenames):
            oids.append(str(item).split("_")[1])

        dict4eval = defaultdict(list)
        for (idx, pair) in enumerate(zip(oids, captions)):
            dict4eval[pair[0]] = [pair[1]]

        with open(model_dir + 'restoredmodel_captions_with_unknown.json', 'w') as f:
            json.dump(dict4eval, f)

