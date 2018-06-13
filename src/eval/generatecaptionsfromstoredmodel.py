import lstm.beam
import lstm.lstm
import tensorflow as tf
import json
import numpy as np
import lstm.params as p
from collections import defaultdict

def final_prediction(self, tensor):
    return tensor[:, -1]

if __name__ == "__main__":

    filenames = np.genfromtxt('../eval/cached_data/raw_dataset_filenames.txt', delimiter=',', dtype=None, encoding=None)

    with open('../eval/cached_data/index2token.json', "r") as f:
        index2token = json.load(f)
    new_index2token= {int(key): value for key, value in index2token.iteritems()}

    # keep unknown tokens
    new_index2token[1] = "UNKNOWN"

    images = np.genfromtxt('../eval/cached_data/raw_dataset.txt', delimiter=',', dtype=None)

    imported_meta = tf.train.import_meta_graph('../eval/model/without_chicken/inject_refcoco_refrnn_compositional_3_512_1/model.meta')
    graph = tf.get_default_graph()
    seq_in = graph.get_tensor_by_name('seq_in:0')
    seq_len = graph.get_tensor_by_name('seq_len:0')
    image = graph.get_tensor_by_name('image:0')
    predictions = graph.get_tensor_by_name('softmax/prediction:0')
    last_prediction = predictions[:, -1]

    with tf.Session() as sess:

        imported_meta.restore(sess, tf.train.latest_checkpoint('../eval/model/without_chicken/inject_refcoco_refrnn_compositional_3_512_1'))

        oids = list()
        captions = list()
        searcher = lstm.beam.Search(new_index2token, True)
        for (i, image_input) in enumerate(images):
            caption = searcher.generate_sequence_beamsearch(lambda prefixes: sess.run(last_prediction, feed_dict={
                seq_in: prefixes,
                seq_len: [len(p) for p in prefixes],
                image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
            }))
            captions.append(caption)
        for (i, item) in enumerate(filenames):
            oids.append(str(item).split("_")[1])

        dict4eval = defaultdict(list)
        for (idx, pair) in enumerate(zip(oids, captions)):
            dict4eval[pair[0]] = [pair[1]]

        with open('../eval/jsons/restoredmodel_captions_chicken.json', 'w') as f:
            json.dump(dict4eval, f)

