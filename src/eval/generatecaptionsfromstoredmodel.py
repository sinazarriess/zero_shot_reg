import lstm.beam
import lstm.lstm
import tensorflow as tf
import json
import numpy as np
import lstm.params as p
from collections import defaultdict, OrderedDict
import time
import operator

model_dir = './model/with_reduced_vocab/'
edge_index = 0
unknown_index = 1

def final_prediction(self, tensor):
    return tensor[:, -1]

class RefsGenerator:
    def __init__(self):
        self.filenames = np.genfromtxt(model_dir + 'raw_dataset_filenames.txt', delimiter=',', dtype=None, encoding=None)
        with open(model_dir + 'index2token.json', "r") as f:
            index2token = json.load(f)
        self.new_index2token= {int(key): value for key, value in index2token.iteritems()}
        # keep unknown tokens
        self.new_index2token[unknown_index] = "UNKNOWN"
        self.new_index2token[edge_index] = "EDGE"
        self.images = np.genfromtxt(model_dir + 'raw_dataset.txt', delimiter=',', dtype=None)
        self.oids = list()
        for (i, item) in enumerate(self.filenames):
            self.oids.append(str(item).split("_")[1])
        self.candidates4eval = defaultdict()

    def initialize_model(self):
        self.imported_meta = tf.train.import_meta_graph(model_dir + 'inject_refcoco_refrnn_compositional_3_512_1/model.meta')
        graph = tf.get_default_graph()
        self.seq_in = graph.get_tensor_by_name('seq_in:0')
        self.seq_len = graph.get_tensor_by_name('seq_len:0')
        self.image = graph.get_tensor_by_name('image:0')
        predictions = graph.get_tensor_by_name('softmax/prediction:0')
        self.last_prediction = predictions[:, -1]

    def generate_refs_greedily(self):
        with tf.Session() as sess:
            average_utterance_length = 0
            utterance_counter = 0
            self.imported_meta.restore(sess, tf.train.latest_checkpoint(model_dir + 'inject_refcoco_refrnn_compositional_3_512_1'))
            self.captions_greedy = list()

            for (i, image_input) in enumerate(self.images):
                predictions_function = (lambda prefixes: sess.run(self.last_prediction, feed_dict={
                    self.seq_in: prefixes,
                    self.seq_len: [len(p) for p in prefixes],
                    self.image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
                }))
                gen_prefix = list()
                gen_prefix.append([0])  # edge index
                isComplete = False
                while not isComplete:
                    indexes_distributions = predictions_function(gen_prefix)
                    candidate_dict = defaultdict()

             #       max_value = 0
             #       max_index = -1
                    for (next_index, next_prob) in enumerate(indexes_distributions[0]):
                        candidate_dict[next_index] = next_prob
             #           if next_prob > max_value:
             #               max_value = next_prob
             #               max_index = next_index

                    # sort distributions to get highest probabilities --> (index,prob)
                   # sorted_distribution = sorted(candidate_dict.items(), key = operator.itemgetter(1))
                    sorted_distribution = OrderedDict(sorted(candidate_dict.items(), key=lambda t: t[1]))
                    max_index = sorted_distribution.items()[-1][0]

                    if max_index == edge_index:
                        isComplete = True
                        average_utterance_length += len(gen_prefix[0][1:])
                        utterance_counter += 1
                        self.captions_greedy.append(' '.join(self.new_index2token[index] for index in gen_prefix[0][1:]))
                    else:
                        gen_prefix[0].append(max_index)
                        if max_index == unknown_index:
                            max_candidates = [self.new_index2token[tuple[0]] for tuple in sorted_distribution.items()[-5:]]
                            #print "max_candidates for image ", self.oids[i], " are (falling probability): ", max_candidates
                            self.candidates4eval[self.oids[i]] = max_candidates
            print "Average utterance length greedy: ", average_utterance_length/float(utterance_counter)

    def generate_refs_with_beam(self):
        with tf.Session() as sess:
            self.imported_meta.restore(sess, tf.train.latest_checkpoint(
                model_dir + 'inject_refcoco_refrnn_compositional_3_512_1'))

            self.captions = list()
            searcher = lstm.beam.Search(self.new_index2token, True)
            for (i, image_input) in enumerate(self.images):
                caption = searcher.generate_sequence_beamsearch(lambda prefixes: sess.run(self.last_prediction, feed_dict={
                      self.seq_in: prefixes,
                      self.seq_len: [len(p) for p in prefixes],
                      self.image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
                }))
                self.captions.append(caption)
            print "Average utterance length beam: ", searcher.get_average_ref_len()


    def save_refs(self, captionslist, name):

        dict4eval = defaultdict(list)
        for (idx, pair) in enumerate(zip(self.oids, captionslist)):  # captions
            dict4eval[pair[0]] = [pair[1]]

        with open(model_dir + name + '.json', 'w') as f:
            json.dump(dict4eval, f)

        with open(model_dir + 'highest_prob_candidates.json', 'w') as f:
            json.dump(self.candidates4eval, f)


if __name__ == "__main__":
    gen = RefsGenerator()
    gen.initialize_model()
    start = time.time()
    gen.generate_refs_greedily()
    end = time.time()
    print (end - start)
    gen.save_refs(gen.captions_greedy, 'restoredmodel_refs_greedy')
    # start = time.time()
    # gen.generate_refs_with_beam()
    # end = time.time()
    # print (end - start)

    # gen.save_refs(gen.captions, 'restoredmodel_refs_beam')









