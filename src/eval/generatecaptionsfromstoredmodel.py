import lstm.beam
import lstm.lstm
import tensorflow as tf
import json
import numpy as np
import lstm.params as p
from collections import defaultdict, OrderedDict
import time
import operator
import ast

## The purpose of this class is to seperate the application of the model (i.e. the generation of sequences) from the
## training code. It works with a model that is stored as a file, this allows to run the script on another computer than
## the training, if desired.
## Methods include a beam search and a greedy search.

edge_index = 0
unknown_index = 1

def final_prediction(self, tensor):
    return tensor[:, -1]

class RefsGenerator:
    def __init__(self, nmbr, modeldir):
        self.model_dir = modeldir
        self.number_of_candidates = nmbr
        self.filenames = np.genfromtxt(self.model_dir + 'raw_dataset_filenames.txt', delimiter=',', dtype=None, encoding=None)
        with open(self.model_dir + 'index2token.json', "r") as f:
            index2token = json.load(f)
        self.new_index2token= {int(key): value for key, value in index2token.iteritems()}
        # keep unknown tokens
        self.new_index2token[unknown_index] = "UNKNOWN"
        self.new_index2token[edge_index] = "EDGE"
        self.images = np.genfromtxt(self.model_dir + 'raw_dataset.txt', delimiter=',', dtype=None)
        self.oids = list()
        for (i, item) in enumerate(self.filenames):
            self.oids.append(str(item).split("_")[1])
        self.candidates4eval = defaultdict()
        self.alternatives_dict = defaultdict()
        self.accumulated_variance = 0

    ## read in the region IDs for which the model is supposed to generate expressions (costum split)
    def read_excluded_ids(self):
        with open(self.model_dir + 'refs_moved_to_test.json', 'r') as f:
            ids = f.readline()
            return ast.literal_eval(ids)

    ## read stored model from file
    def initialize_model(self):
        self.imported_meta = tf.train.import_meta_graph(self.model_dir + 'inject_refcoco_refrnn_compositional_3_512_1/model.meta')
        graph = tf.get_default_graph()
        self.seq_in = graph.get_tensor_by_name('seq_in:0')
        self.seq_len = graph.get_tensor_by_name('seq_len:0')
        self.image = graph.get_tensor_by_name('image:0')
        predictions = graph.get_tensor_by_name('softmax/prediction:0')
        self.last_prediction = predictions[:, -1]

    ## generate expressions for the test set with a greedy strategy
    def generate_refs_greedily(self, excluded_ids = []):
        self.excluded = excluded_ids
        with tf.Session() as sess:
            average_utterance_length = 0
            utterance_counter = 0
            self.imported_meta.restore(sess, tf.train.latest_checkpoint(self.model_dir + 'inject_refcoco_refrnn_compositional_3_512_1'))
            self.captions_greedy = list()

            for (i, image_input) in enumerate(self.images):
                if self.oids[i] in excluded_ids:
                    predictions_function = (lambda prefixes: sess.run(self.last_prediction, feed_dict={
                        self.seq_in: prefixes,
                        self.seq_len: [len(p) for p in prefixes],
                        self.image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
                    }))
                    gen_prefix = list()
                    gen_prefix.append([0])  # edge index
                    isComplete = False
                    temp_dict = defaultdict()
                    while not isComplete:
                        indexes_distributions = predictions_function(gen_prefix)
                        candidate_dict = defaultdict()

                        for (next_index, next_prob) in enumerate(indexes_distributions[0]):
                            candidate_dict[next_index] = next_prob

                        # probabilities ordered with ascending value
                        sorted_distribution = OrderedDict(sorted(candidate_dict.items(), key=lambda t: t[1]))
                        max_index = sorted_distribution.items()[-1][0]

                        best_candidates = [(self.new_index2token[tuple[0]], str(tuple[1])) for tuple in sorted_distribution.items()[-self.number_of_candidates:]]
                        probabilities_for_the_variance = [tuple[1] for tuple in sorted_distribution.items()[-10:]]

                        if max_index == edge_index:
                            isComplete = True
                            average_utterance_length += len(gen_prefix[0][1:])
                            utterance_counter += 1
                            self.captions_greedy.append(' '.join(self.new_index2token[index] for index in gen_prefix[0][1:]))

                            self.alternatives_dict[self.oids[i]] = temp_dict
                            self.accumulated_variance += np.var(probabilities_for_the_variance)
                        else:
                            # position in utterance
                            temp_dict[len(gen_prefix[0])] = best_candidates

                            gen_prefix[0].append(max_index)
                            if max_index == unknown_index:
                                #print "max_candidates for image ", self.oids[i], " are (falling probability): ", max_candidates
                                self.candidates4eval[self.oids[i]] = best_candidates  #might overwrite in very rare cases ('the UNKNOWN UNKNOWN')

            print "Average utterance length greedy: ", average_utterance_length/float(utterance_counter)

    ## generate expressions for the test set with a beam search algorithm
    def generate_refs_with_beam(self):
        with tf.Session() as sess:
            self.imported_meta.restore(sess, tf.train.latest_checkpoint(
                self.model_dir + 'inject_refcoco_refrnn_compositional_3_512_1'))

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


    ## store expressions in files
    def save_refs(self, captionslist, name):

        dict4eval = defaultdict(list)

        test = list()
        for x in self.oids:
            if x in self.excluded:
                test.append(x)
        for (idx, pair) in enumerate(zip(test, captionslist)):  # captions
            dict4eval[pair[0]] = [pair[1]]

        with open(self.model_dir + name + '.json', 'w') as f:
            json.dump(dict4eval, f)

        # was used for unknown analysis
        with open(self.model_dir + 'highest_prob_candidates_' + str(self.number_of_candidates) + '.json', 'w') as f:
            json.dump(self.candidates4eval, f)

        # basis for zero-shot application!
        with open(self.model_dir + 'all_highest_probs_' + str(self.number_of_candidates) + '.json', 'w') as f:
            json.dump(self.alternatives_dict, f)

if __name__ == "__main__":

    gen = RefsGenerator(10, './model/with_reduced_cats_zebra/')
    gen.initialize_model()
    start = time.time()
    ids = gen.read_excluded_ids()
    gen.generate_refs_greedily(ids)
    end = time.time()
    print (end - start)
    gen.save_refs(gen.captions_greedy, 'restoredmodel_refs_greedy')

    print "mean variance: ", gen.accumulated_variance / float(len(gen.captions_greedy))
    # start = time.time()   #for performance analysis
    # gen.generate_refs_with_beam()
    # end = time.time()
    # print (end - start)

    # gen.save_refs(gen.captions, 'restoredmodel_refs_beam')









