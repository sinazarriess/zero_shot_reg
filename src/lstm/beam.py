import numpy as np
import heapq
import params

## Code extracted from the original experiment.py by Tanti et al., containing all code for the generation of a
# sequence with the LSTM
# https://github.com/mtanti/rnn-role/blob/master/experiment.py

class Search:

    def __init__(self, index_to_token, keep_unknowntoken=False):
        self.average_utt_len = 0
        self.utt_count = 0
        self.index_to_token = index_to_token
        self.keep_unknown = keep_unknowntoken
        if self.keep_unknown:
            if params.unknown_index not in index_to_token:
                raise KeyError('unknown token not in dictionary provided')

    def get_average_ref_len(self):
        return self.average_utt_len/float(self.utt_count)

    def generate_sequence_beamsearch(self, predictions_function, beam_width=3, clip_len=20):
        prev_beam = Beam(beam_width)
        prev_beam.add(np.array(1.0, 'float64'), False, [ params.edge_index ])
        while True:
            curr_beam = Beam(beam_width)

            #Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
            prefix_batch = list()
            prob_batch = list()
            for (prefix_prob, complete, prefix) in prev_beam:
                if complete == True:
                    curr_beam.add(prefix_prob, True, prefix)
                else:
                    prefix_batch.append(prefix)
                    prob_batch.append(prefix_prob)

            #Get probability of each possible next word for each incomplete prefix.
            indexes_distributions = predictions_function(prefix_batch)

            #Add next words
            for (prefix_prob, prefix, indexes_distribution) in zip(prob_batch, prefix_batch, indexes_distributions):
                for (next_index, next_prob) in enumerate(indexes_distribution):
                    if self.keep_unknown:
                        if next_index == params.edge_index:  # if next word is the end token then mark prefix as complete and leave out the end token
                            curr_beam.add(prefix_prob * next_prob, True, prefix)
                        else:  # if next word is a non-end token then mark prefix as incomplete
                            curr_beam.add(prefix_prob * next_prob, False, prefix + [next_index])
                    else:
                        if next_index == params.unknown_index: #skip unknown tokens
                            pass
                        elif next_index == params.edge_index: #if next word is the end token then mark prefix as complete and leave out the end token
                            curr_beam.add(prefix_prob*next_prob, True, prefix)
                        else: #if next word is a non-end token then mark prefix as incomplete
                            curr_beam.add(prefix_prob*next_prob, False, prefix+[next_index])

            (best_prob, best_complete, best_prefix) = max(curr_beam)
            if best_complete == True or len(best_prefix)-1 == clip_len: #if the length of the most probable prefix exceeds the clip length (ignoring the start token) then return it as is
                self.average_utt_len += len(best_prefix[1:])
                self.utt_count += 1
                return ' '.join(self.index_to_token[index] for index in best_prefix[1:]) #return best sentence without the start token

            prev_beam = curr_beam

##################################################################################################################################
class Beam(object):
#For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
#This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)
