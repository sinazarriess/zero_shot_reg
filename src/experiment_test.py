from __future__ import absolute_import, division, print_function, unicode_literals
#from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import tensorflow as tf
import json
import numpy as np
import scipy.io
import sys, os, shutil
import random
import timeit
import collections
import heapq
import io

max_epochs      = 100
num_runs        = 3
minibatch_size  = 50

#results_data_dir   = '/media/data/sinaza/034-iconic-gesture/results-exp'
results_data_dir = '/media/compute/vol/dsg/lilian/testrun/results'
#raw_input_data_dir = '.'
#raw_input_data_dir = '/media/dsgserve1_shkbox/036_object_parts'
min_token_freq = 3
layer_size = 512
method = 'inject' # 'merge'
dataset = 'refcoco_refrnn_compositional'


##################################################################################################################################
def generate_sequence_beamsearch(predictions_function, beam_width=3, clip_len=20):
    prev_beam = Beam(beam_width)
    prev_beam.add(np.array(1.0, 'float64'), False, [ edge_index ])
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
                if next_index == unknown_index: #skip unknown tokens
                    pass
                elif next_index == edge_index: #if next word is the end token then mark prefix as complete and leave out the end token
                    curr_beam.add(prefix_prob*next_prob, True, prefix)
                else: #if next word is a non-end token then mark prefix as incomplete
                    curr_beam.add(prefix_prob*next_prob, False, prefix+[next_index])

        (best_prob, best_complete, best_prefix) = max(curr_beam)
        if best_complete == True or len(best_prefix)-1 == clip_len: #if the length of the most probable prefix exceeds the clip length (ignoring the start token) then return it as is
            return ' '.join(index_to_token[index] for index in best_prefix[1:]) #return best sentence without the start token

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

##################################################################################################################################


if __name__ == '__main__':


    #with io.open(raw_input_data_dir+'/PreProcOut/refcoco_refrnn.json', 'r', encoding='utf-8') as captions_f:
    #with io.open(raw_input_data_dir+'/PreProcOut/refcoco_refrnn_compositionalspl.json', 'r', encoding='utf-8') as captions_f:
    with io.open('/media/compute/vol/dsg/lilian/testrun/refcoco_refrnn.json', 'r', encoding='utf-8') as captions_f:
        captions_data = json.load(captions_f)['images']
    #features = scipy.io.loadmat(raw_input_data_dir+'/visual_genome_vgg19_feats_small.mat')['feats'].T #image features matrix are transposed
    #features = scipy.io.loadmat(raw_input_data_dir+'/ExtrFeatsOut/refcoco_vgg19_rnnpreproc.mat')['feats'].T #image features matrix are transposed
    features = scipy.io.loadmat('/media/compute/vol/dsg/lilian/testrun/refcoco_vgg19_rnnpreproc.mat')['feats'].T

    print(type(features))
    print(features.shape)
    print("vgg_mat[1][0]", features[1][0])#0.0729042887688

    raw_dataset = {
                'train': { 'filenames': list(), 'images': list(), 'captions': list() },
                'val':   { 'filenames': list(), 'images': list(), 'captions': list() },
                'test':  { 'filenames': list(), 'images': list(), 'captions': list() },
            }

    for (image_id, (caption_data, image)) in enumerate(zip(captions_data, features)):
        #assert caption_data['sentences'][0]['imgid'] == image_id

        split = caption_data['split']
        if split == 'restval':
            continue
        filename = caption_data['filename']
        caption_group = [ caption['tokens'] for caption in caption_data['sentences'] ]
        image = image/np.linalg.norm(image)

        raw_dataset[split]['filenames'].append(filename)
        raw_dataset[split]['images'].append(image)
        raw_dataset[split]['captions'].append(caption_group)

    print('raw data set',len(raw_dataset['train']['captions']))
    ################################################################
    #for min_token_freq in [ 3, 4, 5 ]:
    all_tokens = (token for caption_group in raw_dataset['train']['captions'] for caption in caption_group for token in caption)
    token_freqs = collections.Counter(all_tokens)
    vocab = sorted(token_freqs.keys(), key=lambda token:(-token_freqs[token], token))
    while token_freqs[vocab[-1]] < min_token_freq:
        vocab.pop()

    vocab_size = len(vocab) + 2 # + edge and unknown tokens
    print('vocab:', vocab_size)

    token_to_index = { token: i+2 for (i, token) in enumerate(vocab) }
    index_to_token = { i+2: token for (i, token) in enumerate(vocab) }
    edge_index = 0
    unknown_index = 1


   # print(raw_dataset['train']['captions'][0]) # output: [[u'hidden', u'chocolate', u'donut'], [u'space', u'right', u'above', u'game']]
   # print(raw_dataset['train']['captions'][111])  # output : [[u'groom'], [u'groom'], [u'man']]

