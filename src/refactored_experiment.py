from __future__ import absolute_import, division, unicode_literals, print_function
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import json
import numpy as np
import sys, os, shutil
import random
import timeit
import collections
import heapq
import io

############ params ####################################################
max_epochs      = 100
num_runs        = 3
minibatch_size  = 50
results_data_dir = '/media/compute/vol/dsg/lilian/testrun_after_refactoring/results'
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

    ############ load and prepare image features ###########################

    # Image Features  --> numpy npz file includes one key/array, arr_0
    extracted_features = np.load("../data/refcoco/mscoco_vgg19_refcoco.npz")['arr_0']
    #print "type", type (extracted_features)
    #print "extracted_features.shape ",extracted_features.shape # (49865, 4106)
    #print "vgg_mat[1][0] ", extracted_features[1][0] # 1.0
    #extracted_features = extracted_features[:,3:]
    #print "extracted_features.shape ",extracted_features.shape # (49865, 4103)
    #print "vgg_mat[1][0] ", extracted_features[1][0] # 0.0

    test_list =[]
    test_count = 0

    img_counter = 0
    sentence_counter = 0
    selected_img_features = [] #alias img_mat

    ########### load and prepare referring expressions dataset ##############
    refcoco_data = pd.read_json("../data/refcoco/refcoco_refdf.json.gz", orient="split", compression="gzip")
    with open("../data/refcoco/refcoco_splits.json") as f:
        splits = json.load(f)
    #  TODO  generate own split for zero-shot learning
    splitmap = {'val':'val','train':'train','testA':'test','testB':'test'}
    # for every group in split --> for every entry --> make entry in new dict
    # file2split just translates testA and testB to "test"?
    new_split_dict = {val:splitmap[key] for key in splits for val in splits[key]}


    # dict of objectids and ref exps
    obj2phrases = defaultdict(list)
    # dict of objectids and split (train,test or val)
    obj2split = {}
    split2obj = {'train':[],'test':[]}

    # iterate over json "entries"
    for index, row in refcoco_data.iterrows():
        # id is tuple of image and region id
        objectid = (row['image_id'], row['region_id'])
        obj2phrases[objectid].append(row['refexp'].split())
        obj2split[objectid] = new_split_dict[row['image_id']]

    #print "Objects",len(obj2phrases)
    ############ match visual data with referring expressions ###############
    ############### & set up raw data with splits ###########################

    raw_dataset = {
                    'train': { 'filenames': list(), 'images': list(), 'captions': list() },
                    'val':   { 'filenames': list(), 'images': list(), 'captions': list() },
                    'test':  { 'filenames': list(), 'images': list(), 'captions': list() },
                }

    # tqdm visualizes progress in the terminal :)
    for obj2phrases_item in obj2phrases:  #tqdm(obj2phrases):

        # [:,1] means: all indices of x along the first axis, but only index 1 along the second --> this list comprehension filters out features for one image
        features_for_imageId = extracted_features[extracted_features[:,1] == obj2phrases_item[0]]   #obj2phrases_item[0] is image id
        # this filters out features for the correct region
        features_for_objectId = features_for_imageId[features_for_imageId[:,2] == obj2phrases_item[1]]   #obj2phrases_item[1] is region id

        if len(features_for_objectId) > 0:
            image = np.array(features_for_objectId[0])[3:]  # TODO WHY cut of 3 entries?
            test_list.append(np.array(features_for_objectId[0])[3:])
            test_count += 1

            split = obj2split[obj2phrases_item]
            filename = "_".join([str(obj2phrases_item[0]),str(obj2phrases_item[1])])
            caption_group = []
            for ref in obj2phrases[obj2phrases_item]:
                caption_group.append(ref)

            image = image / np.linalg.norm(image)

            raw_dataset[split]['filenames'].append(filename)
            raw_dataset[split]['images'].append(image)
            raw_dataset[split]['captions'].append(caption_group)

    print('raw data set',len(raw_dataset['train']['captions']))  #42279

    print(len(raw_dataset['train']['images']) + len(raw_dataset['val']['images']) + \
          len(raw_dataset['test']['images']))  #should be 49865

    print(raw_dataset['train']['captions'][0]) # output : [[u'hidden', u'chocolate', u'donut'], [u'space', u'right', u'above', u'game']]
    print(raw_dataset['train']['captions'][111])  # output : [[u'groom'], [u'groom'], [u'man']]

    # to compare with original scripts: here, the order is like
    # in im_mat from prepare_refcoco.py.
    print("count", test_count) # 49865
    test_list = np.array(test_list)
    print(test_list.shape)
    print("test:: ", test_list[1][0]) # 0.0729042887688 --> like in original script (random number chosen)


    ################################################################
    # for min_token_freq in [ 3, 4, 5 ]:
    all_tokens = (token for caption_group in raw_dataset['train']['captions'] for caption in caption_group for token in
                  caption)
    token_freqs = collections.Counter(all_tokens)
    vocab = sorted(token_freqs.keys(), key=lambda token: (-token_freqs[token], token))
    # discard words with very low frequency
    while token_freqs[vocab[-1]] < min_token_freq:
        vocab.pop()

    vocab_size = len(vocab) + 2  # + edge and unknown tokens
    print('vocab:', vocab_size)

    # "word embedding"
    token_to_index = {token: i + 2 for (i, token) in enumerate(vocab)}
    index_to_token = {i + 2: token for (i, token) in enumerate(vocab)}
    edge_index = 0
    unknown_index = 1


    ################################################################

    def parse(data):
        indexes = list()
        lens = list()
        images = list()
        for (caption_group, img) in zip(data['captions'], data['images']):
            for caption in caption_group:
                indexes_ = [token_to_index.get(token, unknown_index) for token in caption]
                indexes.append(indexes_)
                lens.append(len(indexes_) + 1)  # add 1 due to edge token
                images.append(img)

        maxlen = max(lens)

        in_mat = np.zeros((len(indexes), maxlen), np.int32)
        out_mat = np.zeros((len(indexes), maxlen), np.int32)
        for (row, indexes_) in enumerate(indexes):
            in_mat[row, :len(indexes_) + 1] = [edge_index] + indexes_
            out_mat[row, :len(indexes_) + 1] = indexes_ + [edge_index]
        return (in_mat, out_mat, np.array(lens, np.int32), np.array(images))


    (train_captions_in, train_captions_out, train_captions_len, train_images) = parse(raw_dataset['train'])
    (val_captions_in, val_captions_out, val_captions_len, val_images) = parse(raw_dataset['val'])
    (test_captions_in, test_captions_out, test_captions_len, test_images) = parse(raw_dataset['test'])
    print("Train captions", np.shape(train_captions_in))
    ################################################################
    print('Training...')

    # for layer_size in [ 128, 256, 512 ]:
    #    for method in [ 'merge', 'inject' ]:
    for run in range(1, num_runs + 1):
        model_name = '_'.join([str(x) for x in [method, dataset, min_token_freq, layer_size, run]])
        if os.path.isdir(results_data_dir + '/' + model_name):
            os.system("rm -r " + results_data_dir + '/' + model_name)
        os.makedirs(results_data_dir + '/' + model_name)

        print()
        print('-' * 100)
        print(dataset, min_token_freq, layer_size, method, run)
        print()

        tf.reset_default_graph()

        # Sequence of token indexes generated thus far included start token (or full correct sequence during training).
        seq_in = tf.placeholder(tf.int32, shape=[None, None], name='seq_in')  # [seq, token index]
        # Length of sequence in seq_in.
        seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')  # [seq len]
        # Images
        image = tf.placeholder(tf.float32, shape=[None, 4103], name='image')  # [seq, image feature]
        # Correct sequence to generate during training without start token but with end token
        seq_target = tf.placeholder(tf.int32, shape=[None, None], name='seq_target')  # [seq, token index]

        # Number of sequences to process at once.
        batch_size = tf.shape(seq_in)[0]
        # Number of tokens in generated sequence.
        num_steps = tf.shape(seq_in)[1]

        with tf.variable_scope('image'):
            # Project image vector into a smaller vector.

            W = tf.get_variable('W', [4103, layer_size], tf.float32, tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [layer_size], tf.float32, tf.zeros_initializer())

            post_image = tf.matmul(image, W) + b

        with tf.variable_scope('prefix_encoder'):
            # Encode each generated sequence prefix into a vector.

            # Embedding matrix for token vocabulary.           -> xavier: weight initialization!
            embeddings = tf.get_variable('embeddings', [vocab_size, layer_size], tf.float32,
                                         tf.contrib.layers.xavier_initializer())  # [vocabulary token, token feature]

            # 3tensor of tokens in sequences replaced with their corresponding embedding.
            # look up ids in seq_in in full vocab
            embedded = tf.nn.embedding_lookup(embeddings, seq_in)  # [seq, token, token feature]

            print ("embedded shape: ", np.shape(embedded))
           # t = tf.expand_dims(post_image, 1)
            if method == 'inject':
                rnn_input = tf.concat([embedded, tf.tile(tf.expand_dims(post_image, 1), [1, num_steps, 1])], axis=2)
            else:
                rnn_input = embedded

            # Use an LSTM to encode the generated prefix.
            init_state = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros([batch_size, layer_size]),
                                                       h=tf.zeros([batch_size, layer_size]))
            cell = tf.contrib.rnn.BasicLSTMCell(layer_size)
            (prefix_vectors, _) = tf.nn.dynamic_rnn(cell, rnn_input, sequence_length=seq_len,
                                                    initial_state=init_state)  # [seq, prefix position, prefix feature]

            # Mask of which positions in the matrix of sequences are actual labels as opposed to padding.
            token_mask = tf.cast(tf.sequence_mask(seq_len, num_steps), tf.float32)  # [seq, token flag]

        with tf.variable_scope('softmax'):
            # Output a probability distribution over the token vocabulary (including the end token)

            if method == 'merge':

                softmax_input = tf.concat([prefix_vectors, tf.tile(tf.expand_dims(post_image, 1), [1, num_steps, 1])],
                                          axis=2)
                softmax_input_size = layer_size + layer_size  # state + image
            else:
                softmax_input = prefix_vectors
                softmax_input_size = layer_size

            W = tf.get_variable('W', [softmax_input_size, vocab_size], tf.float32,
                                tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
            logits = tf.reshape(tf.matmul(tf.reshape(softmax_input, [-1, softmax_input_size]), W) + b,
                                [batch_size, num_steps, vocab_size])
            predictions = tf.nn.softmax(logits)  # [seq, prefix position, token probability]
            last_prediction = predictions[:, -1]

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seq_target, logits=logits) * token_mask
        total_loss = tf.reduce_sum(losses)
        train_step = tf.train.AdamOptimizer().minimize(total_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        num_params = 0
        for v in sess.graph.get_collection('trainable_variables'):
            num_params += np.prod(v.get_shape()).value

        print('epoch', 'val loss', 'duration', sep='\t')
        run_start = start = timeit.default_timer()

        # validation_loss = 0
        # for i in range(len(test_images)//minibatch_size):
        #    minibatch_validation_loss = sess.run(total_loss, feed_dict={
        #                                                            seq_in:     val_captions_in [i*minibatch_size:(i+1)*minibatch_size],
        #                                                            seq_len:    val_captions_len[i*minibatch_size:(i+1)*minibatch_size],
        #                                                            seq_target: val_captions_out[i*minibatch_size:(i+1)*minibatch_size],
        #                                                            image:      test_images[i*minibatch_size:(i+1)*minibatch_size]
        #                                                        })
        #    validation_loss += minibatch_validation_loss
        # print(0, round(validation_loss, 3), round(timeit.default_timer() - start), sep='\t')
        last_validation_loss = 1000000

        trainingset_indexes = list(range(len(train_images)))
        for epoch in range(1, max_epochs + 1):
            random.shuffle(trainingset_indexes)

            start = timeit.default_timer()
            for i in range(len(trainingset_indexes) // minibatch_size):
                minibatch_indexes = trainingset_indexes[i * minibatch_size:(i + 1) * minibatch_size]
                sess.run(train_step, feed_dict={
                    seq_in: train_captions_in[minibatch_indexes],
                    seq_len: train_captions_len[minibatch_indexes],
                    seq_target: train_captions_out[minibatch_indexes],
                    image: train_images[minibatch_indexes]
                })

            validation_loss = 0
            for i in range(len(test_images) // minibatch_size):
                minibatch_validation_loss = sess.run(total_loss, feed_dict={
                    seq_in: val_captions_in[i * minibatch_size:(i + 1) * minibatch_size],
                    seq_len: val_captions_len[i * minibatch_size:(i + 1) * minibatch_size],
                    seq_target: val_captions_out[i * minibatch_size:(i + 1) * minibatch_size],
                    image: val_images[i * minibatch_size:(i + 1) * minibatch_size]  # test images
                })
                validation_loss += minibatch_validation_loss
            print(epoch, round(validation_loss, 3), round(timeit.default_timer() - start), sep='\t')
            if validation_loss > last_validation_loss:
                break
            last_validation_loss = validation_loss
            print("save model", results_data_dir + '/' + model_name + '/model')
            saver.save(sess, results_data_dir + '/' + model_name + '/model')

        saver.restore(sess, tf.train.latest_checkpoint(results_data_dir + '/' + model_name))

        print()
        print('evaluating...')
        print()

        captions = list()
        for (i, image_input) in enumerate(raw_dataset['test']['images']):
            caption = generate_sequence_beamsearch(lambda prefixes: sess.run(last_prediction, feed_dict={
                seq_in: prefixes,
                seq_len: [len(p) for p in prefixes],
                image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
            }))
            captions.append(caption)

        vocab_used = len({word for caption in captions for word in caption.split(' ')})

        with open(results_data_dir + '/' + model_name + '/generated_captions.json', 'w') as f:
            print(str(json.dumps([
                {
                    'image_id': image_id,
                    'caption': caption
                }
                for (image_id, caption) in enumerate(captions)
            ])), file=f)

        print()
        print('Duration:', round(timeit.default_timer() - run_start), 's')
        print()

######### new
        oids = list()
        captions_new = list()
        for (i, image_input) in enumerate(raw_dataset['test']['images']):
            caption = generate_sequence_beamsearch(lambda prefixes: sess.run(last_prediction, feed_dict={
                seq_in: prefixes,
                seq_len: [len(p) for p in prefixes],
                image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
            }))
            captions_new.append([caption])

        for (i, item) in enumerate(raw_dataset['test']['filenames']):
            oids.append(item.split("_")[1])

        dict4eval = defaultdict(list)
        for (idx, pair) in enumerate(zip(oids, captions_new)):
            dict4eval[pair[0]] = pair[1]
        with open(results_data_dir + '/' + '4evalrefactoredexp_' + model_name + '.json', 'w') as f:
            json.dump(dict4eval, f)
