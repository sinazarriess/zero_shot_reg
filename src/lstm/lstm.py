import params as p
import os
import tensorflow as tf
import numpy as np

from collections import defaultdict
from tensorflow.contrib.tensorboard.plugins import projector

class LSTM():

    def __init__(self, run, vocab_size):
        self.method = 'inject' # 'merge'
        self.model_name = '_'.join([str(x) for x in [self.method, p.dataset, p.min_token_freq, p.layer_size, run]])
        if os.path.isdir(p.results_data_dir + '/' + self.model_name):
            os.system("rm -r " + p.results_data_dir + '/' + self.model_name)
        os.makedirs(p.results_data_dir + '/' + self.model_name)
        self.vocab_size = vocab_size

        print('\n-' * 100)
        print(p.dataset, p.min_token_freq, p.layer_size, self.method, run, '\n')

    def final_prediction(self, tensor):
        return tensor[:, -1]

    def build_network(self):
        tf.reset_default_graph()

        # Create Projector config
        self.config = projector.ProjectorConfig()
        # Add embedding visualizer
        self.ambedding = self.config.embeddings.add()

        # Sequence of token indexes generated thus far included start token (or full correct sequence during training).
        self.seq_in = tf.placeholder(tf.int32, shape=[None, None], name='seq_in')  # [seq, token index]
        # Length of sequence in seq_in.
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')  # [seq len]
        # Images
        self.image = tf.placeholder(tf.float32, shape=[None, 4103], name='image')  # [seq, image feature]
        # Correct sequence to generate during training without start token but with end token
        self.seq_target = tf.placeholder(tf.int32, shape=[None, None], name='seq_target')  # [seq, token index]

        # Number of sequences to process at once.
        batch_size = tf.shape(self.seq_in)[0]
        # Number of tokens in generated sequence.
        num_steps = tf.shape(self.seq_in)[1]

        with tf.variable_scope('image'):
            # Project image vector into a smaller vector.

            W = tf.get_variable('W', [4103, p.layer_size], tf.float32, tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [p.layer_size], tf.float32, tf.zeros_initializer())

            post_image = tf.matmul(self.image, W) + b

        with tf.variable_scope('prefix_encoder'):
            # Encode each generated sequence prefix into a vector.

            # Embedding matrix for token vocabulary.           -> xavier: weight initialization!
            self.embeddings = tf.get_variable('embeddings', [self.vocab_size, p.layer_size], tf.float32,
                                         tf.contrib.layers.xavier_initializer())  # [vocabulary token, token feature]

            # 3tensor of tokens in sequences replaced with their corresponding embedding.
            # look up ids in seq_in in full vocab
            embedded = tf.nn.embedding_lookup(self.embeddings, self.seq_in)  # [seq, token, token feature]

            print ("embedded shape: ", np.shape(embedded))
            # t = tf.expand_dims(post_image, 1)
            if self.method == 'inject':
                rnn_input = tf.concat([embedded, tf.tile(tf.expand_dims(post_image, 1), [1, num_steps, 1])], axis=2)
            else:
                rnn_input = embedded

            # Use an LSTM to encode the generated prefix.
            init_state = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros([batch_size, p.layer_size]),
                                                       h=tf.zeros([batch_size, p.layer_size]))
            cell = tf.contrib.rnn.BasicLSTMCell(p.layer_size)
            (prefix_vectors, _) = tf.nn.dynamic_rnn(cell, rnn_input, sequence_length = self.seq_len,
                                                    initial_state=init_state)  # [seq, prefix position, prefix feature]

            # Mask of which positions in the matrix of sequences are actual labels as opposed to padding.
            token_mask = tf.cast(tf.sequence_mask(self.seq_len, num_steps), tf.float32)  # [seq, token flag]

        with tf.variable_scope('softmax'):
            # Output a probability distribution over the token vocabulary (including the end token)

            if self.method == 'merge':

                softmax_input = tf.concat([prefix_vectors, tf.tile(tf.expand_dims(post_image, 1), [1, num_steps, 1])],
                                          axis=2)
                softmax_input_size = p.layer_size + p.layer_size  # state + image
            else:
                softmax_input = prefix_vectors
                softmax_input_size = p.layer_size

            W = tf.get_variable('W', [softmax_input_size, self.vocab_size], tf.float32,
                                tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.zeros_initializer())
            logits = tf.reshape(tf.matmul(tf.reshape(softmax_input, [-1, softmax_input_size]), W) + b,
                                [batch_size, num_steps, self.vocab_size])
            self.predictions = tf.nn.softmax(logits, name = 'prediction')  # [seq, prefix position, token probability]
            self.last_prediction = tf.py_func(self.final_prediction, [self.predictions], tf.float32 , name = "last-prediction") #self.predictions[:, -1]

        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.seq_target, logits=logits) * token_mask
        self.total_loss = tf.reduce_sum(self.losses)
        self.train_step = tf.train.AdamOptimizer().minimize(self.total_loss)

        tf.summary.scalar("loss", self.total_loss)
        tf.summary.histogram("histogram loss", self.total_loss)

        self.summary_op = tf.summary.merge_all()

    def generate_captions(self, raw_dataset, beam, sess):
        oids = list()
        captions = list()
        for (i, image_input) in enumerate(raw_dataset['test']['images']):
            caption = beam.generate_sequence_beamsearch(lambda prefixes: sess.run(self.last_prediction, feed_dict={
                self.seq_in: prefixes,
                self.seq_len: [len(p) for p in prefixes],
                self.image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)  # -1 means unspecified dimension
            }))
            captions.append([caption])

        for (i, item) in enumerate(raw_dataset['test']['filenames']):
            oids.append(item.split("_")[1])

        dict4eval = defaultdict(list)
        for (idx, pair) in enumerate(zip(oids, captions)):
            dict4eval[pair[0]] = pair[1]
        return dict4eval



 #       sess = tf.Session()
 #       sess.run(tf.global_variables_initializer())
 #       writer = tf.summary.FileWriter('./logs/train ', sess.graph)
 #       tf.summary.histogram('predictions', self.predictions)
 #       summary_op = tf.summary.merge_all()


#if __name__ == '__main__':
#    model = LSTM(1,200)
#    model.build_network()