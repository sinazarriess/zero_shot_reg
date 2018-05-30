from __future__ import division, unicode_literals, print_function
import tensorflow as tf
import timeit
import random
import params as p
import numpy as np
import beam
import json

class Learn:

    def run_training(self, model, data):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(p.results_data_dir + '/' + model.model_name + '/model/graph')

        num_params = 0
        for v in sess.graph.get_collection('trainable_variables'):
            num_params += np.prod(v.get_shape()).value

        print('epoch\t', 'val loss\t', 'duration\t')
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

        trainingset_indexes = list(range(len(data.train_images)))
        for epoch in range(1, p.max_epochs + 1):
            random.shuffle(trainingset_indexes)

            start = timeit.default_timer()
            for i in range(len(trainingset_indexes) // p.minibatch_size):
                minibatch_indexes = trainingset_indexes[i * p.minibatch_size:(i + 1) * p.minibatch_size]
                sess.run(model.train_step, feed_dict={
                    model.seq_in: data.train_captions_in[minibatch_indexes],
                    model.seq_len: data.train_captions_len[minibatch_indexes],
                    model.seq_target: data.train_captions_out[minibatch_indexes],
                    model.image: data.train_images[minibatch_indexes]
                })

            validation_loss = 0
            for i in range(len(data.test_images) // p.minibatch_size):
                minibatch_validation_loss = sess.run(model.total_loss, feed_dict={
                    model.seq_in: data.val_captions_in[i * p.minibatch_size:(i + 1) * p.minibatch_size],
                    model.seq_len: data.val_captions_len[i * p.minibatch_size:(i + 1) * p.minibatch_size],
                    model.seq_target: data.val_captions_out[i * p.minibatch_size:(i + 1) * p.minibatch_size],
                    model.image: data.val_images[i * p.minibatch_size:(i + 1) * p.minibatch_size]  # test images
                })
                validation_loss += minibatch_validation_loss
            print(epoch, '\t', round(validation_loss, 3), '\t', round(timeit.default_timer() - start))
            if validation_loss > last_validation_loss:
                break
            last_validation_loss = validation_loss
            print("save model", p.results_data_dir + '/' + model.model_name + '/model')
            saver.save(sess, p.results_data_dir + '/' + model.model_name + '/model')

        saver.restore(sess, tf.train.latest_checkpoint(p.results_data_dir + '/' + model.model_name))



        captions = list()
        searcher = beam.Search(data.index_to_token)
        for (i, image_input) in enumerate(data.raw_dataset['test']['images']):
            caption = searcher.generate_sequence_beamsearch(lambda prefixes: sess.run(model.last_prediction, feed_dict={
                model.seq_in: prefixes,
                model.seq_len: [len(p) for p in prefixes],
                model.image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
            }))
            captions.append(caption)

        vocab_used = len({word for caption in captions for word in caption.split(' ')})

        with open(p.results_data_dir + '/' + model.model_name + '/generated_captions.json', 'w') as f:
            print(str(json.dumps([
                {
                    'image_id': image_id,
                    'caption': caption
                }
                for (image_id, caption) in enumerate(captions)
            ])), file = f)

        print('\nDuration:', round(timeit.default_timer() - run_start), 's\n')
        print('\nevaluating...\n')

#### new
        dict4eval = model.generate_captions(data.raw_dataset, searcher, sess)
        with open('/media/compute/vol/dsg/lilian/src/' + '4evalafter2ndrefactoring' + model.model_name + '.json', 'w') as f:
            json.dump(dict4eval, f)

        print('\n wrote new json\n')