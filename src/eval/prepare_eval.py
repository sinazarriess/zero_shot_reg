import tensorflow as tf
import lstm.beam as beam
import lstm.data
from tensorflow.python.tools import inspect_checkpoint as chkp




# convert test set into one compact dict (keys: region_id, values: refexps)
# ToDo how should refexps look?

# generate captions for test set with beam search


class Evalutator:
    def __init__(self, modeldir):
        self.data_interface = lstm.data.Data() #todo??
        chkp.print_tensors_in_checkpoint_file(modeldir, tensor_name='', all_tensors=True)




        print("v2 : %s" % v2.eval())


  #  def generate_candidate_captions(self):
  #      with tf.Session() as sess:
  #          saver = tf.train.Saver()
  #          saver.restore(sess, tf.train.latest_checkpoint(p.results_data_dir + '/' + model.model_name))
  #
  #           print('\nevaluating...\n')
  #
  #           captions = list()
  #           searcher = beam.Search(self.data_interface.index_to_token)
  #           for (i, image_input) in enumerate(self.data_interface.raw_dataset['test']['images']):
  #               caption = searcher.generate_sequence_beamsearch(lambda prefixes: sess.run(model.last_prediction, feed_dict={  #todo! last prediction found?
  #                   model.seq_in: prefixes,
  #                   model.seq_len: [len(p) for p in prefixes],
  #                   model.image: image_input.reshape([1, -1]).repeat(len(prefixes), axis=0)
  #               }))
  #               captions.append(caption)
  #
  #           vocab_used = len({word for caption in captions for word in caption.split(' ')})
  #
  #           with open(p.results_data_dir + '/' + model.model_name + '/generated_captions.json', 'w') as f:
  #               print(str(json.dumps([
  #                   {
  #                       'image_id': image_id,
  #                       'caption': caption
  #                   }
  #                   for (image_id, caption) in enumerate(captions)
  #               ])), file = f)
  #
  #           print('\nDuration:', round(timeit.default_timer() - run_start), 's\n')


if __name__ == '__main__':
    modeldir = '/media/compute/vol/dsg/lilian/testrun/results/inject_refcoco_refrnn_compositional_3_512_1/model'
    eval = Evalutator(modeldir)