import lstm
import data
import train
import params
import numpy as np
import json
import params as p
import pandas as pd
from collections import defaultdict
from pprint import pprint
import os
#import statistics
import csv

bb_path = 'lstm/mscoco_bbdf.json.gz'
results_data_dir = '/media/compute/vol/dsg/lilian/exp/with_reduced_cats_' + 'test'

#  ATTENTION enter words that are supposed to be in the dictionary for zero-shot naming - e.g. the name of
#  an excluded category or excluded words, if using this variable
#  words = [ "laptop" ] #["bus"] #"juice", "soldier", "cookie", "watch", "lemon", "suv"

#  words_excluded = ["juice", "soldier", "cookie", "watch", "lemon", "suv"]
#  categories_excluded = [ 73 ]  # 19-horse  73-laptop 6-bus 24-zebra


# store data for later application of stored LSTM (eval/generatecaptionsfromstoredmodel.py)
def generate_indextotoken(data, results_dir, words):

    with open(results_dir + '/index2token.json', 'w') as f:
        json.dump(data.index_to_token, f)

    with open(results_dir+ '/vocab_list.txt', 'w') as f:
        np.savetxt(f, data.vocab, delimiter=', ', fmt="%s")

    # with open(results_dir + '/raw_dataset_filenames.txt', 'w') as f:
    #     np.savetxt(f, data.raw_dataset['test']['filenames'], delimiter=', ', fmt="%s")
    #
    #
    # with open(results_dir + '/raw_dataset.txt', 'w') as f:
    #     np.savetxt(f, data.raw_dataset['test']['images'], delimiter=', ')
    #
    #
    # with open(results_dir + '/raw_dataset_filenames_train.txt', 'w') as f:
    #     np.savetxt(f, data.raw_dataset['train']['filenames'], delimiter=', ', fmt="%s")
    #
    # with open(results_dir + '/raw_dataset_train.txt', 'w') as f:
    #     np.savetxt(f, data.raw_dataset['train']['images'], delimiter=', ')

    with open(results_dir + '/additional_vocab.txt', 'w') as f:
        np.savetxt(f, words, delimiter=', ', fmt='%s')

# find all regions displaying a category in order to set up new split
def parse_categories(excluded_cats):
    tmp = np.zeros(91, dtype=int)
    bounding_boxes = pd.read_json(bb_path, orient="split", compression="gzip")
    #bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")
    for index, row in bounding_boxes.iterrows():
   #     if str(row['cat']) in excluded_cats:
        tmp[row['cat']] += 1 #(str(row['region_id']))
    pprint(tmp)
    category_freqs = defaultdict()

    count = 0
    for i in range(0,91):
        category_freqs[str(i)] = tmp[i]
        count += tmp[i]

    #print "median freq of cats: ", statistics.median(sorted(category_freqs.values()))
    print "mean freq of cats: ", count/float(len(tmp))

    excluded_ids = []
    for index, row in bounding_boxes.iterrows():
        if str(row['cat']) in excluded_cats:
            excluded_ids.append(row['region_id'])
    return excluded_ids

   # with open('category_freqs.json', 'w') as f:
   #     json.dump(category_freqs, f)
   # pprint(category_freqs)

if __name__ == '__main__':

    print "******** Train model ****************"
    data_interface = data.Data(results_data_dir, [], [], [])
    print data_interface.vocab_size
    training = train.Learn(results_data_dir)
    for run in range(1, params.num_runs + 1):
        # this is the configuration for a model that knows all categories! Therefore, the list of excluded IDs is empty.
        model = lstm.LSTM(run, data_interface.vocab_size, results_data_dir, [], data_interface.index_to_token)
        model.build_network()
        training.run_training(model, data_interface)

    generate_indextotoken(data_interface, results_data_dir, [])


    ##### run training for all categories
    # categories = defaultdict()
    # # reader = csv.reader(open("../eval/cats.txt"))
    # reader = csv.reader(open("cats.txt"))
    # for row in reader:
    #     categories[row[0].strip()] = row[1:]
    #
    # for key in categories.keys():
    #     print "******** Train model without:", categories[key][0].strip()
    #     cat_ids = parse_categories([key])
    #
    #     if len(cat_ids) == 0:
    #         continue
    #     words = [categories[key][0].strip()]
    #     # print cat_ids, words
    #     results_data_dir = '/media/compute/vol/dsg/lilian/exp/with_reduced_cats_' + key
    #     # if not os.path.exists(results_data_dir + '/inject_refcoco_refrnn_compositional_3_512_1'):
    #
    #     data_interface = data.Data(results_data_dir, [], cat_ids, words)
    #     if len(data_interface.refs_moved_to_test) > 0:  # if category is in training set!
    #         print data_interface.vocab_size
    #         training = train.Learn(results_data_dir)
    #         for run in range(1, params.num_runs + 1):
    #             model = lstm.LSTM(run, data_interface.vocab_size, results_data_dir, cat_ids,
    #                               data_interface.index_to_token)
    #             model.build_network()
    #             training.run_training(model, data_interface)
    #
    #         generate_indextotoken(data_interface, results_data_dir, words)
    #     else:
    #         print "was not trained: ", key
    #     # else:
    #     #  print "was already trained: ", key
    # #
    # # cat_ids = parse_categories(categories_excluded)