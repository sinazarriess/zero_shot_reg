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
#import statistics
import csv

#words_excluded = ["juice", "soldier", "cookie", "watch", "lemon", "suv"]
#categories_excluded = [ 73 ]  # 19-horse  73-laptop 6-bus 24-zebra

#  ATTENTION enter words that are supposed to be in the dictionary for zero-shot naming - e.g. the name of
#  an excluded category or excluded words
#words = [ "laptop" ] #["bus"] #"juice", "soldier", "cookie", "watch", "lemon", "suv"

def generate_indextotoken(data, results_dir):
    with open(results_dir + '/raw_dataset_filenames.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['test']['filenames'], delimiter=', ', fmt="%s")

    with open(results_dir + '/index2token.json', 'w') as f:
        json.dump(data.index_to_token, f)

    with open(results_dir + '/raw_dataset.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['test']['images'], delimiter=', ')


    with open(results_dir+ '/vocab_list.txt', 'w') as f:
        np.savetxt(f, data.vocab, delimiter=', ', fmt="%s")

    with open(results_dir + '/raw_dataset_filenames_train.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['train']['filenames'], delimiter=', ', fmt="%s")

    with open(results_dir + '/raw_dataset_train.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['train']['images'], delimiter=', ')

    with open(results_dir+ '/additional_vocab.txt', 'w') as f:
        np.savetxt(f, words, delimiter=', ', fmt='%s')

def parse_categories(excluded_cats):
    tmp = np.zeros(91, dtype=int)
    bounding_boxes = pd.read_json('lstm/mscoco_bbdf.json.gz', orient="split", compression="gzip") #todo refactor  # ../../data/..  lstm
    for index, row in bounding_boxes.iterrows():
   #     if str(row['cat']) in excluded_cats:
        tmp[row['cat']] += 1 #(str(row['region_id']))
    pprint(tmp)
    category_freqs = defaultdict()

    count = 0
    for i in range(0,91):
        category_freqs[i] = tmp[i]
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

    categories = defaultdict()
    reader = csv.reader(open("cats.txt"))
    for row in reader:
        categories[row[0].strip()] = row[1:]

    for key in categories.keys():
        print key, categories[key]

        print "******** Train model without ", categories[key]
        cat_ids = parse_categories([key])

        if len(cat_ids) == 0:
            print "category not in test set: ", categories[key]
            continue
        words = categories[key]
        print cat_ids, words
        results_data_dir = '/media/compute/vol/dsg/lilian/exp/with_reduced_cats_' + key

        data_interface = data.Data(results_data_dir, [], cat_ids, words)
        print data_interface.vocab_size
        training = train.Learn(results_data_dir)
        for run in range(1, params.num_runs + 1):
            model = lstm.LSTM(run, data_interface.vocab_size, results_data_dir, cat_ids, data_interface.index_to_token)
            model.build_network()
            training.run_training(model, data_interface)

        generate_indextotoken(data_interface, results_data_dir)
    #
           #cat_ids = parse_categories(categories_excluded)