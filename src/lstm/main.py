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


def generate_indextotoken(data):
    with open(p.results_data_dir + '/raw_dataset_filenames.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['test']['filenames'], delimiter=', ', fmt="%s")

    with open(p.results_data_dir + '/index2token.json', 'w') as f:
        json.dump(data.index_to_token, f)

    with open(p.results_data_dir + '/raw_dataset.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['test']['images'], delimiter=', ')


def parse_categories(excluded_cats):
    tmp = np.zeros(91, dtype=int)
    bounding_boxes = pd.read_json('lstm/mscoco_bbdf.json.gz', orient="split", compression="gzip") #todo refactor  # ../../data/..
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
        if row['cat'] in categories_excluded:
            excluded_ids.append(row['region_id'])
    return excluded_ids

   # with open('category_freqs.json', 'w') as f:
   #     json.dump(category_freqs, f)
   # pprint(category_freqs)

if __name__ == '__main__':

   # words_excluded = ["juice", "soldier", "cookie", "watch", "lemon", "suv"]
    categories_excluded = [ 73 ] # 19-horse  73-laptopd
    cat_ids = parse_categories(categories_excluded)

    # #data_interface = data.Data(words_excluded)
    data_interface = data.Data([], cat_ids)
    print data_interface.vocab_size
    training = train.Learn()

    for run in range(1, params.num_runs + 1):
        model = lstm.LSTM(run, data_interface.vocab_size)
        model.build_network()
        training.run_training(model, data_interface)

    generate_indextotoken(data_interface)

