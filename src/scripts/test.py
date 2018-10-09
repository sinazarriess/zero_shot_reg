import numpy as np
from pprint import pprint
from collections import defaultdict, OrderedDict
import pandas as pd
import statistics

def parse_categories():
    tmp = np.zeros(91, dtype=int)
    bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")
    #bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")

    print "bb count: ", len(bounding_boxes.index)
    for index, row in bounding_boxes.iterrows():
   #     if str(row['cat']) in excluded_cats:
        tmp[row['cat']] += 1 #(str(row['region_id']))
    pprint(tmp)
    category_freqs = defaultdict()

    count = 0
    for i in range(1,91):
        category_freqs[str(i)] = tmp[i]
        count += tmp[i]
        if tmp[i] > 0:
            print "Category index: ", i, ", Frequency: ", tmp[i]

    print "median freq of cats: ", statistics.median(sorted(category_freqs.values()))
    print "mean freq of cats: ", count/float(len(tmp))




if __name__ == "__main__":
    parse_categories()