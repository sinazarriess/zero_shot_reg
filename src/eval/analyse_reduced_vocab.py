import ast
import lstm.data
from collections import defaultdict
import pandas as pd
import json
import utils
from pprint import pprint
import numpy as np

modelpath = './model/with_reduced_cats/'
file_to_analyse = 'restoredmodel_refs_greedy.json'#'inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json'
#rawdata_path = "../data/refcoco/refcoco_refdf.json.gz"

class Analyse:
    def __init__(self):
        with open(modelpath + file_to_analyse) as f:
            self.refs = json.load(f)

        with open(modelpath + 'test.json', 'r') as f:
            self.reference = json.load(f)

        with open(modelpath + 'highest_prob_candidates.json', 'r') as f:
            self.candidates = json.load(f)

        with open(modelpath + 'refs_moved_to_test.json', 'r') as f:
            ids = f.readline()
            self.extra_items_list = ast.literal_eval(ids)

        self.bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")
        self.analysis_dict = defaultdict()
        self.categories = utils.read_in_cats()

    def first_step(self):
        print "Number of phrases left out in training: ", len(self.extra_items_list)

        counter = 0
        for reg_id in self.extra_items_list:
            for word in self.refs[reg_id]:
                if "UNKNOWN" in word:
                    counter += 1
        print "Number of phrases containing UNKNOWN: ", counter

        for index, row in self.bounding_boxes.iterrows():
            id = str(row['region_id'])
            if id in self.extra_items_list:
                tmpdict = defaultdict()
                tmpdict['cat'] = self.categories[ str(row['cat']) ]
                tmpdict['RE'] = self.refs[id]
                tmpdict['reference'] = self.reference[id]
                if id in self.candidates.keys():
                    tmpdict['alternatives'] = self.candidates[id]
                else:
                    tmpdict['alternatives'] = ""
                self.analysis_dict[id] = tmpdict

        for item in self.analysis_dict:
            pprint(self.analysis_dict[item])


if __name__ == "__main__":
    a = Analyse()
    a.first_step()
