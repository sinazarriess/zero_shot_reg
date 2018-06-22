import ast
import lstm.data
from collections import defaultdict
import pandas as pd
import json
import utils
from pprint import pprint

modelpath = './model/with_reduced_vocab/'
file_to_analyse = 'inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json'
rawdata_path = "../data/refcoco/refcoco_refdf.json.gz"

class Analyse:
    def __init__(self):
        with open(modelpath + file_to_analyse) as f:
            self.refs = json.load(f)

        with open(modelpath + 'refs_moved_to_test.json', 'r') as f:
            ids = f.readline()
            self.extra_items_list = ast.literal_eval(ids)

        self.bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")
        self.analysis_dict = defaultdict()
        self.categories = utils.read_in_cats()

    def first_step(self):
        for reg_id in self.extra_items_list:
            #print reg_id
            print self.refs[reg_id]

        for index, row in self.bounding_boxes.iterrows():
            id = str(row['region_id'])
            if id in self.extra_items_list:
                tmpdict = defaultdict()
                tmpdict['cat'] = self.categories[ str(row['cat']) ]
                tmpdict['ref'] = self.refs[id]
                self.analysis_dict[id] = tmpdict

        for item in self.analysis_dict:
            pprint(self.analysis_dict[item])


if __name__ == "__main__":
    a = Analyse()
    a.first_step()
