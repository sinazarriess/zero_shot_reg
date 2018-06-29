import ast
import lstm.data
from collections import defaultdict
import pandas as pd
import json
import utils
from pprint import pprint
import numpy as np

modelpath = './model/with_reduced_cats_2/'
file_to_analyse = 'restoredmodel_refs_greedy.json'#'inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json'
#rawdata_path = "../data/refcoco/refcoco_refdf.json.gz"

categories_excluded = [ 73 ]

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
        cow_counter = 0
        animal_counter = 0
        animals = ['sheep', 'bear', 'elephant', 'giraffe', 'dog', 'bird']
        laptop_counter = 0
        screen_counter = 0
        device_counter = 0
        devices = [ 'phone', 'keyboard', 'monitor', 'tv']
        phrases_with_excluded_category = list()
        for index, row in self.bounding_boxes.iterrows():
            id = str(row['region_id'])
            if row['cat'] in categories_excluded:
                if id in self.refs.keys():
                    tmpdict = defaultdict()
                    tmpdict['cat'] = self.categories[ str(row['cat']) ]
                    tmpdict['RE'] = self.refs[id]
                    tmpdict['reference'] = self.reference[id]
                    if id in self.candidates.keys():
                        tmpdict['alternatives'] = self.candidates[id]
                    else:
                        tmpdict['alternatives'] = ""
                    self.analysis_dict[id] = tmpdict

                    animal = False
                    unknown = False
                    cow = False
                    screen = False
                    device = False
                    laptop = False
                    tmp = [e for e in self.refs[id] if any(x in e for x in devices)]
                    tmp2 = [e for e in self.refs[id] if any(x in e for x in animals)]
                    if len(tmp) > 0:
                        device = True
                    if len(tmp2) > 0:
                        animal = True
                    for word in self.refs[id]:
                        if 'screen' in word:
                            screen = True
                        if 'UNKNOWN' in word:
                            unknown = True
                        if 'laptop' in word:
                            laptop = True
                        if 'cow' in word:
                            cow = True

                    if animal:
                        animal_counter += 1
                    if unknown:
                        counter += 1
                    if cow:
                        cow_counter += 1
                    if device:
                        device_counter += 1
                    if screen:
                        screen_counter += 1
                    if laptop:
                        laptop_counter += 1

        print "Number of phrases containing UNKNOWN: ", counter
        print "Number of phrases describing excluded categories: ", len(self.analysis_dict)
        print "Cow prediction: ", cow_counter / float(len(self.analysis_dict))
        print "Animal prediction: ", animal_counter / float(len(self.analysis_dict))
        print "Laptop prediction: ", laptop_counter / float(len(self.analysis_dict))
        print "Screen prediction: ", screen_counter / float(len(self.analysis_dict))
        print "Device prediction: ", device_counter / float(len(self.analysis_dict))

        for item in self.analysis_dict:
            pprint(self.analysis_dict[item])


if __name__ == "__main__":
    a = Analyse()
    a.first_step()
