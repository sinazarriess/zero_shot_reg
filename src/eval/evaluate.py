import pandas as pd
import json
import os
from collections import defaultdict
import numpy as np
import bleu
import ast

test_ids = []

candidate_path_1 = 'restoredmodel_captions_' #'jsons/4evalrefactoredexpinject_refcoco_refrnn_compositional_3_512_'  # original script output
candidate_path_2 = 'jsons/4evalafter2ndrefactoringinject_refcoco_refrnn_compositional_3_512_' # output of oo code
#reference_dict_path = 'jsons/test.json'
reference_data_path = '../../data/refcoco_refdf.json.gz'

class Evalutator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.reference_dict_path = model_path + 'test.json'
        print  "Reference dict (test.json) exists: ", os.path.exists(self.reference_dict_path)
        if os.path.exists(self.reference_dict_path):
            with open(self.reference_dict_path, "r") as f:
                self.refdict4eval = json.load(f)  # correct format

        else:
            self.refdict4eval = defaultdict()
            self.prepare_ref_data()


    def run_eval(self, candidate):
        with open(candidate, "r") as f:
            cand = json.load(f)  # correct format

        # keys without image features have to be filtered out - this is the easiest way without checking image vectors
        for key in self.refdict4eval.keys():
            if not key in cand.keys():
                del self.refdict4eval[key]

        assert len(self.refdict4eval.keys()) == len(cand.keys())

        with open(self.reference_dict_path, 'w') as f:
            json.dump(self.refdict4eval, f)

        return bleu.evaluate(self.reference_dict_path, candidate, True)

    def prepare_ref_data(self):
        with open(self.model_path + 'refs_moved_to_test.json', 'r') as f:
            ids = f.readline()
            extra_items_list = ast.literal_eval(ids)

        refcoco_data = pd.read_json(reference_data_path, orient="split", compression="gzip")
        with open("../../data/refcoco_splits.json") as f:
            splits = json.load(f)
        splitmap = {'val': 'val', 'train': 'train', 'testA': 'test', 'testB': 'test'}
        # for every group in split --> for every entry --> make entry in new dict
        # file2split just translates testA and testB to "test"?
        new_split_dict = {val: splitmap[key] for key in splits for val in splits[key]}

        # dict of objectids and ref exps
        self.obj2phrases = defaultdict(list)
        # dict of objectids and split (train,test or val)
        self.obj2split = {}

        # iterate over json "entries"
        for index, row in refcoco_data.iterrows():
            objectid = row['region_id']
            self.obj2phrases[objectid].append(row['refexp'])#.split())
            self.obj2split[objectid] = new_split_dict[row['image_id']]

        for obj2phrases_item in self.obj2phrases:  # tqdm(obj2phrases):
            split = self.obj2split[obj2phrases_item]

            if split == 'test' or str(obj2phrases_item) in extra_items_list:
                test_ids.append(obj2phrases_item)
                caption_group = []
                for ref in self.obj2phrases[obj2phrases_item]:
                    caption_group.append(ref)
                self.refdict4eval[str(obj2phrases_item)] = caption_group


if __name__ == '__main__':

    #eval.run_eval('./jsons/no_unknown_for_comp.json')
    model_path = 'model/with_reduced_cats_2/'
    eval = Evalutator(model_path)
    eval.run_eval(model_path + 'inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json') #'restoredmodel_refs_beam.json')
    score_1 = 0
    score_2 = 0

#    for i in range(1,4):
 #       print "evaluate " + candidate_path_1 + str(i) + '.json'
 #       score_1 += eval.run_eval(candidate_path_1 + str(i) + '.json')['Bleu_1']

 #   for i in range(1, 4):
 #       print "evaluate " + candidate_path_2 + str(i) + '.json'
 #       score_2 += eval.run_eval(candidate_path_2 + str(i) + '.json')['Bleu_1']

 #   print "Final average BLEU_1 score Candidate 1: " + str(score_1 / 3.0)
 #   print "Final average BLEU_1 score Candidate 2: " + str(score_2 / 3.0)