import pandas as pd
import json
from collections import defaultdict
import numpy as np
import bleu

test_ids = []

old = "candrefdict_inject_refcoco_refrnn_compositional_3_512_1.json"
candidate_path_1 = "jsons/4evalrefactoredexpinject_refcoco_refrnn_compositional_3_512_1.json"  # original script output
candidate_path_2 = "jsons/4evalafter2ndrefactoringinject_refcoco_refrnn_compositional_3_512_1.json" # output of oo code
reference_data_path = "../../data/refcoco_refdf.json.gz"

class Evalutator:
    def __init__(self):

        with open(candidate_path_1, "r") as f:
            self.candidate_1 = json.load(f)  # correct format

        self.prepare_ref_data()


        print len(self.refdict4eval.keys())
        print len(self.candidate_1.keys())

        print self.refdict4eval["1721891"][0]
    #    print self.candidate_1["1721891"]

        for key in self.refdict4eval.keys():
            if not key in self.candidate_1.keys():
                del self.refdict4eval[key]

        print len(self.refdict4eval.keys())
        print len(self.candidate_1.keys())

        f = open('test.out', 'wb')
        for i in range(len(test_ids)):
            f.write("%i\n" % (test_ids[i]))
        f.close()

        with open('jsons/test.json', 'w') as f:
            json.dump(self.refdict4eval, f)

        bleu.evaluate("jsons/test.json", candidate_path_1)

    def prepare_ref_data(self):
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

        self.refdict4eval = defaultdict()

        for obj2phrases_item in self.obj2phrases:  # tqdm(obj2phrases):
            split = self.obj2split[obj2phrases_item]

            if split == 'test':
                test_ids.append(obj2phrases_item)
                caption_group = []
                for ref in self.obj2phrases[obj2phrases_item]:
                    caption_group.append(ref)
                self.refdict4eval[str(obj2phrases_item)] = caption_group


if __name__ == '__main__':
    eval = Evalutator()