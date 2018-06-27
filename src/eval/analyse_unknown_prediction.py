import json
from collections import defaultdict
import pandas as pd
import cv2 as cv
from sets import Set
from pprint import pprint
import csv
import statistics
import utils
import ast

modelpath = 'model/with_unknown/'
file2analyse = 'restoredmodel_refs_greedy.json' #'inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json'


class Analyse:

    def __init__(self):
        self.analysis_dict = defaultdict()
        self.categories = defaultdict()
        self.images = set()
        self.regionids = list()

    def read_files(self):
        with open(modelpath + 'test.json', "r") as f: # TODO Achtung
            self.reference = json.load(f)

        with open('./model/without_unknown/inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json') as f:
            self.without_unknown_candidate = json.load(f)

        try:
            with open(modelpath + 'highest_prob_candidates.json', 'r') as f:
                self.candidate_words = json.load(f)
        except IOError:
            print "no candidate words"

        with open(modelpath + file2analyse) as f:
            self.unknown_candidate = json.load(f)

        self.categories = utils.read_in_cats()
        self.refcoco_data = pd.read_json("./../../data/refcoco_refdf.json.gz", orient="split", compression="gzip")
        self.bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")

        print "Number of images in test set: ", len(self.reference.keys())


    def parse(self):
        unknown_counter = 0
        for refex_id in self.unknown_candidate:
            comparison = defaultdict()
            for word in self.unknown_candidate[refex_id]:
                if "UNKNOWN" in word:
                    unknown_counter += 1
                    comparison['reflist'] = self.reference[refex_id]
#                    comparison['original generated caption'] = self.without_unknown_candidate[refex_id] #TODO
                    comparison['with unknown'] = self.unknown_candidate[refex_id]
                    self.analysis_dict[refex_id] = comparison

        print "Number of unknown occurences: ", unknown_counter


    def add_additional_info(self):
        for id in self.analysis_dict.keys():
            self.regionids.append(str(id))

        for index, row in self.refcoco_data.iterrows():
            if str(row['region_id']) in self.regionids:
                self.images.add(row['image_id'])
                self.analysis_dict[str(row['region_id'])]['image_id'] = row['image_id']
                self.analysis_dict[str(row['region_id'])]['refexp_id'] = row['rex_id']

        for index, row in self.bounding_boxes.iterrows():
            if str(row['region_id']) in self.regionids:
                self.analysis_dict[str(row['region_id'])]['cat'] = row['cat']
                self.analysis_dict[str(row['region_id'])]['bb'] = row['bb']

        print self.analysis_dict[self.analysis_dict.keys()[0]]

        print len(self.images)
        print self.images

    def save(self):
        with open(modelpath + 'unknown_analysis.json', 'w') as f:
            json.dump(self.analysis_dict, f)

        dict4eval = defaultdict()
#        for reg_id in self.analysis_dict.keys():
#            dict4eval[reg_id] = self.analysis_dict[reg_id]['original generated caption']
        with open(modelpath + 'unknown_analysis_eval_original.json', 'w') as f:
            json.dump(dict4eval, f)

        dict4comparison = defaultdict()
        for reg_id in self.without_unknown_candidate.keys():
            if reg_id not in self.analysis_dict.keys():
                dict4comparison[reg_id] = self.without_unknown_candidate[reg_id]
        with open(modelpath + 'no_unknown_for_comp.json', 'w') as f:
            json.dump(dict4comparison, f)

        # sanity check
        overlapping_keys = [k for k in dict4eval if k in dict4comparison]
        assert len(overlapping_keys) == 0


    def analyse(self):
        self.read_files()
        self.parse()
        self.add_additional_info()
        self.save()

    def visualize_unknown(self, region_id=-1):
        with open(modelpath + 'unknown_analysis.json', 'r') as f:
            data = json.load(f)

        if region_id >= 0:
            img_id = data[region_id]['image_id']
            filename = "/mnt/Data/zero_shot_reg/coco-caption/images/train2014/COCO_train2014_" + str(img_id).zfill(
                12) + ".jpg"
            bb_pt1 = (int(data[region_id]['bb'][0]), int(data[region_id]['bb'][1]))
            bb_pt2 = (bb_pt1[0] + int(data[region_id]['bb'][2]), bb_pt1[1] + int(data[region_id]['bb'][3]))
            img = cv.imread(filename)
            img = cv.rectangle(img, bb_pt1, bb_pt2, (0, 255, 0), 4)
            print "\n\n\n"
            pprint(data[region_id])
            cv.imshow('test', img)
            c = cv.waitKey(0);
            return
        else:
            average_length = 0
            for reg_id in data:
                print data[reg_id]['with unknown']
                average_length += len(data[reg_id]['with unknown'])
            print "average utterance length (with unknown): ", average_length / len(data)
            for reg_id in data:
                img_id = data[reg_id]['image_id']
                filename = "/mnt/Data/zero_shot_reg/coco-caption/images/train2014/COCO_train2014_" + str(img_id).zfill(
                    12) + ".jpg"
                bb_pt1 = (int(data[reg_id]['bb'][0]), int(data[reg_id]['bb'][1]))
                bb_pt2 = (bb_pt1[0] + int(data[reg_id]['bb'][2]), bb_pt1[1] + int(data[reg_id]['bb'][3]))
                img = cv.imread(filename)
                img = cv.rectangle(img, bb_pt1, bb_pt2, (0, 255, 0), 4)
                print "\n\n\n"
                print "region_id : ", reg_id
                print "category : ", self.categories[str(data[reg_id]['cat'])]
             #   print "alternative candidates : ", self.candidate_words[reg_id] #TODO
                pprint(data[reg_id])
                cv.imshow('test', img)
                c = cv.waitKey(0);
                if c == ord('q'):
                    return


    def analyze_freqs(self):
        with open('resultstoken_freqs.json') as f: #TODO generisch?
            alltokens = json.load(f)
        mean_freq = 0
        counter = 0
        freqs = []
        for val in alltokens.values():
            if val > 2:
                mean_freq += int(val)
                counter += 1
                freqs.append(val)

        mean_freq = mean_freq / float(counter)

        print "control count - in-vocab word number : ", counter  # 2965, vocab without edge and unknown
        print "mean frequency of in-vocab words : ", mean_freq
        print "median :", statistics.median(sorted(freqs))
        for key in alltokens.keys():
            if alltokens[key] in range(55,70):
                print key, alltokens[key]


if __name__ == "__main__":
    a = Analyse()
    a.analyse()
    a.visualize_unknown()
    a.analyze_freqs()