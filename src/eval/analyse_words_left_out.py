import ast
import json
import utils
from collections import defaultdict
import pandas as pd
import cv2 as cv
from pprint import pprint

model_path = 'model/with_reduced_vocab/'


class Analyse():
    def __init__(self):
        self.analysis_dict = defaultdict()
        self.images = set()
        self.regionids = list()

        with open(model_path + 'refs_moved_to_test.json', 'r') as f:
            ids = f.readline()
            self.extra_items_list = ast.literal_eval(ids)
        print "Number of regions put into extra test set: ", len(self.extra_items_list)

        with open(model_path + 'inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json') as f:
            self.generated_captions = json.load(f)

        with open(model_path + 'test.json', "r") as f: # TODO Achtung
            self.reference = json.load(f)

        self.categories = utils.read_in_cats()
        self.refcoco_data = pd.read_json("./../../data/refcoco_refdf.json.gz", orient="split", compression="gzip")
        self.bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")

        print "Number of images in test set: ", len(self.reference.keys())

    def parse(self):
        unknown_counter = 0
        for refex_id in self.generated_captions:
            comparison = defaultdict()
            if refex_id in self.extra_items_list:
                if "UNKNOWN" in self.generated_captions[refex_id]:
                    unknown_counter += 1
                comparison['reflist'] = self.reference[refex_id]
                # comparison['original generated caption'] = self.without_unknown_candidate[refex_id]
                comparison['with reduced vocab'] = self.generated_captions[refex_id]
                self.analysis_dict[refex_id] = comparison

        print "Number of unknown occurences in subset: ", unknown_counter

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
        with open(model_path + 'reducedvocab_analysis.json', 'w') as f:
            json.dump(self.analysis_dict, f)

 #       dict4eval = defaultdict()
#        for reg_id in self.analysis_dict.keys():
#            dict4eval[reg_id] = self.analysis_dict[reg_id]['original generated caption']
 #       with open('./jsons/unknown_analysis_eval_original.json', 'w') as f:
#            json.dump(dict4eval, f)


    def visualize_zeroshot(self, region_id=-1):
        with open(model_path + 'reducedvocab_analysis.json', 'r') as f:
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
                pprint(data[reg_id])
                cv.imshow('test', img)
                c = cv.waitKey(0);
                if c == ord('q'):
                    return


if __name__ == "__main__":
    a = Analyse()
    a.parse()
    a.add_additional_info()
    a.save()
    a.visualize_zeroshot()