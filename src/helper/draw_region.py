import pandas as pd
from collections import defaultdict
import cv2 as cv

refcoco_data = pd.read_json("./../../data/refcoco_refdf.json.gz", orient="split", compression="gzip")
bounding_boxes = pd.read_json('../../data/mscoco_bbdf.json.gz', orient="split", compression="gzip")
analysis_dict = defaultdict()

for index, row in refcoco_data.iterrows():
    analysis_dict[str(row['region_id'])] = defaultdict()
    analysis_dict[str(row['region_id'])]['image_id'] = row['image_id']
    analysis_dict[str(row['region_id'])]['refexp_id'] = row['rex_id']

for index, row in bounding_boxes.iterrows():
    if str(row['region_id']) not in analysis_dict.keys():
        analysis_dict[str(row['region_id'])] = defaultdict()
    analysis_dict[str(row['region_id'])]['cat'] = row['cat']
    analysis_dict[str(row['region_id'])]['bb'] = row['bb']



img_id = "219248"
region_id = "274465"

filename = "/mnt/Data/zero_shot_reg/coco-caption/images/train2014/COCO_train2014_" + str(img_id).zfill(12) + ".jpg"
bb_pt1 = (int(analysis_dict[region_id]['bb'][0]), int(analysis_dict[region_id]['bb'][1]))
bb_pt2 = (bb_pt1[0] + int(analysis_dict[region_id]['bb'][2]), bb_pt1[1] + int(analysis_dict[region_id]['bb'][3]))
img = cv.imread(filename)
img = cv.rectangle(img, bb_pt1, bb_pt2, (0, 255, 0), 4)
print "\n\n\n"
cv.imshow('test', img)
c = cv.waitKey(0);