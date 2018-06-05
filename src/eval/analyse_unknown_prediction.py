import json
from collections import defaultdict
import pandas as pd
from sets import Set

with open('./jsons/test.json', "r") as f:
    reference = json.load(f)

with open('./model/without_unknown/inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json') as f:
    without_unknown_candidate = json.load(f)

with open('./model/inject_refcoco_refrnn_compositional_3_512_1/4evalinject_refcoco_refrnn_compositional_3_512_1.json') as f:
    unknown_candidate = json.load(f)


unknown_counter = 0
print "Number of images in test set: ", len(reference.keys())

analysis_dict = defaultdict()

for refex_id in unknown_candidate:
    comparison = defaultdict()
    if "UNKNOWN" in unknown_candidate[refex_id]:
        unknown_counter += 1
        comparison['reflist'] = reference[refex_id]
        comparison['original generated caption'] = without_unknown_candidate[refex_id]
        comparison['with unknown'] = unknown_candidate[refex_id]
        analysis_dict[refex_id] = comparison

print "Number of unknown occurences: ", unknown_counter

with open('./jsons/unknown_analysis.json', 'w') as f:
    json.dump(analysis_dict, f)


images = set()
refcoco_data = pd.read_json("./../../data/refcoco_refdf.json.gz", orient="split", compression="gzip")

regionids = list()
for id in analysis_dict.keys():
    regionids.append(str(id))
print regionids

for index, row in refcoco_data.iterrows():
    if str(row['region_id']) in regionids:
        images.add(row['image_id'])

print len(images)