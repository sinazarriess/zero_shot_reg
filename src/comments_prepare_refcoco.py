import pandas as pd
import json
from collections import defaultdict
import numpy as np
import tqdm
import scipy

refcoco_data = pd.read_json("../data/refcoco/refcoco_refdf.json.gz",orient="split",compression="gzip")
with open("../data/refcoco/refcoco_splits.json") as f:
    splits = json.load(f)

splitmap = {'val':'val','train':'train','testA':'test','testB':'test'}
# for every group in split --> for every entry --> make entry in new dict
# file2split just translates testA and testB to "test"?
new_split_dict = {val:splitmap[key] for key in splits for val in splits[key]}

# dict of objectids and ref exps
obj2phrases = defaultdict(list)
# dict of objectids and split (train,test or val)
obj2split = {}
split2obj = {'train':[],'test':[]}

# iterate over json "entries"
for index, row in refcoco_data.iterrows():
    # id is tuple of image and region id
    objectid = (row['image_id'], row['region_id'])
    obj2phrases[objectid].append(row['refexp'].split())
    obj2split[objectid] = new_split_dict[row['image_id']]

# Image Features  --> numpy npz file includes one key/array, arr_0
npz_file = np.load("../data/refcoco/mscoco_vgg19_refcoco.npz")
extracted_features = npz_file['arr_0']

img_counter = 0
snt_counter = 0
ref_img_list = []
img_mat = []

vgg_mat = np.empty([0,0])

# tqdm visualizes progress in the terminal :)
for obj2phrases_item in tqdm(obj2phrases):

    # [:,1] means: all indices of x along the first axis, but only index 1 along the second
    # this list comp. filters out features for one image
    features_for_imageId = extracted_features[extracted_features[:,1] == obj2phrases_item[0]]   #obj2phrases_item[0] is image id
    # this filters out features for the correct region
    features_for_objectId = features_for_imageId[features_for_imageId[:,2] == obj2phrases_item[1]]   #obj2phrases_item[1] is region id

    if len(features_for_objectId) > 0:
        img_mat.append(features_for_objectId[0])

        ref_img_dict = {'filename':"_".join([str(obj2phrases_item[0]),str(obj2phrases_item[1])]),\
                  'imgid':img_counter,\
                  'split':obj2split[obj2phrases_item],
                  'sentids':[],
                  'sentences':[]
             }
        for ref in obj2phrases[obj2phrases_item]:
            ref_img_dict['sentids'].append(snt_counter)
            ref_img_dict['sentences'].append({'imgid':img_counter,'sentid':snt_counter,'tokens':ref})
            snt_counter += 1
        ref_img_list.append(ref_img_dict)
        img_counter += 1

# make final image features mat
vgg_mat = np.array(img_mat)
vgg_mat = vgg_mat[:, 3:]
vgg_mat = vgg_mat.T

# dump image matrix
scipy.io.savemat('/media/compute/vol/dsg/lilian/testrun/refcoco_vgg19_rnnpreproc.mat', {'feats': vgg_mat})

# dump ref exp data set
dataset = {'images': ref_img_list, 'dataset': 'refcoco_rnn'}
with open('/media/compute/vol/dsg/lilian/testrun/refcoco_refrnn.json', 'w') as f:
    json.dump(dataset, f)












