import pandas as pd
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

############ load and prepare image features ###########################

# Image Features  --> numpy npz file includes one key/array, arr_0
extracted_features = np.load("../data/refcoco/mscoco_vgg19_refcoco.npz")['arr_0']
#print "type", type (extracted_features)
#print "extracted_features.shape ",extracted_features.shape # (49865, 4106)
#print "vgg_mat[1][0] ", extracted_features[1][0] # 1.0
#extracted_features = extracted_features[:,3:]
#print "extracted_features.shape ",extracted_features.shape # (49865, 4103)
#print "vgg_mat[1][0] ", extracted_features[1][0] # 0.0

test_list =[]
test_count = 0

img_counter = 0
sentence_counter = 0
selected_img_features = [] #alias img_mat

########### load and prepare referring expressions dataset ##############
refcoco_data = pd.read_json("../data/refcoco/refcoco_refdf.json.gz",orient="split",compression="gzip")
with open("../data/refcoco/refcoco_splits.json") as f:
    splits = json.load(f)
#  TODO  generate own split for zero-shot learning
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

print "Objects",len(obj2phrases)
############ match visual data with referring expressions ###############
############### & set up raw data with splits ###########################

raw_dataset = {
                'train': { 'filenames': list(), 'images': list(), 'captions': list() },
                'val':   { 'filenames': list(), 'images': list(), 'captions': list() },
                'test':  { 'filenames': list(), 'images': list(), 'captions': list() },
            }

# tqdm visualizes progress in the terminal :)
for obj2phrases_item in obj2phrases:  #tqdm(obj2phrases):

    # [:,1] means: all indices of x along the first axis, but only index 1 along the second --> this list comprehension filters out features for one image
    features_for_imageId = extracted_features[extracted_features[:,1] == obj2phrases_item[0]]   #obj2phrases_item[0] is image id
    # this filters out features for the correct region
    features_for_objectId = features_for_imageId[features_for_imageId[:,2] == obj2phrases_item[1]]   #obj2phrases_item[1] is region id

    if len(features_for_objectId) > 0:
        image = np.array(features_for_objectId[0])[3:]  # TODO WHY cut of 3 entries? !! 0 oben??
        test_list.append(np.array(features_for_objectId[0])[3:])
        test_count += 1

        split = obj2split[obj2phrases_item]
        filename = "_".join([str(obj2phrases_item[0]),str(obj2phrases_item[1])])
        caption_group = []
        for ref in obj2phrases[obj2phrases_item]:
            caption_group.append(ref)

        image = image / np.linalg.norm(image)

        raw_dataset[split]['filenames'].append(filename)
        raw_dataset[split]['images'].append(image)
        raw_dataset[split]['captions'].append(caption_group)

        #todo dimension/count of image features and captions correct?
        # dimensionality of features correct? (transpose(transpose)) = no transpose?

print('raw data set',len(raw_dataset['train']['captions']))  #42279

print len(raw_dataset['train']['images']) + len(raw_dataset['val']['images']) + \
      len(raw_dataset['test']['images'])  #should be 49865

print raw_dataset['train']['captions'][0]  # output : [[u'hidden', u'chocolate', u'donut'], [u'space', u'right', u'above', u'game']]
print raw_dataset['train']['captions'][111]  # output : [[u'groom'], [u'groom'], [u'man']]
#TODO why error?

# to compare with original scripts: here, the order is like
# in im_mat from prepare_refcoco.py.
print "count", test_count # 49865
test_list = np.array(test_list)
print test_list.shape
print "test:: ", test_list[1][0] # 0.0729042887688 --> like in original script (random number chosen)
