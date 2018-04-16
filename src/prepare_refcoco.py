import pandas as pd
import numpy as np
import nltk
from random import sample
from tqdm import tqdm
import scipy.io
import json
from collections import defaultdict

refdf = pd.read_json("/media/dsgserve1_shkbox/036_object_parts/PreProcOut/refcoco_refdf.json",orient="split",compression="gzip")
with open("/media/dsgserve1_shkbox/036_object_parts/PreProcOut/refcoco_splits.json") as f:
    splits = json.load(f)

splmap = {'val':'val','train':'train','testA':'test','testB':'test'}
file2split = {val:splmap[key] for key in splits for val in splits[key]}

print refdf.head(5)
print splits.keys()
print splits['train'][:10]

obj2phrases = defaultdict(list)
obj2split = {}
split2obj = {'train':[],'test':[]}



for ix,row in refdf.iterrows():
    #phrases = [preprocess_annotation(p.strip()) for p in row['collected_annotation'].split(',')]
    #print phrases
    objectid = (row['image_id'],row['region_id'])
    obj2phrases[objectid].append(row['refexp'].split())
    obj2split[objectid] = file2split[row['image_id']]


print "Objects",len(obj2phrases)


X = np.load("/media/dsgserve1_shkbox/036_object_parts/ExtrFeatsOut/mscoco_vgg19.npz")
X = X['arr_0']

print "X",X.shape
print X[:5]

img_counter = 0
snt_counter = 0
im_list = []
im_mat = []

vgg_mat = np.empty([0,0])

for oitem in tqdm(obj2phrases):
    
    
    vgg_file = X[X[:,1]==oitem[0]]
    vgg_obj = vgg_file[vgg_file[:,2]==oitem[1]]
    
    if len(vgg_obj) > 0:

        im_mat.append(vgg_obj[0])
        
        imdict = {'filename':"_".join([str(oitem[0]),str(oitem[1])]),\
                  'imgid':img_counter,\
                  'split':obj2split[oitem],
                  'sentids':[],
                  'sentences':[]
             }
        for ref in obj2phrases[oitem]:
            imdict['sentids'].append(snt_counter)
            imdict['sentences'].append({'imgid':img_counter,'sentid':snt_counter,'tokens':ref})
            snt_counter += 1
        im_list.append(imdict)
        img_counter += 1



print "make final mat"
vgg_mat = np.array(im_mat)
vgg_mat = vgg_mat[:,3:]
print "shape", vgg_mat.shape
    
vgg_mat =vgg_mat.T

print "dump image matrix"
scipy.io.savemat('/media/dsgserve1_shkbox/036_object_parts/ExtrFeatsOut/refcoco_vgg19_rnnpreproc.mat',{'feats':vgg_mat})

print "dump data set"
dataset = {'images':im_list,'dataset':'refcoco_rnn'}
with open('/media/dsgserve1_shkbox/036_object_parts/PreProcOut/refcoco_refrnn.json', 'w') as f:
    json.dump(dataset,f)
