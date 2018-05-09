
import numpy as np
#import nltk
#from random import sample
from tqdm import tqdm
import scipy.io
import json
import pandas as pd
from collections import defaultdict

refdf = pd.read_json("../data/refcoco/refcoco_refdf.json.gz",orient="split",compression="gzip")
with open("../data/refcoco/refcoco_splits.json") as f:
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


X = np.load("../data/refcoco/mscoco_vgg19_refcoco.npz")
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
print "vgg_mat[1][0]", vgg_mat[1][0]  # 1.0
vgg_mat = vgg_mat[:,3:]
print "shape", vgg_mat.shape #(49865, 4103)
print "vgg_mat[1][0]", vgg_mat[1][0]  # 0.0729042887688
    
vgg_mat =vgg_mat.T
print "shape after t", vgg_mat.shape #(4103, 49865)
print "vgg_mat[1][0]", vgg_mat[1][0] # 0.0

print "dump image matrix"
scipy.io.savemat('/media/compute/vol/dsg/lilian/testrun/refcoco_vgg19_rnnpreproc.mat',{'feats':vgg_mat})

print "dump data set"
dataset = {'images':im_list,'dataset':'refcoco_rnn'}
with open('/media/compute/vol/dsg/lilian/testrun/refcoco_refrnn.json', 'w') as f:
    json.dump(dataset,f)
