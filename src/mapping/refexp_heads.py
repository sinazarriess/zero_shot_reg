
import json
import os
import numpy as np
import pandas as pd
import gzip
from collections import Counter,defaultdict
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec




# some words which are often tagged as nouns, but we know that they are not heads/object names
NOHEAD = ['left','right','middle','center','corner','front','rightmost','leftmost','back',\
         'blue','red','white','purple','green','gray','grey','brown','blond','black','yellow',\
         'tall','standing','empty','tan','bald','silver','leftest','far','farthest','furthest',\
         'whole','lol','sorry','tallest','nearest','round','taller','righter','rightest','background',\
          'top','okay','pic','part','bottom','click','mid','closest','teal','hot','wooden',\
         'short','shortest','bottm','topmost','dirty','spiders','wtf','skinny','fat','sexy','jumping',
         'frontmost','yes','wearing','lil','creepy','serving','beside','beige','upper','lower','side',
         'facing','blurry','color','colour','dark','pink','foreground','facing','holding','forefront',\
         'second','third','row','image','ok','hi','thanks','botom','towards','photo']


#NOUNS = open('noun_list_long.txt').readlines()

RELUNI = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'by',
            'near',
            'with',
            'at',
            'that',
            'who',
            'beside',
            'besides']

RELOF = ['front', 'right', 'left', 'ontop', 'top', 'middle','side','out']

RELTO = ['next', 'close', 'closest']

MIN_FREQ = 5

print('load glove')
glove_model = KeyedVectors.load_word2vec_format("glove.6B/glove.6B.300d.w2v.bin")
print('done')

print('load X')
X_path = "/Volumes/SHK-DropBox/036_object_parts/ExtrFeatsOut/mscoco_vgg19_refcoco.npz"
X = np.load(X_path)['arr_0']
print('done')
region2ix = {int(regionid):xix for xix,regionid in enumerate(X[:,2])}
print(list(region2ix.items())[:20])

refdf_path = "/Volumes/SHK-DropBox/036_object_parts/PreProcOut/refcoco_refdf.json.gz"
fulldf = pd.read_json(refdf_path,compression='gzip', orient='split')


WORDS = Counter(' '.join(list(fulldf['refexp'])).split())

# pattern-pos-based hack
# that identifies the head of a refex
def get_head_noun(tagged):
    
    for (word,pos) in tagged:
        if len(word) > 1 and \
         not word in NOHEAD and \
         WORDS[word] > 1:
            if pos == 'NN':
                return word
            if pos == 'NNS':
                return word
            #if word in NOUNS:
            #    return word
    #print tagged
    return ""

# getting the index of the relational prep.
# if refex is relational at all
def get_rel_index(tagged):
    for ix,(word,_) in enumerate(tagged):
        if word in RELUNI:
            return ix
        if word in RELOF:
            if ix+1 < len(tagged):
                if tagged[ix+1][0] == 'of':
                    return ix+1

        if word in RELTO:
            if ix+1 < len(tagged):
                if tagged[ix+1][0] == 'to':
                    return ix+1

    return -1


fulldf['relindex'] = fulldf['tagged'].apply(lambda x: get_rel_index(x))

# looking for heads
headlist1 = []
HEADS = Counter()
for x,row in fulldf.iterrows():

    # if taggers disagree, we simply collect both heads
    if row['relindex'] > -1:
        refex1 = row['tagged'][:row['relindex']]
    else:
        refex1 = row['tagged']
    #print x,row['relindex'],
    #print refex
    h1 = get_head_noun(refex1)
    
    headlist1.append(h1)
    if len(h1) > 0:
        HEADS[h1] += 1

assert len(headlist1) == len(fulldf)
freqheads = [h for h in HEADS if HEADS[h] > MIN_FREQ and h in glove_model]
print("frequent heads",len(freqheads))

final_headlist = []
for x,row in fulldf.iterrows():

    if (headlist1[x] in freqheads) and (headlist1[x] != ""):
        final_headlist.append(headlist1[x])

    else:
        words = row['refexp'].split()
        words = words[:row['relindex']]
        new_head = ""
        for w in words:
            if w in freqheads:
                #print(words)
                #print("head found",w)
                new_head = w
                break
        final_headlist.append(new_head)

VOCAB = list(set(final_headlist))
print("VOCAB",len(VOCAB))

fulldf['head'] = final_headlist

bbdf_path = "/Volumes/SHK-DropBox/036_object_parts/PreProcOut/mscoco_bbdf.json.gz"
bbdf = pd.read_json(bbdf_path,compression='gzip', orient='split')

cats = open('cats_glove.txt','r').readlines()
id2cat = {}
for line in cats:
    (cid,cname) = line.strip().split(' , ')
    id2cat[int(cid)] = cname.strip()

region2cat = {rid:id2cat[cid] for (rid,cid) in zip(bbdf.region_id,bbdf.cat)}

region2names = {}
region2refexp = {}
for ix, row in fulldf.iterrows():
    if row['region_id'] not in region2names:
        region2names[row['region_id']] = []
        region2refexp[row['region_id']] = []
    if row['head'] != "":
        region2names[row['region_id']].append(row['head'])
        region2refexp[row['region_id']].append(row['refexp'])

    
    cat_name = region2cat[row['region_id']]
    if cat_name in glove_model:
        region2names[row['region_id']].append(cat_name)
        if cat_name not in VOCAB:
            VOCAB.append(cat_name)


region2image = {rid:imid for (rid,imid) in zip(bbdf.region_id,bbdf.image_id)}
region2catid = {rid:cid for (rid,cid) in zip(bbdf.region_id,bbdf.cat)}

newdf = []
for some_r in region2names:
    if len(region2names[some_r]) > 0:
        if some_r in region2ix:
            namelist = list(set(region2names[some_r]))
            some_tup = [1,region2image[some_r],some_r,namelist,region2refexp[some_r],region2catid[some_r],region2ix[some_r]]
            newdf.append(some_tup)

newdf = pd.DataFrame(newdf,columns=['i_corpus','image_id','region_id','names','refexps','cat','ix_Xfile'])

newdf.to_json('refcoco_refdf_heads.json.gz',compression='gzip', force_ascii=False, orient='split')

w2vfile = open("glove.6B/glove.6B.300d.heads.txt","w")
for w in VOCAB:
    if w != "":
        print(w,' '.join([str(i) for i in glove_model[w]]),file=w2vfile)
w2vfile.close()

glove2word2vec("glove.6B/glove.6B.300d.heads.txt", "glove.6B/glove.6B.300d.heads.w2v.bin")
