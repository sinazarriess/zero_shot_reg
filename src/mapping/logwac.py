import pandas as pd
import numpy as np
import gzip
import json
from tqdm import tqdm
from collections import Counter
import sys
from scipy import stats
from sklearn import linear_model







def train_logreg(this_X,this_y):

    reg = linear_model.LogisticRegression(penalty='l1')
    try: 
        print("...training")
        reg.fit(this_X,this_y)
    except:
        print("Could not train")
        reg = None

    return reg



def train_names():

    refdf_path = "refcoco_refdf_heads.json.gz"
    refdf = pd.read_json(refdf_path,compression='gzip', orient='split')

    X_path = "/Volumes/SHK-DropBox/036_object_parts/ExtrFeatsOut/mscoco_vgg19_refcoco.npz"
    X = np.load(X_path)['arr_0']

    cats = set(list(refdf.cat))

    for cat in cats:

        wac = {}
        traindf = refdf[refdf.cat != cat]
        trainvocab = []
        for name_l in traindf.names:
            trainvocab += name_l
        trainvocab = list(set(trainvocab))
        print("Names:",len(trainvocab))

        if cat == 2:
            for name in trainvocab:
                print("Name:",name)
                wac[name] = train_name(X,traindf,name)

def train_name(X,traindf,name):

    nneg = 25000

    train_index = np.array(traindf.ix_Xfile)

    pos_instances = np.array(traindf.names.apply(lambda x: name in x))
    neg_instances = np.invert(pos_instances)

    pos_index = train_index[pos_instances]
    neg_index = train_index[neg_instances]
    if len(neg_index) > nneg:
        neg_index = np.random.choice(neg_index,nneg,replace=False)

    y = np.hstack([np.ones(pos_index.shape), np.zeros(neg_index.shape)])
    Xindex = np.hstack([pos_index, neg_index])
    Xtrain = X[Xindex]


    return train_logreg(Xtrain,y)




if __name__ == '__main__':

    train_names()