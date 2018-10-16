import pandas as pd
import numpy as np
import gzip
import json
from tqdm import tqdm
import scipy.stats
from collections import Counter
import sys
import random
import yaml
from scipy.spatial.distance import pdist, squareform
import sys
from sklearn import linear_model
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation



def get_target_vecs(glove,names):

    print ()

    Y = []
    for namelist in names:
        #print(namelist)
        vec = np.mean([glove[n] for n in namelist if n in glove],axis=0)
        Y.append(vec)
        #print(vec)
    Y = np.array(Y)
    print("Y",Y.shape)

    return Y

def load(zero_cat=1):

    refdf_path = "refcoco_refdf_heads.json.gz"
    refdf = pd.read_json(refdf_path,compression='gzip', orient='split')

    testdf = refdf[refdf.cat==zero_cat]
    traindf = refdf[refdf.cat!=zero_cat]

    glove_model = KeyedVectors.load_word2vec_format("glove.6B/glove.6B.300d.heads.w2v.bin")

    Ytrain = get_target_vecs(glove_model,list(traindf.names))
    Ytest = get_target_vecs(glove_model,list(testdf.names))

    X_path = "/Volumes/SHK-DropBox/036_object_parts/ExtrFeatsOut/mscoco_vgg19_refcoco.npz"
    X = np.load(X_path)['arr_0']
    print(X.shape)
    

    train_index = list(traindf.ix_Xfile)
    Xtrain = X[train_index][:,3:]

    test_index = list(testdf.ix_Xfile)
    Xtest = X[test_index][:,3:]

    print("Xtrain",Xtrain.shape)
    print("Xtest",Xtest.shape)

    return ((Xtrain,Ytrain),(Xtest,Ytest))


def train_mappings(X,Y):

    batch_size = 32
    epochs = 5


    model = Sequential()
    model.add(Dense(1024, input_shape=(X.shape[1],)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1]))
    #model.add(Activation('relu'))

    model.compile(loss='cosine_proximity',\
              optimizer='adam',\
              metrics=['mae'])

    model.fit(X, Y,batch_size=batch_size, epochs=epochs,verbose=1,validation_split=0.1)



if __name__ == '__main__':

    train,test = load(zero_cat=1)
    train_mappings(train[0],train[1])


