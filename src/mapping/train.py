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

    print(testdf.head(n=10))

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

    return ((Xtrain,Ytrain,traindf),(Xtest,Ytest,testdf),glove_model)


def train_mappings(X,Y):

    batch_size = 50
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

    model.fit(X, Y,batch_size=batch_size, epochs=epochs,verbose=1,validation_split=0.02)

    return model

def test_mappings(model,X,testdf,glove_model):

    acc = {1:0,2:0,5:0,10:0}
    acc_2 = 0
    acc_5 = 0
    acc_10 = 0

    print("Test input:",X.shape)
    pred = model.predict(X)
    print("Predictions:",pred.shape)

    for ix in range(pred.shape[0]):
        pred_names = [w for (w,sim) in glove_model.similar_by_vector(pred[ix],topn=10)]
        gold_names = set(testdf.iloc[ix]['names'])

        for t in [1,2,5,10]:
            t_pred = set(pred_names[:t])
            if len(gold_names & t_pred) > 0:
                acc[t] += 1

        if ix < 15:
            print("prediction:",pred_names)
            print("actual names:",gold_names)

    print(acc)
    for t in acc:
        print("Acc@%d:%.2f"%(t,acc[t]/X.shape[0]))

    return acc


if __name__ == '__main__':

    train,test,glove = load(zero_cat=73)
    zeroshot_m1 = train_mappings(train[0],train[1])
    test_mappings(zeroshot_m1,test[0],test[2],glove)


