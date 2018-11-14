import pandas as pd
import numpy as np
import gzip
import json
from tqdm import tqdm
from collections import Counter
import sys
from scipy import stats
from sklearn import linear_model
import multiprocessing as mp






def train_logreg(this_X,this_y):

    reg = linear_model.LogisticRegression(penalty='l1',solver='liblinear')
    #reg = linear_model.LogisticRegression(penalty='l1',solver='saga')
    try: 
#        print("...training")
        reg.fit(this_X,this_y)
    except:
        print("Could not train")
        reg = None

    return reg



def train_names(X,refdfd,cat):



    wac = {}
    traindf = refdf[refdf.cat != cat]
    trainvocab = []
    for name_l in traindf.names:
        trainvocab += name_l
    trainvocab = list(set(trainvocab))
    print("Names:",len(trainvocab))

    for name in tqdm(trainvocab):
        #print("Name:",name)
        wac[name] = train_name(X,traindf,name)

    testdf = refdf[refdf.cat == cat]
    names,predictions = apply_wac(X,testdf,wac)
    #print("Predictions",predictions.shape)

    topn = 10

    word_matrix = np.column_stack(1-predictions)
    word_sort = np.argsort(word_matrix,axis=1)
    word_sort = word_sort[:,:topn]
    #print("Wordsort",word_sort.shape)
    
    region2predictions = {}
    for x in range(len(testdf)):
        for y in range(topn):
            region = testdf.iloc[x]['region_id']
            topn_region = word_sort[x]
            topn_words = [(names[windex],"%.6f"%(1-word_matrix[x][windex])) for windex in topn_region]
            region2predictions[str(region)] = topn_words

    #print(region2predictions)


    with open('wacexp/wac_with_reduced_cat_'+str(cat)+".json",'w') as f:
       json.dump(region2predictions,f)

    return True

def apply_wac(X,testdf,wac):

    test_index = np.array(testdf.ix_Xfile)
    Xtest = X[test_index]
    Xtest = Xtest[:,3:]

    words = wac.keys()
    word_probs = []
    for name in words:
        probs = wac[name].predict_proba(Xtest)[:,1]
        word_probs.append(probs)
    return words,np.array(word_probs)    

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
    Xtrain = Xtrain[:,3:]

    return train_logreg(Xtrain,y)




if __name__ == '__main__':
    refdf_path = "refcoco_refdf_heads.json.gz"
    refdf = pd.read_json(refdf_path,compression='gzip', orient='split')

    #X_path = "/Volumes/SHK-DropBox/036_object_parts/ExtrFeatsOut/mscoco_vgg19_refcoco.npz"
    X_path = "/media/dsgshk/036_object_parts/ExtrFeatsOut/mscoco_vgg19_refcoco.npz"
    X = np.load(X_path)['arr_0']


    cats = list(set(refdf.cat))
    for ix in range(2,len(cats),3):
        cat1 = cats[ix]
        if ix+2 < len(cats):
            cat3 = cats[ix+2]
            cat2 = cats[ix+1]
            print("cats:",cat1,cat2,cat3)
    	    processes = [mp.Process(target=train_names,args=(X,refdf,cat1)),\
                    mp.Process(target=train_names,args=(X,refdf,cat2)),\
                    mp.Process(target=train_names,args=(X,refdf,cat3))]
        elif ix+1 < len(cats):
            cat2 = cats[ix+1]
    	    processes = [mp.Process(target=train_names,args=(X,refdf,cat1)),\
                    mp.Process(target=train_names,args=(X,refdf,cat2))]
        else:
    	    processes = [mp.Process(target=train_names,args=(X,refdf,cat1)),\
                    mp.Process(target=train_names,args=(X,refdf,1))]


        for p in processes:
            p.start()
        for p in processes:
            p.join()

