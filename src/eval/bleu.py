import cPickle as pickle
import os
import sys
import json
sys.path.append('../coco-caption')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores
    

def evaluate(reference_path, candidate_path, doPrint, split='val'):
    
    # load caption data
    with open(reference_path, 'rb') as f:
        ref = json.load(f)
    with open(candidate_path, 'rb') as f:
        cand = json.load(f)

    if doPrint:
        print "ref keys",len(ref), ref.keys()[:10]
        print "cand keys",len(cand), cand.keys()[:10]
        print set(ref.keys())-set(cand.keys())
    sortkeys = sorted(ref.keys())
    refsort = {k:ref[k] for k in sortkeys}
    candsort = {k:cand[k][:1] for k in sortkeys}
    # compute bleu score
    final_scores = score(refsort, candsort)

    if doPrint:
    # print out scores
        print 'Bleu_1:\t',final_scores['Bleu_1']
        print 'Bleu_2:\t',final_scores['Bleu_2']
        print 'Bleu_3:\t',final_scores['Bleu_3']
        print 'Bleu_4:\t',final_scores['Bleu_4']
        print 'METEOR:\t',final_scores['METEOR']
        print 'ROUGE_L:',final_scores['ROUGE_L']
        print 'CIDEr:\t',final_scores['CIDEr']
    
    
    return final_scores
    
   
    
    
    
    
    
    
    
    
    
    


