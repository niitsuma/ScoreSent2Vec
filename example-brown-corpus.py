#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence, LineScoredSentence, ScoredSent2Vec
import nltk
import numpy
from random import shuffle


def mysentscore1(sent,scale):
    mywords=['can', 'could', 'may', 'might', 'must', 'will']
    return  [sent.count(w)*scale for w in mywords]

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']

brown_sents_label_pre=[[ [sent,g] for sent in nltk.corpus.brown.sents(categories=[g]) ] for g in genres]

brown_sents_labels=[]
for n in range(len(genres)):
    brown_sents_labels.extend(brown_sents_label_pre[n])




def test_sentvec(sg_v,scale):
    sents_scores_labels=[[sl[0],mysentscore1(sl[0],scale),sl[1]] for sl  in brown_sents_labels]
    shuffle(sents_scores_labels) ##avoid affect orderd label

    sents_scores=[[scl[0],scl[1]] for scl in sents_scores_labels]
    sents=[scl[0] for scl in sents_scores_labels]

    modelbrw = Word2Vec(sents , size=100, window=5, sg=sg_v, min_count=5, workers=8)
    modelbrw.save('braword.model')
    modelbrw.save_word2vec_format('braword.vec')

    modelbrsc = ScoredSent2Vec(sents_scores, model_file='braword.model',sg=sg_v) 
    modelbrsc.save_sent2vec_format('brasentsc.vec')

    modelbrs = Sent2Vec(sents, model_file='braword.model',sg=sg_v)
    modelbrs.save_sent2vec_format('brasent.vec')



    N=len(brown_sents_labels)
    X1 =[modelbrs.sents[n] for n in range(N)]
    X2 =[modelbrsc.sents[n] for n in range(N)]
    X3 =[numpy.append(modelbrs.sents[n],sents_scores[n][1])  for n in range(N)] ##Concatenate
    Y=[scl[2] for scl in sents_scores_labels]

    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import cross_val_score

    scores= cross_val_score(LogisticRegression(), X1, Y, scoring='accuracy', cv=5)
    ret1=scores.mean()

    scores = cross_val_score(LogisticRegression(), X2, Y, scoring='accuracy', cv=5)
    ret2=scores.mean()

    scores = cross_val_score(LogisticRegression(), X3, Y, scoring='accuracy', cv=5)
    ret3=scores.mean()

    result=[sg_v,scale,ret1,ret2,ret3]
    print result

    return result




from sklearn.grid_search import ParameterSampler, ParameterGrid
#params={'sg_v':[0,1],'scale':[2.0**(-n) for n in range(20)]}
#params={'sg_v':[1],'scale':[2.0**(-n) for n in range(20)]}
#params={'sg_v':[1],'scale':[2.0**(n+1) for n in range(20)]}
params={'sg_v':[0,1],'scale':[1.0]}
print params
param_list = list(ParameterGrid(params))
print param_list

result=[test_sentvec(**param) for param in param_list]
print result
