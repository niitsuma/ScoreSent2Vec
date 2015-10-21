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
 
def mysentscore1(sent):
    mywords=['can', 'could', 'may', 'might', 'must', 'will']
    return  [sent.count(w)/10.0 for w in mywords]

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
brown_sents_label_pre=[[ [sent,g] for sent in nltk.corpus.brown.sents(categories=[g]) ] for g in genres]

brown_sents_labels=[]
for n in range(len(genres)):
    brown_sents_labels.extend(brown_sents_label_pre[n])

sents=[sent_label[0] for sent_label in brown_sents_labels]
sents_scores=[[s,mysentscore1(s)] for s in sents]

sg_v=0
#sg_v=1

modelbrw = Word2Vec(sents , size=100, window=5, sg=sg_v, min_count=5, workers=8)
modelbrw.save('braword.model')
modelbrw.save_word2vec_format('braword.vec')

modelbrs = Sent2Vec(sents, model_file='braword.model',sg=sg_v)
modelbrs.save_sent2vec_format('brasent.vec')

modelbrsc = ScoredSent2Vec(sents_scores, model_file='braword.model',sg=sg_v)
modelbrsc.save_sent2vec_format('brasentsc.vec')


N=len(brown_sents_labels)

X1 =[modelbrs.sents[n] for n in range(N)]
X2 =[modelbrsc.sents[n] for n in range(N)]
Y=[brown_sents_labels[n][1] for n in range(N)]


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

scores= cross_val_score(LogisticRegression(), X1, Y, scoring='accuracy', cv=5)
print scores.mean()

scores = cross_val_score(LogisticRegression(), X2, Y, scoring='accuracy', cv=5)
print scores.mean()


