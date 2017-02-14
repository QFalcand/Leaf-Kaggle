# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:20:44 2017

@author: Quentin PC2
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss

"""
#pour enlever les warnings
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
"""

appr = pd.read_csv('train.csv')
print(appr)
appr = appr.as_matrix()

appr_Y = appr[:, 1]
appr_X = appr[:, 2:]

classifieurs = []
classifieurs.append(MLPClassifier())
classifieurs.append(DecisionTreeClassifier(max_depth = 10))
classifieurs.append(KMeans(99))
classifieurs.append(SVC())

for clf in classifieurs:
    print(type(clf))
#    if type(clf) != KMeans:
    clf.fit(appr_X, appr_Y)
    print(clf.score(appr_X, appr_Y))




