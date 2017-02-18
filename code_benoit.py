# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:40:24 2017

@author: Utilisateur
"""

import pandas as pd
import random as rd
from sklearn.metrics import *
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.externals import joblib

def reduction_dim(X):
    '''ACP et projection dans un ensemble à moins de dimensions. Retourne X
    projettée dans le nouvel espace'''

    col = ['Feature', 'dim1', 'dim2', 'dim3']
    df = pd.DataFrame([], columns = col)
    
    test = True
    for x in X:
        
        pca = PCA(n_components = 3)
        pca.fit(x[1])
        
        a = [x[0]] + list(pca.explained_variance_ratio_)
        
        d = pd.DataFrame([a], columns = col)
        df = df.append(d, ignore_index = True)
        
        xtransform = pca.fit_transform(x[1])
        
        if test:
            Xprim = xtransform
        else:
            Xprim = np.c_[Xprim, xtransform]
    
        test = False
    
    print(df)
    
    return Xprim      
                 
def test_classifieurs(skf, classifieurs, X, Y, nCV):
    '''Teste et compare différents classifieurs. Les classifieurs sont d'abord entraînés
    et testés par cross validation'''
    
    col = ['Classifieurs', 'Fmesure', 'precision', 'recall', 'acc']
    
    test = True
    compteur = 0
    for train_index, test_index in skf.split(X, Y):
        
        compteur += 1
        print(compteur)
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    
        t = True
        for cf in classifieurs:
    
            clf = cf[1].fit(X_train, Y_train)
            train_prediction = clf.predict(X_test)
            
            acc = accuracy_score(Y_test, train_prediction)
            Fmesure = f1_score(Y_test, train_prediction, average = 'macro')
            precision = precision_score(Y_test, train_prediction, average = 'macro')
            recall = recall_score(Y_test, train_prediction, average = 'macro') 
            
            if t:
                df = pd.DataFrame([[cf[0]]+[Fmesure, precision, recall, acc]], columns = col)
            else:
                df = df.append(pd.DataFrame([cf[0]+[Fmesure, precision, recall, acc]], columns = col), ignore_index = True)
                
            t = False
            
            if compteur == 1:
                joblib.dump(clf, cf[0]+'.pkl') #Enregistrement du classifieur
        
        if test:
            results = df.copy(deep = True)
        else:
            for c in list(results.columns[1:]):
                results[c] = results[c] + df[c]
        
        test = False
    
    for c in list(results.columns[1:]):
        results[c] = results[c]/nCV
    
    print(results)
    
def shuffle(X, Y):
    '''Retourne les matrices X et Y avec les lignes mélangées'''
    
    indices = [i for i in range(0, X.shape[0])] #X.shape[0] : nombre de lignes de X
    
    rd.shuffle(indices)
    
    X_new = X[[indices]]
    Y_new = Y[[indices]]

    return(X_new, Y_new)

def etude_MLP(skf, X, Y, nCV, n_iter):
    '''Etude des MLP : suivi erreur en test et erreur en apprentissage selon le nombre
    d'époques. Plot un graphique'''
    
    col = ['iter','Train', 'Test']
    test= True
    
    compteur = 0
    for train_index, test_index in skf.split(X, Y):
        
        compteur += 1
        print(compteur)
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        sample = {'train':[X_train, Y_train], 'test':[X_test, Y_test]}

        clf = MLPClassifier(max_iter = 10, learning_rate_init = 0.01)
        classes = np.unique(Y_train)
        
        acc_plot = {'train':[], 'test':[]}
        
        t = True
        for i in range(0, n_iter, 10):
            
            X_train, Y_train = shuffle(X_train, Y_train) #On re-mélange les exemples
            X_test, Y_test = shuffle(X_test, Y_test)
            
            clf = clf.partial_fit(X_train, Y_train, classes)
            
            acc = [i]
            for samp in ['train', 'test']:
                prediction = clf.predict(sample[samp][0])
                acc.append(accuracy_score(sample[samp][1], prediction))
                acc_plot[samp].append(accuracy_score(sample[samp][1], prediction))
                
            if t:
                df = pd.DataFrame([acc], columns = col)
            else:
                df = df.append(pd.DataFrame([acc], columns = col), ignore_index = True)
                
            t = False
        
        if test:
            results = df.copy(deep = True)
        else:
            df = pd.DataFrame([acc], columns = col)
            for c in list(results.columns[1:]):
                results[c] = results[c] + df[c]

        #Courbe d'erreur en test et en apprentissage pour chaque classifieur        
        fig, ax = plt.subplots()
        ax.plot(range(0, n_iter, 10), acc_plot['train'], label = 'Train')
        ax.plot(range(0, n_iter, 10), acc_plot['test'], c = 'r', label = 'Test')
        plt.title('MLP'+str(compteur), fontsize = 'medium')
        plt.xlabel('Epochs number', fontsize = 'small')
        plt.ylabel('Accuracy', fontsize = 'small')
        plt.legend(fontsize = 'x-small', loc = 'center right')
        plt.savefig('MLP'+str(compteur)+'.pdf', bbox_inches='tight')
        plt.show()

        joblib.dump(clf, 'clf'+str(compteur)+'.pkl') #Enregistrement du classifieur
        
        test = False
    
    for c in list(results.columns[1:]):
        results[c] = results[c]/nCV
    df.to_csv('results.csv')
    print(results)
    
    #Courbe d'erreur en test et en apprentissage moyennée
    plt.plot(range(0, n_iter, 10), list(results['Train']))
    plt.plot(range(0, n_iter, 10), list(results['Test']), c = 'r')
    plt.show()
    
def compare_results(classifieurs, X_test, Y_test, cumul = False):
    '''Compare les résultats en prédictions de plusieurs classifieurs DEJA entraînés.
    Dans le cas de cumul = True, on additionne les probabilités de sorties pour 
    chaque classifieur et on regarde la prédiction de l'ensemble'''
    
    if cumul:
        Sorties = np.zeros((X_test.shape[0], 99))
    col = ['Classifieur', 'acc']
    res = pd.DataFrame([], columns = col)
    
    for clf in classifieurs:
        
        prediction = clf[1].predict(X_test)
        acc = accuracy_score(Y_test, prediction)
        df = pd.DataFrame([[clf[0], acc]], columns = col)
        res = res.append(df, ignore_index = True)
        
        if cumul:
            Sorties = Sorties + clf[1].predict_proba(X_test)

    if cumul:
        predic_cumulees = np.copy(Y_test)
        for ligne in range(0, Sorties.shape[0]):
            ind_max = np.argmax(Sorties[ligne])
            predic_cumulees[ligne] = clf[1].classes_[ind_max]
        acc = accuracy_score(predic_cumulees, Y_test)
        df = pd.DataFrame([['Proba cumulees', acc]], columns = col)
        res = res.append(df, ignore_index = True)
    
    print(res)
    
def generation_train_test(X, Y):
    '''Génère un ensemble de test et un un ensemble d'apprentissage à partir
    d'un échantillon donné en entrée, avec conservation du pourcentage d'exemples
    de chaque classe. Ensemble de test = 10% du départ'''
    
    skf = StratifiedShuffleSplit(n_splits = 1, test_size=0.1)
    
    list_ind = list(skf.split(X, Y))
    
    train_ind = list_ind[0][0]
    test_ind = list_ind[0][1]

    X_train, X_test = X[train_ind], X[test_ind]
    Y_train, Y_test = Y[train_ind], Y[test_ind]

    return X_train, Y_train, X_test, Y_test
    
def etude_SVM(skf, X, Y, nCV):
    '''Etude des SVM, faire varier C'''
    
    col = ['C', 'acc', 'n_pointsupp']

    compteur = 0
    test = True
    for train_index, test_index in skf.split(X, Y):
        
        compteur += 1
        print(compteur)
        
        t = True
        for C in [1,1000,10000,20000,50000]:
        
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
    
            clf = svm.SVC(decision_function_shape='ovo', C = C)
            clf = clf.fit(X_train, Y_train)
            
            train_prediction = clf.predict(X_test)
            acc = accuracy_score(Y_test, train_prediction)
            n_pointsupp = np.mean(clf.n_support_)
            
            if t:
                df = pd.DataFrame([[C, acc, n_pointsupp]], columns = col)
                      
            else:
                df = df.append(pd.DataFrame([[C, acc, n_pointsupp]], columns = col), ignore_index = True)
            
            
            if C == 50000:
                joblib.dump(clf, 'clf'+str(compteur)+'.pkl') #Enregistrement du classifieur
                
            t = False
        
        if test:
            results = df.copy(deep = True)
            
        else:
            for c in list(results.columns[1:]):
                results[c] = df[c] + results[c]

        test = False
   
    for col in list(results.columns[1:]):
        results[col] = results[col]/nCV

    print(results)
    
def prediction(classifieurs, X):
    '''Retourne les prédictions de différents classifieurs pour des exemples dont on ne connaît 
    pas la classe'''
    
    col = []
    for c in classifieurs:
        col.append(c[0])
    
    test = True
    for clf in classifieurs:
        
        prediction = clf[1].predict(X)
        
        if test:
            results = pd.DataFrame({clf[0]:list(prediction)}, columns = [clf[0]])
        else:
            results.loc[:,clf[0]] = pd.Series(list(prediction), index = results.index)
            
    print(results)
        
    
if __name__ == '__main__':
    
    '''Ouverture et lecture des fichiers'''
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    t1 = train.as_matrix()
    t2 = test.as_matrix()

    
    '''Réduction de dimension'''
#    X = [['margins', t1[:,2:65]], ['shapes', t1[:,66:129]], ['textures', t1[:,130:194]]]
#    Y = t1[:,1]
#    X = reduction_dim(X)
    
    '''Pas de réduction de dimension'''
    X = t1[:,2:]
    Y = t1[:,1]
    Xapredire = t2[0,1:]

    '''Découpage en un ensemble d'apprentissage et un ensemble de test. On fera de la
    cross-validation sur l'ensemble d'apprentissage, et on testera ensuite les classifieurs 
    appris sur l'ensemble de test'''
    X_train, Y_train, X_test, Y_test = generation_train_test(X, Y)
   
    '''Comparaison de différents classifieurs'''
#    classifieurs = [['Arbre de décision', tree.DecisionTreeClassifier(max_depth = 10)],
#                ['SVM', svm.SVC(decision_function_shape='ovo')],
#                ['Random Forest', RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)],
#                ['Adaboost avec SVM', AdaBoostClassifier(n_estimators=100, algorithm = 'SAMME', base_estimator = svm.SVC(decision_function_shape='ovo'))],
#                ['Adaboost avec Random Forest', AdaBoostClassifier(n_estimators=100, algorithm = 'SAMME', base_estimator = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0))],
#                ['Reseau Neurones', MLPClassifier()],
#                ['Kppv', KNeighborsClassifier(3)]]            
#    nCV = 9
#    skf = StratifiedKFold(n_splits=nCV)  
#    test_classifieurs(skf, classifieurs, X, Y, nCV)    
    
    '''Test du MLP par cross validation du ensemble d'apprentissage, on trace les courbes d'erreur
    en apprentissage et en test'''
    nCV = 9
    skf = StratifiedKFold(n_splits=nCV)           
    etude_MLP(skf, X_train, Y_train, nCV, n_iter = 1000)
    
    '''Comparaison des prédictions des MLP enregistrés + vote de l'ensemble sur ensemble de test
    ensemble de test = ensemble vu par aucun classifieur'''
#    classifieurs = []
#    for i in range(1,nCV):
#        clf = joblib.load('clf'+str(i)+('.pkl'))
#        classifieurs.append([i, clf])
#    compare_results(classifieurs, X_test, Y_test, cumul = True)
    
    '''Etude du SVM en fonction de C. Affiche le nombre de points support moyens par cross validation,
    en fonction de C'''
#    nCV = 9
#    skf = StratifiedKFold(n_splits=nCV)
#    etude_SVM(skf, X_train, Y_train, nCV)
    
    '''Comparaison des prédictions des SVM enregistrés sur ensemble de test. Contraierement aux MLP,
    on ne peut pas avoir accès aux proba de sorties, donc pas de prédiction à partir des sommes des 
    probas. On pourrait les faire voter en regardant l'espèce prédite majoritaire'''
#    classifieurs = []
#    for i in range(1,nCV):
#        clf = joblib.load('clf'+str(i)+('.pkl'))
#        classifieurs.append([i, clf])
#    compare_results(classifieurs, X_test, Y_test)

    '''Comparaison des prédictions sur le "vrai" ensemble de test'''
#    classifieurs = []
#    for i in range(1, nCV):
#        clf = joblib.load('clf'+str(i)+('.pkl'))
#        classifieurs.append([i, clf])
#    prediction(classifieurs, )

    

    
    
    
        
    
