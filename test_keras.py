# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:26:36 2017

@author: Quentin PC2
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Dropout, MaxPooling2D
from keras.layers import Reshape, Flatten, Merge
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class CNN(Sequential):
    """
    Classe représentant un réseau de neurone à convolutions étendant le réseau 
    de neurone de base de Keras.
    """
    def __init__(self, input_shape):
        Sequential.__init__(self)
        self.add(Convolution2D(10, 5, 5, input_shape=input_shape, border_mode='valid'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
#        self.add(Dropout(0.5))
        self.add(Convolution2D(15, 5, 5))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(99))
        self.add(Activation('softmax'))
#        self.add(Dense(1))
#        self.add(Activation('relu'))
        #on utilise sparse_categorical_crossentropy pour éviter d'avoir des problèmes 
        #liés à la différence entre le nombre de sorties et les catégories
        self.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                     metrics = ['categorical_accuracy', 'precision', 'recall', 'fmeasure'])

def chargement_images(ids, input_shape):
    """
    Renvoie un np.array contenant les matrices représentant les images dont les 
    identifiants sont passés en entrée.
    """
    print("Chargement des images....")
    images = []
    for i in ids:
#        print('image', i)
        image = load_img('images/'+str(i)+'.jpg', grayscale=True, target_size=input_shape)
#        image = load_img('images/'+str(i)+'.jpg', grayscale=True)
#        image.thumbnail(input_shape)
#        image.save('images_reduites/' + str(i) + '.jpg')
        img = img_to_array(image)
        images.append(img)
    print("Images chargées !")
    return np.array(images)

def apprentissage_100x100():
    """
    Entraine un réseau de neurones convolutif définit comme précédemment avec 
    les images redimensionnées en 100x100.
    """
    print('CNN')
    #on charge les données
    train = pd.read_csv('train.csv')
    
    t = train.as_matrix()
    #features
    X = t[:,2:]
    #objectif
    Y = t[:,1]
    #on recode Y pour avoir des labels qui ne soient plus textuels
    Y = LabelEncoder().fit(Y).transform(Y)
    Y = to_categorical(Y)
    
    #id de l'ensemble d'apprentissage
    ids = t[:, 0]
    #on récupère les images
    images = chargement_images(ids, (100, 100))
#    return []
    #on prépare la génération de données supplémentaires
    #on s'autorise des rotations de 10degrés, des symmétires verticales et horizontales 
    #et des zooms de 10%
    idg = ImageDataGenerator(rotation_range = 0.1, vertical_flip = True,
                             horizontal_flip = True, zoom_range = 0.1)
    
    #on sépare l'ensemble d'apprentissage en test et apprentissage, 
    #avec 10% de données en test
    Nb_splits = 1
    skf = StratifiedShuffleSplit(n_splits=Nb_splits)
    #on fixe le nombre d'époques d'apprentissage et le pas de visualisation
    max_epoch = 10
    epoch_iter = 1
    
    results = {'train':{'Accuracy':[0], 'F mesure':[0], 'Précision':[0], 'Rappel':[0], 'Perte':[]},
               'test':{'Accuracy':[0], 'F mesure':[0], 'Précision':[0], 'Rappel':[0]}}
    #pour chaque séparation test/apprentissage
    for train_index, test_index in skf.split(X, Y):
        #on apprend un réseau
        reseau = CNN((100, 100, 1))
        
        #on récupère les données correspondantes
#        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        img_train, img_test = images[train_index], images[test_index]
        #on récupère les images via l'augmenteur de données
        iterateur = idg.flow(img_train, Y_train, save_to_dir = 'images_gen')

        print("Début d'apprentissage......")
        for i in range(0, max_epoch, epoch_iter):
            print("Etape", i+1)
            #on entraine le réseau
#            historique = reseau.fit(img_train, Y_train, verbose = 0, nb_epoch = epoch_iter, 
#                                    initial_epoch = 0)
            historique = reseau.fit_generator(iterateur, 
                                              samples_per_epoch = Y_train.shape[0],
                                              nb_epoch = epoch_iter, 
                                              verbose = 0, 
                                              validation_data = (img_test, Y_test))
#            print(historique.history.keys())
            #on récupère la perte
            results['train']['Perte'].extend(historique.history['loss'])
            
            results = ajout_resultat(results, reseau, img_train, Y_train, img_test, Y_test)
        print("Fin d'apprentissage")
    graphique('CNN', results)

class RNN(Sequential):
    """
    Classe représentant un réseau de neurone étendant le réseau de neurone 
    de base de Keras.
    """
    def __init__(self):
        Sequential.__init__(self)
        self.add(Dense(100, input_dim = 192))
        self.add(Activation('relu'))
#        self.add(Dense(60))
#        self.add(Activation('relu'))
#        self.add(Dense(40))
#        self.add(Activation('relu'))
        self.add(Dense(99))
        self.add(Activation('softmax'))
        #hyper nul
#        self.compile(loss='mse', optimizer='sgd',
        #nul mais moins nul
#        self.compile(loss='categorical_crossentropy', optimizer='sgd',
        #à peu près équivalent à celui du dessous
#        self.compile(loss='mse', optimizer='rmsprop',
        #super bien
        self.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                     metrics = ['categorical_accuracy', 'precision', 'recall', 'fmeasure'])

def convertion_proba_classe(entree):
    """
    Fonction qui convertie une liste de vecteurs de probabilité d'appertance en 
    liste de classe
    """
    sortie = []
    for ligne in entree:
        ligne = ligne.tolist()
        sortie.append(ligne.index(max(ligne)) + 1)
    return np.array(sortie)

def apprentissage_features():
    """
    Fonction qui fait apprendre le réseau de neurones dédié aux features
    """
    print('RNN')
    #on charge les données
    train = pd.read_csv('train.csv')
    t = train.as_matrix()
    #features
    X = t[:,2:]
    #objectif
    Y = t[:,1]
    #on recode Y pour avoir des labels qui ne soient plus textuels
    Y = LabelEncoder().fit(Y).transform(Y)
    Y = to_categorical(Y)
    #on fixe le nombre d'époques d'apprentissage
    max_epoch = 200
    
    results = {'train':{'Accuracy':[0], 'F mesure':[0], 'Précision':[0], 'Rappel':[0], 'Perte':[]},
               'test':{'Accuracy':[0], 'F mesure':[0], 'Précision':[0], 'Rappel':[0]}}
    Nb_splits = 1
    skf = StratifiedShuffleSplit(n_splits=Nb_splits)
    for train_index, test_index in skf.split(X, Y):
        reseau = RNN()
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        #on entraine le réseau
        print("Début d'apprentissage......")
        for ep in range(max_epoch):
            historique = reseau.fit(X_train, Y_train, verbose = 0, nb_epoch = 1)
            #on récupère la perte
            results['train']['Perte'].extend(historique.history['loss'])
            
            results = ajout_resultat(results, reseau, X_train, Y_train, X_test, Y_test)
        print("Fin d'apprentissage")
            
    graphique('RNN', results)
    
def graphique(titre, results):
    """
    Fonction qui crée le plot en prenant un dictionnaire de dictionnaire de features
    """
    plt.title(titre)
    #nombre de plots à afficher
    n = max(len(results['train']), len(results['test']))
    for clef in results:
        i = 1
        for clef2 in results[clef]:
            #on affiche la perte à part
            if clef2 != 'Perte':
                #si n est un carré parfait, on le prend en compte
                if int(sqrt(n)) == sqrt(n):
                    axe = plt.subplot(sqrt(n), sqrt(n), i)
                else:
                    axe = plt.subplot(int(sqrt(n)), int(sqrt(n)) + 1, i)
                axe.plot(results[clef][clef2], label = clef)
                plt.ylabel(clef2)
                plt.legend(loc = 'lower right')
                i += 1
            elif clef2 == 'Perte':
                #si n est un carré parfait, on le prend en compte
                if int(sqrt(n)) == sqrt(n):
                    axe = plt.subplot(sqrt(n), sqrt(n), n)
                else:
                    axe = plt.subplot(int(sqrt(n)), int(sqrt(n)) + 1, n)
                axe.plot(results[clef]['Perte'], label = clef)
                plt.ylabel('Perte')
    plt.show()
    
def ajout_resultat(results, reseau, X_train, Y_train, X_test, Y_test):
    """
    Fonction qui ajoute dans results les performances en apprentissage et en 
    test du réseau
    """
    #on évalue son accuracy en test
    proba_prediction_test = reseau.predict(X_test)
    prediction_test = convertion_proba_classe(proba_prediction_test)
    Y_test_cat = convertion_proba_classe(Y_test)

    results['test']['Accuracy'].append(accuracy_score(Y_test_cat, prediction_test))
    results['test']['F mesure'].append(f1_score(Y_test_cat, prediction_test, average = 'macro'))
    results['test']['Précision'].append(precision_score(Y_test_cat, prediction_test, average = 'macro'))
    results['test']['Rappel'].append(recall_score(Y_test_cat, prediction_test, average = 'macro'))
    
    #on récupère son accuracy en apprentissage
#            results['train'].append(historique.history['categorical_accuracy'][-1])
    #on évalue son accuracy en apprentissage
    proba_prediction_train = reseau.predict(X_train)
    prediction_train = convertion_proba_classe(proba_prediction_train)
    Y_train_cat = convertion_proba_classe(Y_train)

    results['train']['Accuracy'].append(accuracy_score(Y_train_cat, prediction_train))
    results['train']['F mesure'].append(f1_score(Y_train_cat, prediction_train, average = 'macro'))
    results['train']['Précision'].append(precision_score(Y_train_cat, prediction_train, average = 'macro'))
    results['train']['Rappel'].append(recall_score(Y_train_cat, prediction_train, average = 'macro'))
    
    return results
    
class Dual(Sequential):
    """
    Réseau de neurones qui mélange réseau à convolution et réseau de neurone de 
    base.
    """
    def __init__(self, input_shape):
        Sequential.__init__(self)
        #partie RNN
        RNN = Sequential()
        RNN.add(Dense(100, input_dim = 192))
        RNN.add(Activation('relu'))
        #partie CNN
        CNN = Sequential()
        CNN.add(Convolution2D(10, 5, 5, input_shape=input_shape, border_mode='same'))
        CNN.add(Activation('relu'))
        CNN.add(MaxPooling2D(pool_size=(2, 2)))
        CNN.add(Convolution2D(15, 5, 5))
        CNN.add(Activation('relu'))
        CNN.add(MaxPooling2D(pool_size=(2, 2)))
        CNN.add(Dropout(0.5))
        CNN.add(Flatten())
        #mélange des deux
        self.add(Merge([RNN, CNN], mode = 'concat'))
        self.add(Dense(100))
        self.add(Activation('relu'))
        self.add(Dense(99))
        self.add(Activation('softmax'))
        #fonction de calcul de la perte
        self.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                     metrics = ['categorical_accuracy', 'precision', 'recall', 'fmeasure'])
    
def apprentissage_mixte():
    """
    Fonction qui fait apprendre le réseau de neurones mixte
    """
    print('Dual')
    #on charge les données
    train = pd.read_csv('train.csv')
    
    t = train.as_matrix()
    #features
    X = t[:,2:]
    #objectif
    Y = t[:,1]
    #on recode Y pour avoir des labels qui ne soient plus textuels
    Y = LabelEncoder().fit(Y).transform(Y)
    Y = to_categorical(Y)
    
    #id de l'ensemble d'apprentissage
    ids = t[:, 0]
    #on récupère les images
    images = chargement_images(ids, (100, 100))
    #on prépare la génération de données supplémentaires
    #on s'autorise des rotations de 10degrés, des symmétires verticales et horizontales 
    #et des zooms de 10%
#    idg = ImageDataGenerator(rotation_range = 0.1, vertical_flip = True,
#                             horizontal_flip = True, zoom_range = 0.1)
    
    #on sépare l'ensemble d'apprentissage en test et apprentissage, 
    #avec 10% de données en test
    Nb_splits = 1
    skf = StratifiedShuffleSplit(n_splits=Nb_splits)
    #on fixe le nombre d'époques d'apprentissage et le pas de visualisation
    max_epoch = 100
    
    results = {'train':{'Accuracy':[0], 'F mesure':[0], 'Précision':[0], 'Rappel':[0], 'Perte':[]},
               'test':{'Accuracy':[0], 'F mesure':[0], 'Précision':[0], 'Rappel':[0]}}
    #pour chaque séparation test/apprentissage
    for train_index, test_index in skf.split(X, Y):
        #on apprend un réseau
        reseau = Dual((100, 100, 1))
        
        #on récupère les données correspondantes
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        img_train, img_test = images[train_index], images[test_index]
        #on récupère les images via l'augmenteur de données
#        iterateur = idg.flow(img_train, Y_train, save_to_dir = 'images_gen')

        print("Début d'apprentissage......")
        for i in range(max_epoch):
            #on entraine le réseau
            if not i%10:
                print("Etape", i + 1)
            historique = reseau.fit([X_train, img_train], Y_train, verbose = 0,
                                    nb_epoch = 1)
#            historique = reseau.fit_generator([iterateur, X_train],
#                                              samples_per_epoch = Y_train.shape[0],
#                                              nb_epoch = epoch_iter, 
#                                              verbose = 0)
            #on récupère la perte
            results['train']['Perte'].extend(historique.history['loss'])
            
            results = ajout_resultat(results, reseau, [X_train, img_train], Y_train, [X_test, img_test], Y_test)
        print("Fin d'apprentissage")
            
    graphique('Dual', results)
#    plt.title("Dual")
#    for clef in results:
#        plt.plot(range(max_epoch), results[clef], label = clef)
#    plt.legend(loc = 'upper left')
#    plt.ylabel('Accuracy')
#    plt.show()
    
if __name__ == '__main__':
    #test sur le réseau de neurones normal
#    inputs = np.random.random((1000, 192))
#    outputs = np.random.randint(2, size=(1000, 1))
#    reseau = RNN()
#    reseau.fit(inputs, outputs)
#    X_test = np.random.random((250, 192))
#    Y_test = np.random.randint(2, size=(250, 1))
#    score = reseau.evaluate(X_test, Y_test, batch_size=16)
#    
#    print('\n\nscore', score)
    #test sur le réseau de neurones convolutif
#    images = []
#    for i in (1, 2):
#        image = load_img('images/'+str(i)+'.jpg', grayscale=True, target_size=(467, 526))
#        img = img_to_array(image)
#        images.append(img)
#    images = np.array(images)
#    print(img.shape)
#    print(len(images))
#    inputs = np.random.random((2, 467, 526, 1))
#    objectif = np.random.randint(1, size=(2, ))
#    print(inputs.shape)
#    print(objectif.shape)
#    reseau = CNN((467, 526, 1))
#    reseau.fit(images, objectif, nb_epoch = 2, batch_size = 2)
    apprentissage_100x100()
#    apprentissage_features()
#    apprentissage_mixte()
    


