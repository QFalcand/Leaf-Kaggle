# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:03:17 2017

@author: Quentin PC2
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Dropout, MaxPooling2D
from keras.layers import Reshape, Flatten, Merge
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from PIL import Image

def graphique(titre, results):
    """
    Fonction qui crée le plot en prenant un dictionnaire de dictionnaire de features
    """
    plt.title(titre)
    #nombre de plots à afficher
    n = max(len(results['Apprentissage']), len(results['Validation']))
    for clef in results:
        i = 1
        for clef2 in results[clef]:
            #on affiche la perte à part
            if clef2 != 'Perte':
                loca = 'lower right'
                sub = i
                i += 1
            elif clef2 == 'Perte':
                loca = 'upper right'
                sub = n
            #si n est un carré parfait, on le prend en compte
            if int(sqrt(n)) == sqrt(n):
                axe = plt.subplot(sqrt(n), sqrt(n), sub)
            else:
                axe = plt.subplot(int(sqrt(n)), int(sqrt(n)) + 1, sub)
            axe.plot(results[clef][clef2], label = clef)
            plt.ylabel(clef2)
            plt.xlabel("Nombre d'époques")
            plt.legend(loc = loca)
    plt.show()

def chargement_images(ids, input_shape):
    """
    Renvoie un np.array contenant les matrices représentant les images dont les 
    identifiants sont passés en entrée.
    """
    print("Chargement des images....")
    images = []
    for i in ids:
    #        print('image', i)
        #pour créer les images réduites
#        image = load_img('images/'+str(int(i))+'.jpg', grayscale=True)
#        image.thumbnail(input_shape)
#        nouv = Image.new('L', input_shape, 0)
#        coord_hg = (int((nouv.size[0] - image.size[0]) / 2),
#                    int((nouv.size[1] - image.size[1]) / 2))
#        nouv.paste(image, coord_hg)
#        nouv.save('images_reduites/' + str(int(i)) + '.jpg')
#        img = img_to_array(nouv)
        #pour charger directement les images réduites
        image = load_img('images_reduites/'+str(int(i))+'.jpg', grayscale=True)
        img = img_to_array(image)
        #dans tous les cas
        images.append(img)
    print("Images chargées !")
    return np.array(images)


str_reseau = 'histpur_RNN_norm'

#on charge les données
train = pd.read_csv('train_norm.csv')

t = train.as_matrix()
#id de l'ensemble d'apprentissage
ids = t[:, 0]
#features
if 'CNN' not in str_reseau:
    X = t[:,2:]
#    print(t.shape, X.shape)
if 'hist' in str_reseau:
    hist = pd.read_csv('hist_norm.csv', header = None)
    h = hist.as_matrix()
    id2 = ids - 1
#    X = h[id2.tolist(), 1:]
    X2 = h[id2.tolist(), 1:]
    X = np.concatenate((X, X2), axis = 1)
#    print(c.shape)
#objectif
Y = t[:,1]
#on recode Y pour avoir des labels qui ne soient plus textuels
Y = LabelEncoder().fit(Y).transform(Y)
Y = to_categorical(Y)

if "CNN" in str_reseau:
    #on récupère les images
    X = chargement_images(ids, (100, 100))
    
    #on prépare la génération de données supplémentaires
    #on s'autorise des rotations de 10degrés, des symmétires verticales et horizontales 
    #et des zooms de 10%
    idg = ImageDataGenerator(rotation_range = 0, vertical_flip = True,
                             horizontal_flip = True, zoom_range = 0.1)

#on sépare l'ensemble d'apprentissage en test et apprentissage, 
#avec 10% de données en test
Nb_splits = 1
skf = StratifiedShuffleSplit(n_splits=Nb_splits)
#on fixe le nombre d'époques d'apprentissage et le pas de visualisation
max_epoch = 200

liste_hist = []
#pour chaque séparation test/apprentissage
for train_index, test_index in skf.split(X, Y):
    #on essaie de récupérer le réseau appris aupparavant s'il existe
    try:
        reseau = load_model(str_reseau + '.h5')
    except :
        #on apprend un réseau
        reseau = Sequential()
        if 'CNN' in str_reseau:
    #        reseau.add(Reshape((1, 100, 100), input_shape=(100, 100, 1)))
            reseau.add(Convolution2D(10, 5, 5, input_shape=X[0, :, :, :].shape,
                                     border_mode='valid'))
            reseau.add(Activation('relu'))
            reseau.add(MaxPooling2D(pool_size=(2, 2)))
    #        self.add(Dropout(0.5))
            reseau.add(Convolution2D(15, 5, 5))
            reseau.add(Activation('relu'))
            reseau.add(MaxPooling2D(pool_size=(2, 2)))
            reseau.add(Dropout(0.5))
            reseau.add(Flatten())
    #        reseau.add(Dense(99))
    #        reseau.add(Activation('softmax'))
        elif 'RNN' in str_reseau:
            reseau.add(Dense(100, input_dim = X.shape[1]))
            reseau.add(Activation('relu'))
#            reseau.add(Dense(100))
#            reseau.add(Activation('relu'))
        
        #couches communes à tous les réseaux
        reseau.add(Dense(99))
        reseau.add(Activation('softmax'))
        reseau.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                       metrics = ['categorical_accuracy', 'precision',
                                  'recall', 'fmeasure'])
    
    #on récupère les données correspondantes
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    if 'CNN' in str_reseau:
#        img_train, img_test = images[train_index], images[test_index]
        #on récupère les images via l'augmenteur de données
        iterateur = idg.flow(X_train, Y_train)

    print("Début d'apprentissage......")
#        for i in range(0, max_epoch, epoch_iter):
#            print("Etape", i+1)
        #on entraine le réseau
    if 'CNN' in str_reseau:
        historique = reseau.fit_generator(iterateur, 
                                          samples_per_epoch = Y_train.shape[0],
                                          nb_epoch = max_epoch, 
                                          verbose = 1, 
                                          validation_data = (X_test, Y_test))
    else:
        historique = reseau.fit(X_train, Y_train, verbose = 0, nb_epoch = max_epoch, 
                                validation_data = (X_test, Y_test))
    liste_hist.append(historique.history)
    print("Fin d'apprentissage")

init = [0] * max_epoch
results = {'Apprentissage':{'Accuracy':init.copy(), 'F mesure':init.copy(), 
                            'Précision':init.copy(), 'Rappel':init.copy(),
                            'Perte':init.copy()},
           'Validation':{'Accuracy':init.copy(), 'F mesure':init.copy(),
                         'Précision':init.copy(), 'Rappel':init.copy(), 
                         'Perte':init.copy()}}
for elem in liste_hist:
    results['Apprentissage']['Accuracy'] = [x + y / Nb_splits for x, y in zip(results['Apprentissage']['Accuracy'], elem['categorical_accuracy'])]
    results['Apprentissage']['F mesure']  = [x + y / Nb_splits for x, y in zip(results['Apprentissage']['F mesure'], elem['fmeasure'])]
    results['Apprentissage']['Précision'] = [x + y / Nb_splits for x, y in zip(results['Apprentissage']['Précision'], elem['precision'])]
    results['Apprentissage']['Rappel'] = [x + y / Nb_splits for x, y in zip(results['Apprentissage']['Rappel'], elem['recall'])]
    results['Apprentissage']['Perte'] = [x + y / Nb_splits for x, y in zip(results['Apprentissage']['Perte'], elem['loss'])]
    results['Validation']['Accuracy'] = [x + y / Nb_splits for x, y in zip(results['Validation']['Accuracy'], elem['val_categorical_accuracy'])]
    results['Validation']['F mesure'] = [x + y / Nb_splits for x, y in zip(results['Validation']['F mesure'], elem['val_fmeasure'])]
    results['Validation']['Précision'] = [x + y / Nb_splits for x, y in zip(results['Validation']['Précision'], elem['val_precision'])]
    results['Validation']['Rappel'] = [x + y / Nb_splits for x, y in zip(results['Validation']['Rappel'], elem['val_recall'])]
    results['Validation']['Perte'] = [x + y / Nb_splits for x, y in zip(results['Validation']['Perte'], elem['val_loss'])]
graphique(str_reseau, results)
reseau.save(str_reseau + '.h5')
print('réseau enregistré')