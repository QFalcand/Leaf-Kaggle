rm(list=ls()) #Nettoyer la mémoire
setwd("") #Working directory


tabdata = read.table('train.csv',sep = ',', header = TRUE)

tabdata

summary(tabdata)

#Statistiques de base

attach(tabdata)

summary(tabdata[species == 'Viburnum_x_Rhytidophylloides' , ])

cor(tabdata[, 3:194])
heatmap(abs(cor(tabdata[, 3:194])), symm = TRUE)

library('FactoMineR')

par(mfrow=c(1,1))
#ACP sur l'ensemble
acp.res = PCA(tabdata[,-(1:2)], scale.unit = TRUE, ncp = 30, graph = FALSE)
plot(acp.res, choix = 'ind', col.ind = as.factor(tabdata[,2]))

par(mfrow=c(3,2))
#------------------------------------------------------------
#ACP sur les margins
acp.res = PCA(tabdata[,3:66], scale.unit = TRUE, ncp = 11, graph = FALSE)
acp.res
plot(acp.res, choix = 'var')
round(acp.res$eig,  4)

coord12 = acp.res$ind$coord[,1:2]

coscar1 = acp.res$ind$cos2[,1]
coscar2 = acp.res$ind$cos2[,2]
coscar12 = coscar1 + coscar2

contrib12 = acp.res$ind$contrib[,1:2]

Info.ind = round(data.frame(coord12, coscar12, contrib12), 2)
colnames(Info.ind) = c("coord1", "coord2", "coscar12", "contrib1", "contrib2")

Species = tabdata[,2]
Species = Species[!duplicated(Species)]

for (i in 1:length(Species)) {
  
  indices = which(tabdata[,2] == Species[i])
  espece = as.character(Species[i])
  coor1 = mean(coord12[indices, 1])
  coor2 = mean(coord12[indices, 2])
  coscar = mean(coscar12[indices])
  
  if (i == 1) {
    coord12_species = data.frame(espece, coor1, coor2, coscar)
    colnames(coord12_species)=c('Species', 'Dim.1', 'Dim.2', 'coscar12')}
  else {
    df = data.frame(espece, coor1, coor2, coscar)
    colnames(df)=c('Species', 'Dim.1', 'Dim.2', 'coscar12')
    coord12_species = rbind(coord12_species, df)}
  
}

#Plot des individus moyens/espèce dans le nouvel espace en prenant les 2 meilleurs dimensions
plot(coord12_species[,2], coord12_species[,3], col = 0, cex = 0.8, xlim = c(-10,10), ylim = c(-10,10))
text(coord12_species[,2], coord12_species[,3], coord12_species[,1], cex = 0.3+log(1+coord12_species[,4]))
abline(h = 0, v = 0)
title('Margins - individus moyens')

#------------------------------------------------------------
#ACP sur les shapes
acp.res = PCA(tabdata[,67:130], scale.unit = TRUE, ncp = 11, graph = FALSE)
acp.res
plot(acp.res, choix = 'var')
round(acp.res$eig,  4)

coord12 = acp.res$ind$coord[,1:2]

coscar1 = acp.res$ind$cos2[,1]
coscar2 = acp.res$ind$cos2[,2]
coscar12 = coscar1 + coscar2

contrib12 = acp.res$ind$contrib[,1:2]

Info.ind = round(data.frame(coord12, coscar12, contrib12), 2)
colnames(Info.ind) = c("coord1", "coord2", "coscar12", "contrib1", "contrib2")

Species = tabdata[,2]
Species = Species[!duplicated(Species)]

for (i in 1:length(Species)) {
  
  indices = which(tabdata[,2] == Species[i])
  espece = as.character(Species[i])
  coor1 = mean(coord12[indices, 1])
  coor2 = mean(coord12[indices, 2])
  coscar = mean(coscar12[indices])
  
  if (i == 1) {
    coord12_species = data.frame(espece, coor1, coor2, coscar)
    colnames(coord12_species)=c('Species', 'Dim.1', 'Dim.2', 'coscar12')}
  else {
    df = data.frame(espece, coor1, coor2, coscar)
    colnames(df)=c('Species', 'Dim.1', 'Dim.2', 'coscar12')
    coord12_species = rbind(coord12_species, df)}
  
}

#Plot des individus moyens/espèce dans le nouvel espace en prenant les 2 meilleurs dimensions
plot(coord12_species[,2], coord12_species[,3], col = 0, cex = 0.8, xlim = c(-5,5), ylim = c(-5,5))
text(coord12_species[,2], coord12_species[,3], coord12_species[,1], cex = 0.3+log(1+coord12_species[,4]))
abline(h = 0, v = 0)
title('Shapes - individus moyens')

#------------------------------------------------------------
#ACP sur les textures

acp.res = PCA(tabdata[,131:194], scale.unit = TRUE, ncp = 11, graph = FALSE)
acp.res
plot(acp.res, choix = 'var')
round(acp.res$eig,  4)

coord12 = acp.res$ind$coord[,1:2]

coscar1 = acp.res$ind$cos2[,1]
coscar2 = acp.res$ind$cos2[,2]
coscar12 = coscar1 + coscar2

contrib12 = acp.res$ind$contrib[,1:2]

Info.ind = round(data.frame(coord12, coscar12, contrib12), 2)
colnames(Info.ind) = c("coord1", "coord2", "coscar12", "contrib1", "contrib2")

Species = tabdata[,2]
Species = Species[!duplicated(Species)]

for (i in 1:length(Species)) {
  
  indices = which(tabdata[,2] == Species[i])
  espece = as.character(Species[i])
  coor1 = mean(coord12[indices, 1])
  coor2 = mean(coord12[indices, 2])
  coscar = mean(coscar12[indices])
  
  if (i == 1) {
    coord12_species = data.frame(espece, coor1, coor2, coscar)
    colnames(coord12_species)=c('Species', 'Dim.1', 'Dim.2', 'coscar12')}
  else {
    df = data.frame(espece, coor1, coor2, coscar)
    colnames(df)=c('Species', 'Dim.1', 'Dim.2', 'coscar12')
    coord12_species = rbind(coord12_species, df)}
  
}

#Plot des individus moyens/espèce dans le nouvel espace en prenant les 2 meilleurs dimensions
plot(coord12_species[,2], coord12_species[,3], col = 0, cex = 0.8, xlim = c(-5,5), ylim = c(-5,5))
text(coord12_species[,2], coord12_species[,3], coord12_species[,1], cex = 0.3+log(1+coord12_species[,4]))
abline(h = 0, v = 0)
title('Textures - individus moyens')


