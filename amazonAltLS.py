#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:49:24 2017

latent factor variable recommender system using alternating least squares
trained on amazon dataset with anonymized user and item hashes

@author: daniel riley
"""
#%%
import numpy as np
import gzip
import random
from collections import defaultdict
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

from collections import Counter

class altLeastSquares(object):
    ###########################################################################
    #Estimate user rating of item based on latent user and item biases
    #according to:
    #   rating(u,i) = a + b_i + b_u + g_i•g_u
    #where a is a constant bias, b_i is a rating bias associated with the item,
    #b_u is a rating bias associated with the user, and g_u,g_i are vectors
    #of the same length associated with the user and item respectively
    ###########################################################################
    def __init__(self,R):
        self.R = R
        self.trLen = len(self.R)
        
    def train(self,lam = 3,eps = 1e-40,gStep0 = 0.00001,annealingRate = 1):
        np.random.seed(0)
        datLen = 200000 #datLen = 200000 uses entire dataset
        self.gamLen = 6
        alpha = 8
        iteration = 1
        rKeys = list(self.R.keys())
        for bigIter in range(1):
            random.shuffle(rKeys)
            self.batchKeys = rKeys[:datLen]
            self.uRatingCount = Counter([k[0] for k in self.batchKeys])
            self.iRatingCount = Counter([g[1] for g in self.batchKeys])
            
            #initialize biases randomly
            betaUser = {u:(np.random.rand() - 0.5) for u in self.uRatingCount}
            betaItem = {i:(np.random.rand() - 0.5) for i in self.iRatingCount}
            iLen = len(betaItem)
            uLen = len(betaUser)
            gammaUser = {u:np.random.rand(self.gamLen)*0.00001 for u in self.uRatingCount}
            gammaItem = {i:np.random.rand(self.gamLen)*0.00001 for i in self.iRatingCount}
            
            for smallIter in range(10):
                alChange = 0.5
                biChange = 1
                buChange = 1
                giChange = 1
                guChange = 1
                change1 = True
                gstep = gStep0
                while(change1 == True):
                #hold g_u constant and perform least squares
                #to fit a, b_u,b_i, and g_i to the data
                    print iteration
                    gStep = gstep*annealingRate
                    if alChange > eps:
                        alpha,alChange = self.regaUpdate(betaItem,betaUser,alpha,gammaItem,gammaUser)
                
                    if buChange > eps:
                        betaUser,buChange = self.regbUpdate(betaUser,betaItem,alpha,self.uRatingCount,gammaUser,gammaItem,lam)
                
                    if biChange > eps:
                        betaItem,biChange = self.regbUpdate(betaItem,betaUser,alpha,self.iRatingCount,gammaItem,gammaUser,lam)
                    if giChange > eps:
                        gammaItem, giChange = self.reggUpdate(betaItem,betaUser,alpha,gammaItem,gammaUser,lam,gStep)
                    iteration = iteration + 1
                    change1 = biChange > eps*iLen or buChange > eps*uLen or alChange > eps or giChange > iLen*self.gamLen*eps
                
                gStep = gStep0
                alChange = 0.5
                biChange = 1
                buChange = 1
                giChange = 1
                guChange = 1
                change2 = True
                while(change2 == True):
                #hold g_i constant and perform least squares
                #to fit a, b_u,b_i, and g_u to the data
                    print iteration
                    gStep = gStep*annealingRate
                    if alChange > eps:
                        alpha,alChange = self.regaUpdate(betaItem,betaUser,alpha,gammaItem,gammaUser)            
                    if buChange > eps:
                        betaUser,buChange = self.regbUpdate(betaUser,betaItem,alpha,self.uRatingCount,gammaUser,gammaItem,lam)               
                    if biChange > eps:
                        betaItem,biChange = self.regbUpdate(betaItem,betaUser,alpha,self.iRatingCount,gammaItem,gammaUser,lam)         
                    if guChange > eps:
                        gammaUser, guChange = self.reggUpdate(betaUser,betaItem,alpha,gammaUser,gammaItem,lam,gStep)
                    iteration = iteration + 1
                    change2 = biChange > eps*iLen or buChange > eps*uLen or alChange > eps or guChange > uLen*self.gamLen*eps 
        return alpha,betaUser,betaItem,gammaUser,gammaItem
        
    def regaUpdate(self,betaItem,betaUser,alpha,gItem,gUser):
        alpha1 = 0
        for (j,k) in self.batchKeys:
            alpha1 = alpha1 + self.R[(j,k)] - betaItem[k] - betaUser[j] +np.dot(gItem[k],gUser[j])
        alpha1 = alpha1/self.trLen
        alChange = np.abs(alpha1 - alpha)
        #print 'alChange = ' + str(alChange)
        return alpha1, alChange

    def regbUpdate(self,lastBias,otherBias,alpha,numRatings,gCurr, gOther,lam):
        beta1 = {u:0 for u in lastBias}
        if lastBias.keys()[0][0] == 'U':
            for (j,k) in self.batchKeys:
                beta1[j] = beta1[j] + self.R[(j,k)] - alpha - otherBias[k] + np.dot(gOther[k],gCurr[j])
            for j in numRatings:
                beta1[j] = beta1[j]*1.0/(lam + numRatings[j])
        else:
            for (j,k) in self.batchKeys:
                beta1[k] = beta1[k] + self.R[(j,k)] - alpha - otherBias[j] + np.dot(gCurr[k],gOther[j])
            for k in numRatings:
                beta1[k] = beta1[k]*1.0/(lam + numRatings[k])
        bChange = sum([abs(beta1[j] - lastBias[j]) for j in beta1])
        print 'buChange = ' + str(bChange)
        return beta1, bChange
        
    def reggUpdate(self,bCurr,bOther,alpha,gCurr,gOther,lam,gStep):
        if gCurr.keys()[0][0] == 'U':
            gCurr1 = {u:np.zeros(self.gamLen) for u in self.uRatingCount}
            dg = {u:np.zeros(self.gamLen) for u in self.uRatingCount}
            for j,k in self.batchKeys:
                dg[j] += gOther[k]*(alpha + bOther[k] + bCurr[j] + np.dot(gCurr[j],gOther[k]) - self.R[(j,k)]) + lam*gCurr[j]  
            for j in self.uRatingCount:
                gCurr1[j] = gCurr[j] - gStep*dg[j]
            gChange = sum(sum([np.abs(gCurr1[new] - gCurr[new]) for new in gCurr1]))
        else:
            gCurr1 = {i:np.zeros(self.gamLen) for i in self.iRatingCount}
            dg = {i:np.zeros(self.gamLen) for i in self.iRatingCount}
            for j,k in self.batchKeys:
                dg[k] += gOther[j]*(alpha + bOther[j] + bCurr[k] + np.dot(gCurr[k],gOther[j]) - self.R[(j,k)]) + lam*gCurr[k]
            for k in self.iRatingCount:
                gCurr1[k] = gCurr[k] - gStep*dg[k]
            gChange = sum(sum([np.abs(gCurr1[new] - gCurr[new]) for new in gCurr1]))
        print 'g' + gCurr.keys()[0][0] + 'Change = ' + str(gChange)
        return gCurr1, gChange
    
#%%read user,item data from json
trLen = 200000
r = defaultdict()
i = 1
for l in readGz("train.json.gz"):
    if i >=trLen + 1:
        break
    user, item,rating = l['reviewerID'],l['itemID'],l['rating']
    r[(user,item)] = rating
    i = i+1
r = dict(r)
#%%
ALS = altLeastSquares(r)
alpha,betaUser,betaItem,gammaUser,gammaItem = ALS.train()

#%%
#valRatings = []
#valPred = []
#reviewText2 = []
#minRatings = 12
#i = 1
#for l in readGz('train.json.gz'):
#    if i >= trLen +1:
#        user, item = l['reviewerID'],l['itemID']
#        valRatings.append(l['rating'])
#       # revText = l['reviewText'].lower().split()
#        if user in betaUser and item in betaItem:    
#            valPred.append(alpha + betaItem[item] + betaUser[user] + np.dot(gammaItem[item],gammaUser[user]))
#        elif user in betaUser:
#            valPred.append(alpha + betaUser[user] + np.dot(mnGItem,gammaUser[user]))
#        elif item in betaItem:
#            valPred.append(alpha + betaItem[item] +np.dot(gammaItem[item],mnGUser))
#        else:    
#            valPred.append(alpha)
#   # reviewText2.append(revText)
#    i = i+1
    
    #%%
#for i in range(len(valPred)):
#    if valPred[i] >5:
#        valPred[i] = 5
#mnGUser = np.average([gammaUser[j] for j in gammaUser],axis = 0)
#mnGItem = np.average([gammaItem[j] for j in gammaItem],axis = 0)
#mseVal = sum([(valRatings[u] - valPred[u])**2 for u in range(len(valRatings))])/(200000 - trLen)

#%% print test set predictions to file in kaggle submission format
predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
    #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')

    if u in betaUser and i in betaItem:
        predictions.write(u + '-' + i + ',' + str(alpha + betaItem[item] + betaUser[user] + np.dot(gammaItem[item],gammaUser[user])) + '\n')
    elif u in betaUser:
        predictions.write(u + '-' + i + ',' + str(alpha + betaUser[user]) + '\n')
    elif i in betaItem:
        predictions.write(u + '-' + i + ',' + str(alpha + betaItem[item]) + '\n')
    else:
        predictions.write(u + '-' + i + ',' + str(alpha)+'\n')
predictions.close()

