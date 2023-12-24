#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY


def naivebayes(x, y, x1):
# =============================================================================
#function logratio = naivebayes(x,y,x1);
#
#Computation of log P(Y|X=x1) using Bayes Rule
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#x1: input vector of d dimensions (dx1)
#
#Output:
#logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
# =============================================================================


    
    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    X1= np.matrix(x1)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
# =============================================================================
# fill in code here
#     print("X shape", x.shape)
#     print("Y shape", y.shape)
#     print("X1 shape", x1.shape)

    prob_y_success, prob_y_fail = naivebayesPY(x,y)
    print("Break point 1 ")
    posprob, negprob = naivebayesPXY(x,y)
    print("Break point 2")
    pos_prob = 1.0
    neg_prob = 1.0

    for i in range(d):
        if x1[i] == 1:
            pos_prob *= posprob[i]
            neg_prob *= negprob[i]
        else:
            pos_prob *= 1 - posprob[i]
            neg_prob *= 1 - negprob[i]

    logratio = np.log((pos_prob*prob_y_success)/(neg_prob*prob_y_fail))
    logratio = np.array([logratio])
    print("Log Ratio: ", logratio)
    print("Log Ratio Shape:", logratio.shape)

    return logratio
# # =============================================================================
# x,y = genTrainFeatures()
# featureVector = name2features("Sylvie")
# logratio = naivebayes(x,y,featureVector)
