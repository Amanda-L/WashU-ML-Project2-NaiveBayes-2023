#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: Yichen
@author: M.Joo
"""

import numpy as np
from genTrainFeatures import genTrainFeatures
import numpy as np

def naivebayesPXY(x, y):
    print("Entered NaiveBayesPXY")
    # Convert input lists to numpy arrays
    X = np.array(x)
    Y = np.array(y)

    d, n = X.shape

    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d, 2))
    # print("XO:",X0)
    Y0 = np.array([[-1, 1]])
    print("X0, Y0 initialized")
    # add one all-ones positive and negative example
    print("X shape:", X.shape)
    print("X0 shape:", X0.shape)
    Xnew = np.hstack((X, X0)) # stack arrays in sequence horizontally (column-wise)
    print("Xnew not problematic")
    print("Y shape:", Y.shape)
    print("Y0 shape:", Y0.shape)
    # Check if the array has 1 dimension
    print("Y Dimensions:", Y.ndim)
    if Y.ndim == 1:
        print("This is the problem")
        # Reshape the array to have 2 dimensions (unknown size for each dimension)
        Y = Y.reshape(1, Y.shape[0])
        
    Ynew = np.hstack((Y, Y0))
    print("Initial hstack complete")
    # matrix of all-zeros -
    X1 = np.zeros((d, 2))
    Xnew = np.hstack((Xnew, X1))
    Ynew = np.hstack((Ynew, Y0))

    print("Full hstack complete")
    # Re-configuring the size of matrix Xnew
    d, n = Xnew.shape
    # print("N is: ", n)
    # print("D is:", d)
    # print("Xnew2",Xnew)
    # Initialize Laplace smoothing constant
    alpha = 1.0

    # Total neg/pos in y
    total_negs = np.sum(Ynew == -1)
    print(Ynew)

    print("Total Negs:",total_negs)
    total_pos = np.sum(Ynew == 1)
    posprob = np.zeros((d,1))
    negprob = np.zeros((d,1))

    for i in range(d):
        vector_at_i = Xnew[i, :]

        # Counts for X=0 and X=1
        # count_0 = np.where(vector_at_i == 0)[0]  # Keep [1] for indexing
        count_1 = np.where(vector_at_i == 1)[0] #[1]  # Keep [1] for indexing

        # Laplace smoothing for X=0 and X=1
        countX1_y_success = np.sum((Ynew[0, count_1] == 1))  #+ alpha
        countX1_y_fail = np.sum((Ynew[0, count_1] == -1))  #+ alpha


        # Probability X=1|Y=1,-1
        probX1_ysuccess = countX1_y_success / total_pos
        probX1_yfail = countX1_y_fail / total_negs

        # Posprob and negprob calculations
        # posprob[i] = probX0_ysuccess * probX1_ysuccess
        posprob[i] = probX1_ysuccess
        negprob[i] = probX1_yfail

    # print("Posprob: ", posprob)
    # print("Negprob: ", negprob)
    print("Posprob Shape (BREAKPOINT 1.5)", posprob.shape)
    print("Negprob Shape", negprob.shape)
    return posprob, negprob


# =============================================================================
x,y = genTrainFeatures()
yTest = y[:,5:10]
xTest = x[40:45,40:45]

print("XTest",xTest)
print("Ytest",yTest.shape)
posprob, negprob = naivebayesPXY(x,y)

