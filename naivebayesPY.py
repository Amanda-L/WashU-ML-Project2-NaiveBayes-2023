#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from genTrainFeatures import genTrainFeatures

def naivebayesPY(x, y):

    # function [pos,neg] = naivebayesPY(x,y);
    #
    # Computation of P(Y)
    # Input:
    # x : n input vectors of d dimensions (dxn)
    # y : n labels (-1 or +1) (1xn)
    #
    # Output:
    # pos: probability p(y=1)
    # neg: probability p(y=-1)
    #
    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    Y = np.matrix(y)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.matrix('-1, 1')
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
    Ynew = np.hstack((Y, Y0))

    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    # print("XNew",Xnew.shape)
    # print("YNew,",Ynew.shape)
    rows,columns = Ynew.shape
    ## fill in code here
    pos = 0
    neg = 0
    for i in range(0,columns):
        item = Ynew[:,i]
        if item == 1:
            pos+=1
        elif item == -1:
            neg +=1
    pos = pos/columns
    neg = neg/columns
    return pos,neg

# #Test naivebayes
# x,y = genTrainFeatures()
# pos, neg = naivebayesPY(x,y)
# # print("Pos: ",pos)
# # print("Neg: ", neg)