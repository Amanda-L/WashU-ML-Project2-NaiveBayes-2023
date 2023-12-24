#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from genTrainFeatures import genTrainFeatures
from naivebayesCL import naivebayesCL
def classifyLinear(x, w, b):
# =============================================================================
#function preds=classifyLinear(x,w,b);
#
#Make predictions with a linear classifier
#Input:
#x : n input vectors of d dimensions (dxn)
#w : weight vector
#b : bias
#
#Output:
#preds: predictions
# =============================================================================

    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    W = np.matrix(w)
    
# =============================================================================
# fill in code here
    preds = np.dot(W.T,X) + b
    preds = np.where(preds > 0, 1, -1)
    return preds
# =============================================================================


[x,y]=genTrainFeatures();
[w,b]=naivebayesCL(x,y);
preds=classifyLinear(x,w,b)
print("Preds:", preds)
trainingerror=np.sum(preds==y)/(y.shape[1])
print(trainingerror)