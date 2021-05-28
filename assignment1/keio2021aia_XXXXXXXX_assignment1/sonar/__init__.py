##
import numpy as np
import pickle as pkl
import random
##

##
def perceptron(z):
    return -1 if z <= 0 else 1

def ploss(yhat, y):
    return max(0, -yhat*y)

def ppredict(self, x):
    return self(x)
##
