"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

from simulation.markov_models.gauss import Gauss
from simulation.markov_models.mult_gauss import MultiGauss
import math

"""
Discere convolution of two functions f1 and f2 represented as lists
"""

def discrete_convolution(f1, f2):
    conv = []
    for i in range(len(f1)):
        for j in range(len(f2)):
            conv.append(f1[i]+f2[j])
    return conv

"""
Convolution of two gaussians
"""

def gauss_convolution(g1, g2):
    conv = Gauss(g1.mean+g2.mean, math.sqrt((g1.deviation)**2+(g2.deviation)**2))
    return conv

"""
Convolution of two sums of gaussians
mult1, mult2 are weighted sums of gaussians
returns another mult
"""
threshold = 0.001

def mult_gauss_convolution(mult1, mult2):
    mult = MultiGauss([],[])
    for i in range(len(mult1.probabilities)):
        for j in range(len(mult2.probabilities)):
            mult.probabilities.append(mult1.probabilities[i]*mult2.probabilities[j])
            mult.gaussians.append(gauss_convolution(mult1.gaussians[i], mult2.gaussians[j]))
    mult.unify_small_prob_gauss(threshold)
    return mult

def mult_gauss_self_convolution(mult1, k):
    mult = MultiGauss([1], [Gauss(0,0)])
    for i in range(k):
        mult = mult_gauss_convolution(mult, mult1)
    mult.unify_small_prob_gauss(threshold)
    return mult

def mult_gauss_sum(mult1, mult2, p1, p2):
    sum = MultiGauss([],[])
    for i in range(len(mult1.probabilities)):
        if p1 > 0:
            sum.probabilities.append(p1*mult1.probabilities[i])
            sum.gaussians.append(mult1.gaussians[i])
    
    for i in range(len(mult2.probabilities)):
        if p2 > 0:
            sum.probabilities.append(p2*mult2.probabilities[i])
            sum.gaussians.append(mult2.gaussians[i])
    
    sum.unify_small_prob_gauss(threshold)
    return sum
