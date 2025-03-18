"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as special
from simulation.markov_models.gauss import Gauss
import math
from copy import deepcopy
from scipy.stats import truncnorm

from scipy.stats import entropy as kl_div
from scipy.stats import kstest
import pylab

class MultiGauss:
    def __init__(self, probabilities, gaussians):
        self.probabilities = probabilities 
        self.gaussians = gaussians 

    def plot_mult_gauss(self, x, label, color):
        f = [0] * len(x)

        for i in range(len(self.probabilities)):
            f += self.probabilities[i] * stats.norm.pdf(x, self.gaussians[i].mean, self.gaussians[i].deviation)
        plt.plot(x, f, color=color, linewidth=1, label=label)

    def fitted_results(self,data):
        x = [i for i in range(len(data))]
        f = [0] * len(x)

        for i in range(len(self.probabilities)):
            f += self.probabilities[i] * stats.norm.pdf(x, self.gaussians[i].mean, self.gaussians[i].deviation)

        # y, x = np.histogram(data, bins=bins, density=density)
        kullback_leibler = kl_div(f,data)
        cdf = np.cumsum(f)

        ks_stat, ks_pval = kstest(data,cdf)
        return ks_stat, ks_pval, kullback_leibler

    def remove_out_bounds_gauss(self, x):
        i =0 
        while i < len(self.probabilities):
            if (self.gaussians[i].mean > max(x)):
                del self.probabilities[i]
                del self.gaussians[i]
            else:
                i+=1
        self.normalise_gauss()

    def mult_gauss_values(self):
        t = np.arange(0, 20, 0.1)
        f = [0] * len(t)
        for i in range(len(self.probabilities)):
            f += self.probabilities[i] * stats.norm.pdf(t, self.gaussians[i].mean, self.gaussians[i].deviation)
        return f
    
    def normalise_gauss(self):
        sum_p = 0
        for i in range(len(self.probabilities)):
            sum_p += self.probabilities[i]

        for i in range(len(self.probabilities)):
            self.probabilities[i] /= sum_p

        return self

    def truncate_gauss(self, threshold):
        for i in range(len(self.gaussians)):
            negative_area = self.gaussians[i].calc_negative_area()
            if  negative_area > threshold:
                self.gaussians[i].deviation = - (self.gaussians[i].mean/(math.sqrt(2)*special.erfinv(2*threshold - 1)))
        return self
    
    def remove_small_prob_gauss(self, threshold):
        i =0
        while i < len(self.probabilities):
            if (self.probabilities[i] < threshold):
                del self.probabilities[i]
                del self.gaussians[i]
            else: 
                i += 1
        self.normalise_gauss()
    
    def unify_small_prob_gauss(self, threshold):
        i = 0
        sum_prob = 0
        sum_means = 0
        sum_deviations = 0
        while i < len(self.probabilities):
            if (self.probabilities[i] < threshold):
                sum_prob += self.probabilities[i]
            i += 1  
        if sum_prob > 0:
            i = 0
            while i < len(self.probabilities):
                if (self.probabilities[i] < threshold):
                    sum_means += self.probabilities[i]/sum_prob*self.gaussians[i].mean
                    sum_deviations += self.probabilities[i]/sum_prob*(self.gaussians[i].mean**2 + self.gaussians[i].deviation**2)    
                    del self.probabilities[i]
                    del self.gaussians[i]
                else:
                    i += 1
            
        sum_deviations = np.sqrt(sum_deviations - sum_means**2)
        if sum_prob > 0:
            self.probabilities.append(sum_prob)  
            self.gaussians.append(Gauss(sum_means, sum_deviations))
    
    def remove_zero(self):
        i=0
        while i < len(self.probabilities):
            if (self.gaussians[i].mean == 0) and (self.gaussians[i].deviation == 0):
                del self.probabilities[i]
                del self.gaussians[i]
            else:
                i += 1
        self.normalise_gauss()
    
    def calculate_mean(self):
        i =0
        mean = 0 
        while i < len(self.probabilities):
            mean += self.probabilities[i]*self.gaussians[i].mean
            i += 1
        return mean
    
    def calculate_mode(self):
        i =0
        mode = 0
        max_prob = 0 
        while i < len(self.probabilities):
            if max_prob < self.probabilities[i]:
                mode = self.gaussians[i].mean
                max_prob = self.probabilities[i]
            i += 1
        return mode
    
    def calculate_peaks(self):
        i =0
        prob = 0
        max_probabilities = []
        max_means = []
        max_length = 5 
        while i < len(self.probabilities):
            prob = self.probabilities[i]
            if i < max_length:
                max_probabilities.append(prob)
            else:
                for max_prob in max_probabilities:
                    if prob > max_prob:
                        max_probabilities.remove(max_prob)
                        max_probabilities.append(prob)
                        break
            i += 1
        i = 0
        while i < len(max_probabilities):
            max_means.append(self.gaussians[i].mean)
            i += 1

        return max_means, max_probabilities
    
    def calculate_sum_probabilities(self):
        i =0
        prob = 0 
        while i < len(self.probabilities):
            prob += self.probabilities[i]
            i += 1
        return prob
    
    def calc_observed(self, lower_bound, upper_bound, samp):
        cnt = 0
        for s in samp:
            if s <= upper_bound and s >= lower_bound:
                cnt += 1
        return cnt

    def calc_expected(self, lower_bound, upper_bound):
        expected = 0
        for i in range(len(self.probabilities)):
            cdf_lower, cdf_upper = stats.norm.cdf([lower_bound, upper_bound], self.gaussians[i].mean, self.gaussians[i].deviation)
            expected += self.probabilities[i] * (cdf_upper-cdf_lower)
        return expected


    def calc_expected_truncated(self, lower_bound, upper_bound):
        expected = 0
        for i in range(len(self.probabilities)):
            a, b = (0 - self.gaussians[i].mean) / self.gaussians[i].deviation, np.inf
            cdf_lower, cdf_upper = truncnorm.cdf([lower_bound, upper_bound], a=a, b=b, loc=self.gaussians[i].mean, scale=self.gaussians[i].deviation)
            expected += self.probabilities[i] * (cdf_upper-cdf_lower)
        return expected

    def calc_chi_square(self, bins, samp):
        chi_square = 0
        upper_bound = np.max(samp)
        step = np.max(samp)/bins

        for i in range(bins):
           lower_bound = i*step
           upper_bound = (i+1)*step
           observed = self.calc_observed(lower_bound, upper_bound, samp)
           expected = self.calc_expected_truncated(lower_bound, upper_bound, samp)
           chi_square += (observed-expected)**2 / expected
           

        return chi_square

    def calc_kl_divergence(self, bins, samp, all_samp):
        kl_divergence = 0
        upper_bound = np.max(samp)
        step = np.max(samp)/bins

        for i in range(bins):
           lower_bound = i*step
           upper_bound = (i+1)*step
           observed = self.calc_observed(lower_bound, upper_bound, samp)/len(all_samp)
           expected = self.calc_expected_truncated(lower_bound, upper_bound)
           if observed != 0 and expected != 0:
                kl_divergence += observed*(np.log(observed/expected))
           

        return kl_divergence

    def calc_kl_divergence_uniform(self, bins, samp):
        kl_divergence = 0
        upper_bound = np.max(samp)
        step = np.max(samp)/bins
        print(step)
        print(np.max(samp))

        for i in range(bins):
           lower_bound = i*step
           upper_bound = (i+1)*step
           observed = self.calc_observed(lower_bound, upper_bound, samp)/len(samp)
           expected = 1 / bins
           if observed != 0 and expected != 0:
                kl_divergence += observed*(np.log(observed/expected))
           

        return kl_divergence

    def calc_chi_square_uniform(self, bins, samp):
        chi_square = 0
        upper_bound = np.max(samp)
        step = np.max(samp)/bins

        for i in range(bins):
           lower_bound = i*step
           upper_bound = (i+1)*step
           observed = self.calc_observed(lower_bound, upper_bound, samp)
           expected = step
           chi_square += (observed-expected)**2 / expected
           

        return chi_square


    def plot_trunc_mult_gauss(self, x, label, color):
        f = [0] * len(x)
        for i in range(len(self.probabilities)):
            a, b = (0 - self.gaussians[i].mean) / self.gaussians[i].deviation, np.inf
            f += self.probabilities[i] * truncnorm.pdf(x, a=a, b=b, loc=self.gaussians[i].mean, scale=self.gaussians[i].deviation)
        plt.plot(x, f, color, linewidth=1, label=label)

    
    def calc_mean_with_truncations(self):
        mean = 0
        for i in range(len(self.probabilities)):
            a, b = (0 - self.gaussians[i].mean) / self.gaussians[i].deviation, np.inf
            mean += self.probabilities[i] * truncnorm.mean(a=a, b=b, loc=self.gaussians[i].mean, scale=self.gaussians[i].deviation)
        return mean

    def to_json(self):
        gaus_list = []
        for gaussian in self.gaussians:
            gaus_list.append(gaussian.to_json())
        return {'probabilities': self.probabilities,'gaussians': gaus_list}

def mg_from_json(json_data):
    gaus_list = []
    for gaussian in json_data['gaussians']:
        g = Gauss(gaussian['mean'], gaussian['deviation'])
        gaus_list.append(g)
    multi_gauss = MultiGauss(json_data['probabilities'], gaus_list)
    return multi_gauss