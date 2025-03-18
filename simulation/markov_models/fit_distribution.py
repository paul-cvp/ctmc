
"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

from scipy.optimize import curve_fit
import numpy as np
from simulation.markov_models.gauss import Gauss
from simulation.markov_models.mult_gauss import MultiGauss
from scipy.signal import find_peaks
import math

"""
Smooth a function
"""
def moving_average(y, n=3) :
    sum = np.zeros(len(y))
    for i in range(len(y)-n+1):
        sum[i] = 0
        for j in range (i,i+n-1):
            sum[i] += y[j]
        sum[i] /= 3
    y = sum
    return y 

"""
ctr - mu
amp - p * 1/wid * 1/sqrt(pi)
wid - sqrt (2) * sigma
Note that if mean is not in x, the peak will not be properly visualised
"""
def gauss_func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        #print("i:")
        #print(i)
        #print(params)
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp(-((x - ctr)/wid)**2)
    return y

"""
ctr - mu
amp - p * 1/wid * 1/sqrt(pi)
wid - sqrt (2) * sigma
Note that if mean is not in x, the peak will not be properly visualised
"""
def gauss_func_2(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        mu = params[i]
        p = params[i+1]
        sigma = params[i+2]
        y = y + p * 1 / (math.sqrt(2*math.pi)*sigma) * np.exp(-((x - mu)/(math.sqrt(2)*sigma))**2)
    return y

def build_multi_gauss_from_params(*params):
    m = MultiGauss([], [])
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        mean = ctr
        deviation = wid / math.sqrt(2) 
        g = Gauss(mean=mean, deviation=deviation)
        p = amp * wid * math.sqrt(math.pi)
        m.probabilities.append(p)
        m.gaussians.append(g)
    return m

def build_multi_gauss_from_params_2(*params):
    m = MultiGauss([], [])
    for i in range(0, len(params), 3):
        mean = params[i]
        p = params[i+1]
        deviation = params[i+2]
        g = Gauss(mean=mean, deviation=deviation)
        m.probabilities.append(p)
        m.gaussians.append(g)
    return m


def find_peaks_custom(x,y,max_number):
    peaks = []
    max_x = len(x)
    for i in range (0, max_x - max_x//max_number, max_x//max_number):
        max = 0
        peak_x = 0
        for j in range(i,i + max_x//max_number):
            if y[j] > max:
                max = y[j]
                peak_x = x[j]
        peaks.append(peak_x)
    return peaks

def find_peaks_lib(x,y,min_height,width):
    peaks_x = []
    if width == None:
        peaks_i, _ = find_peaks(y, height=[min_height, 1])
    if min_height == None:
        peaks_i, _ = find_peaks(y, width=width)
    for peak_i in peaks_i:
        for i in range(len(x)):
            if (i==peak_i):
                if (x[i] > 0):
                    peaks_x.append(x[i])
                else:
                    peaks_x.append(0)
    return peaks_x

def prepare_init_param(peaks):
    init_params = []
    for peak in peaks:
        init_params.append(peak)
        init_params.append(1)
        init_params.append(1)
    return init_params

def fit_gauss(x, y, real_xs):
    min_height = 0.0001
    peaks = find_peaks_lib(x,y,min_height=min_height,width=None)

    # when peaks are narrow --> calculate mean value
    if peaks == []:
        mean = 0
        for i in range(len(x)):
            mean += x[i] * y[i]
        fit = gauss_func_2(x, *(mean, 1, 1))
        m = build_multi_gauss_from_params_2([1.0], [Gauss(mean, 1)])
    else:
        init_params = prepare_init_param(peaks)
        bounds = ([0] * len(init_params), [np.inf,1,np.inf] * (len(init_params)//3))
        try:
            popt, pcov = curve_fit(gauss_func_2, x, y, p0 = init_params, maxfev=5000000, bounds=bounds)
        except np.linalg.LinAlgError:
                print("Exception")
                print(y)
                mean = max(real_xs, key = real_xs.count)
                m = MultiGauss([1.0],[Gauss(mean, 1)])
                return m
        fit = gauss_func_2(x, *popt)
        m = build_multi_gauss_from_params_2(*popt)
        # Avoiding flat fitting curves
        while max(fit) < 0.001:
            min_height *= 2
            peaks = find_peaks_lib(x,y,min_height=min_height,width=None)
            init_params = prepare_init_param(peaks)
            bounds = ([0] * len(init_params), [np.inf,1,np.inf] * (len(init_params)//3))
            try:
                popt, p = curve_fit(gauss_func_2, x, y, p0 = init_params, maxfev=5000000, bounds=bounds)
                fit = gauss_func_2(x, *popt)
                m = build_multi_gauss_from_params_2(*popt)
            except TypeError:
                print("Exception")
                mean = max(real_xs, key = real_xs.count)
                m = MultiGauss([1.0],[Gauss(mean, 1)])
                break   
            except RuntimeError:
                print("Exception")
                mean = max(real_xs, key = real_xs.count)
                m = MultiGauss([1.0],[Gauss(mean, 1)])
                break
        

    m.remove_out_bounds_gauss(x)
    m.normalise_gauss()
    return m

def max_value(x, y):
    max = 0
    for i in range(len(x)):
        if y(x) > max:
            max = y(x)
    return max

def filter_peaks(peaks, x, y, max_number):
    filtered_peaks = []
    ind = collect_indexes_of_peaks(peaks, x)
    for i in ind:
        cnt = 0
        for j in ind:
            if y[i] < y[j]:
                cnt += 1
        if cnt < max_number:
            filtered_peaks.append(x[i])
    return filtered_peaks

def collect_indexes_of_peaks(peaks, x):
    ind = []
    for i in range(len(peaks)):
        for j in range(len(x)):
            if x[j]==peaks[i]:
                ind.append(j)
    return ind

def max_peak(x,y):
    max_i = 0
    max = 0
    for i in range(len(x)):
        if y[i] > max:
            max_i = i
            max = y[i]
    return x[max_i]

