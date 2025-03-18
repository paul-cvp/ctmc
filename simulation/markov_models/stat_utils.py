"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

import statistics
import numpy as np

def calculate_means(dfg,times,log_activities):
    all_means = {}
    for activity1 in log_activities:
        i = 0
        mean = 0
        for activity2 in log_activities:
            if dfg[activity1,activity2] > 0:
                #print(times[activity1+";"+activity2])
                for time in times[activity1+";"+activity2]:
                    mean += time.total_seconds()
                    i += 1
        mean /= i
        all_means[activity1]=mean
    return all_means

def calculate_standard_deviation_times(dfg,times,log_activities):
    all_standard_deviation_times = {}
    all_times = {}
    for activity1 in log_activities:
        for activity2 in log_activities:
            if dfg[activity1,activity2] > 0:
                all_times[activity1] = []
                for time in times[activity1+";"+activity2]:
                    all_times[activity1].append(time.total_seconds())

    for activity1 in all_times:
            times = all_times[activity1]
            if (len(times) > 1):
                all_standard_deviation_times[activity1] = statistics.stdev(times)
            else:
                all_standard_deviation_times[activity1] = 0

    return all_standard_deviation_times

def calc_observed(lower_bound, upper_bound, samp):
    cnt = 0
    for s in samp:
        if s < upper_bound and s >= lower_bound:
            cnt += 1
    return cnt

def discrete_kl_divergence(y, samp, bins):
    kl_divergence = 0
    step = len(y)/bins
    #print("Step:")
    #print(step)

    for i in range(0, bins-1, 1):
        expected = 0
        lower_bound = i*step
        upper_bound = (i+1)*step
        
        for j in range (int(lower_bound), int(upper_bound), 1):
            expected += (y[j] + y[j+1]) / 2

        observed = calc_observed(lower_bound, upper_bound, samp)/len(samp)   
        #print(expected)
        #print(observed)
        if observed != 0 and expected != 0:
            kl_divergence += observed*(np.log(observed/expected))
    return kl_divergence       