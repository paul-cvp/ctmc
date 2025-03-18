"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

from pm4py.objects.log.importer.xes import importer as xes_importer
import matplotlib.pyplot as plt
import statsmodels.api as sm
from simulation.markov_models import log_parser
from pm4py import discover_dfg
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import time 
from copy import deepcopy
from simulation.markov_models.mult_gauss import MultiGauss
from simulation.markov_models.gauss import Gauss
from simulation.markov_models.semi_markov_discrete import SemiMarkovDiscrete
import sys, dfg_utils, stat_utils
import numpy as np
from array import array
from scipy import signal
import pm4py
from simulation.markov_models import stat_utils
import scipy.stats as st

def extract_times_with_future(log):
    for trace in log:
        first = True
        for next_event in trace:
            if not first and next_event['concept:name'] != 'end' and event['concept:name'] != 'start':
                time = next_event['time:timestamp'] - event['time:timestamp']
                if not event['concept:name'] + '->' + next_event['concept:name'] in times_dictionary.keys():
                    times_dictionary[event['concept:name'] + '->' + next_event['concept:name']] = [time.total_seconds()//3600]
                else:
                    times_dictionary[event['concept:name'] + '->' + next_event['concept:name']].append(time.total_seconds()//3600)
            event = next_event
            first = False

def extract_times_event_log():
    times = []
    for trace in log:
        start = trace[0]['time:timestamp']
        end = trace[len(trace)-1] ['time:timestamp']   
        time = end - start
        times.append(time.total_seconds()//3600)
    return times

# retrieve distribution of values from given y
def retrieve_distribution(y):
    result = {}
    for i in y:
        count_i = y.count(i)
        result[i] = count_i
        result[i] /= len(y)
    return result

def build_semi_markov(dfg):
    
    states = set()
    transitions = set()
    out_frequences = {}

    for key in dfg.keys():
        states.add(key[0])
    
    for key1 in states:
        out_frequences[key1] = 0
        for key2 in states:
            if dfg[key1,key2] > 0:
                out_frequences[key1] += dfg[key1,key2]


    transition_times = {}
    total = 0
    max_len = 0
    for key1 in states:
        for key2 in states:
            if dfg[key1,key2] > 0:
                if ((key1 == 'start') or  (key1 == 'end') or (key2 == 'start') or  (key2 == 'end')):
                    t = (key1, key2, dfg[key1,key2]/out_frequences[key1], MultiGauss([1], [Gauss(0, 0)]))
                    transitions.add(t)
                    transition_times[(key1, key2)] = [1.0]
                else:
                    t = (key1, key2, dfg[key1,key2]/out_frequences[key1], MultiGauss([1], [Gauss(0, 0)]))
                    transitions.add(t)
                    bins = np.max(times_dictionary.get(str(key1)+"->"+str(key2))) + 1
                    transition_times[(key1, key2)] = np.histogram(times_dictionary.get(str(key1)+"->"+str(key2)), bins=int(bins), density=True)[0]
                    #ydata = build_linear_approxaimation(np.histogram(times_dictionary.get(str(key1)+"->"+str(key2)), bins=300, density=True)[1], transition_times[(key1, key2)])
                    #transition_times[(key1, key2)] = ydata
                    #print(ydata)
                    transition_times[(key1, key2)] = transition_times[(key1, key2)] / np.sum(transition_times[(key1, key2)])
                    total += len(transition_times[(key1, key2)])                     
                    if len(transition_times[(key1, key2)]) > max_len:
                        max_len = len(transition_times[(key1, key2)])
    print(max_len)
    print(total / len(transition_times))
    print(len(states))
    print(len(transitions)/(len(states)*len(states)))
    print('DFG is built')
    return SemiMarkovDiscrete(states, transitions, transition_times)

def cut_tail(list_d):
    result = 0
    for i in range(len(list_d)):
        if list_d[i] > 0:
            result = i
    return list_d[:result]

def build_linear_approxaimation(xdata, ydata):
    result = []
    for i in range(len(ydata)):
        if ydata[i] == 0:
            l = 0
            r = 0
            #find first preceding that is not zero
            for j in range(i, -1, -1):
                if ydata[j] > 0:
                    l = j
                    break

            #find first following that is not zero
            for j in range(i, len(ydata), 1):
                if ydata[j] > 0:
                    r = j
                    break
                    
            if (l > 0) and (r > 0) and (l != r):
                a = (ydata[l]-ydata[r])/(xdata[l]-xdata[r])
                b = ydata[l] - a*xdata[l]
                res = a*xdata[i] + b
            else: 
                res = 0
            if  (r > build_semi_markov0):
                res = ydata[r]
            else: 
                res = 0
            result.append(res)
        else:
            result.append(ydata[i])
    return result    

variant = xes_importer.Variants.ITERPARSE
parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
log = xes_importer.apply('logs/' + sys.argv[1], variant=variant, parameters=parameters)
reduction_times = {}
kl_divergences = {}

#log = pm4py.filter_case_size(log, 4, 4)
event_log_times = extract_times_event_log()

filtered_event_log_times = []
for times in event_log_times:
    if times < 1200:
        filtered_event_log_times.append(times)


for k in [5]:
 
    print()
    print("Order: k=" + str(k))

    start = time.time()
    log_for_discovery = deepcopy(log)
    times_dictionary = {}
    log_processed = log_parser.prepare_log(log_for_discovery, k)
    dfg, start_activities, end_activities = discover_dfg(log_processed)
    dfg["end", "start"] = 1

    end = time.time()
    print()
    print("Discovery time:")
    print(end-start)

   
    
    "Full analysis"
    
    start = time.time()
    extract_times_with_future(log_processed)
    """
    conv = [1.0]
    count = 0
    for key in sorted(times_dictionary.keys()):
        count += 1
        data = np.histogram(times_dictionary.get(key), bins=30000, density=True)
        #for i in range(1, len(data[0])):
        #    if data[0][i] == 0:
        #        data[0][i] = data[0][i-1]
        l_data = list(data[0]) / sum (data[0])
        #l_data = build_linear_approxaimation(data[1], l_data)
        print(l_data)
        #conv = [i * 0.1 for i in conv]
        #l_data = [i * 0.9 for i in l_data]
        conv = np.convolve(conv,l_data,'full')
        print("Convolution:")
        print(len(conv))
        print(conv)
        plt.plot(conv, color="tab:red", label='Convolution')
        plt.show()
        #fit_gammas(data[1], l_data)
    """
    
    #print("end")
    """
    Fitting using Gaussian KDE
    """
    #mult_gausses = {}
    #for key in sorted(times_dictionary.keys()):
    #    kde = sm.nonparametric.KDEUnivariate(times_dictionary.get(key))
    #    kde.fit(bw=4, kernel='gau')  # Estimate the densities
    #    multi_gauss = fit_gauss(kde.support, kde.density, times_dictionary.get(key))
    #    mult_gausses[str([key.partition('->')[0], key.partition('->')[2]])] = multi_gauss

#   end = time.time()
#    print()
#    print("Fitting time:")
#    print(end - start)

    semi_markov = build_semi_markov(dfg)
    print("Number of states: " + str(len(semi_markov.states)))
    print("Number of transitions: " + str(len(semi_markov.transitions)))
    states = deepcopy(semi_markov.states)
    
    print("Simulation...")
    sim_times = semi_markov.simulate(times_dictionary)

    #y_sim, x_sim, _ = plt.hist(sim_times, bins=int(max(sim_times)), fc="tab:red", density=True, label='Simulation')
    #discrete_kl_divergence_simulation = stat_utils.discrete_kl_divergence(y_sim, event_log_times, 20)
    #print("KL Divergence Simulation:")
    #print(discrete_kl_divergence_simulation) 
                 

    
    y, x, _ = plt.hist(filtered_event_log_times, bins=500, fc="orange", density=True, label='Event log')
    y_sim, x_sim, _ = plt.hist(sim_times, bins=int(max(sim_times)), fc="tab:red", density=True, label='Simulation')



    plt.xlim([-10, 1000])
    plt.legend(loc="upper right")
    plt.title('')
    plt.xlabel('Overall time in hours')
    plt.ylabel('Probability')
    plt.show()


    start = time.time()
    while len(semi_markov.states) > 2:
        #print("Selecting node...")
        next_state = semi_markov.select_next_state()
        #print(next_state)
        semi_markov.reduce_node(next_state)
        #print("Reduced")
    end = time.time()     
    print()
    print("Reduction time:")
    print(end-start)
    if k not in reduction_times:
        reduction_times[k] = {end-start}      
    else:
        reduction_times[k].add(end-start)

    for transition in semi_markov.transitions:
        if transition[0] == 'start':
            times = semi_markov.transition_times[(transition[0], transition[1])]
            print(len(times))
            #print(times)
            #print(y_sim)
            discrete_kl_divergence = stat_utils.discrete_kl_divergence(times, event_log_times, 20)
            discrete_kl_divergence_simulation = stat_utils.discrete_kl_divergence(y_sim, event_log_times, 20)
            print("KL Divergence:")
            print(discrete_kl_divergence) 
            print("KL Divergence Simulation:")
            print(discrete_kl_divergence_simulation) 
            if k not in kl_divergences:
                kl_divergences[k] = {discrete_kl_divergence}
            else:
                kl_divergences[k].add(discrete_kl_divergence)
            print()
            plt.plot(times, label="Discrete Semi-Markov Model order="+str(k))

for k in [2]:
    print()
    print("Metrics for order " + str(k) + ":")
    print()

    reduction_times_values = reduction_times[k]
    print(reduction_times_values)
    reduction_times_average = np.mean(list(reduction_times_values))
    print("Reduction times average:")
    print(reduction_times_average)
    reduction_times_interval = st.t.interval(0.95, df=len(reduction_times_values)-1, 
              loc=np.mean(list(reduction_times_values)), 
              scale=st.sem(list(reduction_times_values)))
    print("Reduction times interval:")
    print(reduction_times_interval[1]-reduction_times_average)
    print()

    kl_divergence_values = kl_divergences[k]
    print(kl_divergence_values)
    kl_divergence_average = np.mean(list(kl_divergence_values))
    print("KL-divergence average:")
    print(kl_divergence_average)
    kl_divergence_interval = st.t.interval(0.95, df=len(kl_divergence_values)-1, 
              loc=np.mean(list(kl_divergence_values)), 
              scale=st.sem(list(kl_divergence_values)))
    print("KL-divergence interval:")
    print(kl_divergence_interval[1]-kl_divergence_average)
    print()                

"""
Plotting event log
"""

cm = plt.cm.get_cmap('OrRd')
y, x, _ = plt.hist(filtered_event_log_times, bins=200, fc=cm(0.25), density=True, label='Event log')


 
plt.xlim([-10, 1000])
plt.legend(loc="upper right")
plt.title('')
plt.xlabel('Overall time in hours')
plt.ylabel('Probability')
plt.show()



