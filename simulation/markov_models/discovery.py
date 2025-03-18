"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""
import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import timedelta
from simulation.markov_models.fit_distribution import fit_gauss
from simulation.markov_models import log_parser
from pm4py import discover_dfg
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import time 
from copy import deepcopy
from simulation.markov_models.mult_gauss import MultiGauss
from simulation.markov_models.gauss import Gauss
from simulation.markov_models.semi_markov import SemiMarkov
from simulation.markov_models import dfg_utils, stat_utils
import numpy as np, scipy.stats as st

from simulation import util as sim_util

def extract_resource_times_with_future(event_log):
    log = pm4py.convert_to_event_log(event_log)
    times_dictionary = {}
    for trace in log:
        first = True
        for next_event in trace:
            if not first and next_event['concept:name'] != 'end' and event['concept:name'] != 'start':
                time = next_event['time:timestamp'] - event['time:timestamp']
                key = (event['concept:name'], next_event['concept:name'], next_event['org:role'])
                if not key in times_dictionary.keys():
                    times_dictionary[key] = [time.total_seconds()]
                else:
                    times_dictionary[key].append(time.total_seconds())
            event = next_event
            first = False
    return times_dictionary

def extract_times_with_future(log):
    times_dictionary = {}
    for trace in log:
        first = True
        for next_event in trace:
            if not first and next_event['concept:name'] != 'end' and event['concept:name'] != 'start':
                time = next_event['time:timestamp'] - event['time:timestamp']
                if not event['concept:name'] + '->' + next_event['concept:name'] in times_dictionary.keys():
                    times_dictionary[event['concept:name'] + '->' + next_event['concept:name']] = [time.total_seconds() // 3600]
                else:
                    times_dictionary[event['concept:name'] + '->' + next_event['concept:name']].append(time.total_seconds() // 3600)
            event = next_event
            first = False
    return times_dictionary

def extract_resources(log, resource_column="org:resource"):
    resources_dictionary = {}
    df_log = pm4py.convert_to_dataframe(deepcopy(log))
    for event, resources in df_log.groupby('concept:name', sort=False):
        # NOTE: by default group by will sort the values
        resources_choice = resources.groupby(resource_column, sort=False ,dropna=False).count()[['concept:name']].to_dict()['concept:name']
        total = len(resources)
        resources_list = []
        p_vals = []
        for resource, val in resources_choice.items():
            resources_list.append(resource)
            p_vals.append(val / total)
        resources_dictionary[event] = {'resources': resources_list, 'p': p_vals }
    return resources_dictionary

def extract_times_event_log(log):
    times = []
    for trace in log:
        start = trace[0]['time:timestamp']
        end = trace[len(trace) - 1]['time:timestamp']
        time = end - start
        times.append(time.total_seconds() // 3600)
    return times


# retrieve distribution of values from given y
def retrieve_distribution(y):
    result = {}
    for i in y:
        count_i = y.count(i)
        result[i] = count_i
        result[i] /= len(y)
    return result


def build_semi_markov(dfg, multi_gausses_time, multi_gausses_resources=None):
    states = set()
    transitions = set()
    out_frequences = {}

    for key in dfg.keys():
        states.add(key[0])

    for key1 in states:
        out_frequences[key1] = 0
        for key2 in states:
            if dfg[key1, key2] > 0:
                out_frequences[key1] += dfg[key1, key2]

    for key1 in states:
        for key2 in states:
            if dfg[key1, key2] > 0:
                if ((key1 == 'start') or (key1 == 'end') or (key2 == 'start') or (key2 == 'end')):
                    transitions.add(tuple([key1, key2, dfg[key1, key2] / out_frequences[key1], MultiGauss([1], [Gauss(0, 0)])]))
                else:
                    transitions.add(tuple([key1, key2, dfg[key1, key2] / out_frequences[key1],
                                           multi_gausses_time["['" + str(key1) + "', '" + str(key2) + "']"]]))
    print()
    print('DFG is built')
    return SemiMarkov(states, transitions)

def add_timestamps_to_simulated_log(mc_sim_log):
    simulated_timed_dict = {
        'case:concept:name': [],
        'concept:name': [],
        'time:timestamp': []
    }

    for i, case in mc_sim_log.groupby('case:concept:name'):
        events = case[['concept:name', 'time:since:start']].reset_index().T.to_dict()
        start_timestamp = case['time:timestamp'].to_list()[0]
        simulated_timed_dict['case:concept:name'].append(i)
        simulated_timed_dict['concept:name'].append(events[0]['concept:name'])
        simulated_timed_dict['time:timestamp'].append(start_timestamp)
        for k, data in events.items():
            if k != 0:
                simulated_timed_dict['case:concept:name'].append(i)
                simulated_timed_dict['concept:name'].append(data['concept:name'])
                current_timestamp = start_timestamp + timedelta(hours=data['time:since:start'])
                current_timestamp = sim_util.check_is_workday(current_timestamp)
                while not sim_util.is_time_within_work_hours(current_timestamp):
                    start_timestamp += timedelta(hours=1)
                    current_timestamp = start_timestamp + timedelta(hours=data['time:since:start'])
                simulated_timed_dict['time:timestamp'].append(current_timestamp)
    simulated_timed_log = pd.DataFrame(simulated_timed_dict)
    return simulated_timed_log

def apply(log_df, number_of_traces, trace_inter_arrival_list):
    log = pm4py.convert_to_event_log(log_df)
    discovery_times = {}
    fitting_times = {}
    reduction_times = {}
    kl_divergences = {} #TODO see how the kl divergence is used here

    event_log_times = extract_times_event_log(log)

    filtered_event_log_times = []
    for times in event_log_times:
        if times < 1200:
            filtered_event_log_times.append(times)

    for k in [1]:

        print()
        print("Order: k=" + str(k))

        start = time.time()
        log_for_discovery = deepcopy(log)
        log_processed = log_parser.prepare_log(log_for_discovery, k)
        # print(log_processed)
        dfg, start_activities, end_activities = discover_dfg(log_processed)
        dfg["end", "start"] = 1

        end = time.time()
        print()
        print("Discovery time:")
        print(end - start)
        if k not in discovery_times:
            discovery_times[k] = {end - start}
        else:
            discovery_times[k].add(end - start)

        "Express analysis"

        # cut the log to get better precision of the limiting probabilities
        number_of_chunks = len(log_for_discovery)
        overall_times = []
        # cnt = 0
        temp_log_for_discovery = deepcopy(log_for_discovery)
        for traces in np.array_split(np.array(temp_log_for_discovery, dtype=object), number_of_chunks):
            processed_traces = log_parser.prepare_log(traces, k)
            # for i in range(len(traces[0])):
            #    print(traces[0][i])
            # print(cnt)
            #    cnt += 1
            dfg_express = dfg_discovery.apply(traces, variant=dfg_discovery.Variants.FREQUENCY)
            dfg_express["end", "start"] = 1
            log_activities = log_parser.log_activities(traces)
            times = log_parser.calculate_times(traces)
            means = stat_utils.calculate_means(dfg_express, times, log_activities)
            # print(means)
            limiting_probabilities = dfg_utils.calculate_limiting_probabilities(dfg_express, log_activities)
            # print(limiting_probabilities)
            overall_time = 0
            for i in range(0, len(log_activities)):
                overall_time += limiting_probabilities[log_activities[i]] * means[log_activities[i]]
            overall_time /= limiting_probabilities['start']
            overall_times.append(overall_time)
        estimated_mean_time = np.average(overall_times)

        print(str(round(estimated_mean_time // 86400)) + 'd ' + str(round(estimated_mean_time % 86400 // 3600)) + 'h ' + str(round(estimated_mean_time % 3600 // 60)) + 'm ' + str(
            round(estimated_mean_time % 60)) + 's ')

        "Full analysis"

        start = time.time()
        times_dictionary = extract_times_with_future(log_processed)
        multinomial_resources = extract_resources(log_df)

        """
        Fitting time using Gaussian KDE
        """
        mult_gausses_time = {}
        for key in sorted(times_dictionary.keys()):
            kde = sm.nonparametric.KDEUnivariate(times_dictionary.get(key))
            kde.fit(bw=4, kernel='gau')  # Estimate the densities
            multi_gauss = fit_gauss(kde.support, kde.density, times_dictionary.get(key))
            mult_gausses_time[str([key.partition('->')[0], key.partition('->')[2]])] = multi_gauss

        end = time.time()
        print()
        print("Fitting time duration:")
        print(end - start)

        if k not in fitting_times:
            fitting_times[k] = {end - start}
        else:
            fitting_times[k].add(end - start)

        semi_markov = build_semi_markov(dfg, mult_gausses_time, multinomial_resources)

        print("Number of states: " + str(len(semi_markov.states)))
        print("Number of transitions: " + str(len(semi_markov.transitions)))
        state_degrees = semi_markov.state_degrees()
        avg_state_degree = np.average(state_degrees)
        print("Average state degree: " + str(avg_state_degree))
        max_state_degree = np.max(state_degrees)
        print("Max state degree: " + str(max_state_degree))

        print("Simulation...")
        times, sim_log_dict_list = semi_markov.simulate(number_of_traces=number_of_traces)
        simulated_log = pd.DataFrame(sim_log_dict_list)
        simulated_log.loc[simulated_log['time:since:start'] == 0, 'time:timestamp'] = trace_inter_arrival_list
        simulated_timed_log = add_timestamps_to_simulated_log(simulated_log)
        print(times)

        states = deepcopy(semi_markov.states)

        start = time.time()
        while len(semi_markov.states) > 2:
            next_state = semi_markov.select_next_state()
            semi_markov.reduce_node(next_state)
        end = time.time()
        print()
        print("Reduction time:")
        print(end - start)
        if k not in reduction_times:
            reduction_times[k] = {end - start}
        else:
            reduction_times[k].add(end - start)

        for transition in semi_markov.transitions:
            if transition[0] == 'start':
                multi_gauss = transition[3]
                multi_gauss.remove_zero()
                color = {
                    1: "tab:red",
                    2: "tab:blue",
                    3: "k",
                    4: "tab:green",
                    5: "tab:purple"
                }
                multi_gauss.plot_trunc_mult_gauss(range(-10, 400, 1), label="Semi-Markov Model, order=" + str(k), color=color.get(k))
                print()
                print("Peaks:")
                print(multi_gauss.calculate_peaks())

                print()
                print("KL Divergence:")
                kl_divergence = multi_gauss.calc_kl_divergence(20, filtered_event_log_times, event_log_times)
                print(kl_divergence)
                if k not in kl_divergences:
                    kl_divergences[k] = {kl_divergence}
                else:
                    kl_divergences[k].add(kl_divergence)
                print()

    for k in [1]:
        print()
        print("Metrics for order " + str(k) + ":")
        discovery_times_values = discovery_times[k]
        print(discovery_times_values)
        discovery_times_average = np.mean(list(discovery_times_values))
        print("Discovery times average:")
        print(discovery_times_average)
        discovery_times_interval = st.t.interval(0.95, df=len(discovery_times_values) - 1,
                                                 loc=np.mean(list(discovery_times_values)),
                                                 scale=st.sem(list(discovery_times_values)))
        print("Discovery times interval:")
        print(discovery_times_interval[1] - discovery_times_average)
        print()

        fitting_times_values = fitting_times[k]
        print(fitting_times_values)
        fitting_times_average = np.mean(list(fitting_times_values))
        print("Fitting times average:")
        print(fitting_times_average)
        fitting_times_interval = st.t.interval(0.95, df=len(fitting_times_values) - 1,
                                               loc=np.mean(list(fitting_times_values)),
                                               scale=st.sem(list(fitting_times_values)))
        print("Fitting times interval:")
        print(fitting_times_interval[1] - fitting_times_average)
        print()

        reduction_times_values = reduction_times[k]
        print(reduction_times_values)
        reduction_times_average = np.mean(list(reduction_times_values))
        print("Reduction times average:")
        print(reduction_times_average)
        reduction_times_interval = st.t.interval(0.95, df=len(reduction_times_values) - 1,
                                                 loc=np.mean(list(reduction_times_values)),
                                                 scale=st.sem(list(reduction_times_values)))
        print("Reduction times interval:")
        print(reduction_times_interval[1] - reduction_times_average)
        print()

        kl_divergence_values = kl_divergences[k]
        print(kl_divergence_values)
        kl_divergence_average = np.mean(list(kl_divergence_values))
        print("KL-divergence average:")
        print(kl_divergence_average)
        kl_divergence_interval = st.t.interval(0.95, df=len(kl_divergence_values) - 1,
                                               loc=np.mean(list(kl_divergence_values)),
                                               scale=st.sem(list(kl_divergence_values)))
        print("KL-divergence interval:")
        print(kl_divergence_interval[1] - kl_divergence_average)
        print()

    """
    Plotting event log
    """

    cm = plt.cm.get_cmap('OrRd')
    y, x, _ = plt.hist(filtered_event_log_times, bins=150, fc=cm(0.25), density=True, label='Event log')

    plt.xlim([-10, 400])
    plt.legend(loc="upper right")
    plt.title('')
    plt.xlabel('Overall time in hours')
    plt.ylabel('Probability')
    plt.show()

    return semi_markov, multi_gauss, simulated_timed_log
