"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""
import pm4py
import statsmodels.api as sm

from simulation.markov_models.fit_distribution import fit_gauss
from simulation.markov_models import log_parser
from copy import deepcopy
from simulation.markov_models.mult_gauss import MultiGauss
from simulation.markov_models.gauss import Gauss
from simulation.markov_models.semi_markov import SemiMarkov, SemiMarkovTransition, SemiMarkovState

from fitter import Fitter, get_common_distributions

def extract_times_with_future(log):
    times_dictionary = {}
    for trace in log:
        first = True
        for next_event in trace:
            if not first and next_event['concept:name'] != 'end' and event['concept:name'] != 'start':
                time = next_event['time:timestamp'] - event['time:timestamp']
                transition_encoding = SemiMarkovTransition(event['concept:name'], next_event['concept:name'])

                if not hash(transition_encoding) in times_dictionary.keys():
                    times_dictionary[hash(transition_encoding)] = []
                times_dictionary[hash(transition_encoding)].append(time.total_seconds() // 3600)
            event = next_event
            first = False

    return times_dictionary

def extract_resources(log, resource_column="org:role"):
    resources_dictionary = {}
    for event, resources in log.groupby('concept:name', sort=False):
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
        times.append(time.total_seconds() // 3600) # times in hours
    return times


# retrieve distribution of values from given y
def retrieve_distribution(y):
    result = {}
    for i in y:
        count_i = y.count(i)
        result[i] = count_i
        result[i] /= len(y)
    return result


def build_semi_markov(dfg, multi_gausses_time, multi_resources=None):
    states = set()
    transitions = set()
    out_frequences = {}
    for key in dfg.keys():
        if (key[0] == 'start') or (key[0] == 'end'):
            states.add(SemiMarkovState(key[0],{'resources':[None],'p':[1]}))
        else:
            states.add(SemiMarkovState(key[0],multi_resources[key[0]]))

    for key1 in states:
        out_frequences[key1.state] = 0
        for key2 in states:
            if dfg[key1.state, key2.state] > 0:
                out_frequences[key1.state] += dfg[key1.state, key2.state]
    for key1 in states:
        for key2 in states:
            if dfg[key1.state, key2.state] > 0:
                if ((key1.state == 'start') or (key1.state == 'end') or (key2.state == 'start') or (key2.state == 'end')):
                    transitions.add(SemiMarkovTransition(key1,
                                           key2,
                                           dfg[key1.state, key2.state] / out_frequences[key1.state],
                                           MultiGauss([1], [Gauss(0, 0)])))
                else:
                    transitions.add(SemiMarkovTransition(key1,
                                           key2,
                                           dfg[key1.state, key2.state] / out_frequences[key1.state],
                                           multi_gausses_time[hash((key1,key2))]))
    return SemiMarkov(states, transitions)


def apply(log_df,use_kde=True):
    log = pm4py.convert_to_event_log(log_df)

    log_for_discovery = deepcopy(log)
    log_processed = log_parser.add_start_end(log_for_discovery)
    dfg, start_activities, end_activities = pm4py.discover_dfg(log_processed)
    dfg["end", "start"] = 1

    times_dictionary = extract_times_with_future(log_processed)
    multinomial_resources = extract_resources(log_df)

    dist_fits = {}
    if use_kde:
        #Fitting using Gaussian KDE
        for key in sorted(times_dictionary.keys()):
            kde = sm.nonparametric.KDEUnivariate(times_dictionary.get(key))
            kde.fit(bw=4, kernel='gau')  # Estimate the densities
            multi_gauss = fit_gauss(kde.support, kde.density, times_dictionary.get(key))
            dist_fits[key] = multi_gauss
    else:
        for k, values in times_dictionary.items():
            f = Fitter(values, distributions=get_common_distributions())
            f.fit()
            best_dist, fitted_params = f.get_best().popitem()
            dist_fits[k] = {'best_dist': best_dist, 'fitted_params': fitted_params}

    semi_markov = build_semi_markov(dfg, dist_fits, multinomial_resources)

    return semi_markov.to_json()