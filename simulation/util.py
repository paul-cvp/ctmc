import pm4py
import scipy
import json
import numpy as np
from datetime import timedelta

from copy import deepcopy
import pandas as pd

def scale_event_log_time(event_log, multiplicity=2):
    new_el = []
    for case, temp_df in event_log.groupby('case:concept:name'):
        temp_df['time:timestamp:to'] = temp_df['time:timestamp'].shift(-1).fillna(temp_df['time:timestamp'])
        temp_df['time:delta'] = temp_df['time:timestamp:to'] - temp_df['time:timestamp']
        temp_df['time:delta'] = temp_df['time:delta']*multiplicity
        time_delta = temp_df[['concept:name','time:delta','org:role','org:resource']].T.to_dict()
        current_timestamp = temp_df['time:timestamp'].min()
        for e, t in time_delta.items():
            t_copy = deepcopy(t)
            new_el.append({**t_copy, 'time:timestamp':current_timestamp,'case:concept:name':case})
            current_timestamp += t['time:delta']
    return pd.DataFrame(new_el)

def prepare_input_for_analysis(subset_el, final_states):
    from simulation.timings import Timings
    data_transition_role_frequency = get_transition_resource_dict(subset_el)
    mine_declaratively = True
    if mine_declaratively:
        timings = Timings()
        resource_input_array = timings.create_resource_input_array_from_log(subset_el)
        res_timings = timings.get_timings_per_resource(subset_el, resource_input_array)
        times_dictionary = res_timings
    else:
        timings = Timings()
        times_dictionary = timings.extract_resource_times_with_future(subset_el)
    data_mean_transition_role_time = {}
    tuples_to_discard = set()
    for k, v in data_transition_role_frequency.items():
        if k in ['start', 'end']:
            continue
        for k2, v2 in v.items():
            if k2 in ['start', 'end']:
                continue
            all_freq = 0
            for k3, v3 in v2.items():
                all_freq += v3
                if (k, k2, k3) in times_dictionary:
                    times = times_dictionary[(k, k2, k3)]
                    times = np.array(times)
                    times = times // 3600
                    times = times[times != 0]
                    if len(times) > 1:  # only take times that have more than 1 value
                        expon_loc, expon_scale = scipy.stats.expon.fit(times)

                        # f = Fitter(times, distributions=['expon'])
                        # f.fit()
                        # best = f.get_best()['expon']
                        # expon_loc_fitter, expon_scale_fitter = best['loc'], best['scale']

                        if expon_scale > 0:  # do not take times that cannot be fit into an exponential
                            rate = 1 / expon_scale
                            if k not in data_mean_transition_role_time:
                                data_mean_transition_role_time[k] = {}
                            if k2 not in data_mean_transition_role_time[k]:
                                data_mean_transition_role_time[k][k2] = {}
                            if k3 not in data_mean_transition_role_time[k][k2]:
                                data_mean_transition_role_time[k][k2][k3] = {
                                    # 'loc': expon_loc_fitter,
                                    # 'scale': expon_scale_fitter,
                                    'loc': expon_loc,
                                    'scale': expon_scale,
                                    'lambda': rate
                                }
                        else:
                            print(k, k2, k3)
                            tuples_to_discard.add((k, k2, k3))
                            print(times)
                    else:
                        print(k, k2, k3)
                        tuples_to_discard.add((k, k2, k3))
                        print(times)
    for (e_from, e_to, role) in tuples_to_discard:
        if e_from in data_transition_role_frequency:
            if e_to in data_transition_role_frequency[e_from]:
                if role in data_transition_role_frequency[e_from][e_to]:
                    data_transition_role_frequency[e_from][e_to].pop(role)
    for e_from in data_transition_role_frequency.keys():
        for e_to in data_transition_role_frequency.keys():
            if (e_from == 'start' and e_to == 'start') or (e_from == 'end' and e_to == 'end'):
                data_transition_role_frequency[e_from].pop(e_to)

    def remove_empty_keys(d):
        """Recursively remove empty keys from a three-level nested dictionary."""
        if not isinstance(d, dict):
            return d  # Return non-dict values as they are

        cleaned_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                cleaned_value = remove_empty_keys(value)  # Recursively clean sub-dictionaries
                if cleaned_value:  # Add only if not empty
                    cleaned_dict[key] = cleaned_value
            elif value not in (None, "", [], {}, ()):  # Ignore empty values
                cleaned_dict[key] = value

        return cleaned_dict

    data_transition_role_frequency = remove_empty_keys(data_transition_role_frequency)

    role_resources = get_detailed_weighted_role(subset_el)

    states = set(subset_el['concept:name'].unique()).difference(set(['start', 'end']))
    n = len(states)
    i = 0
    correspondence = {s: i for s, i in zip(states, range(len(states)))}

    non_final_states = list(states.difference(set(final_states)))
    for s in final_states:
        if correspondence[s] == 0:
            correspondence[s] = correspondence[non_final_states[0]]
            correspondence[non_final_states[0]] = 0

    return correspondence, role_resources, data_mean_transition_role_time, data_transition_role_frequency

def sanity_check(sc_data,final_states):
    res = []
    for label, el in sc_data.items():
        correspondence, role_weighted, mean_res_dict, e1_e2_res_dict = prepare_input_for_analysis(el,final_states)
        mean, median, std = get_pm4py_reference_times(el)
        print("Reference times")
        print(f"[{label}] Mean: {timedelta(seconds=mean)}")
        print(f"[{label}] Std: {timedelta(seconds=std)}")
        print(f"[{label}] median: {timedelta(seconds=median)}")
        print()
        role_trials = {k:int(v) for k,v in role_weighted.items()}
        result = get_stormpy_analysis_times(correspondence,final_states, role_trials, role_weighted, e1_e2_res_dict, mean_res_dict)
        print("Analysis times")
        print(f"[{label}] Raw result: {result}")
        offset = mean - result
        if result < np.inf:
            dt_result = timedelta(seconds=result)
            dt_offset = timedelta(seconds=offset)
        else:
            dt_result = result
            dt_offset = offset
        print(f"[{label}] Duration: {dt_result}")
        print(f"[{label}] Delta: {dt_offset}")
        res.append({'label':label,
                    'ref-mean': timedelta(seconds=mean),
                    'ref-median': timedelta(seconds=median),
                    'ref-std': timedelta(seconds=std),
                    'analysis-time':dt_result,
                    'offset':dt_offset})
    return pd.DataFrame(res)

def get_stormpy_analysis_times(correspondence,final_states, role_trials, role_weighted, e1_e2_res_dict, mean_res_dict):
    from simulation.ctmc import create_prism_program_from_log
    import stormpy

    probabilities = create_prism_program_from_log(
                                                  correspondence,
                                                  final_states,
                                                  mean_res_dict,
                                                  role_weighted,
                                                  e1_e2_res_dict,
                                                  role_trials,
                                                  'ctmc.sm')
    prism_program = stormpy.parse_prism_program('ctmc.sm',prism_compat=True,simplify=True)
    model = stormpy.build_model(prism_program)

    labels = ""
    for fs in final_states:
        labels += f'"q_terminal_{fs}" |'
    labels = labels[:-2]

    formula_str = f'R=? [F {labels}]'
    properties = stormpy.parse_properties(formula_str, prism_program)
    result = stormpy.model_checking(model, properties[0])
    initial_state = model.initial_states[0]
    result = result.at(initial_state)
    return result

def get_pm4py_reference_times(event_log):
    case_durations = pm4py.stats.get_all_case_durations(event_log)
    n = len(case_durations)
    median = np.median(case_durations)
    mean = np.mean(case_durations)
    std = np.std(case_durations)

    z_critical = scipy.stats.norm.ppf(q=0.975)  # Get the z-critical value*
    margin_of_error = z_critical * (std / np.sqrt(n))
    return mean, median, margin_of_error

def get_transition_resource_dict(event_log):
    res_dict = {}
    for case, events in event_log.groupby('case:concept:name',sort=False):
        events['concept:name:next'] = events['concept:name'].shift(-1).fillna('end')
        events['org:role:next'] = events['org:role'].shift(-1).fillna('end')
        s1 = 'start'
        s2 = events.reset_index().T.to_dict()[0]['concept:name']
        role = events.reset_index().T.to_dict()[0]['org:role']
        if s1 not in res_dict:
            res_dict[s1] = {}
        if s2 not in res_dict[s1]:
            res_dict[s1][s2] = {}
        if role not in res_dict[s1][s2]:
            res_dict[s1][s2][role] = 0
        res_dict[s1][s2][role] += 1

        temp_dict = events[['concept:name','concept:name:next','org:role:next']].T.to_dict()
        for k,v in temp_dict.items():
            s1 = v['concept:name']
            s2 = v['concept:name:next']
            role = v['org:role:next']
            if s1 not in res_dict:
                res_dict[s1] = {}
            if s2 not in res_dict[s1]:
                res_dict[s1][s2] = {}
            if role not in res_dict[s1][s2]:
                res_dict[s1][s2][role] = 0
            res_dict[s1][s2][role] += 1
    return res_dict

def get_detailed_weighted_role(event_log):
    resource_activity = event_log['org:resource'].value_counts().to_dict()

    res_dict = {}
    for k, v in event_log[['org:resource', 'org:role']].drop_duplicates().groupby('org:role'):
        res_dict[k] = v['org:resource'].to_list()

    resource_in_role = {}
    for resource, roles in event_log[['org:resource', 'org:role']].groupby('org:resource'):
        resource_in_role[resource] = roles['org:role'].value_counts().to_dict()

    role_weighted = {}
    for role, resources in res_dict.items():
        role_weighted[role] = 0
        for resource in resources:
            if role in resource_in_role[resource]:
                res_act = resource_activity[resource]
                res_in_role_act = resource_in_role[resource][role]
                role_weighted[role] += res_in_role_act / res_act  # resource in role divided by resource presence in log
    return role_weighted