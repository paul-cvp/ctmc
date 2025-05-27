import pm4py
import pandas as pd
import numpy as np

class Timings:

    def __init__(self):
        pass

    def get_timings_per_resource(self, event_log: pd.DataFrame, timing_input_array,time_factor = 1):
        res = {}

        for event_resource_pair in timing_input_array:
            filtered_df = self.get_log_with_pair_resource(event_log, event_resource_pair[0], event_resource_pair[1],event_resource_pair[2])
            data = self.get_delta_between_events_resource(filtered_df, event_resource_pair,time_factor=time_factor)
            data = np.array(data)
            data = data[data>0]
            if len(data) > 1:
                res[event_resource_pair] = data
        return res

    def create_timing_input_array_from_log(self, event_log):
        all_conditions_for = self.create_all_possible_conditions(event_log)
        timing_input_array = []
        for event_after, events_before in all_conditions_for.items():
            for event_before in events_before:
                timing_input_array.append((event_before, event_after))
        return timing_input_array

    def get_log_with_pair_resource(self, event_log, e1, e2, resource):
        first_e1 = event_log[event_log['concept:name'] == e1].groupby('case:concept:name')[['case:concept:name', 'time:timestamp']].first().reset_index(drop=True)
        subset_is_in = first_e1.merge(event_log, on='case:concept:name', how='inner', suffixes=('_e1', ''))
        cids = subset_is_in[((subset_is_in['time:timestamp_e1'] < subset_is_in['time:timestamp'])
                             & (subset_is_in['concept:name'] == e2)
                             & (subset_is_in['org:role'] == resource) )]['case:concept:name'].unique()
        return event_log[event_log['case:concept:name'].isin(cids)].copy(deep=True)

    def get_delta_between_events_resource(self, filtered_df, event_pair, time_factor = 1):
        filtered_df['time:timestamp'] = pd.to_datetime(filtered_df['time:timestamp'], utc=True)
        filtered_df = filtered_df[(filtered_df['org:role'] == event_pair[2]) |
                                (filtered_df['concept:name'] == event_pair[1]) |
                                (filtered_df['concept:name'] == event_pair[0])].sort_values(['case:concept:name', 'time:timestamp'])
        temp_df = pd.concat([filtered_df, filtered_df.groupby('case:concept:name').shift(-1)
                            .rename({'concept:name': 'concept:name:to', 'time:timestamp': 'time:timestamp:to'}, axis=1)], axis=1)
        temp_df['delta'] = (temp_df['time:timestamp:to'] - temp_df['time:timestamp']).dt.total_seconds() * time_factor
        temp_df = temp_df[(temp_df['concept:name'] == event_pair[0]) & (temp_df['concept:name:to'] == event_pair[1])]
        data = temp_df['delta'].values
        return data

    def create_all_possible_conditions(self, event_log):
        log = pm4py.project_on_event_attribute(event_log.drop_duplicates(), case_id_key='case:concept:name')
        traces = set(tuple(i) for i in log)
        traces_list = [list(i) for i in traces]
        all_conditions_for = {}
        events = set()
        for trace in traces_list:
            preceding_events = set()
            for event in trace:
                events.add(event)
                if event not in all_conditions_for:
                    all_conditions_for[event] = set()
                all_conditions_for[event] = all_conditions_for[event].union(preceding_events)
                preceding_events.add(event)
        return all_conditions_for

    def create_resource_input_array_from_log(self, event_log):
        from simulation.util import get_transition_resource_dict

        input_array = self.create_timing_input_array_from_log(event_log)
        e1_e2_res_dict = get_transition_resource_dict(event_log)
        resource_input_array = []
        for (e1, e2) in input_array:
            if e1 in e1_e2_res_dict and e2 in e1_e2_res_dict[e1]:
                resource_list = list(e1_e2_res_dict[e1][e2].keys())
                if 'end' in resource_list:
                    resource_list.remove('end')
                for resource in resource_list:
                    resource_input_array.append((e1, e2, resource))
        return resource_input_array

    def extract_imperative_resource_times(self,event_log,time_factor=1):
        '''
        The difference between this unique way of extracting times is that it requires
        the next event to be in sequence.
        :param event_log:
        :return:
        '''
        log = pm4py.convert_to_event_log(event_log)
        times_dictionary = {}
        for trace in log:
            first = True
            for next_event in trace:
                if not first and next_event['concept:name'] != 'end' and event['concept:name'] != 'start':
                    time = next_event['time:timestamp'] - event['time:timestamp']
                    key = (event['concept:name'], next_event['concept:name'], next_event['org:role'])
                    if not key in times_dictionary.keys():
                        times_dictionary[key] = [time.total_seconds() * time_factor]
                    else:
                        times_dictionary[key].append(time.total_seconds() * time_factor)
                event = next_event
                first = False
        return times_dictionary