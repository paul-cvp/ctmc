import pm4py

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