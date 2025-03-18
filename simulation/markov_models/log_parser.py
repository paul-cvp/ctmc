"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

from pm4py.objects.log.obj import Event
import datetime


# k is the order of the model
def prepare_log(log, k):
    add_order(log, k)
    add_start_end(log)
    return log

# adds start and end activities to the log
def add_start_end(log):
    for trace in log:
        sts = trace[0]['time:timestamp']
        srole = trace[0]['org:role']
        sresource = trace[0]['org:resource']
        start_event = Event({"concept:name":"start",'time:timestamp':sts,'org:role':srole,'org:resource':sresource})

        n = len(trace)-1
        ets = trace[n]['time:timestamp']
        erole = trace[n]['org:role']
        eresource = trace[n]['org:resource']
        end_event =  Event({"concept:name":"end",'time:timestamp':ets,'org:role':erole,'org:resource':eresource})
        trace.insert(0,start_event)
        trace.append(end_event)
    return log

def max_length(log):
    max_length = 0
    for i in range(0, len(log)):
        if(len(log[i]) > max_length):
            max_length = len(log[i])
    return max_length

def add_order(log,k):
    a = []
    if k<=1:
        return
    else:
        for i in range(0, len(log)):
           a.append([])
           for j in range(0,len(log[i])):
               a[i].append(log[i][j]['concept:name'])
               for l in range(1,k):
                   if j-l >= 0:
                    a[i][j] = log[i][j-l]['concept:name'] + "," + a[i][j]
    
    for i in range(0, len(log)):
        for j in range(0,len(log[i])):
            log[i][j]['concept:name'] = a[i][j]

def parse(log, k):
    for i in range(0, len(log)):
        occur = {}
        for j in range(0,len(log[i])):
            if log[i][j]['concept:name'] in occur:
                if occur[log[i][j]['concept:name']] < k:
                    occur[log[i][j]['concept:name']] += 1
            else:
                occur[log[i][j]['concept:name']] = 1
            log[i][j]['concept:name'] = log[i][j]['concept:name'] + '_' + str(occur[log[i][j]['concept:name']])
    return log

def log_activities(log):
    all_log_activities = []
    for i in range(0, len(log)):
        for j in range(0,len(log[i])):
            if all_log_activities.count(log[i][j]['concept:name']) == 0:
                all_log_activities.append(log[i][j]['concept:name'])
    return all_log_activities

def calculate_times(log):
    dict_time = {}
    for i in range(0, len(log)):
        for j in range(0,len(log[i])-1):
            name1 = log[i][j]['concept:name']
            name2 = log[i][j+1]['concept:name']
            if ('time:timestamp' in log[i][j]) and ('time:timestamp' in log[i][j+1]):
                time = log[i][j+1]['time:timestamp'] - log[i][j]['time:timestamp']
                if name1 + ";" + name2 in dict_time:
                    dict_time[name1 + ";" + name2].append(time)
                else:
                    dict_time[name1 + ";" + name2] = [time]
            else:
                dict_time[name1 + ";" + name2] = [datetime.timedelta(days=0, seconds=0)]
    dict_time["end;start"] = [datetime.timedelta(days=0, seconds=0)]
    return dict_time

def calculate_real_overall_time(log):
    all_time = 0
    for i in range(0, len(log)):
        if ('time:timestamp' in log[i][len(log[i])-2]) and ('time:timestamp' in log[i][1]):
            time = log[i][len(log[i])-2]['time:timestamp'] - log[i][1]['time:timestamp']
            all_time += time.total_seconds()
    all_time /= len(log)         
    return all_time
