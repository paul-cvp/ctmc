from scipy.linalg import eig
import datetime

def build_matrix(dfg, log_activities, times):
    a = []
    i = 0
    for activity1 in log_activities:
        a.append([])
        sum = 0
        j = 0
        for activity2 in log_activities:
            sum += dfg[activity1,activity2]
        for activity2 in log_activities:
            if not activity1+";"+activity2 in times:
                times[activity1+";"+activity2] = datetime.timedelta(days=0, seconds=0)
            a[i].append([])
            a[i][j].append((dfg[activity1,activity2]/sum, times[activity1+";"+activity2]))
            if activity1=='start' and activity2=='end':
                start = i
                end =j
            j += 1
        i += 1
    return a, start, end

def multiply_elements(e1, e2):
    mult = []
    for (p1, times1) in e1:
        for(p2, times2) in e2:
            if p1 > 0 and p2 > 0:
                new_times = []
                for i in range(0, len(times1)):
                    for j in range(0, len(times2)):
                        new_times.append(times1 + times2)
                        mult.append((p1*p2, new_times))
    return mult

def multiply (a1, a2, end):
    a3 = []
    for i in range(0, len(a1)):
        a3.append([])
        for j in range(0, len(a1)):
            for k in range(0, len(a1)):
                    if k != end:
                        a3[i].append(multiply_elements(a1[i][k], a2[k][j]))
                    else:
                        a3[i].append((0, []))
    return a3

def calculate_all_times(dfg, log_activities, times):
    all_times = []
    a, start, end = build_matrix(dfg, log_activities, times)
    a2 = multiply (a, a, end)

    for (probability, times) in a2[start][end]:
        if probability > 0:
            all_times.append((probability,times))
    return all_times   

def calculate_limiting_probabilities(dfg, log_activities):
    a = []
    i = 0
    for activity1 in log_activities:
        a.append([])
        sum = 0
        j = 0
        for activity2 in log_activities:
            sum += dfg[activity1,activity2]
        for activity2 in log_activities:
            a[i].append(dfg[activity1,activity2]/sum)
            j += 1
        i += 1
    w, vl = eig(a, left=True, right=False)
    j = 0
    for eigv in w:
        if round(eigv.real,4) == 1.0:
            break
        j += 1     
    iegv = vl[:,j].real
    if(iegv[0] < 0):
        iegv = iegv * (-1)
    limiting_probabilities = {}
    for i in range (0,len(log_activities)):
        limiting_probabilities[log_activities[i]] = iegv[i]
    return limiting_probabilities
