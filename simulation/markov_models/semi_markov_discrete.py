"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

from simulation.markov_models.mult_gauss import MultiGauss
from simulation.markov_models.gauss import Gauss
from simulation.markov_models.convolution import mult_gauss_convolution, mult_gauss_sum, threshold
import numpy as np
import time



def mean_time_between_events(e1,e2,skip_events,log):
        times = set()
        for trace in log:
            for i in range(len(trace) - 1):
                if trace[i]['concept:name'] == e1:
                    for k in range (i+1, len(trace)):
                        if trace[k]['concept:name'] == e2:
                            time = trace[k]['time:timestamp'] - trace[i]['time:timestamp']
                            times.add(time.total_seconds()//3600)
                            break
                        if not (trace[k]['concept:name'] in skip_events):
                            break
        if len(times) > 0:
            mean = sum(times)/len(times)
        else: 
            mean = 0
        return mean

class SemiMarkovDiscrete:
    def __init__(self, states, transitions, tranisition_times):
        self.states = states 
        # Each transition is a tuple: (from, to, probability, multi Gauss)
        self.transitions = transitions
        self.transition_times = tranisition_times 

    def start_state(self):
        for state in self.states:
            if state == 'start':
                return state
    
    def end_state(self):
        for state in self.states:
            if state == 'end':
                return state

    def draw_transition(self, state):
        transitions = self.get_out_transitions_with_loop(state)
        prob = []
        trans = []
        for transition in transitions:
            prob.append(transition[2])
            trans.append(transition)
        #print(prob)
        index = np.random.choice(len(trans), 1, p=prob)
        transition = trans[index[0]]
        return transition

    def draw_time(self, transition, times_dictionary):
        time = np.random.choice(times_dictionary[transition[0]+"->"+transition[1]])
        return time


    def simulate(self, times_dictionary):
        times = list()
        #iterations = 100
        #for i in range(iterations):
        t_end = time.time() + 60 * 1
        while time.time() < t_end:
            overall_time = 0
            state = self.start_state()
            start = self.start_state() 
            end = self.end_state() 
            while state != end:
                transition = self.draw_transition(state)
                if state != start and transition[1] != end:
                    overall_time += self.draw_time(transition, times_dictionary)
                state = transition[1]
            #if overall_time < 1200:    
            times.append(overall_time)
        return times

    def reduce_node(self, state):
        if ((state == 'start') or (state == 'end')):
            return
        else:
            # Calsulate self-loop time
            init_self_loop_time = [1.0]
            #for transition in self_loops:
            self_loop_time = self.calculate_self_loop_time(state, 0.1)
            if len(self_loop_time) == 1:
                self_loop_time = init_self_loop_time
            #print("SELF_LOOP")
            #print(self_loop_time)
            #plt.plot(self_loop_time, label="Self-loop Time", color ="black")
            #  Add new transitions
            in_transitions = self.get_in_transitions(state)
            out_transitions = self.get_out_transitions(state)
            for in_transition in in_transitions:
                in_state = in_transition[0]
                for out_transition in out_transitions:
                    out_state = out_transition[1]
                    p = self.get_probability(in_state, out_state)
                    time = self.get_time(in_state, out_state)
                    #plt.plot(time, label="Old Time", color ="blue")
                    new_p = self.get_probability(in_state,state)*self.get_probability(state,out_state)/(1-self.get_probability(state,state))
                    all_p = p + new_p
                    m1 = self.get_time(in_state, state)
                    m2 = self.get_time(state, out_state)
                    #print("Sizes:")
                    #print(len(m1))
                    #printf(len(self_loop_time))
                    #print(len(m2))
                    new_time = np.convolve(m1, self_loop_time, "full")
                    new_time = np.convolve(new_time, m2, "full")
                    all_time = []
                    i = 0
                    while (i < len(time)) and (i < len(new_time)):
                        all_time.append(p/all_p*time[i] + new_p/all_p*new_time[i])
                        i+=1
                    while (i < len(new_time)):
                        all_time.append(new_p/all_p*new_time[i])
                        i+=1
                    while (i < len(time)):
                        all_time.append(p/all_p*time[i])
                        i+=1

                    all_time = all_time / np.sum(all_time)
                    #plt.plot(m1, label="m1 time", color="green")
                    #plt.plot(m2, label="m2 time", color="orange")
                    #plt.plot(all_time, label="New time", color="red")
                    #plt.title(str(in_state)+"->"+str(out_state))
                    #plt.show()

                    # Remove old transition
                    transition_to_remove = set()
                    for transition in self.transitions:
                        if ((transition[0] == in_state) and (transition[1] == out_state)):
                            transition_to_remove.add(transition)
                    for transition in transition_to_remove:
                        self.transitions.remove(transition)
                        del self.transition_times[(transition[0],transition[1])]
                            
                    # Add new transition
                    #print("ADD:")
                    #print(in_state)
                    #print(out_state)
                    t = (in_state, out_state, all_p)
                    self.transitions.add(t)
                    self.transition_times[(in_state, out_state)] = all_time


            # Remove state
            transition_to_remove = set()
            for transition in self.transitions:
                if ((transition[0] == state) or (transition[1] == state)):
                    transition_to_remove.add(transition)
            for transition in transition_to_remove:
                 self.transitions.remove(transition)
                 del self.transition_times[(transition[0],transition[1])]
            self.states.remove(state)


    def  calculate_self_loop_time_old(self, state, threshold):
        m1 = self.get_time(state, state)
        p = self.get_probability(state, state)
        m = MultiGauss([1-p],[Gauss(0,0)])
        p_current = p * (1-p)
        conv = MultiGauss([1],[Gauss(0,0)])
        while (p_current > threshold):
            conv = mult_gauss_convolution(m1, conv)
            m = mult_gauss_sum(m, conv, 1, p_current)
            p_current *= p
        m.normalise_gauss()
        return m
    
    def calculate_self_loop_time(self, state, threshold):
        m1 = self.get_time(state, state)
        #plt.plot(m1, label="m1 time", color="black")
        #plt.show()
        p = self.get_probability(state, state)
        p_current = p * (1-p)
        conv = [1.0]
        m = [1-p]
        #print("SELF CONVOLUTION")
        while (p_current > threshold):
            #print(p_current)
            #print(threshold)
            conv = np.convolve(conv, m1, "full")
            #plt.plot(conv, label="conv", color="yellow")
            #plt.show()
            res =[]
            i = 0
            while (i < len(conv)) and (i < len(m)):
                res.append(p_current*conv[i] + m[i])
                i+=1
            while (i < len(conv)):
                res.append(p_current*conv[i])
                i+=1
            while (i < len(m)):
                res.append(m[i])
                i+=1
            m = res
            #plt.plot(m, label="m time", color="green")
            #plt.show()
            p_current *= p
        m = m /np.sum(m)
        return m
    
    def get_in_transitions(self, state):
        in_transitions = set()
        for transition in self.transitions:
            if (transition[1] == state) and (transition[0] != state):
                in_transitions.add(transition)
        return in_transitions
    
    def get_in_transitions_with_loop(self, state):
        in_transitions = set()
        for transition in self.transitions:
            if (transition[1] == state):
                in_transitions.add(transition)
        return in_transitions
    
    def get_out_transitions(self, state):
        out_transitions = set()
        for transition in self.transitions:
            if (transition[0] == state) and (transition[1] != state):
                out_transitions.add(transition)
        return out_transitions
    
    def get_out_transitions_with_loop(self, state):
        out_transitions = set()
        for transition in self.transitions:
            if (transition[0] == state):
                out_transitions.add(transition)
        return out_transitions
 
    def get_probability(self, state1, state2):
        for transition in self.transitions:
            if ((transition[0] == state1) and (transition[1] == state2)):
                return transition[2]
        return 0
    
    def get_time_old(self, state1, state2):
        for transition in self.transitions:
            if ((transition[0] == state1) and (transition[1] == state2)):
                return transition[4]
        return MultiGauss([],[])
    
    def get_time(self, state1, state2):
        for transition in self.transitions:
            if ((transition[0] == state1) and (transition[1] == state2)):
                return self.transition_times[(state1, state2)]
        return [1.0]
    
    def select_next_state(self):
        min_degree = 100000000
        for state in self.states:
            degree = 0
            degree = len(self.get_in_transitions_with_loop(state))*len(self.get_out_transitions_with_loop(state))
            if (degree < min_degree) and (state != 'start') and (state != 'end'):
                next_state = state
                min_degree = degree
        return next_state



    