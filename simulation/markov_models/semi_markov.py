"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""
from typing import Set

import numpy as np
import scipy

from simulation.markov_models.mult_gauss import MultiGauss, mg_from_json
from simulation.markov_models.gauss import Gauss
from simulation.markov_models.convolution import mult_gauss_convolution, mult_gauss_sum

self_loop_threshold = 0.1

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

class SemiMarkovState:

    def __init__(self, state, resources=None):
        self.state = state
        self.resources = resources

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, __value):
        return self.state == __value.state

    def __str__(self):
        return self.state

    def to_json(self):
        return {
            'state': self.state,
            'resources': self.resources['resources'],
            'p': self.resources['p']
        }

class SemiMarkovTransition:
    def __init__(self, e_from, e_to, prob=None, multi_gaussian_time=None):
        self.e_from = e_from
        self.e_to = e_to
        self.prob = prob
        self.multi_gaussian_time = multi_gaussian_time

    def __hash__(self):
        return hash((self.e_from, self.e_to))

    def __eq__(self, __value):
        return (self.e_from == __value.e_from and self.e_to == __value.e_to)

    def __str__(self):
        return f"{self.e_from} --> {self.e_to}"

    def to_json(self):
        if isinstance(self.multi_gaussian_time,MultiGauss):
            mgt = self.multi_gaussian_time.to_json()
        else:
            mgt = self.multi_gaussian_time
        return {'from': self.e_from,
                'to': self.e_to,
                'prob': self.prob,
                'multi_gaussian_time': mgt}

class SemiMarkov:
    def __init__(self, states: Set[SemiMarkovState], transitions: Set[SemiMarkovTransition]):
        self.states = states
        # Each transition is a tuple: (from, to, probability, multi Gauss time, multi Gauss resource)
        self.transitions = transitions 

    def start_state(self):
        for state in self.states:
            if state.state == 'start':
                return state
    
    def end_state(self):
        for state in self.states:
            if state.state == 'end':
                return state

    def draw_transition(self, state):
        transitions = self.get_out_transitions_with_loop(state)
        prob = []
        trans = []
        for transition in transitions:
            prob.append(transition.prob)
            trans.append(transition)
        index = np.random.choice(len(trans), 1, p=prob)
        transition = trans[index[0]]
        return transition

    def draw_time(self, transition, resource=None):
        # TODO make the time resource specific
        multi_gauss = transition.multi_gaussian_time
        time = -1
        tries = 0
        while time <= 0:
            if isinstance(multi_gauss, MultiGauss):
                gauss = np.random.choice(multi_gauss.gaussians, 1, p=multi_gauss.probabilities)
                time = np.random.normal(loc=gauss[0].mean, scale=gauss[0].deviation)
            elif 'best_dist' in multi_gauss:
                best_dist = multi_gauss['best_dist']
                fitted_params = multi_gauss['fitted_params']
                func = lambda : getattr(scipy.stats, best_dist).rvs(**fitted_params, size=1)
                time = func()[0]
            else:
                print(multi_gauss)
            tries += 1
            if tries > 100:
                time = 1
        return time

    def draw_resource(self, state):
        multinom = state.resources
        index = np.random.choice(len(multinom['resources']), 1, p=multinom['p'])
        resource = multinom['resources'][index[0]]
        return resource

    def simulate(self,number_of_traces=1):
        times = set()
        iterations = number_of_traces
        sim_log_dict_list = []
        for i in range(iterations):
            time = 0
            transition = self.draw_transition(self.start_state())
            state = transition.e_to
            end = self.end_state()
            while state != end:
                resource = self.draw_resource(state)
                sim_log_dict_list.append({'case:concept:name': f'simid{i}',
                                          'concept:name':state.state,
                                          'time:since:start':time,
                                          'org:resource':resource})
                transition = self.draw_transition(state)#,previous_resource)

                time += self.draw_time(transition, resource) #+ resource)
                state = transition.e_to
            times.add(time)
        return times, sim_log_dict_list

    def reduce_node(self, state):
        if ((state == 'start') or (state == 'end')):
            return
        else:
            # Calculate self-loop time
            self_loop_time = MultiGauss([1], [Gauss(0,0)])
            #for transition in self_loops:
            self_loop_time = self.calculate_self_loop_time(state)
            #  Add new transitions
            in_transitions = self.get_in_transitions(state)
            out_transitions = self.get_out_transitions(state)
            i = 1
            for in_transition in in_transitions:
                in_state = in_transition[0]
                for out_transition in out_transitions:
                    out_state = out_transition[1]
                    p = self.get_probability(in_state, out_state)
                    time = self.get_time(in_state, out_state)
                    new_p = self.get_probability(in_state,state)*self.get_probability(state,out_state)/(1-self.get_probability(state,state))
                    all_p = p + new_p
                    m1 = self.get_time(in_state, state)
                    m2 = self.get_time(state, out_state)
                    new_time = mult_gauss_convolution(m1, self_loop_time)
                    new_time = mult_gauss_convolution(new_time, m2)
                    all_time = mult_gauss_sum(time, new_time, p/all_p, new_p/all_p)

                    # Remove old transition
                    transition_to_remove = set()
                    for transition in self.transitions:
                        if ((transition[0] == in_state) and (transition[1] == out_state)):
                            transition_to_remove.add(transition)
                    for transition in transition_to_remove:
                        self.transitions.remove(transition)

                    # Add new transition
                    self.transitions.add((in_state, out_state, all_p, all_time))


            # Remove state
            transition_to_remove = set()
            for transition in self.transitions:
                if ((transition[0] == state) or (transition[1] == state)):
                    transition_to_remove.add(transition)
            for transition in transition_to_remove:
                 self.transitions.remove(transition)
            self.states.remove(state)

    def  calculate_self_loop_time(self, state):
        m1 = self.get_time(state, state)
        p = self.get_probability(state, state)
        m = MultiGauss([1-p],[Gauss(0,0)])
        p_current = p * (1-p)
        conv = MultiGauss([1],[Gauss(0,0)])
        while (p_current > self_loop_threshold):
            conv = mult_gauss_convolution(m1, conv)
            m = mult_gauss_sum(m, conv, 1, p_current)
            p_current *= p
        m.normalise_gauss()
        return m
    
    def get_in_transitions(self, state):
        in_transitions = set()
        for transition in self.transitions:
            if (transition.e_to.state == state.state) and (transition.e_from.state != state.state):
                in_transitions.add(transition)
        return in_transitions
    
    def get_in_transitions_with_loop(self, state):
        in_transitions = set()
        for transition in self.transitions:
            if (transition.e_to.state == state.state):
                in_transitions.add(transition)
        return in_transitions
    
    def get_out_transitions(self, state):
        out_transitions = set()
        for transition in self.transitions:
            if (transition.e_from.state == state.state) and (transition.e_to.state != state.state):
                out_transitions.add(transition)
        return out_transitions
    
    def get_out_transitions_with_loop(self, state):
        out_transitions = set()
        for transition in self.transitions:
            if (transition.e_from.state == state.state):
                out_transitions.add(transition)
        return out_transitions
 
    def get_probability(self, state1, state2):
        for transition in self.transitions:
            if ((transition.e_from.state == state1.state) and (transition.e_to.state == state2.state)):
                return transition.prob
        return 0
    
    def get_time(self, state1, state2):
        for transition in self.transitions:
            if ((transition.e_from.state == state1.state) and (transition.e_to.state == state2.state)):
                return transition.multi_gaussian_time
        return MultiGauss([],[])
    
    def select_next_state(self):
        min_degree = 100000000
        for state in self.states:
            degree = 0
            degree = len(self.get_in_transitions_with_loop(state))*len(self.get_out_transitions_with_loop(state))
            if (degree < min_degree) and (state.state != 'start') and (state.state != 'end'):
                next_state = state
                min_degree = degree
        return next_state

    def state_degrees(self):
        state_degrees = []
        for state in self.states:
            if ((state.state != 'start') and (state.state != 'end')):
                in_transitions = len(self.get_in_transitions(state))
                out_transitions = len(self.get_out_transitions(state))
                state_degrees.append(max(in_transitions, out_transitions))
        return state_degrees

    def to_json(self):
        json_transition_list = []
        for smt in self.transitions:
            json_transition_list.append(smt.to_json())
        json_state_list = []
        for ste in self.states:
            json_state_list.append(ste.to_json())
        return {'states': json_state_list, 'transitions': json_transition_list}

def semi_markov_from_json(json_dict):
    transition_set = set()
    for transition in json_dict['transitions']:
        if 'gaussians' in transition['multi_gaussian_time']:
            multi_gaussian = mg_from_json(transition['multi_gaussian_time'])
        else:
            multi_gaussian = transition['multi_gaussian_time']
        smt = SemiMarkovTransition(transition['from'], transition['to'],transition['prob'],multi_gaussian)
        transition_set.add(smt)
    states_set = set()
    for state in json_dict['states']:
        ste = SemiMarkovState(state['state'],{'resources':state['resources'],'p':state['p']})
        states_set.add(ste)
    return SemiMarkov(states_set, transition_set)