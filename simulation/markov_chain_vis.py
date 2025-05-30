import pm4py.objects.dfg.obj
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.objects.transition_system import obj as ts
from pm4py.objects.transition_system import utils
from pm4py.visualization.transition_system import visualizer

from simulation.markov_models.semi_markov import SemiMarkovState, SemiMarkovTransition
from simulation.markov_models.mult_gauss import mg_from_json

def view_markov_chain(sim_model,percentage=True):
    map_states = {}
    trans_sys = ts.TransitionSystem()
    for s in sim_model['states']:
        # if str(s['state']) in ['start','end']:
        #     continue

        s = SemiMarkovState(s['state'],{'resources':s['resources'],'p':s['p']})
        map_states[s.state] = TransitionSystem.State(s.state)
        trans_sys.states.add(map_states[s.state])
    for i,t in enumerate(sim_model['transitions']):
        # if str(t['from']) in ['start','end'] or str(t['to']) in ['start','end']:
        #     continue

        if 'gaussians' in t['multi_gaussian_time']:
            mg = mg_from_json(t['multi_gaussian_time'])
        else:
            mg = t['multi_gaussian_time']
        t = SemiMarkovTransition(t['from'], t['to'],t['prob'],mg)
        if percentage:
            utils.add_arc_from_to(f"{t.prob:.0%}",map_states[t.e_from.state],map_states[t.e_to.state],trans_sys,data=t.multi_gaussian_time)
        else:
            utils.add_arc_from_to(f"{t.prob:.0%}",map_states[t.e_from.state],map_states[t.e_to.state],trans_sys,data=t.multi_gaussian_time)
    gviz = visualizer.apply(trans_sys, parameters={visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    visualizer.view(gviz)
    visualizer.save(gviz,'plots/markov-chain.png')

def view_resource_dfg(data_transition_role_frequency,mean_times=None,lambdas=None,start_states=[],end_states=[],percentage=False):
    map_states = {}
    trans_sys = ts.TransitionSystem()
    for k, v in data_transition_role_frequency.items():
        if k not in map_states:
            map_states[k] = TransitionSystem.State(k)
            trans_sys.states.add(map_states[k])

        for k2,v2 in v.items():
            if k2 not in map_states:
                map_states[k2] = TransitionSystem.State(k2)
                trans_sys.states.add(map_states[k2])

            for k3,v3 in v2.items():
                if percentage:
                    utils.add_arc_from_to(f"{k3}:{v3:.0%}",map_states[k],map_states[k2],trans_sys)
                else:
                    name = f"{k3}:{int(v3)}"
                    if mean_times:
                        name+=f"\n{mean_times[(k,k2,k3)]:.2}h"
                    utils.add_arc_from_to(name,map_states[k],map_states[k2],trans_sys)

    start = TransitionSystem.State("<&#9655;>")
    trans_sys.states.add(start)
    end = TransitionSystem.State("<&#9633;>")
    trans_sys.states.add(end)
    for ss in start_states:
        utils.add_arc_from_to('Manager:1000',start,map_states[ss],trans_sys)
    for es in end_states:
        utils.add_arc_from_to('1000',map_states[es],end,trans_sys)

    gviz = visualizer.apply(trans_sys, parameters={visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    visualizer.view(gviz)
    visualizer.save(gviz,'plots/resource-dfg.png')

def view_non_resource_dfg(data_transition_role_frequency,start_states=[],end_states=[],percentage=True):
    map_states = {}
    trans_sys = ts.TransitionSystem()
    for k, v in data_transition_role_frequency.items():
        if k not in map_states:
            map_states[k] = TransitionSystem.State(k)
            trans_sys.states.add(map_states[k])

        for k2,v2 in v.items():
            if k2 not in map_states:
                map_states[k2] = TransitionSystem.State(k2)
                trans_sys.states.add(map_states[k2])
            if percentage:
                utils.add_arc_from_to(f"{v2:.0%}",map_states[k],map_states[k2],trans_sys)
            else:
                utils.add_arc_from_to(f"{int(v2)}",map_states[k],map_states[k2],trans_sys)

    start = TransitionSystem.State("<&#9655;>")
    trans_sys.states.add(start)
    end = TransitionSystem.State("<&#9633;>")
    trans_sys.states.add(end)
    for ss in start_states:
        utils.add_arc_from_to('1000',start,map_states[ss],trans_sys)
    for es in end_states:
        utils.add_arc_from_to('1000',map_states[es],end,trans_sys)

    gviz = visualizer.apply(trans_sys, parameters={visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    visualizer.view(gviz)
    visualizer.save(gviz,'plots/dfg.png')

def view_resource_markov_chain(data_transition_role_frequency,percentage=True):
    map_states = {}
    trans_sys = ts.TransitionSystem()
    for k, v in data_transition_role_frequency.items():
        if k not in map_states:
            map_states[k] = TransitionSystem.State(k)
            trans_sys.states.add(map_states[k])

        for k2,v2 in v.items():
            if k2 not in map_states:
                map_states[k2] = TransitionSystem.State(k2)
                trans_sys.states.add(map_states[k2])

            for k3,v3 in v2.items():
                if percentage:
                    utils.add_arc_from_to(f"{k3}:{v3:.0%}",map_states[k],map_states[k2],trans_sys)
                else:
                    utils.add_arc_from_to(f"{k3}:{int(v3)}",map_states[k],map_states[k2],trans_sys)

    gviz = visualizer.apply(trans_sys, parameters={visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    visualizer.view(gviz)
    visualizer.save(gviz,'plots/resource-markov-chain.png')

def view_non_resource_markov_chain(data_transition_role_frequency,percentage=True):
    map_states = {}
    trans_sys = ts.TransitionSystem()
    for k, v in data_transition_role_frequency.items():
        if k not in map_states:
            map_states[k] = TransitionSystem.State(k)
            trans_sys.states.add(map_states[k])

        for k2,v2 in v.items():
            if k2 not in map_states:
                map_states[k2] = TransitionSystem.State(k2)
                trans_sys.states.add(map_states[k2])
            if percentage:
                utils.add_arc_from_to(f"{v2:.0%}",map_states[k],map_states[k2],trans_sys)
            else:
                utils.add_arc_from_to(f"{int(v2)}",map_states[k],map_states[k2],trans_sys)
    gviz = visualizer.apply(trans_sys, parameters={visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    visualizer.view(gviz)
    visualizer.save(gviz,'plots/markov-chain.png')