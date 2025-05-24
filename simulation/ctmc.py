import json
import copy

def create_prism_program_from_log(
        correspondence,
        final_states,
        data_mean_transition_role_time,
        data_role_number_of_resources,
        data_transition_role_frequency,
        data_role_trials,
        sm_file_path,
        show_print=False):
    '''
    :param correspondence: All the keys and unique ids numbered from 0 to len(correspondence)
    :param final_states: everything labelled as q_terminal
    :param data_mean_transition_role_time:
    :param data_role_number_of_resources:
    :param data_transition_role_frequency:
    :param data_role_trials:
    :param sm_file_path:
    :param show_print:
    :return:

    None of the final states can be labelled as 0
    '''

    file = open(sm_file_path, 'w')

    file.write('ma\n\n')

    n = len(correspondence)-1  # total number of states

    file.write('const int N = ' + str(n) + ';\n')

    for key in data_role_trials:
        file.write('const double ' + str(key) + ' = ' + str(data_role_trials[key]) + ';\n')

    file.write('\n\n\nmodule ctest\n')

    for key in correspondence:
        file.write('\t //' + key + ' : ' + str(correspondence[key]) + '\n')
    # constants are set
    file.write('\n\n\t q : [0 ..N];') # list of states by numbers based on the correspondence
    file.write('\n\n\t started : bool;') # variable that indicates if we have been initialized
    # this only runs once
    file.write('\n\n\t [] !started -> ')

    started = {}
    # this defines the start probabilities
    for key in data_transition_role_frequency["start"]:
        buffer = '('
        for role in data_transition_role_frequency["start"][key]:
            if role in data_role_number_of_resources:
                # for each role and frequency we apply the rule of 3 based on the desired role trials (regula de trei simpla)
                # each rule of 3 is summed
                buffer = buffer + str(data_transition_role_frequency["start"][key][role]) + ' / ' + str(data_role_number_of_resources[role]) + ' * ' + role + '+'
        buffer = buffer[:-1] + ')'

        started[key] = copy.deepcopy(buffer)

    # this creates the normalization so that we have probabilities in started
    buffer = ''
    for key in started:
        buffer = buffer + started[key] + '+'
    buffer = buffer[:-1]

    buffer1 = ''
    for key in started:
        if key == 'start':
            continue
        # start probabilities divided by the normalization
        buffer1 += started[key] + ' / (' + buffer + ') : (started\'=true) & (q\'=' + str(correspondence[key]) + ')+\n\t'
    buffer1 = buffer1[:-3] + ';\n\n'
    file.write(buffer1)
    # now we are finished with started

    frequences_total = {}
    # frequences_local = {}
    # here we make transition probabilities (we need it for the rule of three)
    for key in data_transition_role_frequency:
        if key in ['start','end']:
            continue
        # frequences_local[key] = {}
        frequences_total[key] = {}
        n = 0
        for key_1 in data_transition_role_frequency[key]:
            if key_1 in ['start','end']:
                continue
            # m = 0
            for role in data_transition_role_frequency[key][key_1]:
                n = n + data_transition_role_frequency[key][key_1][role]
                # m = m + data_transition_role_frequency[key][key_1][role]
            # this tells us that we moved from state key to key_1 m many times
            # frequences_local[key][key_1] = m
        # this tells us that we moved from state key to anywhere n many times
        frequences_total[key] = n

    if show_print:
        # print(frequences_local)
        print(frequences_total)

    for key in correspondence:
        if key in final_states:
            file.write('\t <> started & q = ' + str(correspondence[key]) + ' -> true;\n\n')
            continue
        # if q is this specific state -> -(then)->
        if key in data_transition_role_frequency:
            file.write('\t <> started & q = ' + str(correspondence[key]) + ' -> ')
            for key_1 in data_transition_role_frequency[key]:
                if key_1 == 'end':
                    continue
                for role in data_transition_role_frequency[key][key_1]:
                    if role == 'end':
                        continue
                    if (key in data_mean_transition_role_time and
                        key_1 in data_mean_transition_role_time[key] and
                        role in data_mean_transition_role_time[key][key_1] and
                        data_mean_transition_role_time[key][key_1][role]["lambda"] != 0):
                        # again the rule of three
                        #TODO: rename lambda to rate
                        file.write('( ' + str(data_mean_transition_role_time[key][key_1][role]["lambda"]) + ' * ' + str(role) + '/' + str(data_role_number_of_resources[role]) + ') + ')
                    else:
                        file.write('0 +')
                file.write(' 0 : (q\' = ' + str(correspondence[key_1]) + ') + \n')
            file.write(' 0:true ;\n\n\t')

    file.write('\n\nendmodule\n\n')
    # now we have set the transition probabilities

    # here we have the terminal states
    for key in correspondence:
        if key in final_states:
            file.write(f'label "q_terminal_{key}" = (q =' + str(correspondence[key]) + ');\n\n')

    file.write('rewards "num"\n\t')
    # here we have the rewards
    for key in correspondence:
        if key in final_states:
            file.write('[] q = ' + str(correspondence[key]) + ' : 0;\n\t')
        else:
            file.write('[] q = ' + str(correspondence[key]) + ' : 1;\n\t')
    file.write('\nendrewards')

    ###############################################################
    # This is when you do the what if analysis, once you have found
    # your values you continue to go to the computation of the probabilities
    # I think you should put the next part in another cell in jupyter
    # Remeber the role_trials.json is the number of workers the user
    # initialises for the what if analysis
    ##############################################################

    final_probabilities = {}
    # to compute the final probabilities you use the seen frequency / (weighted by all the times we leave a specific state) * current role trials / number of roles
    for key in data_transition_role_frequency:
        if key in ['start', 'end']:
            continue
        final_probabilities[key] = {}
        for key_1 in data_transition_role_frequency[key]:
            if key_1 in ['start', 'end']:
                continue
            final_probabilities[key][key_1] = 0
            for role in data_transition_role_frequency[key][key_1]:
                if role in ['start', 'end']:
                    continue
                # again the rule of 3
                final_probabilities[key][key_1] = final_probabilities[key][key_1] + (data_transition_role_frequency[key][key_1][role] / frequences_total[key]) * (
                            data_role_trials[role] / data_role_number_of_resources[role])
    # normalization to make it a probability
    for key in final_probabilities:
        total = 0
        for key_1 in final_probabilities[key]:
            total = total + final_probabilities[key][key_1]
        for key_1 in final_probabilities[key]:
            final_probabilities[key][key_1] = final_probabilities[key][key_1] / total

    if show_print:
        print(final_probabilities)

    return final_probabilities

if __name__ == '__main__':
    # Open and read the JSON file
    with open('mean_transition_role_time.json', 'r') as file:
        data_mean_transition_role_time = json.load(file)

    with open('role_number_of_resources.json', 'r') as file:
        data_role_number_of_resources = json.load(file)

    with open('transition_role_frequency.json', 'r') as file:
        data_transition_role_frequency = json.load(file)

    with open('role_trials.json', 'r') as file:
        data_role_trials = json.load(file)
    final_probabilities = create_prism_program_from_log(
        ['Completed'],
        data_mean_transition_role_time,
        data_role_number_of_resources,
        data_transition_role_frequency,
        data_role_trials,
        'ctmc.sm')
    #TODO: these are the final probabilities we can put in the display of the ctmc
    with open('result.json', 'w') as file:
        json.dump(final_probabilities, file)
