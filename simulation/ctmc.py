import json
import copy

def create_prism_program_from_log(
        final_states,
        data_mean_transition_role_time,
        data_role_number_of_resources,
        data_transition_role_frequency,
        data_role_trials,
        sm_file_path,
        show_print=False):

    file = open(sm_file_path, 'w')

    file.write('ctmc\n\n')

    n = 0  # total number of states

    for key in data_mean_transition_role_time:
        n = n + 1

    file.write('const int N = ' + str(n) + ';\n')

    for key in data_role_trials:
        file.write('const int ' + str(key) + ' = ' + str(data_role_trials[key]) + ';\n')

    file.write('\n\n\nmodule ctest\n')

    n = 0
    correspondance = {}
    events = set(data_mean_transition_role_time.keys()).union(*[set(k.keys()) for k in data_mean_transition_role_time.values()])

    events = sorted(events.difference(set(['start', 'end'])))
    for key in events:
        correspondance[key] = n
        n = n + 1

    for key in correspondance:
        file.write('\t //' + key + ' : ' + str(correspondance[key]) + '\n')

    file.write('\n\n\t q : [0 ..N];')
    file.write('\n\n\t started : bool;')

    file.write('\n\n\t [] !started -> ')

    started = {}

    for key in data_transition_role_frequency["start"]:
        buffer = '('
        for role in data_transition_role_frequency["start"][key]:
            if role in data_role_number_of_resources:
                buffer = buffer + str(data_transition_role_frequency["start"][key][role]) + ' / ' + str(data_role_number_of_resources[role]) + ' * ' + role + ' + '
        buffer = buffer + '0)'

        started[key] = copy.deepcopy(buffer)

    buffer = ''
    for key in started:
        buffer = buffer + started[key] + ' + '

    buffer = buffer + '0'

    for key in started:
        if key == 'start':
            continue
        file.write(started[key] + ' / (' + buffer + ') : (started\'=true) & (q\'=' + str(correspondance[key]) + ')  + \n\t')
    file.write('0 : true;\n\n\n')

    frequences_total = {}

    frequences_local = {}

    for key in data_transition_role_frequency:
        if key == 'start':
            continue
        if key == 'end':
            continue
        frequences_local[key] = {}
        frequences_total[key] = {}
        n = 0
        for key_1 in data_transition_role_frequency[key]:
            if key_1 == 'start':
                continue
            if key_1 == 'end':
                continue
            m = 0
            for role in data_transition_role_frequency[key][key_1]:
                n = n + data_transition_role_frequency[key][key_1][role]
                m = m + data_transition_role_frequency[key][key_1][role]
            frequences_local[key][key_1] = m
        frequences_total[key] = n

    if show_print:
        print(frequences_local)
        print(frequences_total)

    for key in correspondance:
        if key in final_states:
            file.write('\t <> q = ' + str(correspondance[key]) + ' -> true;\n\n')
            continue
        file.write('\t <> q = ' + str(correspondance[key]) + ' -> ')

        for key_1 in data_transition_role_frequency[key]:
            if key_1 == 'end':
                continue
            for role in data_transition_role_frequency[key][key_1]:
                if role == 'end':
                    continue
                if role in data_mean_transition_role_time[key][key_1] and data_mean_transition_role_time[key][key_1][role] != 0:
                    file.write('( ' + str(data_mean_transition_role_time[key][key_1][role]["lambda"]) + ' * ' + str(role) + '/' + str(data_role_number_of_resources[role]) + ') + ')
                    # file.write(str(data_transition_role_frequency[key][key_1][role]) + '/' + str(frequences_local[key][key_1]) + ' * ( ' + str(
                    #     data_mean_transition_role_time[key][key_1][role]["lambda"]) + ' * ' + str(role) + '/' + str(data_role_number_of_resources[role]) + ') + ')  # 1/ kai bgazo to kleidi
                else:
                    file.write('0 +')
            file.write(' 0 : (q\' = ' + str(correspondance[key_1]) + ') + \n')
        file.write(' true ;\n\n\t')

    file.write('\n\nendmodule\n\n')

    for key in correspondance:
        if key in final_states:
            file.write(f'label "q_terminal_{key}" = (q =' + str(correspondance[key]) + ');\n\n')

    file.write('rewards "num"\n\t')

    for key in correspondance:
        if key in final_states:
            file.write('[] q = ' + str(correspondance[key]) + ' : 0;\n\t')
        else:
            file.write('[] q = ' + str(correspondance[key]) + ' : 1;\n\t')
    file.write('\nendrewards')

    ###############################################################
    # This is when you do the what if analysis, once you have found
    # your values you continue to go to the computation of the probabilities
    # I think you should put the next part in another cell in jupyter
    # Remeber the role_trials.json is the number of workers the user
    # initialises for the what if analysis
    ##############################################################

    final_probabilities = {}

    for key in data_transition_role_frequency:
        if key == 'start':
            continue
        if key == 'end':
            continue
        final_probabilities[key] = {}
        for key_1 in data_transition_role_frequency[key]:
            if key_1 == 'start':
                continue
            if key_1 == 'end':
                continue
            final_probabilities[key][key_1] = 0
            for role in data_transition_role_frequency[key][key_1]:
                if role == 'start':
                    continue
                if role == 'end':
                    continue
                final_probabilities[key][key_1] = final_probabilities[key][key_1] + (data_transition_role_frequency[key][key_1][role] / frequences_total[key]) * (
                            data_role_trials[role] / data_role_number_of_resources[role])

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
