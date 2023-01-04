import numpy as np

def generate_random_selection_1d(array_size, selection_size):
    selection = np.zeros(array_size, dtype='bool')
    selection[:selection_size] = True
    np.random.shuffle(selection)
    return selection

def generate_double_complementary_random_selection_1d(array_size, selection_sizes):
    assert len(selection_sizes) == 2
    selection_1 = generate_random_selection_1d(array_size, selection_sizes[0])
    temp_selec = generate_random_selection_1d(array_size - selection_sizes[0], selection_sizes[1])
    selection_2 = np.zeros(array_size, dtype='bool')
    selection_2[np.logical_not(selection_1)] = temp_selec

    return selection_1, selection_2


def generate_simple_random_selection_tracking(pids, selection_size, N_garbage_identities):
    np.random.seed(42)

    # assert that the number of garbage identities asked in the selection is not too high
    assert N_garbage_identities <= 2798

    N_normal_identities = selection_size - N_garbage_identities



    indexes = np.arange(len(pids))

    selection_indexes = np.zeros(0,dtype='int')

    unique_pids, pids_count = get_unique_pids_count(pids)
    pids_to_be_selected = unique_pids[1:]
    np.random.shuffle(pids_to_be_selected)

    counter = 0
    while len(selection_indexes) < N_normal_identities:
        current_pid_to_be_selected = pids_to_be_selected[counter]
        counter += 1
        to_be_added_to_selection_indexes = indexes[pids == current_pid_to_be_selected]
        selection_indexes = np.append(selection_indexes, to_be_added_to_selection_indexes)
        print(len(to_be_added_to_selection_indexes))
    #print(selection_indexes, len(selection_indexes), pids_count)

    N_garbage_identities = selection_size - len(selection_indexes)

    if N_garbage_identities <= 0:
        N_garbage_identities = 0

    garbages_identities_indexes = indexes[pids == 0]
    np.random.shuffle(garbages_identities_indexes)
    selection_indexes = np.append(selection_indexes, garbages_identities_indexes[:N_garbage_identities])

    return selection_indexes









# For the composition of the pure appearance graph pair selection, we need
# to know how many images of a person there are for each pids
def get_unique_pids_count(pids):
    unique_pids = np.unique(pids)
    pids_count = np.zeros(len(unique_pids))
    for i in range(len(unique_pids)):
        pids_count[i] = np.count_nonzero(pids == unique_pids[i])
    return unique_pids, np.array(pids_count,dtype='int')