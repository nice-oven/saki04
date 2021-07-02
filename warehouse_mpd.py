import mdptoolbox.mdp as mdptb
import numpy as np
from itertools import product
import multiprocessing as mp
from scipy.sparse import lil_matrix, csr_matrix
import pickle
import sys

"""
What we are going to do:
1)	Function to read different datasets (potentially clean)
2)	Setup 
    a.	Transitions matrix with shape #action, #states, #states
        actions {put <color> into <x, y>, take <color> from <x, y>}
        states {2x2 fields, either empty or <color>}
        probability of success 100% 
    b.	Reward:
        general reward function: value of storing – cost of putting as function of distance (no only x or y movement, not combined)
        tremendous punishment for being full and having to store something – even though this is more the fault of the operator, right?!
        let her be of shape #s
        make sure to make it easy to change concrete values here
    c.	discount

"""

def get_dataset(name):
    datasets = {
        "test_l": './data/warehouseorder2x2.txt',
        "test_s": './data/warehouseordernew.txt',
        "train": './data/warehousetraining2x2.txt'
    }

    filename = datasets[name]

    out = []
    with open(filename, 'r') as in_file:
        for line in in_file.readlines():
            split_at = line.index("\t")
            out.append([line[:split_at], line[split_at+1:-1]])
    print("read", len(out), "lines")
    return out


def ds_to_np(ds):
    colors = ['red', 'white', 'blue']
    actions = ['store', 'restore']

    encoded = [[actions.index(el[0]), colors.index(el[1])] for el in ds]

    return np.array(encoded, dtype=int)


def generate_states(properties):
    """
    represent each state as an array with the following values:
    <1, 1><1, 2><2, 1><2, 2><n_act><n_col>
    :return:
    """

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']

    out = np.ndarray(shape=((nr_fillings**nr_fields)*(nr_actions*nr_next_col), nr_fields+2))

    field_states = list(product(np.arange(nr_fillings), repeat=nr_fields))

    for c_fs, field_state in enumerate(field_states):
        for c_a in range(nr_actions):
            for c_nc in range(nr_next_col):
                idx = nr_actions * nr_next_col * c_fs\
                      + c_a * nr_next_col \
                      + c_nc

                out[idx, :] = *field_state, c_a, c_nc

    return out


def gen_transition_matrix(states, properties):
    # for each action -> put in field x, y
    # for each state
    # for each state
    #   if put:
    #       if all same but one
    #       for each loc:
    #           if empty
    #
    #   if take:

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]

    trans = np.zeros((nr_fields, n_states, n_states))

    for a in range(nr_fields):
        print("a:", a)
        for s1 in range(n_states):
            print("\ts1:", s1)
            for s2 in range(n_states):
                st1 = states[s1]
                st2 = states[s2]

                n_act = st1[-2]
                n_col = st1[-1]

                if n_act == 0:  # put
                    if st1[a] == 3 and st2[a] == n_col:  # todo constant for empty field
                        trans[a, s1, s2] = 1
                else:  # take
                    if st1[a] == n_col and st2[a] == 3:
                        trans[a, s1, s2] = 1
            # normalize row
            row_sum = np.sum(trans[s1])
            if row_sum > 1:
                trans[s1] /= row_sum
    return trans


def pickle_tran_mat(transition_matrix, properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    filename = './data/tran_ma/' + str(nr_fields) + "_" + str(nr_fillings) + "_" + str(nr_actions) + ".pickle"
    with open(filename, 'wb') as file:
        pickle.dump(transition_matrix, file)

def unpickle_tran_mat(properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    filename = './data/tran_ma/' + str(nr_fields) + "_" + str(nr_fillings) + "_" + str(nr_actions) + ".pickle"
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_rew_mat(transition_matrix, properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    filename = './data/rew_ma/' + str(nr_fields) + "_" + str(nr_fillings) + "_" + str(nr_actions) + ".pickle"
    with open(filename, 'wb') as file:
        pickle.dump(transition_matrix, file)

def unpickle_rew_mat(properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    filename = './data/rew_ma/' + str(nr_fields) + "_" + str(nr_fillings) + "_" + str(nr_actions) + ".pickle"
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data



def gen_transition_matrix_q(states, properties, a, q):
    # for each action -> put in field x, y
    # for each state
    # for each state
    #   if put:
    #       if all same but one
    #       for each loc:
    #           if empty
    #
    #   if take:
    print("started", a)
    last_percent = 0

    a = int(a)


    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]

    put_take = 2

    trans = lil_matrix((n_states, n_states))

    n_act_arr = np.array(list(product(range(put_take), range(nr_next_col))))
    counters_fields = [nr_fillings] * nr_fields
    counters_actions = [put_take, nr_next_col]
    index_counters = np.array(counters_fields + counters_actions)
    index_counters[:-1] = index_counters[1:]
    index_counters[-1] = 1
    index_counters = np.cumprod(index_counters[::-1])[::-1]
    for i, s in enumerate(states):
        # report progress
        curr_percent = int((i * 100) / n_states)
        if curr_percent > last_percent + 4:
            print("a:", a, "at", curr_percent)
            last_percent = curr_percent

        # find adjacent sates
        # on warehouse state level
        n_col = s[-1]
        n_move = s[-2]
        s_new = None
        if n_move == 0 and s[a] == 3:
            s_new = np.concatenate((s[:a], [n_col], s[a + 1:nr_fields]))
        if n_move == 1 and s[1] == n_col:
            s_new = np.concatenate((s[:a], [3], s[a + 1: nr_fields]))
        if s_new is not None:
            s_new_r = np.tile(s_new, (n_act_arr.shape[0], 1))
            s_new_full = np.concatenate((s_new_r, n_act_arr), axis=1)
            # calculate their index
            indices = np.dot(s_new_full, index_counters)
            # set that index to 1/#states
            trans[i, indices] = 1 / indices.shape[0]
        else:
            trans[i, i] = 1
            pass
        # check
    q.put(trans)


def partitioned_gen_tran_ma(states, properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]

    # trans = np.zeros((nr_fields, n_states, n_states))  # using list of lil_matrix matrices
    trans = []

    processes = []
    queues = []
    for a in range(nr_fields):
        queues.append(mp.Queue())
        p = mp.Process(target=gen_transition_matrix_q, args=(states, properties, a, queues[-1]))
        p.start()

    for proc in processes:
        proc.join()

    for i, q in enumerate(queues):
        trans.append(q.get().tocsr())
    return trans


def partitioned_gen_rew_ma(states, properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]

    # trans = np.zeros((nr_fields, n_states, n_states))  # using list of lil_matrix matrices
    trans = []

    processes = []
    queues = []
    for a in range(nr_fields):
        queues.append(mp.Queue())
        p = mp.Process(target=gen_reward_matrix_q, args=(states, properties, a, queues[-1]))
        p.start()

    for proc in processes:
        proc.join()

    for i, q in enumerate(queues):
        trans.append(q.get().tocsr())
    return trans


def gen_reward_matrix(states, trans, properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    layout = properties['layout']

    rows = np.arange(layout[0])
    columns = np.arange(layout[1])

    distances = np.ones(layout)

    distances += columns
    distances = (distances.T + rows).T

    distances = distances.flatten()

    results = []

    # reward characteristics
    # put_bonus
    # move_malus * nr_of_moves
    for a_nr, sxs in enumerate(trans):
        states_sq = sxs.toarray()
        indices = np.where(np.all(1 > states_sq > 0))
        out = np.zeros(states_sq.shape)
        out[indices] = 10 - distances[a_nr]

        results.append(csr_matrix(out))

    return results

def gen_reward_matrix_q(states, properties, a, q):
    # for each action -> put in field x, y
    # for each state
    # for each state
    #   if put:
    #       if all same but one
    #       for each loc:
    #           if empty
    #
    #   if take:

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]

    layout = properties['layout']
    put_take = 2

    rows = np.arange(layout[0])
    columns = np.arange(layout[1])

    distances = np.ones(layout)

    distances += columns
    distances = (distances.T + rows).T

    distances = distances.flatten()

    trans = lil_matrix((n_states, n_states))
    print("started", a)
    last_percent = 0


    n_act_arr = np.array(list(product(range(put_take), range(nr_next_col))))
    counters_fields = [nr_fillings] * nr_fields
    counters_actions = [put_take, nr_next_col]
    index_counters = np.array(counters_fields + counters_actions)
    index_counters[:-1] = index_counters[1:]
    index_counters[-1] = 1
    index_counters = np.cumprod(index_counters[::-1])[::-1]
    for i, s in enumerate(states):
        # report progress
        curr_percent = int((i * 100) / n_states)
        if curr_percent > last_percent + 4:
            print("a:", a, "at", curr_percent)
            last_percent = curr_percent

        # find adjacent sates
        # on warehouse state level
        n_col = s[-1]
        n_move = s[-2]
        s_new = None
        if n_move == 0 and s[a] == 3:
            s_new = np.concatenate((s[:a], [n_col], s[a + 1:nr_fields]))
        if n_move == 1 and s[1] == n_col:
            s_new = np.concatenate((s[:a], [3], s[a + 1: nr_fields]))
        if s_new is not None:
            s_new_r = np.tile(s_new, (n_act_arr.shape[0], 1))
            s_new_full = np.concatenate((s_new_r, n_act_arr), axis=1)
            # calculate their index
            indices = np.dot(s_new_full, index_counters)
            # set that index to 1/#states
            trans[i, indices] = 10 - (.5 * distances[a])

    q.put(trans)


def eval_mdp(mdp):
    """
    - get the utility per action / field
    - visualize
    - have an experiment, where one item is very unlikely to appear say less than 5%
    - see if we get different distribution
    :return:
    """
    pass


def unpickle_mdp(properties, type_):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    # type_ = str(type_)
    # type_ = type_[type_.rfind(".")+1:type_.rfind("\\")-1]
    filename = "./data/mdp/" + type_ + str(nr_fields) + "_" + str(nr_fillings) + "_" + str(nr_actions) + ".pickle"
    with open(filename, "rb") as file:
        mdp = pickle.load(file)
    return mdp


def pickle_mdp(properties, mdp):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    type_ = str(type(mdp))
    type_ = str(type_)
    type_ = type_[type_.rfind(".")+1:type_.rfind("\\")-1]
    filename = "./data/mdp/" + type_ + str(nr_fields) + "_" + str(nr_fillings) + "_" + str(nr_actions) + ".pickle"
    with open(filename, "wb") as file:
        pickle.dump(pi, file)


def utility_one_item(properties, states, type_):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]
    layout = properties['layout']

    fields = np.zeros(layout)
    field_count = np.zeros(layout)

    mdp = unpickle_mdp(properties, type_)

    v_arr = np.array(mdp.V)
    unique_vals, f = np.unique(v_arr, return_counts=True)

    idc = np.argsort(f)

    unique_vals = unique_vals[idc]
    f = f[idc]


    print(":)")


if __name__ == "__main__":
    print(ds_to_np(get_dataset("test_l")))
    get_dataset("test_s")
    get_dataset("train")

    properties = {
        'nr_fields': 6,
        'nr_fillings': 4,
        'nr_actions': 2,
        'nr_next_col': 3,
        'layout': (2, 3)
    }

    states = generate_states(properties)

    utility_one_item(properties, states, "ValueIteration")

    print("generated", len(states), "states")

    # pickle_tran_mat(partitioned_gen_tran_ma(states, properties), properties)
    # pickle_rew_mat(partitioned_gen_rew_ma(states, properties), properties)

    data = unpickle_tran_mat(properties)
    rew = unpickle_rew_mat(properties)

    print("loaded data")

    pi = mdptb.ValueIteration(transitions=data,
                              reward=rew,
                              discount=.95)
    pi.setVerbose()

    print("running")
    results = pi.run()

    pickle_mdp(properties, pi)

    print("ok")

