import mdptoolbox.mdp as mdptb
import numpy as np
from itertools import product
import multiprocessing as mp
from scipy.sparse import lil_matrix, csr_matrix
import pickle
import sys
import matplotlib.pyplot as plt
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

"""
# IO
"""


def get_dataset(name):
    datasets = {
        "test_l": './data/warehouseorder2x2.txt',
        "test_s": './data/warehouseordernew.txt',
        "train": './data/warehousetraining2x2.txt',
        "cwo": './data/curr_warehouseorder.txt',
        "cwt": './data/curr_warehousetraining.txt'
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


def filename_from_properties(properties):
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    layout = properties['layout']
    color_frequencies_str = "_".join( "{:5f}".format(cf) for cf in properties['color_frequencies'])
    return "_".join((str(nr_fields),
                     str(nr_fillings),
                     str(nr_actions),
                     str(layout[0]),
                     str(layout[1]),
                     color_frequencies_str))


def pickle_tran_mat(transition_matrix, properties):
    filename = filename_from_properties(properties)
    filename = './data/tran_ma/' + filename + ".pickle"
    with open(filename, 'wb') as file:
        pickle.dump(transition_matrix, file)


def unpickle_tran_mat(properties):
    filename = filename_from_properties(properties)
    filename = './data/tran_ma/' + filename + ".pickle"
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_rew_mat(transition_matrix, properties):
    filename = filename_from_properties(properties)
    filename = './data/rew_ma/' + filename + ".pickle"
    with open(filename, 'wb') as file:
        pickle.dump(transition_matrix, file)


def unpickle_rew_mat(properties):
    filename = filename_from_properties(properties)
    filename = './data/rew_ma/' + filename + ".pickle"
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def unpickle_mdp(properties):
    filename = filename_from_properties(properties)
    filename = "./data/mdp/" + properties['mdp'].__name__ + filename + ".pickle"
    with open(filename, "rb") as file:
        mdp = pickle.load(file)
    return mdp


def pickle_mdp(properties, mdp):
    type_ = mdp.__class__.__name__
    filename = filename_from_properties(properties)
    filename = "./data/mdp/" + type_ + filename + ".pickle"
    with open(filename, "wb") as file:
        pickle.dump(mdp, file)


"""
# Prepare Inputs of MDPs
"""


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

    out = np.ndarray(shape=((nr_fillings**nr_fields)*(nr_actions*nr_next_col), nr_fields+2), dtype=np.short)

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
    """
    for each action for each state for each state check if a transition can be made
    :param states: possible states, flat
    :param properties: properties of the world
    :return: transition matrix of shape (actions, states, states)
    """

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

                r1 = np.concatenate((st1[:a], st1[a+1:-2]))
                r2 = np.concatenate((st2[:a], st2[a+1:-2]))

                if s1 == 3 and s2 == 1152:
                    print("pause")

                if not np.any(r1 != r2):
                    n_act = st1[-2]
                    n_col = st1[-1]

                    if n_act == 0:  # put
                        if st1[a] == 3 and st2[a] == n_col:  # todo constant for empty field
                            trans[a, s1, s2] = 1
                    else:  # take
                        if st1[a] == n_col and st2[a] == 3:
                            trans[a, s1, s2] = 1
            # normalize row
            row_sum = np.sum(trans[a, s1])
            if row_sum == 0:
                trans[a, s1, s1] = 1
            else:
                trans[a, s1] /= row_sum
            if np.sum(trans[a, s1]) != 1:
                print("bad in", str(s1))
    return trans

def gen_reward_matrix(states, trans, properties):
    """
    takes the transition matrix and writes rewards where there are transitions indicated
    :param states:
    :param trans:
    :param properties:
    :return: rewards of shape (actions, states, states) - unnecessaryily complex as reward is determined by action
    """
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


"""
# Partitioned Generation
"""


def partitioned_gen_tran_ma(states, properties):
    return start_partitioned_job(states, properties, gen_transition_matrix_q)


def start_partitioned_job(states, properties, target):
    """
    just start one process per action - doesnt make sense for small problems...
    :param states:
    :param properties:
    :param target:
    :return:
    """
    nr_fields = properties['nr_fields']

    # trans = np.zeros((nr_fields, n_states, n_states))  # using list of lil_matrix matrices
    results = []

    processes = []
    queues = []
    for a in range(nr_fields):
        queues.append(mp.Queue())
        p = mp.Process(target=target, args=(states, properties, a, queues[-1]))
        p.start()

    for proc in processes:
        proc.join()

    for i, q in enumerate(queues):
        results.append(q.get().tocsr())
    return results


def gen_transition_matrix_q(states, properties, a, q):
    """
    basic idea is to calculate the indices of the possible next states from the current state, then set the trans proba
    :param states:
    :param properties:
    :param a:
    :param q:
    :return: transition matrix of shape (actions, states, states)
    """
    print("started", a)
    last_percent = 0

    a = int(a)

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]
    color_frequencies = properties['color_frequencies']

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
        if n_move == 1 and s[a] == n_col:
            s_new = np.concatenate((s[:a], [3], s[a + 1: nr_fields]))
        if s_new is not None:
            s_new_r = np.tile(s_new, (n_act_arr.shape[0], 1))
            s_new_full = np.concatenate((s_new_r, n_act_arr), axis=1)
            # calculate their index
            indices = np.dot(s_new_full, index_counters)

            # check indices
            actual_s = states[indices]
            res = np.where(np.any(actual_s != s_new_full))
            if len(res[0]) > 0:
                print("nicht gut")
            # set that index to 1/#states
            test_cf = color_frequencies[s_new_full[:, -1]]
            trans[i, indices] = color_frequencies[s_new_full[:, -1]] / np.sum(color_frequencies[s_new_full[:, -1]])
        else:
            trans[i, i] = 1
            pass
        # check
        if not np.isclose(np.sum(trans[i]), 1.0):
            print("bad in ", str(i))
    q.put(trans)


def partitioned_gen_rew_ma(states, properties):
    return start_partitioned_job(states, properties, gen_reward_matrix_q)


def gen_reward_matrix_q(states, properties, a, q):
    """
    same idea as in gen_transition_matrix_q but with rewards
    :param states:
    :param properties:
    :param a:
    :param q:
    :return:
    """

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
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
        if n_move == 1 and s[a] == n_col:
            s_new = np.concatenate((s[:a], [3], s[a + 1: nr_fields]))
        if s_new is not None:
            s_new_r = np.tile(s_new, (n_act_arr.shape[0], 1))
            s_new_full = np.concatenate((s_new_r, n_act_arr), axis=1)
            # calculate their index
            indices = np.dot(s_new_full, index_counters)
            # set that index to 1/#states
            trans[i, indices] = 16 - (distances[a]**2)  # 10 - (.5 * distances[a])

    q.put(trans)


"""
# evaluations
"""


def eval_mdp(mdp):
    """
    - get the utility per action / field
    - visualize
    - have an experiment, where one item is very unlikely to appear say less than 5%
    - see if we get different distribution
    :return:
    """
    pass


def count_expected_reward(rewards):
    sum_rew = []
    for r in rewards:
        sum_rew.append(np.sum(r) / r.nnz)

    print("avg rew", sum_rew)


def value_per_color(properties, states):
    """
    calculate the avarage expected value per color and field in the warehouse
    :param properties:
    :param states:
    :return:
    """
    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]
    layout = properties['layout']

    fields = np.zeros((2, nr_fillings, *layout))
    field_count = np.zeros((2, nr_fillings, *layout))

    mdp = unpickle_mdp(properties)

    for i, state in enumerate(states):
        color = state[-1]
        action = state[-2]
        suggested_action = mdp.policy[i]
        if suggested_action == 0 and mdp.P[suggested_action][i, i] == 1:
            continue
        field_index = np.unravel_index(int(suggested_action), layout)
        fields[(action, color, *field_index)] += mdp.V[i]
        field_count[(action, color, *field_index)] += 1
    # normalize
    fields[:, :3] /= field_count[:, :3]
    # field_count /= np.sum(field_count)
    print(":)")

    visualize_value_per_color(fields, "expected utility of action")
    visualize_value_per_color(field_count, "nr. of policy suggestions")
    return fields


def visualize_value_per_color(fields, name="show"):
    n_colors = fields.shape[1]
    n_actions = fields.shape[0]
    fig, axs = plt.subplots(n_colors - 1, n_actions, figsize=(4.2, 4.1))
    fig.suptitle(name)

    im_list = []
    # fill subplots
    for i in range(n_colors - 1):
        for j in range(n_actions):
            im_list.append(axs[i, j].imshow(fields[j, i]))

            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # ticks
            if i == n_colors -2:
                axs[i, j].set_xticks(np.arange(fields.shape[-1]))
                axs[i, j].set_xticklabels(np.arange(fields.shape[-1]) + 1)
            else:
                axs[i, j].set_xticklabels([])
            if j == 0:
                axs[i, j].set_yticks(np.arange(fields.shape[-2]))
                axs[i, j].set_yticklabels(np.arange(fields.shape[-2]) + 1)
            else:
                axs[i, j].set_yticklabels([])


            # color number
            if j == 0:
                axs[i, j].set_ylabel("color #" + str(i))

    # set colorbar
    cbaxes = fig.add_axes([0.86, 0.125, 0.03, 0.75])
    plt.colorbar(im_list[0], cax=cbaxes)

    # set labels
    axs[0, 0].set_title("put")
    axs[0, 1].set_title("take")
    # fig.tight_layout(h_pad=.5)
    fig.subplots_adjust(top=0.89, wspace=.16, hspace=.01, left=.1, right=.83)
    plt.show()
    fn = name.replace(" ", "_") + ".png"
    plt.savefig("./data/pics/" + fn)


def reward_from_following_policy(properties, ds_name='test_l'):
    states = generate_states(properties)

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]
    layout = properties['layout']

    mdp = unpickle_mdp(properties)
    moves = get_dataset(ds_name)
    moves_arr = ds_to_np(moves)

    fields = [3] * nr_fields  # empty fields
    fields.extend(moves_arr[0])  # set initial transition
    curr_state = np.array(fields)

    counters_fields = [nr_fillings] * nr_fields
    counters_actions = [nr_actions, nr_next_col]
    index_counters = np.array(counters_fields + counters_actions)
    index_counters[:-1] = index_counters[1:]
    index_counters[-1] = 1
    index_counters = np.cumprod(index_counters[::-1])[::-1]

    total_reward = 0
    for i in range(1, len(moves_arr)):
        curr_idx = np.dot(curr_state, index_counters)
        proposed_move = mdp.policy[curr_idx]

        # next state?
        next_state = curr_state.copy()
        if curr_state[-2] == 0:
            if curr_state[proposed_move] == 3:
                next_state[proposed_move] = curr_state[-1]
                total_reward += mdp.R[proposed_move][curr_idx]
            else:
                print("[!] Attempted to store in a non-empty field [!]")
                is_set = False
                for k in range(nr_fields):
                    if next_state[k] == 3:
                        print("alternative field:", k)
                        next_state[k] = curr_state[-1]
                        total_reward += mdp.R[k].max
                        is_set = True
                        break
                if not is_set:
                    print("[1] UNABLE TO PUT")
        else:
            if curr_state[proposed_move] != curr_state[-1]:
                print("[!] Attempted to take item that does not exist in that position [!]")
                is_set = False
                for j in range(nr_fields):
                    if next_state[j] == curr_state[-1]:
                        print("alternative field:", i)
                        next_state[j] = 3
                        total_reward += mdp.R[j].max
                        is_set = True
                        break
                if not is_set:
                    print("[1] UNABLE TO TAKE")
            else:
                next_state[proposed_move] = 3  # empty
                total_reward += mdp.R[proposed_move][curr_idx]

        next_state[-2:] = moves_arr[i]
        next_idx = np.dot(next_state, index_counters)
        # print("plus", mdp.R[proposed_move][curr_idx])

        curr_state = next_state
    return total_reward


def run_experiment(properties, new_tran=False, new_rew=False):
    print("starting experiment with the following properties:")
    print(properties)
    print("")
    states = generate_states(properties)
    print("generated", len(states), "states")


    if new_tran:
        print("generating new transition matrix")
        pickle_tran_mat(partitioned_gen_tran_ma(states, properties), properties)
    if new_rew:
        print("generating new reward matrix")
        pickle_rew_mat(partitioned_gen_rew_ma(states, properties), properties)

    data = unpickle_tran_mat(properties)
    rew = unpickle_rew_mat(properties)

    print("")
    print("loaded data - mdp prechecks may take a while")

    pi = properties['mdp'](transitions=data,
                              reward=rew,
                              discount=.95)
    pi.setVerbose()

    print("running")
    pi.run()

    print("saving results")
    pickle_mdp(properties, pi)
    print("done")

    return states


def real_greedy(properties, ds_name):
    """
    greedily fill the warehouse following reward based on Manhattan distance
    :param properties:
    :param ds_name:
    :return: sum of rewards
    """
    states = generate_states(properties)

    nr_fields = properties['nr_fields']
    nr_fillings = properties['nr_fillings']
    nr_actions = properties['nr_actions']
    nr_next_col = properties['nr_next_col']
    n_states = states.shape[0]
    layout = properties['layout']

    moves = get_dataset(ds_name)
    moves_arr = ds_to_np(moves)

    fields = [3] * nr_fields  # empty fields
    fields.extend(moves_arr[0])  # set initial transition
    curr_state = np.array(fields)

    counters_fields = [nr_fillings] * nr_fields
    counters_actions = [nr_actions, nr_next_col]
    index_counters = np.array(counters_fields + counters_actions)
    index_counters[:-1] = index_counters[1:]
    index_counters[-1] = 1
    index_counters = np.cumprod(index_counters[::-1])[::-1]

    rows = np.arange(layout[0])
    columns = np.arange(layout[1])

    distances = np.ones(layout)

    distances += columns
    distances = (distances.T + rows).T

    distances = distances.flatten()
    idcs = np.argsort(distances)

    total_reward = 0
    for i in range(1, len(moves_arr)):
        task = curr_state[-2]
        color = curr_state[-1]

        if task == 0:
            is_set = False
            for idc in idcs:
                if curr_state[idc] == 3:  # empty
                    curr_state[idc] = color
                    curr_state[-2:] = moves_arr[i]
                    total_reward += 16 - (distances[idc]**2)
                    is_set = True
                    break
            if not is_set:
                print("could not put item", curr_state, "with color", color)
                break
        elif task == 1:
            is_set = False
            for idc in idcs:
                if curr_state[idc] == color:
                    curr_state[idc] = 3  # empty after taking out
                    curr_state[-2:] = moves_arr[i]  # set up next state
                    total_reward += 16 - (distances[idc]**2)
                    is_set = True
                    break
            if not is_set:
                print("could not take item", curr_state, "with color", color)
                break

    return total_reward


def verify_trans(properties, states):
    old = np.array(gen_transition_matrix(states, properties), dtype=float)
    new_l = partitioned_gen_tran_ma(states, properties)
    new = np.ndarray((4, 1536, 1536))
    for k, item in enumerate(new_l):
        new[k] = item.toarray()

    diff = []
    wo = []
    for i, item in enumerate(old):
        diff.append(new[i]-old[i])
        print("diff:", np.sum(np.abs(diff[-1])))
        wo.append(np.isclose(new[i], old[i]))

    print("ok")


def distribution_from_data():
    # for each file
    datasets = {
        "cwt": './data/curr_warehousetraining.txt'
    }

    ds_freq = datasets.copy()

    for ds_name in datasets.keys():
        data = get_dataset(ds_name)
        data = ds_to_np(data)
        colors = np.array([r[1] for r in data])
        un, freq = np.unique(colors, return_counts=True)
        print("unique values", un)
        print("frequencies", freq)
        print("rel freq", freq / np.sum(freq))
        ds_freq[ds_name] = freq/np.sum(freq)

    return ds_freq


def check_min_size():
    """
    check if real_greedy has enough space here
    :return:
    """
    # for each file
    datasets = {
        "test_l": './data/warehouseorder2x2.txt',
        "test_s": './data/warehouseordernew.txt',
        "train": './data/warehousetraining2x2.txt',
        "cwo": './data/curr_warehouseorder.txt',
        "cwt": './data/curr_warehousetraining.txt'
    }

    properties = {
        'nr_fields': 4,
        'nr_fillings': 4,
        'nr_actions': 2,
        'nr_next_col': 3,
        'layout': (2, 2),
        'color_frequencies': np.array([0.33333, 0.33333, 0.33334]),
        # [0.33333, 0.6, 0.06667], [0.33333, 0.33333, 0.33334]
        'mdp': mdptb.PolicyIterationModified
    }

    layouts = [(2, 2), (2, 3), (3, 3)]

    for ds_name in datasets.keys():
        for layout in layouts:
            print("at", layout, ds_name)
            properties["layout"] = layout
            properties['nr_fields'] = layout[0] * layout[1]

            real_greedy(properties, ds_name)

def run_evaluation():
    """
    ok so we want to run:
    - sizes: 2x2, 2x3
    - mdps: ValueIteration
    - rewards: dist^2, linear
    - discounts: .5, .9, .99
    :return:
    """
    properties = {
        'nr_fields': 6,
        'nr_fillings': 4,
        'nr_actions': 2,
        'nr_next_col': 3,
        'layout': (2, 3),
        'color_frequencies': np.array([0.33333, 0.33333, 0.33334]),
        'mdp': mdptb.ValueIteration
    }

    # generate frequencies
    ds_freq = distribution_from_data()['cwt']

    # generate data
    # run mdp with different data
    states = run_experiment(properties, True, True)
    r_assumed_equal = reward_from_following_policy(properties, 'cwt')
    value_per_color(properties, states)

    properties['color_frequencies'] = ds_freq
    states = run_experiment(properties, True, True)

    # calculate reward of each policy
    r_ds_distribution = reward_from_following_policy(properties, 'cwt')
    value_per_color(properties, states)
    # calculate baseline
    r_baseline = real_greedy(properties, 'cwt')

    # very extreme distro
    extreme_distribution = np.array([.998, .001, .001])
    properties['color_frequencies'] = extreme_distribution

    states = run_experiment(properties, True, True)
    r_extreme = reward_from_following_policy(properties, 'cwt')
    value_per_color(properties, states)

    # print results
    print("baseline score:", r_baseline)
    print("equal dist.   :", r_assumed_equal)
    print("actual distrib:", r_ds_distribution)
    print("extreme dist  :", r_extreme)


if __name__ == "__main__":
    """
    run_evaluation produces the results that are described in the homework
    interim results will be saved in the data directory
    
    """
    run_evaluation()
