"""
HMM based smoothing of local ancestry
code adapted from https://github.com/LauritsSkov/Introgression-detection
(MIT License)
"""
import numpy as np
from numba import jit
from numba.typed import List
import pdb
import pickle
import pandas as pd


@jit
def log_with_inf(x):
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


@jit(nopython=True)
def add_in_log_space(y):

    if len(set(y)) == 1 and np.any(y == -np.inf):
        result = -np.inf
    else:

        x_star = np.max(y)
        result = x_star + np.log(np.exp(y - x_star).sum())

    return result


def log_with_inf_array(matrix):

    res = np.zeros((len(matrix), len(matrix[0])))
    for rows in range(len(matrix)):
        for col in range(len(matrix[0])):
            res[rows, col] = log_with_inf(matrix[rows, col])

    return res


def log_with_inf_array3d(matrix):

    res = np.zeros((len(matrix), len(matrix[0]), len(matrix[0][0])))
    for rows in range(len(matrix)):
        for col in range(len(matrix[0])):
            for col2 in range(len(matrix[0][0])):
                res[rows, col, col2] = log_with_inf(matrix[rows, col, col2])
    return res


def trees_to_bp(
    probability,
    tree_left_bp,
    tree_right_bp,
    tree_left_bp_gen,
    tree_right_bp_gen,
    target_branch_length,
    window_size=1e3,
):
    assert len(probability[0]) == len(tree_left_bp)
    assert len(tree_left_bp) == len(tree_right_bp)
    res = []
    gen_grid = []
    bp_grid = []
    for i, (l, r) in enumerate(zip(tree_left_bp, tree_right_bp)):
        for j in range(int(l / window_size), int(r / window_size)):
            number_of_windows = target_branch_length[i]
            # number_of_windows = 3*(int(r/window_size) - int(l/window_size))
            res.append(probability[:, i] / number_of_windows)
            recomb_rate = (tree_right_bp_gen[i] - tree_left_bp_gen[i]) / (
                tree_right_bp[i] - tree_left_bp[i]
            )
            gen_grid.append(tree_left_bp_gen[i] + recomb_rate * (j * window_size - l))
            bp_grid.append(j * window_size)
    return np.array(res).T, np.array(gen_grid), np.array(bp_grid)


def bp_to_trees(probability, tree_left_bp, tree_right_bp, window_size=1e3):
    assert len(tree_left_bp) == len(tree_right_bp)
    res = np.zeros((len(probability), len(tree_left_bp)))
    count = 0
    for i, (l, r) in enumerate(zip(tree_left_bp, tree_right_bp)):
        count_i = 0
        for j in range(int(l / window_size), int(r / window_size)):
            res[:, i] += probability[:, count]
            count += 1
            count_i += 1
        res[:, i] /= count_i
    return np.array(res)


def make_hmm_from_file(markov_param):

    with open(markov_param) as data:
        for line in data:
            if "#" != line[0] and line != "\n":
                # Statenames
                if "states" in line:
                    txt = (
                        line.strip()
                        .split("=")[1]
                        .replace("'", "")
                        .replace(" ", "")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    states = np.array(txt.split(","))

                # transversions
                if "transitions" in line:
                    txt = (
                        line.strip()
                        .split("=")[1]
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "")
                    )
                    transitions = np.array([float(x) for x in txt.split(",")]).reshape(
                        len(states), len(states)
                    )

    # Log transform the transitions
    for i, row in enumerate(transitions):
        for j, col in enumerate(row):
            transitions[i][j] = log_with_inf(col)

    return transitions


def make_transition_matrix(tree_left_bp_gen, t_admix, props):
    tree_location = t_admix * np.abs(np.diff(tree_left_bp_gen, prepend=0))
    tree_location = np.minimum(tree_location, 1)
    transition_arr = np.zeros((len(tree_location), 2, 2))
    transition_arr[:, 0, 1] = tree_location * props[1]
    transition_arr[:, 1, 0] = tree_location * props[0]
    transition_arr[:, 0, 0] = 1 - transition_arr[:, 0, 1]
    transition_arr[:, 1, 1] = 1 - transition_arr[:, 1, 0]
    transition_arr = log_with_inf_array3d(transition_arr)
    return transition_arr


@jit(nopython=True)
def Forward_prob(
    init_start, transitions, probabilities, state_nums, number_observations, forwards_in
):
    """
    Returns the probability of seeing the given `observations` sequence,
    using the Forward algorithm.
    """

    for t in np.arange(1, number_observations):
        for state in state_nums:
            toadd = np.zeros(len(state_nums))
            for state2 in state_nums:
                toadd[state2] = (
                    transitions[t, state2, state]
                    + probabilities[state, t]
                    + forwards_in[state2, t - 1]
                )

            forwards_in[state, t] = add_in_log_space(toadd)

    toadd = np.zeros(len(init_start))
    for state in state_nums:
        toadd[state] = forwards_in[state, -1]

    final = add_in_log_space(toadd)

    return (final, forwards_in)


# weights, mutrates in prob also add factorials


@jit(nopython=True)
def Backward_prob(
    init_start, transitions, probabilities, state_nums, backwards, reversedlist
):
    """
    Returns the probability of seeing the given `observations` sequence,
    using the Backward algorithm.
    """

    # Fill out the matrix
    for t in reversedlist:
        for state in state_nums:

            toadd = np.zeros(len(state_nums))
            for state2 in state_nums:
                toadd[state2] = (
                    transitions[t, state, state2]
                    + probabilities[state2, t]
                    + backwards[state2, t]
                )

            backwards[state, t - 1] = add_in_log_space(toadd)

    toadd = np.zeros(len(init_start))
    for state in state_nums:
        toadd[state] = init_start[state] + probabilities[state, 0] + backwards[state, 0]

    final = add_in_log_space(toadd)

    return (final, backwards)


def Forward_backward(init_start, transition_arr, probabilities):
    """
    Posterior decoding, using the forward-backward algorithm.
    """
    number_observations = len(probabilities[0])
    state_nums = np.arange(len(init_start))

    forwards_in = np.zeros((len(init_start), number_observations))
    forwards_in[:, 0] = init_start + probabilities[:, 0]
    backwards_in = np.zeros((len(init_start), number_observations))

    forward_prob, forwards = Forward_prob(
        init_start,
        transition_arr,
        probabilities,
        state_nums,
        number_observations,
        forwards_in,
    )

    reversedlist = List()
    [reversedlist.append(x) for x in range(number_observations - 1, 0, -1)]
    backward_prob, backwards = Backward_prob(
        init_start,
        transition_arr,
        probabilities,
        state_nums,
        backwards_in,
        reversedlist,
    )

    results = np.exp(forwards + backwards - forward_prob)
    return results, forward_prob


@jit(nopython=True)
def makeprobability_of_transition_matrix(
    state_nums,
    number_observations,
    forwards,
    transitions,
    probabilities,
    backwards,
    forward_prob,
):

    pot = np.zeros((len(state_nums), len(state_nums), number_observations - 1))
    for state1 in state_nums:
        for t in range(number_observations - 1):
            pot[state1, :, t] = (
                forwards[state1, t]
                + transitions[state1, :]
                + probabilities[:, t + 1]
                + backwards[:, t + 1]
                - forward_prob
            )

    return pot


def TrainBaumWelsch(init_start, transitions, probabilities):
    """
    Trains the model once, using the forward-backward algorithm.
    """

    number_observations = len(probabilities[0])
    state_nums = np.arange(len(init_start))

    # Make and initialise forwards matrix
    forwards_in = np.zeros((len(init_start), number_observations))
    forwards_in[:, 0] = init_start + probabilities[:, 0]
    backwards_in = np.zeros((len(init_start), number_observations))

    forward_prob, forwards = Forward_prob(
        init_start,
        transitions,
        probabilities,
        state_nums,
        number_observations,
        forwards_in,
    )

    reversedlist = List()
    [reversedlist.append(x) for x in range(number_observations - 1, 0, -1)]
    backward_prob, backwards = Backward_prob(
        init_start, transitions, probabilities, state_nums, backwards_in, reversedlist
    )

    posat = forwards + backwards - forward_prob
    pot = makeprobability_of_transition_matrix(
        state_nums,
        number_observations,
        forwards,
        transitions,
        probabilities,
        backwards,
        forward_prob,
    )

    # Initial starting probabilities
    normalize = np.exp(posat).sum()
    start_prob = np.zeros(len(state_nums))

    # Transition probs
    trans = np.zeros((len(init_start), len(init_start)))

    for state in state_nums:
        state_prob = np.logaddexp.reduce(posat[state])
        start_prob[state] = np.exp(state_prob) / normalize

        for oth in state_nums:
            trans[state, oth] = np.exp(
                np.logaddexp.reduce(pot[state, oth]) - state_prob
            )

    for i, row in enumerate(trans):
        old_sum = row.sum()

        for j, col in enumerate(row):
            if old_sum == 0:
                trans[i][j] = 0
            else:
                trans[i][j] = col / old_sum

    return (start_prob, trans, forward_prob)


def TrainModel(model, probabilities, starting_probabilities):

    # Parameters (path to observations file, output file, model, weights file)

    # Load data
    transitions = make_hmm_from_file(model)
    print(np.exp(transitions))
    # Train model
    epsilon = 0.0001
    starting_probabilities, transitions, old_prob = TrainBaumWelsch(
        starting_probabilities, transitions, probabilities
    )
    print(transitions)
    for i in range(1000):
        transitions = log_with_inf_array(transitions)
        starting_probabilities = np.log(starting_probabilities)
        starting_probabilities, transitions, new_prob = TrainBaumWelsch(
            starting_probabilities, transitions, probabilities
        )
        print(transitions)

        if new_prob - old_prob < epsilon:
            break

        old_prob = new_prob

    return (np.log(starting_probabilities), log_with_inf_array(transitions))


def Decode(tree_left_bp_gen, t_admix_guess, probabilities, tau):

    # infered proportions
    starting_probabilities = np.log(tau)

    # Train Baum-Welsch
    # _, transitions = TrainModel(model, probabilities, starting_probabilities)

    # transitions = make_hmm_from_file(model)
    if starting_probabilities[0] > starting_probabilities[0]:
        a = [0.95, 0.05]
    else:
        a = [0.05, 0.95]
    transition_arr = make_transition_matrix(
        tree_left_bp_gen,
        t_admix_guess,
        a,
    )
    # Posterior decode the file
    post_seq, forward_prob = Forward_backward(
        starting_probabilities, transition_arr, probabilities
    )
    post_seq /= np.sum(post_seq, axis=0)
    return post_seq, forward_prob


def Decode_grid(
    tree_left_bp,
    tree_right_bp,
    tree_left_bp_gen,
    tree_right_bp_gen,
    target_branch_length,
    t_admix_guess,
    probabilities,
    tau,
    window_size=1000,
):
    """
    Use this for GB fitting
    """
    # infered proportions
    starting_probabilities = np.log(tau)

    # transitions = make_hmm_from_file(model)
    if starting_probabilities[0] > starting_probabilities[0]:
        a = [0.95, 0.05]
    else:
        a = [0.05, 0.95]

    starting_probabilities = np.log(a)
    ## transfor probabilities to per-kb + scaling
    print(np.unique(probabilities[1]).shape)
    probabilities, gen_grid, bp_grid = trees_to_bp(
        probabilities,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        target_branch_length,
        window_size=window_size,
    )

    ## change tree_left_bp_gen to a bp grid of 1kb interval
    transition_arr = make_transition_matrix(
        gen_grid,
        t_admix_guess,
        a,
    )

    # Posterior decode the file
    post_seq, forward_prob = Forward_backward(
        starting_probabilities, transition_arr, probabilities
    )
    post_seq /= np.sum(post_seq, axis=0)

    ## transform post_seq back to per-tree
    post_seq = bp_to_trees(
        post_seq, tree_left_bp, tree_right_bp, window_size=window_size
    )

    return post_seq, forward_prob


def Decode_save_output(
    tree_left_bp,
    tree_right_bp,
    tree_left_bp_gen,
    tree_right_bp_gen,
    target_branch_length,
    t_admix_guess,
    probabilities,
    tau,
    output,
    window_size=1000,
):

    # infered proportions
    starting_probabilities = np.log(tau)

    # transitions = make_hmm_from_file(model)
    if starting_probabilities[0] > starting_probabilities[0]:
        a = [0.95, 0.05]
    else:
        a = [0.05, 0.95]

    starting_probabilities = np.log(a)
    ## transfor probabilities to per-kb + scaling
    probabilities, gen_grid, bp_grid = trees_to_bp(
        probabilities,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        target_branch_length,
        window_size=window_size,
    )

    ## change tree_left_bp_gen to a bp grid of 1kb interval
    transition_arr = make_transition_matrix(
        gen_grid,
        t_admix_guess,
        a,
    )

    # Posterior decode the file
    post_seq, forward_prob = Forward_backward(
        starting_probabilities, transition_arr, probabilities
    )
    post_seq /= np.sum(post_seq, axis=0)

    ## combine post_seq with bp_grd and save as csv
    pd.DataFrame(
        data=np.vstack((bp_grid, post_seq)).T, columns=["start", "prob_0", "prob_1"]
    ).to_csv(output + "_posterior.csv", index=False, sep="\t")


if __name__ == "__main__":
    prefix = "../output/deni_relate_ghost_all"
    post = np.load(prefix + "_overall_membership_50.npy")
    tau = np.load(prefix + "_props_50.npy")
    gb_likelihood = (post.T / tau.T).T
    gb_likelihood = log_with_inf_array(gb_likelihood)
    gt = np.load(prefix + "_ground_truth_membership_50.npy")[0]
    print(np.corrcoef(post[0], gt))
    print(np.mean(post[0]))

    mask = np.load(prefix + "_mask_50.npy")
    f_pkl = open(prefix + "_tree_stats_1.pkl", "rb")
    tree_stats = pickle.load(f_pkl)
    f_pkl.close()
    tree_left_bp = np.array(tree_stats[1])[mask]
    tree_right_bp = np.array(tree_stats[2])[mask]
    tree_left_bp_gen = np.array(tree_stats[3])[mask]
    tree_right_bp_gen = np.array(tree_stats[4])[mask]
    target_branch_length = np.array(tree_stats[19])[mask]

    # post_seq, forward_prob = Decode(tree_left_bp_gen, 1850, gb_likelihood, tau)
    # print(np.corrcoef(post_seq[0], gt))
    # print(np.mean(post_seq[0]))

    # post_seq, forward_prob = Decode_grid(
    #     tree_left_bp,
    #     tree_right_bp,
    #     tree_left_bp_gen,
    #     tree_right_bp_gen,
    #     target_branch_length,
    #     1850,
    #     gb_likelihood,
    #     tau,
    # )
    # mask = ~np.isnan(post_seq[0])
    # print(np.corrcoef(post_seq[0][mask], gt[mask]))
    # print(np.mean(post_seq[0][mask]))
    # np.save("../output/deni_relate_ghost_hmm_all_posterior_membership_50.npy", post_seq)

    Decode_save_output(
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        target_branch_length,
        1850,
        gb_likelihood,
        tau,
        output="../output/deni_relate_ghost_hmm_all",
    )
