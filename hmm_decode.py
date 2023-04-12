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
from scipy.stats import hmean, gmean


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
    window_size=1e3,
):
    assert len(probability[0]) == len(tree_left_bp)
    assert len(tree_left_bp) == len(tree_right_bp)
    res = []
    gen_grid = []
    bp_grid = []
    for i, (l, r) in enumerate(zip(tree_left_bp, tree_right_bp)):
        for j in range(int(l / window_size), int(r / window_size)):
            # number_of_windows = target_branch_length[i]
            # number_of_windows = 3*(int(r/window_size) - int(l/window_size))
            res.append(probability[:, i])
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
                + transitions[t, state1, :]
                + probabilities[:, t + 1]
                + backwards[:, t + 1]
                - forward_prob
            )

    return pot


def Forward_backward(init_start, transition_arr, probabilities, gen_grid):
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

    posat = forwards + backwards - forward_prob
    results = np.exp(posat)
    pot = makeprobability_of_transition_matrix(
        state_nums,
        number_observations,
        forwards,
        transition_arr,
        probabilities,
        backwards,
        forward_prob,
    )
    gen_grid_diff = np.abs(np.diff(gen_grid, prepend=-1))
    trans = np.nan * np.ones((len(init_start), len(init_start)))
    for state in state_nums:
        state_prob = np.logaddexp.reduce(posat[state])
        for oth in state_nums:
            if state != oth:
                trans[state, oth] = np.exp(
                    np.logaddexp.reduce(
                        np.nan_to_num(
                            pot[state, oth] - np.log(gen_grid_diff[:-1]), nan=-np.inf
                        )
                    )
                    - state_prob
                )
    print(trans[0, 1] + trans[1, 0])
    trans = trans[0, 1] + trans[1, 0]  ### approximate updates, set temporarily
    return results, trans, forward_prob


def Decode_grid(
    tree_left_bp,
    tree_right_bp,
    tree_left_bp_gen,
    tree_right_bp_gen,
    transition_arr,
    probabilities,
    tau,
    window_size,
    per_tree_output=False,
):

    """
    Use this for GB fitting
    """
    # infered proportions
    starting_probabilities = np.log(tau)

    # if starting_probabilities[0] > starting_probabilities[1]:
    #     a = [0.95, 0.05]
    # else:
    #     a = [0.05, 0.95]
    # starting_probabilities = np.log(a)

    ## transfor probabilities to per-kb + scaling
    probabilities, gen_grid, bp_grid = trees_to_bp(
        probabilities,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        window_size=window_size,
    )

    # Posterior decode the file
    post_seq, trans, forward_prob = Forward_backward(
        starting_probabilities, transition_arr, probabilities, gen_grid
    )
    post_seq /= np.sum(post_seq, axis=0)

    ## transform post_seq back to per-tree
    if per_tree_output:
        post_seq = bp_to_trees(
            post_seq, tree_left_bp, tree_right_bp, window_size=window_size
        )

    return post_seq, trans, forward_prob


if __name__ == "__main__":
    prefix = "../output/deni_relate_ghost_hmm_all"
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
    target_branch_length = []
    for tid in range(len(tree_stats[19][0])):
        if mask[tid]:
            target_branch_length.append(
                np.mean(
                    tree_stats[19][0][tid]
                    + [tree_stats[2][tid] // 1000 - tree_stats[1][tid] // 1000]
                )
            )
