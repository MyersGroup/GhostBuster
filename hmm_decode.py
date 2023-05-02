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


@jit(nopython=True)
def add_in_log_space(y):

    if len(set(y)) == 1 and np.any(y == -np.inf):
        result = -np.inf
    else:

        x_star = np.max(y)
        result = x_star + np.log(np.exp(y - x_star).sum())

    return result


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
                    transitions[t - 1, state2, state]
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
                    transitions[t - 1, state, state2]
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


def get_transition_arr(t_admix, tau, gen_grid):
    gen_grid_diff = np.abs(np.diff(gen_grid))
    transition_arr = np.zeros((len(gen_grid_diff), len(tau), len(tau)))
    for i in range(len(gen_grid_diff)):
        scaling = 1 - np.exp(-gen_grid_diff[i] * t_admix)
        for state1 in range(len(tau)):
            for state2 in range(len(tau)):
                if state1 != state2:
                    transition_arr[i, state1, state2] = (
                        tau[state2] * scaling / np.sum(tau)
                    )

    for state in range(len(tau)):
        transition_arr[:, state, state] = 1 - np.sum(transition_arr[:, state], axis=1)

    transition_arr = np.log(transition_arr)
    return transition_arr


def Forward_backward(init_start, t_admix, probabilities, gen_grid):
    """
    Posterior decoding, using the forward-backward algorithm.
    """
    number_observations = len(probabilities[0])
    state_nums = np.arange(len(init_start))

    forwards_in = np.zeros((len(init_start), number_observations))
    forwards_in[:, 0] = init_start + probabilities[:, 0]
    backwards_in = np.zeros((len(init_start), number_observations))

    transition_arr = get_transition_arr(t_admix, np.exp(init_start), gen_grid)

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
    pot = np.exp(pot)
    gen_grid_diff = np.abs(np.diff(gen_grid))
    prob_of_recomb_prior = np.exp(-t_admix * gen_grid_diff)
    prob_of_atleast_one_recomb_and_state = np.zeros(
        (len(state_nums), number_observations - 1)
    )
    for state in state_nums:
        for oth in state_nums:
            if state != oth:
                prob_of_atleast_one_recomb_and_state[oth] += pot[state, oth]
            else:
                scaling = np.exp(init_start[state]) * (1 - prob_of_recomb_prior)
                scaling = scaling / (scaling + prob_of_recomb_prior)
                prob_of_atleast_one_recomb_and_state[oth] += pot[state, oth] * scaling

    prob_of_atleast_one_recomb = np.sum(prob_of_atleast_one_recomb_and_state, axis=0)

    numerator = np.sum(
        np.nan_to_num(
            t_admix
            * gen_grid_diff
            * prob_of_atleast_one_recomb
            / (1 - prob_of_recomb_prior),
            nan=0,
        )
    )

    denominator = np.sum(gen_grid_diff)
    t_admix_update = numerator / denominator

    pi_update = np.sum(prob_of_atleast_one_recomb_and_state, axis=1)
    pi_update /= np.sum(pi_update)

    print("t_admix = ", t_admix_update)
    return results, t_admix_update, pi_update, forward_prob


def Decode_grid(
    tree_left_bp,
    tree_right_bp,
    tree_left_bp_gen,
    tree_right_bp_gen,
    t_admix,
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
    post_seq, t_admix_update, pi_update, forward_prob = Forward_backward(
        starting_probabilities, t_admix, probabilities, gen_grid
    )
    post_seq /= np.sum(post_seq, axis=0)

    ## transform post_seq back to per-tree
    if per_tree_output:
        post_seq = bp_to_trees(
            post_seq, tree_left_bp, tree_right_bp, window_size=window_size
        )

    return post_seq, t_admix_update, pi_update, forward_prob
