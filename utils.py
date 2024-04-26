"""
utils.py: contains basic helper function used throughout the code
Includes: boolean(), make_one_hot(), mask_for_dodgy_trees(), downsample_trees(), write_coal()
"""

import argparse
import numpy as np
from sklearn.calibration import calibration_curve
import pandas as pd
import copy
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
import pdb
import numba as nb
from numba import jit
from numba.typed import List
import pickle
import time 

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_numba_nested_list(arr_list):
    ## careful with empty lists
    arr_list_nb = List()
    for i in arr_list:
        inner = List()
        for j in i:
            inner.append(j)
        arr_list_nb.append(inner)
    return arr_list_nb


def make_one_hot(X, max_X=None):
    X = np.array(X, dtype="int")
    classes = np.arange(0, max_X, 1) if max_X is not None else np.sort(np.unique(X))
    # Y = []
    if len(X.shape) == 2:
        Y = np.zeros((len(classes), X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        Y = np.zeros((len(classes), X.shape[0]))
    for c in range(len(classes)):
        # Y.append(scipy.sparse.csr_matrix(np.array(X == c, dtype='int')))
        Y[c] = np.array(X == classes[c], dtype="int")
    return Y


def mask_for_dodgy_trees(recomb_rates, masking_thresh):
    recomb_rates = np.array(recomb_rates)
    mask = recomb_rates <= np.nanpercentile(recomb_rates, (masking_thresh) * 100)
    return mask


def downsample_trees(ground_truth, pop_index, downsample_frac):
    ## higher downsample_frac means less downsampling, range (0, 1)
    assert downsample_frac < 1 and downsample_frac > 0
    mask = np.ones_like(ground_truth[0], dtype=bool)
    downsample_mask = (
        np.random.rand(np.sum(ground_truth[pop_index] == 1)) < downsample_frac
    )
    print(
        "Downsampling population "
        + str(pop_index)
        + " to remove "
        + str(np.sum(1 - downsample_mask))
        + " trees"
    )
    mask[ground_truth[pop_index] == 1] = downsample_mask
    return mask

def write_coal(gamma_arr, filename, labs, output, epoch_intervals):
    if type(labs) == dict:
        labs = list(labs.keys())
    if len(labs) > 10:
        epoch_intervals_pow = np.power(10, epoch_intervals)
        f = open(output + "_" + filename + '.all', "w")
        f.write(" ".join(labs) + "\n")
        for val in epoch_intervals_pow:
            f.write("%s " % val)
        f.write("\n")
        for i in range(gamma_arr.shape[0]):
            for j in range(gamma_arr.shape[1]):
                f.write(str(i) + " " + str(j) + " ")
                for e in range(gamma_arr.shape[2]):
                    f.write(str(gamma_arr[i][j][e]) + " ")
                f.write("\n")

        f.close()

        min_avg_coal = []
        mean_popsize = -np.log10(np.nanmean(gamma_arr[:,:,1:-1], axis=2))
        for i in range(gamma_arr.shape[1]):
            min_diff_arr = []
            for j in range(gamma_arr.shape[0]):
                min_diff_with_other_comp = -np.inf
                for k in range(gamma_arr.shape[0]):
                    if k != j:
                        min_diff_with_other_comp = max(min_diff_with_other_comp, mean_popsize[j, i] - mean_popsize[k, i])
                min_diff_arr.append(min_diff_with_other_comp)
            min_avg_coal.append(min_diff_arr)

        min_avg_coal = np.array(min_avg_coal)  ## N_ref x N_clusters
        top_groups = []
        for j in range(gamma_arr.shape[0]):
            top_groups.extend(np.argsort(min_avg_coal[:, j])[0:10//gamma_arr.shape[0]])
        
        labs = np.array(labs)[top_groups].tolist()
        gamma_arr = gamma_arr[:, top_groups]

    epoch_intervals_pow = np.power(10, epoch_intervals)
    filename = output + "_" + filename
    f = open(filename, "w")
    f.write(" ".join(labs) + "\n")
    for val in epoch_intervals_pow:
        f.write("%s " % val)
    f.write("\n")
    for i in range(gamma_arr.shape[0]):
        for j in range(gamma_arr.shape[1]):
            f.write(str(i) + " " + str(j) + " ")
            for e in range(gamma_arr.shape[2]):
                f.write(str(gamma_arr[i][j][e]) + " ")
            f.write("\n")

    f.close()


def calculate_accuracy(own_membership, ground_truth_membership):
    membership_thresh = make_one_hot(
        np.argmax(own_membership, axis=0), len(own_membership)
    )
    ## Evaluate accuracy
    acc_arr = np.zeros((len(own_membership), len(ground_truth_membership)))
    ## Permute the ground truth and membership accordingly
    ground_truth_membership = ground_truth_membership[
        np.array(np.argsort(np.sum(ground_truth_membership, axis=1)), dtype=int)
    ]
    membership_thresh = membership_thresh[
        np.array(np.argsort(np.sum(membership_thresh, axis=1)), dtype=int)
    ]
    for i in range(len(membership_thresh)):
        for j in range(len(ground_truth_membership)):
            acc = np.sum(
                (membership_thresh[i] == 1) & (ground_truth_membership[j] == 1)
            )
            acc_arr[i][j] = acc
    print("Confusion matrix = " + str(acc_arr))
    print(
        "R2 = "
        + str(np.abs(np.corrcoef(own_membership[i], ground_truth_membership[i])[0, 1]))
    )


def write_calibration(args, own_membership, ground_truth_membership):
    num_rows, num_cols = ground_truth_membership.shape

    filename = "calibration.txt"
    filename = args.output + "_" + filename
    with open(filename, "w") as f:
        for i in range(len(own_membership)):
            for j in range(num_rows):
                y, x = calibration_curve(
                    ground_truth_membership[j],
                    own_membership[i],
                    n_bins=20,
                )
                for k in range(len(x)):
                    f.write(
                        str(i) + " " + str(j) + " " + str(x[k]) + " " + str(y[k]) + "\n"
                    )


def filter_recomb_rate(
    masking_threshold,
    tree_left_bp,
    recomb_rates,
    chr_map,
):
    recomb_rates = np.array(recomb_rates)
    mask_dodgy = np.ones_like(recomb_rates, dtype='bool')
    for chr in np.unique(chr_map):
        recomb_rates_chr = recomb_rates[chr_map == chr]
        for i, r in enumerate(recomb_rates_chr):
            if np.isnan(r):
                recomb_rates_chr[
                    np.abs(np.array(tree_left_bp)[chr_map == chr] - np.array(tree_left_bp)[chr_map == chr][i]) < 500000
                ] = np.inf
        recomb_rates_chr = np.nan_to_num(recomb_rates_chr, posinf=np.nan)
        mask_dodgy[chr_map == chr] = ~np.isnan(recomb_rates_chr)

        recomb_0_thresh = np.sum(np.array(recomb_rates_chr) <= 0) / len(recomb_rates_chr)
        mask_dodgy[chr_map == chr]  *= ~mask_for_dodgy_trees(
            recomb_rates_chr,
            recomb_0_thresh,
        )
        mask_dodgy[chr_map == chr] *= mask_for_dodgy_trees(
            recomb_rates_chr,
            1 - masking_threshold,
        )

    mask_dodgy = np.array(mask_dodgy)

    print(
        "Filtering based on recombination rate, trees remaining: "
        + str(sum(mask_dodgy))
        + " average recomb. rate: "
        + str(np.mean(np.array(recomb_rates)[mask_dodgy]))
    )
    return mask_dodgy


def filter_bstat(
    masking_threshold,
    tree_left_bp,
    b_stat,
    chr_map,
):
    b_stat = np.array(b_stat)
    mask_dodgy = np.ones_like(b_stat, dtype='bool')
    if masking_threshold is None:
        return mask_dodgy
    for chr in np.unique(chr_map):
        b_stat_chr = b_stat[chr_map == chr]
        for i, r in enumerate(b_stat_chr):
            if np.isnan(r):
                b_stat_chr[
                    np.abs(np.array(tree_left_bp)[chr_map == chr] - np.array(tree_left_bp)[chr_map == chr][i]) < 500000
                ] = np.inf
        b_stat_chr = np.nan_to_num(b_stat_chr, posinf=np.nan)
        mask_dodgy[chr_map == chr] = ~np.isnan(b_stat_chr)

        bstat_0_thresh = np.sum(np.array(b_stat_chr) <= 0) / len(b_stat_chr)
        mask_dodgy[chr_map == chr]  *= ~mask_for_dodgy_trees(
            b_stat_chr,
            bstat_0_thresh,
        )
        mask_dodgy[chr_map == chr] *= mask_for_dodgy_trees(
            b_stat_chr,
            1 - masking_threshold,
        )

    mask_dodgy = np.array(mask_dodgy)

    print(
        "Filtering based on b-statistic, trees remaining: "
        + str(sum(mask_dodgy))
        + " average b-stat: "
        + str(np.mean(np.array(b_stat)[mask_dodgy]))
    )
    return mask_dodgy


def filter_prior_likelihood(
    args,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    n_unique_groups,
    n_epochs,
    n_trees,
):
    own_membership = np.ones((1, n_trees))
    masked_trees_index = np.arange(0, n_trees)
    gamma_arr = np.zeros(
        (len(own_membership), n_unique_groups, n_epochs - 1),
        dtype="float64",
    )
    for j in range(len(own_membership)):
        n = compute_gamma_num(
            own_membership[j],
            None,
            proportion_of_coalescing_all,
            epoch_index_all,
            n_unique_groups,
            masked_trees_index,
            n_epochs,
        )
        for i in range(n_unique_groups):
            d = compute_gamma_denom(own_membership[j], denom[i], n_epochs)
            gamma_arr[j][i] = copy.deepcopy(n[i] / d)  # n/d #

    tau = np.mean(own_membership, axis=1)

    ll_k1 = get_epochwise_likelihood(
        args,
        gamma_arr,
        tau,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        n_unique_groups,
        n_epochs,
        n_trees,
        "likelihood_k1.npy",
    )

    for i in range(n_unique_groups):
        d = compute_gamma_denom(own_membership[j], np.sum(denom, axis=0), n_epochs)
        gamma_arr[j][i] = copy.deepcopy(np.sum(n, axis=0) / d)  # n/d #
    ll_prior = get_epochwise_likelihood(
        args,
        gamma_arr,
        tau,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        n_unique_groups,
        n_epochs,
        n_trees,
        "likelihood_prior.npy",
    )

    print(
        "Filtered "
        + str(np.sum(np.sum(ll_k1, axis=1) < np.sum(ll_prior, axis=1)))
        + " trees based on likelihood filter"
    )
    return np.sum(ll_k1, axis=1) > np.sum(ll_prior, axis=1)


def filter_opportunity(
    args,
    ts_list,
    mutrate_opportunity_target,
    mutrate_logpmf_target,
    epoch_index_all,
    proportion_of_coalescing_all,
    denom,
    mask_dodgy,
):
    chr_list = []
    for c in range(len(ts_list)):
        num_of_trees_in_chr = [c + 1] * ts_list[c].num_trees
        chr_list.extend(num_of_trees_in_chr)

    # mutrate_opportunity_target = np.tile(
    #    np.array(mutrate_opportunity_target), (len(args.sample_id), 1)
    # )
    mutrate_opportunity_target_masked = np.array(mutrate_opportunity_target)[mask_dodgy]
    mutrate_opportunity_thresh = np.percentile(
        mutrate_opportunity_target_masked, args.masking_threshold * 100, axis=0
    )

    mutrate_logpmf_target_masked = np.array(mutrate_logpmf_target)[mask_dodgy]
    mutrate_logpmf_thresh = []
    for epoch in range(mutrate_opportunity_target_masked.shape[1]):
        mutrate_logpmf_target_masked_ep = mutrate_logpmf_target_masked[:, epoch]
        if sum(np.isnan(mutrate_logpmf_target_masked_ep)) == len(
            mutrate_logpmf_target_masked_ep
        ):
            mutrate_logpmf_thresh.append(0)
        else:
            mutrate_logpmf_thresh.append(
                np.percentile(
                    mutrate_logpmf_target_masked_ep[
                        ~np.isnan(mutrate_logpmf_target_masked_ep)
                    ],
                    args.masking_threshold * 100,
                )
            )

    epoch_bin_mask = np.ones(
        (
            mutrate_opportunity_target_masked.shape[0],
            mutrate_opportunity_target_masked.shape[1] - 1,
        ),
        dtype=bool,
    )
    for epoch in range(epoch_bin_mask.shape[1]):
        epoch_bin_mask[:, epoch] = (
            mutrate_opportunity_target_masked[:, epoch]
            >= mutrate_opportunity_thresh[epoch]
        ) & (mutrate_logpmf_target_masked[:, epoch] >= mutrate_logpmf_thresh[epoch])

    for ref_gp in range(len(denom)):
        denom[ref_gp] = denom[ref_gp] * (epoch_bin_mask.T)
    for tid in range(len(epoch_index_all)):
        indices_to_remove = []
        for ce in range(len(epoch_index_all[tid])):
            if not epoch_bin_mask[tid, epoch_index_all[tid][ce]]:
                indices_to_remove.append(ce)
        for i_remove in sorted(indices_to_remove, reverse=True):
            del epoch_index_all[tid][i_remove]
            del proportion_of_coalescing_all[tid][i_remove]

    if args.ignore_first_epoch and args.ignore_last_epoch:
        mask_dodgy_low_evidence = (
            np.sum(epoch_bin_mask[:, 1:-1], axis=1) == epoch_bin_mask[:, 1:-1].shape[1]
        )
    if args.ignore_first_epoch and not args.ignore_last_epoch:
        mask_dodgy_low_evidence = (
            np.sum(epoch_bin_mask[:, 1:], axis=1) == epoch_bin_mask[:, 1:].shape[1]
        )
    if not args.ignore_first_epoch and args.ignore_last_epoch:
        mask_dodgy_low_evidence = (
            np.sum(epoch_bin_mask[:, :-1], axis=1) == epoch_bin_mask[:, :-1].shape[1]
        )
    else:
        mask_dodgy_low_evidence = (
            np.sum(epoch_bin_mask, axis=1) == epoch_bin_mask.shape[1]
        )

    mask_dodgy_low_evidence = np.sum(epoch_bin_mask, axis=1) > 8

    print(
        "Filtering based on opportunity filter, trees remaining: "
        + str(sum(mask_dodgy[mask_dodgy] * mask_dodgy_low_evidence))
    )
    return mask_dodgy_low_evidence


def load_mask_csv(args, membership_mask, tree_left_bp, tree_right_bp, chr_map):
    mask_dodgy = np.zeros(len(tree_left_bp), dtype="bool")
    count = 0
    for chr_count, chr in enumerate(np.unique(chr_map)):
        membership_mask_chr = membership_mask[membership_mask[membership_mask.columns[0]] == chr]
        membership_mask_chr[membership_mask.columns[1]] = membership_mask_chr[membership_mask.columns[1]] // args.force_build
        membership_mask_chr[membership_mask.columns[1]] = membership_mask_chr[membership_mask.columns[1]].astype(int)

        for tree_left_i, tree_right_i in zip(np.array(tree_left_bp)[chr_map == chr], np.array(tree_right_bp)[chr_map == chr]):
            for tree_pos in range(int(tree_left_i // args.force_build), int(tree_right_i // args.force_build)):
                if (
                    tree_pos
                    in membership_mask_chr[membership_mask.columns[1]].values.tolist()
                ):
                    mask_dodgy[count] = True
            
            count += 1
    print("Number of trees = " + str(np.sum(mask_dodgy)))
    return mask_dodgy


def get_epoch_interval(args, ts_list):
    coal_times = []
    for ts in ts_list:
        tree = ts.first()
        for tid in range(ts.num_trees):
            for sample_id in args.sample_id:
                parent = sample_id
                while parent != tree.root:
                    parent = tree.parent(parent)
                    coal_times.append(tree.time(parent))
            tree.next()

    coal_times = np.array(coal_times)
    if args.ignore_first_epoch:
        coal_times = coal_times[coal_times > math.pow(10, args.start_time) / 28]
    if args.ignore_last_epoch:
        coal_times = coal_times[coal_times < math.pow(10, args.end_time) / 28]

    def equalObs(x, nbin):
        nlen = len(x)
        return np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))

    n, bins, patches = plt.hist(
        coal_times, equalObs(coal_times, args.num_epochs - 2), edgecolor="black"
    )
    return np.array([-np.inf] + np.log10(bins).tolist() + [np.inf], dtype="float64")


@jit(nopython=True, fastmath=True)
def compute_gamma_num(
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    num_ref_groups,
    n_epochs,
    target_branch_length,
    ignore_first_epoch,
    ignore_last_epoch,
):
    num_full_tree = np.zeros((num_ref_groups, n_epochs - 1), dtype="float64")
    for n_site in range(len(own_membership)):
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[n_site]
        epoch_index_in_tree = epoch_index_all[n_site]
        target_branch_length_tree = target_branch_length[n_site]
        for i in range(len(proportion_of_coalescing_in_tree)):
            if (
                (
                    ignore_first_epoch
                    and not ignore_last_epoch
                    and epoch_index_in_tree[i] >= 1
                )
                or (
                    ignore_last_epoch
                    and not ignore_first_epoch
                    and epoch_index_in_tree[i] < n_epochs - 2
                )
                or (
                    ignore_first_epoch
                    and ignore_last_epoch
                    and epoch_index_in_tree[i] >= 1
                    and epoch_index_in_tree[i] < n_epochs - 2
                )
                or (not ignore_first_epoch and not ignore_last_epoch)
            ):
                epoch = epoch_index_in_tree[i]
                prev_gamma_e = prev_gamma[:, epoch]
                num = prev_gamma_e * proportion_of_coalescing_in_tree[i]
                sum_of_num = np.sum(num)
                if (
                    sum_of_num != 0
                ):  ## sometimes the num are less than python float64 precision, we ignore those coal events while calculating
                    num = num / sum_of_num
                common_term = (
                    own_membership[n_site] / target_branch_length_tree[i]
                )
                num_full_tree[:, epoch] += common_term * num
    return num_full_tree


def compute_gamma_denom(own_membership, denom):
    eps = 1e-200
    denom_1 = np.sum((own_membership * denom.T).T, axis=0)
    return denom_1 + eps


@jit(nopython=True, fastmath=True)
def compute_gamma_denom_eventwise(
    own_membership,
    denom,
    epoch_index_all,
    num_ref_groups,
    n_epochs,
    target_branch_length,
    ignore_first_epoch,
    ignore_last_epoch,
    tree_left_bp,
    tree_right_bp,
    window_size,
):
    eps = 1e-200
    denom_1 = np.zeros((num_ref_groups, n_epochs - 1), dtype="float64")
    count_site = 0
    for tree in range(len(denom)):
        epoch_index_in_tree = epoch_index_all[tree]
        denom_in_tree = denom[tree]
        for _ in range(
            int(tree_left_bp[tree] / window_size),
            int(tree_right_bp[tree] / window_size),
        ):
            count_i = 0
            for i, denom_coal in enumerate(denom_in_tree):
                epoch = epoch_index_in_tree[i]
                if (
                    (ignore_first_epoch and not ignore_last_epoch and epoch >= 1)
                    or (
                        ignore_last_epoch
                        and not ignore_first_epoch
                        and epoch < n_epochs - 2
                    )
                    or (
                        ignore_first_epoch
                        and ignore_last_epoch
                        and epoch >= 1
                        and epoch < n_epochs - 2
                    )
                    or (not ignore_first_epoch and not ignore_last_epoch)
                ):
                    denom_1 += (
                        denom_coal
                        * own_membership[count_site]
                        / target_branch_length[tree][count_i]
                    )
                    count_i += 1
            count_site += 1

    return denom_1 + eps


def update_membership_epochwise(
    proportion_of_coalescing_in_tree,
    epoch_index_in_tree,
    denom,
    gamma_arr,
    tid,
    ignore_first_epoch,
    ignore_last_epoch,
    n_epochs,
):
    log_num_em_j_i = np.zeros(n_epochs - 3)
    for i in range(len(proportion_of_coalescing_in_tree)):
        if (
            ignore_first_epoch
            and ignore_last_epoch
            and epoch_index_in_tree[i] >= 1
            and epoch_index_in_tree[i] < n_epochs - 2
        ):
            log_num_em_j_i[epoch_index_in_tree[i] - 1] += np.log(
                sum(
                    gamma_arr[:, epoch_index_in_tree[i]]
                    * proportion_of_coalescing_in_tree[i]
                ),
            )

    if ignore_first_epoch and ignore_last_epoch:
        log_denom_em_j = -sum(gamma_arr[:, 1:-1] * denom[:, 1:-1, tid])
    elif ignore_first_epoch and not ignore_last_epoch:
        log_denom_em_j = -sum(gamma_arr[:, 1:] * denom[:, 1:, tid])
    elif ignore_last_epoch and not ignore_first_epoch:
        log_denom_em_j = -sum(gamma_arr[:, :-1] * denom[:, :-1, tid])
    else:
        log_denom_em_j = -sum(gamma_arr * denom[:, :, tid])
    return log_num_em_j_i, log_denom_em_j


def get_epochwise_likelihood(
    args,
    gamma_arr,
    tau,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    n_unique_groups,
    n_epochs,
    n_trees,
    name=None,
):
    masked_trees_index = np.arange(0, n_trees)
    assert (gamma_arr >= 0).all()
    print(gamma_arr)
    own_membership_update = np.ones((1, n_trees), dtype="float64")
    log_num_em = np.zeros((1, n_trees, n_epochs - 3), dtype="float64")
    log_denom_em = np.zeros((1, n_trees, n_epochs - 3), dtype="float64")
    count_masked_trees = 0

    for tid in masked_trees_index:
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
        epoch_index_in_tree = epoch_index_all[tid]
        for j in range(1):
            log_num_em_j, log_denom_em_j = update_membership_epochwise(
                proportion_of_coalescing_in_tree,
                epoch_index_in_tree,
                denom,
                gamma_arr[j],
                tid,
                args.ignore_first_epoch,
                args.ignore_last_epoch,
                n_epochs,
            )
            log_num_em[j, count_masked_trees] = log_num_em_j
            log_denom_em[j, count_masked_trees] = log_denom_em_j
        count_masked_trees += 1
    own_membership_update = np.exp(log_num_em + log_denom_em)

    for j in range(1):
        own_membership_update[j] *= tau[j]

    ll_per_tree = np.log(np.sum(own_membership_update, axis=0))
    if name is not None:
        np.save(name, ll_per_tree)
    return ll_per_tree


def neighbour_smoothing(post, dist, window=3, alpha=1e-6):
    ## input: post - C x #Trees matrix containing posterior probabilities to smooth
    ## input: dist - #Trees x 1 vector containing position in cM for each tree
    ## input: alpha - smoothing hyperparameter
    ## output: post_smooth - C x #Trees matrix containing smooth posterior probs
    window = int(window)
    post = np.pad(post, ((0, 0), (window, window)), "constant", constant_values=0)
    dist = np.pad(dist, (window, window), "constant", constant_values=0)
    post_smooth = copy.deepcopy(post)
    for i in range(window, post.shape[1] - window):
        post_smooth[:, i] = post[:, i]
        for w in range(1, window):
            post_smooth[:, i] += (
                np.exp(-alpha * np.abs(dist[i] - dist[i - w])) * post[:, i - w]
                + np.exp(-alpha * np.abs(dist[i + w] - dist[i])) * post[:, i + w]
            )
    post_smooth /= np.sum(post_smooth, axis=0)
    return post_smooth[:, window:-window]


def load_gamma(path, groups, ref_groups):
    if ".npy" in path:
        return np.load(path)
    elif ".coal" in path:
        with open(path) as f:
            header = f.readline().strip("\n").split(" ")
        header = np.array(header)
        if len(np.intersect1d(ref_groups, header)) != len(ref_groups):
            print(
                "Groups in header do not match groups in input, groups in header are: "
                + str(header)
            )
            import pdb; pdb.set_trace()
            raise ValueError
        groups_to_index = []
        ref_groups_to_index = []
        for g in groups:
            try:
                groups_to_index.append(np.where(header == g)[0][0])
            except:
                groups_to_index.append(np.nan)
        for g in ref_groups:
            try:
                ref_groups_to_index.append(np.where(header == g)[0][0])
            except:
                ref_groups_to_index.append(np.nan)
        df = pd.read_csv(path, sep="\s+", header=None, skiprows=[0, 1])
        gamma_arr = np.nan*np.ones((len(groups), len(ref_groups), df.shape[1] - 2))
        for i, gid1 in enumerate(groups_to_index):
            for j, gid2 in enumerate(ref_groups_to_index):
                if not np.isnan(gid2):
                    gamma_arr[i, j] = df[(df[0] == gid1) & (df[1] == gid2)].values[:, 2:]
        
        return gamma_arr
    else:
        print("Unsupported file format for gamma files")

# def load_gamma(path, groups, ref_groups):
#     if ".npy" in path:
#         return np.load(path)
#     elif ".coal" in path:
#         with open(path) as f:
#             header = f.readline().strip("\n").split(" ")
#         header = np.array(header)
#         if len(np.intersect1d(ref_groups, header)) != len(ref_groups):
#             print(
#                 "Groups in header do not match groups in input, groups in header are: "
#                 + str(header)
#             )
#             raise ValueError
#         groups_to_index = [np.where(header == g)[0][0] for g in groups]
#         ref_groups_to_index = [np.where(header == g)[0][0] for g in ref_groups]
#         df = pd.read_csv(path, sep="\s+", header=None, skiprows=[0, 1])
#         gamma_arr = np.zeros((len(groups), len(ref_groups), df.shape[1] - 2))
#         for i, gid1 in enumerate(groups_to_index):
#             for j, gid2 in enumerate(ref_groups_to_index):
#                 gamma_arr[i, j] = df[(df[0] == gid1) & (df[1] == gid2)].values[:, 2:]
#         print(np.nan_to_num(gamma_arr, nan=0))
#         return np.nan_to_num(gamma_arr, nan=0)
#     else:
#         print("Unsupported file format for gamma files")

def load_props(path):
    if ".npy" in path:
        return np.load(path)
    elif ".txt" in path:
        return np.loadtxt(path)
    else:
        return np.array(path.split(" "), dtype="float")


def get_target_branch_length(
    args,
    poplabels,
    ts_list,
    chrs,
    mask_dodgy,
    sample_list,
):
    """
    Calculates the branch length of the target population
    """
    target_branch_length = []
    for sample_no, sample in enumerate(sample_list):
        count_all_tree, count_all_tree2 = 0, 0
        target_branch_length_sample = List()
        leave_one_sample_out = list(set(sample_list) - set([sample]))
        for chr_no, chr in enumerate(chrs):
            if args.branch_persistence_file_prefix is not None:
                branch_persistence_file_name = args.branch_persistence_file_prefix + "_chr" + str(chr) + "_sample" + str(sample) + ".pkl"
            else:
                branch_persistence_file_name = args.output + "_branch_persistence_chr" + str(chr) + "_sample" + str(sample) + ".pkl"
            try:
                f_pkl = open(branch_persistence_file_name, "rb")
                (force_build_file, start_time, end_time, ignore_first_epoch, ignore_last_epoch, masking_threshold, poplabels_file, target_branch_length_sample_chr) = pickle.load(f_pkl)
                f_pkl.close()
                if (force_build_file == args.force_build) & (start_time == args.start_time) & (end_time == args.end_time) & (ignore_first_epoch == args.ignore_first_epoch) & (ignore_last_epoch == args.ignore_last_epoch) & (masking_threshold==args.masking_threshold) & np.all(poplabels_file[list(set(np.arange(len(poplabels_file))) - set(args.sample_id))] == poplabels.values[list(set(np.arange(len(poplabels_file))) - set(args.sample_id))]):
                    ##convert to numba list
                    for i in target_branch_length_sample_chr:
                        numba_i = List().empty_list(nb.types.float64)
                        for j in i:
                            numba_i.append(j)
                        target_branch_length_sample.append(numba_i)
                    print("Loaded branch persistence statistics from: " + str(branch_persistence_file_name))
                    continue
                else:
                    print("Branch persistence statistics file does not match the current settings, recomputing...")
                    raise Exception
            except:
                print("Saving branch persistence statistics to: " + str(branch_persistence_file_name))
                target_branch_length_sample_chr = []
                ts = ts_list[chr_no]
                ts_edges = ts.edges()

                ## calculate tree_left and tree_right for masked trees
                tree_left_bp_chr, tree_right_bp_chr = [], []
                for tree in ts.trees():
                    if (
                        tree.interval[1] // args.force_build - tree.interval[0] // args.force_build
                        > 0
                    ):
                        if mask_dodgy[sample_no][count_all_tree]:
                            tree_left_bp_chr.append(tree.interval[0])
                            tree_right_bp_chr.append(tree.interval[1])
                        count_all_tree += 1

                ## calculate bp_grid for masked trees
                bp_grid = []
                num_sites_per_tree = []
                for i, (l, r) in enumerate(zip(tree_left_bp_chr, tree_right_bp_chr)):
                    num_sites_per_tree.append(r // args.force_build - l // args.force_build)
                    for j in range(int(l / args.force_build), int(r / args.force_build)):
                        bp_grid.append(j)
                bp_grid = np.array(bp_grid)
                num_sites_per_tree = np.array(num_sites_per_tree, dtype='int')

                ## calculate the target branch persistence
                tree = ts.first()
                poplabels_included = poplabels[poplabels.INCLUDE == 1].index.values
                for tid in tqdm(range(ts.num_trees)):
                    if (
                        tree.interval[1] // args.force_build - tree.interval[0] // args.force_build
                        > 0
                    ):
                        if mask_dodgy[sample_no][count_all_tree2]:
                            number_window_list = [] #List().empty_list(nb.types.float64)
                            parent = copy.deepcopy(sample)
                            while parent != tree.root:
                                edge_id = tree.edge(parent)
                                edge = ts_edges[edge_id]
                                parent = tree.parent(parent)
                                if True:
                                    tree_childrens = tree.children(parent)
                                    tree_leaves_left = list(tree.leaves(tree_childrens[0]))
                                    tree_leaves_right = list(tree.leaves(tree_childrens[1]))
                                    if (
                                        np.intersect1d(
                                            tree_leaves_left,
                                            poplabels_included,
                                        ).size
                                        - np.intersect1d(
                                            tree_leaves_left,
                                            leave_one_sample_out,
                                        ).size
                                        > 0
                                    ) and (
                                        np.intersect1d(
                                            tree_leaves_right,
                                            poplabels_included,
                                        ).size
                                        - np.intersect1d(
                                            tree_leaves_right,
                                            leave_one_sample_out,
                                        ).size
                                        > 0
                                    ):
                                        if args.hmm:
                                            edge_right = max(
                                                float(edge.metadata.decode().split(" ")[1])
                                                // args.force_build,
                                                tree.interval[1] // args.force_build,
                                            )
                                            edge_left = min(
                                                float(edge.metadata.decode().split(" ")[0])
                                                // args.force_build,
                                                tree.interval[0] // args.force_build,
                                            )
                                            number_of_overlaps = np.sum(
                                                (edge_right > bp_grid)
                                                & (edge_left <= bp_grid)
                                            )
                                            number_window_list.append(
                                                np.float64(2.0 * number_of_overlaps)
                                            )
                                        else:
                                            number_window_list.append(np.float64(1.0))

                            target_branch_length_sample_chr.append(number_window_list)
                            
                        count_all_tree2 += 1
                    tree.next()

                target_branch_length_sample_chr = np.repeat(target_branch_length_sample_chr, num_sites_per_tree, axis=0)
                f_pkl = open(branch_persistence_file_name, "wb")
                pickle.dump([args.force_build, args.start_time, args.end_time, args.ignore_first_epoch, args.ignore_last_epoch, args.masking_threshold, poplabels.values, target_branch_length_sample_chr], f_pkl)
                f_pkl.close()                
                ##convert to numba list
                for i in target_branch_length_sample_chr:
                    numba_i = List().empty_list(nb.types.float64)
                    for j in i:
                        numba_i.append(j)
                    target_branch_length_sample.append(numba_i)

        target_branch_length.append(target_branch_length_sample)

    return target_branch_length  ## num_samples x num_trees x num_branches


if __name__ == "__main__":
    load_gamma(
        "../real_apr23/1000G_sub_aDNA_Mar2023/result/1000G_sub_Nea_pp_v2_chr1.coal",
        ["Vindija", "CHB"],
        ["Vindija", "CHB", "YRI"],
    )
