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

from calc_ground_truth import make_ground_truth


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_one_hot(X, max_X=None):
    X = np.array(X, dtype="int")
    classes = np.arange(0, max_X, 1) if max_X is not None else np.unique(X)
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
    mask = recomb_rates <= np.percentile(recomb_rates, (masking_thresh) * 100)
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
    args,
    ts_list,
    tree_left_bp,
    recomb_rates,
    frac_branches,
    num_snps_on_lineage,
    fine_map=False,
):
    chrs = list(map(int, args.chrs.split(",")))
    num_trees = int(np.sum([ts.num_trees for ts in ts_list]))
    chr_list = []
    for c in range(len(ts_list)):
        num_of_trees_in_chr = [c + 1] * ts_list[c].num_trees
        chr_list.extend(num_of_trees_in_chr)

    mask_dodgy = ~np.isnan(recomb_rates)
    recomb_0_thresh = np.sum(np.array(recomb_rates) <= 0) / len(recomb_rates)
    mask_dodgy *= ~mask_for_dodgy_trees(
        recomb_rates,
        recomb_0_thresh,
    )

    if fine_map is False:
        mask_dodgy *= mask_for_dodgy_trees(
            recomb_rates,
            1 - args.masking_threshold,
        )
    else:
        print("Fine-mapping to get near independent trees")
        coverage = copy.deepcopy(np.array(mask_dodgy, dtype="float"))
        coverage[coverage == 0] = np.inf
        mask_dodgy = copy.deepcopy(np.zeros_like(coverage, dtype="bool"))
        while np.sum(coverage != np.inf) > 0 and np.sum(mask_dodgy) <= (
            len(mask_dodgy) * (1 - args.masking_threshold)
        ):
            ## choose the best tree
            best_id = np.argmin(np.nan_to_num(recomb_rates * coverage, nan=np.inf))
            print(np.sum(mask_dodgy))
            mask_dodgy[best_id] = True

            ## mask (remove) the near-ones
            recomb_rate = recomb_rates[best_id]
            window = 0.001 / recomb_rate  ## 0.01 cM around the tree
            for i in range(max(0, best_id - 10), min(best_id + 10, len(tree_left_bp))):
                if (
                    tree_left_bp[i] < tree_left_bp[best_id] + window
                    and tree_left_bp[i] > tree_left_bp[best_id] - window
                ):
                    coverage[i] = np.inf

    ### Added frac_branch_target to filter trees aswell
    # mask_dodgy *= ~mask_for_dodgy_trees(
    #    frac_branches,
    #    args.masking_threshold,
    # )
    mask_dodgy = np.array(mask_dodgy)
    # mask_dodgy = np.tile(mask_dodgy, len(args.sample_id))
    # if args.mode == "sim":
    #   #### Caution: manually downsampling HAN (1) !! 🌵
    #   print("Downsampling !! Caution !!")
    #   ground_truth_membership = make_ground_truth(
    #       ts_list,
    #       np.sum(mask_dodgy),
    #       mask_dodgy=mask_dodgy,
    #       path=args.ground_truth_path,
    #       sample=args.sample_id,
    #       chrs=chrs,
    #       force_build=args.force_build,
    #   )
    #   mask_dodgy[mask_dodgy] *= downsample_trees(ground_truth_membership, 1, 0.5)

    print(
        "Filtering based on recombination rate, trees remaining: "
        + str(sum(mask_dodgy))
        + " average recomb. rate: "
        + str(np.mean(np.array(recomb_rates)[mask_dodgy]))
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


def load_mask_csv(args, sample_id_list, ts_list, mask_dodgy, chrs):
    membership_mask = pd.read_csv(args.load_mask, sep="\s+")
    count = 0
    for sample_no in range(len(sample_id_list)):
        for chr_count, chr in enumerate(chrs):
            tree = ts_list[chr_count].first()
            membership_mask_chr = membership_mask[membership_mask.chr == chr]
            for tid in range(ts_list[chr_count].num_trees):
                if (
                    tree.interval[1] // args.force_build
                    - tree.interval[0] // args.force_build
                    > 0
                ):
                    if (
                        tree.interval[0] // args.force_build
                        in membership_mask_chr["pos"].values.tolist()
                    ):
                        mask_dodgy[count] = True
                    else:
                        mask_dodgy[count] = False
                    count += 1
                tree.next()

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


def compute_gamma_num(
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    num_ref_groups,
    masked_trees_index,
    n_epochs,
):
    num_full_tree = np.zeros((num_ref_groups, n_epochs - 1), dtype="float64")
    count_masked_trees = 0
    if not (isinstance(prev_gamma, np.ndarray)):
        for tid in masked_trees_index:
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                epoch = epoch_index_in_tree[i]
                num = proportion_of_coalescing_in_tree[i]
                num = num / sum(num)
                num_full_tree[:, epoch] += own_membership[count_masked_trees] * num
            count_masked_trees += 1
    else:
        for tid in masked_trees_index:
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                epoch = epoch_index_in_tree[i]
                prev_gamma_e = prev_gamma[:, epoch]
                num = prev_gamma_e * proportion_of_coalescing_in_tree[i]
                sum_of_num = sum(num)
                if (
                    sum_of_num != 0
                ):  ## sometimes the num are less than python float64 precision, we ignore those coal events while calculating
                    num = num / sum_of_num
                num_full_tree[:, epoch] += own_membership[count_masked_trees] * num
            count_masked_trees += 1
    return num_full_tree


def compute_gamma_denom(own_membership, denom, n_epochs):
    eps = 1e-200
    denom_1 = np.zeros(n_epochs - 1, dtype="float64")
    for epoch in range(n_epochs - 1):  #
        denom_1[epoch] = sum(denom[epoch] * own_membership)
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
