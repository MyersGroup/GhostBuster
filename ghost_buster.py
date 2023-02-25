import pandas as pd
import numpy as np
import math
import time
import tskit
import argparse
import copy
from tqdm import tqdm
import random
from functools import partial
from joblib import Parallel, delayed

from calc_tree_stats import load_tree_stats
from calc_fixed_params import fixed_parameters
from calc_ground_truth import get_groundtruth_reference, make_ground_truth
from utils import (
    make_one_hot,
    filter_recomb_rate,
    load_mask_csv,
    write_coal,
    write_calibration,
    calculate_accuracy,
    boolean,
    compute_gamma_num,
    compute_gamma_denom,
)
import pdb
import warnings

warnings.filterwarnings("ignore")


def input_assertions(args):
    if (args.load_gamma == None and args.load_props != None) or (
        args.load_props != None and args.load_gamma == None
    ):
        raise ValueError(
            "Stop specifying gammas without proportions or proportions without gamma"
        )
    if not args.evaluate_local_ancestry and not args.evaluate_gamma:
        raise ValueError(
            "If you don't want to evaluate the population sizes or local ancestry, what are you here for ?"
        )
    if args.init_at_truth and args.mode == "real":
        raise ValueError(
            "How can you initialize your local ancestry in real-world data ?"
        )
    if args.props_per_chrs and args.load_props:
        raise ValueError(
            "Propotions per chromosomes not supported when you load proportions"
        )
    if args.mode == "sim" and args.ground_truth_path is None:
        raise ValueError(
            "Supply the location of ground_truth file or run in mode = real"
        )
    if args.opportunity_filter and (args.mutden is None or args.allmuts is None):
        raise ValueError(
            "Supply the location for mutden and allmuts file to filter trees based on opportunity"
        )


def load_trees(args, poplabels):
    chrs = list(map(int, args.chrs.split(",")))
    ts_list = []
    if args.trees != None:
        for chr in chrs:
            ts = tskit.load(args.trees + str(chr) + ".trees")  ## relate trees
            ts_list.append(ts)

    return ts_list


def update_membership(
    proportion_of_coalescing_in_tree,
    epoch_index_in_tree,
    denom,
    gamma_arr,
    tid,
    ignore_first_epoch,
    ignore_last_epoch,
    n_epochs,
):
    log_num_em_j_i = 0
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
            log_num_em_j_i += np.log(
                sum(
                    gamma_arr[:, epoch_index_in_tree[i]]
                    * proportion_of_coalescing_in_tree[i]
                ),
            )

    if ignore_first_epoch and ignore_last_epoch:
        log_denom_em_j = -sum(sum(gamma_arr[:, 1:-1] * denom[:, 1:-1, tid]))
    elif ignore_first_epoch and not ignore_last_epoch:
        log_denom_em_j = -sum(sum(gamma_arr[:, 1:] * denom[:, 1:, tid]))
    elif ignore_last_epoch and not ignore_first_epoch:
        log_denom_em_j = -sum(sum(gamma_arr[:, :-1] * denom[:, :-1, tid]))
    else:
        log_denom_em_j = -sum(sum(gamma_arr * denom[:, :, tid]))
    return log_num_em_j_i, log_denom_em_j


def combine_local_ancestry(arr, n):
    ## combines the arr values every n elements
    ## Input: arr of shape c x N => output of shape c x N/n
    out = np.zeros((arr.shape[0], arr.shape[1] // n))
    for i in range(n):
        out += arr[:, i::n]
    return out / n


def e_m_step(
    args,
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    n_unique_groups,
    n_epochs,
    n_trees,
    n_samples,
    epoch,
):

    masked_trees_index = np.arange(0, n_trees)
    n = np.zeros(
        (args.num_clusters, n_unique_groups, n_epochs - 1),
        dtype="float64",
    )
    d = np.zeros(
        (args.num_clusters, n_unique_groups, n_epochs - 1),
        dtype="float64",
    )
    tau = np.zeros(args.num_clusters, dtype="float64")
    for sample_no in range(n_samples):
        own_membership_sample = own_membership[
            :, sample_no * n_trees : (sample_no + 1) * n_trees
        ]
        for j in range(len(own_membership_sample)):
            if epoch == 0:
                n[j] += compute_gamma_num(
                    own_membership_sample[j],
                    None,
                    proportion_of_coalescing_all[sample_no],
                    epoch_index_all[sample_no],
                    n_unique_groups,
                    masked_trees_index,
                    n_epochs,
                )
            else:
                n[j] += compute_gamma_num(
                    own_membership_sample[j],
                    prev_gamma[j],
                    proportion_of_coalescing_all[sample_no],
                    epoch_index_all[sample_no],
                    n_unique_groups,
                    masked_trees_index,
                    n_epochs,
                )
            for i in range(n_unique_groups):
                d[j, i] += compute_gamma_denom(
                    own_membership_sample[j], denom[sample_no][i], n_epochs
                )

        tau += np.mean(own_membership_sample, axis=1) / n_samples

    gamma_arr = n / d
    if epoch == 0 and args.load_gamma != None and args.load_props != None:
        print("Using initial gamma specified in file: " + str(args.load_gamma))
        gamma_arr = np.load(args.load_gamma)
        tau = np.load(args.load_props)  ### load taus only works for not(props_per_chrs)

    if tau[0] < tau[1]:
        tau = [0.05, 0.95]  ## CAUTION: Fixing tau!!!
    else:
        tau = [0.95, 0.05]
    tau = np.array(tau)

    gamma_arr = np.maximum(gamma_arr, 0)
    prev_gamma = copy.deepcopy(gamma_arr)

    log_num_em = np.zeros((args.num_clusters, n_trees * n_samples), dtype="float64")
    log_denom_em = np.zeros((args.num_clusters, n_trees * n_samples), dtype="float64")
    count_masked_trees = 0

    for sample_no in range(n_samples):
        for tid in masked_trees_index:
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[sample_no][
                tid
            ]
            epoch_index_in_tree = epoch_index_all[sample_no][tid]
            for j in range(args.num_clusters):
                log_num_em_j, log_denom_em_j = update_membership(
                    proportion_of_coalescing_in_tree,
                    epoch_index_in_tree,
                    denom[sample_no],
                    gamma_arr[j],
                    tid,
                    args.ignore_first_epoch,
                    args.ignore_last_epoch,
                    n_epochs,
                )
                log_num_em[j, count_masked_trees] = log_num_em_j
                log_denom_em[j, count_masked_trees] = log_denom_em_j
            count_masked_trees += 1

    log_num_em = 1 * combine_local_ancestry(log_num_em, args.num_subtrees)
    log_denom_em = 1 * combine_local_ancestry(log_denom_em, args.num_subtrees)

    own_membership_update = np.exp(
        log_num_em
        + log_denom_em
        - np.repeat(
            np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
            len(own_membership),
            axis=1,
        ).T
    )
    own_membership_update = np.nan_to_num(own_membership_update, nan=1)
    for j in range(len(own_membership)):
        own_membership_update[j] *= tau[j]

    log_likelihood = np.sum(
        np.log(np.sum(own_membership_update, axis=0))
        + np.max(log_num_em + log_denom_em, axis=0)
    )
    own_membership = own_membership_update / (np.sum(own_membership_update, axis=0))

    own_membership = np.repeat(own_membership, args.num_subtrees, axis=1)
    return own_membership, gamma_arr, tau, log_likelihood


def estimate_gt_ref(
    gamma_arr,
    tau,
    args,
    ts_list,
    poplabels,
    n_trees,
    n_epochs,
    mask_dodgy,
    epoch_intervals_pow,
):
    ## random init gt_ref & unique_groups

    poplabels_included = poplabels[poplabels.INCLUDE == 1]
    unique_groups, i = {}, 0
    for group in np.unique(poplabels_included.GROUP):
        if poplabels_included.GROUP.iloc[args.sample_id[0]] == group:
            for c in range(args.num_clusters):
                unique_groups[group + str(c + 1)] = i
                i += 1
        else:
            unique_groups[group] = i
            i += 1

    gt_ref = np.zeros((len(poplabels.GROUP), n_trees), dtype="object")
    for sample_no, sample in enumerate(poplabels.index):
        group = poplabels.GROUP.loc[sample]
        if group == poplabels.GROUP.loc[args.sample_id[0]]:
            gt_ref[sample_no] = {
                poplabels.GROUP.iloc[sample] + str(c + 1): tau[c]
                for c in range(args.num_clusters)
            }
        else:
            try:
                gt_ref[sample_no] = unique_groups[group]
            except:
                gt_ref[sample_no] = "NA"

    chrs = list(map(int, args.chrs.split(",")))

    gt_ref = np.array(gt_ref, dtype="object")
    gt_ref_update = np.zeros_like(gt_ref, dtype="object")
    for outer_iter in range(1):
        r2 = 0.0
        for sample in tqdm(
            np.random.permutation(
                poplabels[
                    (poplabels.GROUP == poplabels.GROUP.iloc[args.sample_id[0]])
                    & poplabels.INCLUDE
                    == 1
                ].index
            )
        ):
            ## Calc fixed params
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                n_trees,
                mask_dodgy,
                [sample],
                epoch_intervals_pow,
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=gt_ref,
            )
            ## E-step to infer local ancestry
            own_membership_trial = np.ones(
                (args.num_clusters, n_trees), dtype="float64"
            )
            log_num_em = np.zeros((args.num_clusters, n_trees), dtype="float64")
            log_denom_em = np.zeros((args.num_clusters, n_trees), dtype="float64")
            count_masked_trees = 0
            for tid in np.arange(0, n_trees):
                proportion_of_coalescing_in_tree = proportion_of_coalescing_all1[tid]
                epoch_index_in_tree = epoch_index_all1[tid]
                for j in range(args.num_clusters):
                    log_num_em_j, log_denom_em_j = update_membership(
                        proportion_of_coalescing_in_tree,
                        epoch_index_in_tree,
                        denom1,
                        gamma_arr[j],
                        tid,
                        args.ignore_first_epoch,
                        args.ignore_last_epoch,
                        n_epochs,
                    )
                    log_num_em[j, count_masked_trees] = log_num_em_j
                    log_denom_em[j, count_masked_trees] = log_denom_em_j
                count_masked_trees += 1
            own_membership_trial = np.exp(
                log_num_em
                + log_denom_em
                - np.repeat(
                    np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
                    args.num_clusters,
                    axis=1,
                ).T
            )
            own_membership_trial = np.nan_to_num(own_membership_trial, nan=1)
            for j in range(args.num_clusters):
                own_membership_trial[j] *= tau[j]

            own_membership_trial = own_membership_trial / (
                np.sum(own_membership_trial, axis=0)
            )

            # Sample from the posteriors
            for n_t in range(n_trees):
                gt_ref_update[sample, n_t] = {
                    poplabels.GROUP.iloc[sample]
                    + str(c + 1): own_membership_trial[c, n_t]
                    for c in range(args.num_clusters)
                }
        for sample in tqdm(
            np.random.permutation(
                poplabels[
                    (poplabels.GROUP == poplabels.GROUP.loc[args.sample_id[0]])
                ].index
            )
        ):
            gt_ref[sample] = copy.deepcopy(gt_ref_update[sample])

    return gt_ref, unique_groups


def random_sweep_iter(
    args,
    n_clusters,
    n_unique_groups,
    n_epochs,
    n_trees,
    n_iters,
    ts_list,
    poplabels,
    mask_dodgy,
    epoch_intervals_pow,
    num,
    denom,
    proportion_of_coalescing_all,
    epoch_index_all,
):
    gamma_arr = np.power(
        np.e,
        np.random.uniform(-16.11, -5.3, (n_clusters, n_unique_groups, n_epochs - 1)),
    )
    tau = np.random.uniform(0.01, 0.99, n_clusters)
    for c in range(n_clusters):
        not_c = np.delete(np.arange(n_clusters), c)
        gamma_arr[c, c] = np.power(
            np.e, np.log(gamma_arr[c, c]) + np.log(np.max(gamma_arr[c, not_c], axis=0))
        )
    if args.load_gamma is not None and args.load_props is not None and n_iters == 0:
        gamma_arr = np.load(args.load_gamma)
        tau = np.load(args.load_props)

    masked_trees_index = np.arange(0, n_trees)
    n_samples = len(args.sample_id)
    if args.joint_fit:
        gt_ref, unique_groups = estimate_gt_ref(
            gamma_arr,
            tau,
            args,
            ts_list,
            poplabels,
            n_trees,
            n_epochs,
            mask_dodgy,
            epoch_intervals_pow,
        )

    else:
        gt_ref = None
        unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)
    own_membership_trial = np.ones((n_clusters, n_trees * n_samples), dtype="float64")
    log_num_em = np.zeros((n_clusters, n_trees * n_samples), dtype="float64")
    log_denom_em = np.zeros((n_clusters, n_trees * n_samples), dtype="float64")
    count_masked_trees = 0

    for sample_no, sample in enumerate(args.sample_id):
        if args.joint_fit:
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                n_trees,
                mask_dodgy,
                [sample],
                epoch_intervals_pow,
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=gt_ref,
            )
            num.append(num1)
            denom.append(denom1)
            proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
            epoch_index_all.append(epoch_index_all1)

        for tid in masked_trees_index:
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[sample_no][
                tid
            ]
            epoch_index_in_tree = epoch_index_all[sample_no][tid]
            for j in range(n_clusters):
                log_num_em_j, log_denom_em_j = update_membership(
                    proportion_of_coalescing_in_tree,
                    epoch_index_in_tree,
                    denom[sample_no],
                    gamma_arr[j],
                    tid,
                    args.ignore_first_epoch,
                    args.ignore_last_epoch,
                    n_epochs,
                )
                log_num_em[j, count_masked_trees] = log_num_em_j
                log_denom_em[j, count_masked_trees] = log_denom_em_j
            count_masked_trees += 1

    own_membership_trial = np.exp(
        log_num_em
        + log_denom_em
        - np.repeat(
            np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
            n_clusters,
            axis=1,
        ).T
    )
    own_membership_trial = np.nan_to_num(own_membership_trial, nan=1)
    for j in range(n_clusters):
        own_membership_trial[j] *= tau[j]

    own_membership_trial = own_membership_trial / (np.sum(own_membership_trial, axis=0))

    for epoch in range(args.sweep_num_iters):
        own_membership_trial, gamma_arr, tau, log_likelihood = e_m_step(
            args,
            own_membership_trial,
            gamma_arr,
            proportion_of_coalescing_all,
            epoch_index_all,
            denom,
            n_unique_groups,
            n_epochs,
            n_trees,
            n_samples,
            epoch,
        )
    return (
        log_likelihood,
        own_membership_trial,
        num,
        denom,
        proportion_of_coalescing_all,
        epoch_index_all,
        unique_groups,
    )


def random_sweep(
    args,
    n_clusters,
    n_unique_groups,
    n_epochs,
    n_trees,
    n_repeats,
    ts_list,
    poplabels,
    mask_dodgy,
    epoch_intervals,
):
    print("Performing a random sweep for better initialization")
    masked_trees_index = np.arange(0, n_trees)
    best_loglikelihood = -np.inf
    epoch_intervals_pow = np.power(10, epoch_intervals)

    if not args.joint_fit:
        num, denom, proportion_of_coalescing_all, epoch_index_all = [], [], [], []
        for sample_no, sample in enumerate(args.sample_id):
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                poplabels[poplabels.INCLUDE == 1].GROUP.unique(),
                n_trees,
                mask_dodgy,
                [sample],
                epoch_intervals_pow,
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=None,
            )
            num.append(num1)
            denom.append(denom1)
            proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
            epoch_index_all.append(epoch_index_all1)
    else:
        num, denom, proportion_of_coalescing_all, epoch_index_all = (
            [],
            [],
            [],
            [],
        )

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(random_sweep_iter)(
            args,
            n_clusters,
            n_unique_groups,
            n_epochs,
            n_trees,
            n_iters,
            ts_list,
            poplabels,
            mask_dodgy,
            epoch_intervals_pow,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
        )
        for n_iters in range(n_repeats)
    )
    for i in range(len(out)):
        (
            log_likelihood,
            own_membership_trial,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
            unique_groups,
        ) = out[i]
        if log_likelihood > best_loglikelihood or i == 0:
            best_loglikelihood = log_likelihood
            own_membership = own_membership_trial
            (
                best_num,
                best_denom,
                best_proportion_of_coalescing_all,
                best_epoch_index_all,
            ) = (num, denom, proportion_of_coalescing_all, epoch_index_all)
    return (
        own_membership,
        best_num,
        best_denom,
        best_proportion_of_coalescing_all,
        best_epoch_index_all,
        unique_groups,
    )


def write_membership_grid(
    own_membership,
    tree_left_bp,
    tree_right_bp,
    n_clusters,
    sample_id_label,
    output,
    window_size=1e3,
):
    assert len(own_membership[0]) == len(tree_left_bp)
    assert len(tree_left_bp) == len(tree_right_bp)
    res = []
    for i, (l, r) in enumerate(zip(tree_left_bp, tree_right_bp)):
        for j in range(int(l / window_size), int(r / window_size)):
            res.append([j * window_size] + list(own_membership[:, i]))
    pd.DataFrame(
        data=np.array(res),
        columns=["start"] + ["prob_" + str(i) for i in range(n_clusters)],
    ).to_csv(
        output + "_overall_membership_" + sample_id_label + ".csv",
        index=False,
        sep="\t",
    )


def write_membership_gamma(
    args,
    own_membership,
    gamma_arr,
    tau,
    mask_dodgy,
    chr_map,
    tree_left_bp,
    tree_right_bp,
    epoch_intervals,
    unique_groups,
    sample_id_label,
):
    ## gamma and membership plots
    filename = (
        "overall_membership_" + sample_id_label + ".npy"
    )  ## this saves membership for all the trees (without the filtering)
    filename = args.output + "_" + filename
    with open(filename, "wb") as f:
        np.save(f, own_membership)

    write_membership_grid(
        own_membership,
        tree_left_bp,
        tree_right_bp,
        args.num_clusters,
        sample_id_label,
        args.output,
        args.force_build,
    )

    write_coal(
        gamma_arr,
        sample_id_label + ".coal",
        unique_groups,
        args.output,
        epoch_intervals,
    )

    with open(
        args.output + "_gamma_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, gamma_arr)

    with open(
        args.output + "_props_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, tau)

    tree_position = []
    for tid in range(len(tree_left_bp) // len(args.sample_id)):
        tree_position.append(
            [np.array(chr_map)[mask_dodgy][tid], tree_left_bp[tid] // args.force_build]
        )
    filename = (
        "mask_" + sample_id_label + ".csv"
    )  ## this saves membership for all the trees (without the filtering)
    filename = args.output + "_" + filename
    pd.DataFrame(np.array(tree_position), columns=["chr", "pos"]).to_csv(
        filename, index=False, sep="\t"
    )
    np.save(args.output + "_" + "mask_" + sample_id_label + ".npy", mask_dodgy)


def main(args):
    ### Initialize some global variables
    epoch_intervals = np.array(
        [-np.inf]
        + np.linspace(
            args.start_time - math.log(28, 10),
            args.end_time - math.log(28, 10),
            args.num_epochs - 1,
        ).tolist()
        + [np.inf],
        dtype="float64",
    )

    sample_id = []
    for i in range(len(args.sample_id)):
        if "-" in args.sample_id[i]:
            sample_id.extend(
                np.arange(
                    int(args.sample_id[i].split("-")[0]),
                    int(args.sample_id[i].split("-")[1]),
                ).tolist()
            )
        else:
            sample_id.append(int(args.sample_id[i]))
    args.sample_id = sample_id

    sample_id_label = "_".join([str(e) for e in args.sample_id])
    poplabels = pd.read_csv(args.poplabels, sep="\s+")
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))

    ### Load all the trees in a list
    ts_list = load_trees(args, poplabels)
    if len(poplabels) == ts_list[0].num_samples // 2:
        poplabels = pd.DataFrame(
            np.repeat(poplabels.values, 2, axis=0), columns=poplabels.columns
        )
    if len(poplabels) != ts_list[0].num_samples:
        raise ValueError(
            "Number of samples in trees doesnt match number of samples in poplabels.txt"
        )

    num_samples = len(poplabels)
    for sample in args.sample_id:
        if sample >= num_samples or sample < 0:
            raise ValueError("The sample ids are out of range")
        else:
            print(str(sample) + " is: " + str(poplabels.GROUP.iloc[sample]))

    ### Load tree stats
    (
        recomb_rates,
        mutrate_opportunity_target,
        tree_left_bp,
        tree_right_bp,
        chr_map,
        frac_branches_with_snp_target,
        mutrate_logpmf_target,
        num_snps_on_lineage,
    ) = load_tree_stats(args, ts_list, poplabels)

    ### Filter based on recombination rates
    if args.load_mask is None:
        mask_dodgy = filter_recomb_rate(
            args,
            ts_list,
            tree_left_bp,
            recomb_rates,
            frac_branches_with_snp_target,
            num_snps_on_lineage,
        )

    if args.load_mask is not None:
        mask_dodgy = np.zeros(len(recomb_rates), dtype="bool")
        mask_dodgy = load_mask_csv(args, args.sample_id, ts_list, mask_dodgy, chrs)

    mask_dodgy = np.repeat(mask_dodgy, args.num_subtrees)

    ### Use ground-truth local ancestry for reference samples
    num_trees = np.sum(mask_dodgy)
    poplabels_included = poplabels[poplabels.INCLUDE == 1]
    tree_left_bp = np.array(
        np.array(tree_left_bp)[mask_dodgy].tolist() * len(args.sample_id)
    )
    tree_right_bp = np.array(
        np.array(tree_right_bp)[mask_dodgy].tolist() * len(args.sample_id)
    )
    ### Calculate ground truth local ancestry
    if args.mode == "sim":
        ground_truth_membership = []
        for sample in args.sample_id:
            ground_truth_membership_sample = make_ground_truth(
                ts_list,
                num_trees // args.num_subtrees,
                mask_dodgy=mask_dodgy[:: args.num_subtrees],
                path=args.ground_truth_path,
                sample=[sample],
                chrs=chrs,
                force_build=args.force_build,
            )
            ground_truth_membership_sample = np.repeat(
                ground_truth_membership_sample, args.num_subtrees, axis=1
            )
            ground_truth_membership.append(ground_truth_membership_sample)
        ground_truth_membership = (
            np.array(ground_truth_membership)
            .transpose(1, 0, 2)
            .reshape(ground_truth_membership_sample.shape[0], -1)
        )

    ### Initialize local ancestry
    if args.init_at_truth and args.joint_fit:
        unique_groups, i = {}, 0
        for group in np.unique(poplabels_included.GROUP):
            if poplabels_included.GROUP.iloc[args.sample_id[0]] == group:
                for c in range(args.num_clusters):
                    unique_groups[group + str(c + 1)] = i
                    i += 1
            else:
                unique_groups[group] = i
                i += 1

        gt_ref = np.zeros((len(poplabels.GROUP), num_trees), dtype="object")
        for sample_no, sample in enumerate(poplabels.index):
            group = poplabels.GROUP.loc[sample]
            if group == poplabels.GROUP.loc[args.sample_id[0]]:
                gt_ref[sample_no] = {
                    poplabels.GROUP.iloc[sample] + str(c + 1): 1
                    for c in range(args.num_clusters)
                }
            else:
                try:
                    gt_ref[sample_no] = unique_groups[group]
                except:
                    gt_ref[sample_no] = "NA"

        gt_ref = np.array(gt_ref, dtype="object")

        print("Calculating ground-truth ancestry of the reference...")
        gt_ref_orig, _ = get_groundtruth_reference(
            ts_list,
            poplabels_included,
            np.sum(mask_dodgy) // args.num_subtrees,
            mask_dodgy[:: args.num_subtrees],
            args.ground_truth_path,
            chrs,
            poplabels.GROUP.loc[args.sample_id[0]],
            args.force_build,
        )
        for sample_no, sample in enumerate(
            poplabels_included[
                poplabels_included.GROUP == poplabels.GROUP.loc[args.sample_id[0]]
            ].index
        ):
            for n_t in range(gt_ref.shape[1]):
                gt_ref[sample, n_t] = unique_groups[
                    poplabels.GROUP.iloc[args.sample_id[0]]
                    + str(gt_ref_orig[sample_no, n_t] + 1)
                ]

        n = np.zeros(
            (args.num_clusters, len(unique_groups), len(epoch_intervals) - 1),
            dtype="float64",
        )
        d = np.zeros(
            (args.num_clusters, len(unique_groups), len(epoch_intervals) - 1),
            dtype="float64",
        )
        tau = np.zeros(args.num_clusters, dtype="float64")
        for sample in args.sample_id:
            own_membership_sample = make_one_hot(gt_ref[sample])
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                num_trees,
                mask_dodgy,
                [sample],
                np.power(10, epoch_intervals),
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=gt_ref,
            )
            for j in range(len(own_membership_sample)):
                n[j] += compute_gamma_num(
                    own_membership_sample[j],
                    None,
                    proportion_of_coalescing_all1,
                    epoch_index_all1,
                    len(unique_groups),
                    np.arange(0, num_trees),
                    len(epoch_intervals),
                )
                for i in range(len(unique_groups)):
                    d[j, i] += compute_gamma_denom(
                        own_membership_sample[j], denom1[i], len(epoch_intervals)
                    )

            print(np.mean(own_membership_sample, axis=1))

            tau += np.mean(own_membership_sample, axis=1)

        tau /= len(args.sample_id)
        gamma_arr = n / d

        print(tau)
        print(gamma_arr)

        with open(
            args.output + "_gamma_all.npy",
            "wb",
        ) as f:
            np.save(f, gamma_arr)

        with open(
            args.output + "_props_all.npy",
            "wb",
        ) as f:
            np.save(f, tau)

        return

    elif args.init_at_truth and not args.joint_fit:
        own_membership = ground_truth_membership
        num, denom, proportion_of_coalescing_all, epoch_index_all = [], [], [], []
        for sample_no, sample in enumerate(args.sample_id):
            (
                num1,
                denom1,
                proportion_of_coalescing_all1,
                epoch_index_all1,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                num_trees,
                mask_dodgy,
                [sample],
                np.power(10, epoch_intervals),
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=None,
            )
            num.append(num1)
            denom.append(denom1)
            proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
            epoch_index_all.append(epoch_index_all1)
    else:
        (
            own_membership,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
            unique_groups,
        ) = random_sweep(
            args,
            args.num_clusters,
            len(np.unique(poplabels_included.GROUP)) + args.num_clusters - 1
            if args.joint_fit
            else len(unique_groups),
            len(epoch_intervals),
            num_trees,
            args.n_repeats,
            ts_list,
            poplabels,
            mask_dodgy,
            epoch_intervals,
        )

    if args.load_gamma:
        gamma_arr = np.load(args.load_gamma)
    else:
        gamma_arr = None
    if args.load_props:
        tau = np.load(args.load_props)

    ### EM
    if args.evaluate_gamma:
        filename = (
            "overall_membership_iter0_" + sample_id_label + ".npy"
        )  ## this saves membership for all the trees (without the filtering)
        filename = args.output + "_" + filename
        with open(filename, "wb") as f:
            np.save(f, own_membership)

        log_likelihood_arr = []
        start_time_em = time.time()
        print("Starting the EM..")

        filename_logl = args.output + "_" + sample_id_label + ".logl"
        filename_tau = args.output + "_" + sample_id_label + ".tau"
        f_logl = open(filename_logl, "w")
        f_tau = open(filename_tau, "w")

        for epoch in range(args.num_iters):
            own_membership, gamma_arr, tau, log_likelihood = e_m_step(
                args,
                own_membership,
                gamma_arr,
                proportion_of_coalescing_all,
                epoch_index_all,
                denom,
                len(unique_groups),
                len(epoch_intervals),
                num_trees,
                len(args.sample_id),
                epoch,
            )
            print(tau)

            if epoch == 0:
                write_coal(
                    gamma_arr,
                    sample_id_label + "_iter0.coal",
                    unique_groups,
                    args.output,
                    epoch_intervals,
                )
            for i in range(np.shape(tau)[0]):
                f_tau.write(str(tau[i]) + " ")
            f_tau.write("\n")
            log_likelihood_arr.append(log_likelihood)

            if args.mode == "sim":
                calculate_accuracy(own_membership, ground_truth_membership)

            ## Early-stopping
            print("log-likelihood = " + str(log_likelihood_arr[-1]), flush=True)
            f_logl.write(str(log_likelihood_arr[-1]) + "\n")

        if args.mode == "sim":
            write_calibration(args, own_membership, ground_truth_membership)
            filename = (
                "ground_truth_membership_" + sample_id_label + ".npy"
            )  ## this saves membership for all the trees (without the filtering)
            filename = args.output + "_" + filename
            with open(filename, "wb") as f:
                np.save(f, ground_truth_membership)

        write_membership_gamma(
            args,
            own_membership,
            gamma_arr,
            tau,
            mask_dodgy,
            chr_map,
            tree_left_bp,
            tree_right_bp,
            epoch_intervals,
            unique_groups,
            sample_id_label,
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-fb",
        "--force_build",
        help="force build size to subsample the trees in bp",
        type=float,
        default=10000,
    )
    parser.add_argument("-r", "--rec", help="Filename of rec maps.", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="Output prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "-chrs",
        "--chrs",
        help="Comma-seperated list of chromosomes to be considered",
        type=str,
        default="1,2",
    )
    parser.add_argument(
        "-masking_thresh",
        "--masking_threshold",
        help="Remove top x cent of high recombination regions",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-init_at_truth",
        "--init_at_truth",
        help="Do you wish to initialize at ground-truth",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "--n_repeats",
        help="Number of restarts when randomly initializing local ancestry, we choose best out of n_trails",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-load_gamma",
        "--load_gamma",
        help="Starting gamma values written in a file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-load_mask",
        "--load_mask",
        help="Load mask csv file with chr, tree_position_left//force_build",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-load_props",
        "--load_props",
        help="Starting taU values written in a file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-trees",
        "--trees",
        help="Location to trees in tskit format",
        type=str,
        default=None,
    )
    parser.add_argument("--poplabels", help="Location to poplabels file", type=str)
    parser.add_argument(
        "--mutden",
        help="Location prefix to mutation density file from Relate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--allmuts",
        help="Location prefix to allmuts file from Relate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ground_truth_path",
        help="Location prefix to groundtruth file generated from tskit in simulations",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-i",
        "--num_iters",
        help="Number of iterations for EM",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-k",
        "--num_clusters",
        help="Number of clusters to find using the EM",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-mode",
        "--mode",
        help="Which mode do you want to run in? Simulation or Real-world ?",
        type=str,
        default="real",
        choices=("sim", "real"),
    )
    parser.add_argument(
        "-evaluate_gamma",
        "--evaluate_gamma",
        help="Run EM to calculate the gammas (population sizes)",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "-start_time",
        "--start_time",
        help="Starting time for the population size plots, measured in log-scale",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "-end_time",
        "--end_time",
        help="Ending time for the population size plots, measured in log-scale",
        type=float,
        default=6,
    )
    parser.add_argument(
        "-num_epochs",
        "--num_epochs",
        help="Num epochs",
        type=int,
        default=9,
    )
    parser.add_argument(
        "-ignore_first_epoch",
        "--ignore_first_epoch",
        help="Ignore first epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "-ignore_last_epoch",
        "--ignore_last_epoch",
        help="Ignore last epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "--opportunity_filter",
        help="check mutations on target lineage: criterion to choose trees",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "--num_subtrees",
        help="Number of subtrees in the composite likelihood",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_per_group",
        help="Maximum number of individuals in a subtrees for the composite likelihood",
        type=int,
        default=-1,
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--joint_fit",
        action="store_const",
        const=True,
        default=False,
        help="Joint fitting of multiple samples",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of threads")
    parser.add_argument(
        "--sweep_num_iters",
        type=int,
        default=10,
        help="Number of iterations to run EM for in random sweep",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)  ## fix the random seed
    random.seed(args.seed)
    main(args)
