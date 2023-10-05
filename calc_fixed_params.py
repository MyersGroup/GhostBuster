"""
calc_fixed_params.py: contains code to calculate numerator and denominator for each tree & 
helper functions for the EM
Includes: fixed_parameters(), compute_gamma_num(), compute_gamma_denom(), update_membership()
"""

import numpy as np
import copy
from tqdm import tqdm
import pickle
import math
from calc_ground_truth import make_ground_truth
from utils import get_epoch_interval, make_numba_nested_list
import random


def subsample_poplabels(poplabels, sample_list, max_per_group):
    """
    Takes a poplabels file and returns a poplabels file with few samples included
    """
    poplabels = copy.deepcopy(poplabels)
    for s in sample_list:
        poplabels.loc[s, "INCLUDE"] = 0
    for group in np.unique(poplabels[poplabels.INCLUDE == 1].GROUP):
        idx = poplabels[
            (poplabels.INCLUDE == 1) & (poplabels.GROUP == group)
        ].index.tolist()
        if max_per_group >= 1:
            keep_idx = random.sample(idx, min(len(idx), max_per_group))
        else:
            keep_idx = idx
        for i in list(set(idx) - set(keep_idx)):
            poplabels.loc[i, "INCLUDE"] = 0
    for s in sample_list:
        poplabels.loc[s, "INCLUDE"] = 1
    return poplabels


def fixed_parameters(
    ts_list,
    poplabels_orig,
    unique_groups,
    num_trees,
    mask_dodgy,
    sample,
    sample_list,
    epoch_intervals_pow,
    force_build=1,
    num_subtrees=1,
    max_per_group=-1,
    gt_ref=None,
    ignore_first_epoch=False,
):
    assert [
        s in poplabels_orig[poplabels_orig.INCLUDE == 1].index.tolist()
        for s in sample_list
    ] == [True] * len(sample_list)
    ### You have removed target samples from the poplabels file
    eps = 1e-20
    num_samples = len(list(ts_list[0].first().samples()))
    num_nodes = len(list(ts_list[0].first().nodes()))
    coal_count = np.zeros(
        (
            len(unique_groups),
            len(epoch_intervals_pow) - 1,
            num_trees,
        ),
        dtype="float64",
    )
    opportunity = np.zeros(
        (
            len(unique_groups),
            len(epoch_intervals_pow) - 1,
            num_trees,
        ),
        dtype="float64",
    )
    proportion_of_coalescing_all = []
    epoch_index_all = []
    denom_all = []
    count_mut_trees = -1
    count_all_tree = 0
    group_id = {}
    for u in range(len(np.unique(poplabels_orig[poplabels_orig.INCLUDE == 1].GROUP))):
        group_id[np.unique(poplabels_orig[poplabels_orig.INCLUDE == 1].GROUP)[u]] = u

    lineage_content_init = np.zeros(
        (2 * num_samples - 1, len(unique_groups)),
        dtype="float32",
    )
    for m in range(len(poplabels_orig)):
        ## Only count lineage content for included samples
        if poplabels_orig.INCLUDE.iloc[m]:
            lineage_content_init[m, group_id[poplabels_orig.GROUP.iloc[m]]] = 1.0
    for t in sample_list:
        lineage_content_init[
            t
        ] = 0.0  ## setting lineage content of target sequences = 0

    for sample_no, target_seq_ in enumerate([sample]):
        count_mut_trees_prev = copy.deepcopy(count_mut_trees)
        target_sampling_time = poplabels_orig.SAMPLING_TIME.iloc[target_seq_]
        for chr_no, ts in enumerate(ts_list):
            tree = ts.first()
            for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
                if (
                    tree.interval[1] // force_build - tree.interval[0] // force_build
                    > 0
                ):
                    if mask_dodgy[count_all_tree]:
                        sample_list_tree = copy.deepcopy(sample_list)
                        ## Make the coalescene table and sort it
                        coal_events_matrix = []
                        mapping = {}
                        count = num_samples
                        for s in tree.nodes():
                            if s < num_samples:
                                mapping[s] = s
                            else:
                                mapping[s] = count
                                count += 1
                        for s in tree.nodes():
                            if tree.children(s) != ():
                                a = tree.children(s)[0]
                                b = tree.children(s)[1]
                                c = s
                                t = tree.time(c)
                                coal_events_matrix.append(
                                    [
                                        int(mapping[a]),
                                        int(mapping[b]),
                                        int(mapping[c]),
                                        t,
                                    ]
                                )
                        coal_events_matrix = np.array(
                            coal_events_matrix, dtype="float64"
                        )
                        coal_events_matrix = coal_events_matrix[
                            coal_events_matrix[:, 3].argsort()
                        ]  ## sorting based on coalescene times
                        target_seq = target_seq_
                        for _ in range(num_subtrees):
                            count_mut_trees += 1
                            if max_per_group == -1:
                                poplabels = poplabels_orig.copy()
                            else:
                                poplabels = subsample_poplabels(
                                    poplabels_orig, sample_list, max_per_group
                                )
                            if gt_ref is None:
                                lineage_content = lineage_content_init.copy()
                            else:
                                lineage_content = np.zeros(
                                    (2 * num_samples - 1, len(unique_groups)),
                                    dtype="float64",
                                )
                                for m in range(len(poplabels)):
                                    ## Only count lineage content for included samples
                                    if poplabels.INCLUDE.iloc[m]:
                                        if (
                                            type(
                                                gt_ref[
                                                    m, count_mut_trees // num_subtrees
                                                ]
                                            )
                                            == dict
                                        ):
                                            for k, v in gt_ref[
                                                m, count_mut_trees // num_subtrees
                                            ].items():
                                                lineage_content[
                                                    m, int(unique_groups[k])
                                                ] = v
                                        else:
                                            lineage_content[
                                                m,
                                                int(
                                                    gt_ref[
                                                        m,
                                                        count_mut_trees // num_subtrees,
                                                    ]
                                                ),
                                            ] = 1

                                for t in sample_list_tree:
                                    lineage_content[
                                        t
                                    ] = 0  ## setting lineage content of target sequences = 0
                            prev_branch_length = np.sum(
                                lineage_content, axis=0
                            )  # np.sum(lineage_content[:,1])
                            lineage_content_sum = np.sum(lineage_content, axis=1)
                            proportion_of_coalescing_in_tree = []
                            coalescene_times_in_tree = []
                            epoch_index_in_tree = []
                            denom_in_tree = []
                            event_count = 0
                            for epoch in range(len(epoch_intervals_pow) - 1):
                                coal_events_submatrix = coal_events_matrix[
                                    (
                                        coal_events_matrix[:, 3]
                                        >= epoch_intervals_pow[epoch]
                                    )
                                    & (
                                        coal_events_matrix[:, 3]
                                        < epoch_intervals_pow[epoch + 1]
                                    )
                                ]
                                tprev = max(
                                    epoch_intervals_pow[epoch],
                                    target_sampling_time,
                                )  ##only considering coalescene events after the sampling time of the target
                                if epoch == 0 and ignore_first_epoch:
                                    for i in range(len(lineage_content)):
                                        if lineage_content_sum[i] > 0:
                                            lineage_content[i] /= np.sum(
                                                lineage_content[i]
                                            )
                                            lineage_content_sum[i] = 1
                                for (a, b, c, t) in coal_events_submatrix:
                                    event_count += 1
                                    a = int(a)
                                    b = int(b)
                                    c = int(c)
                                    opportunity[:, epoch, count_mut_trees] += (
                                        max(t, target_sampling_time) - tprev
                                    ) * (
                                        prev_branch_length
                                    )  ##only considering coalescene events after the sampling time of the target
                                    # print("pre: " + str(time.time() - st2))
                                    if (
                                        a in sample_list_tree and b in sample_list_tree
                                    ):  ## sometimes the target sequences coalesces with each other, in that case we append the coalesced node to the sample's list for that tree
                                        sample_list_tree.append(c)

                                    if (a == target_seq and b in sample_list_tree) or (
                                        b == target_seq and a in sample_list_tree
                                    ):  ## in case the target sequences coalesces with other target sequence, we don't count that coalescene count and opportunity
                                        target_seq = c
                                        lineage_content[c] = 0
                                        lineage_content_sum[c] = 0

                                    elif (
                                        a == target_seq and lineage_content_sum[b] == 0
                                    ) or (
                                        b == target_seq and lineage_content_sum[a] == 0
                                    ):
                                        ## This happens when target coalesces with a sample not included, in that case don't count that event
                                        target_seq = c
                                        lineage_content[c] = 0
                                        lineage_content_sum[c] = 0

                                    elif a == target_seq:
                                        proportion_of_coalescing = copy.deepcopy(
                                            lineage_content[b]
                                        )  # / (sum(lineage_content[b]))
                                        coal_count[
                                            :, epoch, count_mut_trees
                                        ] += proportion_of_coalescing
                                        target_seq = c
                                        lineage_content[c] = 0
                                        lineage_content_sum[c] = 0
                                        proportion_of_coalescing_in_tree.append(
                                            proportion_of_coalescing
                                        )
                                        epoch_index_in_tree.append(epoch)
                                        denom_in_tree.append(
                                            copy.deepcopy(
                                                opportunity[:, :, count_mut_trees]
                                            )
                                        )
                                        opportunity[:, :, count_mut_trees] = 0
                                        prev_branch_length = (
                                            prev_branch_length
                                            - lineage_content[b]
                                            / (lineage_content_sum[b])
                                        )
                                    elif b == target_seq:
                                        proportion_of_coalescing = copy.deepcopy(
                                            lineage_content[a]
                                        )  # / (sum(lineage_content[a]))
                                        ## sum() faster than np.sum()
                                        coal_count[
                                            :, epoch, count_mut_trees
                                        ] += proportion_of_coalescing
                                        target_seq = c
                                        lineage_content[c] = 0
                                        lineage_content_sum[c] = 0
                                        proportion_of_coalescing_in_tree.append(
                                            proportion_of_coalescing
                                        )
                                        epoch_index_in_tree.append(epoch)
                                        denom_in_tree.append(
                                            copy.deepcopy(
                                                opportunity[:, :, count_mut_trees]
                                            )
                                        )
                                        opportunity[:, :, count_mut_trees] = 0
                                        prev_branch_length = (
                                            prev_branch_length
                                            - lineage_content[a]
                                            / (lineage_content_sum[a])
                                        )

                                    else:  ## we don't count the branch lengths for the samples in sample_list_tree because they are the target sequences
                                        lineage_content[c] = (
                                            lineage_content[a] + lineage_content[b]
                                        )
                                        lineage_content_sum[c] = (
                                            lineage_content_sum[a]
                                            + lineage_content_sum[b]
                                        )
                                        if (
                                            lineage_content_sum[a] == 0
                                            or lineage_content_sum[b] == 0
                                        ):
                                            ### If a coalescene event involving atleast 1 non-included sequence, we ignore that event
                                            pass

                                        elif (
                                            a in sample_list_tree
                                            and b not in sample_list_tree
                                        ):
                                            prev_branch_length = (
                                                prev_branch_length
                                                - lineage_content[b]
                                                / (lineage_content_sum[b])
                                                + lineage_content[c]
                                                / (lineage_content_sum[c])
                                            )
                                        elif (
                                            b in sample_list_tree
                                            and a not in sample_list_tree
                                        ):
                                            prev_branch_length = (
                                                prev_branch_length
                                                - lineage_content[a]
                                                / (lineage_content_sum[a])
                                                + lineage_content[c]
                                                / (lineage_content_sum[c])
                                            )
                                        elif (
                                            a not in sample_list_tree
                                            and b not in sample_list_tree
                                        ):
                                            prev_branch_length = (
                                                prev_branch_length
                                                - lineage_content[a]
                                                / (lineage_content_sum[a])
                                                - lineage_content[b]
                                                / (lineage_content_sum[b])
                                                + lineage_content[c]
                                                / (lineage_content_sum[c])
                                            )
                                    lineage_content[a] = 0
                                    lineage_content[b] = 0
                                    lineage_content_sum[a] = 0
                                    lineage_content_sum[b] = 0
                                    tprev = max(t, target_sampling_time)
                                if epoch < len(epoch_intervals_pow) - 2:
                                    opportunity[:, epoch, count_mut_trees] += (
                                        max(
                                            epoch_intervals_pow[epoch + 1],
                                            target_sampling_time,
                                        )
                                        - max(
                                            tprev,
                                            target_sampling_time,
                                        )
                                    ) * (prev_branch_length)
                                if (event_count == num_samples - 1) and epoch <= len(
                                    epoch_intervals_pow
                                ) - 2:
                                    opportunity[:, epoch + 1 :, count_mut_trees] = 0.0
                                    break
                            proportion_of_coalescing_all.append(
                                proportion_of_coalescing_in_tree
                            )
                            epoch_index_all.append(epoch_index_in_tree)
                            denom_all.append(denom_in_tree)
                            sample_list_tree = copy.deepcopy(sample_list)

                    count_all_tree += num_subtrees
                tree.next()

        ## correcting opportunity for ancestral reference samples
        for m in range(len(poplabels)):
            if (not m in sample_list) and (poplabels.INCLUDE.iloc[m]):
                for epoch in range(len(epoch_intervals_pow) - 1):
                    if (
                        epoch_intervals_pow[epoch + 1]
                        < poplabels.SAMPLING_TIME.iloc[target_seq_]
                    ):
                        continue
                    if poplabels.SAMPLING_TIME.iloc[m] > epoch_intervals_pow[epoch]:
                        for tid, denom_tree in enumerate(denom_all):
                            for denom_coal in denom_tree:
                                if (
                                    denom_coal[
                                        group_id[poplabels.GROUP.iloc[m]],
                                        epoch,
                                    ]
                                    > min(
                                        poplabels.SAMPLING_TIME.iloc[m],
                                        epoch_intervals_pow[epoch + 1],
                                    )
                                    - epoch_intervals_pow[epoch]
                                ):
                                    denom_coal[
                                        group_id[poplabels.GROUP.iloc[m]],
                                        epoch,
                                    ] -= (
                                        min(
                                            poplabels.SAMPLING_TIME.iloc[m],
                                            epoch_intervals_pow[epoch + 1],
                                        )
                                        - epoch_intervals_pow[epoch]
                                    )
                                else:
                                    denom_coal[
                                        group_id[poplabels.GROUP.iloc[m]],
                                        epoch,
                                    ] = 0

    return (
        coal_count,
        denom_all,
        proportion_of_coalescing_all,
        epoch_index_all,
    )


def load_fixed_params(args, ts_list, sample, poplabels, mask_dodgy, chr_map, epoch_intervals, fixed_params_file_prefix=None):
    chrs = list(map(int, args.chrs.split(",")))
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)
    epoch_intervals_pow = np.power(10, epoch_intervals)

    denom_all = []
    proportion_of_coalescing_all = []
    epoch_index_all = []

    for chr_no, chr in enumerate(chrs):
        if fixed_params_file_prefix is not None:
            fixed_params_file_name = fixed_params_file_prefix + "_chr" + str(chr) + "_sample" + str(sample) + ".pkl"
        else:
            fixed_params_file_name = args.output + "_fixed_params_chr" + str(chr) + "_sample" + str(sample) + ".pkl"
        try:
            f_pkl = open(fixed_params_file_name, "rb")
            (num_subtrees, max_per_group, force_build, start_time, end_time, ignore_first_epoch, ignore_last_epoch, masking_threshold, poplabels_file, coal_count, denom, proportion_of_coalescing, epoch_index) = pickle.load(f_pkl)
            f_pkl.close()
            if (num_subtrees == args.num_subtrees) & (max_per_group == args.max_per_group) & (force_build == args.force_build) & (start_time == args.start_time) & (end_time == args.end_time) & (ignore_first_epoch == args.ignore_first_epoch) & (ignore_last_epoch == args.ignore_last_epoch) & (masking_threshold==args.masking_threshold) & np.all(poplabels_file == poplabels.values):
                ##convert to numba list
                denom_all.extend(denom)
                proportion_of_coalescing_all.extend(proportion_of_coalescing)
                epoch_index_all.extend(epoch_index)
                print("Loaded fixed parameters from: " + str(fixed_params_file_name))
                continue

        except:
            print("Fixed parameters file not found, calculating fixed parameters..")
            mask_dodgy_sam_chr = mask_dodgy[np.array(chr_map) == chr]
            num_trees = np.sum(mask_dodgy_sam_chr)
            (coal_count, denom, proportion_of_coalescing, epoch_index) = fixed_parameters(
                ts_list[chr_no:chr_no + 1],
                poplabels,
                unique_groups,
                num_trees,
                mask_dodgy_sam_chr,
                sample,
                args.sample_id,
                epoch_intervals_pow,
                args.force_build,
                args.num_subtrees,
                args.max_per_group,
                gt_ref=None,
                ignore_first_epoch=args.ignore_first_epoch,
            )
            f_pkl = open(fixed_params_file_name, "wb")
            pickle.dump([args.num_subtrees, args.max_per_group, args.force_build, args.start_time, args.end_time, args.ignore_first_epoch, args.ignore_last_epoch, args.masking_threshold, poplabels.values, coal_count, denom, proportion_of_coalescing, epoch_index], f_pkl)
            f_pkl.close()
            denom_all.extend(denom)
            proportion_of_coalescing_all.extend(proportion_of_coalescing)
            epoch_index_all.extend(epoch_index)
            print("Fixed parameters stored in: " + str(fixed_params_file_name))

    
    denom_all = make_numba_nested_list(denom_all)
    proportion_of_coalescing_all = make_numba_nested_list(proportion_of_coalescing_all)
    epoch_index_all = make_numba_nested_list(epoch_index_all)


    return (
        coal_count,
        denom_all,
        proportion_of_coalescing_all,
        epoch_index_all,
    )
