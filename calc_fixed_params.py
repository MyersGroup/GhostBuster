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
from utils import get_epoch_interval
import random

def subsample_poplabels(poplabels, sample_list, max_per_group):
    '''
    Takes a poplabels file and returns a poplabels file with few samples included
    '''
    poplabels = copy.deepcopy(poplabels)
    for s in sample_list:
        poplabels.loc[s,'INCLUDE'] = 0
    for group in np.unique(poplabels[poplabels.INCLUDE == 1].GROUP):
        idx = poplabels[(poplabels.INCLUDE == 1) & (poplabels.GROUP == group)].index.tolist()
        if max_per_group >= 1:
            keep_idx = random.sample(idx, min(len(idx), max_per_group))
        else:
            keep_idx = idx
        for i in list(set(idx) - set(keep_idx)):
            poplabels.loc[i,'INCLUDE'] = 0
    for s in sample_list:
        poplabels.loc[s,'INCLUDE'] = 1
    return poplabels

def fixed_parameters(
    ts_list,
    poplabels_orig,
    unique_groups,
    num_trees,
    mask_dodgy,
    sample_list,
    epoch_intervals_pow,
    force_build=1,
    num_subtrees=1,
    max_per_group=-1
):
    assert [
        s in poplabels_orig[poplabels_orig.INCLUDE == 1].index.tolist() for s in sample_list
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
    count_mut_trees = -1
    count_all_tree = 0
    group_id = {}
    for u in range(len(unique_groups)):
        group_id[unique_groups[u]] = u

    for sample_no, target_seq_ in enumerate(sample_list):
        count_mut_trees_prev = copy.deepcopy(count_mut_trees)
        for chr_no, ts in enumerate(ts_list):
            tree = ts.first()
            for tid in tqdm(range(len(list(ts.trees())))):  # len(list(ts.trees()))
                if (
                    tree.interval[1] // force_build - tree.interval[0] // force_build
                    > 0
                ):
                    sample_list_tree = copy.deepcopy(sample_list)
                    if mask_dodgy[count_all_tree]:
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
                        for _ in range(num_subtrees):
                            count_mut_trees += 1
                            poplabels = subsample_poplabels(poplabels_orig, sample_list, max_per_group)
                            lineage_content = np.zeros(
                                (2 * num_samples - 1, len(unique_groups)), dtype="float64"
                            )
                            target_seq = target_seq_
                            for m in range(len(poplabels)):
                                ## Only count lineage content for included samples
                                if poplabels.INCLUDE.iloc[m]:
                                    lineage_content[
                                        m, group_id[poplabels.GROUP.iloc[m]]
                                    ] = 1

                            for t in sample_list_tree:
                                lineage_content[
                                    t
                                ] = 0  ## setting lineage content of target sequences = 0
                            prev_branch_length = np.sum(
                                lineage_content, axis=0
                            )  # np.sum(lineage_content[:,1])
                            proportion_of_coalescing_in_tree = []
                            coalescene_times_in_tree = []
                            epoch_index_in_tree = []
                            event_count = 0
                            for epoch in range(len(epoch_intervals_pow) - 1):
                                coal_events_submatrix = coal_events_matrix[
                                    (coal_events_matrix[:, 3] >= epoch_intervals_pow[epoch])
                                    & (
                                        coal_events_matrix[:, 3]
                                        < epoch_intervals_pow[epoch + 1]
                                    )
                                ]
                                tprev = max(
                                    epoch_intervals_pow[epoch],
                                    poplabels.SAMPLING_TIME.iloc[target_seq_],
                                )  ##only considering coalescene events after the sampling time of the target

                                for (a, b, c, t) in coal_events_submatrix:
                                    event_count += 1
                                    a = int(a)
                                    b = int(b)
                                    c = int(c)
                                    opportunity[:, epoch, count_mut_trees] += (
                                        max(t, poplabels.SAMPLING_TIME.iloc[target_seq_])
                                        - tprev
                                    ) * (
                                        prev_branch_length
                                    )  ##only considering coalescene events after the sampling time of the target
                                    if (
                                        a in sample_list_tree and b in sample_list_tree
                                    ):  ## sometimes the target sequences coalesces with each other, in that case we append the coalesced node to the sample's list for that tree
                                        sample_list_tree.append(c)

                                    if (a == target_seq and b in sample_list_tree) or (
                                        b == target_seq and a in sample_list_tree
                                    ):  ## in case the target sequences coalesces with other target sequence, we don't count that coalescene count and opportunity
                                        target_seq = c
                                        lineage_content[c] = 0

                                    elif (
                                        a == target_seq and sum(lineage_content[b]) == 0
                                    ) or (b == target_seq and sum(lineage_content[a]) == 0):
                                        ## This happens when target coalesces with a sample not included, in that case don't count that event
                                        target_seq = c
                                        lineage_content[c] = 0

                                    elif a == target_seq:
                                        proportion_of_coalescing = copy.deepcopy(
                                            lineage_content[b]
                                        ) / (sum(lineage_content[b]))
                                        coal_count[
                                            :, epoch, count_mut_trees
                                        ] += proportion_of_coalescing
                                        target_seq = c
                                        lineage_content[c] = 0
                                        proportion_of_coalescing_in_tree.append(
                                            proportion_of_coalescing
                                        )
                                        epoch_index_in_tree.append(epoch)
                                        prev_branch_length = (
                                            prev_branch_length
                                            - lineage_content[b] / (sum(lineage_content[b]))
                                        )
                                    elif b == target_seq:
                                        proportion_of_coalescing = copy.deepcopy(
                                            lineage_content[a]
                                        ) / (
                                            sum(lineage_content[a])
                                        )  ## sum() faster than np.sum()
                                        coal_count[
                                            :, epoch, count_mut_trees
                                        ] += proportion_of_coalescing
                                        target_seq = c
                                        lineage_content[c] = 0
                                        proportion_of_coalescing_in_tree.append(
                                            proportion_of_coalescing
                                        )
                                        epoch_index_in_tree.append(epoch)
                                        prev_branch_length = (
                                            prev_branch_length
                                            - lineage_content[a] / (sum(lineage_content[a]))
                                        )

                                    else:  ## we don't count the branch lengths for the samples in sample_list_tree because they are the target sequences
                                        lineage_content[c] = (
                                            lineage_content[a] + lineage_content[b]
                                        )
                                        if (
                                            sum(lineage_content[a]) == 0
                                            or sum(lineage_content[b]) == 0
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
                                                / (sum(lineage_content[b]))
                                                + lineage_content[c]
                                                / (sum(lineage_content[c]))
                                            )
                                        elif (
                                            b in sample_list_tree
                                            and a not in sample_list_tree
                                        ):
                                            prev_branch_length = (
                                                prev_branch_length
                                                - lineage_content[a]
                                                / (sum(lineage_content[a]))
                                                + lineage_content[c]
                                                / (sum(lineage_content[c]))
                                            )
                                        elif (
                                            a not in sample_list_tree
                                            and b not in sample_list_tree
                                        ):
                                            prev_branch_length = (
                                                prev_branch_length
                                                - lineage_content[a]
                                                / (sum(lineage_content[a]))
                                                - lineage_content[b]
                                                / (sum(lineage_content[b]))
                                                + lineage_content[c]
                                                / (sum(lineage_content[c]))
                                            )
                                    lineage_content[a] = 0
                                    lineage_content[b] = 0
                                    tprev = max(
                                        t, poplabels.SAMPLING_TIME.iloc[target_seq_]
                                    )
                                if epoch < len(epoch_intervals_pow) - 2:
                                    opportunity[:, epoch, count_mut_trees] += (
                                        max(
                                            epoch_intervals_pow[epoch + 1],
                                            poplabels.SAMPLING_TIME.iloc[target_seq_],
                                        )
                                        - max(
                                            tprev, poplabels.SAMPLING_TIME.iloc[target_seq_]
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
                    if (
                        poplabels.SAMPLING_TIME.iloc[m]
                        >= epoch_intervals_pow[epoch + 1]
                    ):
                        opportunity[
                            group_id[poplabels.GROUP.iloc[m]],
                            epoch,
                            count_mut_trees_prev + 1 : count_mut_trees + 1,
                        ] -= epoch_intervals_pow[epoch + 1] - max(
                            epoch_intervals_pow[epoch],
                            poplabels.SAMPLING_TIME.iloc[target_seq_],
                        )
                    elif (
                        poplabels.SAMPLING_TIME.iloc[m]
                        > max(
                            epoch_intervals_pow[epoch],
                            poplabels.SAMPLING_TIME.iloc[target_seq_],
                        )
                        and poplabels.SAMPLING_TIME.iloc[m]
                        < epoch_intervals_pow[epoch + 1]
                    ):
                        opportunity[
                            group_id[poplabels.GROUP.iloc[m]],
                            epoch,
                            count_mut_trees_prev + 1 : count_mut_trees + 1,
                        ] -= poplabels.SAMPLING_TIME.iloc[m] - max(
                            epoch_intervals_pow[epoch],
                            poplabels.SAMPLING_TIME.iloc[target_seq_],
                        )

    return coal_count, opportunity, proportion_of_coalescing_all, epoch_index_all


def load_fixed_params(args, ts_list, poplabels, mask_dodgy):
    chrs = list(map(int, args.chrs.split(",")))
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    num_trees = np.sum(mask_dodgy)
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)

    # epoch_intervals = np.array(
    #     [-np.inf]
    #     + np.linspace(
    #         args.start_time - math.log(28, 10),
    #         args.end_time - math.log(28, 10),
    #         args.num_epochs - 1,
    #     ).tolist()
    #     + [np.inf],
    #     dtype="float64",
    # )
    epoch_intervals = get_epoch_interval(args, ts_list)
    epoch_intervals_pow = np.power(10, epoch_intervals)

    fixed_params_file_name = (
        args.output
        + "_fixed_params_"
        + sample_id_label
        + "_"
        + str(args.force_build)
        + "_"
        + str(args.chrs)
        + "_"
        + str(args.masking_threshold)
        + ".pkl"
    )
    try:
        f_pkl = open(fixed_params_file_name, "rb")
        if args.mode == "sim":
            (
                num,
                denom,
                proportion_of_coalescing_all,
                epoch_index_all,
                ground_truth_membership,
            ) = pickle.load(f_pkl)
        else:
            (
                num,
                denom,
                proportion_of_coalescing_all,
                epoch_index_all,
            ) = pickle.load(f_pkl)
            f_pkl.close()
            ground_truth_membership = None
        print("Done loading fixed parameters from: " + str(fixed_params_file_name))

    except:
        print("Fixed parameters file not found, calculating fixed parameters..")
        if args.mode == "sim":
            ground_truth_membership = make_ground_truth(
                ts_list,
                num_trees//args.num_subtrees,
                mask_dodgy=mask_dodgy[::args.num_subtrees],
                path=args.ground_truth_path,
                sample=args.sample_id,
                chrs=chrs,
                force_build=args.force_build,
            )
            ground_truth_membership = np.repeat(ground_truth_membership, args.num_subtrees, axis=1)
        else:
            ground_truth_membership = None
        (num, denom, proportion_of_coalescing_all, epoch_index_all,) = fixed_parameters(
            ts_list,
            poplabels,
            unique_groups,
            num_trees,
            mask_dodgy,
            args.sample_id,
            epoch_intervals_pow,
            args.force_build,
            args.num_subtrees,
            args.max_per_group
        )
        f_pkl = open(fixed_params_file_name, "wb")
        if args.mode == "sim":
            pickle.dump(
                [
                    num,
                    denom,
                    proportion_of_coalescing_all,
                    epoch_index_all,
                    ground_truth_membership,
                ],
                f_pkl,
            )
        else:
            pickle.dump(
                [
                    num,
                    denom,
                    proportion_of_coalescing_all,
                    epoch_index_all,
                ],
                f_pkl,
            )
            f_pkl.close()
        print("Fixed parameters stored in: " + str(fixed_params_file_name))

    if (denom < -1e-8).any():
        raise ValueError(
            "The opportunity has negative values, check the sampling times in poplabels.txt"
        )

        ### Clipping the opportunity to zero (because there might be some very small -ve values cause of numerical instabilities)
    denom = copy.deepcopy(np.maximum(denom, 0))
    return (
        num,
        denom,
        proportion_of_coalescing_all,
        epoch_index_all,
        ground_truth_membership,
    )
