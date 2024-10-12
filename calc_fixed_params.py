"""
calc_fixed_params.py: contains code to calculate numerator and denominator for each tree & 
helper functions for the EM
"""

import numpy as np
import copy
from tqdm import tqdm
import pickle
import math
from utils import make_numba_nested_list
import random
import numba
import time

@numba.jit(nopython=True)
def process_epochs(
    epoch_intervals_pow,
    coal_events_matrix,
    lineage_content,
    target_sampling_time,
    target_seq,
    opportunity,
    count_mut_trees,
    num_samples,
    coal_count,
    ignore_first_epoch
):
    prev_branch_length = lineage_content.sum(axis=0)
    lineage_content_sum = lineage_content.sum(axis=1)
    proportion_of_coalescing_in_tree = []
    epoch_index_in_tree = []
    denom_in_tree = []

    event_count = 0
    for epoch in range(len(epoch_intervals_pow) - 1):
        coal_events_submatrix = coal_events_matrix[
            (coal_events_matrix[:, 3] >= epoch_intervals_pow[epoch])
            & (coal_events_matrix[:, 3] < epoch_intervals_pow[epoch + 1])
        ]

        tprev = max(epoch_intervals_pow[epoch], target_sampling_time)  ## Only considering coalescence events after the sampling time of the target

        if epoch == 0 and ignore_first_epoch:
            for i in range(len(lineage_content)):
                if lineage_content_sum[i] > 0:
                    lineage_content[i] /= np.sum(lineage_content[i])
                    lineage_content_sum[i] = 1

        for (a, b, c, t) in coal_events_submatrix:
            event_count += 1
            a, b, c = int(a), int(b), int(c)
            opportunity[:, epoch, count_mut_trees] += (max(t, target_sampling_time) - tprev) * prev_branch_length
            if (a == target_seq and lineage_content_sum[b] == 0) or (b == target_seq and lineage_content_sum[a] == 0):
                target_seq = c
                lineage_content[c], lineage_content_sum[c] = 0, 0
            elif a == target_seq:
                proportion_of_coalescing = lineage_content[b].copy()
                coal_count[count_mut_trees] += 1
                target_seq = c
                lineage_content[c], lineage_content_sum[c] = 0, 0
                proportion_of_coalescing_in_tree.append(proportion_of_coalescing)
                epoch_index_in_tree.append(epoch)
                denom_in_tree.append(opportunity[:, :, count_mut_trees].copy())
                opportunity[:, :, count_mut_trees] = 0
                prev_branch_length -= lineage_content[b] / lineage_content_sum[b]
            elif b == target_seq:
                proportion_of_coalescing = lineage_content[a].copy()
                coal_count[count_mut_trees] += 1
                target_seq = c
                lineage_content[c], lineage_content_sum[c] = 0, 0
                proportion_of_coalescing_in_tree.append(proportion_of_coalescing)
                epoch_index_in_tree.append(epoch)
                denom_in_tree.append(opportunity[:, :, count_mut_trees].copy())
                opportunity[:, :, count_mut_trees] = 0
                prev_branch_length -= lineage_content[a] / lineage_content_sum[a]
            else:
                lineage_content[c] = lineage_content[a] + lineage_content[b]
                lineage_content_sum[c] = lineage_content_sum[a] + lineage_content_sum[b]
                if lineage_content_sum[a] == 0 or lineage_content_sum[b] == 0:
                    pass
                elif a == target_seq and b != target_seq:
                    prev_branch_length -= lineage_content[b] / lineage_content_sum[b]
                    prev_branch_length += lineage_content[c] / lineage_content_sum[c]
                elif b == target_seq and a != target_seq:
                    prev_branch_length -= lineage_content[a] / lineage_content_sum[a]
                    prev_branch_length += lineage_content[c] / lineage_content_sum[c]
                elif a != target_seq and b != target_seq:
                    prev_branch_length -= (lineage_content[a] / lineage_content_sum[a]) + (lineage_content[b] / lineage_content_sum[b])
                    prev_branch_length += lineage_content[c] / lineage_content_sum[c]
            lineage_content[a], lineage_content[b] = 0, 0
            lineage_content_sum[a], lineage_content_sum[b] = 0, 0
            tprev = max(t, target_sampling_time)
        if epoch < len(epoch_intervals_pow) - 2:
            opportunity[:, epoch, count_mut_trees] += (
                max(epoch_intervals_pow[epoch + 1], target_sampling_time)
                - max(tprev, target_sampling_time)
            ) * prev_branch_length
        if (event_count == num_samples - 1) and epoch <= len(
            epoch_intervals_pow
        ) - 2:
            opportunity[:, epoch + 1 :, count_mut_trees] = 0.0
            break

    return (
        proportion_of_coalescing_in_tree,
        epoch_index_in_tree,
        denom_in_tree,
        coal_count,
    )

def fixed_parameters(
    ts_list,
    poplabels_orig,
    unique_groups,
    num_trees,
    mask_dodgy,
    sample,
    epoch_intervals_pow,
    target_branch_length_masked,
    mutscale_masked,
    chr,
    force_build=1,
    ignore_first_epoch=False,
    gt_ref=None,
    exact_pos=None,
):
    eps = 1e-20
    if exact_pos is not None:
        exact_pos_chr = exact_pos[exact_pos['chr'] == chr]
    num_samples = len(list(ts_list[0].first().samples()))
    num_nodes = len(list(ts_list[0].first().nodes()))
    coal_count = np.zeros(
        num_trees,
        dtype="int32",
    )
    num_sites_per_tree = np.zeros(num_trees, dtype="int32")
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
        (2 * num_samples - 1, len(np.unique(poplabels_orig[poplabels_orig.INCLUDE == 1].GROUP))),
        dtype="float32",
    )
    for m in range(len(poplabels_orig)):
        ## Only count lineage content for included samples
        if poplabels_orig.INCLUDE.iloc[m]:
            lineage_content_init[m, group_id[poplabels_orig.GROUP.iloc[m]]] = 1.0
    for t in [sample]:
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
                    np.ceil(tree.interval[1] / force_build) - np.ceil(tree.interval[0] / force_build)
                    > 0
                ):
                    if mask_dodgy[count_all_tree]:
                        ## Make the coalescene table and sort it
                        coal_events_matrix = []
                        mapping = {}
                        count = num_samples
                        for s in tree.nodes():
                            if s < num_samples:
                                mapping[s] = int(s)
                            else:
                                mapping[s] = int(count)
                                count += 1
                        for s in tree.nodes(order='timeasc'):
                            ch = tree.children(s)
                            if ch != ():
                                a, b = ch
                                c = s
                                t = tree.time(c)
                                coal_events_matrix.append(
                                    [
                                        mapping[a],
                                        mapping[b],
                                        mapping[c],
                                        t,
                                    ]
                                )
                        coal_events_matrix = np.array(coal_events_matrix, dtype="float64")
                        target_seq = target_seq_
                        count_mut_trees += 1
                        if exact_pos is None:
                            num_sites_per_tree[count_mut_trees] = (np.ceil(tree.interval[1] / force_build) - np.ceil(tree.interval[0] / force_build))
                        else:
                            num_sites_per_tree[count_mut_trees] = np.searchsorted(exact_pos_chr['pos'].values, tree.interval[1]) - np.searchsorted(exact_pos_chr['pos'].values, tree.interval[0])
                        poplabels = poplabels_orig.copy()
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
                                    if np.isnan(gt_ref[m, count_mut_trees]):
                                        lineage_content[m] = 0
                                    else:
                                        lineage_content[m, int(gt_ref[m,count_mut_trees])] = 1
                            for t in [sample]:
                                lineage_content[t] = 0.0

                        proportion_of_coalescing_in_tree, epoch_index_in_tree, denom_in_tree, coal_count = process_epochs(
                            epoch_intervals_pow,
                            coal_events_matrix,
                            lineage_content,
                            target_sampling_time,
                            target_seq,
                            opportunity,
                            count_mut_trees,
                            num_samples,
                            coal_count,
                            ignore_first_epoch,
                        )
                        proportion_of_coalescing_all.append(
                            proportion_of_coalescing_in_tree
                        )
                        epoch_index_all.append(epoch_index_in_tree)
                        denom_all.append(denom_in_tree)

                    count_all_tree += 1
                tree.next()
        sampling_time = poplabels.SAMPLING_TIME.values
        group_ids = poplabels.GROUP.values
        include_flags = poplabels.INCLUDE.values
        for m in range(len(poplabels)):
            if m != sample and include_flags[m]:
                m_sampling_time = sampling_time[m]
                group_id_m = group_id[group_ids[m]]
                for epoch in range(len(epoch_intervals_pow) - 1):
                    if epoch_intervals_pow[epoch + 1] < sampling_time[target_seq_]:
                        continue
                    if m_sampling_time > epoch_intervals_pow[epoch]:
                        epoch_diff = max(min(m_sampling_time, epoch_intervals_pow[epoch + 1]) - epoch_intervals_pow[epoch], 0)
                        for tid, denom_tree in enumerate(denom_all):
                            for denom_coal in denom_tree:
                                if gt_ref is None:
                                    denom_value = denom_coal[group_id_m, epoch]
                                else:
                                    denom_value = denom_coal[int(gt_ref[m, tid]), epoch]
                                if denom_value > epoch_diff:
                                    if gt_ref is None:
                                        denom_coal[group_id_m, epoch] -= epoch_diff
                                    else:
                                        denom_coal[int(gt_ref[m, tid]), epoch] -= epoch_diff
                                else:
                                    if gt_ref is None:
                                        denom_coal[group_id_m, epoch] = 0
                                    else:
                                        denom_coal[int(gt_ref[m, tid]), epoch] = 0

    proportion_of_coalescing_all = [sublist for sublist, count in zip(proportion_of_coalescing_all, num_sites_per_tree) for _ in range(count)]
    epoch_index_all = [sublist for sublist, count in zip(epoch_index_all, num_sites_per_tree) for _ in range(count)]
    denom_all = [sublist for sublist, count in zip(denom_all, num_sites_per_tree) for _ in range(count)]
    coal_count = [sublist for sublist, count in zip(coal_count, num_sites_per_tree) for _ in range(count)]

    denom_epochwise = np.zeros((len(denom_all), len(denom_all[0][0]), len(denom_all[0][0][0])), dtype='float64')
    denom_epochwise_unscaled = np.zeros((len(denom_all), len(denom_all[0][0]), len(denom_all[0][0][0])), dtype='float64')
    for n_t in range(len(denom_all)):
        epoch_index_in_tree = epoch_index_all[n_t]
        for c_t in range(len(denom_all[n_t])):
            denom_epochwise[n_t] += denom_all[n_t][c_t]/target_branch_length_masked[n_t][c_t]
            denom_epochwise_unscaled[n_t] += denom_all[n_t][c_t]/mutscale_masked[n_t][c_t]
            
    return (
        coal_count,
        denom_epochwise,
        denom_epochwise_unscaled,
        proportion_of_coalescing_all,
        epoch_index_all,
    )


def load_fixed_params(args, ts_list, sample, poplabels, mask_dodgy, chr_map, epoch_intervals, target_branch_length_masked, mutscale_masked, gt_ref=None, unique_groups=None, exact_pos=None):
    chrs = list(map(int, args.chrs.split(",")))
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP) if unique_groups is None else unique_groups
    epoch_intervals_pow = np.power(10, epoch_intervals)

    denom_all = []
    denom_all_unscaled = []
    proportion_of_coalescing_all = []
    epoch_index_all = []
    num_trees_prev = 0
    for chr_no, chr in enumerate(chrs):
        if args.fixed_params_file_prefix is not None:
            fixed_params_file_name = args.fixed_params_file_prefix + "_chr" + str(chr) + "_sample" + str(sample) + ".pkl"
        else:
            fixed_params_file_name = args.output + "_fixed_params_chr" + str(chr) + "_sample" + str(sample) + ".pkl"

        try:
            f_pkl = open(fixed_params_file_name, "rb")
            (mut_scaling_file, hmm_file, force_build, start_time, end_time, ignore_first_epoch, ignore_last_epoch, masking_threshold, poplabels_file, coal_count, denom, denom_unscaled, proportion_of_coalescing, epoch_index, gt_ref_file, unique_groups_file, exact_pos_file) = pickle.load(f_pkl)
            f_pkl.close()
            if exact_pos is not None:
                if (exact_pos_file[exact_pos_file[:,0] == chr] != exact_pos[exact_pos['chr'] == chr].values).any():
                    print("Exact position file doesn't match, calculating fixed parameters..")
                    raise Exception
            if gt_ref is not None and gt_ref_file is None:
                if (mut_scaling_file == args.mut_scaling) & (hmm_file == args.hmm) & (gt_ref_file == gt_ref).all() & (unique_groups_file == unique_groups).all() & (force_build == args.force_build) & (start_time == args.start_time) & (end_time == args.end_time) & (ignore_first_epoch == args.ignore_first_epoch) & (ignore_last_epoch == args.ignore_last_epoch) & (masking_threshold==args.masking_threshold) & np.all(poplabels_file[list(set(np.arange(len(poplabels_file))) - set(args.sample_id))] == poplabels.values[list(set(np.arange(len(poplabels_file))) - set(args.sample_id))]) & (denom.shape[2] == args.num_epochs):
                    ##convert to numba list
                    denom_all.extend(denom)
                    denom_all_unscaled.extend(denom_unscaled)
                    proportion_of_coalescing_all.extend(proportion_of_coalescing)
                    epoch_index_all.extend(epoch_index)
                    print("Loaded fixed parameters from: " + str(fixed_params_file_name))
                    continue
                else:
                    print("Fixed parameters file found but parameters don't match, calculating fixed parameters..")
                    raise Exception
            elif gt_ref is None and gt_ref_file is None:
                if (mut_scaling_file == args.mut_scaling) & (hmm_file == args.hmm) &  (unique_groups_file == unique_groups).all() & (force_build == args.force_build) & (start_time == args.start_time) & (end_time == args.end_time) & (ignore_first_epoch == args.ignore_first_epoch) & (ignore_last_epoch == args.ignore_last_epoch) & (masking_threshold==args.masking_threshold) & np.all(poplabels_file[list(set(np.arange(len(poplabels_file))) - set(args.sample_id))] == poplabels.values[list(set(np.arange(len(poplabels_file))) - set(args.sample_id))]) & (denom.shape[2] == args.num_epochs):
                    ##convert to numba list
                    denom_all.extend(denom)
                    denom_all_unscaled.extend(denom_unscaled)
                    proportion_of_coalescing_all.extend(proportion_of_coalescing)
                    epoch_index_all.extend(epoch_index)
                    print("Loaded fixed parameters from: " + str(fixed_params_file_name))
                    continue
                else:
                    print("Fixed parameters file found but parameters don't match, calculating fixed parameters..")
                    raise Exception

        except:
            print("Fixed parameters file not found, calculating fixed parameters..")
            mask_dodgy_sam_chr = mask_dodgy[np.array(chr_map) == chr]
            num_trees = np.sum(mask_dodgy_sam_chr)
            target_branch_length_masked_chr = []
            mutscale_masked_chr = []
            # make faster #
            if chr_no == 0:
                num_sites_per_tree = np.zeros(np.sum(mask_dodgy), dtype='int32')            
                i, count_i  = 0,0 
                for chr_, tseq in zip(chrs, ts_list):
                    if exact_pos is not None:
                        exact_pos_chr = exact_pos[exact_pos['chr'] == chr_]
                    for tree in tseq.trees():
                        if np.ceil(tree.interval[1] / args.force_build) - np.ceil(tree.interval[0] / args.force_build) > 0:
                            if mask_dodgy[count_i]:
                                if exact_pos is None:
                                    num_sites_per_tree[i] = (np.ceil(tree.interval[1] / args.force_build) - np.ceil(tree.interval[0] / args.force_build))
                                else:
                                    num_sites_per_tree[i] = np.searchsorted(exact_pos_chr['pos'].values, tree.interval[1]) - np.searchsorted(exact_pos_chr['pos'].values, tree.interval[0])
                                i+= 1
                            count_i += 1
                chr_map_masked = np.array(chr_map)[mask_dodgy]
                chr_map_masked = [sublist for sublist, count in zip(chr_map_masked, num_sites_per_tree) for _ in range(count)]

            for t in range(len(target_branch_length_masked)):
                if chr_map_masked[t] == chr:
                    target_branch_length_masked_chr.append(target_branch_length_masked[t])
                    mutscale_masked_chr.append(mutscale_masked[t])
            
            (coal_count, denom, denom_unscaled, proportion_of_coalescing, epoch_index) = fixed_parameters(
                ts_list[chr_no:chr_no + 1],
                poplabels,
                unique_groups,
                num_trees,
                mask_dodgy_sam_chr,
                sample,
                epoch_intervals_pow,
                target_branch_length_masked_chr,
                mutscale_masked_chr,
                chr,
                args.force_build,
                ignore_first_epoch=args.ignore_first_epoch,
                gt_ref=gt_ref,
                exact_pos=exact_pos
            )
            f_pkl = open(fixed_params_file_name, "wb")
            pickle.dump([args.mut_scaling, args.hmm, args.force_build, args.start_time, args.end_time, args.ignore_first_epoch, args.ignore_last_epoch, args.masking_threshold, poplabels.values, coal_count, denom, denom_unscaled, proportion_of_coalescing, epoch_index, gt_ref, unique_groups, exact_pos.values if exact_pos is not None else None], f_pkl)
            f_pkl.close()
            denom_all.extend(denom)
            denom_all_unscaled.extend(denom_unscaled)
            proportion_of_coalescing_all.extend(proportion_of_coalescing)
            epoch_index_all.extend(epoch_index)
            print("Fixed parameters stored in: " + str(fixed_params_file_name))

    proportion_of_coalescing_all = make_numba_nested_list(proportion_of_coalescing_all)
    epoch_index_all = make_numba_nested_list(epoch_index_all)

    return (
        coal_count,
        np.array(denom_all),
        np.array(denom_all_unscaled),
        proportion_of_coalescing_all,
        epoch_index_all,
    )
