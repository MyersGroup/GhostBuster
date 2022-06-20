"""
utils.py: contains basic helper function used throughout the code
Includes: boolean(), make_one_hot(), mask_for_dodgy_trees(), downsample_trees(), write_coal()
"""

import argparse
import numpy as np
from sklearn.calibration import calibration_curve
import pandas as pd

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


def make_one_hot(X, max_X):
    X = np.array(X, dtype="int")
    classes = np.arange(0, max_X, 1)
    # Y = []
    if len(X.shape) == 2:
        Y = np.zeros((len(classes), X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        Y = np.zeros((len(classes), X.shape[0]))
    for c in classes:
        # Y.append(scipy.sparse.csr_matrix(np.array(X == c, dtype='int')))
        Y[c] = np.array(X == c, dtype="int")
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


def filter_recomb_rate(args, ts_list, tree_left_bp, recomb_rates):
    chrs = list(map(int, args.chrs.split(",")))
    num_trees = int(np.sum([ts.num_trees for ts in ts_list]))
    chr_list = []
    for c in range(len(ts_list)):
        num_of_trees_in_chr = [c + 1] * ts_list[c].num_trees
        chr_list.extend(num_of_trees_in_chr)

    mask_dodgy = mask_for_dodgy_trees(
        recomb_rates,
        1 - args.masking_threshold,
    )
    recomb_0_thresh = np.sum(np.array(recomb_rates) <= 0) / len(recomb_rates)
    mask_dodgy *= ~mask_for_dodgy_trees(
        recomb_rates,
        recomb_0_thresh,
    )
    mask_dodgy = np.array(mask_dodgy)
    if args.mode == "sim":
       ##### Caution: manually downsampling HAN (1) !! 🌵
       print("Downsampling !! Caution !!")
       ground_truth_membership = make_ground_truth(
           ts_list,
           np.sum(mask_dodgy),
           mask_dodgy=mask_dodgy,
           path=args.ground_truth_path,
           sample=args.sample_id,
           chrs=chrs,
           force_build=args.force_build,
       )
       mask_dodgy[mask_dodgy] *= downsample_trees(ground_truth_membership, 1, 0.25)

    print(
        "Filtering based on recombination rate, trees remaining: "
        + str(sum(mask_dodgy))
        + " average recomb. rate: "
        + str(np.mean(np.array(recomb_rates)[mask_dodgy]))
    )
    mask_dodgy = np.tile(mask_dodgy, len(args.sample_id))
    return mask_dodgy


def filter_opportunity(
    args,
    ts_list,
    mutrate_opportunity_target,
    epoch_index_all,
    proportion_of_coalescing_all,
    denom,
    mask_dodgy,
):
    chr_list = []
    for c in range(len(ts_list)):
        num_of_trees_in_chr = [c + 1] * ts_list[c].num_trees
        chr_list.extend(num_of_trees_in_chr)

    mutrate_opportunity_target = np.tile(
        np.array(mutrate_opportunity_target), (len(args.sample_id), 1)
    )
    mutrate_opportunity_target_masked = np.array(mutrate_opportunity_target)[mask_dodgy]
    mutrate_opportunity_thresh = np.percentile(
        mutrate_opportunity_target_masked, args.masking_threshold * 100, axis=0
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
        )

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
