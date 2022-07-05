import pandas as pd
import numpy as np
import math
import time
import tskit
import argparse
import copy
from tqdm import tqdm

from calc_tree_stats import load_tree_stats
from calc_fixed_params import load_fixed_params
from utils import (
    filter_recomb_rate,
    filter_opportunity,
    load_mask_csv,
    write_coal,
    write_calibration,
    calculate_accuracy,
    boolean,
)


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


def e_m_step(
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    n_unique_groups,
    n_epochs,
    n_trees,
    epoch,
):
    masked_trees_index = np.arange(0, n_trees)
    gamma_arr = np.zeros(
        (len(own_membership), n_unique_groups, n_epochs - 1),
        dtype="float64",
    )
    for j in range(len(own_membership)):
        if epoch == 0:
            n = compute_gamma_num(
                own_membership[j],
                None,
                proportion_of_coalescing_all,
                epoch_index_all,
                n_unique_groups,
                masked_trees_index,
                n_epochs,
            )
        else:
            n = compute_gamma_num(
                own_membership[j],
                prev_gamma[j],
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
    if epoch == 0 and args.load_gamma != None and args.load_props != None:
        print("Using initial gamma specified in file: " + str(args.load_gamma))
        gamma_arr = np.load(args.load_gamma)
        tau = np.load(args.load_props)  ### load taus only works for not(props_per_chrs)

    #if tau[0] < tau[1]:
    #    tau = [0.05, 0.95]  ## CAUTION: Fixing tau!!!
    #else:
    #    tau = [0.95, 0.05]
    #tau = np.array(tau)

    assert (gamma_arr >= 0).all()
    prev_gamma = copy.deepcopy(gamma_arr)
    own_membership_update = np.ones((len(own_membership), n_trees), dtype="float64")
    log_num_em = np.zeros((len(own_membership), n_trees), dtype="float64")
    log_denom_em = np.zeros((len(own_membership), n_trees), dtype="float64")
    count_masked_trees = 0

    for tid in masked_trees_index:
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
        epoch_index_in_tree = epoch_index_all[tid]
        for j in range(len(own_membership)):
            log_num_em_j, log_denom_em_j = update_membership(
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
    own_membership_update = np.exp(
        log_num_em
        + log_denom_em
        - np.repeat(
            np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
            len(own_membership),
            axis=1,
        ).T
    )

    for j in range(len(own_membership)):
        own_membership_update[j] *= tau[j]

    log_likelihood = np.sum(
        np.log(np.sum(own_membership_update, axis=0))
        + np.max(log_num_em + log_denom_em, axis=0)
    )
    own_membership = own_membership_update / (np.sum(own_membership_update, axis=0))
    return own_membership, gamma_arr, tau, log_likelihood


def random_sweep(
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    n_clusters,
    n_unique_groups,
    n_epochs,
    n_trees,
    n_repeats,
):
    print("Performing a random sweep for better initialization")
    masked_trees_index = np.arange(0, n_trees)
    best_loglikelihood = -np.inf
    for n_iters in tqdm(range(n_repeats)):
        gamma_arr = np.power(
            np.e,
            np.random.uniform(
                -16.11, -4.6, (n_clusters, n_unique_groups, n_epochs - 1)
            ),
        )
        tau = np.random.uniform(0.01, 0.99, n_clusters)

        own_membership_trial = np.ones((n_clusters, n_trees), dtype="float64")
        log_num_em = np.zeros((n_clusters, n_trees), dtype="float64")
        log_denom_em = np.zeros((n_clusters, n_trees), dtype="float64")
        count_masked_trees = 0

        for tid in masked_trees_index:
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for j in range(n_clusters):
                log_num_em_j, log_denom_em_j = update_membership(
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
        own_membership_trial = np.exp(
            log_num_em
            + log_denom_em
            - np.repeat(
                np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
                n_clusters,
                axis=1,
            ).T
        )

        for j in range(n_clusters):
            own_membership_trial[j] *= tau[j]

        own_membership_trial = own_membership_trial / (
            np.sum(own_membership_trial, axis=0)
        )

        for epoch in range(10):
            own_membership_trial, gamma_arr, tau, log_likelihood = e_m_step(
                own_membership_trial,
                gamma_arr,
                proportion_of_coalescing_all,
                epoch_index_all,
                denom,
                n_unique_groups,
                n_epochs,
                n_trees,
                epoch,
            )
        if log_likelihood > best_loglikelihood:
            best_loglikelihood = log_likelihood
            own_membership = own_membership_trial

    return own_membership


def write_membership_gamma(
    args,
    own_membership,
    gamma_arr,
    mask_dodgy,
    chr_map,
    tree_left_bp,
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

    tree_position = []
    for tid in range(len(mask_dodgy) // len(args.sample_id)):
        if mask_dodgy[tid]:
            tree_position.append([chr_map[tid], tree_left_bp[tid] // args.force_build])
    filename = (
        "mask_" + sample_id_label + ".csv"
    )  ## this saves membership for all the trees (without the filtering)
    filename = args.output + "_" + filename
    pd.DataFrame(np.array(tree_position), columns=["chr", "pos"]).to_csv(
        filename, index=False, sep="\t"
    )


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
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    poplabels = pd.read_csv(args.poplabels, sep="\s+")
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))

    ### Load all the trees in a list
    ts_list = load_trees(args, poplabels)
    if len(poplabels) == ts_list[0].num_samples // 2:
        ## If the poplabels files is one entry per individual (not haplotype)
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
        chr_map,
        frac_branches_with_snp_target,
    ) = load_tree_stats(args, ts_list, poplabels)

    ### Filter based on recombination rates
    if args.load_mask is None:
        mask_dodgy = filter_recomb_rate(
            args, ts_list, frac_branches_with_snp_target, recomb_rates
        )
    if args.load_mask is not None:
        mask_dodgy = np.zeros(len(recomb_rates), dtype="bool")
        mask_dodgy = load_mask_csv(args, args.sample_id, ts_list, mask_dodgy, chrs)

    ### Load fixed params
    (
        num,
        denom,
        proportion_of_coalescing_all,
        epoch_index_all,
        ground_truth_membership,
    ) = load_fixed_params(args, ts_list, poplabels, mask_dodgy)

    ### Filter based on opportunity filter
    if args.opportunity_filter and args.load_mask is None:
        mask_dodgy_low_evidence = filter_opportunity(
            args,
            ts_list,
            mutrate_opportunity_target,
            epoch_index_all,
            proportion_of_coalescing_all,
            denom,
            mask_dodgy,
        )
        if args.mode == "sim":
            ground_truth_membership = ground_truth_membership[
                :, mask_dodgy_low_evidence
            ]
        denom = denom[:, :, mask_dodgy_low_evidence]
        for tid in sorted(range(len(epoch_index_all)), reverse=True):
            if not mask_dodgy_low_evidence[tid]:
                del epoch_index_all[tid]
                del proportion_of_coalescing_all[tid]

        mask_dodgy[mask_dodgy] *= mask_dodgy_low_evidence

    ### Initialize local ancestry
    if args.init_at_truth:
        own_membership = ground_truth_membership
    else:
        num_trees = np.sum(mask_dodgy)
        own_membership = random_sweep(
            proportion_of_coalescing_all,
            epoch_index_all,
            denom,
            args.num_clusters,
            len(unique_groups),
            len(epoch_intervals),
            np.sum(mask_dodgy),
            args.n_repeats,
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
                own_membership,
                gamma_arr,
                proportion_of_coalescing_all,
                epoch_index_all,
                denom,
                len(unique_groups),
                len(epoch_intervals),
                np.sum(mask_dodgy),
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
            mask_dodgy,
            chr_map,
            tree_left_bp,
            epoch_intervals,
            unique_groups,
            sample_id_label,
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-fb",
        "--force_build",
        help="force build size to subsample the trees in bp",
        type=int,
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
        default=100,
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
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    np.random.seed(args.seed)  ## fix the random seed
    main(args)

## python ghost_buster.py --mode sim --trees example/stdpopsim_homsap_chr --poplabels example/poplabels.txt --ground_truth example/local_ancestry_chr  --rec example/genetic_map_GRCh37_chr --sample_id 51 --chr 22 --output example/stdpopsim_homsap --init_at_truth 1
## python ghost_buster.py --mode sim --trees example/relate_homsap_chr --poplabels example/poplabels.txt --ground_truth example/local_ancestry_chr  --rec example/genetic_map_GRCh37_chr --mutden example/relate_homsap_chr --allmuts example/relate_homsap_chr --opportunity_filter 1 --sample_id 51 52 --chr 22 --output example/relate_homsap --init_at_truth 1
