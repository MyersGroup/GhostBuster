from numpy.lib.arraysetops import unique
from numpy.testing._private.utils import decorate_methods
import wandb
import argparse
import numpy as np
import math
import pandas as pd
from pathlib import Path
import tskit
import pickle
import copy


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def update_membership(
    proportion_of_coalescing_in_tree,
    epoch_index_in_tree,
    denom,
    gamma_arr,
    tid,
    ignore_first_epoch,
    ignore_last_epoch,
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
                and epoch_index_in_tree[i] < len(epoch_intervals) - 2
            )
            or (
                ignore_first_epoch
                and ignore_last_epoch
                and epoch_index_in_tree[i] >= 1
                and epoch_index_in_tree[i] < len(epoch_intervals) - 2
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


def compute_gamma_num(
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    num_ref_groups,
    masked_trees_index,
):
    num_full_tree = np.zeros(
        (num_ref_groups, len(epoch_intervals) - 1), dtype="float64"
    )
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


def compute_gamma_denom(own_membership, denom, mask_dodgy):
    eps = 1e-200
    denom_1 = np.zeros(len(epoch_intervals) - 1, dtype="float64")
    for epoch in range(len(epoch_intervals) - 1):  #
        denom_1[epoch] = sum(denom[epoch][mask_dodgy] * own_membership)
    return denom_1 + eps


def load_data(args):
    ## loads the necessary data, e.g. trees, tree stats, fixed parameters, etc.
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    print("Considering sample ids: " + str(args.sample_id))
    poplabels = pd.read_csv(Path(args.path) / "poplabels.txt", sep=" ")
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)

    ts_list = []
    # ts_list_subsampled = []
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))
    for chr in chrs:
        ts = tskit.load(
            Path(args.path) / str(args.trees + "_chr" + str(chr) + ".trees")
        )
        ts_list.append(ts)
    if len(poplabels) != ts_list[0].num_samples:
        raise ValueError(
            "Number of samples in trees doesnt match number of samples in poplabels.txt"
        )

    f_pkl = open(args.tree_stats_file_name, "rb")
    (
        num_trees,
        trees_per_chr,
        tree_size,
        no_of_mutations,
        tmrca,
        recomb_rates,
        rank_zero_snp_branches_target,
        frac_branches_with_snp_target,
        frac_branches_with_snp,
        num_snps_on_tree,
        mask_dodgy,
    ) = pickle.load(f_pkl)
    f_pkl.close()
    print("Done loading tree statistics from: " + str(args.tree_stats_file_name))
    masked_trees_index = np.arange(0, num_trees * len(args.sample_id))[mask_dodgy]

    f_pkl = open(args.fixed_params_file_name, "rb")
    (
        num,
        denom,
        proportion_of_coalescing_all,
        epoch_index_all,
    ) = pickle.load(f_pkl)
    f_pkl.close()
    print("Done loading fixed parameters from: " + str(args.fixed_params_file_name))
    denom = copy.deepcopy(np.maximum(denom, 0))
    return (
        masked_trees_index,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        num_trees,
        mask_dodgy,
    )


def e_step(
    masked_trees_index,
    gamma_arr,
    tau,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    num_clusters,
    num_trees,
    mask_dodgy=None,
):
    own_membership_update = np.ones((num_clusters, num_trees), dtype="float64")
    log_num_em = np.zeros((num_clusters, num_trees), dtype="float64")
    log_denom_em = np.zeros((num_clusters, num_trees), dtype="float64")
    count_masked_trees = 0

    for tid in masked_trees_index:
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
        epoch_index_in_tree = epoch_index_all[tid]
        for j in range(num_clusters):
            log_num_em_j, log_denom_em_j = update_membership(
                proportion_of_coalescing_in_tree,
                epoch_index_in_tree,
                denom,
                gamma_arr[j],
                tid,
                args.ignore_first_epoch,
                args.ignore_last_epoch,
            )
            log_num_em[j, count_masked_trees] = log_num_em_j
            log_denom_em[j, count_masked_trees] = log_denom_em_j
        count_masked_trees += 1
    own_membership_update = np.exp(
        log_num_em
        + log_denom_em
        - np.repeat(
            np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
            num_clusters,
            axis=1,
        ).T
    )
    own_membership_update = np.nan_to_num(own_membership_update, nan=1)
    for j in range(num_clusters):
        own_membership_update[j] *= tau[j]

    if mask_dodgy is None:
        log_likelihood = np.sum(
            np.log(np.sum(own_membership_update, axis=0))
            + np.max(log_num_em + log_denom_em, axis=0)
        )
    else:
        log_likelihood = np.sum(
            np.log(np.sum(own_membership_update, axis=0))[mask_dodgy]
            + np.max(log_num_em + log_denom_em, axis=0)[mask_dodgy]
        )

    own_membership = own_membership_update / (np.sum(own_membership_update, axis=0))
    return log_likelihood, own_membership


def m_step(
    masked_trees_index,
    own_membership,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    mask_dodgy,
):
    gamma_arr = np.zeros(
        (len(own_membership), num_reference_groups, num_epochs),
        dtype="float64",
    )
    for j in range(len(own_membership)):
        n = compute_gamma_num(
            own_membership[j],
            None,
            proportion_of_coalescing_all,
            epoch_index_all,
            num_reference_groups,
            masked_trees_index,
        )
        for i in range(num_reference_groups):
            d = compute_gamma_denom(own_membership[j], denom[i], mask_dodgy)
            gamma_arr[j][i] = copy.deepcopy(n[i] / d)  # n/d #

    tau = np.mean(own_membership, axis=1)
    return gamma_arr, tau


def get_log_likelihood(
    masked_trees_index,
    gamma_arr,
    tau,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    num_clusters,
    num_trees,
    mask_dodgy,
):
    ## Gives the loglikelihood of observing the data given the coalescene rates
    _, own_membership = e_step(
        masked_trees_index,
        gamma_arr,
        tau,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        num_clusters,
        len(masked_trees_index),
    )
    gamma_arr, tau = m_step(
        masked_trees_index,
        own_membership,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        mask_dodgy,
    )
    log_likelihood, own_membership = e_step(
        np.arange(len(args.sample_id) * num_trees),
        gamma_arr,
        tau,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        num_clusters,
        num_trees,
        mask_dodgy,
    )
    return log_likelihood, own_membership


def main(config=None):
    with wandb.init(config=config):
        args = wandb.config

        gamma_arr = np.zeros(
            (num_clusters, num_reference_groups, num_epochs),
            dtype="float64",
        )
        tau = np.zeros(num_clusters, dtype="float64")
        for k in range(num_clusters):
            for j in range(num_reference_groups):
                for e in range(num_epochs):
                    gamma_arr[k, j, e] = args["gamma" + str(k) + str(j) + str(e)]

        for k in range(num_clusters):
            tau[k] = args["tau" + str(k)]
            assert tau[k] > 0
        tau = tau / np.sum(tau)

        ## calculate log-likelihood

        log_likelihood, own_membership = get_log_likelihood(
            masked_trees_index,
            gamma_arr,
            tau,
            proportion_of_coalescing_all,
            epoch_index_all,
            denom,
            num_clusters,
            num_trees,
            mask_dodgy,
        )

        ## log the log-likelihood & local ancestry
        np.save("bayesopt/membership.npy", own_membership)
        wandb.log({"log_likelihood": log_likelihood})


if __name__ == "__main__":
    ## constants
    num_clusters = 2
    num_reference_groups = 2
    num_epochs = 4

    ## wandb arguments
    parser = argparse.ArgumentParser()
    wandb_group = parser.add_argument_group("WandB")
    wandb_mode = wandb_group.add_mutually_exclusive_group()
    wandb_mode.add_argument(
        "--wandb_offline",
        dest="wandb_mode",
        default=None,
        action="store_const",
        const="offline",
    )
    wandb_mode.add_argument(
        "--wandb_disabled",
        dest="wandb_mode",
        default=None,
        action="store_const",
        const="disabled",
    )

    wandb_group.add_argument(
        "--wandb_project_name",
        help="wandb project name",
        default="ghost_buster",
    )

    wandb_group.add_argument(
        "--wandb_run_path",
        help="The wandb run_path to load the checkpoint from, e.g., nfrc/cavia-debug-2/1i1an80e",
        default=None,
    )

    wandb_group.add_argument(
        "--wandb_job_type",
        help="Wandb job type. This is useful for grouping runs together.",
        default=None,
    )
    parser.add_argument(
        "-start_time",
        "--start_time",
        help="Starting time for the population size plots, measured in log-scale",
        type=float,
        default=4,
    )
    parser.add_argument(
        "-end_time",
        "--end_time",
        help="Ending time for the population size plots, measured in log-scale",
        type=float,
        default=7,
    )
    parser.add_argument(
        "-path",
        "--path",
        help="Location to the trees, ground truth assignments and recombination maps ",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-trees",
        "--trees",
        help="Prefix of the trees file present in args.path folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-chrs",
        "--chrs",
        help="Comma-seperated list of chromosomes to be considered",
        type=str,
        default="1,2",
    )
    parser.add_argument(
        "--tree_stats_file_name",
        help="Location to the tree stats file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fixed_params_file_name",
        help="Location to the fixed params file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-ignore_first_epoch",
        "--ignore_first_epoch",
        help="Ignore first epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "-ignore_last_epoch",
        "--ignore_last_epoch",
        help="Ignore last epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )

    for k in range(num_clusters):
        for j in range(num_reference_groups):
            for e in range(num_epochs):
                parser.add_argument(
                    "--gamma" + str(k) + str(j) + str(e), type=float, default=None
                )

    for j in range(num_clusters):
        parser.add_argument("--tau" + str(j), type=float)

    args = parser.parse_args()
    if args.gamma000 is None:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "log_likelihood", "goal": "maximize"},
            "parameters": {
                "gamma000": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma001": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma002": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma003": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma010": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma011": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma012": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma013": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma100": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma101": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma102": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma103": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma110": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma111": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma112": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "gamma113": {"distribution": "log_uniform", "min": -16.11, "max": -4.6},
                "tau0": {"distribution": "uniform", "min": 0, "max": 1},
                "tau1": {"distribution": "uniform", "min": 0, "max": 1},
                "path": {"value": args.path},
                "trees": {"value": args.trees},
                "chrs": {"value": args.chrs},
                "tree_stats_file_name": {"value": args.tree_stats_file_name},
                "fixed_params_file_name": {"value": args.fixed_params_file_name},
                "sample_id": {"value": args.sample_id},
                "ignore_first_epoch": {"value": args.ignore_first_epoch},
                "ignore_last_epoch": {"value": args.ignore_last_epoch},
            },
        }
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.wandb_project_name,
            entity="hrushikeshloya",
        )

    epoch_intervals = np.array(
        [-np.inf]
        + np.linspace(
            args.start_time - math.log(28, 10),
            args.end_time - math.log(28, 10),
            num_epochs - 1,
        ).tolist()
        + [np.inf],
        dtype="float64",
    )
    epoch_intervals_pow = np.power(10, epoch_intervals)

    ## Load the data
    (
        masked_trees_index,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom,
        num_trees,
        mask_dodgy,
    ) = load_data(args)

    if args.gamma000 is None:
        wandb.agent(sweep_id, main, count=100)
    else:
        main(args)
# python em_true_ancient_sim_subsampled.py --start_time 5 --end_time 7 --num_epochs 4 -k 2 --path ../sims/stdpopsim_ancient_small/relate_trees_force_100/ --trees stdpopsim_homsap_conv --chrs 1,2,3,4,5 --sample_id 206 --ignore_first_epoch True --mode real --relate_trees True --masking_thresh 0.5 --window_size 10000 --output bayesopt/stdpopsim_homsap -i 1 --rec ../msprime_maps/genetic_map_GRCh37
# python bayesopt.py --start_time 5 --end_time 7 --path ../sims/stdpopsim_ancient_small/relate_trees_force_100/ --trees stdpopsim_homsap_conv --chrs 1,2,3,4,5 --tree_stats_file_name bayesopt/stdpopsim_homsap_tree_stats_206_10000_True_1,2,3,4,5_0.5.pkl  --fixed_params_file_name bayesopt/stdpopsim_homsap_fixed_params_206_10000_True_1,2,3,4,5_0.5.pkl  --sample_id 206 --ignore_first_epoch True --gamma000 1e-4 --gamma001 1e-4 --gamma002 1e-4 --gamma003 1e-4 --gamma010 1e-4 --gamma011 1e-4 --gamma012 1e-4 --gamma013 1e-4 --gamma100 1e-4 --gamma101 1e-4 --gamma102 1e-4 --gamma103 1e-4 --gamma110 1e-4 --gamma111 1e-4 --gamma112 1e-4 --gamma113 1e-4 --tau0 0.5 --tau1 0.5
