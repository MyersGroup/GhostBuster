import pandas as pd
import numpy as np
import matplotlib as mpl
from tqdm import tqdm

mpl.use("Agg")
import math
import time
from collections import Counter
import tskit
from sklearn.calibration import calibration_curve
import argparse
import pickle
import copy
import warnings
import os


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    "-window_size",
    "--window_size",
    help="Window size to subsample the trees",
    type=int,
    default=50000,
)
parser.add_argument(
    "-relate_trees",
    "--relate_trees",
    help="Do you wish to work with Relate trees",
    type=boolean,
    default=True,
)
parser.add_argument("-r", "--rec", help="Filename of rec maps.", type=str)
parser.add_argument(
    "-plot_int",
    "--plot_intermediate_gammas",
    help="Plotting gammas for each iteration",
    type=boolean,
    default=False,
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
    default=0.8,
)
parser.add_argument(
    "-verbose",
    "--verbose",
    help="Print intermediate results",
    type=boolean,
    default=True,
)
parser.add_argument(
    "-init_at_truth",
    "--init_at_truth",
    help="Do you wish to initialize at ground-truth",
    type=boolean,
    default=False,
)
parser.add_argument(
    "-load_gamma",
    "--load_gamma",
    help="Starting gamma values written in a file",
    type=str,
    default=None,
)
parser.add_argument(
    "-load_props",
    "--load_props",
    help="Starting cluster proportions (input space seperated numbers)",
    nargs="+",
    type=float,
    default=None,
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
    "-evaluate_local_ancestry",
    "--evaluate_local_ancestry",
    help="Evaluate the local ancestry on all the trees",
    type=boolean,
    default=True,
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
parser.add_argument(
    "-props_per_chrs",
    "--props_per_chrs",
    help="Assume different proportions across chromosomes",
    type=boolean,
    default=False,
)
args = parser.parse_args()


epoch_intervals = np.array(
    [-np.inf]
    + np.linspace(
        args.start_time - math.log(28, 10), args.end_time - math.log(28, 10), 9
    ).tolist()
    + [np.inf],
    dtype="float64",
)
epoch_intervals_pow = np.power(10, epoch_intervals)

# path = "/well/myers/users/ooz218/workspace/MixedAncestryCoalescenceRates/sim_debug/transfer/transfer/input/"
# path="/data/smew1/speidel/genomics/relate_analyses/MixedCoalRates/stdpopsim_homsap/"
path = args.path


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


def make_ground_truth(ts_list, num_trees, window_size, sample=None, chrs=None):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")

    true_assignment_chr = []
    true_assignment_bp = []
    true_assignment_group = []

    for sample_no, ind in enumerate(sample):
        with open(path + "/assignment.txt", "r") as fp:

            line = fp.readline().split()
            line = fp.readline().split()
            while line:
                true_assignment_chr.append(str(line[0]))
                true_assignment_bp.append(int(line[1]))
                true_assignment_group.append(line[ind + 2])
                line = fp.readline().split()

        true_assignment_group = pd.Series(true_assignment_group).astype("category")
        true_assignment_group = true_assignment_group.cat.codes

        ground_truth_membership_one_hot = np.zeros(
            (max(true_assignment_group) + 1, len(sample) * num_trees)
        )
        count = 0
        num_tree = 0
        for chr in chrs:
            ts = ts_list[count]
            count += 1
            tree = ts.first()
            prev_interval = tree.interval[0]
            for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
                if tree.interval[1] >= prev_interval + window_size:
                    prev_interval = prev_interval + window_size

                    assigned = False
                    for j in range(0, len(true_assignment_bp)):
                        # print([tree.interval, true_assignment_bp[j], true_assignment_bp[j+1], true_assignment_chr[j+1] == str(chr), chr])
                        if true_assignment_chr[j] != str(chr):
                            continue
                        if true_assignment_chr[j] == str(chr) and j + 1 == len(
                            true_assignment_bp
                        ):
                            ground_truth_membership_one_hot[
                                true_assignment_group[j],
                                sample_no * num_trees + num_tree,
                            ] = 1
                            assigned = True
                            break
                        if true_assignment_chr[j] == str(chr) and true_assignment_chr[
                            j + 1
                        ] != str(chr):
                            ground_truth_membership_one_hot[
                                true_assignment_group[j],
                                sample_no * num_trees + num_tree,
                            ] = 1
                            assigned = True
                            break
                        if (
                            true_assignment_chr[j] == str(chr)
                            and tree.interval[0] < true_assignment_bp[j + 1]
                            and tree.interval[1] >= true_assignment_bp[j]
                        ):
                            # print(true_assignment_group[j])
                            ground_truth_membership_one_hot[
                                true_assignment_group[j],
                                sample_no * num_trees + num_tree,
                            ] = 1
                            assigned = True
                            break
                    if assigned == False:
                        print(chr, tree.interval)

                    num_tree += 1
                tree.next()

    print("Done in " + str(time.time() - start_time))
    return ground_truth_membership_one_hot


def fixed_parameters(
    ts_list, poplabels, unique_groups, num_trees, window_size, sample_list
):
    eps = 1e-20
    num_samples = len(list(ts_list[0].first().samples()))
    coal_count = np.zeros(
        (
            len(unique_groups),
            len(epoch_intervals_pow) - 1,
            num_trees * len(sample_list),
        ),
        dtype="float64",
    )
    opportunity = np.zeros(
        (
            len(unique_groups),
            len(epoch_intervals_pow) - 1,
            num_trees * len(sample_list),
        ),
        dtype="float64",
    )
    proportion_of_coalescing_all = []
    epoch_index_all = []
    count_mut_trees = -1
    group_id = {}
    for u in range(len(unique_groups)):
        group_id[unique_groups[u]] = u

    for sample_no, target_seq_ in enumerate(sample_list):
        for ts in ts_list:
            tree = ts.first()
            prev_interval = tree.interval[0]
            for tid in tqdm(range(len(list(ts.trees())))):  # len(list(ts.trees()))
                if tree.interval[1] < prev_interval + window_size:
                    tree.next()
                    continue
                prev_interval = prev_interval + window_size
                count_mut_trees += 1
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
                            [int(mapping[a]), int(mapping[b]), int(mapping[c]), t]
                        )
                coal_events_matrix = np.array(coal_events_matrix, dtype="float64")
                coal_events_matrix = coal_events_matrix[
                    coal_events_matrix[:, 3].argsort()
                ]  ## sorting based on coalescene times
                lineage_content = np.zeros(
                    (2 * num_samples - 1, len(unique_groups)), dtype="float64"
                )
                target_seq = target_seq_
                for m in range(len(poplabels)):
                    lineage_content[
                        2 * m : 2 * m + 2, group_id[poplabels.GROUP.loc[m]]
                    ] = 1
                lineage_content[
                    target_seq
                ] = 0  ## setting lineage content of target sequence = 0
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
                        & (coal_events_matrix[:, 3] < epoch_intervals_pow[epoch + 1])
                    ]
                    tprev = epoch_intervals_pow[epoch]
                    for (a, b, c, t) in coal_events_submatrix:
                        event_count += 1
                        a = int(a)
                        b = int(b)
                        c = int(c)
                        opportunity[:, epoch, count_mut_trees] += (t - tprev) * (
                            prev_branch_length
                        )
                        if a == target_seq:
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
                            prev_branch_length = prev_branch_length - lineage_content[
                                b
                            ] / (sum(lineage_content[b]))
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
                            prev_branch_length = prev_branch_length - lineage_content[
                                a
                            ] / (sum(lineage_content[a]))
                        else:
                            lineage_content[c] = lineage_content[a] + lineage_content[b]

                            if (
                                sum(lineage_content[a]) == 0
                                or sum(lineage_content[b]) == 0
                            ):
                                print(tree.interval)
                                print(a, b)
                                print(lineage_content[a])
                                print(lineage_content[b])
                            prev_branch_length = (
                                prev_branch_length
                                - lineage_content[a] / (sum(lineage_content[a]))
                                - lineage_content[b] / (sum(lineage_content[b]))
                                + lineage_content[c] / (sum(lineage_content[c]))
                            )
                        lineage_content[a] = 0
                        lineage_content[b] = 0
                        tprev = t
                    if epoch < len(epoch_intervals_pow) - 2:
                        opportunity[:, epoch, count_mut_trees] += (
                            epoch_intervals_pow[epoch + 1] - tprev
                        ) * (prev_branch_length)
                    if (event_count == num_samples - 1) and epoch <= len(
                        epoch_intervals_pow
                    ) - 2:
                        opportunity[:, epoch + 1 :, count_mut_trees] = 0.0
                        break
                proportion_of_coalescing_all.append(proportion_of_coalescing_in_tree)
                epoch_index_all.append(epoch_index_in_tree)
                tree.next()

    ## correcting opportunity for ancestral samples
    for epoch in range(len(epoch_intervals_pow) - 1):
        for m in range(len(poplabels)):
            if poplabels.SAMPLING_TIME.loc[m] >= epoch_intervals_pow[epoch + 1]:
                opportunity[group_id[poplabels.GROUP.loc[m]], epoch, :] -= (
                    epoch_intervals_pow[epoch + 1] - epoch_intervals_pow[epoch]
                )
            elif (
                poplabels.SAMPLING_TIME.loc[m] > epoch_intervals_pow[epoch]
                and poplabels.SAMPLING_TIME.loc[m] < epoch_intervals_pow[epoch + 1]
            ):
                opportunity[group_id[poplabels.GROUP.loc[m]], epoch, :] -= (
                    poplabels.SAMPLING_TIME.loc[m] - epoch_intervals_pow[epoch]
                )

    return coal_count, opportunity, proportion_of_coalescing_all, epoch_index_all


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


def compute_tree_stats(ts_list, chrs, window_size):
    num_trees = 0
    tree_size = []
    no_of_mutations = []
    tmrca = []
    recomb_rates = []
    frac_branches_with_snp = []
    num_snps_on_tree = []
    fraction_snps_not_mapping = []
    count = 0
    for chr in chrs:
        print(chr)
        recomb_map = pd.read_csv(
            # "/well/myers/speidel/SharedWithHrushi/stdpopsim_Han"
            # "/camp/lab/skoglundp/working/leo/datasets/human_genome/recomb_maps/HapmapII/genetic_map_GRCh37_chr"
            # + "/recomb_maps/msprime_maps/genetic_map_GRCh37_chr"
            args.rec + "_chr" + str(chr) + ".txt",
            sep="\t",
        )
        recomb_map_arr = np.array(recomb_map[recomb_map.columns[1:]])
        recomb_map["Start Position(bp)"] = np.array(
            [0] + recomb_map_arr[:-1, 0].tolist()
        )
        relate_quality_output = pd.read_csv(
            path + args.trees + "_chr" + str(chr) + ".qual",
            sep=" ",
        )
        ts = ts_list[count]
        count += 1
        tree = ts.first()
        # prev_interval = 0
        prev_interval = tree.interval[0]
        i = 0
        for tid in range(ts.num_trees):  # len(list(ts.trees()))
            if tree.interval[1] >= prev_interval + window_size:
                prev_interval = prev_interval + window_size
                recomb_events = recomb_map[
                    ~(
                        (recomb_map["Start Position(bp)"] > tree.interval[1])
                        | (recomb_map["Position(bp)"] < tree.interval[0])
                    )
                ]
                recomb_rates.append(np.mean(recomb_events["Rate(cM/Mb)"]))

                num_trees += 1
                i += 1
                tree_size.append(tree.interval[1] - tree.interval[0])
                no_of_mutations.append(
                    tree.num_mutations
                )  ###changed to mutations from sites
                tmrca.append(tree.time(tree.root))
                relate_quality = relate_quality_output[
                    (relate_quality_output.pos < tree.interval[1])
                    & (relate_quality_output.pos >= tree.interval[0])
                ].iloc[0]
                frac_branches_with_snp.append(relate_quality["frac_branches_with_snp"])
                num_snps_on_tree.append(relate_quality["num_snps_on_tree"])
                fraction_snps_not_mapping.append(
                    relate_quality["fraction_snps_not_mapping"]
                )

            tree.next()

        del tree
        del ts
    return (
        tree_size,
        no_of_mutations,
        tmrca,
        recomb_rates,
        frac_branches_with_snp,
        num_snps_on_tree,
        fraction_snps_not_mapping,
    )


def mask_for_dodgy_trees(frac_branches_with_snp, num_snps_on_tree, masking_thresh):
    # print(np.percentile(frac_branches_with_snp, masking_thresh * 100))
    # print(np.percentile(num_snps_on_tree, masking_thresh * 100))
    mask = (
        frac_branches_with_snp
        > np.percentile(frac_branches_with_snp, masking_thresh * 100)
    ) & (num_snps_on_tree > np.percentile(num_snps_on_tree, masking_thresh * 100))
    return mask


def write_coal(gamma_arr, filename, labs, is_relate):

    if is_relate:
        filename = "RelateTrees_" + filename
    else:
        filename = "TrueTrees_" + filename

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


def main(args, plot=False, gamma_arr=None):
    print("Considering sample ids: " + str(args.sample_id))

    ### Add all input assertions here:
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
    start_time = time.time()
    num_clusters = args.num_clusters
    poplabels = pd.read_csv(path + "poplabels.txt", sep=" ")
    unique_groups = np.unique(poplabels.GROUP)

    ts_list = []
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))
    tree_stats_file_name = (
        "tree_stats_"
        + str(args.sample_id)
        + "_"
        + str(args.window_size)
        + "_"
        + str(args.relate_trees)
        + "_"
        + str(args.chrs)
        + "_"
        + str(args.masking_threshold)
        + ".pkl"
    )
    fixed_params_file_name = (
        "fixed_params_"
        + str(args.sample_id)
        + "_"
        + str(args.window_size)
        + "_"
        + str(args.relate_trees)
        + "_"
        + str(args.chrs)
        + "_"
        + str(args.masking_threshold)
        + ".pkl"
    )
    if args.trees == None and (
        not os.path.isfile(tree_stats_file_name)
        or not os.path.isfile(fixed_params_file_name)
    ):
        raise ValueError(
            "Specify the location to the trees under the args.trees argument"
        )

    if args.trees != None:
        for chr in chrs:
            ts = tskit.load(
                path + args.trees + "_chr" + str(chr) + ".trees"
            )  ## relate trees
            ts_list.append(ts)
        if len(poplabels) != ts_list[0].num_samples / 2:  ## only valid for diploid
            raise ValueError(
                "Number of samples in trees doesnt match number of samples in poplabels.txt"
            )

    num_samples = len(list(ts_list[0].first().samples()))
    for sample in args.sample_id:
        if sample >= num_samples or sample < 0:
            raise ValueError("The sample ids are out of range")

    filename = ".treepos"
    if args.relate_trees:
        filename = "Relate" + filename
    else:
        filename = "True" + filename
    f = open(filename, "w")

    if args.trees != None:
        trees_per_chr = []
        num_trees = 0
        for ts in ts_list:
            tree = ts.first()
            prev_interval = tree.interval[0]
            # prev_interval = 0
            start_pos = copy.deepcopy(num_trees)
            for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
                if tree.interval[1] >= prev_interval + args.window_size:
                    f.write(str(tree.interval[0]) + " " + str(tree.interval[1]) + "\n")
                    prev_interval = prev_interval + args.window_size
                    num_trees += 1
                tree.next()
            trees_per_chr.append((start_pos, num_trees))
        f.close()
        print("Total number of trees = " + str(num_trees))

    if args.relate_trees:
        try:
            f_pkl = open(tree_stats_file_name, "rb")
            (
                num_trees,
                trees_per_chr,
                tree_size,
                no_of_mutations,
                tmrca,
                recomb_rates,
                frac_branches_with_snp,
                num_snps_on_tree,
                fraction_snps_not_mapping,
                mask_dodgy,
            ) = pickle.load(f_pkl)
            f_pkl.close()
            print("Done loading tree statistics from: " + str(tree_stats_file_name))
        except:
            print("Tree statistics file not found, calculating tree statistics..")
            (
                tree_size,
                no_of_mutations,
                tmrca,
                recomb_rates,
                frac_branches_with_snp,
                num_snps_on_tree,
                fraction_snps_not_mapping,
            ) = compute_tree_stats(ts_list, chrs=chrs, window_size=args.window_size)
            mask_dodgy = mask_for_dodgy_trees(
                frac_branches_with_snp * len(args.sample_id),
                num_snps_on_tree * len(args.sample_id),
                args.masking_threshold,
            )

            f_pkl = open(tree_stats_file_name, "wb")
            pickle.dump(
                [
                    num_trees,
                    trees_per_chr,
                    tree_size,
                    no_of_mutations,
                    tmrca,
                    recomb_rates,
                    frac_branches_with_snp,
                    num_snps_on_tree,
                    fraction_snps_not_mapping,
                    mask_dodgy,
                ],
                f_pkl,
            )
            f_pkl.close()
            print("Tree statistics stored in: " + str(tree_stats_file_name))

    else:
        mask_dodgy = np.ones(
            num_trees * len(args.sample_id), dtype=bool
        )  ## No masking needed for true trees

    print("Trees with high certainty = " + str(np.sum(mask_dodgy)))
    masked_trees_index = np.arange(0, num_trees * len(args.sample_id))[mask_dodgy]

    if args.props_per_chrs:
        trees_per_chr_masked = []
        for (start, end) in trees_per_chr:
            start_in_masked = len(masked_trees_index) - len(
                masked_trees_index[masked_trees_index >= start]
            )
            end_in_masked = len(masked_trees_index) - len(
                masked_trees_index[masked_trees_index >= end]
            )
            trees_per_chr_masked.append(
                (start_in_masked, end_in_masked)
            )  ## [start, end)

    if args.relate_trees:

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
            print("Done loading fixed parameters from: " + str(fixed_params_file_name))

        except:
            print("Fixed parameters file not found, calculating fixed parameters..")
            if args.mode == "sim":
                ground_truth_membership = make_ground_truth(
                    ts_list,
                    num_trees,
                    window_size=args.window_size,
                    sample=args.sample_id,
                    chrs=chrs,
                )
            (
                num,
                denom,
                proportion_of_coalescing_all,
                epoch_index_all,
            ) = fixed_parameters(
                ts_list,
                poplabels,
                unique_groups,
                num_trees,
                args.window_size,
                args.sample_id,
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

    else:
        ground_truth_membership = make_ground_truth(
            ts_list,
            num_trees,
            window_size=args.window_size,
            sample=args.sample_id,
            chrs=chrs,
        )
        num, denom, proportion_of_coalescing_all, epoch_index_all = fixed_parameters(
            ts_list,
            poplabels,
            unique_groups,
            num_trees,
            args.window_size,
            args.sample_id,
        )

    if (denom < -1e-8).any():
        raise ValueError(
            "The opportunity has negative values, check the sampling times in poplabels.txt"
        )

    ### Clipping the opportunity to zero (because there might be some very small -ve values cause of numerical instabilities)
    denom = copy.deepcopy(np.maximum(denom, 0))

    if args.init_at_truth:
        own_membership = ground_truth_membership[:, mask_dodgy]
    else:
        own_membership = np.array(
            np.random.dirichlet(
                np.ones(num_clusters), len(args.sample_id) * num_trees
            ).T,
            dtype="float64",
        )[:, mask_dodgy]

    if args.load_gamma:
        gamma_arr = np.load(args.load_gamma)
    if args.load_props:
        tau = args.load_props

    if args.evaluate_gamma:
        log_likelihood_arr = []
        start_time_em = time.time()
        print("Starting the EM..")
        for epoch in range(args.num_iters):
            ## M-step
            start_m_time = time.time()
            gamma_arr = np.zeros(
                (len(own_membership), len(unique_groups), len(epoch_intervals) - 1),
                dtype="float64",
            )
            for j in range(len(own_membership)):
                if epoch == 0:
                    n = compute_gamma_num(
                        own_membership[j],
                        None,
                        proportion_of_coalescing_all,
                        epoch_index_all,
                        len(unique_groups),
                        masked_trees_index,
                    )
                else:
                    n = compute_gamma_num(
                        own_membership[j],
                        prev_gamma[j],
                        proportion_of_coalescing_all,
                        epoch_index_all,
                        len(unique_groups),
                        masked_trees_index,
                    )
                for i in range(len(unique_groups)):
                    d = compute_gamma_denom(own_membership[j], denom[i], mask_dodgy)
                    gamma_arr[j][i] = copy.deepcopy(n[i] / d)  # n/d #

            tau = np.mean(own_membership, axis=1)
            if args.props_per_chrs:
                tau = np.zeros((len(trees_per_chr_masked), len(own_membership)))
                for chr, (start, end) in enumerate(trees_per_chr_masked):
                    tau[chr] = np.mean(own_membership[:, start:end], axis=1)

            if epoch == 0 and args.load_gamma != None and args.load_props != None:
                print("Using initial gamma specified in file: " + str(args.load_gamma))
                gamma_arr = np.load(args.load_gamma)
                tau = args.load_props  ### load taus only works for not(props_per_chrs)

            assert (gamma_arr >= 0).all()
            prev_gamma = copy.deepcopy(gamma_arr)

            if args.verbose:
                print(gamma_arr)
                print("Iter" + str(epoch))
                print(tau)
            # print("m-step: " + str(time.time() - start_m_time))
            ## E-step

            start_e_time = time.time()
            own_membership_update = np.ones(
                (len(own_membership), int(np.sum(mask_dodgy))), dtype="float64"
            )
            log_num_em = np.zeros(
                (len(own_membership), int(np.sum(mask_dodgy))), dtype="float64"
            )
            log_denom_em = np.zeros(
                (len(own_membership), int(np.sum(mask_dodgy))), dtype="float64"
            )
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

            if not args.props_per_chrs:
                for j in range(len(own_membership)):
                    own_membership_update[j] *= tau[j]
            else:
                for chr, (start, end) in enumerate(trees_per_chr_masked):
                    for j in range(len(own_membership)):
                        own_membership_update[j, start:end] *= tau[chr, j]

            log_likelihood = np.sum(
                np.log(np.sum(own_membership_update, axis=0))
                + np.max(log_num_em + log_denom_em, axis=0)
            )
            log_likelihood_arr.append(log_likelihood)
            own_membership = own_membership_update / (
                np.sum(own_membership_update, axis=0)
            )
            membership_thresh = make_one_hot(
                np.argmax(own_membership, axis=0), len(own_membership)
            )
            # print("e-step: " + str(time.time() - start_e_time))

            if args.mode == "sim":
                ## Evaluate accuracy
                acc_arr = np.zeros(
                    (len(own_membership), len(ground_truth_membership[:, mask_dodgy]))
                )
                for i in range(len(own_membership)):
                    for j in range(0, len(ground_truth_membership[:, mask_dodgy])):
                        acc = np.sum(
                            membership_thresh[i]
                            == ground_truth_membership[j, mask_dodgy]
                        )
                        acc_arr[i][j] = acc
                overall_acc = (
                    np.sum(np.max(acc_arr, axis=1))
                    / len(membership_thresh)
                    / len(membership_thresh[0])
                )
                print(
                    "Sample = "
                    + str(args.sample_id)
                    + " Accuracy = "
                    + str(overall_acc)
                )

            ## Tree stats
            if args.verbose and args.relate_trees:
                proportion_of_coalescing_top2 = np.zeros(
                    (len(args.sample_id) * num_trees, 2, len(unique_groups))
                )
                for tid in range(len(args.sample_id) * num_trees):
                    proportion_of_coalescing_top2[
                        tid, :, :
                    ] = proportion_of_coalescing_all[tid][0:2]
                recomb_x_all = []
                recomb_y_all = []
                size_x_all = []
                size_y_all = []
                muts_x_all = []
                muts_y_all = []
                frac_branch_x_all = []
                num_snps_x_all = []
                frac_snps_x_all = []
                for i in range(len(own_membership)):
                    tree_size_i = np.array(len(args.sample_id) * tree_size)[mask_dodgy][
                        np.argmax(own_membership, axis=0) == i
                    ]
                    num_mutations_i = np.array(len(args.sample_id) * no_of_mutations)[
                        mask_dodgy
                    ][np.argmax(own_membership, axis=0) == i]
                    recomb_rates_i = np.array(len(args.sample_id) * recomb_rates)[
                        mask_dodgy
                    ][np.argmax(own_membership, axis=0) == i]
                    recomb_rates_i = recomb_rates_i[~np.isnan(np.array(recomb_rates_i))]
                    frac_branches_with_snp_i = np.array(
                        len(args.sample_id) * frac_branches_with_snp
                    )[mask_dodgy][np.argmax(own_membership, axis=0) == i]
                    num_snps_on_tree_i = np.array(
                        len(args.sample_id) * num_snps_on_tree
                    )[mask_dodgy][np.argmax(own_membership, axis=0) == i]
                    fraction_snps_not_mapping_i = np.array(
                        len(args.sample_id) * fraction_snps_not_mapping
                    )[mask_dodgy][np.argmax(own_membership, axis=0) == i]
                    frac_branch_x_all.extend(frac_branches_with_snp_i)
                    num_snps_x_all.extend(num_snps_on_tree_i)
                    frac_snps_x_all.extend(fraction_snps_not_mapping_i)
                    recomb_x_all.extend(recomb_rates_i)
                    size_x_all.extend(tree_size_i)
                    muts_x_all.extend(num_mutations_i)
                    recomb_y_all.extend(np.repeat(i, len(recomb_rates_i)))
                    size_y_all.extend(np.repeat(i, len(tree_size_i)))
                    muts_y_all.extend(np.repeat(i, len(num_mutations_i)))
                    proportion_of_coalescing_i = proportion_of_coalescing_top2[
                        mask_dodgy
                    ][np.argmax(own_membership, axis=0) == i]
                    print(
                        "Cluster: "
                        + str(i)
                        + " Median tree size: "
                        + str(np.median(tree_size_i))
                        + " Median num of muts: "
                        + str(np.median(num_mutations_i))
                        + " Median recomb rate: "
                        + str(np.median(recomb_rates_i))
                        + " Median frac_branches_with_snp: "
                        + str(np.median(frac_branches_with_snp_i))
                        + " Median num_snps_on_tree: "
                        + str(np.median(num_snps_on_tree_i))
                        + " Median fraction_snps_not_mapping: "
                        + str(np.median(fraction_snps_not_mapping_i))
                    )
                    print(
                        " Mean 1st coal. proportion: "
                        + str(np.mean(proportion_of_coalescing_i[:, 0, :], axis=0))
                        + " Mean 2nd coal. proportion: "
                        + str(np.mean(proportion_of_coalescing_i[:, 1, :], axis=0))
                    )

            ## Gamma plots
            if args.plot_intermediate_gammas:
                write_coal(
                    gamma_arr,
                    "stdpopsim_" + str(args.sample_id) + "_iter" + str(epoch) + ".coal",
                    unique_groups,
                    args.relate_trees,
                )
                with open(
                    "gamma_" + str(args.sample_id) + "_iter" + str(epoch) + ".npy", "wb"
                ) as f:
                    np.save(f, gamma_arr)

                filename = "membership_" + str(args.sample_id) + ".npy"
                if args.relate_trees:
                    filename = "RelateTrees_" + filename
                else:
                    filename = "TrueTrees_" + filename
                with open(filename, "wb") as f:
                    np.save(f, own_membership)

            ## Early-stopping
            print("log-likelihood = " + str(log_likelihood_arr[-1]), flush=True)
            # if epoch > 100: ##min-iters = 100
            #    if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.00001:
            #        break ## stop if log-likelihood isn't changing much

        if (
            np.abs(
                (log_likelihood_arr[-1] - log_likelihood_arr[-2])
                / log_likelihood_arr[-2]
            )
            > 0.001
        ):
            warnings.warn(
                "The log-likelihood is still increasing, you should consider running for longer"
            )

        print(
            "Sample = "
            + str(args.sample_id)
            + " Epochs = "
            + str(epoch)
            + " Total time = "
            + str(time.time() - start_time)
            + " EM time = "
            + str(time.time() - start_time_em)
        )

        ## gamma plots
        write_coal(
            gamma_arr,
            "stdpopsim_" + str(args.sample_id) + ".coal",
            unique_groups,
            args.relate_trees,
        )

        if args.mode == "sim":
            ## Calculate the calibration on the rich-trees
            mapping = np.argmax(acc_arr, axis=1)
            for i in range(len(own_membership)):
                y, x = calibration_curve(
                    ground_truth_membership[mapping[i], mask_dodgy],
                    own_membership[i],
                    n_bins=20,
                )
                print("Calibration x: " + str(x))
                print("Calibration y: " + str(y))

    if args.evaluate_local_ancestry:
        ## Final local ancestry inference on all trees
        own_membership_update = np.ones(
            (len(own_membership), len(args.sample_id) * num_trees), dtype="float64"
        )

        log_num_em = np.zeros(
            (len(own_membership), len(args.sample_id) * num_trees), dtype="float64"
        )
        log_denom_em = np.zeros(
            (len(own_membership), len(args.sample_id) * num_trees), dtype="float64"
        )
        for tid in range(len(args.sample_id) * num_trees):
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
                )
                log_num_em[j, tid] = log_num_em_j
                log_denom_em[j, tid] = log_denom_em_j

        own_membership_update = np.exp(
            log_num_em
            + log_denom_em
            - np.repeat(
                np.max(log_num_em + log_denom_em, axis=0).reshape(-1, 1),
                len(own_membership),
                axis=1,
            ).T
        )

        # In-case of highly uncertain trees, only depend on the prior
        own_membership_update = np.nan_to_num(own_membership_update, nan=1)

        if not args.props_per_chrs:
            for j in range(len(own_membership)):
                own_membership_update[j] *= tau[j]
        else:
            for chr, (start, end) in enumerate(trees_per_chr):
                for j in range(len(own_membership)):
                    own_membership_update[j, start:end] *= tau[chr, j]

        log_likelihood = np.sum(
            np.log(np.sum(own_membership_update, axis=0))[mask_dodgy]
            + np.max(log_num_em + log_denom_em, axis=0)[mask_dodgy]
        )

        print("Test log-likelihood: " + str(log_likelihood))

        own_membership = own_membership_update / (np.sum(own_membership_update, axis=0))

        filename = (
            "overall_membership_" + str(args.sample_id) + ".npy"
        )  ## this saves membership for all the trees (without the filtering)
        if args.relate_trees:
            filename = "RelateTrees_" + filename
        else:
            filename = "TrueTrees_" + filename
        with open(filename, "wb") as f:
            np.save(f, own_membership)

    if args.mode == "sim":
        return overall_acc
    else:
        return 0


acc = main(args, plot=False, gamma_arr=None)  ##Han(106), Sardinian(52)
if args.mode == "sim":
    print("Average accuracy = " + str(acc))

# python ../RelateLocalAncestry/em_true_ancient_sim_subsampled.py --chr 1,2 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas True --window_size 0 --sample_id 0
