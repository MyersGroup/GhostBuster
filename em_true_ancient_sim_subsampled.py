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
from sklearn.metrics import silhouette_score
import argparse
import pickle
import copy
import warnings
import os
from pathlib import Path
import scipy.stats as stats


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
    default=0,
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
    "-load_membership",
    "--load_membership",
    help="Starting gamma values written in a file",
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
    "-load_mask",
    "--load_mask",
    help="Load precomputed mask file (0/1s of length number of trees)",
    type=str,
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
    "-num_epochs",
    "--num_epochs",
    help="Num epochs",
    type=int,
    default=10,
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
parser.add_argument(
    "--check_muts_target",
    help="check mutations on target lineage: criterion to choose trees",
    type=boolean,
    default=False,
)
args = parser.parse_args()

np.random.seed(2)  ## fix the random seed

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


def lineage_nodes(tree, sample_ids):
    assert sample_ids is not None  ## shouldn't be going here
    num_samples = len(set(tree.samples()))
    assert (np.array(sample_ids) < num_samples).all()
    out = []
    for sample_id in sample_ids:
        parent = sample_id
        out.append(parent)
        while parent != tree.root:
            parent = tree.parent(parent)
            relate_format = num_samples + (parent - num_samples) % (num_samples - 1)
            out.append(relate_format)
    return out


def count_num_muts(mut_t, nodes):
    mut_t_b = mut_t.branchID
    focal_mutations = np.isin(mut_t_b, np.fromiter(nodes, mut_t_b.dtype))
    return np.sum(mut_t.num_muts[focal_mutations])


def count_lineage_branch_has_muts(mut_t, nodes):
    mut_t = mut_t[mut_t.num_muts > 0].branchID
    focal_mutations = np.isin(mut_t, np.fromiter(nodes, mut_t.dtype))
    return np.sum(focal_mutations) / (len(nodes) - 1)


def get_multinomial_frac_branches(mut_t, nodes, mut_rate=1e-6, num_samples=10000):
    mut_t_b = mut_t.branchID
    focal_mutations = np.isin(mut_t_b, np.fromiter(nodes, mut_t_b.dtype))
    opportunity = (
        mut_rate * mut_t.dist[focal_mutations] * mut_t.branch_length[focal_mutations]
    )
    # return np.sum(opportunity)
    opportunity /= np.sum(opportunity)
    opportunity = np.maximum(opportunity, 1e-8)
    opportunity /= np.sum(opportunity)
    num_muts = np.sum(mut_t.num_muts[focal_mutations])
    assert opportunity.ndim == 1
    # assert np.abs(opportunity.sum() - 1) < 1e-6
    if num_muts == 0:
        return num_samples
    rv = stats.multinomial(num_muts, opportunity.tolist())
    num_zero_branches = np.sum(rv.rvs(size=num_samples) == 0, axis=1)  # num_samples x 1
    rank = np.sum(num_zero_branches < np.sum(mut_t.num_muts[focal_mutations] == 0))
    return rank


def samples_below(tree, node):
    out = []
    if node in list(tree.samples()):
        out.append(node)
    else:
        out.extend(samples_below(tree, tree.children(node)[0]))
        out.extend(samples_below(tree, tree.children(node)[1]))
    return out


def nodes_to_keep(tree, sample_ids):
    ## returns list of nodes which are to appear in ts.simplify(sample_ids)
    out = sample_ids
    num_samples = len(set(tree.samples()))
    for node in tree.nodes():
        if tree.children(node) != ():
            left_samples_below = samples_below(tree, tree.children(node)[0])
            right_samples_below = samples_below(tree, tree.children(node)[1])
            if (
                np.isin(left_samples_below, sample_ids).any()
                and np.isin(right_samples_below, sample_ids).any()
            ):
                out.append(node)
                # out.append(num_samples + (node - num_samples) % (num_samples - 1))
    return out


# def make_ground_truth(ts_list, num_trees, window_size, sample=None, chrs=None):
#     ## Extracts the ground truth membership from the simulations
#     start_time = time.time()
#     print("Calculating the ground truth local ancestry..")

#     true_assignment_chr = []
#     true_assignment_bp = []
#     true_assignment_group = []

#     for sample_no, ind in enumerate(sample):
#         with open(Path(path) / "assignment.txt") as fp:

#             line = fp.readline().split()
#             line = fp.readline().split()
#             while line:
#                 true_assignment_chr.append(str(line[0]))
#                 true_assignment_bp.append(int(line[1]))
#                 true_assignment_group.append(line[ind + 2])
#                 line = fp.readline().split()

#         true_assignment_group = pd.Series(true_assignment_group).astype("category")
#         true_assignment_group = true_assignment_group.cat.codes

#         ground_truth_membership_one_hot = np.zeros(
#             (max(true_assignment_group) + 1, len(sample) * num_trees)
#         )
#         count = 0
#         num_tree = 0
#         for chr in chrs:
#             ts = ts_list[count]
#             count += 1
#             tree = ts.first()
#             prev_interval = tree.interval[0]
#             for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
#                 if tree.interval[1] >= prev_interval + window_size:
#                     prev_interval = prev_interval + window_size

#                     assigned = False
#                     for j in range(0, len(true_assignment_bp)):
#                         # print([tree.interval, true_assignment_bp[j], true_assignment_bp[j+1], true_assignment_chr[j+1] == str(chr), chr])
#                         if true_assignment_chr[j] != str(chr):
#                             continue
#                         if true_assignment_chr[j] == str(chr) and j + 1 == len(
#                             true_assignment_bp
#                         ):
#                             ground_truth_membership_one_hot[
#                                 true_assignment_group[j],
#                                 sample_no * num_trees + num_tree,
#                             ] = 1
#                             assigned = True
#                             break
#                         if true_assignment_chr[j] == str(chr) and true_assignment_chr[
#                             j + 1
#                         ] != str(chr):
#                             ground_truth_membership_one_hot[
#                                 true_assignment_group[j],
#                                 sample_no * num_trees + num_tree,
#                             ] = 1
#                             assigned = True
#                             break
#                         if (
#                             true_assignment_chr[j] == str(chr)
#                             and tree.interval[0] < true_assignment_bp[j + 1]
#                             and tree.interval[1] >= true_assignment_bp[j]
#                         ):
#                             # print(true_assignment_group[j])
#                             ground_truth_membership_one_hot[
#                                 true_assignment_group[j],
#                                 sample_no * num_trees + num_tree,
#                             ] = 1
#                             assigned = True
#                             break
#                     if assigned == False:
#                         print(chr, tree.interval)

#                     num_tree += 1
#                 tree.next()

#     print("Done in " + str(time.time() - start_time))
#     return ground_truth_membership_one_hot


def make_ground_truth(ts_list, num_trees, window_size, sample=None, chrs=None):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print("Calculating the ground truth local ancestry..")
    ground_truth_membership_one_hot = None

    num_tree = 0
    for sample_no, ind in enumerate(sample):
        count = 0
        for chr in chrs:
            ground_truth = pd.read_csv(
                Path(path)
                / str("local_ancestry_chr" + str(chr) + "_" + str(ind) + ".csv"),
                names=["startpos", "endpos", "dest"],
            )
            if ground_truth_membership_one_hot is None:
                num_ref_groups = int(np.max(ground_truth["dest"]))
                ground_truth_membership_one_hot = np.zeros(
                    (num_ref_groups, int(len(sample) * num_trees))
                )
            tree = ts_list[count].first()
            prev_interval = tree.interval[0]
            for tid in range(len(list(ts_list[count].trees()))):
                if tree.interval[1] >= prev_interval + window_size:
                    prev_interval = prev_interval + window_size
                    if tree.num_sites > 0:
                        for j in range(len(ground_truth)):
                            if ~(
                                (tree.interval[1] < ground_truth["startpos"].loc[j])
                                | (tree.interval[0] > ground_truth["endpos"].loc[j])
                            ):
                                ground_truth_membership_one_hot[
                                    int(ground_truth["dest"].loc[j]) - 1, num_tree
                                ] = 1
                    num_tree += 1
                tree.next()
            count += 1
    print("Done in " + str(time.time() - start_time))
    ## only return ground truth of groups which actually contribute
    print(np.sum(ground_truth_membership_one_hot, axis=1))
    return np.vstack(
        (
            ground_truth_membership_one_hot[
                np.sum(ground_truth_membership_one_hot, axis=1) != 0
            ],
            1 - np.sum(ground_truth_membership_one_hot, axis=0),
        )
    )


def fixed_parameters(
    ts_list,
    poplabels,
    unique_groups,
    num_trees,
    window_size,
    sample_list,
):
    assert [
        s in poplabels[poplabels.INCLUDE == 1].index.tolist() for s in sample_list
    ] == [True] * len(sample_list)
    ### You have removed target samples from the poplabels file
    eps = 1e-20
    num_samples = len(list(ts_list[0].first().samples()))
    num_nodes = len(list(ts_list[0].first().nodes()))
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
        count_mut_trees_prev = copy.deepcopy(count_mut_trees)
        for chr_no, ts in enumerate(ts_list):
            tree = ts.first()
            prev_interval = tree.interval[0]
            for tid in tqdm(range(len(list(ts.trees())))):  # len(list(ts.trees()))
                sample_list_tree = copy.deepcopy(sample_list)
                if tree.interval[1] >= prev_interval + window_size:
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
                        ## Only count lineage content for included samples
                        if poplabels.INCLUDE.iloc[m]:
                            lineage_content[m, group_id[poplabels.GROUP.iloc[m]]] = 1

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

                            elif (a == target_seq and sum(lineage_content[b]) == 0) or (
                                b == target_seq and sum(lineage_content[a]) == 0
                            ):
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
                                    a in sample_list_tree and b not in sample_list_tree
                                ):
                                    prev_branch_length = (
                                        prev_branch_length
                                        - lineage_content[b] / (sum(lineage_content[b]))
                                        + lineage_content[c] / (sum(lineage_content[c]))
                                    )
                                elif (
                                    b in sample_list_tree and a not in sample_list_tree
                                ):
                                    prev_branch_length = (
                                        prev_branch_length
                                        - lineage_content[a] / (sum(lineage_content[a]))
                                        + lineage_content[c] / (sum(lineage_content[c]))
                                    )
                                elif (
                                    a not in sample_list_tree
                                    and b not in sample_list_tree
                                ):
                                    prev_branch_length = (
                                        prev_branch_length
                                        - lineage_content[a] / (sum(lineage_content[a]))
                                        - lineage_content[b] / (sum(lineage_content[b]))
                                        + lineage_content[c] / (sum(lineage_content[c]))
                                    )
                            lineage_content[a] = 0
                            lineage_content[b] = 0
                            tprev = max(t, poplabels.SAMPLING_TIME.iloc[target_seq_])
                        if epoch < len(epoch_intervals_pow) - 2:
                            opportunity[:, epoch, count_mut_trees] += (
                                max(
                                    epoch_intervals_pow[epoch + 1],
                                    poplabels.SAMPLING_TIME.iloc[target_seq_],
                                )
                                - max(tprev, poplabels.SAMPLING_TIME.iloc[target_seq_])
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


def compute_tree_stats(
    ts_list, chrs, window_size, check_muts_target_name, sample_list=None
):
    num_trees = 0
    tree_size = []
    no_of_mutations = []
    tmrca = []
    recomb_rates = []
    rank_zero_snp_branches_target = []
    frac_branches_with_snp_target = []
    frac_branches_with_snp = []
    num_snps_on_tree = []
    count = 0
    recomb_window_size = 10000  ## window size for measure recombination rates
    num_nodes = len(list(ts_list[0].first().nodes()))
    first_tree_nodes = list(ts_list[0].first().nodes())[0:-1]
    for chr_no, chr in enumerate(chrs):
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
        # relate_quality_output = pd.read_csv(
        #     Path(path) / str(args.trees + "_chr" + str(chr) + ".qual"),
        #     sep=" ",
        # )
        if check_muts_target_name is not None:
            relate_allmuts_file = pd.read_csv(
                check_muts_target_name[chr_no],
                sep=" ",
                engine="c",
            )
        ts = ts_list[count]
        count += 1
        tree = ts.first()
        prev_interval = tree.interval[0]
        i = 0
        start_idx = 0
        for tid in tqdm(range(ts.num_trees)):  # len(list(ts.trees()))
            if tree.interval[1] >= prev_interval + window_size:
                prev_interval = prev_interval + window_size
                recomb_events = recomb_map[
                    ~(
                        (
                            recomb_map["Start Position(bp)"]
                            > tree.interval[1] + recomb_window_size
                        )
                        | (
                            recomb_map["Position(bp)"]
                            < tree.interval[0] - recomb_window_size
                        )
                    )
                ]
                recomb_rates.append(np.mean(recomb_events["Rate(cM/Mb)"]))
                num_trees += 1
                i += 1
                tree_size.append(tree.interval[1] - tree.interval[0])
                no_of_mutations.append(tree.num_mutations)
                tmrca.append(tree.time(tree.root))
                if check_muts_target_name is not None:
                    relate_allmuts_tree = relate_allmuts_file.iloc[
                        tid * num_nodes : (tid + 1) * num_nodes
                    ]
                    rank_zero_snp_branches_target.append(
                        get_multinomial_frac_branches(
                            relate_allmuts_tree, lineage_nodes(tree, sample_list)
                        )
                    )
                    frac_branches_with_snp_target.append(
                        count_lineage_branch_has_muts(
                            relate_allmuts_tree, lineage_nodes(tree, sample_list)
                        )
                    )
                    frac_branches_with_snp.append(
                        count_lineage_branch_has_muts(
                            relate_allmuts_tree, first_tree_nodes
                        )
                    )
                    num_snps_on_tree.append(
                        count_num_muts(relate_allmuts_tree, first_tree_nodes)
                    )
                else:
                    rank_zero_snp_branches_target.append(0)
                    frac_branches_with_snp_target.append(0)
                    frac_branches_with_snp.append(0)
                    num_snps_on_tree.append(0)

                # relate_quality = relate_quality_output[
                #     (relate_quality_output.pos < tree.interval[1])
                #     & (relate_quality_output.pos >= tree.interval[0])
                # ]
                # start_index = relate_quality.index[-1]
                # relate_quality_mean = relate_quality.mean()
                # frac_branches_with_snp.append(
                #     relate_quality_mean["frac_branches_with_snp"]
                # )
                # num_snps_on_tree.append(relate_quality_mean["num_snps_on_tree"])
                # fraction_snps_not_mapping.append(
                #     relate_quality_mean["fraction_snps_not_mapping"]
                # )

            tree.next()

        del tree
        del ts
    return (
        tree_size,
        no_of_mutations,
        tmrca,
        recomb_rates,
        rank_zero_snp_branches_target,
        frac_branches_with_snp_target,
        frac_branches_with_snp,
        num_snps_on_tree,
    )


# def mask_for_dodgy_trees(frac_branches_with_snp, num_snps_on_tree, masking_thresh):
#     print(np.percentile(frac_branches_with_snp, masking_thresh * 100))
#     print(np.percentile(num_snps_on_tree, masking_thresh * 100))
#     mask = (
#         frac_branches_with_snp
#         > np.percentile(frac_branches_with_snp, masking_thresh * 100)
#     ) & (num_snps_on_tree > np.percentile(num_snps_on_tree, masking_thresh * 100))
#     return mask


def mask_for_dodgy_trees(recomb_rates, masking_thresh):
    recomb_rates = np.array(recomb_rates)
    print(np.percentile(recomb_rates, (masking_thresh) * 100))
    mask = recomb_rates < np.percentile(recomb_rates, (masking_thresh) * 100)
    print(np.sum(mask) / len(mask))
    return mask


def downsample_trees(ground_truth, pop_index, downsample_frac):
    ## higher downsample_frac means less downsampling, range (0, 1)
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


def write_coal(gamma_arr, filename, labs, is_relate):

    if is_relate:
        filename = args.output + "_" + filename
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


def _silhouette_score(coal_top2, local_ancestry_assignments):
    ## coal_top2 shape is num_of_treesx 2 x num_of_reference_groups
    num_trees = coal_top2.shape[0]
    coal_top2 = coal_top2.reshape(num_trees, -1)
    assert len(coal_top2) == len(local_ancestry_assignments)
    silhouette_scores = silhouette_score(
        coal_top2, labels=local_ancestry_assignments, metric="euclidean", random_state=2
    )
    return silhouette_scores


def main(args, plot=False, gamma_arr=None):

    sample_id_label = "_".join([str(e) for e in args.sample_id])
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
    poplabels = pd.read_csv(Path(path) / "poplabels.txt", sep=" ")
    if poplabels.shape[1] != 4:
        poplabels = pd.read_csv(Path(path) / "poplabels.txt", sep="\t")
    assert poplabels.shape[1] == 4
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)

    ts_list = []
    # ts_list_subsampled = []
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))
    tree_stats_file_name = (
        args.output
        + "_tree_stats_"
        + sample_id_label
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
        args.output
        + "_fixed_params_"
        + sample_id_label
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
    if args.check_muts_target is True:
        print(
            "Note: Filtering based on mutations on target lineage.. ancestry proportion estimates might be biased"
        )
        check_muts_target_name = []
        for chr in chrs:
            check_muts_target_name.append(
                str(Path(path) / str(args.trees + "_chr" + str(chr) + ".allmuts"))
            )
    else:
        check_muts_target_name = None

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
                Path(path) / str(args.trees + "_chr" + str(chr) + ".trees")
            )  ## relate trees
            ts_list.append(ts)
            # if len(poplabels[poplabels.INCLUDE]) != len(poplabels):
            #     ts_list_subsampled.append(
            #         ts.simplify(poplabels[poplabels.INCLUDE].index.tolist())
            #     )
            # else:
            #     ts_list_subsampled.append(ts)
        if len(poplabels) != ts_list[0].num_samples:
            ## num_samples is number of haplotypes
            raise ValueError(
                "Number of samples in trees doesnt match number of samples in poplabels.txt"
            )

    num_samples = len(poplabels)
    for sample in args.sample_id:
        if sample >= num_samples or sample < 0:
            raise ValueError("The sample ids are out of range")
        else:
            print(str(sample) + " is: " + str(poplabels.GROUP.iloc[sample]))

    filename = ".treepos"
    if args.relate_trees:
        filename = "Relate" + filename
    else:
        filename = "True" + filename
    f = open(filename, "w")

    if args.trees != None:
        trees_per_chr = []
        num_trees = 0
        for sample_no in range(len(args.sample_id)):
            for chr_no, ts in enumerate(ts_list):
                tree = ts.first()
                prev_interval = tree.interval[0]
                # prev_interval = 0
                start_pos = copy.deepcopy(num_trees)
                for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
                    if tree.interval[1] >= prev_interval + args.window_size:
                        prev_interval = prev_interval + args.window_size
                        f.write(
                            str(tree.interval[0]) + " " + str(tree.interval[1]) + "\n"
                        )
                        num_trees += 1
                    tree.next()
                trees_per_chr.append((start_pos, num_trees))
        f.close()
        num_trees = int(num_trees / len(args.sample_id))
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
                rank_zero_snp_branches_target,
                frac_branches_with_snp_target,
                frac_branches_with_snp,
                num_snps_on_tree,
                mask_dodgy,
            ) = pickle.load(f_pkl)
            f_pkl.close()
            print("Done loading tree statistics from: " + str(tree_stats_file_name))
        except:
            print("Tree statistics file not found, calculating tree statistics..")
            ## mapping samples back to their original names
            (
                tree_size,
                no_of_mutations,
                tmrca,
                recomb_rates,
                rank_zero_snp_branches_target,
                frac_branches_with_snp_target,
                frac_branches_with_snp,
                num_snps_on_tree,
            ) = compute_tree_stats(
                ts_list,
                chrs,
                args.window_size,
                check_muts_target_name,
                poplabels.index.values[args.sample_id],
            )
            mask_dodgy = mask_for_dodgy_trees(
                recomb_rates * len(args.sample_id),
                # frac_branches_with_snp * len(args.sample_id),
                # num_snps_on_tree * len(args.sample_id),
                1 - args.masking_threshold,
            )
            if check_muts_target_name is not None:
                mask_dodgy *= mask_for_dodgy_trees(
                    rank_zero_snp_branches_target * len(args.sample_id),
                    1 - args.masking_threshold,
                )

            if args.load_mask:
                mask_dodgy2 = np.load(args.load_mask)
                print(np.mean(mask_dodgy))
                mask_dodgy = np.multiply(mask_dodgy, mask_dodgy2)
                print(np.mean(mask_dodgy))

            f_pkl = open(tree_stats_file_name, "wb")
            pickle.dump(
                [
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
                ],
                f_pkl,
            )
            f_pkl.close()
            print("Tree statistics stored in: " + str(tree_stats_file_name))

    else:
        mask_dodgy = np.ones(
            num_trees * len(args.sample_id), dtype=bool
        )  ## No masking needed for true trees

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

    ##### Caution: manually downsampling HAN !! 🌵
    # print("Downsampling !! Caution !!")
    # mask_dodgy *= downsample_trees(ground_truth_membership, 0, 0.25)

    print(
        "Trees with high certainty = " + str(np.sum(mask_dodgy) / len(args.sample_id))
    )
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

    if args.init_at_truth:
        own_membership = ground_truth_membership[:, mask_dodgy]
    elif args.load_membership:
        own_membership = np.load(args.load_membership)[:, mask_dodgy]
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
        tau = np.load(args.load_props)

    if args.evaluate_gamma:
        log_likelihood_arr = []
        start_time_em = time.time()
        print("Starting the EM..")

        filename_logl = args.output + "_" + sample_id_label + ".logl"
        filename_tau = args.output + "_" + sample_id_label + ".tau"
        f_logl = open(filename_logl, "w")
        f_tau = open(filename_tau, "w")

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
            print(tau)
            if args.props_per_chrs:
                tau = np.zeros((len(trees_per_chr_masked), len(own_membership)))
                for chr, (start, end) in enumerate(trees_per_chr_masked):
                    tau[chr] = np.mean(own_membership[:, start:end], axis=1)

            if epoch == 0 and args.load_gamma != None and args.load_props != None:
                print("Using initial gamma specified in file: " + str(args.load_gamma))
                gamma_arr = np.load(args.load_gamma)
                tau = np.load(
                    args.load_props
                )  ### load taus only works for not(props_per_chrs)

            # if tau[0] < tau[1]:
            #     tau = [0.02, 0.98]  ## CAUTION: Fixing tau!!!
            # else:
            #     tau = [0.98, 0.02]
            # tau = np.array(tau)

            assert (gamma_arr >= 0).all()
            prev_gamma = copy.deepcopy(gamma_arr)

            if args.verbose:
                print(gamma_arr)
                print("Iter" + str(epoch))
                print(tau)
                print(len(np.shape(tau)))
            if len(np.shape(tau)) == 1:
                for i in range(np.shape(tau)[0]):
                    f_tau.write(str(tau[i]) + " ")
                f_tau.write("\n")
            else:
                for i in range(np.shape(tau)[0]):
                    for j in range(np.shape(tau)[1]):
                        f_tau.write(str(tau[i, j]) + " ")
                    f_tau.write("\n")
                f_tau.write("\n")

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
                ## Permute the ground truth and membership accordingly
                ground_truth_membership = ground_truth_membership[
                    np.array(
                        np.argsort(np.sum(ground_truth_membership, axis=1)), dtype=int
                    )
                ]
                membership_thresh = membership_thresh[
                    np.array(np.argsort(np.sum(membership_thresh, axis=1)), dtype=int)
                ]
                for i in range(len(membership_thresh)):
                    for j in range(len(ground_truth_membership[:, mask_dodgy])):
                        acc = np.sum(
                            (membership_thresh[i] == 1)
                            & (ground_truth_membership[j, mask_dodgy] == 1)
                        )
                        acc_arr[i][j] = acc
                print(
                    "Sample = "
                    + sample_id_label
                    + " Confusion matrix = "
                    + str(acc_arr)
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
                    frac_branch_x_all.extend(frac_branches_with_snp_i)
                    num_snps_x_all.extend(num_snps_on_tree_i)
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
                    sample_id_label + "_iter" + str(epoch) + ".coal",
                    unique_groups,
                    args.relate_trees,
                )
                with open(
                    args.output
                    + "_gamma_"
                    + sample_id_label
                    + "_iter"
                    + str(epoch)
                    + ".npy",
                    "wb",
                ) as f:
                    np.save(f, gamma_arr)

                with open(
                    args.output
                    + "_props_"
                    + sample_id_label
                    + "_iter"
                    + str(epoch)
                    + ".npy",
                    "wb",
                ) as f:
                    np.save(f, tau)

                filename = "membership_" + sample_id_label + ".npy"
                if args.relate_trees:
                    filename = args.output + "_" + filename
                else:
                    filename = "TrueTrees_" + filename
                with open(filename, "wb") as f:
                    np.save(f, own_membership)

            ## Early-stopping
            print("log-likelihood = " + str(log_likelihood_arr[-1]), flush=True)
            f_logl.write(str(log_likelihood_arr[-1]) + "\n")
            # if epoch > 100: ##min-iters = 100
            #    if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.00001:
            #        break ## stop if log-likelihood isn't changing much

        # if (
        #    np.abs(
        #        (log_likelihood_arr[-1] - log_likelihood_arr[-2])
        #        / log_likelihood_arr[-2]
        #    )
        #    > 0.001
        # ):
        #    warnings.warn(
        #        "The log-likelihood is still increasing, you should consider running for longer"
        #    )

        print(
            "Sample = "
            + sample_id_label
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
            sample_id_label + ".coal",
            unique_groups,
            args.relate_trees,
        )

        with open(
            args.output + "_gamma_" + sample_id_label + ".npy",
            "wb",
        ) as f:
            np.save(f, gamma_arr)

        if args.mode == "sim":
            ## Calculate the calibration on the rich-trees
            num_rows, num_cols = ground_truth_membership.shape

            filename = "calibration.txt"
            if args.relate_trees:
                filename = args.output + "_" + filename
            else:
                filename = "TrueTrees_" + filename
            with open(filename, "w") as f:
                for i in range(len(own_membership)):
                    for j in range(num_rows):
                        y, x = calibration_curve(
                            ground_truth_membership[j, mask_dodgy],
                            own_membership[i],
                            n_bins=20,
                        )
                        for k in range(len(x)):
                            f.write(
                                str(i)
                                + " "
                                + str(j)
                                + " "
                                + str(x[k])
                                + " "
                                + str(y[k])
                                + "\n"
                            )

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

        own_membership = own_membership_update / (np.sum(own_membership_update, axis=0))

        proportion_of_coalescing_top2 = np.zeros(
            (len(args.sample_id) * num_trees, 2, len(unique_groups))
        )
        for tid in range(len(args.sample_id) * num_trees):
            proportion_of_coalescing_top2[tid, :, :] = proportion_of_coalescing_all[
                tid
            ][0:2]
            ## assuming each tree has atleast 2 coal events with the target

        # silhouette_scores = _silhouette_score(
        #     proportion_of_coalescing_top2[mask_dodgy],
        #     np.argmax(own_membership[:, mask_dodgy], axis=0),
        # )  ## calculating silhouette scores only on masked trees
        print("Test log-likelihood: " + str(log_likelihood))
        # print("silhouette_scores: " + str(silhouette_scores))

        filename = (
            "overall_membership_" + sample_id_label + ".npy"
        )  ## this saves membership for all the trees (without the filtering)
        if args.relate_trees:
            filename = args.output + "_" + filename
        else:
            filename = "TrueTrees_" + filename
        with open(filename, "wb") as f:
            np.save(f, own_membership)

        if args.mode == "sim":
            filename = (
                "ground_truth_membership_" + sample_id_label + ".npy"
            )  ## this saves membership for all the trees (without the filtering)
            if args.relate_trees:
                filename = args.output + "_" + filename
            else:
                filename = "TrueTrees_" + filename
            with open(filename, "wb") as f:
                np.save(f, ground_truth_membership)

        return 0


if __name__ == "__main__":
    acc = main(args, plot=False, gamma_arr=None)  ##Han(106), Sardinian(52)
    if args.mode == "sim":
        print("Average accuracy = " + str(acc))

# python ../RelateLocalAncestry/em_true_ancient_sim_subsampled.py --chr 1,2 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas True --window_size 0 --sample_id 0
