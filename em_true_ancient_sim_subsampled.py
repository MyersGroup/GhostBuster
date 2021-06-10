from numpy.core.fromnumeric import prod
import pandas as pd
import numpy as np
import matplotlib as mpl
from tqdm import tqdm

mpl.use("Agg")
import matplotlib.pyplot as plt
import math
import time
import seaborn as sns
from collections import Counter
import scipy.sparse
import tskit
from sklearn.calibration import calibration_curve
import argparse
import distutils
import pickle

# epoch_intervals = np.array([-np.inf] + np.linspace(3 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])

epoch_intervals = np.array(
    [-np.inf]
    + np.linspace(4 - math.log(28, 10), 7 - math.log(28, 10), 9).tolist()
    + [np.inf]
)
# epoch_intervals = np.array([-np.inf] + np.linspace(5 - math.log(28,10),7 - math.log(28,10), 21).tolist() + [np.inf])  ### recent modification (only ancient past)
epoch_intervals_pow = np.power(10, epoch_intervals)

path = "/well/myers/speidel/SharedWithHrushi/stdpopsim_Han"
# path="/data/smew1/speidel/genomics/relate_analyses/MixedCoalRates/stdpopsim_homsap/"


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


def make_ground_truth(
    ts_list, num_trees, target_group, window_size, sample=None, chrs=None
):
    ## Extracts the ground truth membership from the simulations
    start_time = time.time()
    print(num_trees)
    print("Calculating the ground truth local ancestry..")

    true_assignment_chr = []
    true_assignment_bp = []
    true_assignment_group = []

    ind = 106
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
        (max(true_assignment_group) + 1, num_trees)
    )
    count = 0
    num_tree = 0
    for chr in chrs:
        ts = ts_list[count]
        count += 1
        # ground_truth = pd.read_csv(path + '/stdpopsim_homsap/output/local_ancestry_chr' + str(chr) +'_' + str(sample)+'.csv', names = ['startpos', 'endpos', 'dest'])
        tree = ts.first()
        # prev_interval = tree.interval[0]
        prev_interval = 0
        # num_tree = 0
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
                            true_assignment_group[j], num_tree
                        ] = 1
                        assigned = True
                        break
                    if true_assignment_chr[j] == str(chr) and true_assignment_chr[
                        j + 1
                    ] != str(chr):
                        ground_truth_membership_one_hot[
                            true_assignment_group[j], num_tree
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
                            true_assignment_group[j], num_tree
                        ] = 1
                        assigned = True
                        break
                if assigned == False:
                    print(chr, tree.interval)

                num_tree += 1
            tree.next()

    print(ground_truth_membership_one_hot.sum(axis=0))
    print(all(ground_truth_membership_one_hot.sum(axis=0)))
    print("Done in " + str(time.time() - start_time))
    return ground_truth_membership_one_hot


def fixed_parameters(ts_list, membership, num_trees, window_size, target_seq_):
    eps = 1e-20
    num_samples = len(list(ts_list[0].first().samples()))
    coal_count = np.zeros((len(membership), len(epoch_intervals_pow) - 1, num_trees))
    opportunity = np.zeros((len(membership), len(epoch_intervals_pow) - 1, num_trees))
    proportion_of_coalescing_all = []
    coalescene_times_all = []
    epoch_index_all = []
    count_mut_trees = -1
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
            coal_events_matrix = np.array(coal_events_matrix)
            coal_events_matrix = coal_events_matrix[
                coal_events_matrix[:, 3].argsort()
            ]  ## sorting based on coalescene times
            lineage_content = np.zeros((2 * num_samples - 1, len(membership)))
            target_seq = target_seq_
            for m in membership:
                lineage_content[m[0] : m[1], m[2]] = 1
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
                        proportion_of_coalescing = lineage_content[b] / (
                            sum(lineage_content[b])
                        )
                        coal_count[
                            :, epoch, count_mut_trees
                        ] += proportion_of_coalescing
                        target_seq = c
                        lineage_content[c] = 0
                        proportion_of_coalescing_in_tree.append(
                            proportion_of_coalescing
                        )
                        epoch_index_in_tree.append(epoch)
                        prev_branch_length = prev_branch_length - lineage_content[b] / (
                            sum(lineage_content[b])
                        )
                    elif b == target_seq:
                        proportion_of_coalescing = lineage_content[a] / (
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
                        prev_branch_length = prev_branch_length - lineage_content[a] / (
                            sum(lineage_content[a])
                        )
                    else:
                        lineage_content[c] = lineage_content[a] + lineage_content[b]
                        if sum(lineage_content[a]) == 0 or sum(lineage_content[b]) == 0:
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
                if (event_count == num_samples - 1) and epoch < len(
                    epoch_intervals_pow
                ) - 2:
                    opportunity[:, epoch + 1 :, count_mut_trees] = 0.0
                    break
            proportion_of_coalescing_all.append(proportion_of_coalescing_in_tree)
            epoch_index_all.append(epoch_index_in_tree)
            tree.next()
    return coal_count, opportunity, proportion_of_coalescing_all, epoch_index_all


def compute_gamma_num(
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    num_ref_groups,
):
    num_full_tree = np.zeros((num_ref_groups, len(epoch_intervals) - 1))
    if not (isinstance(prev_gamma, np.ndarray)):
        for tid in range(len(proportion_of_coalescing_all)):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                epoch = epoch_index_in_tree[i]
                num = proportion_of_coalescing_in_tree[i]
                num = num / np.sum(num)
                num_full_tree[:, epoch] += own_membership[tid] * num
    else:
        for tid in range(len(proportion_of_coalescing_all)):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                epoch = epoch_index_in_tree[i]
                prev_gamma_e = prev_gamma[:, epoch]
                num = prev_gamma_e * proportion_of_coalescing_in_tree[i]
                if np.sum(num) > 0:
                    num = num / np.sum(num)
                num_full_tree[:, epoch] += own_membership[tid] * num
    return num_full_tree


def compute_gamma_denom(own_membership, denom):
    eps = 1e-200
    denom_1 = np.zeros(len(epoch_intervals) - 1)
    for epoch in range(len(epoch_intervals) - 1):  #
        denom_1[epoch] = sum(denom[epoch] * own_membership)
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
            path
            + "/recomb_maps/msprime_maps/genetic_map_GRCh37_chr"
            + str(chr)
            + ".txt.gz",
            sep="\t",
        )
        recomb_map_arr = np.array(recomb_map[recomb_map.columns[1:]])
        recomb_map["Start Position(bp)"] = np.array(
            [0] + recomb_map_arr[:-1, 0].tolist()
        )
        relate_quality_output = pd.read_csv(
            path + "/stdpopsim_homsap/Han/relate_homsap_ne_chr" + str(chr) + ".qual",
            sep=" ",
        )
        # relate_quality_output = pd.read_csv(path + '/stdpopsim_homsap/relate_new/relate_homsap_ne_chr' + str(chr) + '.qual', sep = ' ')
        ts = ts_list[count]
        count += 1
        tree = ts.first()
        prev_interval = 0
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

                # num_snps_on_tree.append(relate_quality_output[str(106)][i])
                # num_snps_on_tree.append(relate_quality_output[str(0)][i])
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
    print(np.percentile(frac_branches_with_snp, masking_thresh * 100))
    print(np.percentile(num_snps_on_tree, masking_thresh * 100))
    mask = (
        frac_branches_with_snp
        > np.percentile(frac_branches_with_snp, masking_thresh * 100)
    ) & (num_snps_on_tree > np.percentile(num_snps_on_tree, masking_thresh * 100))
    return mask


# def mask_for_dodgy_trees(recomb_rates, masking_thresh):
#    mask = (recomb_rates < np.percentile(recomb_rates, masking_thresh*100))
#    return mask


def write_coal(gamma_arr, filename, is_relate):

    if is_relate:
        filename = "RelateTrees_" + filename
    else:
        filename = "TrueTrees_" + filename

    # labs = ['Mbuti', 'LBK', 'Sardinian', 'Loschbour', 'MA1', 'Han', 'UstIshim', 'Neanderthal']
    # labs = ['Mbuti', 'Sardinian', 'Han']
    labs = ["Han", "Neanderthal"]
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


def main(args, plot=False, gamma_arr=None):
    start_time = time.time()
    num_clusters = 2
    # membership = [(0,50,0), (50,52,1), (52,102,2), (102,104,3), (104,106,4), (106,156,5), (156,158,6), (158, 160, 7)]   ## (startpos, endpos, groupid)
    # membership = [(0,50,0), (50,100,1), (100, 150, 2)]   ## (startpos, endpos, groupid)
    membership = [(0, 50, 0), (50, 52, 1)]  ## (startpos, endpos, groupid)
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
    for chr in chrs:
        if args.relate_trees:
            # ts = tskit.load(path + '/stdpopsim_homsap/relate_new/relate_homsap_ne_chr'+ str(chr) + '.trees')  ## relate trees
            # ts = tskit.load(path + '/stdpopsim_homsap/moderns_only/relate_homsap_ne_chr'+ str(chr) + '.trees')  ## relate trees
            ts = tskit.load(
                path
                + "/stdpopsim_homsap/Han/relate_homsap_ne_chr"
                + str(chr)
                + ".trees"
            )  ## relate trees
        else:
            ts = tskit.load(
                path
                + "/stdpopsim_homsap/true_trees/stdpopsim_homsap_"
                + str(chr)
                + ".trees"
            )  ## true trees
        ts_list.append(ts)

    filename = ".treepos"
    if args.relate_trees:
        filename = "Relate" + filename
    else:
        filename = "True" + filename
    f = open(filename, "w")

    num_trees = 0
    for ts in ts_list:
        tree = ts.first()
        # prev_interval = tree.interval[0]
        prev_interval = 0
        for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
            if tree.interval[1] >= prev_interval + args.window_size:
                f.write(str(tree.interval[0]) + " " + str(tree.interval[1]) + "\n")
                prev_interval = prev_interval + args.window_size
                num_trees += 1
            tree.next()
    f.close()
    print("Total number of trees = " + str(num_trees))
    if args.relate_trees:
        try:
            f_pkl = open(tree_stats_file_name, "rb")
            (
                tree_size,
                no_of_mutations,
                tmrca,
                recomb_rates,
                frac_branches_with_snp,
                num_snps_on_tree,
                fraction_snps_not_mapping,
                mask_dodgy,
            ) = pickle.load(f_pkl)
            mask_dodgy = mask_for_dodgy_trees(
                frac_branches_with_snp, num_snps_on_tree, args.masking_threshold
            )
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
                frac_branches_with_snp, num_snps_on_tree, args.masking_threshold
            )

            f_pkl = open(tree_stats_file_name, "wb")
            pickle.dump(
                [
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

        # mask_dodgy = mask_for_dodgy_trees(recomb_rates, args.masking_threshold)
    else:
        mask_dodgy = np.ones(num_trees, dtype=bool)  ## No masking needed for true trees
    plt.figure(figsize=(40, 4))
    sns.heatmap(mask_dodgy.reshape(1, -1))
    plt.savefig("ancient_sim_true_mask_dodgy_" + str(args.sample_id) + ".png")
    plt.close()
    print("Trees with high certainty = " + str(np.sum(mask_dodgy)))
    # ground_truth_membership = make_ground_truth(ts, num_trees, sample = sample_id, chrs = [5])
    # ground_truth_membership = make_ground_truth(ts, num_trees, target_group = 2, sample = sample_id, chrs = [1])[[2,3,7,8]]
    # ground_truth_membership = make_ground_truth(ts_list, num_trees, target_group = 5, window_size = args.window_size, sample = args.sample_id, chrs = chrs)
    # num, denom, proportion_of_coalescing_all, epoch_index_all = fixed_parameters(ts_list, membership, num_trees, args.window_size, args.sample_id)

    if args.relate_trees:

        try:
            f_pkl = open(fixed_params_file_name, "rb")
            (
                num,
                denom,
                proportion_of_coalescing_all,
                epoch_index_all,
                ground_truth_membership,
            ) = pickle.load(f_pkl)
            f_pkl.close()
            print("Done loading fixed parameters from: " + str(fixed_params_file_name))

        except:
            print("Fixed parameters file not found, calculating tree statistics..")
            ground_truth_membership = make_ground_truth(
                ts_list,
                num_trees,
                target_group=5,
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
                ts_list, membership, num_trees, args.window_size, args.sample_id
            )
            f_pkl = open(fixed_params_file_name, "wb")
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
            f_pkl.close()
            print("Fixed parameters stored in: " + str(fixed_params_file_name))

    else:
        ground_truth_membership = make_ground_truth(
            ts_list,
            num_trees,
            target_group=5,
            window_size=args.window_size,
            sample=args.sample_id,
            chrs=chrs,
        )
        num, denom, proportion_of_coalescing_all, epoch_index_all = fixed_parameters(
            ts_list, membership, num_trees, args.window_size, args.sample_id
        )

    foo = np.mean(ground_truth_membership * mask_dodgy, axis=1)
    print(foo / sum(foo))
    # initialise membership
    # if args.relate_trees:
    #    with open('groundtruth_membership.npy', 'rb') as f:
    #        ground_truth_membership = np.load(f)

    own_membership = ground_truth_membership  # np.random.dirichlet(np.ones(num_clusters), num_trees).T
    # foo = np.random.dirichlet(np.ones(num_clusters), num_trees).T
    # own_membership += foo
    # foo = np.sum(own_membership, axis = 0)
    # for i in range(len(foo)):
    #    own_membership[0][i] /= foo[i]
    #    own_membership[1][i] /= foo[i]

    # own_membership = np.random.dirichlet(np.ones(num_clusters), num_trees).T

    # own_membership    = np.random.dirichlet(np.ones(num_clusters), num_trees).T
    # random_membership = np.random.dirichlet(np.ones(num_clusters-1), num_trees).T
    # own_membership[0] = ground_truth_membership[1]
    # for i in range(1,num_clusters):
    #    own_membership[i] = np.multiply( (1-own_membership[0]), random_membership[i-1] )
    print(all(own_membership.sum(axis=0)))

    # print(own_membership)
    log_likelihood_arr = []
    start_time_em = time.time()
    print("Starting the EM..")
    for epoch in range(100):  ## max-iters = 40
        gamma_arr = np.zeros(
            (len(own_membership), len(membership), len(epoch_intervals) - 1)
        )
        for j in range(len(own_membership)):
            if epoch == 0:
                n = compute_gamma_num(
                    own_membership[j] * mask_dodgy,
                    None,
                    proportion_of_coalescing_all,
                    epoch_index_all,
                    len(membership),
                )
            else:
                n = compute_gamma_num(
                    own_membership[j] * mask_dodgy,
                    prev_gamma[j],
                    proportion_of_coalescing_all,
                    epoch_index_all,
                    len(membership),
                )  # compute_gamma_num(own_membership[j], prev_gamma[j], proportion_of_coalescing_all, epoch_index_all, len(unique_groups))
            for i in range(len(membership)):
                d = compute_gamma_denom(own_membership[j] * mask_dodgy, denom[i])
                gamma_arr[j][i] = n[i] / d  # n/d #
        assert (gamma_arr >= 0).all()
        prev_gamma = gamma_arr
        tau = np.ones(len(own_membership)) / len(own_membership)
        for j in range(len(own_membership)):
            tau[j] = np.clip(
                np.sum(own_membership[j]) / own_membership[j].shape[0], 1e-10, 1 - 1e-10
            )
        if args.verbose:
            # print(gamma_arr)
            print("Iter" + str(epoch))
            print(tau)

        ## E-step
        own_membership_update = np.ones((len(own_membership), num_trees))

        log_num_em = np.zeros((len(own_membership), num_trees))
        log_denom_em = np.zeros((len(own_membership), num_trees))
        for tid in range(num_trees):
            proportion_of_coalescing_in_tree = proportion_of_coalescing_all[tid]
            epoch_index_in_tree = epoch_index_all[tid]
            for i in range(len(proportion_of_coalescing_in_tree)):
                assert np.sum(proportion_of_coalescing_in_tree[i]) == 1
                assert (np.array(proportion_of_coalescing_in_tree[i]) >= 0).any()
                for j in range(len(own_membership)):
                    log_num_em_j_i = np.log(
                        np.maximum(
                            np.sum(
                                gamma_arr[j, :, epoch_index_in_tree[i]]
                                * proportion_of_coalescing_in_tree[i]
                            ),
                            1e-300,
                        )
                    )
                    log_num_em[j, tid] += log_num_em_j_i
            max_epoch_index = int(
                np.minimum(epoch_index_in_tree[i] + 1, len(epoch_intervals_pow) - 1)
            )
            for j in range(len(own_membership)):
                log_denom_em[j, tid] = -np.sum(
                    gamma_arr[j, :, 0:max_epoch_index]
                    * denom[:, 0:max_epoch_index, tid]
                )  ## summing only till the maximum epoch in that tree
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
            np.log(np.sum(own_membership_update, axis=0))[mask_dodgy]
            + np.max(log_num_em + log_denom_em, axis=0)[mask_dodgy]
        )
        log_likelihood_arr.append(log_likelihood)
        own_membership_update = own_membership_update / (
            np.sum(own_membership_update, axis=0)
        )
        own_membership = own_membership_update
        membership_thresh = make_one_hot(
            np.argmax(own_membership, axis=0), len(own_membership)
        )

        ## Evaluate accuracy
        acc_arr = np.zeros((len(own_membership), len(ground_truth_membership)))
        for i in range(len(own_membership)):
            for j in range(0, len(ground_truth_membership)):
                acc = np.sum(membership_thresh[i] == ground_truth_membership[j])
                acc_arr[i][j] = acc
        overall_acc = (
            np.sum(np.max(acc_arr, axis=1))
            / len(membership_thresh)
            / len(membership_thresh[0])
        )
        print("Sample = " + str(args.sample_id) + " Accuracy = " + str(overall_acc))

        ## Tree stats
        if args.verbose and args.relate_trees:
            proportion_of_coalescing_top2 = np.zeros((num_trees, 2, len(membership)))
            for tid in range(num_trees):
                proportion_of_coalescing_top2[tid, :, :] = proportion_of_coalescing_all[
                    tid
                ][0:2]
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
                tree_size_i = np.array(tree_size)[
                    np.argmax(own_membership, axis=0) == i
                ]
                num_mutations_i = np.array(no_of_mutations)[
                    np.argmax(own_membership, axis=0) == i
                ]
                recomb_rates_i = np.array(recomb_rates)[
                    np.argmax(own_membership, axis=0) == i
                ]
                recomb_rates_i = recomb_rates_i[~np.isnan(np.array(recomb_rates_i))]
                frac_branches_with_snp_i = np.array(frac_branches_with_snp)[
                    np.argmax(own_membership, axis=0) == i
                ]
                num_snps_on_tree_i = np.array(num_snps_on_tree)[
                    np.argmax(own_membership, axis=0) == i
                ]
                fraction_snps_not_mapping_i = np.array(fraction_snps_not_mapping)[
                    np.argmax(own_membership, axis=0) == i
                ]
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
                    np.argmax(own_membership, axis=0) == i
                ]
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
                gamma_arr, "stdpopsim_iter" + str(epoch) + ".coal", args.relate_trees
            )
            filename = "membership.npy"
            if args.relate_trees:
                filename = "RelateTrees_" + filename
            else:
                filename = "TrueTrees_" + filename
            with open(filename, "wb") as f:
                np.save(f, own_membership * mask_dodgy)

        ## Early-stopping
        print("log-likelihood = " + str(log_likelihood_arr[-1]), flush=True)
        # if epoch > 100: ##min-iters = 100
        #    if np.abs((log_likelihood_arr[-1] - log_likelihood_arr[-2])/log_likelihood_arr[-2]) < 0.00001:
        #        break ## stop if log-likelihood isn't changing much

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

    filename = "membership.npy"
    if args.relate_trees:
        filename = "RelateTrees_" + filename
    else:
        filename = "TrueTrees_" + filename
    with open(filename, "wb") as f:
        np.save(f, own_membership * mask_dodgy)

    ## gamma plots
    write_coal(gamma_arr, "stdpopsim.coal", args.relate_trees)
    for i in range(gamma_arr.shape[0]):
        plt.clf()
        for j in range(gamma_arr.shape[1]):
            plt.plot(gamma_arr[i][j], marker="o")
        plt.legend(
            [
                "Mbuti",
                "LBK",
                "Sardinian",
                "Loschbour",
                "MA1",
                "Han",
                "UstIshim",
                "Neanderthal",
            ],
            fontsize=14,
        )
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Gamma", fontsize=14)
        plt.ylim(0, 4e-4)
        plt.show()
        plt.savefig("ancient_sim_true_gamma_" + str(i) + ".png")
        plt.close()

    # Calibration plots
    mapping = np.argmax(acc_arr, axis=1)
    y, x = calibration_curve(
        ground_truth_membership[mapping[0]], own_membership[0], n_bins=20
    )
    plt.clf()
    plt.plot(x, y, marker="o")
    plt.plot(x, x, ":")
    plt.ylabel("True Probability")
    plt.xlabel("Predicted Probability")
    plt.savefig("calibration_plot_relate.png")
    plt.close()

    ## Plotting the heatmaps and likelihood
    plt.clf()
    plt.plot(log_likelihood_arr)
    plt.savefig("ancient_sim_true_log_likelihood.png")
    plt.close()
    plt.clf()
    plt.figure(figsize=(40, 4))
    sns.heatmap(own_membership)
    plt.savefig("ancient_sim_true_own_membership_" + str(args.sample_id) + ".png")
    plt.close()
    plt.clf()
    plt.figure(figsize=(40, 4))
    sns.heatmap(ground_truth_membership)
    plt.savefig(
        "ancient_sim_true_ground_truth_membership_" + str(args.sample_id) + ".png"
    )
    plt.close()

    return overall_acc


parser = argparse.ArgumentParser()
parser.add_argument(
    "-sample_id",
    "--sample_id",
    help="The index of the haplotype you wish local ancestry for",
    type=int,
    default=106,  ## 106 is for the first Han
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
args = parser.parse_args()
acc = main(args, plot=False, gamma_arr=None)  ##Han(106), Sardinian(52)
print("Average accuracy = " + str(acc))

# python ../RelateLocalAncestry/em_true_ancient_sim_subsampled.py --chr 1,2 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas True --window_size 0 --sample_id 0
