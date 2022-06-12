"""
calc_tree_stats.py: contains code to calculate various tree statistics used for filtering
Includes: lineage_nodes(), count_num_muts(), count_lineage_branch_has_muts(),
get_poisson_logpmf_bins(), compute_tree_stats()
"""

import numpy as np
import math
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
import pickle
from pathlib import Path


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


def get_poisson_logpmf_bins(mutrates, num_epochs, mut_rate):
    """
    One gets the mutden file using RelateMutationRate --mode MutationDensity -i relate_homsap_chr22
    -o relate_homsap_chr22 --pop_of_interest 51 --bins 4.5,6.5,0.285714286

    Calculates the normalized logpmf for poisson distribution
    """
    logpmf = np.ones(num_epochs)
    for epoch in range(num_epochs):
        num_muts = mutrates[epoch]
        opportunity = mutrates[num_epochs + epoch] * mut_rate
        if opportunity > 0:
            rv = stats.poisson(opportunity)
            logpmf[epoch] = (
                0.5 * rv.logpmf(math.floor(num_muts))
                + 0.5 * rv.logpmf(math.ceil(num_muts))
            ) / opportunity
        else:
            logpmf[epoch] = np.nan
    return logpmf


def compute_tree_stats(
    ts_list, chrs, check_muts_target_name, rec, sample_list=None, force_build=1
):
    tree_size = []
    tree_left_bp = []
    no_of_mutations = []
    tmrca = []
    recomb_window_size = 50000  ## window size for measure recombination rates
    recomb_rates = []
    rank_zero_snp_branches_target = []
    frac_branches_with_snp_target = []
    frac_branches_with_snp = []
    num_snps_on_tree = []
    num_snps_on_lineage = []
    num_branches_on_target = []
    mutrate_logpmf_target = []
    mutrate_opportunity_target = []
    chr_map = []
    count = 0
    num_nodes = len(list(ts_list[0].first().nodes()))
    first_tree_nodes = list(ts_list[0].first().nodes())[0:-1]
    for chr_no, chr in enumerate(chrs):
        recomb_map = pd.read_csv(
            rec + "_chr" + str(chr) + ".txt",
            sep="\t",
        )
        recomb_map_arr = np.array(recomb_map[recomb_map.columns[1:]])
        recomb_map["Start Position(bp)"] = np.array(
            [0] + recomb_map_arr[:-1, 0].tolist()
        )
        if check_muts_target_name is not None:
            relate_allmuts_file = pd.read_csv(
                check_muts_target_name[chr_no],
                sep=" ",
                engine="c",
            )
            mut_den_filename = check_muts_target_name[chr_no][:-8] + ".mutden"
            mutrates = pd.read_csv(mut_den_filename, sep=" ", header=None)
            mutrates = mutrates.dropna(axis=1)
            epoch_intervals_mutrate = mutrates.iloc[0][0 : int(mutrates.shape[1] / 2)]
            mutrates = mutrates.drop(0)
            mutrates = np.array(mutrates)
            mutrate_num_epochs = int(mutrates.shape[1] / 2)

        ts = ts_list[count]
        count += 1
        tree = ts.first()
        for tid in tqdm(range(ts.num_trees)):  # len(list(ts.trees()))
            tree_size.append(tree.interval[1] - tree.interval[0])
            tree_left_bp.append(tree.interval[0])
            no_of_mutations.append(tree.num_mutations)
            tmrca.append(tree.time(tree.root))
            chr_map.append(chr)
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
            if len(recomb_events) > 1:
                recomb_rate = (
                    recomb_events.iloc[-1]["Map(cM)"] - recomb_events.iloc[0]["Map(cM)"]
                ) / (
                    recomb_events.iloc[-1]["Position(bp)"]
                    - recomb_events.iloc[0]["Position(bp)"]
                )
            else:
                recomb_rate = recomb_events.iloc[0]["Rate(cM/Mb)"] * 1e-6
            recomb_rates.append(recomb_rate)
            if check_muts_target_name is not None:
                relate_allmuts_tree = relate_allmuts_file.iloc[
                    tid * num_nodes : (tid + 1) * num_nodes
                ]
                mut_rates_tid = mutrates[tid]
                rank_zero_snp_branches_target.append(0)
                frac_branches_with_snp_target.append(
                    count_lineage_branch_has_muts(
                        relate_allmuts_tree, lineage_nodes(tree, sample_list)
                    )
                )
                frac_branches_with_snp.append(
                    count_lineage_branch_has_muts(relate_allmuts_tree, first_tree_nodes)
                )
                num_snps_on_tree.append(
                    count_num_muts(relate_allmuts_tree, first_tree_nodes)
                )
                num_snps_on_lineage.append(
                    count_num_muts(
                        relate_allmuts_tree, lineage_nodes(tree, sample_list)
                    )
                )
                num_branches_on_target.append(len(lineage_nodes(tree, sample_list)))
                mutrate_logpmf_target.append(
                    get_poisson_logpmf_bins(
                        mut_rates_tid, mutrate_num_epochs, mut_rate=1e-8
                    )
                )
                mutrate_opportunity_target.append(
                    mut_rates_tid[mutrate_num_epochs : 2 * mutrate_num_epochs]
                )
            else:
                rank_zero_snp_branches_target.append(0)
                frac_branches_with_snp_target.append(0)
                frac_branches_with_snp.append(0)
                num_snps_on_tree.append(0)
                num_snps_on_lineage.append(0)
                num_branches_on_target.append(0)
                mutrate_logpmf_target.append([0])
                mutrate_opportunity_target.append([0])
            tree.next()

        del tree
        del ts
    return (
        tree_size,
        tree_left_bp,
        no_of_mutations,
        tmrca,
        recomb_rates,
        rank_zero_snp_branches_target,
        frac_branches_with_snp_target,
        frac_branches_with_snp,
        num_snps_on_tree,
        num_snps_on_lineage,
        num_branches_on_target,
        mutrate_logpmf_target,
        mutrate_opportunity_target,
        chr_map,
    )


def load_tree_stats(args, ts_list, poplabels):
    chrs = list(map(int, args.chrs.split(",")))
    tree_stats_file_name = args.output + "_tree_stats_" + str(args.chrs) + ".pkl"
    if args.opportunity_filter is True:
        print(
            "Note: Filtering based on mutations on target lineage.. ancestry proportion estimates might be biased"
        )
        check_muts_target_name = []
        for chr in chrs:
            check_muts_target_name.append(
                str(Path(args.path) / str(args.trees + "_chr" + str(chr) + ".allmuts"))
            )
    else:
        check_muts_target_name = None

    try:
        f_pkl = open(tree_stats_file_name, "rb")
        (
            tree_size,
            tree_left_bp,
            no_of_mutations,
            tmrca,
            recomb_rates,
            rank_zero_snp_branches_target,
            frac_branches_with_snp_target,
            frac_branches_with_snp,
            num_snps_on_tree,
            num_snps_on_lineage,
            num_branches_on_target,
            mutrate_logpmf_target,
            mutrate_opportunity_target,
            chr_map,
        ) = pickle.load(f_pkl)
        f_pkl.close()
        print("Done loading tree statistics from: " + str(tree_stats_file_name))
    except:
        print("Tree statistics file not found, calculating tree statistics..")
        ## mapping samples back to their original names
        (
            tree_size,
            tree_left_bp,
            no_of_mutations,
            tmrca,
            recomb_rates,
            rank_zero_snp_branches_target,
            frac_branches_with_snp_target,
            frac_branches_with_snp,
            num_snps_on_tree,
            num_snps_on_lineage,
            num_branches_on_target,
            mutrate_logpmf_target,
            mutrate_opportunity_target,
            chr_map,
        ) = compute_tree_stats(
            ts_list,
            chrs,
            check_muts_target_name,
            args.rec,
            poplabels.index.values[args.sample_id],
        )

        f_pkl = open(tree_stats_file_name, "wb")
        pickle.dump(
            [
                tree_size,
                tree_left_bp,
                no_of_mutations,
                tmrca,
                recomb_rates,
                rank_zero_snp_branches_target,
                frac_branches_with_snp_target,
                frac_branches_with_snp,
                num_snps_on_tree,
                num_snps_on_lineage,
                num_branches_on_target,
                mutrate_logpmf_target,
                mutrate_opportunity_target,
                chr_map,
            ],
            f_pkl,
        )
        f_pkl.close()
        print("Tree statistics stored in: " + str(tree_stats_file_name))

    ### Temporarily only using recomb rates with window_size = 50000
    num_trees = int(np.sum([ts.num_trees for ts in ts_list]))
    mask_dodgy = np.ones(num_trees, dtype=bool)

    if args.load_mask:
        mask_dodgy2 = np.load(args.load_mask)
        mask_dodgy = np.multiply(mask_dodgy, mask_dodgy2)

    return recomb_rates, mutrate_opportunity_target, tree_left_bp
