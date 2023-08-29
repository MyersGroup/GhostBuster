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
import msprime
import copy
import pdb
import os


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
    args,
    poplabels,
    ts_list,
    chrs,
    allmuts,
    mutden,
    rec,
    sample_list=None,
    force_build=1,
):
    tree_size = []
    tree_left_bp = []
    tree_right_bp = []
    tree_left_bp_gen = []
    tree_right_bp_gen = []
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
    snps_not_mapping = []
    snps_flipped = []
    # mutrate_logpmf_target = []
    # mutrate_opportunity_target = []
    chr_map = []
    count = 0
    num_nodes = len(list(ts_list[0].first().nodes()))
    first_tree_nodes = list(ts_list[0].first().nodes())[0:-1]

    cent_telo_hla = pd.read_csv(
        os.path.dirname(os.path.abspath(__file__)) + "/real_data_mask.txt", sep="\t"
    )

    for chr_no, chr in enumerate(chrs):
        if os.path.isfile(rec + str(chr) + ".txt"):
            recomb_map = pd.read_csv(
                rec + str(chr) + ".txt",
                sep="\t",
            )
        elif os.path.isfile(rec + str(chr) + ".txt.gz"):
            recomb_map = pd.read_csv(
                rec + str(chr) + ".txt.gz",
                sep="\t",
            )
        else:
            raise "Recomb map format not identified"
        recomb_map_arr = np.array(recomb_map[recomb_map.columns[1:]])
        recomb_map["Start Position(bp)"] = np.array(
            [recomb_map_arr[0, 0]] + recomb_map_arr[:-1, 0].tolist()
        )
        recomb_map_msprime = msprime.RateMap.read_hapmap(rec + str(chr) + ".txt")
        tree_left_bp_chr, tree_right_bp_chr = [], []
        if allmuts is not None:
            relate_allmuts_file = pd.read_csv(
                allmuts + str(chr) + ".allmuts",
                sep=" ",
                engine="c",
            )
            # relate_mutgz_file = pd.read_csv(
            #     allmuts + str(chr) + ".mut.gz",
            #     sep=";",
            #     engine="c",
            # )
            # relate_mutgz_file = relate_mutgz_file.groupby('tree_index').mean()
            # mut_den_filename = check_muts_target_name[chr_no][1]
            # mutrates = pd.read_csv(mut_den_filename, sep=" ", header=None)
            # mutrates = mutrates.dropna(axis=1)
            # epoch_intervals_mutrate = mutrates.iloc[0][0 : int(mutrates.shape[1] / 2)]
            # mutrates = mutrates.drop(0)
            # mutrates = np.array(mutrates)
            # mutrate_num_epochs = int(mutrates.shape[1] / 2)

        ts = ts_list[count]
        count += 1
        tree = ts.first()
        for tid in tqdm(range(ts.num_trees)):  # len(list(ts.trees()))
            if tree.interval[1] // force_build - tree.interval[0] // force_build > 0:
                tree_size.append(tree.interval[1] - tree.interval[0])
                tree_left_bp_chr.append(tree.interval[0])
                tree_right_bp_chr.append(tree.interval[1])
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
                            recomb_map[recomb_map.columns[1]] ##position(bp)
                            < tree.interval[0] - recomb_window_size
                        )
                    )
                ]
                if (
                    args.mode == "real"
                    and (
                        (
                            tree.interval[0]
                            >= cent_telo_hla[cent_telo_hla.chr == str(chr)].start
                        )
                        & (
                            tree.interval[1]
                            < cent_telo_hla[cent_telo_hla.chr == str(chr)].end
                        )
                    ).any()
                ):
                    recomb_rate = np.nan
                elif len(recomb_events) > 1:
                    recomb_rate = (
                        recomb_events.iloc[-1][recomb_map.columns[3]] ##map(cm)
                        - recomb_events.iloc[0][recomb_map.columns[3]]
                    ) / (
                        recomb_events.iloc[-1][recomb_map.columns[1]] ##position(bp)
                        - recomb_events.iloc[0][recomb_map.columns[1]]
                    )
                else:
                    recomb_rate = np.nan  # recomb_events.iloc[0]["Rate(cM/Mb)"] * 1e-6
                recomb_rates.append(recomb_rate)
                if allmuts is not None:
                    relate_allmuts_tree = relate_allmuts_file.iloc[
                        tid * num_nodes : (tid + 1) * num_nodes
                    ]
                    # relate_mutgz_tree = relate_mutgz_file.iloc[tid]
                    # mut_rates_tid = mutrates[tid]
                    rank_zero_snp_branches_target.append(0)
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
                    num_snps_on_lineage.append(
                        count_num_muts(
                            relate_allmuts_tree, lineage_nodes(tree, sample_list)
                        )
                    )
                    num_branches_on_target.append(len(lineage_nodes(tree, sample_list)))
                    snps_not_mapping.append(0)  # (relate_mutgz_tree['is_not_mapping'])
                    snps_flipped.append(0)  # relate_mutgz_tree['is_flipped'])
                    # mutrate_logpmf_target.append(
                    #     get_poisson_logpmf_bins(
                    #         mut_rates_tid, mutrate_num_epochs, mut_rate=1e-8
                    #     )
                    # )
                    # mutrate_opportunity_target.append(
                    #     mut_rates_tid[mutrate_num_epochs : 2 * mutrate_num_epochs]
                    # )
                else:
                    rank_zero_snp_branches_target.append(0)
                    frac_branches_with_snp_target.append(0)
                    frac_branches_with_snp.append(0)
                    num_snps_on_tree.append(0)
                    num_snps_on_lineage.append(0)
                    num_branches_on_target.append(0)
                    snps_not_mapping.append(0)
                    snps_flipped.append(0)
                    # mutrate_logpmf_target.append([0])
                    # mutrate_opportunity_target.append([0])
            tree.next()

        del tree
        del ts
        tree_left_bp_gen.extend(
            recomb_map_msprime.get_cumulative_mass(tree_left_bp_chr).tolist()
        )
        tree_right_bp_gen.extend(
            recomb_map_msprime.get_cumulative_mass(tree_right_bp_chr).tolist()
        )
        tree_left_bp.extend(tree_left_bp_chr)
        tree_right_bp.extend(tree_right_bp_chr)

    if mutden is not None:
        mutrate_logpmf_target, mutrate_opportunity_target = compute_mutden(
            ts_list, chrs, sample_list, mutden, force_build
        )
    else:
        mutrate_logpmf_target = np.zeros(len(recomb_rates)).tolist()
        mutrate_opportunity_target = np.zeros(len(recomb_rates)).tolist()

    return (
        tree_size,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
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
        snps_not_mapping,
        snps_flipped,
    )


def compute_mutden(ts_list, chrs, samples, mutden, force_build=1):
    print("Using mutden files to get tree statistics on target lineage")
    mutrate_logpmf_target = []
    mutrate_opportunity_target = []
    for sample_no in samples:
        count = 0
        for chr_no, chr in enumerate(chrs):
            mut_den_filename = mutden + str(chr) + "_" + str(sample_no) + ".mutden"
            mutrates = pd.read_csv(mut_den_filename, sep=" ", header=None)
            mutrates = mutrates.dropna(axis=1)
            mutrates = mutrates.drop(0)
            mutrates = np.array(mutrates)
            mutrate_num_epochs = int(mutrates.shape[1] / 2)
            ts = ts_list[count]
            count += 1
            tree = ts.first()
            for tid in tqdm(range(ts.num_trees)):  # len(list(ts.trees()))
                if (
                    tree.interval[1] // force_build - tree.interval[0] // force_build
                    > 0
                ):
                    mut_rates_tid = mutrates[tid]
                    mutrate_logpmf_target.append(
                        get_poisson_logpmf_bins(
                            mut_rates_tid, mutrate_num_epochs, mut_rate=1e-8
                        )
                    )
                    mutrate_opportunity_target.append(
                        mut_rates_tid[mutrate_num_epochs : 2 * mutrate_num_epochs]
                    )
                tree.next()

    return mutrate_logpmf_target, mutrate_opportunity_target


def load_tree_stats(args, ts_list, poplabels):
    chrs = list(map(int, args.chrs.split(",")))
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    tree_stats_file_name = args.output + "_tree_stats_" + str(args.chrs) + ".pkl"
    try:
        f_pkl = open(tree_stats_file_name, "rb")
        (
            tree_size,
            tree_left_bp,
            tree_right_bp,
            tree_left_bp_gen,
            tree_right_bp_gen,
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
            snps_not_mapping,
            snps_flipped,
        ) = pickle.load(f_pkl)
        f_pkl.close()
        print("Done loading tree statistics from: " + str(tree_stats_file_name))
    except:
        print("Tree statistics file not found, calculating tree statistics..")
        ## mapping samples back to their original names
        (
            tree_size,
            tree_left_bp,
            tree_right_bp,
            tree_left_bp_gen,
            tree_right_bp_gen,
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
            snps_not_mapping,
            snps_flipped,
        ) = compute_tree_stats(
            args,
            poplabels,
            ts_list,
            chrs,
            args.allmuts,
            args.mutden,
            args.rec,
            args.sample_id,
            args.force_build,
        )

        f_pkl = open(tree_stats_file_name, "wb")
        pickle.dump(
            [
                tree_size,
                tree_left_bp,
                tree_right_bp,
                tree_left_bp_gen,
                tree_right_bp_gen,
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
                snps_not_mapping,
                snps_flipped,
            ],
            f_pkl,
        )
        f_pkl.close()
        print("Tree statistics stored in: " + str(tree_stats_file_name))

    return (
        recomb_rates,
        mutrate_opportunity_target,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        chr_map,
        frac_branches_with_snp_target,
        mutrate_logpmf_target,
        num_snps_on_lineage,
    )
