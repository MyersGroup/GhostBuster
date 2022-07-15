import pickle
import numpy as np
import pandas as pd
import scipy.stats
from tabulate import tabulate

tree_stats_file_name = "output/nea_ghost_relate_tree_stats_1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22.pkl"
mask_file_name = "output/nea_ghost_relate_mask_51"
post_file_name = "output/nea_ghost_relate_overall_membership_51.npy"

f_pkl = open(tree_stats_file_name, "rb")
tree_stats = pickle.load(f_pkl)
f_pkl.close()

mask_np = np.load(mask_file_name + ".npy")
post = np.load(post_file_name)
mask = pd.read_csv(mask_file_name + ".csv", sep="\s+")
mask["post1"] = post[0]
results = pd.DataFrame(
    np.zeros((len(tree_stats) + 1, 4)),
    columns=["tree_stats", "median-group1", "median-group2", "t-test"],
)
results.loc[0] = [
    "num_trees",
    len(mask[mask.post1 < 0.1]),
    len(mask[mask.post1 > 0.9]),
    "NA",
]
for i, tree_stat in enumerate(
    [
        "tree_size",
        "tree_left_bp",
        "no_of_mutations",
        "tmrca",
        "recomb_rates",
        "rank_zero_snp_branches_target",
        "frac_branches_with_snp_target",
        "frac_branches_with_snp",
        "num_snps_on_tree",
        "num_snps_on_lineage",
        "num_branches_on_target",
        "mutrate_logpmf_target",
        "mutrate_opportunity_target",
        "chr_map",
    ]
):
    if np.array(tree_stats[i]).ndim == 1:
        mask[tree_stat] = np.array(tree_stats[i])[mask_np]
        group1 = mask[mask.post1 < 0.1]
        group2 = mask[mask.post1 > 0.9]
    elif np.array(tree_stats[i]).ndim == 2:
        mask[tree_stat] = np.nansum(np.array(tree_stats[i])[mask_np], axis=1)
        group1 = mask[mask.post1 < 0.1]
        group2 = mask[mask.post1 > 0.9]
    results.loc[i + 1] = [
        tree_stat,
        np.median(group1[tree_stat]),
        np.median(group2[tree_stat]),
        scipy.stats.ttest_ind(group1[tree_stat], group2[tree_stat]).statistic,
    ]

print(tabulate(results, headers="keys", tablefmt="psql"))
