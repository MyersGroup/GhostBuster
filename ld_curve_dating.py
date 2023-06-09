import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib
import pdb

font = {"size": 16}
matplotlib.rc("font", **font)
plt.rc("axes.spines", **{"bottom": True, "left": True, "right": False, "top": False})
sns.set_palette("colorblind")
color_palette = sns.color_palette("colorblind")


def func(dist, a, admixtimes, c):
    return a * np.exp(-admixtimes * dist) + c


def plot_ld_curves(dist, means, admixtimes, output_prefix):
    fig, ax = plt.subplots(len(means), len(means), figsize=(10, 10))
    for i in range(len(means)):
        for j in range(len(means)):
            ax[i, j].scatter(dist, means[i, j], color="black", s=2)
            popt, pcov = curve_fit(func, dist, means[i, j])
            ax[i, j].plot(dist, func(dist, *popt), "--", color="gray", linewidth=1)
    fig.text(0.5, 0.04, "Genetic distance (cM)", ha="center", va="center")
    fig.text(
        0.06, 0.5, "Relative probability", ha="center", va="center", rotation="vertical"
    )
    fig.suptitle(
        "Co-ancestry curves (admix time = {0:.2f} generations)".format(admixtimes)
    )
    plt.savefig(output_prefix + "_ld_curve.png")
    plt.show()


def get_admixtimes(initial_values, dist, means):
    admixtimes = 0
    for i in range(means.shape[0]):
        popt, pcov = curve_fit(func, dist, means[i])
        admixtimes += popt[1]
    admixtimes = admixtimes * 100 / means.shape[0]
    return admixtimes


if __name__ == "__main__":
    bin_size = 0.1
    bin_max = 50
    num_bins = int(bin_max / bin_size)
    initial_values = (
        np.sqrt(np.power(10, np.random.uniform(np.log10(20), np.log10(2000))))
        # if args.t_admix_guess is None
        # else [args.t_admix_guess]
    )
    df = pd.read_csv(
        "../real_apr23/sgdp_relate_trees/output/hazara_all_overall_membership_358_359.csv",
        sep="\s+",
    )
    num_clusters = df.shape[1] - 3
    prob_labels = ["prob_" + str(i) for i in range(num_clusters)]

    df_hap1 = df.iloc[0 : df.shape[0] // 2]
    df_hap2 = df.iloc[df.shape[0] // 2 :]
    for prob_col in prob_labels:
        df_hap1[prob_col] = df_hap1[prob_col].values + df_hap2[prob_col].values
    df_hap1 = df_hap1.sample(20000)
    # df = pd.read_csv(
    #     "../bedouin/output/temp_nohmm_overall_membership_0.csv",
    #     sep="\s+",
    # )

    means = np.zeros((num_bins, num_clusters, num_clusters))
    total = np.zeros(num_bins)
    for chr in np.unique(df_hap1.chr):
        df_chr = df_hap1[df_hap1.chr == chr]
        grids = df_chr.genpos.values
        post = df_chr[prob_labels].values
        for d1, p1 in tqdm(zip(grids, post)):
            grids_within_range = grids[np.abs(grids - d1) < bin_max]
            post_within_range = post[np.abs(grids - d1) < bin_max]
            for d2, p2 in zip(grids_within_range, post_within_range):
                means[int(np.abs(d1 - d2) // bin_size)] += np.outer(p1, p2)
                total[int(np.abs(d1 - d2) // bin_size)] += 1

    means = means / total[:, None, None]
    means = np.array(means).transpose(1, 2, 0)
    dist = np.arange(0, bin_max, bin_size) + bin_size / 2
    # dist = np.arange(0.01, 1, 0.001)  # vector of D genetic distances (cM) to fit
    # means_diag = np.vstack(
    #     (
    #         np.exp(-dist * 500 / 100) + np.random.normal(0, 0.05, size=dist.shape),
    #         np.exp(-dist * 550 / 100) + np.random.normal(0, 0.05, size=dist.shape),
    #     )
    # )
    means_diag = np.array([means[i, i] for i in range(len(means))])
    admixtimes = get_admixtimes(initial_values, dist, means_diag)
    print("Admixtime = " + str(admixtimes))

    plot_ld_curves(dist, means, admixtimes, "sim_bed")
