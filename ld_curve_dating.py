import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib
from scipy import interpolate
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline, BSpline
import glob

font = {"size": 16}
matplotlib.rc("font", **font)
plt.rc("axes.spines", **{"bottom": True, "left": True, "right": False, "top": False})
sns.set_palette("colorblind")
color_palette = sns.color_palette("colorblind")


def func(dist, a, c):
    return a * np.exp(-dist / 100) + c


def plot_ld_curves(dist, means_all, admixtimes, output_prefix, refit=False):
    num_sam = len(means_all)
    num_clusters = len(means_all[0])
    fig, ax = plt.subplots(num_clusters, num_clusters, figsize=(10, 10), dpi=300)
    for sam in range(num_sam):
        for i in range(num_clusters):
            for j in range(num_clusters):
                spl = make_interp_spline(dist, means_all[sam][i, j])
                ax[i, j].plot(dist, spl(dist), color="gray", alpha=0.3, linewidth=0.5)
    mean_of_all_sam = np.mean(means_all, axis=0)
    for i in range(num_clusters):
        for j in range(num_clusters):
            spl = make_interp_spline(dist, mean_of_all_sam[i, j])
            ax[i, j].plot(dist, spl(dist), color="black", alpha=1, linewidth=1)
            if refit:
                initial_values = (np.sqrt(np.power(10, np.random.uniform(np.log10(20), np.log10(2000)))))
                admixtimes = get_admixtimes(initial_values, dist, np.array(means_all)[:,i:i+1,j:j+1])
            popt, pcov = curve_fit(
                func, dist * admixtimes, mean_of_all_sam[i, j], maxfev=5000
            )
            ax[i, j].plot(
                dist, func(dist * admixtimes, *popt), "--", color="green", linewidth=1
            )
            if refit:
                y_cord = 0.5
                ax[i, j].text(0.35, y_cord, '{0:.1f} gens'.format(admixtimes), transform=ax[i, j].transAxes, fontsize=14, verticalalignment='bottom')
    fig.text(0.5, 0.04, "Genetic distance (cM)", ha="center", va="center")
    fig.text(
        0.06, 0.5, "Relative probability", ha="center", va="center", rotation="vertical"
    )
    if not refit:
        fig.suptitle(
            "Co-ancestry curves (admix time = {0:.1f} generations)".format(admixtimes)
        )
    plt.savefig(output_prefix + "ld_curve.pdf")
    plt.show()


def get_admixtime_lhood(admixtimes, dist, means_all):
    lhood = 0
    for means in means_all:
        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                popt, pcov = curve_fit(
                    func, dist * admixtimes, means[i, j], maxfev=5000
                )
                res = (means[i, j] - func(dist * admixtimes, *popt)) ** 2
                lhood += len(res) * np.log(np.mean(res)) / 2
    return lhood


def get_admixtimes(initial_guess, dist, means):
    res = minimize(
        get_admixtime_lhood, initial_guess, method="Nelder-Mead", args=(dist, means)
    )
    admixtimes = res.x[0]
    return admixtimes


def get_coancestry_per_sample(df_hap1, bin_size, bin_max, num_clusters):
    num_bins = int(bin_max / bin_size)
    means = np.zeros((num_bins, num_clusters, num_clusters))
    dist = []
    for chr in np.unique(df_hap1.chr):
        df_chr = df_hap1[df_hap1.chr == chr]
        grids = df_chr.genpos.values
        post = df_chr[prob_labels].values
        f = interpolate.interp1d(
            df_chr.genpos.values, df_chr[prob_labels].values, axis=0
        )
        prob_values = f(
            np.arange(df_chr.genpos.values[0], df_chr.genpos.values[-1], bin_size)
        )
        props = np.mean(prob_values, axis=0)
        print(props)
        for count, bin in enumerate(np.arange(0, bin_max, bin_size)):
            ## shift prob_values by bin and multiply with itself
            for cluster1 in range(num_clusters):
                for cluster2 in range(num_clusters):
                    means[count, cluster1, cluster2] += np.sum(
                        prob_values[:, cluster1]
                        * np.array(
                            prob_values[count:, cluster2].tolist()
                            + np.zeros(count).tolist()
                        )
                    ) / (len(prob_values) - count)

        for cluster1 in range(num_clusters):
            for cluster2 in range(num_clusters):
                means[:, cluster1, cluster2] -= props[cluster1] * props[cluster2]
                means[:, cluster1, cluster2] /= props[cluster1] * (1 - props[cluster1])
                means[:, cluster1, cluster2] += 1

    means = np.array(means).transpose(1, 2, 0) / len(np.unique(df_hap1.chr))
    dist = np.arange(0, bin_max, bin_size)
    return means, dist


if __name__ == "__main__":
    bin_size = 0.05
    bin_max = 10
    initial_values = (
        np.sqrt(np.power(10, np.random.uniform(np.log10(20), np.log10(2000))))
        # if args.t_admix_guess is None
        # else [args.t_admix_guess]
    )
    # df = pd.read_csv(
    #     "../../hgdp_1gp/output/sindhi_all_overall_membership_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39.csv",
    #     sep="\s+",
    # )
    output_prefix = '../../Relate_wolfdog/output/american_all_'
    num_hap = 0
    len_all = []
    for chr in range(1, 7):
        file_list = glob.glob(output_prefix + '{0}_overall_membership_*.csv'.format(chr))
        sorted_hap_no = []
        sorted_file_list = []
        for file in file_list:
            hap_no = int(file.split(output_prefix + '{0}_overall_membership_'.format(chr))[1].split('.csv')[0])
            sorted_hap_no.append(hap_no)
        
        num_hap += len(sorted_hap_no)
        for hap_no in np.sort(sorted_hap_no):
            file = output_prefix + '{0}_overall_membership_'.format(chr) + str(hap_no) + '.csv'
            df_i = pd.read_csv(
                file,
                "\s+",
            )
            len_all.append(len(df_i))
            try:
                df = pd.concat([df, df_i], axis=0)
            except:
                df = df_i.copy()

    print("Number of haplotypes = " + str(num_hap))
    num_clusters = df.shape[1] - 3
    prob_labels = ["prob_" + str(i) for i in range(num_clusters)]

    means_all = []
    len_all_cumsum = np.cumsum(len_all)
    for sam in range(num_hap):
        if sam >= 1:
            df_sam = df.iloc[
                len_all_cumsum[sam - 1] : len_all_cumsum[sam]
            ]
        else:
            df_sam = df.iloc[0: len_all_cumsum[0]]
        if sam % 2 == 0:
            df_hap1 = df_sam
        else:
            for prob_col in prob_labels:
                df_hap1[prob_col] += df_sam[prob_col]
                df_hap1[prob_col] /= 2
            means, dist = get_coancestry_per_sample(
                df_hap1, bin_size, bin_max, num_clusters
            )
            means_all.append(means)

    # for i in range(2):
    #     for j in range(2):
    #         means[i,j] = np.exp(-dist * 500 / 100) + np.random.normal(0, 0.05, size=dist.shape)
    # means_all = [means]

    admixtimes = get_admixtimes(initial_values, dist, means_all)
    print("Admixtime = " + str(admixtimes))

    plot_ld_curves(dist, means_all, admixtimes, output_prefix, refit=False)
