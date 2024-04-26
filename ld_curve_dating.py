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
        0.03, 0.5, "Relative probability", ha="center", va="center", rotation="vertical"
    )
    if not refit:
        fig.suptitle(
            "Co-ancestry curves (admix time = {0:.1f} generations)".format(admixtimes)
        )
    plt.savefig(output_prefix + "ld_curve.png", dpi=300)
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


def get_coancestry_per_sample(df_hap1, bin_size, bin_max, num_clusters, normalize=False):
    num_bins = int(bin_max / bin_size)
    means_num = np.zeros((num_bins, num_clusters, num_clusters))
    means_denom = np.zeros((num_bins, num_clusters, num_clusters))
    dist = []
    for chr in np.unique(df_hap1.chr):
        df_chr = df_hap1[df_hap1.chr == chr]
        f = interpolate.interp1d(
            df_chr.genpos.values, df_chr[prob_labels].values, axis=0
        )
        interp_genpos = np.arange(df_chr.genpos.values[0], df_chr.genpos.values[-1], bin_size)
        prob_values = f(interp_genpos)

        # Replace regions where the closest index is more than 1kb away as NaN
        for i in range(len(interp_genpos)):
            if min(abs(df_chr.genpos.values - interp_genpos[i])) > 0.001:
                prob_values[i] = np.nan

        print(np.isnan(prob_values).sum() / prob_values.size)
        props = np.nansum(prob_values, axis=0)
        lens = np.sum(~np.isnan(prob_values[:,0]))
        for count, bin in enumerate(np.arange(0, bin_max, bin_size)):
            ## shift prob_values by bin and multiply with itself
            for cluster1 in range(num_clusters):
                for cluster2 in range(num_clusters):
                    if count == 0:
                        cross = prob_values[:, cluster1] * prob_values[:, cluster2]
                    else:
                        cross = prob_values[0:-count, cluster1] * prob_values[count:, cluster2]
                    means_num[count, cluster1, cluster2] += np.nansum(cross) 
                    means_denom[count, cluster1, cluster2] += np.sum(~np.isnan(cross))

    means_num = np.array(means_num).transpose(1, 2, 0) / len(np.unique(df_hap1.chr))
    means_denom = np.array(means_denom).transpose(1, 2, 0) / len(np.unique(df_hap1.chr))
    dist = np.arange(0, bin_max, bin_size)
    return means_num, means_denom, dist, props, lens


if __name__ == "__main__":
    bin_size = 0.05
    bin_max = 5
    jn_blocks = 20
    initial_values = (
        np.sqrt(np.power(10, np.random.uniform(np.log10(20), np.log10(2000))))
        # if args.t_admix_guess is None
        # else [args.t_admix_guess]
    )
    # df = pd.read_csv(
    #     "../../hgdp_1gp/output/sindhi_all_overall_membership_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39.csv",
    #     sep="\s+",
    # )




    
    # for pop in ['mandenka', 'san', 'yoruba', 'mbuti', 'biaka', 'bantukenya', 'bantusafrica']:
    # for pop in ['mozabite', 'hazara', 'yakut', 'maya', 'tuscan', 'bedouin', 'mbuti', 'biaka']:
    for pop in ['relate_wg']:
        for jn_block in range(jn_blocks):
            means_all = []
            # output_prefix =  '../../Bergstrom2018HGDP/ghost_with_eurasian/{0}_wg'.format(pop)
            # output_prefix =  '../../Bergstrom2018HGDP/recent_admix/{0}'.format(pop)
            output_prefix = '../../denisovan_sim/output_ghost/{0}'.format(pop)
            
            num_hap = 0
            len_all = []
            
            file_list = glob.glob(output_prefix + '_overall_membership_*.csv')
            sorted_hap_no = []
            sorted_file_list = []
            for file in file_list:
                hap_no = int(file.split(output_prefix + '_overall_membership_')[1].split('.csv')[0].split('sample_id_')[1])
                sorted_hap_no.append(hap_no)
        
            num_hap += len(sorted_hap_no)
            df = pd.DataFrame()
            for hap_no in np.sort(sorted_hap_no):
                file = glob.glob(output_prefix + '_overall_membership_*' + str(hap_no) + '.csv')[0]
                dfc = pd.read_csv(file, '\s+')
                ## remove jack-knife block (removing 5% from each chromosome)
                for chr in dfc.chr.unique():
                    dfc.loc[dfc.chr == chr, 'block'] = pd.cut(dfc[dfc.chr == chr].pos, bins=jn_blocks, labels=False)
                    dfc = dfc.drop(dfc[(dfc.chr == chr) & (dfc.block == jn_block)].index)

                # dfc = pd.DataFrame()
                # for chr in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22]:
                #     file = glob.glob(output_prefix + '{0}_overall_membership_*'.format(chr) + str(hap_no) + '.csv')[0]
                #     df_i = pd.read_csv(
                #         file,
                #         "\s+",
                #     )
                #     dfc = pd.concat([dfc, df_i], axis=0)
                
                len_all.append(len(dfc))
                df = pd.concat([df, dfc], axis=0)

            # print(df)
            # print("Number of haplotypes = " + str(num_hap))
            num_clusters = df.shape[1] - 4
            prob_labels = ["prob_" + str(i) for i in range(num_clusters)]
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
                    # print()
                    means_whole_genome_num = np.zeros((num_clusters, num_clusters, int(bin_max / bin_size)))
                    means_whole_genome_denom = np.zeros((num_clusters, num_clusters, int(bin_max / bin_size)))
                    props_whole_genome = np.zeros(num_clusters)
                    len_whole_genome = 0
                    
                    for chr in np.unique(df_hap1.chr):
                        means_num, means_denom, dist, props_num, props_denom = get_coancestry_per_sample(
                            df_hap1[df_hap1['chr'] == chr], bin_size, bin_max, num_clusters
                        )
                        means_whole_genome_num += means_num
                        means_whole_genome_denom += means_denom
                        props_whole_genome += props_num
                        len_whole_genome += props_denom
                    
                    means_whole_genome = means_whole_genome_num / means_whole_genome_denom
                    props_whole_genome /= len_whole_genome
                    # print(props_whole_genome)

                    for cluster1 in range(num_clusters):
                        for cluster2 in range(num_clusters):
                            means_whole_genome[cluster1, cluster2] /= props_whole_genome[cluster1] * props_whole_genome[cluster2]
                            # means_whole_genome[cluster1, cluster2] -= props_whole_genome[cluster1] * props_whole_genome[cluster2]
                            # means_whole_genome[cluster1, cluster2] /= props_whole_genome[cluster1] * (1 - props_whole_genome[cluster1])
                            # means_whole_genome[cluster1, cluster2] += 1

                    means_all.append(means_whole_genome)

            # for i in range(2):
            #     for j in range(2):
            #         means[i,j] = np.exp(-dist * 500 / 100) + np.random.normal(0, 0.05, size=dist.shape)
            # means_all = [means]

            admixtimes = get_admixtimes(initial_values, dist, means_all)
            print(str(pop) +  " " + str(admixtimes))

        plot_ld_curves(dist, means_all, admixtimes, output_prefix, refit=False)
