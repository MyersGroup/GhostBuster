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
import argparse

from utils import boolean

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
    plt.savefig(output_prefix + "ld_curve.svg", dpi=300)
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

def simulate_local_ancestry_markov(output, generations, n_samples=10, total_length_cm=250, bin_size_cm=0.05, a=0.5):
    ### Simulates local ancestry under the model
    ### Used for testing if the dating works or not
    num_bins = int(total_length_cm / bin_size_cm)
    chrom_positions = np.arange(0, total_length_cm, bin_size_cm)
    '''
    # Random genomic position grid
    random_increments = np.random.uniform(low=0, high=2*bin_size_cm, size=num_bins)
    chrom_positions = np.cumsum(random_increments)
    chrom_positions = chrom_positions[chrom_positions <= total_length_cm]
    '''
    for sample in range(n_samples):
        local_ancestry = np.zeros(num_bins)
        current_ancestry = 0 if np.random.rand() < a else 1
        local_ancestry[0] = current_ancestry
        for i in range(1, num_bins):
            r = (chrom_positions[i]-chrom_positions[i-1]) * generations / 100.0
            if current_ancestry == 0:
                P_same = np.exp(-r) + a * (1 - np.exp(-r))
                if np.random.rand() < P_same:
                    current_ancestry_next = 0
                else:
                    current_ancestry_next = 1
            else:
                P_same = (1 - a) * (1 - np.exp(-r)) + np.exp(-r)
                if np.random.rand() < P_same:
                    current_ancestry_next = 1
                else:
                    current_ancestry_next = 0
            local_ancestry[i] = current_ancestry_next
            current_ancestry = current_ancestry_next
        prob_0 = 1 - local_ancestry  
        prob_1 = local_ancestry 
        df_sample = pd.DataFrame({
            'chr': [1] * num_bins, 
            'pos': 0.01*chrom_positions,  # Physical positions (in M)
            'genpos': 0.01*chrom_positions,
            'prob_0': prob_0,
            'prob_1': prob_1
        })
        output_file = f"{output}_overall_membership_{sample + 1}_sample_id_{sample}.csv"
        df_sample.to_csv(output_file, sep="\t", index=False)
        print(f"Sample {sample + 1} saved to {output_file}")

def get_coancestry_per_sample(df_hap1, bin_size, bin_max, num_clusters, prob_labels):
    num_bins = int(bin_max / bin_size)
    means_num = np.zeros((num_bins, num_clusters, num_clusters))
    means_denom = np.zeros((num_bins, num_clusters, num_clusters))
    for chr in np.unique(df_hap1.chr):
        df_chr = df_hap1[df_hap1.chr == chr]
        f = interpolate.interp1d(
            df_chr.genpos.values, df_chr[prob_labels].values, axis=0
        )
        interp_genpos = np.arange(df_chr.genpos.values[0], df_chr.genpos.values[-1], bin_size)
        prob_values = f(interp_genpos)

        # Replace regions where the closest index is more than 0.1cm away as NaN
        # for i in range(len(interp_genpos)):
        #     if min(abs(df_chr.genpos.values - interp_genpos[i])) > 0.001:
        #         prob_values[i] = np.nan
        # print(np.isnan(prob_values).sum() / prob_values.size)
        
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

def run_ld_curve_dating(args):
    bin_size = args.bin_size
    bin_max = args.bin_max
    output_prefix = args.output
    jn_blocks = 20
    initial_values = (
        np.sqrt(np.power(10, np.random.uniform(np.log10(20), np.log10(2000))))
    )
    means_all_blocks = []
    admixtimes_all_blocks = []
    for jn_block in range(jn_blocks):
        means_all = []            
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
            dfc = pd.read_csv(file, sep='\s+')
            dfc['genpos'] = 100 * dfc['genpos'] ## converting M to cM
            ## remove jack-knife block (removing 5% from each chromosome)
            for chr in dfc.chr.unique():
                dfc.loc[dfc.chr == chr, 'block'] = pd.cut(dfc[dfc.chr == chr].pos, bins=jn_blocks, labels=False)
                dfc.loc[(dfc.chr == chr) & (dfc.block == jn_block), ["prob_" + str(i) for i in range(dfc.shape[1] - 4)]] = np.nan
                # dfc = dfc.drop(dfc[(dfc.chr == chr) & (dfc.block == jn_block)].index)                
            len_all.append(len(dfc))
            df = pd.concat([df, dfc], axis=0)

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
                        df_hap1[df_hap1['chr'] == chr], bin_size, bin_max, num_clusters, prob_labels
                    )
                    means_whole_genome_num += means_num
                    means_whole_genome_denom += means_denom
                    props_whole_genome += props_num
                    len_whole_genome += props_denom
                
                means_whole_genome = means_whole_genome_num / means_whole_genome_denom
                props_whole_genome /= len_whole_genome

                for cluster1 in range(num_clusters):
                    for cluster2 in range(num_clusters):
                        means_whole_genome[cluster1, cluster2] /= props_whole_genome[cluster1] * props_whole_genome[cluster2]

                means_all.append(means_whole_genome)

        admixtimes = get_admixtimes(initial_values, dist, means_all)
        print("Block: " + str(jn_block) +  ", Admixture time: " + str(admixtimes))
        admixtimes_all_blocks.append(admixtimes)
        means_all_blocks.append(np.mean(means_all, axis=0))
        # avg_ = 0
        # for _ in range(100):
        #     means_all_test = np.array([[[1 + np.exp(-dist*20) + 0.1*np.random.randn(len(dist))]]])
        #     admixtimes_test = get_admixtimes(initial_values, dist, means_all_test)
        #     print(admixtimes_test)
        #     avg_ += admixtimes_test
    print((np.mean(admixtimes_all_blocks), np.std(admixtimes_all_blocks)))
    plot_ld_curves(dist, means_all_blocks, np.mean(admixtimes_all_blocks), output_prefix, refit=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_size", help="Bin size in ld curves (in cM)", default=0.05, type=float)
    parser.add_argument("--bin_max", help="Maximum bin location in ld curves (in cM)", default=10, type=float)
    parser.add_argument(
        "-o",
        "--output",
        help="Output prefix (same as what was used when calling ghost_buster.py)",
        type=str,
        default=None,
    )
    parser.add_argument("--test", help="Only for debugging purpose, test with simulated local ancestry", type=boolean, default=False)
    args = parser.parse_args()
    print(args)
    if args.test:
        print("Simulating local ancestry under the model...")
        simulate_local_ancestry_markov(args.output, generations=1500, n_samples=10, total_length_cm=250, bin_size_cm=args.bin_size, a=0.05)
    run_ld_curve_dating(args)
    # python -W ignore RelateLocalAncestry/ld_curve_dating.py --output ground_truth_genpos
