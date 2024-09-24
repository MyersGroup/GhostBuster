import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

font = {'family': 'normal', 'size': 22}
mpl.rc('font', **font)
plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

def plot_histogram_from_csv(df, output):
    if not all(col in df.columns for col in ['prob_0', 'prob_1']):
        print("Required columns prob_0, prob_1 not found.")
        return
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.hist(df['prob_0'], bins=50, color='blue', alpha=0.7)
    plt.title('Histogram for component 1', fontsize=24)
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.subplot(1, 2, 2)
    plt.hist(df['prob_1'], bins=50, color='green', alpha=0.7)
    plt.title('Histogram for component 2', fontsize=24)
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.tight_layout()
    plt.savefig(output + '_histogram.svg', dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":
    post_file_name = sys.argv[1]
    dfc = []
    for file in glob.glob(post_file_name + "_overall_membership_*_sample_id_*.csv"):
        df = pd.read_csv(file, sep='\s+')
        dfc.append(df)
    
    combined_df = pd.concat(dfc, ignore_index=True)
    plot_histogram_from_csv(combined_df, post_file_name)
