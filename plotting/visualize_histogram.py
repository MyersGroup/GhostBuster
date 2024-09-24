import argparse
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

def plot_histogram_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, delimiter='\t')
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
    plt.show()

# Example usage:
csv_file_path = 'ancestry_probabilities.csv'
plot_histogram_from_csv(csv_file_path)
