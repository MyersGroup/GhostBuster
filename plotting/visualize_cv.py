import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# Font and plot style configurations
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

# Argument parsing using nargs=[+]
parser = argparse.ArgumentParser(description="Scatter plot with nargs for k, ll, and r2 values")
parser.add_argument('--k', nargs='+', type=int, help='Cluster numbers (k)')
parser.add_argument('--ll', nargs='+', type=float, help='Log-likelihood values')
parser.add_argument('--r2', nargs='+', type=float, help='R² values')
parser.add_argument('--output', type=str, help='Output file name')

args = parser.parse_args()
assert len(args.k) == len(args.ll) == len(args.r2)

# Convert inputs into a DataFrame
data = {'k': args.k, 'll': args.ll, 'r2': args.r2}
df = pd.DataFrame(data)
print(df)

# Create subplots stacked on top of each other
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)  # Share the x-axis

# First subplot for ll values
ax1.scatter(df['k'], df['ll'], s=150, c='blue', edgecolors='black', linewidths=2, label='Held-out Log-Likelihood')  # Larger points
ax1.plot(df['k'], df['ll'], linestyle='--', color='blue', linewidth=3)  # Fatter lines
ax1.set_ylabel(None)
min_lim_ll = df.ll.min() - 0.2 * (df.ll.max() - df.ll.min())
max_lim_ll = df.ll.max() + 0.2 * (df.ll.max() - df.ll.min())
ax1.set_ylim(min_lim_ll, max_lim_ll)
ax1.legend(loc='upper right', fontsize=16, frameon=False)  # Smaller legend font
ax1.set_xticks(args.k)  # Set xticks only at args.k

# Second subplot for r2 values
ax2.scatter(df['k'], df['r2'], s=150, c='green', edgecolors='black', linewidths=2, label='Adjusted R²')  # Larger points
ax2.plot(df['k'], df['r2'], linestyle='--', color='green', linewidth=3)  # Fatter lines
ax2.set_ylabel(None)
min_lim_r2 = df.r2.min() - 0.2 * (df.r2.max() - df.r2.min())
max_lim_r2 = df.r2.max() + 0.2 * (df.r2.max() - df.r2.min())
ax2.set_ylim(min_lim_r2, max_lim_r2)
ax2.legend(loc='upper right', fontsize=16, frameon=False)  # Smaller legend font
ax2.set_xticks(args.k)  # Set xticks only at args.k

# Common x-label
fig.text(0.5, 0.04, 'Number of clusters', ha='center', fontsize=22)  # Adjust position and size

# Adjust layout
plt.tight_layout(rect=[0, 0.04, 1, 1])  # Adjust the bottom space for the common xlabel

# Save the figure
plt.savefig(args.output + '.svg', dpi=300, transparent=True)

# For testing: uncomment below to simulate input in an interactive environment
# parser.parse_args(['--k', '1', '2', '3', '--ll', '-104', '-199', '-130', '--r2', '0.8', '0.6', '0.9', '--output', 'test_output'])
