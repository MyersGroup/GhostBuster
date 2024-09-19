import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import scipy.stats as stats
import random
import matplotlib as mpl
font = {'family' : 'normal', 'size' : 22}
mpl.rc('font', **font)
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
mpl.rcParams['xtick.labelsize'] = 20          # Set global font size for x-tick labels
mpl.rcParams['ytick.labelsize'] = 20          # Set global font size for y-tick labels
mpl.rcParams['xtick.major.size'] = 10           # Set global length for major x-ticks
mpl.rcParams['ytick.major.size'] = 10           # Set global length for major y-ticks
mpl.rcParams['axes.linewidth'] = 2            # Set global thickness for axis lines
mpl.rcParams['xtick.major.size'] = 10         # Set global length for major x-ticks
mpl.rcParams['ytick.major.size'] = 10         # Set global length for major y-ticks
mpl.rcParams['xtick.major.width'] = 2         # Set global width for major x-ticks
mpl.rcParams['ytick.major.width'] = 2         # Set global width for major y-ticks

data = {
    'k': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4],
    'll': [-104, -90, -130, -101, -103, -103, -199, -200, -101, -102, np.nan, np.nan]
}
df = pd.DataFrame(data)
df = df.dropna()
def mean_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    n = len(data)
    stderr = stats.sem(data)
    h = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, h

grouped = df.groupby('k')['ll'].apply(lambda x: mean_confidence_interval(x)).apply(pd.Series)
grouped.columns = ['mean', 'ci']

plt.figure(figsize=(6,6))
plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['ci'], fmt='o', capsize=10, markersize=12, linewidth=3, elinewidth=3)
plt.xlabel('Number of clusters')
plt.xticks(grouped.index)
plt.title('Mean held-out likelihood')
plt.savfig(output + '_cross_validation.svg', dpi=300, transparent=True)