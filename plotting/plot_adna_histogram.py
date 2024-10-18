import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Font and style settings
font = {"size": 15}
matplotlib.rc("font", **font)
plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})

# Clear the current figure
plt.clf()

# Create a new figure and axes
fig, ax = plt.subplots(figsize=(35, 4), dpi=300)

# Read the data
df = pd.read_csv('all_ancient_dna_samples.csv', skiprows=[0])
df.loc[df['MeanYBP'] <= 500, 'MeanYBP'] = 501

# Create the histogram plot with KDE
sns.histplot(df['MeanYBP'], ax=ax, log_scale=(True, False), kde=True, color='green')

# Remove x and y axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Set x-axis to extend to 1e6 years and flip the axis
ax.set_xlim(1e6, 500)  # Flip x-axis and set limits

# Increase x-ticks size
plt.xticks([], fontsize=30)

# Set and increase y-ticks size, make them bold, and set specific y-ticks
plt.yticks([100, 500, 1000], fontsize=30)

# Save the figure
plt.savefig('adna_sample_histogram.svg', dpi=300, transparent=True)
