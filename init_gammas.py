import pandas as pd
import numpy as np
import matplotlib as mpl
from tqdm import tqdm

mpl.use("Agg")
import math
import time
from collections import Counter
import tskit
from sklearn.calibration import calibration_curve
import argparse
import pickle
import copy
import warnings

parser = argparse.ArgumentParser()
parser.add_argument(
		"-k",
		"--num_clusters",
		help="Number of clusters to find using the EM",
		type=int,
		default=2,
		)
parser.add_argument(
		"-path",
		"--path",
		help="Location to the trees, ground truth assignments and recombination maps ",
		type=str,
		default="./",
)
parser.add_argument(
		"-start_time",
		"--start_time",
		help="Starting time for the population size plots, measured in log-scale",
		type=float,
		default=4,
		)
parser.add_argument(
		"-end_time",
		"--end_time",
		help="Ending time for the population size plots, measured in log-scale",
		type=float,
		default=7,
)
args = parser.parse_args()
path = args.path

epoch_intervals = np.array(
		[-np.inf]
		+ np.linspace(
			args.start_time - math.log(28, 10), args.end_time - math.log(28, 10), 9
			).tolist()
		+ [np.inf],
		dtype="float64",
		)
epoch_intervals_pow = np.power(10, epoch_intervals)

num_clusters = args.num_clusters
poplabels = pd.read_csv(path + "poplabels.txt", sep=" ")
unique_groups = np.unique(poplabels.GROUP)

gamma_arr = np.zeros(
		(num_clusters, len(unique_groups), len(epoch_intervals) - 1),
		dtype="float64",
		)


n0 = 20000

gamma_arr[0,:,:] = 0.1/n0
gamma_arr[0,np.where(unique_groups == "Denisova"),:] = 1/n0
gamma_arr[1,:,:] = 0.1/n0
gamma_arr[1,np.where(unique_groups == "Vindija"),:] = 1/n0
gamma_arr[1,np.where(unique_groups == "Chagyrskaya"),:] = 1/n0
gamma_arr[1,np.where(unique_groups == "AltaiNeandertal"),:] = 1/n0
gamma_arr[2,:,:] = 0.1/n0
gamma_arr[2,np.where(unique_groups == "Papuan"),:] = 1/n0
gamma_arr[3,:,:] = 1/n0

print(gamma_arr)

with open(
		"init_gamma.npy", "wb"
		) as f:
	np.save(f, gamma_arr)
