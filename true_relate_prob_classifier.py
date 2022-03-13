from heapq import merge
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler


def remove_outlier(arr, thresh=0.01):
    return (arr < np.percentile(arr, 100 * (1 - thresh))) & (
        arr > np.percentile(arr, thresh * 100)
    )


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


# true_prob = pd.read_csv("force_filter_50/stdpopsim_homsap_2_overall_membership_51.csv")
ground_truth = np.load("force_filter_50/relate_homsap_2_ground_truth_membership_51.npy")
relate_prob = pd.read_csv("force_filter_50/relate_homsap_2_overall_membership_51.csv")
true_prob = pd.DataFrame(
    np.hstack((relate_prob[["0", "1"]], ground_truth.T)), columns=["0", "1", "2", "3"]
)

f_pkl = open(
    "force_filter_50/relate_homsap_2_tree_stats_True_1,2,3,4,5.pkl",
    "rb",
)
filter_tree = np.load("force_filter_50/relate_homsap_2.mask.npy")
tree_stats_pickle = pickle.load(f_pkl)
f_pkl.close()

tree_stats = np.vstack(tree_stats_pickle[0:4])
tree_stats_filterd = tree_stats[:, filter_tree].T
relate_prob[
    [
        "tree_size",
        "tree_left_bp",
        "no_of_muts",
        "tmrca",
    ]
] = tree_stats_filterd

tree_stats = np.vstack(tree_stats_pickle[5:-3])
tree_stats_filterd = tree_stats[:, filter_tree].T
relate_prob[
    [
        "rank_zero_snp_branches_target",
        "frac_branch_target",
        "frac_branch_tree",
        "num_snps_tree",
        "num_snps_target",
        "num_branch_target",
    ]
] = tree_stats_filterd

relate_prob["chr"] = np.array(tree_stats_pickle[-1])[filter_tree]


#### See if atleast one epoch is negtaive
opportunity_target = np.array(tree_stats_pickle[-2])
opportunity_target_list = [[] for _ in range(opportunity_target.shape[1])]
for tid in range(opportunity_target.shape[0]):
    if filter_tree[tid]:
        opportunity_target_tid = opportunity_target[tid]
        for i in range(len(opportunity_target_tid)):
            opportunity_target_list[i].append(opportunity_target_tid[i])

opportunity_target_list = np.array(opportunity_target_list)
for i in range(len(opportunity_target_list)):
    relate_prob["opportunity_" + str(i)] = opportunity_target_list[i]


logpmf_target = np.array(tree_stats_pickle[-3])
logpmf_list = [[] for _ in range(logpmf_target.shape[1])]
for tid in range(logpmf_target.shape[0]):
    if filter_tree[tid]:
        logpmf_tid = logpmf_target[tid]
        for i in range(len(logpmf_tid)):
            logpmf_list[i].append(logpmf_tid[i])

logpmf_list = np.array(logpmf_list)
relate_prob["logpmf_sum"] = 0
for i in range(len(logpmf_list)):
    relate_prob["logpmf_target_" + str(i)] = logpmf_list[i]
    relate_prob["logpmf_sum"] += np.nan_to_num(logpmf_list[i], nan=0)


recomb_rates = np.array(tree_stats_pickle[4])
recomb_rates = recomb_rates[:, filter_tree]

for i in range(len(recomb_rates)):
    relate_prob["recomb_rate_" + str(i)] = recomb_rates[i]

merged_prob = pd.merge(true_prob, relate_prob, on=["0", "1"])


merged_prob["2_diff"] = np.abs(merged_prob["2_x"] - merged_prob["2_y"])

# !!!!!! ### ONLY looking at NEA
# merged_prob = merged_prob[merged_prob["2_x"] == 1]
# print(merged_prob.shape)

mask = np.ones(len(merged_prob), dtype=bool)
for ts in [
    "tree_size",
    "tmrca",
    "recomb_rate_0",
    "recomb_rate_1",
    "recomb_rate_2",
    "recomb_rate_3",
    "frac_branch_target",
    "frac_branch_tree",
    "num_snps_tree",
    "num_snps_target",
    "num_branch_target",
]:
    mask = mask & remove_outlier(merged_prob[ts])
merged_prob = merged_prob.loc[mask]
print("removing 1% outlier on all tree statistics..")

print(merged_prob.shape)

bins = histedges_equalN(merged_prob["2_x"].dropna(), 40)
y_list = []
for i in range(1, len(bins)):
    if i > 0:
        y_list.append(
            merged_prob[
                (merged_prob["2_x"] <= bins[i]) & (merged_prob["2_x"] >= bins[i - 1])
            ]["2_y"].mean()
        )
y_list = np.array(y_list)
plt.scatter(bins[1:], y_list, s=20)
plt.plot([0, 1], [0, 1], linestyle="dashed", color="grey")
plt.xlabel("True tree prob.")
plt.ylabel("Relate tree prob.")
plt.savefig("outlier_removed_true_relate.png")

fig, axes = plt.subplots(2, 8, figsize=(10, 4), dpi=200, sharey=True)
fig.suptitle("Comparing tree statistics with absolute error loss")

for j, ts in enumerate(
    [
        "tree_size",
        "tmrca",
        "recomb_rate_0",
        "recomb_rate_1",
        "recomb_rate_2",
        "recomb_rate_3",
        "frac_branch_target",
        "frac_branch_tree",
        "num_snps_tree",
        "num_snps_target",
        "num_branch_target",
        "opportunity_0",
        "opportunity_1",
        "opportunity_2",
        "opportunity_3",
        "chr",
    ]
):
    # sns.scatterplot(data=merged_prob, x="2_x", y="2_y", hue=ts)
    # plt.plot([0, 1], [0, 1], linestyle="dashed", color="grey")
    # plt.xlabel("True tree prob.")
    # plt.ylabel("Relate tree prob.")
    # plt.savefig("outlier_removed_true_relate_1_" + ts + ".png")
    prob_diff = []
    # weights, bins = np.histogram(merged_prob[ts], bins=100)
    bins = histedges_equalN(merged_prob[ts].dropna(), 10)
    for i in range(1, len(bins)):
        if i > 0:
            prob_diff.append(
                merged_prob[
                    (merged_prob[ts] <= bins[i]) & (merged_prob[ts] >= bins[i - 1])
                ]["2_diff"].mean()
            )
    prob_diff = np.array(prob_diff)
    axes[j // 8, j % 8].scatter(bins[1:], prob_diff, s=20)
    regr = LinearRegression()
    regr.fit(
        bins[1:].reshape(-1, 1),
        prob_diff,
    )
    axes[j // 8, j % 8].plot(
        bins[1:].reshape(-1, 1),
        regr.predict(bins[1:].reshape(-1, 1)),
        color="grey",
    )
    axes[j // 8, j % 8].set_xlabel(ts)

plt.tight_layout()
plt.savefig("linear_regr.pdf")

## Correlation within tree statistics on relate trees
fig, axes = plt.subplots(9, 9, figsize=(28, 28), dpi=200)
fig.suptitle("Comparing correlation within tree statistics")

for i, ts1 in enumerate(
    [
        "tree_size",
        "tmrca",
        "recomb_rate_0",
        "frac_branch_target",
        "frac_branch_tree",
        "num_snps_tree",
        "num_snps_target",
        "num_branch_target",
    ]
):
    for j, ts2 in enumerate(
        [
            "tree_size",
            "tmrca",
            "recomb_rate_0",
            "frac_branch_target",
            "frac_branch_tree",
            "num_snps_tree",
            "num_snps_target",
            "num_branch_target",
        ]
    ):
        ts2_list = []
        weights, bins = np.histogram(merged_prob[ts1], bins=40)
        for k in range(1, len(bins)):
            if k > 0:
                ts2_list.append(
                    merged_prob[
                        (merged_prob[ts1] <= bins[k])
                        & (merged_prob[ts1] >= bins[k - 1])
                    ][ts2].mean()
                )
        ts2_list = np.array(ts2_list)
        bins = bins[1:][(weights != 0) & (~np.isnan(ts2_list))].reshape(-1, 1)
        ts2_list = ts2_list[weights != 0]
        weights = weights[(weights != 0)][(~np.isnan(ts2_list))]
        ts2_list = ts2_list[~np.isnan(ts2_list)]
        regr = LinearRegression()
        regr.fit(
            bins,
            ts2_list,
            weights,
        )
        axes[i, j].scatter(bins, ts2_list, s=20 * weights / weights.max())
        axes[i, j].plot(
            bins,
            regr.predict(bins),
            color="grey",
        )
        axes[i, j].set_xlabel(ts1)
        axes[i, j].set_ylabel(ts2)

plt.tight_layout()
plt.savefig("corr_comparision.pdf")

## standardize data


## multiple linear regression
orig_train_cols = [
    "tree_size",
    "tmrca",
    "recomb_rate_0",
    "recomb_rate_1",
    "recomb_rate_2",
    "recomb_rate_3",
    "frac_branch_target",
    "frac_branch_tree",
    "num_snps_tree",
    "num_snps_target",
    "num_branch_target",
    "chr"
    # "opportunity_0",
    # "opportunity_1",
    # "opportunity_2",
    # "opportunity_3",
    # "opportunity_4",
]
scaler = StandardScaler()
merged_prob[orig_train_cols] = scaler.fit_transform(merged_prob[orig_train_cols])
regr = LinearRegression()
regr.fit(
    merged_prob[orig_train_cols],
    merged_prob["2_diff"],
)
for i, i_col in enumerate(orig_train_cols):
    print(i_col + " : " + str(regr.coef_[i]))

print(
    "Explained variance score: "
    + str(
        explained_variance_score(
            merged_prob["2_diff"], regr.predict(merged_prob[orig_train_cols])
        )
    )
)

## multiple linear regression (+ non-linearity)
train_cols = []
for col1 in orig_train_cols:
    for col2 in orig_train_cols:
        merged_prob[col1 + "*" + col2] = merged_prob[col1] * merged_prob[col2]
        train_cols.append(col1 + "*" + col2)
    train_cols.append(col1)
regr = LinearRegression()
regr.fit(
    merged_prob[train_cols],
    merged_prob["2_diff"],
)
print(
    "Explained variance score (non-linear): "
    + str(
        explained_variance_score(
            merged_prob["2_diff"], regr.predict(merged_prob[train_cols])
        )
    )
)
