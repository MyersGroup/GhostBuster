"""
utils.py: contains code for ground-truth local ancestry calculation, only for simulations
Includes: make_ground_truth()
"""

import numpy as np
import pandas as pd


def make_ground_truth(
    ts_list, num_trees, mask_dodgy, path, sample=None, chrs=None, force_build=1
):
    ## Extracts the ground truth membership from the simulations
    print("Calculating the ground truth local ancestry..")
    ground_truth_membership_one_hot = None

    num_tree = 0
    count_all_tree = 0
    for sample_no, ind in enumerate(sample):
        count = 0
        for chr in chrs:
            ground_truth = pd.read_csv(
                path + str(chr) + "_" + str(ind) + ".csv",
                names=["startpos", "endpos", "dest"],
            )
            if ground_truth_membership_one_hot is None:
                num_ref_groups = int(np.max(ground_truth["dest"]))
                ground_truth_membership_one_hot = np.zeros((num_ref_groups, num_trees))
            tree = ts_list[count].first()
            for tid in range(len(list(ts_list[count].trees()))):
                if (
                    tree.interval[1] // force_build - tree.interval[0] // force_build
                    > 0
                ):
                    if mask_dodgy[count_all_tree]:
                        if tree.num_sites > 0:
                            for j in range(len(ground_truth)):
                                for mut in tree.sites():
                                    if (
                                        mut.position > ground_truth["startpos"].loc[j]
                                        and mut.position < ground_truth["endpos"].loc[j]
                                    ):
                                        ground_truth_membership_one_hot[
                                            int(ground_truth["dest"].loc[j]) - 1,
                                            num_tree,
                                        ] = 1
                                        break
                                    else:
                                        break
                        num_tree += 1
                    count_all_tree += 1
                tree.next()
            count += 1
    ## only return ground truth of groups which actually contribute
    ground_truth_membership_one_hot = ground_truth_membership_one_hot[np.sum(ground_truth_membership_one_hot, axis=1) != 0]
    print(ground_truth_membership_one_hot.shape)
    if ground_truth_membership_one_hot.shape[0] == 1:
        return np.vstack(
            (
                ground_truth_membership_one_hot[
                    np.sum(ground_truth_membership_one_hot, axis=1) != 0
                ],
                1 - np.sum(ground_truth_membership_one_hot, axis=0),
            )
        )
    else:
        return ground_truth_membership_one_hot


if __name__ == "__main__":
    import tskit
    import pdb

    ts_list = [tskit.load("example/relate_homsap_chr22.trees")]
    gt = make_ground_truth(
        ts_list,
        ts_list[0].num_trees,
        np.ones(ts_list[0].num_trees),
        "example",
        sample=[51],
        chrs=[22],
    )
    pdb.set_trace()
