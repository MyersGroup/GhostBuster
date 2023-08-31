"""
utils.py: contains code for ground-truth local ancestry calculation, only for simulations
Includes: make_ground_truth()
"""

import numpy as np
import pandas as pd
import copy
import pdb
from utils import make_one_hot


def make_ground_truth(
    ts_list, mask_dodgy, path, sample, chrs, force_build, tree_left_bp, tree_right_bp
):
    ## Extracts the ground truth membership from the simulations
    print("Calculating the ground truth local ancestry..")
    ground_truth_membership = []
    num_tree = 0
    count_all_tree = 0
    for sample_no, ind in enumerate(sample):
        count = 0
        for chr in chrs:
            ground_truth = pd.read_csv(
                path + str(chr) + "_" + str(ind) + ".csv",
                names=["startpos", "endpos", "dest"],
                float_precision="round_trip",
            )
            tree = ts_list[count].first()
            for tid in range(len(list(ts_list[count].trees()))):
                if (
                    tree.interval[1] // force_build - tree.interval[0] // force_build
                    > 0
                ):
                    if mask_dodgy[count_all_tree]:
                        for loc_in_window in range(
                            int(tree_left_bp[num_tree] / force_build),
                            int(tree_right_bp[num_tree] / force_build),
                        ):
                            loc_in_bp = loc_in_window * force_build
                            flag = False
                            for j in range(len(ground_truth)):
                                if (
                                    loc_in_bp >= ground_truth["startpos"].loc[j]
                                    and loc_in_bp < ground_truth["endpos"].loc[j]
                                ):
                                    ground_truth_membership.append(
                                        int(ground_truth["dest"].loc[j]) - 1
                                    )
                                    flag = True
                                    break
                            if not flag:
                                ground_truth_membership.append(0)
                        num_tree += 1
                    count_all_tree += 1
                tree.next()
            count += 1
    ## only return ground truth of groups which actually contribute
    return make_one_hot(np.array(ground_truth_membership))
    ground_truth_membership_one_hot = ground_truth_membership_one_hot[
        np.sum(ground_truth_membership_one_hot, axis=1) != 0
    ]
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


def get_groundtruth_reference(
    ts_list, poplabels, num_trees, mask_dodgy, path, chrs, tgt_group, force_build=1
):
    poplabels = poplabels[poplabels.GROUP == tgt_group]
    ground_truth_ref = np.zeros((len(poplabels), num_trees), dtype=int)

    ## Assign ground_truth_ref based on global ancestry
    for sample_no, ind in enumerate(poplabels.index):
        ground_truth_ref[sample_no] = -127

    group_id_new = {}
    group_id_new[tgt_group + "->" + tgt_group] = 0
    num_groups = 1
    ## Add logic to use local ancestry to make more population categories
    for sample_no, ind in enumerate(poplabels.index):
        print(sample_no)
        num_tree = 0
        count_all_tree = 0
        count = 0
        for chr in chrs:
            ground_truth = pd.read_csv(
                path + str(chr) + "_" + str(ind) + ".csv",
                names=["startpos", "endpos", "dest"],
                float_precision="round_trip",
            )
            tree = ts_list[count].first()
            for tid in range(len(list(ts_list[count].trees()))):
                if (
                    tree.interval[1] // force_build - tree.interval[0] // force_build
                    > 0
                ):
                    if mask_dodgy[count_all_tree]:
                        for j in range(len(ground_truth)):
                            if (
                                tree.interval[0] >= ground_truth["startpos"].loc[j]
                                and tree.interval[0] < ground_truth["endpos"].loc[j]
                            ):
                                try:
                                    ground_truth_ref[
                                        sample_no, num_tree
                                    ] = group_id_new[
                                        str(poplabels.GROUP.iloc[sample_no])
                                        + "->"
                                        + str(ground_truth["dest"].loc[j])
                                    ]
                                except:
                                    group_id_new[
                                        str(poplabels.GROUP.iloc[sample_no])
                                        + "->"
                                        + str(ground_truth["dest"].loc[j])
                                    ] = copy.deepcopy(num_groups)
                                    ground_truth_ref[
                                        sample_no, num_tree
                                    ] = group_id_new[
                                        str(poplabels.GROUP.iloc[sample_no])
                                        + "->"
                                        + str(ground_truth["dest"].loc[j])
                                    ]
                                    num_groups += 1

                        if ground_truth_ref[sample_no, num_tree] == -127:
                            try:
                                ground_truth_ref[sample_no, num_tree] = group_id_new[
                                    str(poplabels.GROUP.iloc[sample_no])
                                    + "->"
                                    + str(poplabels.GROUP.iloc[sample_no])
                                ]
                            except:
                                group_id_new[
                                    str(poplabels.GROUP.iloc[sample_no])
                                    + "->"
                                    + str(poplabels.GROUP.iloc[sample_no])
                                ] = copy.deepcopy(num_groups)
                                ground_truth_ref[sample_no, num_tree] = group_id_new[
                                    str(poplabels.GROUP.iloc[sample_no])
                                    + "->"
                                    + str(poplabels.GROUP.iloc[sample_no])
                                ]
                                num_groups += 1
                        num_tree += 1
                    count_all_tree += 1
                tree.next()
            count += 1
        print(np.mean(ground_truth_ref[sample_no]))
    return ground_truth_ref, group_id_new


if __name__ == "__main__":
    import tskit
    import pdb

    # ts_list = [tskit.load("example/relate_homsap_chr22.trees")]
    # gt = make_ground_truth(
    #     ts_list,
    #     ts_list[0].num_trees,
    #     np.ones(ts_list[0].num_trees),
    #     "example",
    #     sample=[51],
    #     chrs=[22],
    # )
    ts_list = [
        tskit.load(
            "../sims/nea_const_recomb_0.2/relate_trees/relate_homsap_chr22.trees"
        )
    ]
    poplabels = pd.read_csv(
        "../sims/nea_const_recomb_0.2/relate_trees/poplabels.txt", sep="\s+"
    )
    gt_ref, unique_groups = get_groundtruth_reference(
        ts_list,
        poplabels,
        ts_list[0].num_trees,
        np.ones(ts_list[0].num_trees),
        "../sims/nea_const_recomb_0.2/local_ancestry/local_ancestry_chr",
        [22],
        force_build=1,
    )
    pdb.set_trace()
