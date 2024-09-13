import tskit
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

def new_get_num_muts_for_edges(ts):
    ## https://github.com/tskit-dev/tskit/discussions/1307
    mut_edges = np.zeros(ts.num_edges, dtype=int)
    for m in ts.mutations():
        mut_edges[m.edge] += 1
    return mut_edges

def get_coal_times(ts, target):
    ### go along the tree and save the matrix of (t1,t2) for each coal. event on target lineage
    seq_all_trees = []
    tree_intervals = []
    for tree in ts.trees():
        seq_ = []
        p = copy.deepcopy(target)
        while p != tree.root:
            # seq_.append((tree.time(p), tree.time(tree.parent(p)))) ## edge persistence
            seq_.append(tree.time(tree.parent(p))) ## node persistence
            p = tree.parent(p)
        seq_all_trees.append(seq_)
        tree_intervals.append(tree.interval)
    df = pd.DataFrame({'interval': tree_intervals, 'sequence': seq_all_trees})
    df['left'] = df['interval'].apply(lambda x: x[0])
    df['right'] = df['interval'].apply(lambda x: x[1])
    return df

def get_true_node_persistence(df, pos, tree_window=1000):
    """
    Find the min and max persistence intervals for each branch in the target lineage within a subset of trees.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing tree intervals and coalescent sequences.
    pos (int): Position for which to analyze branch persistence.
    tree_window (int): Number of trees to consider before and after the target tree. Default is 1000.
    
    Returns:
    np.ndarray: Minimum persistence intervals for each branch.
    np.ndarray: Maximum persistence intervals for each branch.
    """
    left_array = df['left'].values
    right_array = df['right'].values
    target_tree_idx = np.argmax((left_array <= pos) & (right_array > pos))
    start_idx = max(0, target_tree_idx - tree_window)
    end_idx = min(len(df), target_tree_idx + tree_window + 1)
    df_subset = df.iloc[start_idx:end_idx]
    target_seq = df_subset.loc[(df_subset['interval'].str[0] <= pos) & (df_subset['interval'].str[1] > pos), 'sequence'].values[0]
    matching_intervals = np.full((len(target_seq), 2), np.nan)
    for j, branch in enumerate(target_seq):
        branch_tuple = branch
        found_first = False
        for i, seq in enumerate(df_subset['sequence']):
            if branch_tuple in seq:
                if not found_first:
                    matching_intervals[j, 0] = df_subset['interval'].iloc[i][0]  # Set start
                    found_first = True
                matching_intervals[j, 1] = df_subset['interval'].iloc[i][1]  # Update end
        if not found_first:
            break
    matching_intervals[np.isnan(matching_intervals)] = 0
    ## cheating!
    # matching_intervals[:, 0] = matching_intervals[:, 0] - (matching_intervals[:, 1] - matching_intervals[:, 0]) * 0.5
    # matching_intervals[:, 1] = matching_intervals[:, 1] + (matching_intervals[:, 1] - matching_intervals[:, 0]) * 0.5
    return matching_intervals[:, 0], matching_intervals[:, 1]

def get_coal_descendants(ts, target):
    mut_edges = new_get_num_muts_for_edges(ts)
    seq_all_trees = []
    tree_intervals = [] 
    muts_all_trees = []
    for tree in ts.trees():
        seq_ = []
        muts_ = []
        p = copy.deepcopy(target)
        while p != tree.root:
            parent = tree.parent(p)
            num_muts = mut_edges[tree.edge(p)]
            muts_.append(num_muts)
            descendants = set(tree.samples(parent)) - set(tree.samples(p))
            seq_.append(descendants)
            p = parent
        seq_all_trees.append(seq_)
        muts_all_trees.append(muts_)
        tree_intervals.append(tree.interval)    
    df = pd.DataFrame({'interval': tree_intervals, 'sequence': seq_all_trees, 'mutations' : muts_all_trees})
    df['left'] = df['interval'].apply(lambda x: x[0])
    df['right'] = df['interval'].apply(lambda x: x[1])
    return df

def get_overlap(set1, set2, num_samples):
    p1 = len(set1)/num_samples
    p2 = len(set2)/num_samples
    return (len(set1 & set2)/num_samples - p1*p2)/np.sqrt(p1-p1**2)/np.sqrt(p2-p2**2)

def update_intervals(target_seq, row, matching_intervals, num_muts, found_mismatch, num_samples, overlap_threshold):
    """
    Helper function to update the matching intervals when overlap is detected.
    """
    sequence = row['sequence']
    left = row['left']
    right = row['right']
    muts = row['mutations']
    ncoal1 = len(target_seq)
    overlap_matrix = np.zeros((ncoal1, len(sequence)))
    for j, target_descendants in enumerate(target_seq):
        for k, node_descendants in enumerate(sequence):
            overlap_matrix[j, k] = get_overlap(target_descendants, node_descendants, num_samples) #len(target_descendants & node_descendants)/len(node_descendants.union(target_descendants))
    candidate_pairs = []
    for j in range(ncoal1):
        for k in range(len(sequence)):
            if overlap_matrix[j, k] > overlap_threshold:
                candidate_pairs.append((j, k, overlap_matrix[j, k]))
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_nodes_target = set()
    assigned_nodes_other = set()
    for j, k, overlap_value in candidate_pairs:
        if j not in assigned_nodes_target and k not in assigned_nodes_other and not found_mismatch[j]:
            assigned_nodes_target.add(j)
            assigned_nodes_other.add(k)
            if left < matching_intervals[j, 0]: 
                matching_intervals[j, 0] = left
            if right > matching_intervals[j, 1]:
                matching_intervals[j, 1] = right
            num_muts[j] += muts[k]
    found_mismatch = np.ones(ncoal1, dtype='bool')
    for j in assigned_nodes_target:
        found_mismatch[j] = False
    return found_mismatch, matching_intervals, num_muts

def get_approx_node_persistence(df, pos, num_samples, tree_window=1000, overlap_threshold=0.5):
    """
    Find the min and max persistence intervals for each branch in the target lineage within a subset of trees,
    considering nodes equivalent if their descendants overlap by at least `overlap_threshold`.

    Parameters:
    df (pd.DataFrame): DataFrame containing tree intervals and coalescent sequences from `get_coal_descendants`.
    pos (int): Position for which to analyze branch persistence.
    tree_window (int): Number of trees to consider before and after the target tree. Default is 1000.
    overlap_threshold (float): Threshold for the fraction of overlapping descendants to consider nodes equivalent.

    Returns:
    np.ndarray: Minimum persistence intervals for each branch.
    np.ndarray: Maximum persistence intervals for each branch.
    """
    left_array = df['left'].values
    right_array = df['right'].values
    target_tree_idx = np.argmax((left_array <= pos) & (right_array > pos))
    start_idx = max(0, target_tree_idx - tree_window)
    end_idx = min(len(df), target_tree_idx + tree_window + 1)
    df_subset_forward = df.iloc[target_tree_idx:end_idx]
    df_subset_backward = df.iloc[start_idx:target_tree_idx]
    target_seq = df.iloc[target_tree_idx]['sequence']
    matching_intervals = np.full((len(target_seq), 2), np.nan)
    matching_intervals[:,0] = df.iloc[target_tree_idx]['left']
    matching_intervals[:,1] = df.iloc[target_tree_idx]['right']
    num_muts = np.zeros(len(target_seq))
    ncoal1 = len(target_seq)
    found_mismatch_forward, found_mismatch_backward = np.zeros(ncoal1, dtype=bool), np.zeros(ncoal1, dtype=bool)
    # Forward pass
    for i, row in df_subset_forward.iterrows():
        found_mismatch_forward, matching_intervals, num_muts = update_intervals(target_seq, row, matching_intervals, num_muts, found_mismatch_forward, num_samples, overlap_threshold)
        if np.all(found_mismatch_forward): break
    # Backward pass
    for i, row in df_subset_backward.iloc[::-1].iterrows():
        found_mismatch_backward, matching_intervals, num_muts = update_intervals(target_seq, row, matching_intervals, num_muts, found_mismatch_backward, num_samples, overlap_threshold)
        if np.all(found_mismatch_backward): break
    return matching_intervals[:, 0], matching_intervals[:, 1], num_muts

def get_relate_node_persistence(ts, target, pos):
    tree = ts.at(pos)
    ts_edges = ts.edges()
    p = copy.deepcopy(target)
    matching_intervals = []
    num_mut_all = []
    while p != tree.root:
        edge = ts_edges[tree.edge(p)]
        edge_right = max(float(edge.metadata.decode('utf-8').split(" ")[1]), tree.interval[1])
        edge_left = min(float(edge.metadata.decode('utf-8').split(" ")[0]), tree.interval[0])
        num_mut = int(edge.metadata.decode('utf-8').split(" ")[2].rstrip('\x00'))
        p = tree.parent(p)
        matching_intervals.append((edge_left, edge_right))
        num_mut_all.append(num_mut)
    matching_intervals = np.array(matching_intervals)
    return matching_intervals[:, 0], matching_intervals[:, 1], num_mut_all

if __name__ == "__main__":
    ts = tskit.load("./stdpopsim_homsap_chr1.trees")
    num_samples = ts.num_samples
    target = 0
    df1 = get_coal_times(ts, target)
    df = get_coal_descendants(ts, target)
    overall_scaling = np.zeros(20)
    overall_scaling1 = np.zeros(20)
    count_overall = np.zeros(20)
    sum_num, sum_denom = 0, 0
    b1, b2 = [], []
    relate_muts_all, true_muts_all = [], []
    import time
    for pos in range(0, int(ts.sequence_length), 10000000):
        relate_l, relate_r, relate_muts = get_relate_node_persistence(ts, target, pos)
        true_l_1, true_r_1 = get_true_node_persistence(df1, pos)
        st = time.time()
        true_l, true_r, true_muts = get_approx_node_persistence(df, pos, num_samples)
        print(time.time() - st)
        relate_muts_all.extend(relate_muts)
        true_muts_all.extend(true_muts)
        b1.extend(true_r_1 - true_l_1)
        b2.extend(true_r - true_l)
        sum_num += np.sum(true_r - true_l)
        sum_denom += np.sum(relate_r - relate_l)
        for i in range(len(true_l)):
            overall_scaling[i] += (true_r[i] - true_l[i]) / (relate_r[i] - relate_l[i])
            overall_scaling1[i] += (true_r_1[i] - true_l_1[i]) / (relate_r[i] - relate_l[i])
            count_overall[i] += 1
    
    print(np.sum(overall_scaling) / np.sum(count_overall))
    print(sum_num / sum_denom)
    print(np.corrcoef(b1,b2)[0,1])
    print(np.corrcoef(relate_muts_all, true_muts_all)[0,1])
    print(sum(b1)/sum(b2))
    import matplotlib.pyplot as plt 
    plt.scatter(b1, b2)
    plt.show()
    plt.scatter(relate_muts_all, true_muts_all)
    plt.show()
    import pdb; pdb.set_trace()

## maybe look for two consequtive mismatches ?