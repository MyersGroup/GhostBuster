import tskit
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import scipy.stats as stats

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

def get_true_node_persistence(df, pos, bp_grid, tree_window=1000):
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
    target_tree_idx = np.argmax((left_array <= pos) & (right_array >= pos))
    start_idx = max(0, target_tree_idx - tree_window)
    end_idx = min(len(df)-1, target_tree_idx + tree_window + 1)
    df_subset = df.iloc[start_idx:end_idx]
    target_seq = df_subset.loc[(df_subset['interval'].str[0] <= pos) & (df_subset['interval'].str[1] >= pos), 'sequence'].values[0]
    number_of_overlaps = np.zeros(len(target_seq))
    leftmost = df.iloc[start_idx]['left']
    rightmost = df.iloc[end_idx]['right']
    bp_grid = bp_grid[(bp_grid >= leftmost) & (bp_grid <= rightmost)]
    df_subset = df_subset[df_subset.apply(lambda row: np.any((bp_grid >= row['left']) & (bp_grid <= row['right'])), axis=1)]
    df_subset['num_bp_grid'] = df_subset.apply(lambda row: np.sum((bp_grid >= row['left']) & (bp_grid <= row['right'])), axis=1)
    for j, branch in enumerate(target_seq):
        branch_tuple = branch
        found_first = False
        for i, row in df_subset.iterrows():
            seq = row['sequence']
            num_bp_grid = row['num_bp_grid']
            if branch_tuple in seq:
                number_of_overlaps[j] += num_bp_grid
    return number_of_overlaps

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
            num_muts = mut_edges[tree.edge(parent)]
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

def update_intervals(target_seq, row, number_of_overlaps, num_muts, found_mismatch, num_samples, num_bp_grid, overlap_threshold):
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
            if overlap_matrix[j, k] >= overlap_threshold:
                candidate_pairs.append((j, k, overlap_matrix[j, k]))
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_nodes_target = set()
    assigned_nodes_other = set()
    for j, k, overlap_value in candidate_pairs:
        if j not in assigned_nodes_target and k not in assigned_nodes_other and not found_mismatch[j]:
            assigned_nodes_target.add(j)
            assigned_nodes_other.add(k)
            number_of_overlaps[j] += num_bp_grid
            num_muts[j] += muts[k]
    ## removed the lines below because the descendants may reappear
    found_mismatch = np.ones(ncoal1, dtype='bool')
    for j in assigned_nodes_target:
        found_mismatch[j] = False
    return found_mismatch, number_of_overlaps, num_muts

def get_approx_node_persistence(df, pos, num_samples, bp_grid, overlap_threshold=0.8, tree_window=1000):
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

    Note: num_muts calculated here is wrong, it should be the number of mutations on the edge (as calculated in Relate)
    """
    left_array = df['left'].values
    right_array = df['right'].values
    target_tree_idx = np.argmax((left_array <= pos) & (right_array >= pos))
    start_idx = max(0, target_tree_idx - tree_window)
    end_idx = min(len(df)-1, target_tree_idx + tree_window + 1)
    df_subset_forward = df.iloc[target_tree_idx:end_idx]
    df_subset_backward = df.iloc[start_idx:target_tree_idx]
    target_seq = df.iloc[target_tree_idx]['sequence']
    number_of_overlaps = np.zeros(len(target_seq))
    num_muts = np.zeros(len(target_seq))
    ncoal1 = len(target_seq)
    leftmost = df.iloc[start_idx]['left']
    rightmost = df.iloc[end_idx]['right']
    bp_grid = bp_grid[(bp_grid >= leftmost) & (bp_grid <= rightmost)]
    df_subset_forward = df_subset_forward[
        df_subset_forward.apply(lambda row: np.any((bp_grid >= row['left']) & (bp_grid <= row['right'])), axis=1)
    ]
    df_subset_backward = df_subset_backward[
        df_subset_backward.apply(lambda row: np.any((bp_grid >= row['left']) & (bp_grid <= row['right'])), axis=1)
    ]
    found_mismatch_forward, found_mismatch_backward = np.zeros(ncoal1, dtype=bool), np.zeros(ncoal1, dtype=bool)
    # Forward pass
    for i, row in df_subset_forward.iterrows():
        num_bp_grid = np.sum((row['right'] >= bp_grid) & (row['left'] <= bp_grid))
        found_mismatch_forward, number_of_overlaps, num_muts = update_intervals(target_seq, row, number_of_overlaps, num_muts, found_mismatch_forward, num_samples, num_bp_grid, overlap_threshold)
        if np.all(found_mismatch_forward): break
    # Backward pass
    for i, row in df_subset_backward.iloc[::-1].iterrows():
        num_bp_grid = np.sum((row['right'] >= bp_grid) & (row['left'] <= bp_grid))
        found_mismatch_backward, number_of_overlaps, num_muts = update_intervals(target_seq, row, number_of_overlaps, num_muts, found_mismatch_backward, num_samples, num_bp_grid, overlap_threshold)
        if np.all(found_mismatch_backward): break
    return number_of_overlaps, num_muts

def get_relate_node_persistence(ts, target, pos, bp_grid):
    tree = ts.at(pos)
    ts_edges = ts.edges()
    p = copy.deepcopy(target)
    number_of_overlaps = []
    num_mut_all = []
    while p != tree.root:
        edge = ts_edges[tree.edge(p)]
        edge_right = max(float(edge.metadata.decode('utf-8').split(" ")[1]), tree.interval[1])
        edge_left = min(float(edge.metadata.decode('utf-8').split(" ")[0]), tree.interval[0])
        edgep = ts_edges[tree.edge(tree.parent(p))]
        num_mut = int(edgep.metadata.decode('utf-8').split(" ")[2].rstrip('\x00'))
        num_mut_all.append(num_mut)
        number_of_overlaps.append(np.sum((bp_grid >= edge_left) & (bp_grid <= edge_right)))
        p = tree.parent(p)
    return number_of_overlaps, num_mut_all

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import scipy.stats as stats
    from sklearn.linear_model import LinearRegression
    import matplotlib 
    font = {"size": 16}
    matplotlib.rc("font", **font)
    plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
    sns.set_palette('colorblind')
    
    ts = tskit.load("../../denisovan_sim_2024_08/trees/stdpopsim_homsap_chr1.trees")
    bp_grid = np.arange(0, ts.sequence_length, 1e4)
    num_samples = ts.num_samples
    target = 0
    df1 = get_coal_times(ts, target)
    df = get_coal_descendants(ts, target)
    
    b1, b2, b3 = [], [], []
    m1, m2 = [], []
    
    for thresh in [0.8]:
        print("")
        print(thresh)
        for pos in range(0, int(ts.sequence_length)-1000000, 2000000):
            true_overlaps = get_true_node_persistence(df1, pos, bp_grid)
            approx_overlaps, approx_muts = get_approx_node_persistence(df, pos, num_samples, bp_grid, thresh)
            relate_overlaps, relate_muts = get_relate_node_persistence(ts, target, pos, bp_grid)
            
            b1.extend(true_overlaps)
            b2.extend(approx_overlaps)
            b3.extend(relate_overlaps)
            m1.extend(approx_muts)
            m2.extend(relate_muts)
        
        # Plotting subplots for b1 vs b2, b2 vs b3, and m1 vs m2
        plt.clf()
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        
        # Define a function to handle plotting for each pair
        def plot_comparison(ax, x_data, y_data, xlabel, ylabel, label_fit, label_yx):
            df = pd.DataFrame({xlabel: x_data, ylabel: y_data})
            df = df[(df[xlabel] < 100) & (df[ylabel] < 100)]
            # Fit linear regression line
            x_data = df[xlabel].values.reshape(-1, 1)
            y_data = df[ylabel].values
            linear_regressor = LinearRegression()
            linear_regressor.fit(x_data, y_data)
            slope = linear_regressor.coef_[0]
            intercept = linear_regressor.intercept_
            print(f"Linear fit slope: {slope}")
            print(f"Linear fit intercept: {intercept}")
            # Predict values for linear fit
            y_pred = linear_regressor.predict(x_data)
            sns.scatterplot(data=df, x=xlabel, y=ylabel, alpha=0.5, ax=ax)
            if intercept >0:
                ax.plot(x_data, y_pred, 'r', label=label_fit + f': y={np.round(slope, 2)}x+{np.round(intercept, 2)}', linewidth=2)
            else:
                ax.plot(x_data, y_pred, 'r', label=label_fit + f': y={np.round(slope, 2)}x-{np.abs(np.round(intercept, 2))}', linewidth=2)
            ax.plot(x_data, x_data, 'g', label=label_yx, linewidth=2)  # y=x line
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper left', frameon=False)

        # 1. First subplot: b1 vs b2
        plot_comparison(axes[0], b1, b2, 'True overlaps', 'Approx. overlaps', 'Linear Fit', 'y=x')
        print(stats.spearmanr(b1, b2))

        # 2. Second subplot: b2 vs b3
        plot_comparison(axes[1], b1, b3, 'True overlaps', 'Relate overlaps', 'Linear Fit', 'y=x')
        print(stats.spearmanr(b1, b3))

        # 3. Third subplot: m1 vs m2
        plot_comparison(axes[2], m1, m2, 'Approx. mutations', 'Relate mutations', 'Linear Fit', 'y=x')
        
        # Overall title and save the figure
        plt.suptitle(f'Node Persistence Comparisons at Threshold {thresh}', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'node_persistence_comparisons_{thresh}.svg', dpi=300, transparent=True)
        import pdb; pdb.set_trace()

## maybe look for two consequtive mismatches ?