import tskit
import numpy as np
import pandas as pd
import copy
import numba

def get_coal_times(ts, target, bp_grid):
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
    df['num_bp_grid'] = df.apply(lambda row: np.sum((bp_grid >= row['left']) & (bp_grid <= row['right'])), axis=1)
    df = df[df['num_bp_grid'] > 0]
    return df

def get_true_node_persistence(df, pos, tree_window=100):
    left_array = df['left'].values
    right_array = df['right'].values
    target_tree_idx = np.argmax((left_array <= pos) & (right_array >= pos))
    start_idx = max(0, target_tree_idx - tree_window)
    end_idx = min(len(df), target_tree_idx + tree_window + 1)
    df_subset = df.iloc[start_idx:end_idx]
    target_seq = df.iloc[target_tree_idx]['sequence']
    number_of_overlaps = np.zeros(len(target_seq))
    for j, branch in enumerate(target_seq):
        branch_tuple = branch
        for i, row in df_subset.iterrows():
            seq = row['sequence']
            num_bp_grid = row['num_bp_grid']
            if branch_tuple in seq:
                number_of_overlaps[j] += num_bp_grid
    return number_of_overlaps

def convert_set_to_binary(s, num_samples):
    binary = np.zeros(num_samples, dtype='int')
    for i in s:
        binary[i] = 1
    return binary

def get_coal_descendants(ts, target, bp_grid, num_samples):
    seq_all_trees = []
    tree_intervals = [] 
    num_bp_grid_all = []
    for tree in ts.trees():
        num_bp_grid = np.searchsorted(bp_grid, tree.interval[1], side='right') - np.searchsorted(bp_grid, tree.interval[0])
        if num_bp_grid == 0: continue
        seq_ = []
        p = copy.deepcopy(target)
        while p != tree.root:
            parent = tree.parent(p)
            descendants_binary = convert_set_to_binary(set(tree.samples(parent)) - set(tree.samples(p)), num_samples)
            seq_.append(descendants_binary)
            p = parent
        num_bp_grid_all.append(num_bp_grid)
        seq_all_trees.append(np.array(seq_))
        tree_intervals.append(tree.interval)    
    df = pd.DataFrame({'interval': tree_intervals, 'sequence': seq_all_trees, 'num_bp_grid': num_bp_grid_all})
    df['left'] = df['interval'].apply(lambda x: x[0])
    df['right'] = df['interval'].apply(lambda x: x[1])
    return df

def get_overlap(set1, set2, num_samples):
    p1 = len(set1)/num_samples
    p2 = len(set2)/num_samples
    return (len(set1 & set2)/num_samples - p1*p2)/np.sqrt(p1-p1**2)/np.sqrt(p2-p2**2)

def update_intervals(target_seq, sequence, num_bp_grid, number_of_overlaps, found_mismatch, overlap_threshold):
    ncoal1 = len(target_seq)
    overlap_matrix = np.zeros((ncoal1, len(sequence)))
    num_samples = len(target_seq[0])
    overlap_matrix = (target_seq @ sequence.T) / num_samples
    p1_num = np.mean(target_seq, axis=1, keepdims=True)
    p2_num = np.mean(sequence, axis=1, keepdims=True)
    p1_denom = np.sqrt(p1_num - p1_num**2)
    p2_denom = np.sqrt(p2_num - p2_num**2)
    overlap_matrix = (overlap_matrix - p1_num @ p2_num.T) / (p1_denom @ p2_denom.T)
    candidate_pairs = []
    for j in range(ncoal1):
        for k in range(len(sequence)):
            if overlap_matrix[j, k] >= overlap_threshold:
                candidate_pairs.append((j, k, overlap_matrix[j, k]))
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_nodes_target = set()
    assigned_nodes_other = set()
    for j, k, overlap_value in candidate_pairs:
        if not found_mismatch[j]: 
            if j not in assigned_nodes_target:
                if k not in assigned_nodes_other:
                    assigned_nodes_target.add(j)
                    assigned_nodes_other.add(k)
                    number_of_overlaps[j] += num_bp_grid
    found_mismatch = np.ones(ncoal1, dtype='bool')
    for j in assigned_nodes_target:
        found_mismatch[j] = False
    return found_mismatch, number_of_overlaps

def get_approx_node_persistence(df, pos, overlap_threshold=0.8, tree_window=100):
    left_array = df['left'].values
    right_array = df['right'].values
    target_tree_idx = np.argmax((left_array <= pos) & (right_array >= pos))
    start_idx = max(0, target_tree_idx - tree_window)
    end_idx = min(len(df), target_tree_idx + tree_window + 1)
    df_subset_forward = df.iloc[target_tree_idx:end_idx]
    df_subset_backward = df.iloc[start_idx:target_tree_idx]
    target_seq = df.iloc[target_tree_idx]['sequence']
    number_of_overlaps = np.zeros(len(target_seq))
    ncoal1 = len(target_seq)
    found_mismatch_forward, found_mismatch_backward = np.zeros(ncoal1, dtype=bool), np.zeros(ncoal1, dtype=bool)
    # Forward pass
    for i, row in df_subset_forward.iterrows():
        found_mismatch_forward, number_of_overlaps = update_intervals(target_seq, row['sequence'], row['num_bp_grid'], number_of_overlaps, found_mismatch_forward, overlap_threshold)
        if np.all(found_mismatch_forward): break
    # Backward pass
    for i, row in df_subset_backward.iloc[::-1].iterrows():
        found_mismatch_backward, number_of_overlaps = update_intervals(target_seq, row['sequence'], row['num_bp_grid'], number_of_overlaps, found_mismatch_backward, overlap_threshold)
        if np.all(found_mismatch_backward): break
    return number_of_overlaps

def get_relate_node_persistence(ts, target, pos, bp_grid):
    tree = ts.at(pos)
    ts_edges = ts.edges()
    p = copy.deepcopy(target)
    number_of_overlaps = []
    while p != tree.root:
        edge = ts_edges[tree.edge(p)]
        edge_right = max(float(edge.metadata.decode('utf-8').split(" ")[1]), tree.interval[1])
        edge_left = min(float(edge.metadata.decode('utf-8').split(" ")[0]), tree.interval[0])
        number_of_overlaps.append(np.sum((bp_grid >= edge_left) & (bp_grid <= edge_right)))
        p = tree.parent(p)
    return number_of_overlaps

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import scipy.stats as stats
    from sklearn.linear_model import LinearRegression
    import matplotlib 
    from tqdm import tqdm
    import scipy.stats as stats

    font = {"size": 16}
    matplotlib.rc("font", **font)
    plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
    sns.set_palette('colorblind')
    
    ts = tskit.load("../../denisovan_sim_2024_08/trees/stdpopsim_homsap_chr1.trees")
    bp_grid = np.arange(0, ts.sequence_length, 1e4)
    num_samples = ts.num_samples
    target = 0
    df1 = get_coal_times(ts, target, bp_grid)
    df = get_coal_descendants(ts, target, bp_grid)
    
    for thresh in [0.5, 0.7, 0.8, 0.9]:
        b1, b2, b3 = [], [], []
        print("")
        print(thresh)
        for pos in tqdm(range(0, int(ts.sequence_length)-1000000, 1000000)):
            true_overlaps = get_true_node_persistence(df1, pos)
            approx_overlaps = get_approx_node_persistence(df, pos, num_samples, thresh)
            relate_overlaps = get_relate_node_persistence(ts, target, pos, bp_grid)
            b1.extend(true_overlaps)
            b2.extend(approx_overlaps)
            b3.extend(relate_overlaps)
        
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
        
        # Overall title and save the figure
        plt.suptitle(f'Node Persistence Comparisons at Threshold {thresh}', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'node_persistence_comparisons_{thresh}.svg', dpi=300, transparent=True)
        # import pdb; pdb.set_trace()

## maybe look for two consequtive mismatches ?