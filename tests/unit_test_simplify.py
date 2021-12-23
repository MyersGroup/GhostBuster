import tskit
import numpy as np
import pandas as pd
from pathlib import Path
from em_true_ancient_sim_subsampled import fixed_parameters


def get_num_trees(ts, window_size):
    num_trees = 0
    tree = ts.first()
    prev_interval = tree.interval[0]
    # prev_interval = 0
    for tid in range(len(list(ts.trees()))):  # len(list(ts.trees()))
        if tree.interval[1] >= prev_interval + window_size:
            prev_interval = prev_interval + window_size
            num_trees += 1
        tree.next()
    return num_trees


def test_fixed_parameters(ts, poplabels, num_trees, window_size, sample_id):
    print(
        "Testing if simplifying within fixed_parameters file gives the same result as simplification before"
    )
    print(poplabels)
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)
    ts_simplified = ts.simplify(poplabels[poplabels.INCLUDE == 1].index.tolist())
    (a, b, c, d) = fixed_parameters(
        [ts_simplified],
        poplabels[poplabels.INCLUDE == 1],
        unique_groups,
        num_trees,
        window_size,
        sample_id,
    )
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)
    (a1, b1, c1, d1) = fixed_parameters(
        [ts],
        poplabels,
        unique_groups,
        num_trees,
        window_size,
        sample_id,
    )
    assert np.max(a - a1) < 1e-6
    assert np.max(b - b1) < 1e-6
    for i in range(len(c)):
        for j in range(len(c[i])):
            assert np.max(c[i][j] - c1[i][j]) < 1e-6
    assert d == d1


path = "/well/myers/users/tgh473/workspace/ghost_buster/sims/stdpopsim_ancient_small/relate_trees_force"
window_size = 100000
sample_id = [0, 1, 2]
ts = tskit.load(Path(path) / "relate_homsap_chr6.trees")
poplabels = pd.read_csv(Path(path) / "poplabels.txt", sep=" ")
num_trees = get_num_trees(ts, window_size)

# poplabels.INCLUDE = np.ones(len(poplabels), dtype=int)
# test_fixed_parameters(ts, poplabels, num_trees, window_size, sample_id)

poplabels.INCLUDE = np.array(np.random.rand(len(poplabels)) < 0.5, dtype=int)
for s in sample_id:
    poplabels.INCLUDE.iloc[s] = 1
test_fixed_parameters(ts, poplabels, num_trees, window_size, sample_id)
