import numpy as np
import pandas as pd
import tskit
import time
from tqdm import tqdm

def combine_segs(segs, seq_len, get_segs=False, bin_size=10000):
    # Get sequence length from the segment boundaries
    merged = np.empty([0, 2])
    if len(segs) == 0:
        if get_segs:
            return []
        else:
            return np.zeros(int(seq_len / bin_size))
    sorted_segs = segs[np.argsort(segs[:, 0]), :]
    for higher in sorted_segs:
        if len(merged) == 0:
            merged = np.vstack([merged, higher])
        else:
            lower = merged[-1, :]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1, :] = (lower[0], upper_bound)
            else:
                merged = np.vstack([merged, higher])
    
    if get_segs:
        # Create an array to hold local ancestry at every 10kb bin
        local_ancestry = np.zeros(int(seq_len / bin_size))
        for segment in merged:
            start_bin = int(segment[0] // bin_size)
            end_bin = int(segment[1] // bin_size)
            if end_bin > start_bin:
                local_ancestry[start_bin+1:end_bin+1] = 1
        return local_ancestry
    else:
        return np.sum(merged[:, 1] - merged[:, 0]) / seq_len

admix_source = 3 
admix_dest = 4
Testpopulation = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # ts.get_samples(1)
bin_size = 10000  # 10kb bin size

# Preload the tree sequences for all chromosomes
ts_list = []
chr_list = []
for chrom in range(1, 6):
    ts = tskit.load("data/stdpopsim_homsap_chr" + str(chrom) + ".trees")
    chr_list.append(chrom)
    ts_list.append(ts)

de_seg_all_chr = {l: [] for l in Testpopulation}
for chrom, ts in zip(chr_list, ts_list):
    seq_len = ts.sequence_length
    de_seg = {l: [] for l in Testpopulation}
    migrations = []
    for (m, migration) in enumerate(ts.migrations()):
        if migration.source == admix_source and migration.dest == admix_dest:
            migrations.append(
                {
                    "left": migration.left,
                    "right": migration.right,
                    "node": migration.node,
                    "time": migration.time,
                    "dest": migration.dest,
                    "source": migration.source,
                }
            )
    sorted_migrations = sorted(migrations, key=lambda d: d["left"])
    for mr in tqdm(sorted_migrations):
        for tree in ts.trees(leaf_lists=True):
            if mr['left'] > tree.get_interval()[0]:
                continue
            if mr['right'] <= tree.get_interval()[0]:
                break
            for node in tree.leaves(mr['node']):
                if node in Testpopulation:
                    de_seg[node].append(tree.get_interval())  
    for l in Testpopulation:
        de_seg_all_chr[l].append(np.array(de_seg[l]))

for l in Testpopulation:
    all_local_ancestry = []    
    for chrom_no, (chrom, ts) in enumerate(zip(chr_list, ts_list)):
        seq_len = ts.sequence_length
        true_de_segs = combine_segs(de_seg_all_chr[l][chrom_no], seq_len, True, bin_size)
        chrom_data = np.vstack((chrom * np.ones(len(true_de_segs)), np.arange(0, seq_len, bin_size), true_de_segs)).T
        all_local_ancestry.append(pd.DataFrame(chrom_data, columns=["chr", "pos", "gt"]))
    sample_local_ancestry_df = pd.concat(all_local_ancestry)
    print("Genome-wide local ancestry for sample " +str(l) + " : " + str(sample_local_ancestry_df['gt'].mean()))
    sample_local_ancestry_df.to_csv(f"ground_truth_{l}.csv", sep="\t", index=False)
    print(f"Finished processing sample {l}")
