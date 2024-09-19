import numpy as np
import pandas as pd
import tskit
import time
from tqdm import tqdm
import msprime
import argparse

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

def main(args):
    admix_source = args.admix_source 
    admix_dest = args.admix_dest
    Testpopulation = args.sample_id 
    bin_size = args.force_build if args.cm_grid is None else 1
    # Preload the tree sequences for all chromosomes
    ts_list = []
    chr_list = []
    for chrom in args.chrs:
        ts = tskit.load(args.trees + str(chrom) + ".trees")
        chr_list.append(chrom)
        ts_list.append(ts)
    de_seg_all_chr = {l: [] for l in Testpopulation}
    for chrom, ts in zip(chr_list, ts_list):
        seq_len = ts.sequence_length
        de_seg = {l: [] for l in Testpopulation}
        migrations = []
        for (m, migration) in enumerate(ts.migrations()):
            if migration.source in admix_source and migration.dest in admix_dest:
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
            recomb_map_msprime = msprime.RateMap.read_hapmap(args.rec + str(chrom) + ".txt")
            true_de_segs = combine_segs(de_seg_all_chr[l][chrom_no], seq_len, True, bin_size)
            chrom_data = np.vstack((chrom * np.ones(len(true_de_segs)), np.arange(0, seq_len, bin_size), true_de_segs, 1-true_de_segs)).T
            df = pd.DataFrame(chrom_data, columns=["chr", "pos", "prob_0", "prob_1"])
            df = df[(df.pos > recomb_map_msprime.left[0]) & (df.pos < recomb_map_msprime.right[-1])]
            df['genpos'] = recomb_map_msprime.get_cumulative_mass(df['pos'])
            if args.cm_grid is not None:
                m_grid = args.cm_grid/100
                df['genpos_rounded'] = (df['genpos'] / m_grid).astype('int')* m_grid
                df = df.groupby('genpos_rounded').first().reset_index()
                df = df.drop(columns=['genpos_rounded'])
                ## filter df to only keep values close to integral multiple of args.cm_grid
            all_local_ancestry.append(df)
        sample_local_ancestry_df = pd.concat(all_local_ancestry)
        sample_local_ancestry_df.to_csv(args.output + f"_{l}.csv", sep="\t", index=False)
        print("Genome-wide local ancestry for sample " +str(l) + " : " + str(sample_local_ancestry_df['prob_0'].mean()))
        print(f"Finished processing sample {l}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--admix_source", help="list of IDs of source", nargs="+", type=int, default=None)
    parser.add_argument("--admix_dest", help="list of IDs for destination", nargs="+", type=int, default=None)
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-fb",
        "--force_build",
        help="force build size to subsample the trees in bp",
        type=float,
        default=10000,
    )
    parser.add_argument(
        "-trees",
        "--trees",
        help="Location to trees in tskit format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-chrs",
        "--chrs",
        help="Comma-seperated list of chromosomes to be considered",
        type=str,
        default="1,2",
    )
    parser.add_argument("-r", "--rec", help="Filename of rec maps.", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="Output prefix (same as what was used when calling ghost_buster.py)",
        type=str,
        default='ground_truth',
    )
    parser.add_argument("--cm_grid", type=float, default=None, help="Store local ancestry information per cM instead")
    args = parser.parse_args()
    args.chrs = list(map(int, args.chrs.split(",")))
    print(args)
    main(args)


## python RelateLocalAncestry/helper/calc_ground_truth.py --admix_source 3 --admix_dest 4 --sample_id 50 51 52 53 54 55 56 57 58 59 --trees data/stdpopsim_homsap_chr --chrs 1 --rec genetic_map_GRCh37_chr --output ground_truth
## python RelateLocalAncestry/helper/calc_ground_truth.py --admix_source 3 --admix_dest 4 --sample_id 50 51 52 53 54 55 56 57 58 59 --trees data/stdpopsim_homsap_chr --chrs 5 --rec genetic_map_GRCh37_chr --output ground_truth_genpos --cm_grid 0.05
# for i in {50..59}
# do
# cp ground_truth_genpos_${i}.csv ground_truth_genpos_overall_membership_$((i+1))_sample_id_${i}.csv
# done