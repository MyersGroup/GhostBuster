from tqdm import tqdm
import numpy as np
import pandas as pd
import tskit
from pathlib import Path
import argparse
import pdb


def get_migrating_tracts_with_id(ts, path, migr_time, samples=None):
    if samples is None:
        samples = list(ts.first().samples())
    N = 0
    for s in ts.first().samples():
        N += 1
    migrations = []
    for (m, migration) in enumerate(ts.migrations()):
        if np.abs(migration.time - migr_time) < 1:
            migrations.append(
                {
                    "left": migration.left,
                    "right": migration.right,
                    "node": migration.node,
                    "time": migration.time,
                    "dest": migration.dest,
                }
            )
    sorted_migrations = sorted(migrations, key=lambda d: d["left"])
    migration_array = [[[]] for _ in range(N)]
    tree = ts.first()
    for (m, migration) in enumerate(sorted_migrations):
        print(m)
        for tree in ts.trees(leaf_lists=True):
            if migration["left"] > tree.get_interval()[0]:
                continue
            if migration["right"] <= tree.get_interval()[0]:
                break
            for i in tree.leaves(migration["node"]):
                migration_array[i].append(
                    [tree.interval[0], tree.interval[1], migration["dest"]]
                )

    for j in samples:
        migrating_tracts_i = migration_array[j][1:]
        if migrating_tracts_i != []:
            mig = []
            startpos = migrating_tracts_i[0][0]
            for i in range(0, len(migrating_tracts_i) - 1):
                if (
                    migrating_tracts_i[i][1] != migrating_tracts_i[i + 1][0]
                    or migrating_tracts_i[i][2] != migrating_tracts_i[i + 1][2]
                ):
                    mig.append(
                        [startpos, migrating_tracts_i[i][1], migrating_tracts_i[i][2]]
                    )
                    startpos = migrating_tracts_i[i + 1][0]
            if len(migrating_tracts_i) > 0:
                mig.append(
                    [
                        startpos,
                        migrating_tracts_i[len(migrating_tracts_i) - 1][1],
                        migrating_tracts_i[len(migrating_tracts_i) - 1][2],
                    ]
                )
        else:
            mig = [[ts.first().interval[0], ts.last().interval[1], 2]]
        if mig != []:
            np.savetxt(
                path / str("local_ancestry_chr" + str(chr) + "_" + str(j) + ".csv"),
                np.array(mig),
                delimiter=",",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-chrs",
        "--chrs",
        help="Comma-seperated list of chromosomes to be considered",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22",
    )
    parser.add_argument(
        "--path",
        help="Path of the true trees",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--migr_time",
        help="Migration time around which to search",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    sample_id = []
    for i in range(len(args.sample_id)):
        if "-" in args.sample_id[i]:
            sample_id.extend(
                np.arange(
                    int(args.sample_id[i].split("-")[0]),
                    int(args.sample_id[i].split("-")[1]),
                ).tolist()
            )
        else:
            sample_id.append(int(args.sample_id[i]))
    args.sample_id = sample_id

    for chr in tqdm(args.chrs.split(",")):
        ts = tskit.load(
            Path(args.path) / str("stdpopsim_homsap_chr" + str(chr) + ".trees")
        )
        get_migrating_tracts_with_id(
            ts, Path(args.path), args.migr_time, args.sample_id
        )
