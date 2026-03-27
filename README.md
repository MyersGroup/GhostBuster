# GhostBuster

## About the code
* `ghost_buster.py`: Main script to run the EM and find ghost populations
* `src/`: Supporting Python/R code (`plotting/`, `helper/`, and utility modules)
* `real_data_mask/`: Genome-build mask files used for filtering

## Installation
```
conda create -n gb python=3.13 -y
conda activate gb
conda install -c conda-forge gsl
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## List of options
```
python ghost_buster.py --help
```

| Option                                  | Datatype            | Description                                                                                                           |
|-----------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------|
| `--sample_id`                           | list<int>           | Enter space-separated list of the indices of haplotypes to analyze local ancestry for                                 |
| `--trees`                               | string              | Directory of tskit trees                                                                                              |
| `--poplabels`                           | string              | Population labels file                                                                                                |
| `--rec`                                 | string              | Filename of recombination map                                                                                         |
| `--chrs`                                | list<int>           | Comma-separated list of chromosomes to process                                                                        |
| `--output`                              | string              | Output prefix                                                                                                         |
| `--num_clusters`                        | int                 | Number of clusters to infer by EM                                                                                     |
| `--mut_scaling`                         | float               | Scale likelihood by presence of mutation on lineage (improves robustness to bottlenecks)                              |
| `--ypg`                                 | float               | Years per generation (default: 28)                                                                                    |
| `--node_persist_thresh`                 | float               | Correlation threshold above which nodes are considered equivalent                                                     |
| `--cm_grid`                             | float               | Store local-ancestry information per centiMorgan                                                                      |
| `--masking_threshold`                   | float               | Remove top 100*x percentile of high-recombination regions                                                                 |
| `--t_admix_guess`                       | float               | Initial guess for time of admixture                                                                                   |
| `--start_time`                          | float               | Start time for population-size plots (log-scale)                                                                      |
| `--end_time`                            | float               | End time for population-size plots (log-scale)                                                                        |
| `--num_epochs`                          | int                 | Number of epochs                                                                                                      |
| `--seed`                                | int                 | Random seed for reproducibility                                                                                        |
| `--force_build`                         | int                 | Force build size to subsample the trees (in bp)                                                                       |
| `--num_iters`                           | int                 | Number of EM iterations                                                                                                |
| `--n_repeats`                           | int                 | Number of EM restarts when randomly initializing local ancestry; best run chosen                                      |
| `--sweep_num_iters`                     | int                 | Number of EM iterations per random sweep                                                                               |
| `--groups`                              | list<string>        | Space-separated list of source groups (e.g. `Nea CHB`)                                                                 |
| `--load_gamma`                          | string              | Starting γ values from file                                                                                           |
| `--load_mask`                           | string              | Load mask CSV file (columns: chr, tree_position_left)                                                                 |
| `--load_props`                          | string              | Starting τ values from file or space-separated list                                                                   |
| `--tree_stats_file_prefix`              | string              | Prefix for output tree-statistics file                                                                                 |
| `--branch_persistence_file_prefix`      | string              | Prefix for output branch-persistence file                                                                              |
| `--fixed_params_file_prefix`            | string              | Prefix for output fixed-parameters file                                                                                |
| `--load_membership`                     | string              | Load membership array from `.npy` file                                                                                 |
| `--genome_build`                        | string              | Genome build for filtering (e.g. `hg38`, `hg37`, or `None`)                                                           |
| `--gt_ref`                              | string              | Local ancestry of the reference panel                                                                                  |
| `--ignore_coal_between_targets`         | boolean             | Ignore coalescence events between target samples                                                                       |
| `--hmm`                                 | boolean             | Run HMM (`True`) or treat windows as independent (`False`)                                                            |
| `--ignore_first_epoch`                  | boolean             | Ignore first epoch when calculating local ancestry in EM                                                              |
| `--ignore_last_epoch`                   | boolean             | Ignore last epoch when calculating local ancestry in EM                                                               |


## Examples
```
mkdir output/

python ghost_buster.py \
    --genome_build hg37 \
    --trees example/relate_chr \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 0 \
    --chr 22 \
    --output output/relate
```

## Running GhostBuster on a SLURM cluster
You can split the EM inference by chromosome (e.g. via a SLURM array) and then merge all the pickles in one final pass.
```
#SBATCH --array=1-10
python ghost_buster.py \
    --genome_build hg37 \
    --trees example/relate_chr \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 0 \
    --chr ${SLURM_ARRAY_TASK_ID}$ \
    --output output/relate \
    --num_iters 1 \
    --n_repeats 1 \
    --sweep_num_iters 1
```

Once every chromosome has its pickle in `example/relate`, rerun GhostBuster without `--trees` to aggregate:
```
python ghost_buster.py \
    --genome_build hg37 \
    --poplabels example/poplabels.txt \
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 51 \
    --chr 1,2,3,4,5,6,7,8,9,10 \
    --output example/relate 
```
