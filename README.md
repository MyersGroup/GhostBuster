# GhostBuster
Hrushikesh Loya, Leo Speidel, Simon Myers

## About the code
* `ghost_buster.py`: Main script to run the EM and find ghost populations
* `Rscript/plot.R`: Supporting plotting script to get population size plots
* `get_local_anc.py`: Supporting script to get true local ancestry in simulations

## Installation
`pip install -r requirements.txt`

## List of options
`python ghost_buster.py --help`

## Examples
* Getting true local ancestry from the simulation:
`python get_local_anc.py --path example/ --migr_time 2272 --sample_id 50-100 --chrs 22`

* Running GhostBuster (single fitting):
```
python ghost_buster.py \
    --mode sim \
    --trees example/stdpopsim_homsap_chr \
    --poplabels example/poplabels.txt \
    --ground_truth example/local_ancestry_chr \ 
    --rec example/genetic_map_GRCh37_chr \
    --sample_id 51 \
    --chr 22 \
    --output example/stdpopsim_homsap \
    --init_at_truth 0
```
## Local ancestry inference
* Running GhostBuster (supervised local ancestry inference):
```
python ghost_buster.py \
--trees ${tree} \
--mode real \
--chr ${chr} \
--sample_id ${sample_id}  \
--num_iters 1 \
--rec genetic_map_hg38_ \
--output ${out} \
--n_repeats 1 \
--sweep_num_iters 0 \
--poplabels poplabels.txt \
--load_gamma relate.pairwise.coal \
--load_props "0.98 0.02" \
--groups nfe nea \
--t_admix_guess 1500 
```

## Parallelizing ghostbuster
* Generating fixed params and branch persistence files parallely across chromosomes or samples:
```
python ghost_buster.py \
--trees ${tree} \
--mode real \
--chr ${chr} \
--sample_id ${sample}   \
--num_iters 0 \
--rec genetic_map_hg38_ \
--output ${out} \
--n_repeats 1 \
--sweep_num_iters 0 \
--poplabels poplabels.txt \
--tree_stats_file_prefix output/tree_stats \
--branch_persistence_file_prefix output/branch_persistence  \
--fixed_params_file_prefix output/fixed_params
```

* Running ghostbuster without tskit trees (more memory efficient):
```
python ghost_buster.py \
--mode real \
--chr ${chr} \
--sample_id ${sample} \
--num_iters 200 \
--rec genetic_map_hg38_ \
--output ${out} \
--n_repeats 40 \
--sweep_num_iters 20 \
--poplabels poplabels.txt \
--tree_stats_file_prefix output/tree_stats \
--branch_persistence_file_prefix output/branch_persistence  \
--fixed_params_file_prefix output/fixed_params
```

## More options
* A complete list of options available for ghostbuster:
`python ghost_buster.py --help`
