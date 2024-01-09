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

* Running GhostBuster (supervised local ancestry inference):
```
python ../clean/RelateLocalAncestry/ghost_buster.py \
--trees ${tree} \
-k 2 \
--mode real \
--chr ${chr} \
--sample_id ${sample_id}  \
--num_iters 1 \
--rec genetic_map_hg38_ \
--init_at_truth 0 \
--output ${out} \
--masking_threshold 0.7 \
--n_repeats 1 \
--force_build 1e4 \
--sweep_num_iters 0 \
--hmm True \
--regress_out False \
--ignore_first_epoch True \
--ignore_last_epoch True \
--poplabels poplabels.txt \
--load_gamma relate.pairwise.coal \
--load_props "0.98 0.02" \
--groups nfe nea \
--t_admix_guess 1500 
```