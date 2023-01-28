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
`python get_local_anc.py --path example/ --migr_time 2272 --sample_id 51 --chrs 22`

* Running GhostBuster on true trees:
```
python ghost_buster.py 
    --mode sim 
    --trees example/stdpopsim_homsap_chr 
    --poplabels example/poplabels.txt 
    --ground_truth example/local_ancestry_chr  
    --rec example/genetic_map_GRCh37_chr 
    --sample_id 51 
    --chr 22 
    --output example/stdpopsim_homsap 
    --init_at_truth 1
```
