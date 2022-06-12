# GhostBuster
Hrushikesh Loya, Leo Speidel, Simon Myers

## About the code
* `ghost_buster.py`: Main script to run the EM and find ghost populations
* `Rscript/plot.R`: Supporting plotting script to get population size plots
* `true_local_ancestry/get_local_anc.py`: Supporting script to get true local ancestry in simulations

## Installation (tested on Windows)
`conda env create -f environment.yml -n ghost_buster`

## Examples
* True trees and recombination rate filtering:
`python ghost_buster.py --mode sim --trees example/stdpopsim_homsap_chr --poplabels example/poplabels.txt --ground_truth example/local_ancestry_chr  --rec example/genetic_map_GRCh37_chr --sample_id 51 --chr 22 --output example/stdpopsim_homsap --init_at_truth 1`

* Relate trees and recombination rate and opportunity filtering:
`python ghost_buster.py --mode sim --trees example/relate_homsap_chr --poplabels example/poplabels.txt --ground_truth example/local_ancestry_chr  --rec example/genetic_map_GRCh37_chr --mutden example/relate_homsap_chr --allmuts example/relate_homsap_chr --opportunity_filter 1 --sample_id 51 52 --chr 22 --output example/relate_homsap --init_at_truth 1`