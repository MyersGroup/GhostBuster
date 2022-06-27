#!/bin/bash
source ~/.bashrc
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

#ind=51
out="output/postprocess_relate_homsap"
path="../sims/stdpopsim_ancient_small/relate_postprocess_recomb_filter/"
tree="postprocess_relate_homsap_chr"

python ghost_buster.py \
--mode sim \
--chr 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
--sample_id 51 52 53 54 55 \
-i 200 \
--trees ${path}${tree} \
--poplabels ${path}poplabels.txt \
--rec ../msprime_maps_filtered/genetic_map_GRCh37_chr \
--mutden ${path}${tree} \
--allmuts ${path}${tree} \
--ground_truth_path ${path}local_ancestry_chr \
--init_at_truth 1 \
--opportunity_filter 1 \
--output ${out} \
--masking_threshold 0.5 \
#--load_mask output/postprocess_relate_homsap_mask_51.csv
