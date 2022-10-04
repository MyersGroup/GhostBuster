#!/bin/bash
source ~/.bashrc
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

sam=39
out="output/sgdp_ghost_sardinian"
path="/well/myers/users/tgh473/workspace/ghost_buster/SGDP/result/"
tree="SGDP_Africa_chr"

python ghost_buster.py \
--mode real \
--chr 1,2,3,4,5 \
--sample_id 45 46 \
-i 200 \
--trees ${path}${tree} \
--poplabels ${path}poplabels.txt \
--rec ../msprime_maps_sgdp/genetic_map_GRCh37_chr \
--output ${out} \
--masking_threshold 0.7 \
--start_time 4.5 \
--end_time 6.0 \
--num_clusters 2 \
--n_repeats 20 \
# --mutden ${path}${tree} \
# --allmuts ${path}${tree} \
#--ground_truth_path ${path}local_ancestry_chr \
#--init_at_truth 1 \
#--opportunity_filter 1 \
#--load_mask output/postprocess_relate_homsap_mask_51.csv

# python ghost_buster.py --mode real -fb 0.1 --chr 1 --n_repeats 1 --sample_id 39 -i 1 --trees ${path}${tree} --poplabels ${path}poplabels.txt --rec ../msprime_maps_sgdp/genetic_map_GRCh37_chr --output ${out}_test --masking_threshold 0.0 --load_gamma ${out}_gamma_39_40.npy --load_props ${out}_props_39_40.npy

