#!/bin/bash
source ~/.bashrc
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/new_venv/bin/activate

#ind=51
out="output/nea_gt_ref40"
path="../sims/nea_const_recomb_0.2/true_trees/"
tree="stdpopsim_homsap_conv_chr"

python ghost_buster.py \
-k 2 \
--mode sim \
--chr 22 \
--sample_id 51 \
-i 1 \
--trees ${path}${tree} \
--poplabels ${path}poplabels_ghost10.txt \
--rec ../msprime_maps_filtered_temp/genetic_map_GRCh37_chr \
--ground_truth_path ../sims/nea_const_recomb_0.2/local_ancestry/local_ancestry_chr \
--init_at_truth 0 \
--opportunity_filter 0 \
--output ${out} \
--masking_threshold 0.9 \
--n_repeats 1 \
--start_time 4 \
--load_props output/nea_gt_ref38_props_all.npy \
--load_gamma output/nea_gt_ref38_gamma_all.npy \
# --load_mask prior_based_mask.csv

#python ghost_buster.py -k 1 --mode real --chr 2,4,6,8,10,12,14,16,18,20,22 --n_repeats 1 --sample_id 51 -i 1 --trees ${path}${tree} --poplabels ${path}poplabels.txt --rec ../msprime_maps_filtered_temp/genetic_map_GRCh37_chr --output ${out}_test --masking_threshold 0.8 --load_gamma ${out}_gamma_51.npy --load_props ${out}_props_51.npy
