#!/bin/bash
source ~/.bashrc
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

#ind=51
path="../SGDP/SGDP_aDNA_April2022/"
out="sgdp_output/han"

for k in 2
do
echo ${k}
python em_true_ancient_sim_subsampled.py \
--mode real \
--chr 1,2,3,4,5 \
--relate_trees True \
--masking_thresh 0.5 \
--plot_intermediate_gammas True \
--force_build 10000 \
--sample_id 167 \
--path ${path} \
--trees SGDP_aDNA_pp_b \
--output ${out}_${k} \
-k ${k} \
-i 200 \
-start_time 4.5 \
-end_time 6.0 \
--num_epochs 9 \
-ignore_first_epoch True \
--rec ../msprime_maps/genetic_map_GRCh37 \
--verbose False \
--check_muts_target True
done
