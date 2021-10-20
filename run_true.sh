#!/bin/bash
source ~/.bashrc
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

ind=0
path="../sims/stdpopsim_ancient_small/relate_trees/"

for k in 1 2 3
do
echo ${k}
python em_true_ancient_sim_subsampled.py \
--mode real \
--chr 21 \
--relate_trees True \
--masking_thresh 0.8 \
--plot_intermediate_gammas True \
--window_size 0 \
--sample_id ${ind} \
--init_at_truth 0 \
--path ${path} \
--trees stdpopsim_homsap_conv \
--output stdpopsim_ancient_small/stdpopsim_homsap_conv_${k} \
-k ${k} \
-i 500 \
-start_time 4 \
-end_time 6 \
--num_epochs 7 \
-ignore_first_epoch True \
--rec ../msprime_maps/genetic_map_GRCh37 \
--verbose False
python em_true_ancient_sim_subsampled.py --evaluate_gamma False --mode real --chr 22 --relate_trees True --masking_thresh 0.8 --plot_intermediate_gammas False --window_size 0 --sample_id ${ind} --path ${path} --trees stdpopsim_homsap_conv --output stdpopsim_ancient_small/stdpopsim_homsap_conv_${k} -k ${k} -start_time 4 -end_time 6 --num_epochs 7 --ignore_first_epoch True --rec ../msprime_maps/genetic_map_GRCh37 --load_props stdpopsim_ancient_small/stdpopsim_homsap_conv_${k}_props_${ind}_iter499.npy --load_gamma stdpopsim_ancient_small/stdpopsim_homsap_conv_${k}_gamma_${ind}_iter499.npy
done
