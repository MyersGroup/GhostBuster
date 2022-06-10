#!/bin/bash
source ~/.bashrc
module load Python/3.7.4-GCCcore-8.3.0
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

ind=51
path="../sims/stdpopsim_ancient_small/devel_relate_trees_force_50/"
out="force_filter_50/stdpopsim_homsap"
recomb_map="../msprime_maps/genetic_map_GRCh37"
trees="stdpopsim_homsap_conv"

### Hyperparameters ### 
start_time=4.5
end_time=6.5
num_epochs=9
ignore_first_epoch="True"

i=100
masking_thresh=0.5
window_size=10000
check_muts_target="False"
chrs="1,2,3,4,5"
chrs_test="6"
#######################

for k in 2
do

echo ${k}
## Making the tree stats and fixed params file for BayesOpt

#echo "1. Making tree stats and fixed params file for bayesopt" 
#python em_true_ancient_sim_subsampled.py --start_time ${start_time} --end_time ${end_time} --num_epochs 4 -k ${k} --path ${path} --trees ${trees} --chrs ${chrs} --sample_id ${ind} --ignore_first_epoch ${ignore_first_epoch} --mode real --relate_trees True --masking_thresh ${masking_thresh} --window_size ${window_size} --output ${out}_${k}_bayes -i 1 --rec ${recomb_map} --verbose False --check_muts_target ${check_muts_target}

## Running BayesOpt

#echo "2. Running Bayesian Optimization"
#python bayesopt.py  --start_time ${start_time} --end_time ${end_time} --path ${path} --trees ${trees} --chrs ${chrs} --tree_stats_file_name ${out}_${k}_bayes_tree_stats_${ind}_${window_size}_True_${chrs}_${masking_thresh}.pkl  --fixed_params_file_name ${out}_${k}_bayes_fixed_params_${ind}_${window_size}_True_${chrs}_${masking_thresh}.pkl  --sample_id ${ind} --ignore_first_epoch ${ignore_first_epoch} -k ${k} --num_epochs 4 --output ${out}_${k}_overall_membership_bayes.npy --bayes_steps 100


## Running EM

echo "3. Running the EM"
python em_true_ancient_sim_subsampled.py --mode sim --chr ${chrs} --relate_trees True --masking_thresh ${masking_thresh} --plot_intermediate_gammas True --window_size ${window_size} --sample_id ${ind} --init_at_truth 0 --path ${path} --trees ${trees} --output ${out}_${k} -k ${k} -i ${i} -start_time ${start_time} -end_time ${end_time} --num_epochs ${num_epochs} -ignore_first_epoch ${ignore_first_epoch} --rec ${recomb_map} --verbose True --check_muts_target ${check_muts_target} --load_membership ${out}_${k}_overall_membership_bayes.npy 

## Getting test loglikelihood

#echo "4. Evaluating the test-loglikelihood"
#python em_true_ancient_sim_subsampled.py --evaluate_gamma False --mode real --chr ${chrs_test} --relate_trees True --masking_thresh ${masking_thresh} --plot_intermediate_gammas False --window_size ${window_size} --sample_id ${ind} --path ${path} --trees ${trees} --output ${out}_${k} -k ${k} -start_time ${start_time} -end_time ${end_time} --num_epochs ${num_epochs} --ignore_first_epoch ${ignore_first_epoch} --rec ${recomb_map} --load_props ${out}_${k}_props_${ind}_iter$((i-1)).npy --load_gamma ${out}_${k}_gamma_${ind}_iter$((i-1)).npy

done
