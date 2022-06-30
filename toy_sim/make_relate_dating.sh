#!/bin/bash
module load Boost
module load GCC
module load Python/3.7.4-GCCcore-8.3.0
module load R
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

PATH_TO_RELATE="/well/myers/users/tgh473/workspace/relate_devel/relate"
PATH_TO_RELATE_LIB="/well/myers/users/tgh473/workspace/relate_lib"
PATH_TO_GENETIC_MAP="/well/myers/users/tgh473/workspace/ghost_buster/msprime_maps_filtered"

chr=${SGE_TASK_ID}

pushd relate_dating/

${PATH_TO_RELATE}/bin/RelateCoalescentRate --mode ReEstimateBranchLengths \
  -i ../true_trees/stdpopsim_homsap_chr${chr} \
  --coal ../stdpopsim_homsap.coal \
  -m  1.25e-8 \
  -o relate_homsap_chr${chr}

gzip -f relate_homsap_chr${chr}.anc
gzip -f relate_homsap_chr${chr}.mut

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertToTreeSequence \
  --anc relate_homsap_chr${chr}.anc.gz \
  --mut relate_homsap_chr${chr}.mut.gz \
  -o relate_homsap_chr${chr}

${PATH_TO_RELATE}/bin/RelateExtract --mode CountMutonBranches --anc relate_homsap_chr${chr}.anc.gz --mut relate_homsap_chr${chr}.mut.gz  -o relate_homsap_chr${chr}
${PATH_TO_RELATE}/bin/RelateMutationRate --mode MutationDensity -i relate_homsap_chr${chr} -o relate_homsap_chr${chr}_0 --pop_of_interest 0 --bins 3,5.5,0.357142857
popd
