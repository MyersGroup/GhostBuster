#!/bin/bash
module load Boost
imodule load GCC
module load Python/3.7.4-GCCcore-8.3.0
module load R
source /well/myers/users/tgh473/python/tskit_venv/bin/activate

PATH_TO_RELATE="/well/myers/users/tgh473/workspace/relate_devel/relate"
PATH_TO_RELATE_LIB="/well/myers/users/tgh473/workspace/relate_lib"
PATH_TO_GENETIC_MAP="/well/myers/users/tgh473/workspace/ghost_buster/msprime_maps_filtered"

chr=${SGE_TASK_ID}

pushd relate_randomize/

${PATH_TO_RELATE}/bin/Relate --mode PostProcess --randomise \
  -i ../relate_trees/relate_homsap_chr${chr} \
  --haps ../true_trees/stdpopsim_homsap_chr${chr}.haps.gz \
  --sample ../true_trees/stdpopsim_homsap_chr${chr}.sample.gz \
  -o postprocess_relate_homsap_chr${chr}

gzip -f postprocess_relate_homsap_chr${chr}.anc
gzip -f postprocess_relate_homsap_chr${chr}.mut

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertToTreeSequence \
  --anc postprocess_relate_homsap_chr${chr}.anc.gz \
  --mut postprocess_relate_homsap_chr${chr}.mut.gz \
  -o postprocess_relate_homsap_chr${chr}

${PATH_TO_RELATE}/bin/RelateExtract --mode CountMutonBranches --anc postprocess_relate_homsap_chr${chr}.anc.gz --mut postprocess_relate_homsap_chr${chr}.mut.gz  -o postprocess_relate_homsap_chr${chr}
${PATH_TO_RELATE}/bin/RelateMutationRate --mode MutationDensity -i postprocess_relate_homsap_chr${chr} -o postprocess_relate_homsap_chr${chr}_0 --pop_of_interest 0 --bins 3,5.5,0.357142857
popd
