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

pushd true_trees/
#python3 -m tskit vcf --ploidy 2 ../stdpopsim_homsap_chr${chr}.trees > stdpopsim_homsap_chr${chr}.vcf

${PATH_TO_RELATE}/bin/RelateFileFormats --mode ConvertFromVcf -i ../stdpopsim_homsap_chr${chr} --haps stdpopsim_homsap_chr${chr}.haps --sample stdpopsim_homsap_chr${chr}.sample
gzip -f stdpopsim_homsap_chr${chr}.haps
gzip -f stdpopsim_homsap_chr${chr}.sample
rm stdpopsim_homsap_chr${chr}.vcf

${PATH_TO_RELATE}/scripts/PrepareInputFiles/PrepareInputFiles.sh \
  --haps stdpopsim_homsap_chr${chr}.haps.gz \
  --sample stdpopsim_homsap_chr${chr}.sample.gz \
  --ancestor ../ancestor2.fa.gz \
  -o stdpopsim_homsap_filtered_chr${chr}
mv stdpopsim_homsap_filtered_chr${chr}.haps.gz stdpopsim_homsap_chr${chr}.haps.gz
mv stdpopsim_homsap_filtered_chr${chr}.sample.gz stdpopsim_homsap_chr${chr}.sample.gz

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertFromTreeSequence \
  --anc stdpopsim_homsap_chr${chr}.anc \
  --mut stdpopsim_homsap_chr${chr}.mut \
  -i ../stdpopsim_homsap_chr${chr}.trees

gzip -f stdpopsim_homsap_chr${chr}.anc
gzip -f stdpopsim_homsap_chr${chr}.mut

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertToTreeSequence \
  --anc stdpopsim_homsap_chr${chr}.anc.gz \
  --mut stdpopsim_homsap_chr${chr}.mut.gz \
  -o stdpopsim_homsap_conv_chr${chr}

popd

