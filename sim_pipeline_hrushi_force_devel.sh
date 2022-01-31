#!/bin/bash

module load GCC
module load Python/3.7.4-GCCcore-8.3.0
module load R
source /well/myers/users/tgh473/python/tskit_venv/bin/activate  

PATH_TO_RELATE="/well/myers/users/tgh473/workspace/relate_devel/relate"
PATH_TO_RELATE_LIB="/well/myers/users/tgh473/workspace/relate_lib"
PATH_TO_GENETIC_MAP="/well/myers/users/tgh473/workspace/ghost_buster/msprime_maps"

#chr=${SGE_TASK_ID}
chr=7
N_dipl=25
N=$((2*${N_dipl}))

#Mbuti: Present-day African Mbuti
#LBK: Early European farmer (EEF)
#Sardinian: Present-day Sardinian
#Loschbour: Western hunter-gatherer (WHG)
#MA1: Upper Palaeolithic MAl'ta culture
#Han: Present-day Han Chinese
#UstIshim: early Palaeolithic Ust'-Ishim
#Neanderthal: Altai Neanderthal from Siberia

#python -m stdpopsim HomSap ${N} 2 ${N} 2 2 ${N} 2 2 -c chr${chr} -g HapMapII_GRCh37 -d AncientEurasia_9K19 -o stdpopsim_homsap_chr${chr}.trees &
#python3 -m tskit vcf --ploidy 2 stdpopsim_homsap_chr${chr}.trees > stdpopsim_homsap_chr${chr}.vcf

${PATH_TO_RELATE}/bin/RelateFileFormats --mode ConvertFromVcf -i stdpopsim_homsap_chr${chr} --haps stdpopsim_homsap_chr${chr}.haps --sample stdpopsim_homsap_chr${chr}.sample
gzip -f stdpopsim_homsap_chr${chr}.haps 
gzip -f stdpopsim_homsap_chr${chr}.sample
rm stdpopsim_homsap_chr${chr}.vcf 
#Rscript get_poplabels.R stdpopsim_homsap_chr${chr}
#
${PATH_TO_RELATE}/scripts/PrepareInputFiles/PrepareInputFiles.sh \
  --haps stdpopsim_homsap_chr${chr}.haps.gz \
  --sample stdpopsim_homsap_chr${chr}.sample.gz \
  --ancestor ancestor2.fa.gz \
  -o stdpopsim_homsap_filtered_chr${chr}
mv stdpopsim_homsap_filtered_chr${chr}.haps.gz stdpopsim_homsap_chr${chr}.haps.gz  
mv stdpopsim_homsap_filtered_chr${chr}.sample.gz stdpopsim_homsap_chr${chr}.sample.gz  

pushd ../devel_relate_trees_force_50/

${PATH_TO_RELATE}/bin/Relate --mode All \
  --consistency \
  --haps ../data_50/stdpopsim_homsap_chr${chr}.haps.gz \
  --sample ../data_50/stdpopsim_homsap_chr${chr}.sample.gz \
  --map ${PATH_TO_GENETIC_MAP}/genetic_map_GRCh37_chr${chr}.txt \
  --sample_ages ../data_50/sample_ages.txt \
  --coal stdpopsim_homsap.coal \
  --fb 1e4 \
  -m  1e-8 \
  -N 20000 \
  -o relate_homsap_chr${chr}

gzip relate_homsap_chr${chr}.anc
gzip relate_homsap_chr${chr}.mut

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertToTreeSequence \
  --anc relate_homsap_chr${chr}.anc.gz \
  --mut relate_homsap_chr${chr}.mut.gz \
  -o relate_homsap_chr${chr}

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertFromTreeSequence \
  --anc stdpopsim_homsap_chr${chr}.anc \
  --mut stdpopsim_homsap_chr${chr}.mut \
  -i ../data_50/stdpopsim_homsap_chr${chr}.trees

gzip -f stdpopsim_homsap_chr${chr}.anc
gzip -f stdpopsim_homsap_chr${chr}.mut

${PATH_TO_RELATE_LIB}/bin/Convert \
  --mode ConvertToTreeSequence \
  --anc stdpopsim_homsap_chr${chr}.anc.gz \
  --mut stdpopsim_homsap_chr${chr}.mut.gz \
  -o stdpopsim_homsap_conv_chr${chr}

popd



