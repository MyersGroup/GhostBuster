library(relater)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
#library(nls2)
library(stats)
library(parallel)
library(reticulate)
virtualenv_create(envname = "python_environment",python= "/apps/eb/2020b/skylake/software/Python/3.9.6-GCCcore-11.2.0/bin/python")
reticulate::use_virtualenv("python_environment", required = TRUE)
np <- import("numpy")

set.seed(1)

binsize <- 0.01
binmax  <- 1
recbins <- as.numeric(seq(0,binmax,binsize))
L       <- length(recbins)

filename_local_ancestry <- "../output/sgdp_punjabi_test_overall_membership_39.npy"	
filename_mut <- "/well/myers/users/tgh473/workspace/ghost_buster/SGDP/result/SGDP_Africa"

if(0){

  member <- np$load(filename_local_ancestry)
  #member   <- member[-1,]
  gwmember <- rowSums(member)
  gwmember <- gwmember/sum(gwmember)
  member <- member/gwmember

  chrs <- c(1)

  #Calculate genetic distance between adjacent trees.
  df <- data.frame()
  for(chr in chrs){

    mut    <- as.data.frame(read.mut(paste0(filename_mut, "_chr",chr,".mut.gz")))
    #mut    <- as.data.frame(read.mut(paste0("./relate_homsap_chr",chr,".mut.gz")))
    index  <- c(1, which(diff(mut$tree_index) > 0))
    if(max(index)+1 == nrow(mut)){
      index  <- unique(c(index, nrow(mut)))
      bp_raw <- c(as.numeric(mut[index,"pos_of_snp"]), max(mut$pos_of_snp)+1)
    }else{
      index  <- unique(c(index, nrow(mut)))
      bp_raw <- as.numeric(mut[index,"pos_of_snp"])
    }
    bp     <- (bp_raw[-length(bp_raw)] + bp_raw[-1])/2

    rec <- read.table(paste0("/well/myers/users/tgh473/workspace/ghost_buster/msprime_maps_sgdp/genetic_map_GRCh37_chr",chr,".txt"), header = T)
    rec <- stepfun(as.numeric(rec[-1,2]), as.numeric(rec[,4]))

    recrates_start <- rec(bp_raw[-length(bp_raw)])
    recrates_end   <- rec(bp_raw[-1])
    df <- rbind(df, cbind(CHR = chr, BP = bp, BP_start = bp_raw[-length(bp_raw)], BP_end = bp_raw[-1], rec = as.numeric(recrates_end - recrates_start), recdist_start = as.numeric(recrates_start), recdist_end = as.numeric(recrates_end)))

  }

  print(dim(member))
  print(head(df))
  df <- cbind(df, t(member))
  df %>% group_by(CHR) %>% filter(rec > 0, rec < 0.5) -> df
  df %>% group_by(CHR) -> df
  df$BP_bin <- cut(df$BP, breaks = seq(0,300e6,10e6))
  print(df)

  #read in recombination rates
  #for each tree in the genome, choose midpoint and store as BP
  #populate matrix of form "number of trees x cM away" with coancestry curve, i.e. pi*qj/(p * q)

  df_coan <- data.frame()

  for(comp1 in c("1", "2")){
    for(comp2 in c("1", "2")){

      df %>% group_by(CHR) %>% mutate(ind = 1:length(BP)) -> df_all
      df_all %>% group_by(CHR, BP_bin) %>% summarize( ind = list(ind[sample(1:length(BP),min(length(BP),30), replace = F)]) ) %>% unnest(cols = ind) -> df_chosen

      for(chr in chrs){

        df <- subset(df_all, CHR == chr)

        df[,comp1] <- as.numeric(as.matrix(df[,comp1]))
        df[,comp2] <- as.numeric(as.matrix(df[,comp2]))

        get_coan_for_tree <- function(i){

          df_tmp <- df
          p <- as.numeric(df_tmp[i,comp1])

          if(p > 0){
            #r <- as.numeric((df_tmp[i,"recdist_end"] + df_tmp[i,"recdist_start"])/2)
            r <- runif(n = 1, min = as.numeric(df_tmp[i,"recdist_start"]), max = as.numeric(df_tmp[i,"recdist_end"]))
            df_tmp[,"recdist_end"]   <- as.matrix(round(abs(df_tmp[,"recdist_end"] - r)/binsize) + 1)
            df_tmp[,"recdist_start"] <- as.matrix(round(abs(df_tmp[,"recdist_start"] - r)/binsize) + 1)
            df_tmp$recdist_end[df_tmp$recdist_end > L] <- L
            df_tmp$recdist_start[df_tmp$recdist_start > L] <- L
            df_tmp <- df_tmp[-i,]

            df_tmp %>% filter(recdist_start != L | recdist_end != L) -> df_tmp
            df_tmp$val <- as.numeric(as.matrix(df_tmp[,comp2] * p))
            df_tmp %>% group_by(CHR,BP) %>% mutate(ind = list(recdist_start:recdist_end)) %>% unnest(cols = ind) %>% group_by(ind) %>% summarize(res = mean(val)) -> df_tmp

            res   <- rep(NA,L)
            res[df_tmp$ind] <- df_tmp$res
            return(c(i, res))
          }else{

            res   <- rep(0,L)
            return(c(i,res))
          }

        }

        df_coan_tmp <- cbind(as.data.frame(t(mapply(get_coan_for_tree, subset(df_chosen, CHR == chr)$ind))), comp1 = comp1, comp2 = comp2)
        df_coan_tmp <- cbind(df_coan_tmp, CHR = chr, BP_bin = df$BP_bin[df_coan_tmp[,1]])
        df_coan     <- rbind(df_coan, df_coan_tmp)
      }

    }
  }

  df_coan %>% pivot_longer(cols = !c(CHR,BP_bin,comp1,comp2, V1), names_to = "dist", values_to = "coancestry") -> df_coan
  df_coan$dist <- as.matrix(df_coan$dist)
  foo          <- recbins + binsize/2
  df_coan      <- merge(df_coan, cbind(dist = paste0("V",1+1:length(foo)), val = foo), by = "dist")
  df_coan      <- subset(df_coan, !is.na(coancestry))

  save(df, df_coan, file = "coan.RData")

}


#fit coancestry curves
load("coan.RData")
df_coan$val <- as.numeric(df_coan$val)


if(1){
  num_boot <- 10

  df_params <- data.frame()

  for(c in c("2")){ 

    df_sub <- subset(df_coan, comp1 == paste0(c) & comp2 == paste0(c))

    all_blocks <- 1:length(  unique(paste0(df_sub$CHR,"-",df_sub$BP_bin))  )
    df_sub %>% group_split(CHR, BP_bin) -> df_sub

    for(i in 1:num_boot){

      blocks <- sort(sample(all_blocks,replace = T))
      bind_rows(df_sub[blocks]) %>% filter() %>% group_by(val) %>% summarize(coancestry = mean(coancestry)) -> df_resampled

      astart <- 3

      goo <- lm( log(df_resampled$coancestry[1:20] - 1) ~ df_resampled$val[1:20]  )
      astart <- exp(goo$coefficients[1])
      bstart <- -goo$coefficients[2]*100

      bstart <- 200

      # astart <- mean((df_resampled$coancestry-1)/exp(-as.numeric(df_resampled$val)/100*bstart))
      fitted <- nls(data = df_resampled, algorithm = "port",
                    coancestry ~ c+a*exp(-as.numeric(val)/100*b), start = list(a = astart, b = bstart, c = 1), control = list(minFactor = 1e-10, maxiter = 1e4), lower = c(0,0))
      df_params <- rbind(df_params, data.frame(comp1 = c, comp2 = c, boot = i, date = round(28*summary(fitted)$parameters[2,1]), a = summary(fitted)$parameters[1,1], c = summary(fitted)$parameters[3,1]))

    }

  }


  df_params %>% group_by(comp1, comp2) %>% summarize(mean = mean(date), lower = quantile(date, p = 0.025), upper = quantile(date, p = 0.975),
                                                     mean_a = mean(a), lower_a = quantile(a, p = 0.025), upper_a = quantile(a, p = 0.975),
                                                     mean_c = mean(c), lower_c = quantile(c, p = 0.025), upper_c = quantile(c, p = 0.975),
                                                     ) -> df_params_sum
  print(df_params_sum)
  print(df_params$date)

  df_params %>% group_by(comp1, comp2, boot) %>% mutate( val = list(seq(0,binmax,0.01)), 
                                                        coancestry = list(c+a*exp(-as.numeric(seq(0,binmax,0.01))/100*date/28))
                                                        ) %>% unnest(cols = c(val, coancestry)) %>%
  group_by(comp1, comp2, val) %>% summarize(mean = mean(coancestry), lower = quantile(coancestry, p = 0.025), upper = quantile(coancestry, p = 0.975)) -> df_curves



  df_curves$val <- as.numeric(df_curves$val)
  df_curves$mean <- as.numeric(df_curves$mean)
  df_curves$lower <- as.numeric(df_curves$lower)
  df_curves$upper <- as.numeric(df_curves$upper)

  # print(head(df_curves))
}

df_coan %>% group_by(comp1, comp2, val) %>% summarize(coancestry = mean(coancestry)) -> df_coan

p <- ggplot(df_coan) +  
geom_line(data = df_curves, aes(x = val, y = mean)) + 
geom_ribbon(data = df_curves, aes(x = val, ymin = lower, ymax = upper), alpha = 0.5) + 
geom_point(aes(x = val, y = coancestry)) + facet_grid(comp1~ comp2) +
theme_bw()

ggsave(p, file = "plot_coancestry.pdf", width = 10, height = 10)

#Do a block bootstrap

#also look at posterior vs recombination rate, posterior vs quality, posterior correlation to adjacent trees vs recombination rate


