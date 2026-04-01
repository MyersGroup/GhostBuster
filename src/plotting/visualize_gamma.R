library(relater)
library(ggplot2)
library(cowplot)
library(dplyr)
library(tidyr)
library(pammtools)
library(grid)
library(ggplotify)
library(nls2)
library(ggthemes)
library(reticulate)
#virtualenv_create(envname = "python_environment",python= "/apps/eb/2020b/skylake/software/Python/3.9.6-GCCcore-11.2.0/bin/python")
virtualenv_create(envname = "python_environment")
reticulate::use_virtualenv("python_environment", required = TRUE)
np <- import("numpy")

args <- commandArgs(trailingOnly = TRUE)
filename         <- args[1]
sample_id_label  <- args[2]

colours <- c("#FF5CA8", "#FAF19E", "#BDEFD0", "#8DD4F7", "#CF90F4", "#d0f4de")

## Read .coal file
read.coal <- function(filename){
  groups <- as.matrix(utils::read.table(filename, nrow = 1))
  epochs <- as.matrix(utils::read.table(filename, nrow = 1, skip = 1))
  coal <- utils::read.table(filename, skip = 2)

  coal[,1] <- as.factor(coal[,1])
  coal[,2] <- as.factor(coal[,2])

  colnames(coal)[-c(1:2)] <- epochs
  colnames(coal)[1:2] <- c("group1", "Reference")
  coal <- reshape2::melt(coal, id.vars = c("group1", "Reference"))
  colnames(coal)[3:4] <- c("epoch.start", "haploid.coalescence.rate")
  coal <- coal[order(paste(coal[,1], coal[,2], sep = "")),]
  coal$epoch.start <- as.numeric(as.matrix(coal$epoch.start))
  coal$Reference <- groups[as.numeric(as.matrix(coal$Reference))+1]

  return(coal)
}

## Read and reshape .tau file
prop <- 100*np$load(paste0(filename, '_props_nohmm_', sample_id_label, '.npy'))

## Infer number of components from .tau file
num_components <- length(prop)
print(paste("Number of components:", num_components))

## Prepare the coal data
coal <- data.frame()
filename_coal <- paste0(filename, '_nohmm_', sample_id_label, '.coal.all')
if (file.exists(filename_coal)) {
  coal <- rbind(coal, cbind(read.coal(filename_coal)))
} else {
  filename_coal <- paste0(filename, '_nohmm_', sample_id_label, '.coal')
  coal <- rbind(coal, cbind(read.coal(filename_coal)))
}
# coal <- rbind(coal, cbind(read.coal(paste0(filename, ".coal"))))
coal$group1 <- as.numeric(as.factor(coal$group1))
coal$haploid.coalescence.rate[which(coal$haploid.coalescence.rate == 0 & coal$epoch.start > 2e7/28)] <- coal$haploid.coalescence.rate[which(coal$haploid.coalescence.rate == 0 & coal$epoch.start > 2e7/28)-1]
coal$epoch.start[is.infinite(coal$epoch.start)] <- 1e8
coal %>% filter(!is.na(haploid.coalescence.rate)) %>% group_by(epoch.start, group1, Reference) %>% 
  summarize(mean = mean(haploid.coalescence.rate), lower = quantile(haploid.coalescence.rate, p = 0.025), upper = quantile(haploid.coalescence.rate, p = 0.975)) %>% 
  filter(mean > 0, !is.na(mean)) %>% droplevels() -> coal
coal$group1 <- paste0("comp", coal$group1)
print(coal)

coal <- coal %>% filter(Reference != "focal")
coal <- coal %>% filter(Reference != "gbr")
coal <- coal %>% filter(Reference != "WolfNorth_America")
# coal <- coal %>% filter(Reference != "yoruba")
# coal <- coal %>% filter(Reference == "eurasian" | Reference == 'esn' )


## Generate global xlim and ylim based on all components
global_xlim <- range(28 * coal$epoch.start, na.rm = TRUE)
global_ylim <- range(0.5 / coal$mean, na.rm = TRUE)

# Adjust the limits with a small padding for better visualization (optional)
global_xlim <- c(max(5e1, global_xlim[1] * 0.5), min(2e6, global_xlim[2] * 2))
global_ylim <- c(max(5e2, global_ylim[1] * 0.5), min(2e7, global_ylim[2] * 2))

## Generate plots based on the number of components
for (i in 1:num_components){
  coal_subset <- subset(coal, group1 == paste0('comp', i))
  
  ## Get the corresponding proportion for this component
  proportion_value <- prop[i]

  coal_subset$mean_inv <- pmin(pmax(0.5 / coal_subset$mean, global_ylim[1]), global_ylim[2])

  p <- ggplot(coal_subset) +
    geom_step(aes(x = 28*epoch.start, y = mean_inv, color = Reference), lwd = 1.2) +
    scale_x_continuous(trans = "log10", limits = global_xlim) +
    #                   breaks = c(5e4, 3e5, 2e6)) +  # Set custom x-ticks    
    scale_y_continuous(trans = "log10", limits = global_ylim) +
    xlab("years ago") + 
    ylab(NULL) +   # Set 'Inverse Coalescence Rates' as the y-axis label
    ggtitle(paste0("Component ", i, " (", round(proportion_value, 2), "%)")) +  # Set title
    theme_classic(base_size = 18, base_family = "Helvetica") +  # Explicitly setting the font
    theme(plot.title = element_text(hjust = 0.5))  + # Center the title
    theme(plot.title = element_text(hjust = 0.5, size = 16))

  ## Save each plot based on the filename path
  ggsave(p, file = paste0(filename, "_visual", i, ".svg"), width = 6, height = 4, device = "svg", dpi=400)
}

weights <- prop/100
names(weights) <- paste0("comp", seq_along(weights))
coal$weight <- weights[coal$group1]
weighted <- coal %>% 
  group_by(epoch.start, Reference) %>% 
  summarize(weighted_rate = sum(mean * weight), .groups = 'drop') %>%
  mutate(inverse = pmin(pmax(0.5/weighted_rate, global_ylim[1]), global_ylim[2]))
p_gw <- ggplot(weighted) +
  geom_step(aes(x = 28 * epoch.start, y = inverse, color = Reference), lwd = 1.2) +
  scale_x_continuous(trans = "log10", limits = global_xlim) +
  scale_y_continuous(trans = "log10", limits = global_ylim) +
  xlab("years ago") +
  ylab(NULL) +
  ggtitle("Genome-wide ICRs") +
  theme_classic(base_size = 18, base_family = "Helvetica") +
  theme(plot.title = element_text(hjust = 0.5, size = 16))
ggsave(p_gw, file = paste0(filename, "_gw.svg"), width = 6, height = 4, device = "svg", dpi = 400)

for (i in 1:num_components){
  coal_subset <- subset(coal, group1 == paste0('comp', i))
  proportion_value <- prop[i]
  coal_subset$mean_inv <- pmin(pmax(0.5 / coal_subset$mean, global_ylim[1]), global_ylim[2])

  p <- ggplot(coal_subset) +
    # component-specific curve
    geom_step(aes(x = 28 * epoch.start, y = mean_inv, color = Reference),
              lwd = 1.2) +
    # overlay the GW curve in grey
    geom_step(
      data      = weighted,
      aes(x = 28 * epoch.start, y = inverse),
      color     = "grey",
      alpha     = 0.6,
      lwd       = 1,
      inherit.aes = FALSE
    ) +
    scale_x_continuous(trans = "log10", limits = global_xlim) +
    scale_y_continuous(trans = "log10", limits = global_ylim) +
    xlab("years ago") +
    ylab(NULL) +
    ggtitle(paste0("Component ", i, " (", round(proportion_value, 2), "%)")) +
    theme_classic(base_size = 18, base_family = "Helvetica") +
    theme(plot.title = element_text(hjust = 0.5, size = 16))

  ggsave(p, file = paste0(filename, "_gw_visual", i, ".svg"),
         width = 6, height = 4, device = "svg", dpi = 400)
}

## Now adjust the standalone GW plot:
p_gw <- ggplot(weighted) +
  geom_step(
    aes(x = 28 * epoch.start, y = inverse),
    color = "grey",
    alpha = 1.0,
    lwd   = 1.2
  ) +
  scale_x_continuous(trans = "log10", limits = global_xlim) +
  scale_y_continuous(trans = "log10", limits = global_ylim) +
  xlab("years ago") +
  ylab(NULL) +
  ggtitle("Genome-wide ICRs") +
  theme_classic(base_size = 18, base_family = "Helvetica") +
  theme(plot.title = element_text(hjust = 0.5, size = 16))

ggsave(p_gw, file = paste0(filename, "_gw.svg"),
       width = 6, height = 4, device = "svg", dpi = 400)


## module load R/4.1.2-foss-2021b
## Usage: Rscript visualize_gamma.R ../../../denisovan_sim_2024_08/output_nonghost/relate_50_51_52_53_54_55_56_57_58_59
