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

filename <- commandArgs(trailingOnly = TRUE)[1]

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
foo <- read.table(paste0(filename, ".tau"))
colnames(foo) <- paste0("comp", 1:ncol(foo))
prop <- data.frame(iters = 1:nrow(foo), foo)
prop_long <- prop %>% 
  pivot_longer(cols = !iters, names_to = "Components", values_to = "Proportion")

## Filter for the last iteration's proportions
last_iter_prop <- prop_long %>% filter(iters == max(iters))
last_iter_prop$Proportion <- last_iter_prop$Proportion * 100  # Convert to percentage

## Infer number of components from .tau file
num_components <- ncol(foo)
print(paste("Number of components:", num_components))

## Prepare the coal data
coal <- data.frame()
coal <- rbind(coal, cbind(read.coal(paste0(filename, ".coal"))))
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
  proportion_value <- last_iter_prop %>% filter(Components == paste0("comp", i)) %>% pull(Proportion)

  coal_subset$mean <- pmin(pmax(0.5 / coal_subset$mean, global_ylim[1]), global_ylim[2])

  p <- ggplot(coal_subset) +
    geom_step(aes(x = 28*epoch.start, y = mean, color = Reference), lwd = 1.2) +
    scale_x_continuous(trans = "log10", limits = global_xlim) +
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


## module load R/4.1.2-foss-2021b
## Usage: Rscript visualize_gamma.R ../../../denisovan_sim_2024_08/output_nonghost/relate_50_51_52_53_54_55_56_57_58_59
