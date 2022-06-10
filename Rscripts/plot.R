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
virtualenv_create(envname = "python_environment",python= "/apps/eb/2020b/skylake/software/Python/3.9.6-GCCcore-11.2.0/bin/python")
reticulate::use_virtualenv("python_environment", required = TRUE)
np <- import("numpy")

palette <- c("#E69F00", "#000000", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
colours <- c("#ff99c8", "#fcf6bd", "#d0f4de", "#a9def9")

colours <- c("#FF5CA8", "#FAF19E", "#BDEFD0", "#8DD4F7", "#CF90F4", "#d0f4de")

palette  <- c("#f2cc8f", "#e07a5f", "#3d405b", "#81b29a")


poplabs <- sort(unique(read.table("../../sims/stdpopsim_ancient_small/devel_relate_trees_force_50/poplabels.txt", header = T)[,2]))

#poplabels <- read.table("../result/cond_coal_rates_new_group/SGDP_group.poplabels", header = T)[,2]
#assignments <- unique(poplabels)

read.coal <- function(filename){

	groups <- as.matrix(utils::read.table(filename, nrow = 1))
	epochs <- as.matrix(utils::read.table(filename, nrow = 1, skip = 1))
	coal   <- utils::read.table(filename, skip = 2)

	coal[,1] <- as.factor(coal[,1])
	coal[,2] <- as.factor(coal[,2])

	colnames(coal)[-c(1:2)] <- epochs
	colnames(coal)[1:2]     <- c("group1", "group2")
	coal                    <- reshape2::melt(coal, id.vars = c("group1", "group2"))
	colnames(coal)[3:4]     <- c("epoch.start", "haploid.coalescence.rate")
	coal                    <- coal[order(paste(coal[,1], coal[,2], sep = "")),]
	coal$epoch.start        <- as.numeric(as.matrix(coal$epoch.start))

	coal$group2 <- poplabs[as.numeric(as.matrix(coal$group2))+1]

	#coal <- subset(coal, group2 %in% assignment[grep(as.matrix(assignment), pattern = "v1")] )
	#coal <- subset(coal, group2 %in% c("Sardinian", "Khomani-San", "Mbuti", "Mende"))

	return(coal)

}

filename <- commandArgs(trailingOnly = T)[1]
sam      <- commandArgs(trailingOnly = T)[2]
#filename_log <- commandArgs(trailingOnly = T)[2]

coal <- data.frame()
coal <- rbind(coal, cbind(read.coal(paste0(filename,"_",sam,"_iter0.coal")), iter = paste0("Iteration 1")))
coal <- rbind(coal, cbind(read.coal(paste0(filename,"_",sam,".coal")), iter = paste0("Iteration 1000")))

coal$group1 <- as.numeric(as.factor(coal$group1))

coal$haploid.coalescence.rate[which(coal$haploid.coalescence.rate == 0 & coal$epoch.start > 1e7/28)] <- coal$haploid.coalescence.rate[which(coal$haploid.coalescence.rate == 0 & coal$epoch.start > 1e7/28)-1]
coal$epoch.start[is.infinite(coal$epoch.start)] <- 1e8
coal %>% filter(!is.na(haploid.coalescence.rate)) %>% group_by(epoch.start, group1, group2, iter) %>% summarize(mean = mean(haploid.coalescence.rate), lower = quantile(haploid.coalescence.rate, p = 0.025), upper = quantile(haploid.coalescence.rate, p = 0.975)) %>% filter(mean > 0, !is.na(mean)) %>% droplevels() -> coal

coal$group1 <- paste0("comp", coal$group1)

p1 <- ggplot(coal) + geom_step(aes(x = 28*epoch.start, y = 0.5/mean, colour = group1), lwd = 1.1) +
				#geom_stepribbon(aes(x = 28*epoch.start, ymin = 0.5/upper, ymax = 0.5/lower, fill = group2), alpha = 0.5) +
				ggthemes::theme_few() +
				scale_x_continuous(limit = c(5e3,1e7), trans = "log10") +
				scale_y_continuous(trans = "log10") +
				coord_cartesian(ylim = c(1e3,1e7)) +
				# scale_colour_manual(values = palette) +
				# scale_fill_manual(values = palette) + 
				annotation_logticks(sides = "bl") +
				facet_grid(iter~group2) +
				theme(legend.position = "bottom", legend.title = element_blank()) +
				xlab("years ago") +
				ylab("Inverse coalescence rate") + guides(colour = guide_legend(override.aes = list(size = 3)))

g <- ggplot_gtable(ggplot_build(p1))

strips <- which(grepl('strip-t', g$layout$name))

for (i in seq_along(strips)) {
	  k <- which(grepl('rect', g$grobs[[strips[i]]]$grobs[[1]]$childrenOrder))
    l <- which(grepl('titleGrob', g$grobs[[strips[i]]]$grobs[[1]]$childrenOrder))
	  g$grobs[[strips[i]]]$grobs[[1]]$children[[k]]$gp$fill <- colours[i]
	  #g$grobs[[strips[i]]]$grobs[[1]]$children[[l]]$children[[1]]$gp$col <- pal[i + 1]
}

strips <- which(grepl('strip-r', g$layout$name))

for (i in seq_along(strips)) {
	k <- which(grepl('rect', g$grobs[[strips[i]]]$grobs[[1]]$childrenOrder))
	l <- which(grepl('titleGrob', g$grobs[[strips[i]]]$grobs[[1]]$childrenOrder))
	g$grobs[[strips[i]]]$grobs[[1]]$children[[k]]$gp$fill <- "lightgrey"
	#g$grobs[[strips[i]]]$grobs[[1]]$children[[l]]$children[[1]]$gp$col <- pal[i + 1]
}

p1 <- as.ggplot(g)


########### Likeilhood

foo <- as.matrix(read.table(paste0(filename, "_51.logl")))
logl <- data.frame(iters = 1:length(foo), logl = as.numeric(as.matrix(foo)))

print(head(logl))
p2 <- ggplot(logl) + geom_point(aes(x = iters, y = logl)) + geom_line(aes(x = iters, y = logl)) + ylab("log-likelihood") + xlab("iterations") + coord_cartesian(expand = F, xlim = c(0, max(logl$iters+1)), ylim = range(logl$logl) * c(1.00001,0.99999)) + ggthemes::theme_few() 

########### Proportions


prop <- data.frame()

foo <- read.table(paste0(filename,"_51.tau"))
colnames(foo) <- paste0("comp", 1:ncol(foo))
prop <- rbind(prop, cbind(iters = 1:nrow(foo), foo))
print(head(prop))
prop %>% pivot_longer(cols = !iters, names_to = "Components", values_to = "Proportion") -> prop

p3 <- ggplot(prop) + geom_point(aes(x = iters, y = Proportion, colour = Components)) + geom_line(aes(x = iters, y = Proportion, colour = Components)) + xlab("iterations") + coord_cartesian(expand = F, xlim = c(0, max(prop$iters+1)), ylim = c(0,1.0)) + ggthemes::theme_few() 


########## Histogram of posterior at start and end
member3 <- np$load(paste0(filename,"_overall_membership_iter0_",sam,".npy"))
member <- np$load(paste0(filename,"_overall_membership_",sam,".npy"))

member3 <- member3[-1,] ## need to change this.. every time
member <- member[-1,] ## need to change this.. every time

member3  <- as.data.frame(t(member3))
member  <- as.data.frame(t(member))

colnames(member3) <- 1:ncol(member3)
colnames(member) <- 1:ncol(member)

member3$ID <- 1:nrow(member3)
member$ID <- 1:nrow(member)

member3 %>% pivot_longer(col = !ID, names_to = "comp", values_to = "posterior") -> member3
member %>% pivot_longer(col = !ID, names_to = "comp", values_to = "posterior") -> member

member3$iter <- "iter 0"
member$iter <- "iter 499"

df1 <- rbind(member, member3)

p5 <- ggplot(df1, aes(x=posterior))+
  geom_histogram(color="black", fill="black", bins = 100)+
  facet_grid(iter ~ .)

########### Calibration curves

member <- np$load(paste0(filename,"_overall_membership_",sam,".npy"))
member2 <- np$load(paste0(filename,"_ground_truth_membership_",sam,".npy"))
#member2 <- member2[-1,]
#member <- member[-1,] ## need to change this.. every time
member  <- as.data.frame(t(member))
member2 <- as.data.frame(t(member2))

colnames(member) <- 1:ncol(member)
colnames(member2) <- 1:ncol(member2)
member$ID <- 1:nrow(member)
member2$ID <- 1:nrow(member2)
member %>% pivot_longer(col = !ID, names_to = "comp", values_to = "posterior") -> member
member2 %>% pivot_longer(col = !ID, names_to = "comp", values_to = "truth") -> member2

match <- numeric(0)
for(i in unique(member2$comp)){
 k <- numeric(0)
 c <- -1
 for(j in unique(member$comp)){
   if( cor( subset(member, comp == j)$posterior, subset(member2, comp == i)$truth ) > c ){
     c <- cor( subset(member, comp == j)$posterior, subset(member2, comp == i)$truth )
     k <- j
   }
 }
 match <- append(match, k)
}
member2$comp <- as.numeric(as.matrix(member2$comp))
member2$comp <- match[member2$comp]
df <- merge(member, member2, by = c("comp","ID"))
df$comp <- as.matrix(df$comp)
df$comp <- paste0("comp",df$comp)

df$post_bin <- as.numeric(as.matrix(cut(df$posterior, breaks = seq(0,1,0.05), labels = seq(0,0.95,0.05)+0.025)))
df %>% group_by(post_bin, comp) %>% summarize(truth = mean(truth)) -> df_calib 
p4 <- ggplot(df_calib) + geom_abline(slope = 1) + geom_point(aes(x = post_bin, y = truth, colour = comp), size = 2)  + geom_line(aes(x = post_bin, y = truth, colour = comp)) + ggthemes::theme_few() + coord_cartesian(xlim = c(0,1), ylim = c(0,1), expand = F) + xlab("Mean posterior probability") + ylab("Proportion in component") + scale_colour_manual(values = colours, name = "")

df_calib <- read.table(paste0(filename,"_calibration.txt"))
colnames(df_calib) <- c("component", "match", "x", "y")
df_calib$component <- paste0("comp", df_calib$component+1)
df_calib$component <- as.factor(df_calib$component)
df_calib %>% group_by(component) %>% filter( match == match[which.max(y)] ) -> df_calib
df_calib %>% group_by(component) %>% filter( component == "comp1" ) -> df_calib
p4 <- ggplot(df_calib) + geom_abline(slope = 1) + geom_abline(slope = -1) + geom_point(aes(x = x, y = y, colour = component), size = 2) + geom_line(aes(x = x, y = y, colour = component)) + ggthemes::theme_few() + coord_cartesian(xlim = c(0,1), ylim = c(0,1), expand = F) + xlab("Mean posterior probability") + ylab("Proportion in component") + scale_colour_manual(values = colours, name = "")



p <- plot_grid(p2,p3,p4,p5,ncol = 1, align = "hv", axis = "lr", labels = c("b", "c", "d", "e"), label_size = 17)
p <- plot_grid(p1,p, ncol = 2, rel_widths = c(1.3,1), labels = c("a", ""), label_size = 17)

ggsave(p, file = paste0(filename, "_", sam,".pdf"), width = 11.5, height = 9.5)

