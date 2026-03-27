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
library(stringr)

# pop_colours <- c("Neanderthal" = "#FF5CA8", "Denisovan" = "#FAF19E")
pop_colours <- c(
  "Neanderthal" = "#ff7f0e",  # tab:orange
  "Denisovan"  = "#1f77b4"    # tab:blue
)

plot_coal_rates <- function(filepath, output_path) {
  coal <- read.coal(filepath)
  coal$epoch.start <- 28 * coal$epoch.start
  coal <- coal %>%
    filter(group1 == "Africa", group2 %in% c("Neanderthal", "Denisovan")) %>%
    group_by(group2, epoch.start) %>%
    summarise(
      mean_rate = mean(haploid.coalescence.rate),
      sd_rate   = sd(haploid.coalescence.rate),
      .groups   = "drop"
    ) %>%
    mutate(
      upper = mean_rate + 1.96 * sd_rate,
      lower = mean_rate - 1.96 * sd_rate
    )

  xlim <- c(8e4, 8e6)
  ylim <- c(8e3, 1e7)

  p <- ggplot(coal) +
    geom_stepribbon(aes(x = epoch.start, ymin = 0.5/upper, ymax = 0.5/lower, fill = group2), alpha = 0.5) +
    geom_step(aes(x = epoch.start, y = 0.5/mean_rate, color = group2), size = 1) +
    scale_x_continuous(trans = "log10", limits = xlim) +
    scale_y_continuous(trans = "log10", limits = ylim) +
    scale_color_manual(values = pop_colours) +
    scale_fill_manual(values = pop_colours) +
    labs(x = "years ago", y = NULL, color = "Population", fill = "Population") +
    theme_classic(base_size = 18, base_family = "Helvetica") +
    theme(plot.title = element_text(hjust = 0.5, size = 16))

  ggsave(output_path, p, width = 6, height = 4, device = "svg", dpi = 400)
}

filepath    <- "data/hgdp_1gp_ancients/Chagyrskaya_Denisova_san_mbuti_biaka.coal"
output_path <- "data/hgdp_1gp_ancients/Chagyrskaya_Denisova_coal_rates_afr.svg"

plot_coal_rates(filepath, output_path)
