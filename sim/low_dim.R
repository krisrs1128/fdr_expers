#' Simulate Nonlinear Data
library("tidyverse")
library("reshape2")
losses <- read_csv("losses.csv") %>%
    mutate(iter = row_number())

mlosses <- losses %>%
    melt(id.vars = c("iter", "epoch"), variable.name = "net", value.name = "loss")

ggplot(mlosses) +
    geom_point(
        aes(x = iter, y = loss, col = net)
    ) +
    scale_color_brewer(palette = "Set2")
