#!/usr/bin/Rscript

library(knitr)

# opts_chunk$set(
#   dev="pdf",
#   fig.path="figures/",
# 	fig.height=3,
# 	fig.width=4,
# 	out.width=".47\\textwidth",
# 	fig.keep="high",
# 	fig.align="center",
# 	prompt=TRUE,  # show the prompts; but perhaps we should not do this 
# 	comment=NA    # turn off commenting of ouput (but perhaps we should not do this either
#   )
knit('mackenzie_mattering_march2015.rmd')
