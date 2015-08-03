#!/usr/bin/Rscript

library(knitr)

 opts_chunk$set(
   dev="tiff",
   dpi=400,
   fig.path="figure/",
     fig.height=6,
     fig.width=8,
     out.width=".87\\textwidth",
     fig.keep="high",
     fig.align="center",
     prompt=TRUE,  # show the prompts; but perhaps we should not do this 
     comment=NA    # turn off commenting of ouput (but perhaps we should not do this either
   )
knit('mackenzie_mattering.rmd')
