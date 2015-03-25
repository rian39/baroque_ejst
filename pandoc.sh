#!/bin/sh

#pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex ejors.rmd -o ejors.pdf
#pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex mcmc.md -o mcmc.pdf


##format bibliography and  display 
pandoc --smart --normalize --template=template.latex --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex ejors_nov2012.md -o mackenzie_baroque_2013.pdf
evince mackenzie_baroque_2013.pdf
