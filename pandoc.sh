#!/bin/sh

#pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex ejors.rmd -o ejors.pdf
#pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex mcmc.md -o mcmc.pdf


##format bibliography and  display 
pandoc --smart --normalize --latex-engine=xelatex  --template=/home/mackenza/template.latex --bibliography=/home/mackenza/ref_bibs/data_forms_thought.bib  --bibliography=/home/mackenza/ref_bibs/machine_learning.bib --bibliography=/home/mackenza/ref_bibs/R.bib --csl=references/sage-harvard.csl mackenzie_mattering_march2015.md  -o mackenzie_mattering_march2015.pdf
evince mackenzie_mattering_march2015.pdf

