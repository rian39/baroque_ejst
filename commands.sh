pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex ejors.rmd -o ejors.pdf
pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex mcmc.md -o mcmc.pdf

##
pandoc --smart --normalize --bibliography=references/refs.bib --csl=references/sage-harvard.csl --latex-engine=xelatex ejors_nov2012.rmd -o ejors_nov2012.pdf

rsync -r -t -v --progress -u /home/mackenza/Documents/R-project/baroque_ejst/ /home/mackenza/Dropbox/baroque_ejst/