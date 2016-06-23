
library(latex2exp)
svg('formula_labelled_revised.svg')
form = '$f(x;\\mu, \\sigma^2) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}$'
plot(TeX(form))
dev.off()
