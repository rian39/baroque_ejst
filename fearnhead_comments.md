

I have had a quick read through the chapter. There are some stats/maths-related errors I spotted, and  a couple of places where the writing is (to a mathematician) imprecise. 

* p.7  "The diagram shows the contours of two normally-distributed sets of numbers as they
vary in relation to each other."
My interpretation of this sentence would be that the contours should be that of a bivariate normal distribution, where as (I guess) they are contours of the posterior distribution for fitting a mixture of two normal distributions with unknown mean to some data.

* Figure 2: strictly speaking the Poisson distribution does not have a density -- as it can take only discrete values (0,1,2 etc,) rather than a continuous range. So you cannot plot its probability density. You could plot its probability mass function -- but (strictly speaking) not on the same plot as the probability density of continuous random variables.

* p.15 "differentiation (finding the distribution function for one variable)" Differentiating a probability density function does not give you the marginal distribution for one variable -- you need to integrate to go from a joint density function to a marginal one.

p.18 " that multiplying a uniform distribution by itself" you are not multiplying the uniform distribution by itself, but the uniform random variable by itself (or defining a new random variable that is equal to the square of a uniform random variable).

Equation (2) the "-1" should be in the exponent (which should be "beta-1")

p.21 I looks to me like the quote from Robert&Casella 2010:169 is missing some mathematical symbol between the epsilon(t) and U(0,1)

p.22  "If that state is more likely, the move is allowed; otherwise
the particle goes back to where it was."

actually if the state is less likely there is some probability of it being allowed (it does not automatically go back to where it was).

Paul

PS I think the above are the comments you were interested in -- other than that the descriptions of MCMC (at the more hand-wavy level it is covered in the chapter) looks fine to me.

I should also mention that I, personally, do not see the basis for a number of the central comments in the chapter, nor how they relate to how statisticians would view MCMC; for example comments such as "individuals become like joint distributions" (i.e. I am not sure what is meant by this, and do not see how MCMC could link to this idea). Nor how "individuals appear as populations". 

Or " both interpretations miss the re-distribution of probability as randomly generated but topographically smooth surfaces whose
many dimensions support complicated conjunctions of events." I find hard to fully understand, or see how it could be justified mathematically.

But I guess that is a separate issue.


