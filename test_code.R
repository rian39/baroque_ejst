
# joint annuity calculation -- the type of thing Huyghens might have liked?

joint_annuity  <- function() {  
    death_m <- 75
    death_f <- 80

    # annuity starts at what age? 
    start_paying <- 65

    # population size, and mortalities
    n = 100000
    deaths_mens <- rnorm(n, death_m, 5)
    deaths_fems <- rnorm(n, death_f, 5)

    # couples are the same age
    age_diff <- rnorm(n, 3, 3)


    par(mfrow=c(2,3))
    hist(deaths_mens)
    hist(deaths_fems)
    hist(age_diff)
    years_to_pay <- pmax(deaths_mens, deaths_fems) - start_paying + age_diff
    hist(years_to_pay)

    # what can be done with this simulation?
    prob_10 <- sum(years_to_pay<=10)/n * 100
    cat ('Expectation of 10 years or less:', prob_10, '%\n')

    prob_15 <- sum(years_to_pay >=15)/n * 100
    cat ('Expectation of 15 years or  more:', prob_15, '%\n')

    prob_0 <- sum(years_to_pay <=0)/n*100
    cat ('Expectation of no payment:', prob_0, '%\n')
    plot(x=age_diff, y=years_to_pay)

    age_purchase = 60
    years_to_live <- abs(deaths_fems-deaths_mens) 

}

woman_bayes_simple <- function() {
    p_w = 0.5 # 50% of people are women
    p_m = 1- p_w
    p_l_w = .75  # 75% of women have long hair 
    p_l_m = 0.3 # 30% of men have long hair

    # probability that a given person with long hair is a women
    p_w_l <- p_l_w * p_w / (p_l_w * p_w + p_l_m * p_m)

    p_m_l <- p_l_m* p_m /(p_l_w * p_w + p_l_m * p_m)

    # simulate the chances of encountering a women
    n = 10000
    women_h <- rnorm(n, 30, 5)
    men_h <- rnorm(n, 10, 5)
    
    hist(women_h)
    hist(men_h)

    #meet someone with 10cm long hair -- what is the chance they are a women?

    p_w <- sum(women_h <10)/n*100

}




# generate distributions

generate_distributions <- function(x) {
    library(ggplot2)
    library(reshape2)
    	nval=10^4
      normal <- rnorm(nval,0.5,0.5)
      pois <- rpois(nval, 0.1)
      beta <- rbeta(nval, 6,4)
      dist <- data.frame(normal=normal, poisson = pois, beta=beta)
      dist_m <- melt(dist)
      names(dist_m) <- c('distribution', 'value')
      ggplot(dist_m, aes(x=value, fill=distribution)) + 
      geom_density(alpha=0.6) +
      ggtitle('Probability density functions')


    norm_2 <- 0.5*rnorm(1000,10, 0.5) * 2*rnorm(1000, 4, 0.1)
    qplot(norm_2, geom='density')

}

generate_plots <- function(x) {
        require(ggplot2)
        require(reshape2)
        require(gridExtra)

        x = y = scat <- sort(rnorm(1000) + rchisq(1000, df=4))
        fun <- function(x,y) { r <- sqrt(x^2 + y^2); 10*sin(r)/r}
        z <- outer(x,y,fun)
        scat.df <- data.frame(x= x, y= x, z=z)

        g = ggplot(scat.df, aes(x=x, y=y)) + geom_density2d(position='jitter')

        p = persp(x,y,z,  theta = 30, phi = 30, expand = 0.5,
                shade = 0.75, ticktype = "detailed",
               xlab = "X", ylab = "Y")

        # sidebysideplot <- grid.arrange(g, p, ncol=2)

}

bivar_norm <- function(x) {
    # Édouard Tallent @ TaGoMa.Tech
    # September 2012
    # This code plots simulated bivariate normal distributions

    # Some variable definitions
    mu1 <- 0  # expected value of x
    mu2 <- 0.5  # expected value of y
    sig1 <- 0.5 # variance of x
    sig2 <- 2 # variance of y
    rho <- 0.5  # corr(x, y)

    # Some additional variables for x-axis and y-axis 
    xm <- -3
    xp <- 3
    ym <- -3
    yp <- 3

    x <- seq(xm, xp, length= as.integer((xp + abs(xm)) * 10))  # vector series x
    y <- seq(ym, yp, length= as.integer((yp + abs(ym)) * 10))  # vector series y

    # Core function
    bivariate <- function(x,y){
      term1 <- 1 / (2 * pi * sig1 * sig2 * sqrt(1 - rho^2))
      term2 <- (x - mu1)^2 / sig1^2
      term3 <- -(2 * rho * (x - mu1)*(y - mu2))/(sig1 * sig2)
      term4 <- (y - mu2)^2 / sig2^2
      z <- term2 + term3 + term4
      term5 <- exp((-z / (2*(1-rho^2))))
      return (term1 * exp(- z * (2 * (1 - rho^2))))
    }

    # Computes the density values
    z <- outer(x,y,bivariate)

    # Plot
    persp(x, y, z, main = "Bivariate Normal Distribution",
        sub = bquote(bold(mu[1])==.(mu1)~", "~sigma[1]==.(sig1)~", "~mu[2]==.(mu2)~
        ", "~sigma[2]==.(sig2)~", "~rho==.(rho)),
        col="orchid2", theta = 55, phi = 30, r = 40, d = 0.1, expand = 0.5,
        ltheta = 90, lphi = 180, shade = 0.4, ticktype = "detailed", nticks=5)
}