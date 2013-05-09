# mere's question for pascal -- how many dice rolls needed to get even chances of a double six
doublesix <- function(x) {
    n = 1000
    rolls1 <- sample(1:6, n, replace=TRUE)
    rolls2 <- sample(1:6, n, replace=TRUE)
    doubles <- sum(rolls1==6 & rolls2==6)/n*100

    # simpler
    double_six <- 1/6 * 1/6
    rolls <- cumsum(rep(double_six, 100))
    head(rolls)
}

# toss a coin a hundred times [wasserman_all_2003, 9]

toss_coins <- function(heads =1, tosses = 10) {
    tosses <-rbinom(tosses, 2, 0.5)

    prob <- 1-0.5^tosses
    return(prob)
}


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

generate_markov <- function(x) {
    X <- vector(length=10^4)
    X[1] =runif(1)
    sigma=0.9
    for (t in 1:10^4){
    X[t+1] = sigma*X[t] + runif(1,min=0, max=1)
      }
    Y= rnorm(10^4, 0, 1/(1-sigma**2))
    par(mfrow=c(1,2))
    hist(X, breaks=200, freq=F, main='Markov chain generated normal')
    hist(Y, breaks=200, freq=F, main='Stationary distribution')
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

}

generate_folded_surface <- function(x) {
        require(ggplot2)
        require(reshape2)
        require(gridExtra)

      n =100
      x = y = scat <- sort(rnorm(n) + rchisq(n, df=4))
      fun <- function(x,y) { r <- sqrt(x^2 + y^2); 10*sin(r)/r}
      z <- outer(x,y,fun)
      scat.df <- data.frame(x= x, y= x, z=z)

      p = persp(x,y,z,  theta = 30, phi = 30, expand = 0.5,
              shade = 0.75, ticktype = "detailed",
             xlab = "X", ylab = "Y", main='Folded surface')

}

generate_bivar_norm <- function(x) {
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

generate_beta_distribution <- function(x) {
          set.seed(1234)
        m = 10000
        u = runif(m); 
        x = u^2
        xx = seq(0, 1, by=.001)
        cut.u = (0:100)/100; 
        cut.x = cut.u^2
        par(mfrow=c(1,2))
        hist(u, breaks=cut.u, prob=T, ylim=c(0,10))
        lines(xx, dunif(xx), col="blue")
        hist(x, breaks=cut.x, prob=T, ylim=c(0,10))
        lines(xx, .5*xx^-.5, col="blue")
        par(mfrow=c(1,1))
}

generate_beta_accept_reject <- function(x) {
    a = 2.7
    b = 6.3
    M = 2.67
    y = runif(Nsim)
    u = runif(Nsim, max = M)
    par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
    plot(y, u, col = "grey", pch = 19, cex = 0.4, ylab = expression(u.g(y)))
    points(y[u < dbeta(y, a, b)], u[u < dbeta(y, a, b)], pch = 19, 
        cex = 0.4)
    curve(dbeta(x, a, b), col = "sienna", lwd = 2, add = T)
    abline(h = M, col = "gold4", lwd = 2)
    M = 1.68
    y = rbeta(Nsim, 2, 6)
    u = runif(Nsim, max = M)
    labels = u < dbeta(y, a, b)/dbeta(y, 2, 6)
    plot(y, u * dbeta(y, 2, 6), col = "grey", pch = 19, cex = 0.4, 
        ylab = expression(u.g(y)))
    points(y[labels], u[labels] * dbeta(y[labels], 2, 6), pch = 19, 
        cex = 0.4)
    curve(dbeta(x, a, b), col = "sienna", lwd = 2, add = T)
    curve(M * dbeta(x, 2, 6), col = "gold4", lwd = 2, add = T)


}

#To simulate a beta distribution: 'we can just as well use a Metropolis-Hastings algorithm, where the target density f is the Be(2.7,6.3) density and the candidate q is uniform over [0,1]'  [@robert_introducing_2010,p.172]

generate_beta_mcmc <- function(x) {
    a=2.7; b=6.3; c=2.669
    Nsim=5000
    X=rep(runif(1),Nsim)
    accept <- vector(mode='logical', length=Nsim)
    for (i in 2:Nsim){
      Y=runif(1)
      rho=dbeta(Y,a,b)/dbeta(X[i-1],a,b)
      accept[i]=runif(1)<rho
      X[i]=X[i-1]+ (Y-X[i-1])*(accept[i])
      
    }

    Z= rbeta(5000,a,b)
    par(mfrow=c(1,2))
    hist(X, freq=F, breaks=200, main='Sample generated by Metropolis-Hastings')
    hist(Z, freq=F, breaks=200, main='Sample generated by exact iid' )
    print(ks.test(jitter(X), rbeta(5000,a,b)))
}

generate_normal_bivariate_metro_hastings <- function(x) {
    set.seed(1234); m = 40000; rho = .8; sgm = sqrt(1 - rho^2)
    xc = yc = numeric(m)

    # vectors of state components
    xc[1] = -3; yc[1] = 3

    # arbitrary starting values
    jl = 1; jr = 1

    # l and r limits of proposed jumps
    for (i in 2:m)
    {
        xc[i] = xc[i-1]; yc[i] = yc[i-1]
        # if jump rejected
        xp = runif(1, xc[i-1]-jl, xc[i-1]+jr) # proposed x coord
        yp = runif(1, yc[i-1]-jl, yc[i-1]+jr) # proposed y coord
        nmtr = dnorm(xp)*dnorm(yp, rho*xp, sgm)
        dntr = dnorm(xc[i-1])*dnorm(yc[i-1], rho*xc[i-1], sgm)
        r = nmtr/dntr
        # density ratio
        acc = (min(r, 1) > runif(1))
        # jump if acc == T
        if (acc) {xc[i] = xp; yc[i] = yp}
    }
    x = xc[(m/2+1):m]; y = yc[(m/2+1):m]

    # states after burn-in
    round(c(mean(x), mean(y), sd(x), sd(y), cor(x,y)), 4)
    mean(diff(x)==0)

    # proportion or proposals rejected
    mean(pmax(x,y) >= 1.25)

    # prop. of subj. getting certificates
    par(mfrow = c(1,2), pty="s")
    plot(xc[1:100], yc[1:100], xlim=c(-4,4), ylim=c(-4,4), type="l")
    plot(x, y, xlim=c(-4,4), ylim=c(-4,4), pch=".")

}


gibbs_normal_bivariate <- function(x) {
   Niter=10^4
    v=1
    da = sample(c(rnorm(10^2), 2.5 + rnorm(4 * 10^2)))
    
    like = function(mu) {
        sum(log((0.2 * dnorm(da - mu[1]) + 0.8 * dnorm(da - mu[2]))))
    }

    mu1 = mu2 = seq(-2, 5, le = 250)
    lli = matrix(0, ncol = 250, nrow = 250)
    for (i in 1:250) 
      for (j in 1:250) 
        lli[i, j] = like(c(mu1[i],   mu2[j]))
    
    x = prop = runif(2, -2, 5)
    the = matrix(x, ncol = 2)
    curlike = hval = like(x)
    for (i in 2:Niter) {
        pp = 1/(1 + ((0.8 * dnorm(da, mean = the[i - 1, 2]))/(0.2 * dnorm(da, mean = the[i - 1, 1]))))
        z = 2 - (runif(length(da)) < pp)
        prop[1] = (v * sum(da[z == 1]))/(sum(z == 1) * v + 1) + 
            rnorm(1) * sqrt(v/(1 + sum(z == 1) * v))
        prop[2] = (v * sum(da[z == 2]))/(sum(z == 2) * v + 1) + 
            rnorm(1) * sqrt(v/(1 + sum(z == 2) * v))
        curlike = like(prop)
        hval = c(hval, curlike)
        the = rbind(the, prop)
    }

    image(mu1, mu2, -lli, xlab = expression(mu[1]), ylab = expression(mu[2]))
    contour(mu1, mu2, -lli, nle = 100, add = T)
    points(the[, 1], the[, 2], cex = 0.6, pch = 19)
    lines(the[, 1], the[, 2], cex = 0.6, pch = 19)

}

#code from mcms package hastings() [@robert_introducing_2010]
hastings_1970 <- function (nsim = 10^3) 
{
    a = c(0.1, 1, 10)
    na = length(a)
    x = array(0, c(na, nsim))
    for (i in 1:na) {
        acc = 0
        for (j in 2:nsim) {
            y <- x[i, (j - 1)] + runif(1, min = -a[i], max = a[i])
            r = min(exp(-0.5 * ((y^2) - (x[i, (j - 1)]^2))), 
                1)
            u <- runif(1)
            acc = acc + (u < r)
            x[i, j] <- y * (u < r) + x[i, (j - 1)] * (u > r)
        }
    }
    par(mfrow = c(3, na), mar = c(4, 4, 2, 1))
    for (i in 1:na) plot((nsim - 500):nsim, x[i, (nsim - 500):nsim], 
        ty = "l", lwd = 2, xlab = "Iterations", ylab = "", main = paste("Rate", 
            (length(unique(x[i, ]))/nsim), sep = " "))
    for (i in 1:na) {
        hist(x[i, ], freq = F, xlim = c(-4, 4), ylim = c(0, 0.4), 
            col = "grey", ylab = "", xlab = "", breaks = 35, 
            main = "")
        curve(dnorm(x), lwd = 2, add = T)
    }
    for (i in 1:na) acf(x[i, ], main = "")
}
