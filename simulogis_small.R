library(glmnet)
Iters = 30000
burnin = 5000
# random data generation
n = 50  # samples
p = 100   # predictors
s0 = 10    # sparsity
tau = 1  # in the prior


lasso = mala.hinge = lmc.hinge = lmc.logis = mala.logis = acceptrate = c()

for (ss in 1:100) {
  X = matrix( rnorm(p*n), ncol = p); tX = t(X)
  btrue = matrix( 0, nrow = p)
  btrue[1:s0] = rnorm(s0,sd = 10)
  
  # binomial model
  mu = X%*%btrue  + rnorm(n)
  prob = 1/(1+ exp(-mu) )
  Ytrue = apply(prob, 2, function(a) rbinom(n = n, size = 1, a))
  Ytrue[Ytrue==0] <- -1
  Y = Ytrue
  #Z = sample(c(-1,1),size = n,replace = T,prob = c(0.1 , 0.9))
  #Y = Ytrue*Z
  
  ### glmnet
  cvfit.glmet <- cv.glmnet(X,Y, family = "binomial", type.measure = "class",intercept=FALSE)
  beta_glmnet <- as.vector(coef(cvfit.glmet, s = "lambda.min"))[-1]
  lasso[ss] = mean(sign(X%*%beta_glmnet) != Ytrue) 
  
  ### MALA
  Bm_hinge = matrix( 0 ,nrow = p)
  h = 1/(p)^2.3 # 2.4
  a = 0  
  M = matrix( 0 ,nrow = p)
  for(s in 1:Iters){
    YXm = 1-Y*(X%*%M)
    tam = M + h*tX%*%(Y*( YXm > 0 )) - h*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h)*rnorm(p)
    YXtam = 1-Y*(X%*%tam)
    pro.tam = - sum(YXtam*( YXtam > 0) ) -sum(2*log(tau^2 + tam^2))
    pro.M = - sum( YXm*( YXm > 0 )) -sum(2*log(tau^2 + M^2))
    
    tran.m = -sum((M-tam -h*tX%*%(Y*( YXtam > 0 )) -h*sum(2*log(tau^2 + tam^2)) )^2)/(4*h)
    tran.tam = -sum((tam-M - h*tX%*%(Y*( YXm > 0 )) -h*sum(2*log(tau^2 + M^2)) )^2)/(4*h)
    pro.trans = pro.tam+tran.m-pro.M-tran.tam
    if(log(runif(1)) <= pro.trans){
      M = tam;  a = a+1
    } 
    if (s>burnin)Bm_hinge = Bm_hinge + M/(Iters-burnin)
  }
  acceptrate[ss] = a/Iters
  mala.hinge[ss] = mean(sign(X%*%Bm_hinge) != Ytrue)
  
  ### LMC
  B_lmc = matrix( 0 ,nrow = p)
  h_lmc = h/2
  M = beta_glmnet
  for(s in 1:Iters){
    YXm = 1-Y*(X%*%M)
    M = M + h_lmc*tX%*%(Y*( YXm > 0 )) - h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
    if (s>burnin)B_lmc = B_lmc + M/(Iters-burnin)
  }
  lmc.hinge[ss] = mean(sign(X%*%B_lmc) != Ytrue)
  
  ### LMC with logistic loss
  B_lmc_logit = matrix( 0 ,nrow = p)
  h_lmc = h/2
  M = beta_glmnet
  for(s in 1:Iters){
    exp_YXm = exp(-Y*(X%*%M) )
    M = M + h_lmc*tX%*%(Y*exp_YXm/( 1 + exp_YXm) ) -h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
    if (s>burnin)B_lmc_logit = B_lmc_logit + M/(Iters-burnin)
  }
  lmc.logis[ss] = mean(sign(X%*%B_lmc_logit) != Ytrue) 
  
  ### MALA logistic
  B_mala_logit = matrix( 0 ,nrow = p)
  h = 1/(p)^2.3
  a = 0  
  M = beta_glmnet
  for(s in 1:Iters){
    exp_YXm = exp(-Y*(X%*%M) )
    tam = M + h*tX%*%(Y*exp_YXm/( 1 + exp_YXm) ) - h*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h)*rnorm(p)
    
    exp_YXtam = exp(-Y*(X%*%tam) )
    pro.tam = - sum(log(1+exp_YXtam )) -sum(2*log(tau^2 + tam^2))
    pro.M = - sum(log(1+exp_YXm) ) -sum(2*log(tau^2 + M^2))
    
    tran.m = -sum((M-tam -h*tX%*%(Y*exp_YXtam/( 1 + exp_YXtam)) -h*sum(2*log(tau^2 + tam^2)) )^2)/(4*h)
    tran.tam = -sum((tam-M - h*tX%*%(Y*exp_YXm/( 1 + exp_YXm) ) -h*sum(2*log(tau^2 + M^2)) )^2)/(4*h)
    pro.trans = pro.tam+tran.m-pro.M-tran.tam
    if(log(runif(1)) <= pro.trans){
      M = tam;  a = a+1
    } 
    if (s>burnin)B_mala_logit = B_mala_logit + M/(Iters-burnin)
  }
  a/Iters; print(ss)
  mala.logis[ss] = mean(sign(X%*%B_mala_logit) != Ytrue)
}
setwd("~/Library/CloudStorage/Dropbox/ongoing_works/LOW RANK MODELS/sparse logistic classification/Rcodes/outsimu")
save.image(file = 'simulogit_small_withnoise_noswch.rda')


