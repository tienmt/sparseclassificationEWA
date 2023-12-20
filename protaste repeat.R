

save.image(file='~/Library/CloudStorage/Dropbox/ongoing_works/LOW RANK MODELS/high dim classification hinge loss/Rcodes/real data/out_prostate2.rda')

############################
###########################
tau = .51  # in the prior
Iters = 30000
burnin = 5000

library(spls);library(glmnet)
data("prostate"); n = length(prostate$y) ; p = ncol(prostate$x)

####
lasso = mala.hinge = lmc.hinge = lmc.logis = mala.logis = acceptrate = c()
for (ss in 1:100) {
  test = sample(n, size = 31, replace = F)
  X = prostate$x[-test,]; tX = t(X)
  Y = prostate$y[-test] ; Y[Y==0] <- -1 
  Ytest = prostate$y[test] ;  Ytest[Ytest==0] <- -1 
  Xtest = prostate$x[test,]  
  
  ### glmnet
  cvfit.glmet <- cv.glmnet(X,Y, family = "binomial", intercept=FALSE)
  beta_glmnet <- as.vector(coef(cvfit.glmet, s = "lambda.min"))[-1]
  lasso[ss] = mean(sign(Xtest%*%beta_glmnet) != Ytest) 
  
  ### MALA
  Bm_hinge = matrix( 0 ,nrow = p)
  h = 1/(p)^2.7
  a = 0  
  M = beta_glmnet
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
  mala.hinge[ss] = mean(sign(Xtest%*%Bm_hinge) != Ytest)
  
  ### LMC
  B_lmc = matrix( 0 ,nrow = p)
  h_lmc = h/2
  M = beta_glmnet
  for(s in 1:Iters){
    YXm = 1-Y*(X%*%M)
    M = M + h_lmc*tX%*%(Y*( YXm > 0 )) - h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
    if (s>burnin)B_lmc = B_lmc + M/(Iters-burnin)
  }
  lmc.hinge[ss] = mean(sign(Xtest%*%B_lmc) != Ytest)
  
  ### LMC with logistic loss
  B_lmc_logit = matrix( 0 ,nrow = p)
  h_lmc = h/2
  M = beta_glmnet
  for(s in 1:Iters){
    exp_YXm = exp(-Y*(X%*%M) )
    M = M + h_lmc*tX%*%(Y*exp_YXm/( 1 + exp_YXm) ) -h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
    if (s>burnin)B_lmc_logit = B_lmc_logit + M/(Iters-burnin)
  }
  lmc.logis[ss] = mean(sign(Xtest%*%B_lmc_logit) != Ytest) 
  
  ### MALA logistic
  B_mala_logit = matrix( 0 ,nrow = p)
  h = 1/(p)^2.7
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
  print(a/Iters); print(ss)
  mala.logis[ss] = mean(sign(Xtest%*%B_mala_logit) != Ytest)
}

round(c(mean(lmc.logis) , sd(lmc.logis) ),4)
round(c(mean(lmc.hinge) , sd(lmc.hinge) ),4)
round(c(mean(mala.logis) , sd(mala.logis) ),4)
round(c(mean(mala.hinge) , sd(mala.hinge) ),4)
round(c(mean(lasso) , sd(lasso) ),4)

