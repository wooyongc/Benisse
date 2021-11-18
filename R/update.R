update_A=function(master_dist_e,phi,meta_dedup,SI,I,A,R,Q,LS,hyper_para){
  fn=function(par0,data){
    par=data[[10]]
    par[data[[9]]]=par0
    LA=as.vector(diag(rowSums(par))-par)
    0.5*data[[1]]*norm(data[[5]]-data[[8]]-4*data[[2]]*
                         (data[[6]]*data[[3]]+LA)-data[[4]]/data[[1]],type='F')^2+
      sum(par*as.vector(data[[7]]))
  }
  
  gr=function(par0,data){
    par=data[[10]]
    par[data[[9]]]=par0
    U=as.vector(diag(rowSums(par))-par)-
      (data[[5]]-data[[8]]-data[[4]]/data[[1]])/(4*data[[2]])+
      data[[6]]*data[[3]]
    gradients=2*data[[7]]+16*data[[1]]*data[[2]]^2*(-U-t(U)+2*diag(U))
    gradients[data[[9]]]
  }
  
  # bounds in the format of full matrices
  lower=matrix(0,ncol=nrow(meta_dedup),nrow=nrow(meta_dedup))
  upper=hyper_para$lambda1*(1-I)*SI
  
  # identify the true part of these matrices that correspond to 
  # parameters that are to be optmized
  par_which=which(upper!=0)
  lower=as.vector(lower[par_which])
  upper=as.vector(upper[par_which])
  
  # data to go into fn and gr
  A_0=A
  A_0[]=0
  data=list(hyper_para$rho,hyper_para$gamma,LS,R,Q,hyper_para$lambda2,
            phi,I,par_which,A_0)
  
  # optimize
  control_list=list()
  
  if (dim(meta_dedup)[1]>1000) 
  {
    control_list$maxit=50
    control_list$factr=1e7
  } 
  optim_results=optim(as.vector(A[par_which]),fn=fn,gr=gr,method='L-BFGS-B',
                      lower=lower,upper=upper,data=data,control=control_list)$par
  A[par_which]=optim_results
  A=(A+t(A))/2
  return(A) 
}
update_Q=function(I,A,R,Q,LS,hyper_para){
  LA=diag(rowSums(A))-A
  C=I+4*hyper_para$gamma*(hyper_para$lambda2*LS+LA)+R/hyper_para$rho
  tmp=eigen(C)
  V=tmp$vectors;sigma=tmp$values
  sigma_Q=sigma/2+sqrt(sigma^2/4+hyper_para$m/(2*hyper_para$rho))
  Q=V%*%diag(sigma_Q)%*%solve(V)
  Q=(Q+t(Q))/2
  return(list(Q=Q,LA=LA))
}
update_R=function(I,LA,R,Q,LS,hyper_para){
  R=R-hyper_para$rho*(Q-I-4*hyper_para$gamma*(hyper_para$lambda2*LS+LA))
  return(R)
}