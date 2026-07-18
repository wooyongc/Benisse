update_A_corrected=function(phi,SI,I,A,R,Q,LS,hyper_para){
  upper=hyper_para$lambda1*(1-I)*SI
  edges=which(upper.tri(upper)&upper!=0,arr.ind=TRUE)
  maxit=if(nrow(A)>1000) 50 else 100

  build_A=function(parameters){
    candidate=matrix(0,nrow=nrow(A),ncol=ncol(A))
    if(length(parameters)>0){
      candidate[edges]=parameters
      candidate[cbind(edges[,2],edges[,1])]=parameters
    }
    candidate
  }

  terms=function(parameters){
    candidate=build_A(parameters)
    LA=diag(rowSums(candidate))-candidate
    U=LA-(Q-I-R/hyper_para$rho)/(4*hyper_para$gamma)+
      hyper_para$lambda2*LS
    residual=Q-I-4*hyper_para$gamma*(hyper_para$lambda2*LS+LA)-
      R/hyper_para$rho
    objective=0.5*hyper_para$rho*norm(residual,type='F')^2+sum(candidate*phi)
    diagonal=diag(U)
    gradient=2*phi[edges]+16*hyper_para$rho*hyper_para$gamma^2*(
      diagonal[edges[,1]]+diagonal[edges[,2]]-U[edges]-
      U[cbind(edges[,2],edges[,1])]
    )
    list(objective=objective,gradient=gradient)
  }

  if(nrow(edges)==0){
    value=terms(numeric(0))
    return(list(
      A=build_A(numeric(0)),convergence=0,message='No active crude-graph edges',
      counts=c('function'=0,'gradient'=0),iterations=0,objective=value$objective,
      projected_gradient_norm=0,maxit=maxit
    ))
  }

  fn=function(parameters) terms(parameters)$objective
  gr=function(parameters) terms(parameters)$gradient
  initial=A[edges]
  lower=rep(0,nrow(edges))
  upper_values=upper[edges]
  optimization=optim(
    initial,fn=fn,gr=gr,method='L-BFGS-B',lower=lower,upper=upper_values,
    control=list(maxit=maxit,factr=1e7,pgtol=1e-8,lmm=5)
  )
  final_gradient=gr(optimization$par)
  projected=optimization$par-pmin(pmax(optimization$par-final_gradient,lower),upper_values)
  list(
    A=build_A(optimization$par),convergence=optimization$convergence,
    message=optimization$message,counts=optimization$counts,
    iterations=unname(optimization$counts[['gradient']]),objective=optimization$value,
    projected_gradient_norm=max(abs(projected)),maxit=maxit
  )
}
