args=commandArgs(trailingOnly=TRUE)
output_path=if(length(args)>0) args[1] else 'tests/fixtures/r_core_golden.json'

source('R/update.R')
source('tests/fixtures/corrected_update_A.R')

latent_distances=function(Q,m,gamma){
  inverse=solve(Q)
  ones=rep(1,nrow(inverse))
  m*gamma*(diag(inverse)%*%t(ones)+ones%*%t(diag(inverse))-2*inverse)
}

hyper=list(lambda1=0.8,lambda2=1.5,gamma=0.7,rho=1.2,m=3)
n=4
I=diag(n)
SI=matrix(c(
  0,1,1,0,
  1,0,1,0,
  1,1,0,1,
  0,0,1,0
),nrow=n,byrow=TRUE)
A=0.3*SI
LS=matrix(c(
  -0.20, 0.10, 0.10, 0.00,
   0.10,-0.25, 0.10, 0.05,
   0.10, 0.10,-0.30, 0.10,
   0.00, 0.05, 0.10,-0.15
),nrow=n,byrow=TRUE)
R=matrix(c(
   0.10, 0.02, 0.00, 0.00,
   0.02,-0.05, 0.01, 0.00,
   0.00, 0.01, 0.03, 0.02,
   0.00, 0.00, 0.02,-0.08
),nrow=n,byrow=TRUE)
phi=as.matrix(dist(matrix(c(0,0,1,0,0,2,2,2),ncol=2,byrow=TRUE)))^2

q_result=update_Q(I,A,R,SI,LS,hyper)
R_updated=update_R(I,q_result$LA,R,q_result$Q,LS,hyper)
A_result=update_A_corrected(phi,SI,I,A,R_updated,q_result$Q,LS,hyper)

loop_A=hyper$lambda1*(1-I)*SI
loop_R=loop_Q=SI
loop_q_result=update_Q(I,loop_A,loop_R,loop_Q,LS,hyper)
loop_R=update_R(I,loop_q_result$LA,loop_R,loop_q_result$Q,LS,hyper)
loop_A_result=update_A_corrected(
  phi,SI,I,loop_A,loop_R,loop_q_result$Q,LS,hyper
)

run_small_admm=function(max_iter=30,stop_cutoff=1e-10){
  current_hyper=hyper
  A=current_hyper$lambda1*(1-I)*SI
  Q=R=SI
  history=list()
  optimizer_convergence=c()
  converged=FALSE
  change_mean=NA
  change_sd=NA
  for(iteration in seq_len(max_iter)){
    q_result=update_Q(I,A,R,Q,LS,current_hyper)
    Q=q_result$Q
    R=update_R(I,q_result$LA,R,Q,LS,current_hyper)
    a_result=update_A_corrected(phi,SI,I,A,R,Q,LS,current_hyper)
    if(a_result$convergence!=0) stop('corrected R optimizer failed')
    optimizer_convergence=c(optimizer_convergence,a_result$convergence)
    A=a_result$A
    current_hyper$rho=current_hyper$rho*2/(1+sqrt(5))
    sparse=A>0
    history=append(history,list(sparse))
    if(length(history)>11) history=history[-1]
    if(iteration>10){
      changes=sapply(seq_len(10),function(index){
        sum((history[[index+1]]-history[[index]])^2)/n^2
      })
      change_mean=mean(changes)
      change_sd=sd(changes)
      if(change_mean<stop_cutoff&&change_sd<1e-4){
        converged=TRUE
        break
      }
    }
  }
  list(
    Q=Q,R=R,A=A,sparse_graph=sparse,
    latent=latent_distances(Q,hyper$m,hyper$gamma),iterations=iteration,
    converged=converged,graph_change_mean=change_mean,graph_change_sd=change_sd,
    optimizer_convergence=optimizer_convergence
  )
}

run_legacy_admm=function(max_iter=30,stop_cutoff=1e-10){
  current_hyper=hyper
  A=current_hyper$lambda1*(1-I)*SI
  Q=R=SI
  history=list()
  converged=FALSE
  change_mean=NA
  change_sd=NA
  for(iteration in seq_len(max_iter)){
    q_result=update_Q(I,A,R,Q,LS,current_hyper)
    Q=q_result$Q
    R=update_R(I,q_result$LA,R,Q,LS,current_hyper)
    # Frozen production-R behavior: directed coordinates, mismatched gradient,
    # ignored optim status, and post-hoc symmetrization.
    A=update_A(
      matrix(0,n,n),phi,data.frame(node=seq_len(n)),SI,I,A,R,Q,LS,current_hyper
    )
    current_hyper$rho=current_hyper$rho*2/(1+sqrt(5))
    sparse=A>0
    history=append(history,list(sparse))
    if(length(history)>11) history=history[-1]
    if(iteration>10){
      changes=sapply(seq_len(10),function(index){
        sum((history[[index+1]]-history[[index]])^2)/n^2
      })
      change_mean=mean(changes)
      change_sd=sd(changes)
      if(change_mean<stop_cutoff&&change_sd<1e-4){
        converged=TRUE
        break
      }
    }
  }
  list(
    A=A,sparse_graph=sparse,iterations=iteration,converged=converged,
    graph_change_mean=change_mean,graph_change_sd=change_sd
  )
}

fixture=list(
  schema_version=2,
  generated_with=list(
    R=R.version.string,
    implementation='corrected symmetric-edge update_A + R update_Q/update_R'
  ),
  hyperparameters=hyper,
  optimizer_policy=list(small_maxit=100,large_maxit=50,factr=1e7,pgtol=1e-8),
  small=list(
    inputs=list(I=I,SI=SI,A=A,LS=LS,R=R,phi=phi),
    expected=list(
      LA=q_result$LA,Q=q_result$Q,R=R_updated,A=A_result$A,
      latent=latent_distances(q_result$Q,hyper$m,hyper$gamma),
      graph_change_mse=0.5
    ),
    optimizer=list(
      convergence=A_result$convergence,iterations=A_result$iterations,
      objective=A_result$objective,
      projected_gradient_norm=A_result$projected_gradient_norm
    ),
    admm_one_iteration=list(
      Q=loop_q_result$Q,R=loop_R,A=loop_A_result$A,
      optimizer_convergence=loop_A_result$convergence
    ),
    admm=run_small_admm(),
    legacy_production_gap=run_legacy_admm()
  )
)

dir.create(dirname(output_path),recursive=TRUE,showWarnings=FALSE)
jsonlite::write_json(
  fixture,output_path,pretty=TRUE,auto_unbox=TRUE,digits=17,matrix='rowmajor'
)
