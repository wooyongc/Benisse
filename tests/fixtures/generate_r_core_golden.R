args=commandArgs(trailingOnly=TRUE)
output_path=if(length(args)>0) args[1] else 'tests/fixtures/r_core_golden.json'

source('R/update.R')
source('R/util.R')
source('R/post_analysis.R')

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
meta=data.frame(id=seq_len(n))

q_result=update_Q(I,A,R,SI,LS,hyper)
R_updated=update_R(I,q_result$LA,R,q_result$Q,LS,hyper)
A_updated=update_A(matrix(0,n,n),phi,meta,SI,I,A,R_updated,q_result$Q,LS,hyper)
latent=getLatentTdist(q_result$Q,hyper$m,hyper$gamma)

loop_A=hyper$lambda1*(1-I)*SI
loop_R=loop_Q=SI
loop_q_result=update_Q(I,loop_A,loop_R,loop_Q,LS,hyper)
loop_R=update_R(I,loop_q_result$LA,loop_R,loop_q_result$Q,LS,hyper)
loop_A=update_A(matrix(0,n,n),phi,meta,SI,I,loop_A,loop_R,
                loop_q_result$Q,LS,hyper)

previous=matrix(0,nrow=2,ncol=2)
current=matrix(c(0,1,-1,0),nrow=2,byrow=TRUE)

large_n=1001
large_I=diag(large_n)
large_SI=matrix(0,nrow=large_n,ncol=large_n)
large_SI[1,2]=large_SI[2,1]=1
large_SI[2,3]=large_SI[3,2]=1
large_A=0.8*large_SI
large_phi=matrix(0,nrow=large_n,ncol=large_n)
large_zero=matrix(0,nrow=large_n,ncol=large_n)
large_target=0.25*large_SI
large_LA=diag(rowSums(large_target))-large_target
large_Q=large_I+4*hyper$gamma*large_LA
large_meta=data.frame(id=seq_len(large_n))
large_updated=update_A(
  large_zero,large_phi,large_meta,large_SI,large_I,large_A,
  large_zero,large_Q,large_zero,hyper
)

fixture=list(
  schema_version=1,
  generated_with=list(R=R.version.string,implementation='R/update.R + R/post_analysis.R'),
  hyperparameters=hyper,
  small=list(
    inputs=list(I=I,SI=SI,A=A,LS=LS,R=R,phi=phi),
    expected=list(
      LA=q_result$LA,
      Q=q_result$Q,
      R=R_updated,
      A=A_updated,
      latent=latent,
      graph_change_mse=graphChangeMSE(current,previous)
    ),
    admm_one_iteration=list(Q=loop_q_result$Q,R=loop_R,A=loop_A)
  ),
  large_optimizer_branch=list(
    n=large_n,
    active_edges=matrix(c(1,2,2,1,2,3,3,2),ncol=2,byrow=TRUE),
    expected_weights=c(
      large_updated[1,2],large_updated[2,1],
      large_updated[2,3],large_updated[3,2]
    ),
    expected_maxit=50
  )
)

dir.create(dirname(output_path),recursive=TRUE,showWarnings=FALSE)
jsonlite::write_json(
  fixture,output_path,pretty=TRUE,auto_unbox=TRUE,digits=17,matrix='rowmajor'
)
