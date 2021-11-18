library(data.table)
Benisse=function(hyper_para,cdr3exp,t,meta_dedup,exp_data,max_iter,
                 save_path,mode,stop_cutoff,simu_initiation=NA,sample=NA,rm_cutoff=NA)
{
  #simu_initiation: intermediate results list returened by simu_data function. Used in production mode only
  #sample: vector, same length as nrow(meta_dedup), used when one library contains multiple samples
  #rm_cutoff: numeric when not NA, used to keep clones large crude graph clusters with >#cutoff clones.
  hyper_para_keep=hyper_para
  print('Begin initiation.')
  
  initialize=initiation(t,meta_dedup,exp_data,cdr3exp,hyper_para,sample,rm_cutoff)
  meta_dedup=initialize$meta_dedup;master_dist_e=initialize$master_dist_e
  SI=initialize$SI;I=initialize$I;A=initialize$A;phi=initialize$phi;LS=initialize$LS
  
  print('Initiation success.')
  R=Q=SI #initialize R and Q
  res=vector('list',10)
  for(i in 1:max_iter)
  {
    print(paste('Iteration:',i))
    re_tmp=update_Q(I,A,R,Q,LS,hyper_para)
    Q=re_tmp$Q;LA=re_tmp$LA
    R=update_R(I,LA,R,Q,LS,hyper_para)
    A=update_A(master_dist_e,phi,meta_dedup,SI,I,A,R,Q,LS,hyper_para)
    hyper_para$rho=hyper_para$rho*2/(1+sqrt(5))
    sparse_graph=A
    sparse_graph[A>0]=1
    
    #Stop criterion
    res_back=res
    res[[11]]=sparse_graph
    res=res[-1]
    
    if(mode=='test'){
      save(Q,R,A,file=paste(save_path,'Iter_',i,'.RData',sep=''))
    }
    if(i>10){
      res_vec=sapply(1:10,function(r) sum(res[[r]]-res_back[[r]])^2/nrow(sparse_graph)^2)
      if(mean(res_vec)<stop_cutoff&sd(res_vec)<1e-4){
        print(mean(res_vec))
        print(paste('Convergence: iteration',i))
        break
      }
    }
  }
  results=list(Q=Q,R=R,A=A,sparse_graph=sparse_graph,
               SI=SI,meta_dedup=meta_dedup,master_dist_e=master_dist_e,hyper_para=hyper_para_keep)
  save(results,file=paste(save_path,'Benisse_results.RData',sep='/'))
  return(results)
}
