initiation=function(t,meta_dedup,exp_data,cdr3exp,hyper_para,sample,rm_cutoff){
  #1. calculate t labels, build meta
  if(is.na(sample)){
     meta_dedup$cluster=
      as.numeric(as.factor(paste(meta_dedup$v_gene,
                               meta_dedup$j_gene)))
  }else{
    meta_dedup$cluster=
      as.numeric(as.factor(paste(sample,meta_dedup$v_gene,
                                 meta_dedup$j_gene)))
  }
  meta_dedup$clone=cdr3exp[!duplicated(cdr3exp)]
  #1.5 remove clones below cutoff
  if(!is.na(rm_cutoff)){
    cl_tb=table(meta_dedup$cluster)
    cl2rm=as.numeric(names(cl_tb)[cl_tb<rm_cutoff])
    clone2rm=meta_dedup$clone[meta_dedup$cluster%in%cl2rm]
    exp_data=exp_data[,!cdr3exp%in%clone2rm]
    cdr3exp=cdr3exp[!cdr3exp%in%clone2rm]
    t=t[!meta_dedup$cluster%in%cl2rm,]
    meta_dedup=meta_dedup[!meta_dedup$cluster%in%cl2rm,]
  }
  #2. e dists
  keep_ind=apply(exp_data,1,sd)
  pca=prcomp(t(exp_data[keep_ind>quantile(keep_ind,0.9),]))$x[,1:10]
  tmp=as.matrix(dist(pca))
  colnames(tmp)=rownames(tmp)=1:dim(tmp)[1]
  tmp=data.table(cbind(cdr3exp=cdr3exp,as.data.frame(tmp)))
  tmp=tmp[,lapply(.SD,sum),by=cdr3exp]
  setDF(tmp)
  rownames(tmp)=tmp[,1]
  tmp=as.matrix(tmp[,-1])
  tmp=tmp[order(rownames(tmp)),]
  tmp=t(tmp)
  tmp=data.table(cbind(cdr3exp=cdr3exp,as.data.frame(tmp)))
  tmp=tmp[,lapply(.SD,sum),by=cdr3exp]
  setDF(tmp)
  rownames(tmp)=tmp[,1]
  tmp=as.matrix(tmp[,-1])
  tmp=tmp[order(rownames(tmp)),]
  scaler=(4*sum(tmp))/(length(cdr3exp)*(length(cdr3exp)-1))
  clsize=table(cdr3exp)
  clsize=as.numeric(clsize[match(row.names(tmp),names(clsize))])
  master_dist_e=(t(tmp/clsize)/clsize)/scaler
  master_dist_e=master_dist_e[meta_dedup$clone,meta_dedup$clone]
  master_dist_e=(master_dist_e+t(master_dist_e))/2
  #2. t dist
  phi=as.matrix(dist(t))^2
  #3. initialize I
  SI=matrix(0,ncol=nrow(meta_dedup),nrow=nrow(meta_dedup))
  for(i in unique(meta_dedup$cluster)){
    SI[which(meta_dedup$cluster==i),which(meta_dedup$cluster==i)]=1
  }
  diag(SI)=0
  I=diag(rep(1,nrow(meta_dedup)))
  #4. initialize A
  A=hyper_para$lambda1*(1-I)*SI
  #5. initialize LS
  O=master_dist_e*SI
  LS=-(diag(rowSums(O))-O)/sum(SI)
  return(list(meta_dedup=meta_dedup,
              master_dist_e=master_dist_e,
              phi=phi,SI=SI,I=I,A=A,LS=LS))
}
