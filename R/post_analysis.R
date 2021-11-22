#Functions for the primary check
library(ggplot2)
library(igraph)
#1. get latent t
getLatentTdist=function(Q,m,gamma){
  tmp=solve(Q)
  I=rep(1,nrow(tmp))
  latent_t_dist=m*gamma*(diag(tmp)%*%t(I)+I%*%t(diag(tmp))-2*tmp)
  return(latent_t_dist)
}
#2. check connected nodes distance in the latent t
checkDist=function(Q,sparse_graph,crude_graph,path,hyper_para=NA){
  dist_latent_t=getLatentTdist(Q,hyper_para$m,hyper_para$gamma)
  diag(dist_latent_t)=NA
  data2plot=data.frame(dist=c(as.vector(dist_latent_t[sparse_graph==1]),
                              as.vector(dist_latent_t[(sparse_graph!=1)&(crude_graph==1)]),
                              as.vector(dist_latent_t[crude_graph!=1])),
                       label=c(rep('Connected in\nBCR networks',sum(sparse_graph==1)),
                               rep('Not connected,\nsharing V/J genes',sum((sparse_graph!=1)&(crude_graph==1))),
                               rep('Not connected,\nnot sharing V/J genes',sum(crude_graph!=1))))
  data2plot$label=factor(data2plot$label,levels=c('Connected in\nBCR networks','Not connected,\nsharing V/J genes','Not connected,\nnot sharing V/J genes'))
  data2plot=na.omit(data2plot)
  g=ggplot(data2plot,aes(x=label,y=dist,color=label,fill=label))+theme_bw(base_size = 14)+
    theme(axis.title.x = element_blank(),axis.text.x=element_text(angle = 45, hjust = 1),
          legend.position = 'none',plot.title=element_text(size=12))+
    geom_boxplot(width=0.1,alpha=0.7)+
    scale_color_manual(values = c('coral','forestgreen','goldenrod'))+
    scale_fill_manual(values = c('coral','forestgreen','goldenrod'))+
    ylab("BCR Distances in latent space")
    
  if(!is.na(unlist(hyper_para[1])))
  {
    g=g+ggtitle(paste('lambda1:',hyper_para[[1]],'lambda2:',hyper_para[[2]],'\n gamma:',hyper_para[[3]],
                      'rho',hyper_para[[4]]))
  }
  if(is.na(path)){
    return(g)
    }else{ggsave(plot=g,path=path,filename='/in_cross_dist_check.pdf',width=3,height=4, dpi= 400)}
}
#3. plot clusters
plotClusters=function(t,Q,SI,sparse_matrix,master_dist_e,name,path,
                      clsize=NA,hyper_para=NA){
  cor=testCor(master_dist_e,Q,t,sparse_matrix,hyper_para)
  reEdge=round(sum(sparse_matrix)/sum(SI),digits=4)
  pca=prcomp(t)$x
  if(is.na(clsize[1])){
    clsize=rep(0.3,nrow(latent_t))
  }
  data2plot=data.frame(PC1=pca[,1],
                       PC2=pca[,2],
                       clsize=clsize)
  g=ggplot(data2plot)+theme_bw(base_size = 12)+guides(size=FALSE)+
    xlim(range(pca[,1]))+ylim(range(pca[,2]))+
    geom_point(color='forestgreen',alpha=0.7,aes(x=PC1,y=PC2,size=sqrt(clsize)))+
    ggtitle(paste(name,'cor_ab:',round(cor$c1,digit=3),'cor_ac',round(cor$c2,digit=3)))
  tmp=graph.adjacency(sparse_matrix)
  tmp=get.edgelist(tmp)
  data_tmp=data.frame(PC1=pca[tmp[,1],1],PC2=pca[tmp[,1],2],PC1_end=pca[tmp[,2],1],PC2_end=pca[tmp[,2],2])
  g=g+geom_segment(data=data_tmp,color='grey80', alpha=0.5,
                  aes(x=PC1,y=PC2,xend=PC1_end,yend=PC2_end),size=0.1)
  if(!is.na(unlist(hyper_para[1]))){
    g=g+labs(subtitle=paste('lambda1:',hyper_para[[1]],'lambda2:',hyper_para[[2]],'\n gamma:',hyper_para[[3]],
                      'rho',hyper_para[[4]],'\n Edges remained ratio:',reEdge))
  }
  if(is.na(path)){
    return(g)
  }else{ggsave(plot=g,path=path,filename='/connectionplot.pdf',width=8,height=8, dpi= 400)}
}
#4. conver graph to cluster and test ARI
convertCluster=function(sparse_graph){
  clusters=rep("",dim(sparse_graph)[1])
  id=1
  clusters[1]="cluster 1"
  for (i in 1:dim(sparse_graph)[1]){
    for (j in i:dim(sparse_graph)[1]){
      if (i==j) {next}
      if (results$sparse_graph[i,j]==1){
        if (clusters[i]==""){
          id=id+1
          clusters[i]=paste("cluster",id)
        }
        clusters[j]=clusters[i]
      }
    }
  }
  tmp2rm=names(table(clusters))[table(clusters)<2]
  clusters[clusters%in%tmp2rm]=""
  tmplength=sum(clusters=="")
  clusters[clusters==""]=paste('single',1:tmplength)
  return(clusters)
}
testCor=function(master_dist_e,Q,t,sparse_graph,hyper_para,target_points=300){
  latent_dist=getLatentTdist(Q,hyper_para$m,hyper_para$gamma)
  real_dist=as.matrix(dist(t))
  a=as.vector(master_dist_e[sparse_graph==1])
  b=as.vector(latent_dist[sparse_graph==1])
  c=as.vector(real_dist[sparse_graph==1])
  group_size=floor(length(a)/target_points)
  if(group_size<5){
    group_size=floor(length(a)/50)
  }
  aggrlist=rep(c(1:floor(length(a)/group_size)),each=group_size)
  aggrlist=c(aggrlist,rep(aggrlist[length(aggrlist)],(length(a)-length(aggrlist))))
  b=b[order(a,decreasing = F)]
  c=c[order(a,decreasing = F)]
  a=a[order(a,decreasing = F)]
  if(!is.na(aggrlist[1])){
    b=aggregate(b,by=list(aggrlist),median)
    b=b[,2]
    c=aggregate(c,by=list(aggrlist),median)
    c=c[,2]
    a=aggregate(a,by=list(aggrlist),median)
    a=a[,2]
  }
  return(list(c1=cor(a,b,method='spearman'),c2=cor(a,c,method='spearman')))
}

