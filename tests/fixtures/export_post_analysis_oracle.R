args=commandArgs(trailingOnly=TRUE)
if(length(args)!=3){
  stop('usage: Rscript export_post_analysis_oracle.R RESULTS_RDATA ENCODED OUT')
}

load(args[1])
source('R/post_analysis.R')
encoded=read.csv(args[2],stringsAsFactors=FALSE)[,-1]
encoded=encoded[match(results$meta_dedup$cdr3,encoded$index),]
t=as.matrix(encoded[,1:20])
correlations=testCor(
  results$master_dist_e,results$Q,t,results$sparse_graph,results$hyper_para
)

dir.create(args[3],recursive=TRUE,showWarnings=FALSE)
write.table(
  results$master_dist_e,file=file.path(args[3],'master_dist_e.txt'),
  quote=FALSE,row.names=FALSE,col.names=FALSE
)
write.table(
  results$SI,file=file.path(args[3],'si.txt'),
  quote=FALSE,row.names=FALSE,col.names=FALSE
)
writeLines(
  c(as.character(correlations$c1),as.character(correlations$c2)),
  file.path(args[3],'correlations.txt')
)
