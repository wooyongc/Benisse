args=commandArgs(trailingOnly=TRUE)
if(length(args)!=1) stop('usage: Rscript export_example_core_inputs.R OUTPUT_DIR')
output_dir=args[1]
dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)

library(data.table)
source('R/initiation.R')
source('R/prepare.R')

contigs=read.csv('example/10x_NSCLC_contigs.csv',stringsAsFactors=FALSE)
exp_data=read.csv('example/10x_NSCLC_exp.csv',stringsAsFactors=FALSE)
row.names(exp_data)=exp_data[,1]
contigs_encoded=read.csv('example/encoded_10x_NSCLC.csv',stringsAsFactors=FALSE)[,-1]
contigs=contigs[contigs$cdr3 %in% contigs_encoded$index,]

prepared=preproBCR(contigs,exp_data)
exp_data=as.matrix(prepared$exp_data[,-1])
tmp=prepared$contigs_refined$IGH
colnames(tmp)[3]='contigs'
contigs_refined=tmp[!is.na(tmp$cdr3)&(!is.na(tmp$cdr3_nt)),]
contigs_refined=contigs_refined[contigs_refined$barcode %in% colnames(exp_data),]
exp_data=exp_data[,contigs_refined$barcode]
cdr3exp=paste(contigs_refined$v_gene,contigs_refined$cdr3,
              contigs_refined$j_gene,sep='_')
contigs_encoded=contigs_encoded[match(contigs_refined$cdr3,contigs_encoded$index),]
row.names(contigs_encoded)=contigs_refined$barcode
t=contigs_encoded[!duplicated(cdr3exp),1:20]
meta_dedup=contigs_refined[!duplicated(cdr3exp),c('v_gene','j_gene','cdr3')]
hyper=list(lambda1=1,lambda2=1610,gamma=1,rho=1,m=10)
initialized=initiation(t,meta_dedup,exp_data,cdr3exp,hyper,NA,NA)

load('example/Benisse_results.RData')
n=nrow(initialized$SI)
writeLines(as.character(n),file.path(output_dir,'n.txt'))
writeBin(as.double(initialized$phi),file.path(output_dir,'phi.bin'),size=8)
writeBin(as.double(initialized$SI),file.path(output_dir,'si.bin'),size=8)
writeBin(as.double(initialized$LS),file.path(output_dir,'ls.bin'),size=8)
writeBin(as.double(results$A),file.path(output_dir,'expected_a.bin'),size=8)
writeBin(as.integer(results$sparse_graph),file.path(output_dir,'expected_edges.bin'),size=4)
