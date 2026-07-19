args=commandArgs(trailingOnly=TRUE)
if(length(args)!=9){
  stop(paste(
    'usage: Rscript export_initialized_core.R EXP CONTIGS ENCODED OUT',
    'LAMBDA1 LAMBDA2 GAMMA RHO M'
  ))
}

input_exp_data=args[1]
input_bcr_data=args[2]
input_encoded_data=args[3]
output_dir=args[4]
hyper=list(
  lambda1=as.numeric(args[5]),lambda2=as.numeric(args[6]),
  gamma=as.numeric(args[7]),rho=as.numeric(args[8]),m=as.numeric(args[9])
)
dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)

library(data.table)
source('R/initiation.R')
source('R/prepare.R')

contigs=read.csv(input_bcr_data,stringsAsFactors=FALSE)
exp_data=read.csv(input_exp_data,stringsAsFactors=FALSE)
row.names(exp_data)=exp_data[,1]
contigs_encoded=read.csv(input_encoded_data,stringsAsFactors=FALSE)[,-1]
contigs=contigs[contigs$cdr3 %in% contigs_encoded$index,]

prepared=preproBCR(contigs,exp_data)
exp_data=as.matrix(prepared$exp_data[,-1])
tmp=prepared$contigs_refined$IGH
colnames(tmp)[3]='contigs'
contigs_refined=tmp[!is.na(tmp$cdr3)&(!is.na(tmp$cdr3_nt)),]
contigs_refined=contigs_refined[contigs_refined$barcode %in% colnames(exp_data),]
exp_data=exp_data[,contigs_refined$barcode]
cdr3exp=paste(
  contigs_refined$v_gene,contigs_refined$cdr3,contigs_refined$j_gene,sep='_'
)
contigs_encoded=contigs_encoded[match(contigs_refined$cdr3,contigs_encoded$index),]
row.names(contigs_encoded)=contigs_refined$barcode
t=contigs_encoded[!duplicated(cdr3exp),1:20]
meta_dedup=contigs_refined[!duplicated(cdr3exp),c('v_gene','j_gene','cdr3')]
initialized=initiation(t,meta_dedup,exp_data,cdr3exp,hyper,NA,NA)

n=nrow(initialized$SI)
writeLines(as.character(n),file.path(output_dir,'n.txt'))
writeLines(initialized$meta_dedup$clone,file.path(output_dir,'node_ids.txt'))
writeBin(as.double(initialized$phi),file.path(output_dir,'phi.bin'),size=8)
writeBin(as.double(initialized$SI),file.path(output_dir,'si.bin'),size=8)
writeBin(as.double(initialized$LS),file.path(output_dir,'ls.bin'),size=8)
