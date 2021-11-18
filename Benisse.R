# Benisse: BCR embedding graphical network informed with scRNA Seq

# Following is an example script to run Benisse:
# Rscript ~Benisse/Benisse.R \
# ~Benisse/example/10x_NSCLC_exp.csv \
# ~Benisse/example/10x_NSCLC_contigs.csv \
# ~Benisse/example/encoded_10x_NSCLC.csv \
# ~/Benisse/example \
# 1610 1 100 1 1 10 1e-10

# 1st parameter: input expression matrix. See /example for format.
#   will take the exp data as is (any normalization, log transformation, etc)
# 2nd parameter: contigs files. The BCR seq data from 10X. See /example for format
# 3rd parameter: the encoded BCR matrix. See /example for format
# 4th parameter: output path
# 5th parameter: lambda2. see sup file 1
# 6th parameter: gamma. For smaller datasets, smaller lambda2 and/or gamma
# 7th parameter: max_iter
# 8th parameter: lambda1
# 9th parameter: rho
# 10th parameter: m
# 11th parameter: stop_cutoff

######  read arguments and set up environment  #########

args=commandArgs(trailingOnly = FALSE)
scriptPath=normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))

args = commandArgs(trailingOnly=TRUE)
input_exp_data=args[1] # input exp data
input_bcr_data=args[2] # input bcr data
input_encoded_data=args[3] # input bcr data
save_path=args[4] # output path
if(!dir.exists(save_path)){dir.create(save_path)}

source(paste(scriptPath,'/R/initiation.R',sep=""))
source(paste(scriptPath,'/R/update.R',sep=""))
source(paste(scriptPath,'/R/util.R',sep=""))
source(paste(scriptPath,'/R/post_analysis.R',sep=""))
source(paste(scriptPath,'/R/prepare.R',sep=""))

#######  read and pre-process input data  ##########

contigs=read.csv(input_bcr_data,stringsAsFactors = F)
exp_data=read.csv(input_exp_data,stringsAsFactors = F)
row.names(exp_data)=exp_data[,1]
contigs_encoded=read.csv(input_encoded_data,stringsAsFactors = F)[,-1]
  
tmp=preproBCR(contigs,exp_data)
exp_data=tmp$exp_data
exp_data=as.matrix(exp_data[,-1])
contigs_refined=tmp$contigs_refined

tmp=contigs_refined$IGH
colnames(tmp)[3]='contigs'
contigs_refined=tmp[!is.na(tmp$cdr3)&(!is.na(tmp$cdr3_nt)),]
contigs_refined=contigs_refined[contigs_refined$barcode %in% colnames(exp_data),]
exp_data=exp_data[,contigs_refined$barcode]

cdr3exp=paste(contigs_refined$v_gene,contigs_refined$cdr3,
              contigs_refined$j_gene,sep='_')

contigs_encoded=contigs_encoded[match(contigs_refined$cdr3,
                                      contigs_encoded$index),]
row.names(contigs_encoded)=contigs_refined$barcode
colnames(contigs_encoded)[21]='cdr3'

t=contigs_encoded[!duplicated(cdr3exp),1:20]
meta_dedup=contigs_refined[!duplicated(cdr3exp),c('v_gene','j_gene','cdr3')]

########  other parameters  #############

lambda2=as.numeric(args[5])
gamma=as.numeric(args[6])
max_iter=as.numeric(args[7])
lambda1=as.numeric(args[8])
rho=as.numeric(args[9])
m=as.numeric(args[10])
stop_cutoff=as.numeric(args[11])

mode='default'

hyper_para=list(lambda1=lambda1,
                lambda2=lambda2,
                gamma=gamma,
                rho=rho,
                m=m)

#######  core Benisse  ############

set.seed(123)
results=Benisse(hyper_para,cdr3exp,t,meta_dedup,exp_data,
                max_iter,save_path,mode,stop_cutoff)

cl_tmp=table(cdr3exp)
meta_dedup$clsize=as.numeric(cl_tmp[match(cdr3exp[!duplicated(cdr3exp)],names(cl_tmp))])
meta_dedup$graph_label=convertCluster(results$sparse_graph)

checkDist(results$Q,results$sparse_graph,results$SI,path=save_path,hyper_para)
plotClusters(t,results$Q,results$SI,results$sparse_graph,results$master_dist_e,
  "",path=save_path,clsize = meta_dedup$clsize,hyper_para)

lat_t_dist=getLatentTdist(results$Q,hyper_para$m,hyper_para$gamma)


######  output  ##########

write.csv(meta_dedup,file=paste(save_path,"/clone_annotation.csv",sep=""))
write.table(exp_data,file=paste(save_path,"/cleaned_exp.txt",sep=""),
  quote=F)
write.table(cdr3exp,file=paste(save_path,"/clonality_label.txt",sep=""),
            quote=F,row.names = F,col.names = F)
write.table(cdr3exp,file=paste(save_path,"/clonality_label.txt",sep=""),
            quote=F,row.names = F,col.names = F)
write.table(results$A,file=paste(save_path,"/sparse_graph.txt",sep=""),
            quote=F,row.names = F,col.names = F)
write.table(lat_t_dist,file=paste(save_path,"/latent_dist.txt",sep=""),
            quote=F,row.names = F,col.names = F)

# Ze, we keep you in our heart forever