preproBCR=function(contigs,exp_data)
{
  contigs$barcode=gsub('-','.',contigs$barcode,fixed=TRUE)
  inter=intersect(colnames(exp_data),contigs$barcode)
  exp_data=exp_data[,c('X',inter)]
  
  attach(contigs)
  ind1=is_cell=='True'&high_confidence=='True'&full_length=='True'&productive=='True'&chain=='IGH'
  ind2=is_cell=='True'&high_confidence=='True'&full_length=='True'&productive=='True'&chain=='IGK'
  ind3=is_cell=='True'&high_confidence=='True'&full_length=='True'&productive=='True'&chain=='IGL'
  detach(contigs)
  
  n=0
  contigs_refined=vector(mode = 'list',length = 3)
  for(ind in list(ind1,ind2,ind3)){
    n=n+1
    contigs_tmp=contigs[ind,]
    contigs_tmp=contigs_tmp[order(contigs_tmp$barcode,contigs_tmp$umis,decreasing = T),]
    contigs_tmp=contigs_tmp[!duplicated(contigs_tmp$barcode),]
    contigs_tmp=contigs_tmp[match(inter,contigs_tmp$barcode),]
    contigs_tmp$barcode=inter
    row.names(contigs_tmp)=contigs_tmp$barcode
    contigs_refined[[n]]=contigs_tmp
  }
  names(contigs_refined)=c('IGH','IGK','IGL')
  return(list(exp_data=exp_data,contigs_refined=contigs_refined))
}
