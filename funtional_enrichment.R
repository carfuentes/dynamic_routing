
# A PARTIR DE AHORA VAMOS A TRABAJAR CON LOS DATOS QUE YA SON BUENOS: prad_mds y dge_mds


## Functional enrichment analysis focusing on the Gene Ontology biological processes

# Cargo las librerias necesarias
source("https://bioconductor.org/biocLite.R")
#biocLite()


library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(GOstats)
library(DBI)
library(org.Hs.eg.db)
library(xtable)
library(microRNA)
library(mirbase.db)
library(RmiR.Hs.miRNA)
library(Homo.sapiens)
library(topGO)
library(plyr)
browseVignettes("GOstats")

get_genes_annotations <- function(set_genes, db) {
  ann_dict <- vector(mode="list", length=length(keytypes(db)))
  for(column in 1:length(keytypes(db))) {
    allGeneID <- keys(db, keytype = keytypes(db)[column])
    set_genes_id <- set_genes[set_genes %in% allGeneID]
    head(set_genes_id)
    names(ann_dict)[column]=keytypes(db)[column]
    ann_dict[[column]]<-set_genes_id
  }
   return(ann_dict) 
}



##TODOS LOS GENES
#carga los datos de todos los genes
all_geneIDs<-scan("all_gene_ENCODE",what="character")
all_geneIDs<-all_geneIDs[3:16358]

#Get the genes by annotation
ann_dict<- get_genes_annotations(all_geneIDs, org.Hs.eg.db)

#Unify all of them in a vector in ENTREZID 
mask_list<-sapply(ann_dict, function(x) identical(x,character(0)))
ann_dict<-ann_dict[!mask_list]
entrez_id<-sapply(names(ann_dict), function(x) select(org.Hs.eg.db, keys = ann_dict[[x]], columns = "ENTREZID", keytype = x)$ENTREZID, USE.NAMES=TRUE)
entrez_id_unlisted <-unique(unlist(entrez_id,use.names = FALSE))



##RESERVOIR GENES
#carga los datos del reservoir
res_geneIDs<-scan("genes_res_ENCODE",what="character")
#Get the genes by annotation
ann_dict_res<- get_genes_annotations(res_geneIDs, org.Hs.eg.db)

#Unify all of them in a vector in ENTREZID 
mask_list_res<-sapply(ann_dict_res, function(x) identical(x,character(0)))
ann_dict_res<-ann_dict_res[!mask_list_res]
entrez_id_res<-sapply(names(ann_dict_res), function(x) select(org.Hs.eg.db, keys = ann_dict_res[[x]], columns = "ENTREZID", keytype = x)$ENTREZID, USE.NAMES=TRUE)
entrez_id_unlisted_res <-unique(unlist(entrez_id_res,use.names = FALSE))

##EDGE LIST
edge_list_res <- read.table("Dataset1/network_edge_list_ENCODE.csv",header=TRUE,fill = TRUE)
head(edge_list_res)

#create a dictionary for the edge_list
edge_dict <- vector(mode="list", length=length(ann_dict_res$ALIAS))
table_mapping<-select(org.Hs.eg.db, keys = ann_dict_res$ALIAS, columns = "ENTREZID", keytype ="ALIAS")
#table_mapping<-table_mapping[!duplicated(table_mapping[,1]),]
edge_dict<-table_mapping$ENTREZID
names(edge_dict) <- gsub("mir", "hsa-miR", table_mapping$ALIAS)
names(edge_dict) <- gsub("let", "hsa-let", names(edge_dict))
## write in a file the edge_dict
write.table(edge_dict,"mapping_id_to_entrez.txt",row.names=names(edge_dict),col.names = FALSE,quote = FALSE)

##

# 1. Build a parameter object
help("GOHyperGParams-class")
params <- new("GOHyperGParams", geneIds = entrez_id_unlisted_res, universeGeneIds = entrez_id_unlisted, 
              annotation = "org.Hs.eg.db", ontology = "BP", pvalueCutoff = 0.001, testDirection = "over")
conditional(params) <- TRUE # de la diapo siguiente: esta haciendo que es test sea conditional (sino lo 
# ponemos el test seria unconditional)


# 2. Run the functional enrichment analysis
hgOver <- hyperGTest(params)
geneCounts(hgOver)
hgOver
??hyperGTest
# 3. Store and visualize the results
htmlReport(hgOver, file = "gotests.html")

# the resulting object from a call to "hyperGTest" belong to the class of objects GOHyperGResult.
# We can access to this info using:
# therefore, storing the results in a data.frame object enables an automatic processing and filtering of the results:
goresults <- summary(hgOver) 
head(goresults)


# ANALISIS Y PROCESAMIENTO DE LOS RESULTADOS

# GO terms involving a few genes (e.g., < 5) in their size (m) and in their enrichment by DE genes are likely to be less 
# reliable than those that involve many genes.
# In order to try to spot the more reliable GO terms we can filter the previous results by a minimum value on the Count and 
#Size columns and order them by the OddsRatio column:

goresults <- goresults[goresults$Size >= 5 & goresults$Count >= 5, ]
goresults <- goresults[order(goresults$OddsRatio, decreasing = TRUE), ]
head(goresults)
goresults[1:10,]
head(geneCounts(hgOver))

# We can extract the genes that enrich each GO term and paste it to the result as follows:
geneIDs <- geneIdsByCategory(hgOver)[goresults$GOBPID]
head(geneIDs)
geneSYMs <- sapply(geneIDs, function(id) select(org.Hs.eg.db, columns = "ALIAS", key = id, keytype = "ENTREZID")$ALIAS)
geneSYMs <- sapply(geneSYMs, paste, collapse = ", ")
goresults <- cbind(goresults, Genes = geneSYMs)
sapply(geneIDs, function(id) id %in% entrez_id_unlisted_res)


##get dict para el dynamical routing
geneIDs_to_dict <- geneIDs[1:10]
geneId_to_file<-sapply(names(geneIDs_to_dict),function(x) paste(x,paste(geneIDs_to_dict[[x]],collapse=" ")))
geneId_to_file
write(geneId_to_file,"test.txt")
# We can generate an HTML page from a data.frame object using the xtable package:
library(xtable)
xtab <- xtable(goresults, align = "l|c|r|r|r|r|r|p{3cm}|p{3cm}|")
print(xtab, file = "goresults_table.html", type="html")

# INTERESANTE: By setting the argument type="latex" in the previous call to the print() 
# function we can obtain a LaTeX formated table.









