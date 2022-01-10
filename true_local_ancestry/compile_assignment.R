library(dplyr)

focal <- as.numeric(commandArgs(trailingOnly = T)[1]) + 1
chrs <- as.matrix(read.table("../chr.txt"))

assignment <- data.frame()
for(chr in chrs){

  print(chr)
	df <- read.table(paste0("local_ancestry_chr",chr,"_",focal-1,".csv"), sep = ",")
	df <- df[order(df[,1]),]


	num_ancs   <- length(unique(df[,3]))
	poplabs    <- read.table("msprime.poplabels", header = T)
	labels     <- poplabs[,2]
	#labels     <- as.vector(t(cbind(poplabs[,2],poplabs[,2])))

	#labels[focal]  <- "focal"
	labels     <- as.factor(labels)
	index_tmpl <- as.numeric(labels) - 1
	#nea        <- unique(index_tmpl[which(labels == "Neanderthal")])
	local_comps <- c(max(index_tmpl) + 1:(num_ancs))

	assignment <- rbind(assignment, cbind(data.frame(CHR = chr, bp = 0), t(index_tmpl)))
	current_bp <- 0
	for(i in 1:(nrow(df)-1)){

		index <- index_tmpl
		index[focal] <- local_comps[df[i,3]]
		index_next <- index_tmpl
		index_next[focal] <- local_comps[df[i+1,3]]

		assignment <- rbind(assignment, cbind(data.frame(CHR = chr, bp = round(df[i,1])), t(index)))

	}

}

#print(which(!assignment$bp[-nrow(assignment)] <= assignment$bp[-1]))
print(all(assignment$bp[-nrow(assignment)] <= assignment$bp[-1]))

write.table(t(c(as.matrix(unique(as.matrix(sort(labels)))), 1:num_ancs)), file = "assignment.txt", row.names = F, col.names = F, quote = F)
write.table(assignment, file = "assignment.txt", row.names = F, col.names = F, quote = F, append = T)


#####################

poplabels_all   <- read.table("./msprime.poplabels", header = T)
sample_ages <-  as.numeric(as.matrix(read.table("sample_ages.txt")))
#sample_ages <- as.numeric(as.matrix(sample_ages[seq(1,length(sample_ages),2)]))

sample_ages <- data.frame(ID = poplabels_all[,2], age = sample_ages)
sample_ages %>% group_by(ID) %>% summarize(age = round(mean(age))) -> sample_ages

poplabels <- as.matrix(read.table("./msprime.poplabels", header = T))
sample_ages <- subset(sample_ages, ID %in% poplabels[,2])

colnames(sample_ages)[1] <- "POP"
popl   <- merge(poplabels, sample_ages, by = "POP")
popl   <- popl[,c("ID", "POP", "age")]
colnames(popl)[2] <- "GROUP"
colnames(popl)[3] <- "SAMPLING_TIME"

popl$ID <- factor(popl$ID, levels = unique(poplabels[,1]))
popl <- popl[order(popl$ID),]

write.table(popl, file = "poplabels.txt", row.names = F, quote = F)

