library(psych)
library(nFactors)

scriptsDir <- '/projects/f_mc1689_1/ReliableFC/docs/scripts'

indivDiffs <- read.csv(paste(scriptsDir,'/indivDiffs.csv',sep=""))
row.names(indivDiffs) <- indivDiffs[, 1]
indivDiffs <- indivDiffs[, -1]

nSubjs <- nrow(indivDiffs)
faOutput <- fa(indivDiffs, nfactors=1, covar=FALSE, fm="minres", n.obs=nSubjs, rotate="oblimin", scores='regression')
loads <- faOutput$loadings
scores <-faOutput$scores

print(faOutput)
print(loads)
print(scores)

write.csv(loads,paste(scriptsDir,'/gMeasureLoads.csv',sep=""))
write.csv(scores,paste(scriptsDir,'/gSubjScores.csv',sep=""))


