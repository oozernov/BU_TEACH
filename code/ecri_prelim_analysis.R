setwd("~/Dropbox (MIT)/BU_TEACH/ECRI")
df <- haven::read_sas("ms_s1tier2impact.sas7bdat")

#Sample by group and posttest completion
table(df$Cohort,df$Tx)

#check missin data
sum(complete.cases(df$ORFwc)) #2275
sum(!complete.cases(df$ORFwc)) #945

library(dplyr)
df$na<-ifelse(complete.cases(df$ORFwc)==TRUE,'no_miss','miss')

df$Tx <- factor(df$Tx,
                    levels = c(0,1),
                    labels = c("control", "tx"))
table(df$na,df$Tx)

