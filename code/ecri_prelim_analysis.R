###Ola Ozernov-Palchik oozernov@mit.edu

setwd("/Users/olaozernov-palchik/Dropbox (MIT)/GitHub/BU_TEACH/data")

#import intervention data
#df <- haven::read_sas("ms_s1tier2impact.sas7bdat")
df <- read.csv("ecri2022.csv")

#import moderators
md<-haven::read_sas("ms_s1s2tier2impact.sas7bdat")
md<-read.csv("ms_s1s2tier2impact.csv")
Packages <- c("dplyr", "reshape", "magrittr", "tidyr", "ggplot2", "ggpubr",
              "lme4", "lmerTest","emmeans", "sjstats", "plotrix","dabestr","lmerTest", "grid", "plotrix", "readxl", "lmPerm","gridExtra", "grid","ggpubr",'sjmisc','relaimpo',"pbkrtest","effectsize","lsmeans")

lapply(Packages, library, character.only = TRUE)

#Sample by group and posttest completion
table(df$Cohort,df$Tx)

#check missin data
sum(complete.cases(df$ORFwc)) #2275
sum(!complete.cases(df$ORFwc)) #945

df$na<-ifelse(complete.cases(df$ORFwc)==TRUE,'no_miss','miss')

df$Tx <- factor(df$Tx,
                levels = c(0,1),
                labels = c("control", "tx"))
table(df$na,df$Tx)
df2<-df%>%filter(Keep==1)
df$Time<-as.factor(df$Time)
m1<-lmer(ORFwc~Tx*Time+(1|StuID), data=df)
anova(m1)
lsmeans(m1, list(pairwise ~ Time|Tx), adjust = "tukey") 
library(effsize)


p<-ggplot(data = df, aes(x = Time, y = ORFwc, group = StuID))
p + geom_line() + stat_smooth(aes(group = 1), method = "lm", se = FALSE) +
  stat_summary(aes(group = 1), geom = "point", fun.y = mean, shape = 17, size = 3) +
  xlim("1", "2")+
  facet_grid(. ~ Tx)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))


