
library(Boruta)
traindata = read.csv("F:/sem3/big_data/final_project/kdd.csv", header = T)
traindata[traindata == ""] <- NA
set.seed(123)
boruta.train = Boruta(Result~., data = traindata, doTrace = 2)
print(boruta.train)

