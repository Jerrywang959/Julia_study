library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(caTools)
library(randomForest)
library(MASS)
library(ISLR)


Train_ini=read.csv("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Train.csv")
Test=read.csv("/home/jerrywang/Documents/GitHub/note_viajupyterbook/datas/kaggle_2_Test.csv")

set.seed(3000)
split = sample.split(Train_ini$W, SplitRatio = 0.7)
train = subset(Train_ini, split == TRUE)
test = subset(Train_ini, split == FALSE)

set.seed(333)
myForest = randomForest(W ~ OBP + SLG + BA, data = train, ntree = 2000, nodesize = 100, mtry = 1)
predictForest = predict(myForest, newdata = test)





predict=predictForest


mse=sum(abs((predict-test$W))/test$W)/length(test$W)
print(mse)
