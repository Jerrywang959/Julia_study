x = c(1,3,2,5)
x
x <- c(1,6,2)
x
rm(list=ls())
x=matrix (data=c(1,2,3,4) , nrow=2, ncol =2)
x
x=matrix (c(1,2,3,4) ,2,2)
x
matrix (c(1,2,3,4) ,2,2,byrow =TRUE)
x=rnorm (50)
length(x)
y=x+rnorm (50, mean=50, sd=.1)
cor(x,y)
set.seed (1303)
rnorm (50)
set.seed (3)
y=rnorm (100)
mean(y)
var(y)
sqrt(var(y))
sd(y)
x=rnorm (100)
y=rnorm (100)
plot(x,y)
plot(x,y,xlab=" this is the x-axis",ylab=" this is the y-axis", main=" Plot of X vs Y")
?plot()
seq(0,1,length=10)
x=seq(1,10)
x
x=1:10
x
A=matrix (1:16,4,4)
A
A[2,3]
A[1:3 ,2:4]
A[1:2 ,]
A[ ,1:2]
A[c(1,3) ,c(2,4) ]
A[-c(1,3) ,]
A[c(1,3) ,]
A[-c(1,3) ,-c(1,4)]
dim(A)
setwd('/home/jerrywang/GitHub/Julia_study')
getwd()
Auto=read.table("./R_base/auto.data")
fix(Auto)
Auto=read.table ("./R_base/auto.data", header =T, na.strings="?")
fix(Auto)
dim(Auto)
Auto=na.omit(Auto)
dim(Auto)
names(Auto)
plot(Auto$cylinders , Auto$mpg)
attach(Auto)
plot(cylinders, mpg)
cylinders =as.factor (cylinders)
plot(cylinders , mpg)
plot(cylinders , mpg , col ="red", xlab="cylinders", ylab="MPG")
hist(mpg)
hist(mpg ,col =2)
pairs(Auto)
pairs(~mpg + displacement + horsepower + weight + acceleration , Auto)
summary (Auto)
summary (mpg)
