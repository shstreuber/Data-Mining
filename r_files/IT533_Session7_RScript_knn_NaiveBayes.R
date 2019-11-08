## Classification is awesome! Together with Clustering, it is the most frequently used method
## for data mining.
## We will be using a few new libraries

install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")

###########################################################################
#                      k Nearest Neighbor                                 #
###########################################################################

#### Example 1: Forensic Glass  (http://ugrad.stat.ubc.ca/R/library/e1071/html/Glass.html) ####

# CSI Time!!! We are going to investigate a crime scene and determine from the chemical composition of the glass 
# fragments we find next to the victim what happened.

library(textir) ## needed to standardize the data
library(MASS)   ## a library of example datasets

data(fgl) 		## loads the data into R; see help(fgl)
help(fgl)
fgl

## data consists of 214 cases
## here are illustrative box plots of the features stratified by 
## glass type
par(mfrow=c(3,3), mai=c(.3,.6,.1,.1))
plot(RI ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Al ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Na ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Mg ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Ba ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Si ~ type, data=fgl, col=c(grey(.2),2:6))
plot(K ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Ca ~ type, data=fgl, col=c(grey(.2),2:6))
plot(Fe ~ type, data=fgl, col=c(grey(.2),2:6))

## for illustration, consider the RIxAl plane
## use nt=200 training cases to find the nearest neighbors for 
## the remaining 14 cases. These 14 cases become the evaluation 
## (test, hold-out) cases

n=length(fgl$type)
nt=200
set.seed(1) ## to make the calculations reproducible in repeated runs
train <- sample(1:n,nt)

## Standardization of the data is preferable, especially if 
## units of the features are quite different
## could do this from scratch by calculating the mean and 
## standard deviation of each feature, and use those to 
## standardize.
## Even simpler, use the normalize function in the R-package textir; 
## it converts data frame columns to mean-zero sd-one

## x <- normalize(fgl[,c(4,1)])
x=fgl[,c(4,1)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])

x[1:3,]

library(class)  
nearest1 <- knn(train=x[train,],test=x[-train,],cl=fgl$type[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=fgl$type[train],k=5)
data.frame(fgl$type[-train],nearest1,nearest5)

## plot them to see how it worked
par(mfrow=c(1,2))
## plot for k=1 (single) nearest neighbor
plot(x[train,],col=fgl$type[train],cex=.8,main="1-nearest neighbor")
points(x[-train,],bg=nearest1,pch=21,col=grey(.9),cex=1.25)
## plot for k=5 nearest neighbors
plot(x[train,],col=fgl$type[train],cex=.8,main="5-nearest neighbors")
points(x[-train,],bg=nearest5,pch=21,col=grey(.9),cex=1.25)
legend("topright",legend=levels(fgl$type),fill=1:6,bty="n",cex=.75)

## calculate the proportion of correct classifications on this one 
## training set

pcorrn1=100*sum(fgl$type[-train]==nearest1)/(n-nt)
pcorrn5=100*sum(fgl$type[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5

## cross-validation (leave one out)
pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,fgl$type,k)
  pcorr[k]=100*sum(fgl$type==pred)/n
}
pcorr
## Note: Different runs may give you slightly different results as ties 
## are broken at random

## using all nine dimensions (RI plus 8 chemical concentrations)

## x <- normalize(fgl[,c(1:9)])
x=fgl[,c(1:9)]
for (j in 1:9) {
  x[,j]=(x[,j]-mean(x[,j]))/sd(x[,j])
}

nearest1 <- knn(train=x[train,],test=x[-train,],cl=fgl$type[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=fgl$type[train],k=5)
data.frame(fgl$type[-train],nearest1,nearest5)

## calculate the proportion of correct classifications

pcorrn1=100*sum(fgl$type[-train]==nearest1)/(n-nt)
pcorrn5=100*sum(fgl$type[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5

## cross-validation (leave one out)

pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,fgl$type,k)
  pcorr[k]=100*sum(fgl$type==pred)/n
}
pcorr






#### ******* Example 2: German Credit Data ******* ####
#### ******* data on 1000 loans ****************** ####

library(textir)	## needed to standardize the data
library(class)	## needed for knn

## read data and create some `interesting' variables
credit <- read.csv("http://www.biz.uiowa.edu/faculty/jledolter/datamining/germancredit.csv")
credit
str(credit)

credit$Default <- factor(credit$Default)

## re-level the credit history and a few other variables
credit$history = factor(credit$history, levels=c("A30","A31","A32","A33","A34"))
levels(credit$history) = c("good","good","poor","poor","terrible")
credit$foreign <- factor(credit$foreign, levels=c("A201","A202"), labels=c("foreign","german"))
credit$rent <- factor(credit$housing=="A151")
credit$purpose <- factor(credit$purpose, levels=c("A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A410"))
levels(credit$purpose) <- c("newcar","usedcar",rep("goods/repair",4),"edu",NA,"edu","biz","biz")

## for demonstration, cut the dataset to these variables
credit <- credit[,c("Default","duration","amount","installment","age",                    "history", "purpose","foreign","rent")]
credit[1:3,]
summary(credit) # check out the data

## for illustration we consider just 3 loan characteristics:
## amount,duration,installment
## Standardization of the data is preferable, especially if 
## units of the features are quite different
## We use the normalize function in the R-package textir; 
## it converts data frame columns to mean-zero sd-one

## x <- normalize(credit[,c(2,3,4)])
x=credit[,c(2,3,4)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])
x[,3]=(x[,3]-mean(x[,3]))/sd(x[,3])

x[1:3,]

## training and prediction datasets
## training set of 900 borrowers; want to classify 100 new ones
set.seed(1)
train <- sample(1:1000,900) ## this is training set of 900 borrowers
xtrain <- x[train,]
xnew <- x[-train,]
ytrain <- credit$Default[train]
ynew <- credit$Default[-train]

## k-nearest neighbor method
library(class)
nearest1 <- knn(train=xtrain, test=xnew, cl=ytrain, k=1)
nearest3 <- knn(train=xtrain, test=xnew, cl=ytrain, k=3)
data.frame(ynew,nearest1,nearest3)[1:10,]

## calculate the proportion of correct classifications
pcorrn1=100*sum(ynew==nearest1)/100
pcorrn3=100*sum(ynew==nearest3)/100
pcorrn1
pcorrn3

## plot for 3nn
plot(xtrain[,c("amount","duration")],col=c(4,3,6,2)[credit[train,"installment"]],pch=c(1,2)[as.numeric(ytrain)],main="Predicted default, by 3 nearest neighbors",cex.main=.95)
points(xnew[,c("amount","duration")],bg=c(4,3,6,2)[credit[train,"installment"]],pch=c(21,24)[as.numeric(nearest3)],cex=1.2,col=grey(.7))
legend("bottomright",pch=c(1,16,2,17),bg=c(1,1,1,1),legend=c("data 0","pred 0","data 1","pred 1"),title="default",bty="n",cex=.8)
legend("topleft",fill=c(4,3,6,2),legend=c(1,2,3,4),title="installment %",horiz=TRUE,bty="n",col=grey(.7),cex=.8)

## above was for just one training set
## cross-validation (leave one out)
pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,cl=credit$Default,k)
  pcorr[k]=100*sum(credit$Default==pred)/1000
}
pcorr


###########################################################################
#     NAIVE BAYES (conditional probability)                               #
###########################################################################

## Much of this code comes from the fantastic blog, but has been debugged and annotated where necessary:
## https://eight2late.wordpress.com/2015/11/06/a-gentle-introduction-to-naive-bayes-classification-using-r/

## load mlbench library
library(mlbench)

## load HouseVotes84 dataset
data("HouseVotes84")

## Let's look around that dataset a little

## barplots for specific issue
plot(as.factor(HouseVotes84[,2]))
title(main="Votes cast for issue", xlab="vote", ylab="# reps")
      
## by party
plot(as.factor(HouseVotes84[HouseVotes84$Class=="republican",2]))
title(main="Republican votes cast for issue 1", xlab="vote", ylab="# reps")
plot(as.factor(HouseVotes84[HouseVotes84$Class=="democrat",2]))
title(main="Democrat votes cast for issue 1", xlab="vote", ylab="# reps")

## The classification problem at hand is to figure out the party affiliation 
## from a knowledge of voting patterns. For simplicity let us assume that there 
## are only 3 issues voted on instead of the 16 in the actual dataset. 
## In concrete terms we want to answer the question, 
## "what is the probability that a representative is, say, a democrat (D) 
## given that he or she has voted, say,  (v1 = y, v2=n,v3 = y) on the three issues?" 

## Just for fun, we'll treat the NAs differently. We'll impute (i.e. assign) NA values 
## for a given issue and party by looking at how other representatives 
## from the same party voted on the issue. This is very much in keeping 
## with the Bayesian spirit: we infer unknowns based on a justifiable belief - 
## that is, belief based on the evidence.

## To do this we write two functions: one to  compute the number of NA values 
## for a given issue (vote) and class (party affiliation), and the other to 
## calculate the fraction of yes votes for a given issue (column) and class 
## (party affiliation).

## Functions needed for imputation
## function to return number of NAs by vote and class (democrat or republican)

na_by_col_class <- function (col,cls){return(sum(is.na(HouseVotes84[,col]) & HouseVotes84$Class==cls))}

## Function to compute the conditional probability that a member of a party will cast a 'yes' vote for
## a particular issue. The probability is based on all members of the party who #actually cast a vote on the issue (ignores NAs).

p_y_col_class <- function(col,cls){
  sum_y<-sum(HouseVotes84[,col]=="y" & HouseVotes84$Class==cls,na.rm = TRUE)
  sum_n<-sum(HouseVotes84[,col]=="n" & HouseVotes84$Class==cls,na.rm = TRUE)
  return(sum_y/(sum_y+sum_n))}

## Check that functions work!
p_y_col_class(2,"democrat")
p_y_col_class(2,"republican")
na_by_col_class(2,"democrat")
na_by_col_class(2,"republican")

## We can now impute the NA values based on the above. We do this by randomly 
## assigning values ( y or n) to NAs, based on the proportion of members of a 
## party who have voted y or n. In practice, we do this by invoking the uniform 
## distribution and setting an NA value to y if the random number returned is 
## less than the probability of a yes vote and to n otherwise.

## impute missing values.

for (i in 2:ncol(HouseVotes84)) {
  if(sum(is.na(HouseVotes84[,i])>0)) {
    c1 <- which(is.na(HouseVotes84[,i])& HouseVotes84$Class=="democrat",arr.ind = TRUE)
    c2 <- which(is.na(HouseVotes84[,i])& HouseVotes84$Class=="republican",arr.ind = TRUE)
    HouseVotes84[c1,i] <-
      ifelse(runif(na_by_col_class(i,"democrat"))<p_y_col_class(i,"democrat"),"y","n")
    HouseVotes84[c2,i] <-
      ifelse(runif(na_by_col_class(i,"republican"))<p_y_col_class(i,"republican"),"y","n")}
}

## Note that the which function filters  indices by the criteria specified in the arguments 
## and ifelse is a vectorised conditional function which enables us to apply logical criteria 
## to multiple elements of a vector.  At this point it is a good idea to check that the NAs 
## in each column have been set according to the voting patterns of non-NAs for a given party. 
## You can use the p_y_col_class() function to check that the new probabilities are close to the old ones.

## Then we divide into test and training sets.  
## We also create new col "train" and assign 1 or 0 in 80/20 proportion via random uniform dist

HouseVotes84[,"train"] <- ifelse(runif(nrow(HouseVotes84))<0.80,1,0)

## Get col number of train / test indicator column (needed later)

trainColNum <- grep('train', names(HouseVotes84))

## separate training and test sets and remove training column before modeling

trainHouseVotes84 <- HouseVotes84[HouseVotes84$train==1,-trainColNum]
testHouseVotes84 <- HouseVotes84[HouseVotes84$train==0,-trainColNum]

## Now we can build the Naive Bayes model

## Load e1071 library and invoke naiveBayes method

library(e1071)
nb_model <- naiveBayes(Class~.,data = trainHouseVotes84)
nb_model
summary(nb_model)
str(nb_model)

## Now that we have a model, we can do some predicting. We do this by feeding 
## our test data into our model and comparing the predicted party affiliations 
## with the known ones. The latter is done via the wonderfully named confusion 
## matrix - a table in which true and predicted values for each of the predicted 
## classes are displayed in a matrix format. 

## ... and the moment of reckoning

nb_test_predict <- predict(nb_model,testHouseVotes84[,-1])

## Building the confusion matrix

table(pred=nb_test_predict,true=testHouseVotes84$Class)

## Remember that in the confusion matrix (as defined above), the true values 
## are in columns and the predicted values in rows. 
## The output doesn't look too bad, does it?
## However, we need to keep in mind that this could well be quirk of the choice of dataset. 
## To address this, we should get a numerical measure of the efficacy of the algorithm 
## and for different training and testing datasets. A simple measure of efficacy would be 
## the fraction of predictions that the algorithm gets right.

## fraction of correct predictions

mean(nb_test_predict==testHouseVotes84$Class)

## But how good is this prediction? This question cannot be answered with only a single 
## run of the model; we need to do many runs and look at the spread of the results. To do 
## this, we'll create a function which takes the number of times the model should be run 
## and the training fraction as inputs and spits out a vector containing the proportion 
## of correct predictions for each run.

## Function to create, run and record model results
nb_multiple_runs <- function(train_fraction,n){
  fraction_correct <- rep(NA,n)
  for (i in 1:n){
    HouseVotes84[,"train"] <- ifelse(runif(nrow(HouseVotes84))<train_fraction,1,0)
    trainColNum <- grep('train',names(HouseVotes84))
    trainHouseVotes84 <- HouseVotes84[HouseVotes84$train==1,-trainColNum]
    testHouseVotes84 <- HouseVotes84[HouseVotes84$train==0,-trainColNum]
    nb_model <- naiveBayes(Class~.,data = trainHouseVotes84)
    nb_test_predict <- predict(nb_model,testHouseVotes84[,-1])
    fraction_correct[i] <- mean(nb_test_predict==testHouseVotes84$Class)
  }
  return(fraction_correct)
}

## Let's do 20 runs, 80% of data randomly selected for training set in each run

fraction_correct_predictions <- nb_multiple_runs(0.8,20)
fraction_correct_predictions

## Summary of results

summary(fraction_correct_predictions)

## Standard deviation

sd(fraction_correct_predictions)

## Not too shabby! It looks like all results are reasonably close together!
## The standard deviation is also in the ballpark.



