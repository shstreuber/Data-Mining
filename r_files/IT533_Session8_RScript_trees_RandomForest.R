## Classification is awesome! Together with Clustering, it is the most frequently used method
## for data mining.
## We will be using these libraries

install.packages("tree")
install.packages("party")
install.packages("rpart")
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
install.packages("randomForest")
install.packages("h2o")

## We will be using a few new libraries

install.packages("tree")
install.packages("party")
install.packages("rpart")
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
install.packages("randomForest")

## The Iris dataset is part of the default R package, so we will not be downloading anything
## For a description of the Iris dataset, see https://en.wikipedia.org/wiki/Iris_flower_data_set
## Attributes are Sepal.Length, Sepal.Width, Petal.Length and Petal.Width are used to 
## predict the Species of flowers. 

################# TREE #############################################

## In the "tree" package, function tree() builds a decision tree, 
## Before modeling, the iris data is split below into two subsets: training (70%) and test (30%).

library(tree)

data(iris)
iris
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))
trainDataTree <- iris[ind==1,]
testDataTree <- iris[ind==2,]

myFormula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_tree <- tree(myFormula, data=trainDataTree)

summary(iris_tree)

## Now that we have a model, we can do some predicting. We do this by feeding 
## our test data into our model and comparing the predicted party affiliations 
## with the known ones. The latter is done via the wonderfully named confusion 
## matrix - a table in which true and predicted values for each of the predicted 
## classes are displayed in a matrix format. 

## table(predict(iris_tree), trainDataTree$Species) ## error

## After that, we can have a look at the built tree by printing the rules and 
## plotting the tree.

print(iris_tree)

## Well, isn't that pretty?  How about a real tree plot?

plot(iris_tree)
text(iris_tree)

## The barplot for each leaf node shows the probabilities of an instance
## falling into the three species
## After that, the built tree needs to be tested with the test data (the 30%).

testPred <- predict(iris_tree, newdata = testDataTree) 
## table(testPred, testDataTree$Species) ## error
show(testPred)

## Here is the example from Ledolter

library(MASS) 
library(tree)

## read in the iris data
# data(iris)
iris
iristree <- tree(Species~.,data=iris)
iristree
plot(iristree)
plot(iristree,col=8)
text(iristree,digits=2)
summary(iristree)

## snip.tree has two related functions. If nodes is supplied, it 
## removes those nodes and all their descendants from the tree.
## If nodes is not supplied, the user is invited to select nodes 
## interactively; this makes sense only if the tree has already been plotted.

irissnip=snip.tree(iristree,nodes=c(7,12))
irissnip
plot(irissnip)
text(irissnip)
summary(irissnip)

################# C TREE (conditional inference) ###################

## In the "party" package, function ctree() builds a decision tree, 
## and predict() makes prediction for new data.  
## Before modeling, the iris data is split below into two subsets: training (70%) and test (30%).
## The random seed is set to a fixed value below to make the results reproducible.

data(iris)
iris
set.seed(1234)
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

## We then build a decision tree, and check the prediction result. Function
## ctree() provides some parameters, such as MinSplit, MinBusket, MaxSurrogate and MaxDepth,
## to control the training of decision trees. 
## Below we use default settings to build a decision tree. In the code below, myFormula
## specifies that Species is the target variable and all other variables are independent variables.

library(party)
myFormula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree <- ctree(myFormula, data=trainData)

## Now that we have a model, we can do some predicting. We do this by feeding 
## our test data into our model and comparing the predicted party affiliations 
## with the known ones. The latter is done via the wonderfully named confusion 
## matrix - a table in which true and predicted values for each of the predicted 
## classes are displayed in a matrix format. 

table(predict(iris_ctree), trainData$Species)

## After that, we can have a look at the built tree by printing the rules and 
## plotting the tree.

print(iris_ctree)

## Well, isn't that pretty?  How about a real tree plot?

plot(iris_ctree)

## The barplot for each leaf node shows the probabilities of an instance
## falling into the three species
## After that, the built tree needs to be tested with the test data (the 30%).

testPred <- predict(iris_ctree, newdata = testData)
table(testPred, testData$Species)

#################### R Part Package ###########################

## We are using the function rpart() to build a decision tree,
## and then select the tree with the minimum prediction error. 
## After that, it is applied to new data to make predictions with the predict() function.

data("bodyfat", package="TH.data")
bodyfat
attributes(bodyfat)

## As before, we split the data into training and test subsets and build
## a decision tree on the training data.

set.seed(1234)
ind <- sample(2, nrow(bodyfat), replace=TRUE, prob=c(0.7, 0.3))
bodyfat.train <- bodyfat[ind==1,]
bodyfat.test <- bodyfat[ind==2,]

## Train the decision tree

library(rpart)
myFormula <- DEXfat ~ age + waistcirc + hipcirc + elbowbreadth + kneebreadth
bodyfat_rpart <- rpart(myFormula, data = bodyfat.train, control = rpart.control(minsplit = 10))
attributes(bodyfat_rpart)

## Now we visualize the tree

print(bodyfat_rpart$cptable)
print(bodyfat_rpart)

plot(bodyfat_rpart)
text(bodyfat_rpart, use.n=T)

## Then we select the tree with the minimum prediction error

opt <- which.min(bodyfat_rpart$cptable[,"xerror"])
cp <- bodyfat_rpart$cptable[opt, "CP"]
bodyfat_prune <- prune(bodyfat_rpart, cp = cp)
print(bodyfat_prune)
plot(bodyfat_prune)
text(bodyfat_prune, use.n=T)

## After that, the selected tree is used to make prediction and the predicted values are compared
## with actual labels. In the code below, function abline() draws a diagonal line. The predictions
## of a good model are expected to be equal or very close to their actual values, that is, most points
## should be on or close to the diagonal line.

DEXfat_pred <- predict(bodyfat_prune, newdata=bodyfat.test)
xlim <- range(bodyfat$DEXfat)
plot(DEXfat_pred ~ DEXfat, data=bodyfat.test, xlab="Observed", ylab="Predicted", ylim=xlim, xlim=xlim)
abline(a=0, b=1)



###################### Random Forest ##############################

## We use the randomForest package to build a predictive model for
## the iris data. randomForest() has two limitations:
## 1. It cannot handle data with missing values, and users have to impute data
## before feeding them into the function. 
## 2. There is a limit of 32 to the maximum number of levels of each categorical attribute. 
## An alternative way to build a random forest is to use function cforest() from the party package,
## which is not limited to the above maximum levels. However, generally speaking, categorical
## variables with more levels will make it require more memory and take longer time to build a
## random forest.

## Splitting the iris dataset into test and training data

data(iris)
iris
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))
trainDataRF <- iris[ind==1,]
testDataRF <- iris[ind==2,]

## Then we load package randomForest and train a random forest. In the code below, the formula
## is set to "Species ??? .", which means to predict Species with all other variables in the data

library(randomForest)
rf <- randomForest(Species ~ ., data=trainDataRF, ntree=100, proximity=TRUE)
table(predict(rf), trainDataRF$Species)
print(rf)
attributes(rf)

## After that, we plot the error rates with various number of trees.

plot(rf)

## The importance of variables can be obtained with functions importance() and varImpPlot()

importance(rf)
varImpPlot(rf)

## Finally, the built random forest is tested on test data, and the result is checked with functions
## table() and margin(). The margin of a data point is as the proportion of votes for the correct
## class minus maximum proportion of votes for other classes. Generally speaking, positive margin
## means correct classification.

irisPred <- predict(rf, newdata=testDataRF)
table(irisPred, testDataRF$Species)
plot(margin(rf, testDataRF$Species))
