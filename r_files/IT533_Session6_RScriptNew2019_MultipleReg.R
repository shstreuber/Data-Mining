########################################################################
# SIMPLE LINEAR REGRESSION REVIEW                                      #
########################################################################

library('nutshell')
library('lattice')
library('MASS')
library('OneR')

# 1. Plot values upon one another
FuelEff <- read.csv("http://www.biz.uiowa.edu/faculty/jledolter/datamining/FuelEfficiency.csv")
FuelEff
str(FuelEff)
FuelEff <- FuelEff[-1]

plot(GPM~WT,data=FuelEff)
plot(GPM~DIS,data=FuelEff)
plot(GPM~NC,data=FuelEff)

# 2. Check the correlation--find the attribute that has the highest r value
cor(FuelEff$GPM,FuelEff$WT)
cor(FuelEff$GPM,FuelEff$DIS)
cor(FuelEff$GPM,FuelEff$NC)

# 3. Set up the linear model for the entire dataset
modWT=lm(GPM~WT,data=FuelEff)  # The output shows us that GPM = -0.006101 + 1.514798*WT
summary(modWT)
plot(FuelEff$WT, FuelEff$GPM, main="Weight vs GPM", xlab="Car Weight ", ylab="Gallons per 100 Miles", pch=19)
abline(lm(FuelEff$GPM~FuelEff$WT), col="red") # regression line (y~x) 
lines(lowess(FuelEff$WT,FuelEff$GPM), col="blue") # lowess line (x,y)

# Just in case: LOWESS (Locally Weighted Scatterplot Smoothing), sometimes called LOESS (locally weighted smoothing), 
# is a popular tool used in regression analysis that creates a smooth line through a timeplot 
# or scatter plot to help you to see relationship between variables and foresee trends.

# And here are the 4 regression plots that we already know
plot(modWT)

modDIS=lm(GPM~DIS,data=FuelEff) # The output shows us that GPM = 2.4330 + 0.0107*WT
summary(modDIS)

plot(FuelEff$DIS, FuelEff$GPM, main="Displacement vs GPM", xlab="Car Displacement", ylab="Gallons per 100 Miles", pch=19)
abline(lm(FuelEff$GPM~FuelEff$DIS), col="red") # regression line (y~x) 
lines(lowess(FuelEff$DIS,FuelEff$GPM), col="blue") # lowess line (x,y)

plot(modDIS)

modNC=lm(GPM~NC,data=FuelEff)  # The output shows us that GPM = 1.0581 + 0.066*WT
summary(modNC)
plot(FuelEff$NC, FuelEff$GPM, main="Displacement vs GPM", xlab="# of Cylinders", ylab="Gallons per 100 Miles", pch=19)
abline(lm(FuelEff$GPM~FuelEff$NC), col="red") # regression line (y~x) 
lines(lowess(FuelEff$NC,FuelEff$GPM), col="blue") # lowess line (x,y)

plot(modNC)

# To play with more visualizations, check out https://www.statmethods.net/graphs/scatterplot.html

# 4. Review the residuals--centered around 0? Symmetrical?

# 5. Review the R-squared value--over 70%? Which one is highest? That is the best predictor

# 6. Now subset the data into a training and a test set using WT to predict GPM
# We start with leave-one-out cross validation

n=length(FuelEff$GPM)
diff=dim(n)
percdiff=dim(n)
for (k in 1:n) {
  train1=c(1:n)
  train=train1[train1!=k]
  
## the R expression "train1[train1!=k]" picks from train1 those 
## elements that are different from k and stores those elements in the
## object train. For k=1, the training set consists of elements that are different from 1; that 
## is 2, 3, n.
  
  regWT=lm(GPM~WT,data=FuelEff[train,])
  predWT=predict(regWT,newdat=FuelEff[-train,])
  obs=FuelEff$GPM[-train]
  diff[k]=obs-predWT
  percdiff[k]=abs(diff[k])/obs
}
me=mean(diff)
rmse=sqrt(mean(diff**2))
mape=100*(mean(percdiff))
me   # mean error
rmse # root mean square error
mape # mean absolute percent error

########################################################################
# MULTIPLE REGRESSION                                                  #
########################################################################

# So far, we have looked only at WT as the best predictor for GPM.
# But could it be possible that several variables when put together, give us a better model?

# 1. Check the correlation of ALL variables
cor(FuelEff)

# We see that GPM is highly correlated with WT,DIS,and NC, and that WT is also correlated with DIS, and NC
# Further, we see that DIS and NC are also highly correlated. This is called multicollinearity.
# Not ideal for multiple regression, but we will run it anyway.

# 2. Establishing the linear model
ALLmod=lm(GPM~.,data=FuelEff)
summary(ALLmod)

# The output shows us that GPM = -2.599357 + (0.787768*WT) + (-0.0048906*DIS) + 
# (0.444157*NC) + (0.023599*HP) + (0.0688149*ACC) + (-0.959634*ET)
# Given its R-squared value of 93% and its p-value of < 2.2e-16, this model shows obvious dependence

# In a large dataset, such a calculation is very expensive. How much better is it really?
# This is where best subset regression can help us find the one or two best predictor variables.
# Best subset regression calculates all regressions in the dataset.
# Best Subset Regression in R

library(leaps)  
X=FuelEff[,2:7] # the predictor variables
y=FuelEff[,1]   # the target variable (GPM)
out=summary(regsubsets(X,y,nbest=3,nvmax=ncol(X)))
tab=cbind(out$which,out$rsq,out$adjr2)
tab

# The table shows us that, if we run just a basic linear regression with WT, this will give us 85% fidelity.
# This means that running the model on just WT wasn't a bad idea.
# However, we could also run the model on TWO predictors. WT and DIS would give us an R-squared value of 89%

WTDISmod=lm(GPM~WT+DIS,data=FuelEff)
summary(WTDISmod)

# Let's take a closer look at this model
attributes(WTDISmod)
coefficients(WTDISmod) # The model equation is GPM = -1.296128 + 2.449718*WT + -0.007821*DIS
fitted.values(WTDISmod) # shows the calculated values
residuals(WTDISmod) # shows the value of all residuals, i.e. the difference between actual and calculated values
plot(WTDISmod)

# We can also plot the model in a 3D plot as below, where function scatterplot3d() creates
# a 3D scatter plot and plane3d() draws the fitted plane. Parameter lab specifies the number of
# tickmarks on the x- and y-axes.

install.packages('scatterplot3d')
library(scatterplot3d)
s3d <- scatterplot3d(FuelEff$WT, FuelEff$DIS, FuelEff$GPM, highlight.3d=T, type="h", lab=c(2,3))
s3d$plane3d(WTDISmod)

# Well, that's pretty. But what about prediction? Is the model with 2 variables just as good as the model 
# with one or with all?

n=length(FuelEff$GPM)
diff=dim(n)
percdiff=dim(n)
for (k in 1:n) {
  train1=c(1:n)
  train=train1[train1!=k]
  WTDISmod=lm(GPM~WT+DIS,data=FuelEff[train,])
  predWTDIS=predict(WTDISmod,newdat=FuelEff[-train,])
  obs=FuelEff$GPM[-train]
  diff[k]=obs-predWTDIS
  percdiff[k]=abs(diff[k])/obs
}
me=mean(diff)
rmse=sqrt(mean(diff**2))
mape=100*(mean(percdiff))
me   # mean error
rmse # root mean square error
mape # mean absolute percent error

# MEH. That was anticlimactic. The accuracy of the model on a training set has improved only marginally (from a MAPE of 8.23% for WT alone to 8.19 for WT and DIS.)

# So, let's compare that to all the predictors in the model.
n=length(FuelEff$GPM)
diff=dim(n)
percdiff=dim(n)
for (k in 1:n) {
  train1=c(1:n)
  train=train1[train1!=k]
  ALLmod=lm(GPM~.,data=FuelEff[train,])
  pred=predict(ALLmod,newdat=FuelEff[-train,])
  obs=FuelEff$GPM[-train]
  diff[k]=obs-pred
  percdiff[k]=abs(diff[k])/obs
}
me=mean(diff)
rmse=sqrt(mean(diff**2))
mape=100*(mean(percdiff))
me   # mean error
rmse # root mean square error
mape # mean absolute percent error 

# Yes, so the model with ALL the predictors gives us a MAPE of 6.75%. That's obviously better.

# Let's do some real prediction now, with the 2-variable model

# First, we set up our new test values in a dataframe. I am choosing WT=5 and DIS= 400 to 410
PREDframe <- data.frame(WT=5, DIS=400:410)

# Now, we are storing the data predictions with the model WTDISmod in this dataframe. 
GPMnew <- predict(WTDISmod, newdata=PREDframe)

# Let's plot the old and the new data!
style <- c(rep(1,12), rep(2,4))
plot(c(FuelEff$GPM,GPMnew), xaxt="n", ylab="GPM", xlab="", pch=style, col=style)
axis(1, at=1:49, las=3, labels=c(paste(FuelEff$WT,FuelEff$DIS,sep="--"), "5--400", 
                                 "5--401", "5--402", "5--403", "5--404", "5--405", "5--406", "5--407", 
                                 "5--408", "5--409", "5--410"))

# The axis still needs a bit of debugging to line up perfectly with the values.

