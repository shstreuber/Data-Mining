########################################################################
# LOGISTIC REGRESSION                                                  #
# from https://stats.idre.ucla.edu/r/dae/logit-regression/             #
########################################################################

# We are going to use a dataset about getting into gradschool
gradschool <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
str(gradschool)
summary(gradschool)
head(gradschool)

# This dataset has a binary response (outcome, dependent) variable called admit. 
# There are three predictor variables: gre, gpa and rank. We will treat the variables gre and gpa as continuous. 
# The variable rank takes on the values 1 through 4. Institutions with a rank of 1 have the highest prestige, 
# while those with a rank of 4 have the lowest.
# To get the standard deviations, we use sapply to apply the sd function to each variable in the dataset.

sapply(gradschool,sd)

# Let's set up a two-way contingency table of categorical outcome and predictors.
# We want to make sure there are no 0 cells
xtabs(~admit + rank, data = gradschool)

# Setting up a logistic regression model with the glm (generalized linear model) function. 
# But first, we need to convert rank to a factor to indicate that rank should be treated as a categorical variable.

gradschool$rank <- factor(gradschool$rank)
gradlogit <- glm(admit ~ gre + gpa + rank, data = gradschool, family = "binomial")

# glm() is usually used for a Generalized Linear Model, but with family = "binomial", it becomes logistic

summary(gradlogit)

# Now, how do we read this output?
# Deviance residuals: Just like regression residuals
# Coefficients Section: We see that both gre and gpa are statistically significant, as are the three terms for rank. 
# The logistic regression coefficients give the change in the log odds of the outcome for a one unit increase 
# in the predictor variable
# GRE: For every one unit change in gre, the log odds of admission (versus non-admission) increases by 0.002.
# GPA: For a one unit increase in gpa, the log odds of being admitted to graduate school increases by 0.804.
# RANK: having attended an undergraduate institution with rank of 2, versus an institution with a rank of 1, 
# changes the log odds of admission by -0.675.

# Use the confint function to obtain confidence intervals for the coefficient estimates. 
# Note that for logistic models, confidence intervals are based on the profiled log-likelihood function. 
# We can also get CIs based on just the standard errors by using the default method.

# CIs using profiled log-likelihood
confint(gradlogit)

# CIs using standard errors
confint.default(gradlogit)

# We can test for an overall effect of rank using the wald.test function of the aod library. 
# The order in which the coefficients are given in the table of coefficients is the same as the order of the terms 
# in the model. This is important because the wald.test function refers to the coefficients by their order in the model.
# We use the wald.test function. b supplies the coefficients, while Sigma supplies the variance covariance matrix 
# of the error terms, finally Terms tells R which terms in the model are to be tested, 
# in this case, terms 4, 5, and 6, are the three terms for the levels of rank.
install.packages('aod')
library(aod)
wald.test(b = coef(gradlogit), Sigma = vcov(gradlogit), Terms = 4:6)

# The chi-squared test statistic of 20.9, with three degrees of freedom is associated with a p-value of 0.00011 
# indicating that the overall effect of rank is statistically significant.

# Below we test whether the coefficient for rank=2 is equal to the coefficient for rank=3. 
l <- cbind(0, 0, 0, 1, -1, 0)
wald.test(b = coef(gradlogit), Sigma = vcov(gradlogit), L = l)

# Exponentiate the coefficients and interpret them as odds-ratios

# Odds ratios only
exp(coef(gradlogit))

# Odds ratios and 95% CI
exp(cbind(OR = coef(gradlogit), confint(gradlogit)))

# For a one unit increase in gpa, the odds of being admitted to graduate school 
# (versus not being admitted) increase by a factor of 2.23.

##############################################################
# Using predicted probabilities                              #
##############################################################

# Setting up a new data frame with means
newdata1 <- with(gradschool, data.frame(gre = mean(gre), gpa = mean(gpa), rank = factor(1:4)))
newdata1

# NOTE: These objects must have the same names as the variables in the logistic regression above 
# (e.g. in this example the mean for gre must be named gre)

# 1. Calculate predicted probabilities
newdata1$rankP <- predict(gradlogit, newdata = newdata1, type = "response")
newdata1

# newdata1$rankP creates a new variable in the dataset (data frame) newdata1 called rankP, 
# the rest of the command tells R that the values of rankP should be predictions made using the predict( ) function. 
# The options within the parentheses base the prediction on the analysis gradlogit with values of 
# the predictor variables coming from newdata1 and set the type of prediction to predicted probability (type="response").
# The output shows that the predicted probability of being accepted into a graduate program is 0.52 for students 
# from the highest prestige undergraduate institutions (rank=1), and 0.18 for students from the lowest ranked institutions (rank=4), holding gre and gpa at their means. 

# 2. Create a table of predicted probabilities varying the value of gre and rank. 
# We are going to plot these, so we will create 100 values of gre between 200 and 800, 
# at each value of rank (i.e., 1, 2, 3, and 4). We want standard errors so we can plot a confidence interval

newdata2 <- with(gradschool, data.frame(gre = rep(seq(from = 200, to = 800, length.out = 100),
                                              4), gpa = mean(gpa), rank = factor(rep(1:4, each = 100))))

newdata3 <- cbind(newdata2, predict(gradlogit, newdata = newdata2, type = "link",
                                    se = TRUE))
newdata3 <- within(newdata3, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

head(newdata3)

# For this output, we get the estimates on the link scale and back transform both the predicted values 
# and confidence limits into probabilities.

# 3. Test the quality of the model
# This test asks whether the model with predictors fits significantly better than a model with just an intercept 
# (i.e., a null model). The test statistic is the difference between the residual deviance for the model with 
# predictors and the null model. The test statistic is distributed chi-squared with degrees of freedom equal 
# to the differences in degrees of freedom between the current and the null model (i.e., the number of predictor variables
# in the model).
with(gradlogit, null.deviance - deviance)

# The degrees of freedom for the difference between the two models is equal to the number of predictor variables
# in the model, and can be obtained using:
with(gradlogit, df.null - df.residual)

# p-value
with(gradlogit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

# The chi-square of 41.46 with 5 degrees of freedom and an associated p-value of less than 0.001 
# tells us that our model as a whole fits significantly better than an empty model. 