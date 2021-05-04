#load relevant libraries
library(fda)
library(splines)
library(e1071)
library(MASS)
library(tree)
library(randomForest)
library(ISLR)
library(gbm)

#Question 1 

#load the data
?CanadianWeather
attach(CanadianWeather)

# Question 1A ------------------------------------------------
test_data <- data.frame(CanadianWeather$dailyAv[,"Victoria",])
days <- 1:365
fit1 = lm(test_data$Precipitation.mm ~ bs(days,knots=c(90,181,273)))
fit2 = lm(test_data$Precipitation.mm ~ bs(days,knots=c(181)))
pred1 = predict(fit1, newdata=list(days), se=T)
pred2 = predict(fit2, newdata=list(days), se=T)
plot(days, test_data$Precipitation.mm,  col="gray",
     xlab = "Day of the Year", ylab = "Precipitation (mm)")
title("Splines")

#plot the confidence interval for the spline with three nodes
lines(days, pred1$fit, lwd=2, col = "red")
lines(days, pred1$fit+2*pred1$se, lty="dashed", col = "red")
lines(days, pred1$fit-2*pred1$se, lty="dashed", col = "red")

#plot the confidence interval for the spline with one node
lines(days, pred2$fit, lwd=2, col = "blue")
lines(days, pred2$fit+2*pred2$se, lty="dashed", col = "blue")
lines(days, pred2$fit-2*pred2$se, lty="dashed", col = "blue")

# ANALYSIS-------------------------------------------
# The spline with three nodes is marked in red, and the 
# spline with one node is marked in blue.
# The spline with three nodes oscillates more than the 
# spline with a single node.
# The splines closesly follow each other throughout the first
# ten months of the year,but deviate from each other predictions 
# for November and December.
#-----------------------------------------------------

# Question 1B -------------------------------------------
#perform cross validation to optimize df and lambda
fit3 = smooth.spline(days, test_data$Precipitation.mm, cv=TRUE)
pred3 <- predict(fit3, newdata = days)
print(paste("After cross validation, the optimal degrees of freedom is", fit3$df))
print(paste("After cross validation, the optimal lambda  is", fit3$lambda))

#plot the fitted model
plot(days, test_data$Precipitation.mm,  col="gray",
     xlab = "Day of the Year", ylab = "Precipitation (mm)")
title("Smoothing Splines")
lines(fit3, col = "green", lw = 2)

#Question 1C---------------------------------------------
fit4 <- loess(test_data$Precipitation.mm ~ days, span = 0.2)
fit5 <- loess(test_data$Precipitation.mm ~ days, span = 0.5)
pred4 <- predict(fit4, newdata = days)
pred5 <- predict(fit5, newdata = days)
#plot the data
plot(days, test_data$Precipitation.mm,  col="gray",
     xlab = "Day of the Year", ylab = "Precipitation (mm)")
title("Local Regression")
lines(days, pred4, col = "yellow", lw = 2)
lines(days, pred5, col = "green", lw = 2)
legend("topright", legend=c("Span=0.2","Span=0.5"), col=c("yellow","green"),
       lty=1, lwd=2, cex=.8)

# Question 1D------------------------------------------------
test_vals <- test_data$Precipitation.mm
mspe1 <- sum((pred1$fit - test_vals)^2)
mspe2 <- sum((pred2$fit - test_vals)^2)
mspe3 <- sum((pred3$y - test_vals)^2)
mspe4 <- sum((pred4 - test_vals)^2)
mspe5 <- sum((pred5 - test_vals)^2)
print(paste("The mean squared prediction error for the spline with three nodes is", mspe1))
print(paste("The mean squared prediction error for the spline with one is", mspe2))
print(paste("The mean squared prediction error for the smoothed spline is", mspe3))
print(paste("The mean squared prediction error for the local regression with span = 0.2 is", mspe4))
print(paste("The mean squared prediction error for the local regression with span = 0.5 is", mspe5))


#Question 2-------------------------------------------------------
?iris
train_data = subset(iris, select = c(Sepal.Length, Sepal.Width, Species))

#fit a linear svm
linear_svm = svm(as.factor(Species) ~Sepal.Length + Sepal.Width, data=train_data, kernel="linear", cost=1, gamma = 1, scale=FALSE)
plot(linear_svm, train_data, title = "Linear SVM Classification Plot")
linear_preds = predict(linear_svm, train_data)
table(predict = linear_preds, truth = train_data$Species)

#fit a radial svm
radial_svm = svm(as.factor(Species) ~Sepal.Length + Sepal.Width, data=train_data, kernel="radial", cost=1, gamma = 1, scale=FALSE)
radial_preds = predict(radial_svm, train_data)
table(predict = radial_preds, truth = train_data$Species)
plot(radial_svm, train_data)

#Analysis ---------------------------------------------
# switching from a linear to a radial kernel slightly changes the boundaries.
# The boundaries are slightly curved, and this affects the predictions along
# the boundary between virginica and versicolor irises.
# Both models are perfect in predicting setosa, but the radial model predicts 
# more virginica flowers and the linear model predicts more versicolor flowers
#-------------------------------------------------------

#Question 2B --------------------------------------------
#divide the data into training and testing sets
data = subset(iris, select = c(Sepal.Length, Sepal.Width, Species))
cv = sample(nrow(data), nrow(data)*.5)
train = data[cv,]
test = data[-cv,]
#find the optimal values for cost and gamma
tune.out = tune(svm, Species ~., data = train, kernel="radial",
                ranges=list(cost=c(0.1,1,10,100),
                            gamma=c(0.5,1,2)))
best_model <- tune.out$best.model
print(paste("The optimal cost is", best_model$cost))
print(paste("The optimal gamma is", best_model$gamma))
preds = predict(best_model, new_data = test)
table(true=test$Species,pred=preds)
#Analysis-----------------------------------------------
# The optimal cost is 10, and the optimal gamma is 0.5.
# The number of correctly predicted setosa flowers is 10.
# the number of correctly predicted versicolor flowers is 6.
# the number of correctly predicted virginica flowers is 7.
# Virginica flowers are often predicted to be setosa
# Versicolor flowers are often predicted to be virginica flowers
#-------------------------------------------------------

#Question 2C----------------------------------------------
#create vectors to hold linear and radial errors
linear_errors = c()
radial_errors = c()


data = subset(iris, select = c(Sepal.Length, Sepal.Width, Species))
for (x in 1:10) {
  #divide the data into training and testing sets
  cv = sample(nrow(data), nrow(data)*.5)
  train = data[cv,]
  test = data[-cv,]
  
  #find the optimal gamma for the models
  best_radial = tune(svm, Species ~., data = train, kernel="radial", cost = 10, 
                  ranges=list(gamma=c(0.5,1,2)))$best.model
  best_linear = tune(svm, Species ~., data = train, kernel="linear", cost = 10, 
                     ranges=list(gamma=c(0.5,1,2)))$best.model
  
  #calculate the average misclassification rate for each model
  linear_preds = predict(best_linear, new_data = test)
  radial_preds = predict(best_radial, new_data = test)
  linear_errors  = append(linear_errors, mean(test$Species != linear_preds))
  radial_errors = append(radial_errors, mean(test$Species != radial_preds))
}

print(paste("the average misclassification rate for the linear svm is", mean(linear_errors)))
print(paste("the average misclassification rate for the radial svm is", mean(radial_errors)))

#Analysis -----------------------------------------------------
# the average misclassification rate for the linear svm is 0.686666666666667
# the average misclassification rate for the radial svm is 0.674666666666667
#-----------------------------------------------------------------

#Question 3A ---------------------------------------------------------
?Carseats
#divide the data into training and testing sets
set.seed(5)
cv = sample(nrow(Carseats), nrow(Carseats)*.5)
train = sample(1:nrow(Carseats), nrow(Carseats)/2)

sales_tree = tree(Sales ~ CompPrice + Income + Advertising + Population + Price, data = Carseats, subset = train)
plot(sales_tree)
text(sales_tree, pretty=0, cex=0.7)
summary(sales_tree)

cv_sales = cv.tree((sales_tree))
plot(cv_sales$size, cv_sales$dev, type="b")

#the minimum deviance is achieved with 9 nodes
prune_sales = prune.tree(sales_tree, best = 9)
plot(prune_sales)
text(prune_sales, pretty=0, cex=0.7)

#analysis ------------------------------------------------------------
#the minimum deviance is achieved with 9 nodes
# If the price is less than 89.5 and the population is below 253,500, then
# the tree predicts sales of 8, 673 unit sales. At downstream nodes,
# proceed to the left if the condition is met, and proceed to the right if the condition
# is not met. The predicted value is assigned when the process reaches a 
# terminal node.
# --------------------------------------------------------------------

#Question 3B ----------------------------------------------------------
test_y = Carseats[-train,"Sales"]

bag_sales7 = randomForest(Sales ~ CompPrice + Income + Advertising + Population + Price,
                         data = Carseats, subset = train, mtry=7)
bag_sales3 = randomForest(Sales ~ CompPrice + Income + Advertising + Population + Price,
                          data = Carseats, subset = train, mtry=3)
preds7 = predict(bag_sales7, newdata = Carseats[-train,])
preds3 = predict(bag_sales3, newdata = Carseats[-train,])
par(mfrow = c(1, 2))
plot(preds7, test_y)
title("mtry = 7")
plot(preds3, test_y)
title("mtry = 3")

mse7 = sum((test_y - preds7)^2)
mse3 = sum((test_y - preds3)^2)
print(paste("The MSE for the forest with mtry = 7 is", mse7))
print(paste("The MSE for the forest with mtry = 3 is", mse3))

importance(bag_sales7)
importance(bag_sales3)

#Analysis ------------------------------------------------------------
# The MSE for the forest starting with 7 variables is 1000.9322550228
# The MSE for the forest starting with 3 variables is 1020.10671171583
# The forest with mtry = 7 has a lower MSE
# for the random forest starting with 7 variables, Price is the most important variable,
# followed by ComPrice, then Income, Advertising, and Population.
# For the random forest starting with 3 variables, the order of the importance of the 
# variables is the same, but the magnitude of the IncNodePurity for the 
# variables are different.
# --------------------------------------------------------------------

#Question 3C ---------------------------------------------------------
boost_sales = gbm(Sales ~ CompPrice + Income + Advertising + Population + Price,
                  data=Carseats[train,], distribution="gaussian", n.trees=5000)
# I feel like this is what it means by interpret the summary and each predictor
plot(boost_sales, i="CompPrice")
plot(boost_sales, i="Income")
plot(boost_sales, i="Advertising")
plot(boost_sales, i="Population")
plot(boost_sales, i="Price")
summary(boost_sales)

boost_preds = predict(boost_sales, newdata = Carseats[-train,])
boost_mse = sum((boost_preds - test_y)^2)
print(paste("The MSE for the boosting model is", boost_mse))

#Analysis -------------------------------------------------------
# The MSE for the boosting model is 1354.58865286328. This is larger than
# the MSE for both random forests in part B.
# Price is the most important variable, accounting for 27.05% in the reduction
# of the loss function. The percentage of loss accounted for by the other
# variables is given in the rel.inf column of the summary





