# Installing package
install.packages("caret", dependencies = TRUE)
install.packages("caTools")
install.packages("randomForest")  

# Loading package
library(caret)
library(caTools)
library(randomForest)


data = read.csv('C:/Users/DELL/Documents/heart.csv',sep=",", header= TRUE)
head(data)

data$target = as.factor(data$target)


# Split the data into training and testing set
data_set_size = floor(nrow(data)*0.80)
index = sample(1:nrow(data), size = data_set_size)
training = data[index, ]
testing = data[-index, ]


# Random Forest
rf = randomForest(target ~ age+ sex+ cp+ trestbps+ chol + fbs+ restecg+ thalach+ exang+  oldpeak+ slope+ ca+ thal,data = training ,ntree= 39,maxDepth=9)
rf_predicted_train = predict(rf,newdata = training)
rf_train_conf_matrix = table(rf_predicted_train, training$target)
rf_train_conf_matrix
rf_acc_score_train = sum(diag(rf_train_conf_matrix))/sum(rf_train_conf_matrix)
rf_acc_score_train
rf_predicted_test = predict(rf,newdata = testing)
rf_test_conf_matrix = table(rf_predicted_test, testing$target)
rf_test_conf_matrix
rf_acc_score_test = sum(diag(rf_test_conf_matrix))/sum(rf_test_conf_matrix)
rf_acc_score_test


