setwd("C:/Users/roldy/Desktop")
covid = read.csv("COVID_data.csv")
raw = covid[,4:218] #Data without County Names
library(mlbench)
library(caret)

control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
model = train(deaths~., data = raw, method = "bridge", trControl = control) #Bayesian Ridge Regression to determine variable importance
importance = varImp(model, scale = F)
print(importance)
important = raw[c("Density.per.square.mile.of.land.area...Housing.units", "Density.per.square.mile.of.land.area...Population", "HBAC_FEMALE", "HBA_MALE", "HBAC_MALE", "HBA_FEMALE", "GQ_ESTIMATES_2018", "Bachelor.s.degree.or.higher.2014.18","Oncology..Cancer..specialists..2019.","Total.physician.assistants..2019.","ICU.Beds","Endocrinology.Diabetes.and.Metabolism.specialists..2019.", "Psychiatry.specialists..2019.","Internal.Medicine.Primary.Care..2019.", "Cardiovascular.Disease..AAMC.", "Cardiology.specialists..2019.", "Psychiatry..AAMC.", "Hematology...Oncology..AAMC.", "HIAC_FEMALE", "Total.Specialist.Physicians..2019.","deaths")]
View(important) #dataset with only the most important features


library(randomForest)
library(caTools)
set.seed(4052)

#Use repeated averaged cv to get test mse for the full dataset, the 20 most important features set, and just the two most important features

#Full feature set
fullmse = replicate(100, 0)
for(i in 1:100){
  train = sample(1:nrow(raw), 2357)   #Use ~75% of the data for training
  rf = randomForest(deaths~., data = raw, subset = train, ntree = 500) #fit a rf on the training data
  pred=predict(rf, raw[-train,])
  fullmse[i] = with(raw[-train,], mean((deaths-pred)^2))  #calculate mse
}

#20 most important features
redmse = replicate(100, 0)
for(i in 1:100){
  train = sample(1:nrow(important), 2357)
  rf = randomForest(deaths~., data = important, subset = train, ntree = 500)
  pred=predict(rf, important[-train,])
  redmse[i] = with(important[-train,], mean((deaths-pred)^2))
}

#2 most important features
densitymse = replicate(100,0)
for(i in 1:100){
  train = sample(1:nrow(important), 2357)
  rf = randomForest(deaths~Density.per.square.mile.of.land.area...Housing.units+Density.per.square.mile.of.land.area...Population, data = important, subset = train, ntree = 500)
  pred=predict(rf, important[-train,])
  densitymse[i] = with(important[-train,], mean((deaths-pred)^2))
}

#Linear regression with two most important features
lmmse = replicate(100,0)
for(i in 1:100){
  train = sample(1:nrow(important), 2357)
  lmmod = lm(deaths~Density.per.square.mile.of.land.area...Housing.units+Density.per.square.mile.of.land.area...Population, data = important, subset = train)
  pred=predict(lmmod, important[-train,])
  lmmse[i] = with(important[-train,], mean((deaths-pred)^2))
}



c(mean(fullmse), mean(redmse), mean(densitymse), mean(lmmse))