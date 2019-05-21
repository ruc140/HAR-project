## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/ruc140/HAR-project/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

## Human Activities Recognition Project

This project is to construct a model based on Human Activities Recognition data, which can predict the correctness of movement (5 classes) based on a list of variables. The dataset contains about 20000 observations on 160 variables. I first removed the variables that are not movement measurements and those contain NA data. I then divide the data into 70% training and 30% validation. A random forest model is fitted based on training data and accuracy is measured based on validation.

### Getting and cleaning data
```{r}
data <- read.csv("Project/pml-training.csv")
# str(data)
# remove names and times that can't be used to construct model
HAR <- subset(data, select= -c(1:7))
dim(HAR)
# remove columns containing NA
max <- apply(HAR, 2, max)
nanumber <- which(is.na(max)==TRUE)
nanumber <- data.frame(nanumber)
HAR_noNA <- subset(HAR, select= -nanumber[,1])
dim(HAR_noNA)
```
### Diving training and validation datasets
```{r}
library(caret)
set.seed(12345)
inTrain <- createDataPartition(y=HAR_noNA$classe, p=0.7, list=FALSE)
training <- HAR_noNA[inTrain,]
validation <- HAR_noNA[-inTrain,]
dim(training)
dim(validation)
```
### Preprocess data
remove near zero variable
```{r}
nzv <- nearZeroVar(training, saveMetrics=TRUE)
sum(nzv$nzv)
training1 <- subset(training, select= c(nzv$nzv==FALSE))
dim(training1)
```

### Set up parallel computing and fit model
```{r}
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.8),
                           allowParallel = TRUE)                           
modfit_rf <- train(classe ~., method="rf",data=training1, preProcess = c("center", "scale", "pca"), trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
modfit_rf$finalModel
```
### Out sample error
```{r}
pred_validation <- predict(modfit_rf, newdata=validation)
accuracy <- sum(pred_validation == validation$classe) / length(pred_validation)
error <- 1- accuracy
```
The out of sample error is `r error`.

