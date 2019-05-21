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
1. remove near zero variable
```{r}
nzv <- nearZeroVar(training, saveMetrics=TRUE)
sum(nzv$nzv)
training1 <- subset(training, select= c(nzv$nzv==FALSE))
dim(training1)
```
2. Preprocess with PCA
```{r}
preProc1 <-preProcess(training1[,-53], method = c("center", "scale"))
trainPC1 <-predict(preProc1, training1[,-53])
preProc2 <- preProcess(trainPC1, method="pca", thresh = 0.8)
trainPC2 <- predict(preProc2, trainPC1)
preProc3 <- preProcess(trainPC1, method="pca", thresh = 0.85)
trainPC3 <- predict(preProc3, trainPC1)
preProc4 <- preProcess(trainPC1, method="pca", thresh = 0.9)
trainPC4 <- predict(preProc4, trainPC1)
```

### Set up parallel computing and fit model
```{r}
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)                           
modfit_rf2 <- train(training1$classe ~., method="rf",data=trainPC2, trControl = fitControl)
modfit_rf3 <- train(training1$classe ~., method="rf",data=trainPC3, trControl = fitControl)
modfit_rf4 <- train(training1$classe ~., method="rf",data=trainPC4, trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
modfit_rf$finalModel
```
### In and out sample error
pred_validation <- predict(modfit_rf, data=validation)
accuracy <- sum(pred_validation == validation$classe) / length(pred_validation)

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

