This project is to construct a model based on Human Activities Recognition data linked below, which can predict the correctness of movement (5 classes) based on a list of variables. The dataset contains about 20000 observations on 160 variables. I first removed the variables that are not movement measurements and those contain NA data. I then divide the data into 70% training and 30% validation. A random forest model is fitted based on training data and accuracy is measured based on validation.

http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har#wle_paper_section
