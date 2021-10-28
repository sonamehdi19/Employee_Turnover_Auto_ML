# Importing libraries & dataset ----
library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)

path <- dirname(getSourceEditorContext()$path)
setwd(path)
raw <- fread("HR_turnover.csv")
raw %>% skim()
#if the worker left the job, it is coded as 1 in left col otherwise 0


#for automl model, only target needs to be factorized
raw$left<-raw$left %>% as.factor()   #factorizing the target
raw$left %>% table() %>% prop.table()  #target distribution
#the target is equally distributed 

#column names formatting
names(raw) <- names(raw) %>% 
  str_replace_all(" ","_") %>%
  str_replace_all("-","_") %>%
  str_replace_all("\\(","") %>% 
  str_replace_all("\\)","") %>% 
  str_replace_all("\\'","")

# --------------------------------- Modeling ----------------------------------
h2o.init()
h2o_data <- raw %>% as.h2o()

#1. Splitting data into train and test sets using seed=123 within h2o framework----
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]
target <- 'left'
features <- raw %>% select(-left) %>% names()

#2. Building classification model with h2o.automl()----
model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  nfolds = 10, seed = 123,
  max_runtime_secs=300)

model@leaderboard %>% as.data.frame()
model@leader 

# Predicting the Test set results
pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()

# 3. Finding threshold by max f1 score----
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1') -> threshold

#Model evaluation

#4. Extracting confusion matrix in tibble format----
model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))

# Area Under Curve (AUC)
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.metric() %>% 
  select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(nrow(.),min=0.001,max=1)) %>% 
  mutate(fpr_r=tpr_r) %>% 
  arrange(tpr_r,fpr_r) -> deep_metrics


#5. Calculating AUC score both for train and test sets----

#AUC for train
model@leader %>% 
  h2o.performance(train) %>% 
  h2o.auc() %>% round(2) -> auc_train

#AUC for test
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc

highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr,x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r,x=fpr_r), color='red', name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0,yAxis=0,x=0.3,y=0.6),
      text = glue('AUC = {enexpr(auc)}'))
  ) %>%
  hc_title(text = "ROC Curve") %>% 
  hc_subtitle(text = "Model is performing much better than random guessing") 

#6. Checking overfitting ----
model@leader %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)
# as the train and test scores are near to each other, there is no overfitting


