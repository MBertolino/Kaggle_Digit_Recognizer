# Random search improvement of neural network to predict handwritten digits

# Load Libraries
library(h2o)

# Initialize the H2O server and import the MNIST training/testing datasets.
h2oServer <- h2o.init(nthreads = -1)
test <- read.csv("/home/mattias/R/Kaggle/Digit Recognizer/test.csv", stringsAsFactors = FALSE)
train <- read.csv("/home/mattias/R/Kaggle/Digit Recognizer/train.csv", stringsAsFactors = FALSE)

## Preprocessing
# Delete zero-variance columns
variances <- apply(train[,-1], 2, var)
useless <- names(train[,-1][variances == 0][1,])
train <- train[, -which(names(train) %in% useless)]
test <- test[, -which(names(test) %in% useless)]

train[,1] <- as.factor(train[,1]) # Convert digit to factor for classification in label-column

# Param
n_models <- 10 # If random search
select_search <- 0 # 0 is normal, 1 is random search


## Train model
if (select_search > 0.5) {
  ############################ RANDOM SEARCH ##################################
  # Subset data in train, val and test
  index <- sample(1:nrow(train), round(0.75*nrow(train)))
  train_h2o = as.h2o(train[index,])
  val_h2o = as.h2o(train[-index,]) # Used to choose best model if random search
  test_h2o = as.h2o(test)
  
  # Train several random models
  models <- c()
  for (i in 1:n_models) {
    rand_activation <- c("TanhWithDropout", "RectifierWithDropout")[sample(1:2, 1)]
    rand_numlayers <- sample(2:5,1)
    rand_hidden <- c(sample(1000:3000, rand_numlayers,T))
    rand_l1 <- runif(1, 0, 1e-3)
    rand_l2 <- runif(1, 0, 1e-3)
    rand_dropout <- c(runif(rand_numlayers, 0, 0.5))
    rand_input_dropout <- runif(1, 0, 0.5)
    temp_model <- h2o.deeplearning(x = 2:785, y = 1, training_frame = train_h2o, validation_frame = val_h2o, epochs = 15,
                                   activation = rand_activation, hidden = rand_hidden, l1 = rand_l1, l2 = rand_l2,
                                   input_dropout_ratio = rand_input_dropout, hidden_dropout_ratios = rand_dropout)
    models <- c(models, temp_model)
  }
  
  # Find the best model of the randomized with greedy algorithm
  best_model <- models[[1]]
  for (j in 2:n_models) {
    if (h2o.mse(models[[j]]) < h2o.mse(best_model)) {
      best_model <- models[[j]]
    }
  }
  ###############################################################################
} else {
  ################################## NORMAL #####################################
  # Subset data in train and test
  train_h2o = as.h2o(train)
  test_h2o = as.h2o(test)
  
  best_model = h2o.deeplearning(x = 2:785,
                                y = 1,
                                training_frame = train_h2o,
                                activation = "RectifierWithDropout",
                                input_dropout_ratio = 0.2, # % input dropout
                                l1 = 1e-5,
                                max_w2 = 10,
                                balance_classes = TRUE,
                                hidden = c(200, 500),
                                momentum_stable = 0.99,
                                train_samples_per_iteration = -1, 
                                nesterov_accelerated_gradient = T, # Speed-up
                                epochs = 100)
  ###############################################################################
}

## Predict test data
y_pred <- h2o.predict(best_model, test_h2o)

# Convert H2O format into data frame and save as csv
submit = as.data.frame(y_pred)
submit = data.frame(ImageId = seq(1, length(submit$predict)), Label = submit$predict)
write.csv(submit, file = "/home/mattias/R/Kaggle/Digit Recognizer/Results/digitr_mbertolino_9.csv", row.names = FALSE)
