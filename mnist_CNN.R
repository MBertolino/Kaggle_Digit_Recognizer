# Code used in the laboration "Image classification with Deep Learning" 
# in the course "Statistical Machine Learning"
# given at Uppsala University, spring 2017.
#
# This code smaple trains and evaluates a singe layer neural network 
# for classifying hand-written digits using thr MNIST data set

rm(list = ls())
library(tensorflow)

# The placeholder for the input:  28 x 28 = 784 pixels grayscale image. 
X <- tf$placeholder(dtype = tf$float32, shape = shape(NULL, 784L))
X_im <- tf$reshape(X, shape(-1L, 28L, 28L, 1L))
Y <- tf$placeholder(dtype = tf$float32, shape = shape(NULL, 10L))

# Layer params
K <- 6
L <- 12
M <- 24
N <- 200

# Weights
W1 <- tf$Variable(tf$truncated_normal(shape(6L, 6L, 1L, K), stddev = 0.1))
W2 <- tf$Variable(tf$truncated_normal(shape(5L, 5L, K, L), stddev = 0.1))
W3 <- tf$Variable(tf$truncated_normal(shape(4L, 4L, L, M), stddev = 0.1))
W4 <- tf$Variable(tf$truncated_normal(shape(7L*7L*M, N), stddev = 0.1))
W5 <- tf$Variable(tf$truncated_normal(shape(N, 10L), stddev = 0.1))

# Bias
b1 <- tf$Variable(tf$ones(shape(K))/10)
b2 <- tf$Variable(tf$ones(shape(L))/10)
b3 <- tf$Variable(tf$ones(shape(M))/10)
b4 <- tf$Variable(tf$ones(shape(N))/10)
b5 <- tf$Variable(tf$ones(shape(10L))/10)

# Dropout: feed in 1 when testing, 0.75 when training
pkeep = tf$placeholder(tf$float32)

# Model
Z1 <- tf$nn$relu(tf$nn$conv2d(X_im, W1, strides = c(1L, 1L, 1L, 1L), padding = 'SAME') + b1)
Z2 <- tf$nn$relu(tf$nn$conv2d(Z1, W2, strides = c(1L, 2L, 2L, 1L), padding = 'SAME') + b2)
Z3 <- tf$nn$relu(tf$nn$conv2d(Z2, W3, strides = c(1L, 2L, 2L, 1L), padding = 'SAME') + b3)
Z3flat <- tf$reshape(Z3, shape(-1L, 7*7*M))
Z4 <- tf$nn$relu(tf$matmul(Z3flat, W4) + b4)
Z4d <- tf$nn$dropout(Z4, pkeep)
T <- tf$matmul(Z4d, W5) + b5
Yhat <- tf$nn$softmax(T)

# The loss function: Compute the cross-entropy
cross_entropy <- tf$nn$softmax_cross_entropy_with_logits(logits = T, labels = Y)
cross_entropy <- tf$reduce_mean(cross_entropy)*100 # Normalize with batchsize

# Define the training
gamma_min <- 0.0001
gamma_max <- 0.003
gamma <- tf$placeholder(dtype = tf$float32) # The learning rate
train_step <- tf$train$AdamOptimizer(gamma)$minimize(cross_entropy)

# Accuracy of the trained model. 0 is worst and 1 is the best.
prediction <- tf$argmax(Yhat, 1L) # In each row of Yhat, determine the index of the column with highest probability
true <- tf$argmax(Y, 1L) 
correct_prediction <- tf$equal(true, prediction) # Compare with true labels. TRUE if correct, FALSE if not.
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Create session and initialize  variables
sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Load mnist data
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Initialize the test and training error resutls
test.accuracy <- c()
test.cross_entropy <- c()
test.iter <- c()
train.accuracy <- c()
train.cross_entropy <- c()
train.iter <- c()

# The main training loop
for (i in 1:10000) {
  
  # Get 100 training data points (called a mini-batch)
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  
  # Do one gradient step
  new_gamma <- gamma_min + (gamma_max - gamma_min)*exp(-i/2000)
  sess$run(train_step,
           feed_dict = dict(X = batch_xs, Y = batch_ys, 
                            gamma = new_gamma, pkeep = 0.75))
  
  # Evaluate the performance on training data
  # These lines are for evaluating the performance. Don't bother too much about them!
  if (i%%10 == 0) {
    train.tmp <- sess$run(c(accuracy, cross_entropy),
                          feed_dict = dict(X = batch_xs, Y = batch_ys, pkeep = 1))
    train.accuracy <- c(train.accuracy, train.tmp[1])
    train.cross_entropy <- c(train.cross_entropy, train.tmp[2])
    train.iter <- c(train.iter, i)
  }
  
  # Evaluate the performance on training data at every 100th itertation
  # These lines are for evaluating the performance. Don't bother too much about them!
  if (i%%100 == 0) {
    # Evaluate on test data
    test.tmp <- sess$run(c(accuracy, cross_entropy),
                         feed_dict = dict(X = mnist$test$images, Y = mnist$test$labels, pkeep = 1))
    test.accuracy <- c(test.accuracy, test.tmp[1])
    test.cross_entropy <- c(test.cross_entropy, test.tmp[2])
    test.iter <- c(test.iter, i)
    
    # Also print iteration number
    cat(sprintf("Iteration: %i, Test acc: %f \n", i, test.tmp[1]))
  }
}

# Plot 100 test images
batch.test <- mnist$test$next_batch(100L)
batch.test.xs <- batch.test[[1]]
batch.test.ys <- batch.test[[2]]

testbatch.pred <- sess$run(prediction, feed_dict = dict(X = batch.test.xs, pkeep = 1))
testbatch.true <- sess$run(true, feed_dict = dict(Y = batch.test.ys))

# Sort the data such that we have the missclassifications first
result <- sort(testbatch.true == testbatch.pred, index.return = TRUE)

# Do the plot
oldparams <- par(mfrow = c(10, 10), mai = c(0, 0, 0, 0))
for(i in result$ix){
  im = as.matrix(batch.test.xs[i, 1:784])
  dim(im) = c(28, 28)
  image(im[, nrow(im):1], axes = FALSE, col = gray((255:0)/255))
  if (testbatch.pred[i] == testbatch.true[i]){
    text(0.2, 0, testbatch.pred[i], cex = 3, col = 1, pos = c(3, 4))
  } else {
    text(0.2, 0, testbatch.pred[i], cex = 3, col = 2, pos = c(3, 4))
  }
}
par(oldparams) # Reset to old plot-par

# Plot the training/test cross-entropy
plot(x = train.iter, y = train.cross_entropy, col = "blue", type = "l", ylim = c(0.0, 60), 
     xlab="Iteration", ylab = "Cross-entropy", main = "Cross-entropy")
lines(x = test.iter, y = test.cross_entropy, col = "red", lwd = 3)
legend(x = "topright", legend = c("Training data","Test data"), col = c("blue", "red"), lwd = c(1, 3))

# Plot the training accuracy
plot(x = train.iter, y = train.accuracy, col = "blue", type = "l", 
     ylim = c(0.85, 1), xlab = "Iteration", ylab = "Prediction accuracy", 
     main="Prediction accuracy")
lines(x = test.iter, y = test.accuracy, col = "red", lwd = 3)
legend(x = "bottomright", legend = c("Training data", "Test data"),
       col = c("blue", "red"), lwd = c(1, 3))

# Test trained model
cat('\n',"Prediction accuracy test data", tail(test.accuracy, n = 1)[[1]])

# Close the session
sess$close()

