# Please generate the neccessary data files with 
# /path/to/caffe/tools/extra/parse_log.sh before plotting.
# Example usage: 
#     ./parse_log.sh mnist.log
# Now you have mnist.log.train and mnist.log.test.
#     gnuplot mnist.gnuplot

# The fields present in the data files that are usually proper to plot along
# the y axis are test accuracy, test loss, training loss, and learning rate.
# Those should plot along the x axis are training iterations and seconds.
# Possible combinations:
# 1. Test accuracy (test score 0) vs. training iterations / time;
# 2. Test loss (test score 1) time;
# 3. Training loss vs. training iterations / time;
# 4. Learning rate vs. training iterations / time;
# A rarer one: Training time vs. iterations.

reset
#set terminal dumb
set style data lines
set key right center

file(test_or_train) = sprintf("%s.%s", filename, test_or_train)
ucf_101_title = "Learning on six classes of UCF 101"

###### Fields in the training data
###### Iters Seconds TrainingLoss LearningRate

# Training loss vs. training iterations
set terminal png
set output "it_vs_train-loss.png"
set title "Training loss vs. training iterations"
set xlabel "Training iterations"
set ylabel "Training loss"
plot file("train") using 1:3 title "loss"

# Training loss vs. training time
#set terminal png
#set output "time_vs_train-loss.png"
#set title "Training time vs. training loss"
#set xlabel "Training time"
#set ylabel "Training loss"
#plot file("train") using 2:3 title "loss"

# Learning rate vs. training iterations;
set terminal png
set output "it_vs_lr.png"
set xlabel "Training iterations"
set ylabel "Learning rate"
plot file("train") using 1:4 title "learning rate"

###### Fields in the test data
###### Iters Seconds TestAccuracy TestLoss

# Test loss vs. training iterations
set terminal png
set output "it_vs_test-acc.png"
set title "Training iterations vs. test accuracy"
set xlabel "Training iterations"
set ylabel "Test accuracy"
plot file("test") using 1:3 title "accuracy"
