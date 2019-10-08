# Naive Bayes Classifier
Implements a Python executable file called naive_bayes.py that learns a naive Bayes classifier for a classification problem, given some training data and some additional options.

## Arguments
1. The first argument, \<training_file>, is the path name of the training file, where the training data is stored. The path name can specify any file stored on the local computer.

2. The second argument, \<test_file>, is the path name of the test file, where the test data is stored. The path name can specify any file stored on the local computer.

## Notes
* In certain cases, it is possible that value computed for the standard deviation is equal to zero. Your code should make sure that the variance of the Gaussian is NEVER smaller than 0.0001. Since the variance is the square of the standard deviation, this means that the standard deviation should never be smaller than sqrt(0.0001) = 0.01. Any time the value for the standard deviation is computed to be smaller than 0.01, your code should replace that value with 0.01.
