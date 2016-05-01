
# coding: utf-8

# In[1]:

import training

input_X = [[1, 2, 3, 4],[2, 3, 2, 1], [2, 1, 10, 8]]
Y_test = [1, 0, 1]

#The perdiction function: prediction(method, input_X, param)
#The first parameter: the method name. It could be 'linear_regression', 'ridge_regression', 
#'knn', 'linearsvm', 'polysvm', 'rbfsvm', 'sigsvm', 'bayesgaussin', 'logistic', 'lda', 'lasso',
#'decisionTree', 'randomForest', 'adaBoost', 'qda'
#the second parameter: the input of X(4 dimension features matrix)
#The third parameter: the parameter for the method
#Return: the prediction label 1 or 0
print('rbfsvm')
hat_Y = training.prediction('rbfsvm', input_X, 0.25)
print hat_Y

print('knn')
hat_Y = training.prediction('knn', input_X, 5)
print hat_Y

print('logistic')
hat_Y = training.prediction('logistic', input_X)
print hat_Y

#The score function: prediction_score(method, input_X, Y, param)
#The first parameter: the method name. 
#the second parameter: the input of X(4 dimension features matrix)
#The third parameter is true value of Y
#The fourth parameter: the parameter for the method
#Return: the correct prediction rate
score = training.prediction_score('rbfsvm', input_X, Y_test, 0.25)

#The test error function: prediction_testError(method, input_X, Y, param)
#The first parameter: the method name. 
#the second parameter: the input of X(4 dimension features matrix)
#The third parameter is true value of Y
#The fourth parameter: the parameter for the method
#Return: the test error
test_error = training.prediction_testError('rbfsvm', input_X, Y_test, 0.25)
print score, test_error



# In[ ]:




# In[ ]:



