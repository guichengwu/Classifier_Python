
# coding: utf-8

# In[46]:

import numpy as np
from sklearn import svm, neighbors, naive_bayes
from sklearn import linear_model, neighbors, preprocessing, cross_validation
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel, laplacian_kernel 
from sklearn.metrics.pairwise import safe_sparse_dot,check_pairwise_arrays
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA

########################### Methods #############################################
def linear_regression(X_train, Y_train):
    linear_regr = linear_model.LinearRegression()
    #train the model with train data
    linear_regr.fit(X_train, Y_train)
    return linear_regr

def ridge_regression(X_train, Y_train, alpha=10):
    ridge = linear_model.Ridge(alpha)
    ridge.fit(X_train, Y_train)
    return ridge

def kNearestNeighbors(X_train, Y_train, k=5):
    knn = neighbors.KNeighborsRegressor(k)
    knn.fit(X_train, Y_train)
    return knn

#SVM kernel_type: linear, polynomial, rbf, sigmoid, p
def linear_svm(X_train, Y_train):
    linearsvm = svm.SVC(kernel='linear')
    linearsvm.fit(X_train, Y_train)
    return linearsvm

def polynomial_svm(X_train, Y_train, degree=3):
    polysvm = svm.SVC(kernel='poly', degree=degree)
    polysvm.fit(X_train, Y_train)
    return polysvm

def rbf_svm(X_train, Y_train, gamma=None):
    rbfsvm = svm.SVC(kernel='rbf', gamma=gamma)
    rbfsvm.fit(X_train, Y_train)
    return rbfsvm

def sigmoid_svm(X_train, Y_train, gamma=None):
    sigsvm = svm.SVC(kernel='sigmoid', gamma=gamma)
    sigsvm.fit(X_train, Y_train)
    return sigsvm

def bayes_gaussian(X_train, Y_train):
    gaussianbayes = naive_bayes.GaussianNB()
    gaussianbayes.fit(X_train, Y_train)
    return gaussianbayes

def logistic_regression(X_train, Y_train):
    logistic = linear_model.LogisticRegression()
    logistic.fit(X_train, Y_train)
    return logistic

def linearDiscrminat(X_train, Y_train):
    linearDA = LDA()
    linearDA.fit(X_train, Y_train)
    return linearDA

def lasso_regression(X_train, Y_train):
    lasso_regr = linear_model.Lasso(alpha=1)
    lasso_regr.fit(X_train, Y_train)
    return lasso_regr

def decision_tree(X_train, Y_train):
    decisionTree = tree.DecisionTreeClassifier()
    decisionTree.fit(X_train, Y_train)
    return decisionTree

def random_forest(X_train, Y_train):
    randomForest = ensemble.RandomForestClassifier()
    randomForest.fit(X_train, Y_train)
    return randomForest

def adaBoost(X_train, Y_train):
    ada = ensemble.AdaBoostClassifier(n_estimators=200)
    ada.fit(X_train, Y_train)
    return ada

def QuadDA(X_train, Y_train):
    qda = QDA()
    qda.fit(X_train, Y_train)
    return qda

def decision_rule(hat_Y):
    for i in np.arange(hat_Y.shape[0]):
        if hat_Y[i] > 0.5:
            hat_Y[i] = 1
        else:
            hat_Y[i] = 0
    return hat_Y

def predict_linear_regression(X_train, Y_train, X_test):
    hat_Y = linear_regression(X_train, Y_train).predict(X_test)
    return decision_rule(hat_Y)

def predict_ridge_regression(X_train, Y_train, X_test):
    hat_Y = ridge_regression(X_train, Y_train).predict(X_test)
    return decision_rule(hat_Y)

def predict_knn(X_train, Y_train, X_test, k=5):
    hat_Y = kNearestNeighbors(X_train, Y_train, k).predict(X_test)
    return decision_rule(hat_Y)

def predict_linearsvm(X_train, Y_train, X_test):
    hat_Y = linear_svm(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_polynomialsvm(X_train, Y_train, X_test, degree=3):
    hat_Y = polynomial_svm(X_train, Y_train, degree).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_rbfsvm(X_train, Y_train, X_test, gamma=None):
    hat_Y = rbf_svm(X_train, Y_train, gamma).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_sigmoidsvm(X_train, Y_train, X_test, gamma=None):
    hat_Y = sigmoid_svm(X_train, Y_train, gamma).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y) 

def predict_bayesgaussin(X_train, Y_train, X_test):
    hat_Y = bayes_gaussian(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)  

def predict_logistic(X_train, Y_train, X_test):
    hat_Y = logistic_regression(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_LDA(X_train, Y_train, X_test):
    hat_Y = linearDiscrminat(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_lasso(X_train, Y_train, X_test):
    hat_Y = lasso_regression(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_decisionTree(X_train, Y_train, X_test):
    hat_Y = decision_tree(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_randomForest(X_train, Y_train, X_test):
    hat_Y = random_forest(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y)

def predict_adaBoost(X_train, Y_train, X_test):
    hat_Y = adaBoost(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y) 

def predict_qda(X_train, Y_train, X_test):
    hat_Y = QuadDA(X_train, Y_train).predict(X_test)
    hat_Y = hat_Y.reshape(hat_Y.shape[0], 1)
    return decision_rule(hat_Y) 

def method_selection(method, X_train, Y_train, X_test, param=None):
    if method == 'linear_regression':
        hat_Y = predict_linear_regression(X_train, Y_train, X_test)
    elif method == 'ridge_regression':
        hat_Y = predict_ridge_regression(X_train, Y_train, X_test)
    elif method =='knn':
        hat_Y = predict_knn(X_train, Y_train, X_test, param)
    elif method =='linearsvm':
        hat_Y = predict_linearsvm(X_train, Y_train, X_test)
    elif method =='polysvm':
        hat_Y = predict_polynomialsvm(X_train, Y_train, X_test, param)
    elif method =='rbfsvm':
        hat_Y = predict_rbfsvm(X_train, Y_train, X_test, param)
    elif method =='sigsvm':
        hat_Y = predict_sigmoidsvm(X_train, Y_train, X_test, param) 
    elif method =='bayesgaussin':
        hat_Y = predict_bayesgaussin(X_train, Y_train, X_test)
    elif method =='logistic':
        hat_Y = predict_logistic(X_train, Y_train, X_test)
    elif method =='lda':
        hat_Y = predict_LDA(X_train, Y_train, X_test)    
    elif method =='lasso':
        hat_Y = predict_lasso(X_train, Y_train, X_test)  
    elif method =='decisionTree':
        hat_Y = predict_decisionTree(X_train, Y_train, X_test)  
    elif method =='randomForest':
        hat_Y = predict_randomForest(X_train, Y_train, X_test)
    elif method =='adaBoost':
        hat_Y = predict_adaBoost(X_train, Y_train, X_test)
    elif method =='qda':
        hat_Y = predict_qda(X_train, Y_train, X_test)
        
    return hat_Y

def prediction(method, input_X, param=None):
    filename = '/Users/guichengwu/Desktop/208_mid term/exam.dat'
    data = np.loadtxt(filename, dtype='str')

    for i in range(data.shape[0]):
        for j in range(1,data.shape[1]):
            data[i][j] = data[i][j][2:]

    data_matrix = np.matrix(data).astype(np.float)
    X = data_matrix[:, 1:5]
    Y = data_matrix[:, 0]
    nrow = X.shape[0]
    X = np.vstack((X, input_X))
    
    X = preprocessing.scale(X)
    X = laplacian_kernel(X)

    input_X = X[nrow:X.shape[0], :]
    
    X_train = X[0:nrow, :]
    Y_train = Y
    
    return method_selection(method, X_train, Y_train, input_X, param)

def prediction_score(method, X, Y_test, param=None):
    hat_Y = prediction(method, X, param)
    correctNum = 0
    i = 0
    for y in Y_test:
        if y == hat_Y[i]: 
            correctNum += 1
        i += 1
    correctRate = correctNum / (1.*len(Y_test))   
    return correctRate

def prediction_testError(method, X, Y_test, param=None):
    hat_Y = prediction(method, X, param)
    n = len(Y_test)
    return np.sum(np.asarray((hat_Y-Y_test))**2) / (1.0*n)

def correct_rate(method, X_train, Y_train, X_test, Y_test, param=None):
    hat_Y = method_selection(method, X_train, Y_train, X_test, param=param)
    correctNum = 0
    i = 0
    for y in Y_test:
        if y == hat_Y[i]: 
            correctNum += 1
        i += 1
    correctRate = correctNum / (1.*len(Y_test))   
    return correctRate

def test_error(method, X_train, Y_train, X_test, Y_test, param=None):
    hat_Y = method_selection(method, X_train, Y_train, X_test, param=param)
        
    n = len(Y_test)
    return np.sum(np.asarray((hat_Y-Y_test))**2) / (1.0*n)
########################### Methods ######################################################

###########################Data processing################################################
def dataPreProcess():
    filename = '/Users/guichengwu/Desktop/208_mid term/exam.dat'

    data = np.loadtxt(filename, dtype='str')

    for i in range(data.shape[0]):
        for j in range(1,data.shape[1]):
            data[i][j] = data[i][j][2:]

    data_matrix = np.matrix(data).astype(np.float)
    X = data_matrix[:, 1:5]
    Y = data_matrix[:, 0]
    X = preprocessing.scale(X)
    X = laplacian_kernel(X)
    #pca = decomposition.PCA(n_components=3)
    #pca.fit(X)
    #X = pca.transform(X)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    X, Y, test_size =0.2)

###########################Data processing################################################

########################### Test Result###################################################
def test_all_methods():
    print('linear regression')
    print correct_rate('linear_regression', X_train, Y_train, X_test, Y_test)
    print test_error('linear_regression', X_train, Y_train, X_test, Y_test)

    print('ridge regression')
    print correct_rate('ridge_regression', X_train, Y_train, X_test, Y_test)
    print test_error('ridge_regression', X_train, Y_train, X_test, Y_test)

    print('knn')
    print correct_rate('knn', X_train, Y_train, X_test, Y_test, 9)
    print test_error('knn', X_train, Y_train, X_test, Y_test, 9)

    print('linearsvm')
    print correct_rate('linearsvm', X_train, Y_train, X_test, Y_test)
    print test_error('linearsvm', X_train, Y_train, X_test, Y_test)

    print('polysvm')
    print correct_rate('polysvm', X_train, Y_train, X_test, Y_test, 3)
    print test_error('polysvm', X_train, Y_train, X_test, Y_test, 3)

    print('rbfsvm')
    print correct_rate('rbfsvm', X_train, Y_train, X_test, Y_test, 0.25)
    print test_error('rbfsvm', X_train, Y_train, X_test, Y_test, 0.25)

    print('sigsvm')
    print correct_rate('sigsvm', X_train, Y_train, X_test, Y_test, 0.1)
    print test_error('sigsvm', X_train, Y_train, X_test, Y_test, 0.1)

    print('bayesgaussin')
    print correct_rate('bayesgaussin', X_train, Y_train, X_test, Y_test)
    print test_error('bayesgaussin', X_train, Y_train, X_test, Y_test)

    print('logistic')
    print correct_rate('logistic', X_train, Y_train, X_test, Y_test)
    print test_error('logistic', X_train, Y_train, X_test, Y_test)

    print('lda')
    print correct_rate('lda', X_train, Y_train, X_test, Y_test)
    print test_error('lda', X_train, Y_train, X_test, Y_test)

    print('lasso')
    print correct_rate('lasso', X_train, Y_train, X_test, Y_test)
    print test_error('lasso', X_train, Y_train, X_test, Y_test)

    print('decisionTree')
    print correct_rate('decisionTree', X_train, Y_train, X_test, Y_test)
    print test_error('decisionTree', X_train, Y_train, X_test, Y_test)

    print('randomForest')
    print correct_rate('randomForest', X_train, Y_train, X_test, Y_test)
    print test_error('randomForest', X_train, Y_train, X_test, Y_test)

    print('adaBoost')
    print correct_rate('adaBoost', X_train, Y_train, X_test, Y_test)
    print test_error('adaBoost', X_train, Y_train, X_test, Y_test)

    print('qda')
    print correct_rate('qda', X_train, Y_train, X_test, Y_test)
    print test_error('qda', X_train, Y_train, X_test, Y_test)
    
########################### Test Result###################################################

########################### Draw preprocessing garphs#####################################
def preprocessGraph():
    filename = '/Users/guichengwu/Desktop/208_mid term/exam.dat'

    data = np.loadtxt(filename, dtype='str')

    for i in range(data.shape[0]):
        for j in range(1,data.shape[1]):
            data[i][j] = data[i][j][2:]

    data_matrix = np.matrix(data).astype(np.float)
    X = data_matrix[:, 1:5]
    Y = np.asarray(data_matrix[:, 0])
    X = preprocessing.scale(X)
    X = laplacian_kernel(X)
    #X = polynomial_kernel(X)
    #X = laplacian_kernel(X)
    #X = rbf_kernel(X)
    #X = sigmoid_kernel(X)

    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    data_fig1 = plt.figure(1, figsize=(8, 6))
    plt.clf()
    #Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Projection Vector 1')
    plt.ylabel('Projection Vector 2')
    plt.show()
    data_fig1.savefig('/Users/guichengwu/Desktop/208_mid term/data_2d.png')

    data_fig2 = plt.figure(2)
    ax2 = data_fig2.add_subplot(111, projection='3d')
    ax2.scatter(np.asarray(X[:,0]), np.asarray(X[:,1]), np.asarray(X[:, 2]), c=Y, cmap=plt.cm.Paired)
    plt.show()
    data_fig2.savefig('/Users/guichengwu/Desktop/208_mid term/data_3d.png')
########################### Draw preprocessing garphs############################################

################draw Test Error and correct rate graphs##########################################
def drawResultGraphs():
    L = 30
    k_neighbor = np.arange(L) + 1
    rbf_gamma = np.arange(L)/30.+0.1
    sigmoid_gamma = rbf_gamma
    poly_degree = np.arange(L)/5. + 0.5

    knn_score = []
    rbf_svm_score = []
    sigmoid_svm_score = []
    poly_svm_score = []

    knn_error = []
    rbf_svm_error = []
    sigmoid_svm_error = []
    poly_svm_error = []

    for i in range(L):
        rbf_par = rbf_gamma[i]
        sigmoid_par = sigmoid_gamma[i]
        poly_par = poly_degree[i]
        k = k_neighbor[i]

        knn_score.append(correct_rate('knn', X_train, Y_train, X_test, Y_test, k))
        knn_error.append(test_error('knn', X_train, Y_train, X_test, Y_test, k))

        rbf_svm_score.append(correct_rate('rbfsvm', X_train, Y_train, X_test, Y_test, rbf_par))
        rbf_svm_error.append(test_error('rbfsvm', X_train, Y_train, X_test, Y_test, rbf_par))

        sigmoid_svm_score.append(correct_rate('sigsvm', X_train, Y_train, X_test, Y_test, sigmoid_par))
        sigmoid_svm_error.append(test_error('sigsvm', X_train, Y_train, X_test, Y_test, sigmoid_par))

        poly_svm_score.append(correct_rate('polysvm', X_train, Y_train, X_test, Y_test, poly_par))
        poly_svm_error.append(test_error('polysvm', X_train, Y_train, X_test, Y_test, poly_par))

    knn_fig1 = plt.figure(1)
    plt.plot(k_neighbor, knn_score)
    plt.xlabel('k')
    plt.ylabel("Correct Rate")
    plt.title('K neareast neighbors')
    plt.show()
    knn_fig1.savefig('/Users/guichengwu/Desktop/208_mid term/knn_kernel_score.png')

    knn_fig2 = plt.figure(2)
    plt.plot(k_neighbor, knn_error)
    plt.xlabel('k')
    plt.ylabel("Test Error")
    plt.title('K neareast neighbors')
    plt.show()
    knn_fig2.savefig('/Users/guichengwu/Desktop/knn_kernel_error.png')

    rbfsvm_fig3 = plt.figure(3)
    plt.plot(rbf_gamma, rbf_svm_score)
    plt.xlabel('gamma')
    plt.ylabel("Correct Rate")
    plt.title('RBF kernel SVM')
    plt.show()
    rbfsvm_fig3.savefig('/Users/guichengwu/Desktop/208_mid term/rbfsvm_score.png')

    rbfsvm_fig4 = plt.figure(4)
    plt.plot(rbf_gamma, rbf_svm_error)
    plt.xlabel('gamma')
    plt.ylabel("Test Error")
    plt.title('RBF kernel SVM')
    plt.show()
    rbfsvm_fig4.savefig('/Users/guichengwu/Desktop/208_mid term/rbfsvm_error.png')

    sigsvm_fig5 = plt.figure(5)
    plt.plot(sigmoid_gamma, sigmoid_svm_score)
    plt.xlabel('gamma')
    plt.ylabel("Correct Rate")
    plt.title('Sigmoid kernel SVM')
    plt.show()
    sigsvm_fig5.savefig('/Users/guichengwu/Desktop/208_mid term/sigsvm_score.png')

    sigsvm_fig6 = plt.figure(6)
    plt.plot(sigmoid_gamma, sigmoid_svm_error)
    plt.xlabel('gamma')
    plt.ylabel("Test Error")
    plt.title('Sigmoid kernel SVM')
    plt.show()
    sigsvm_fig6.savefig('/Users/guichengwu/Desktop/208_mid term/sigsvm_error.png')

    polysvm_fig7 = plt.figure(7)
    plt.plot(poly_degree, poly_svm_score)
    plt.xlabel('degree')
    plt.ylabel("Correct Rate")
    plt.title('Polynomial kernel SVM')
    plt.show()
    polysvm_fig7.savefig('/Users/guichengwu/Desktop/208_mid term/polysvm_score.png')

    polysvm_fig8 = plt.figure(8)
    plt.plot(poly_degree, poly_svm_error)
    plt.xlabel('degree')
    plt.ylabel("Test Error")
    plt.title('Polynomial kernel SVM')
    plt.show()
    polysvm_fig8.savefig('/Users/guichengwu/Desktop/208_mid term/polysvm_error.png')
################draw Test Error and correct rate graphs########################################

################draw algorithms comparison graph###############################################
def drawAlgoCompGraph():
    h = 0.02
    names = ["ridge", "KNN", "Linear SVM", "RBF SVM", "LDA",
             "Random Forest", "AdaBoost", "Naive Bayes", "QDA", "Logistic"]

    kernel_names =['laplacian kernel', 'RBF kernel', 'Sigmoid kernel']
    classifiers = [
        linear_model.Ridge(),
        KNeighborsClassifier(9),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="rbf", gamma=0.25),
        LDA(),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QDA(),
        linear_model.LogisticRegression()]

    filename = '/Users/guichengwu/Desktop/208_mid term/exam.dat'

    data = np.loadtxt(filename, dtype='str')

    for i in range(data.shape[0]):
        for j in range(1,data.shape[1]):
            data[i][j] = data[i][j][2:]

    data_matrix = np.matrix(data).astype(np.float)
    X = data_matrix[:, 1:5]
    y = np.asarray(data_matrix[:, 0])
    X = preprocessing.scale(X)

    Lap_X = laplacian_kernel(X)
    pca1 = decomposition.PCA(n_components=2)
    pca1.fit(Lap_X)
    Lap_X = pca1.transform(Lap_X)

    RBF_X = rbf_kernel(X)
    pca2 = decomposition.PCA(n_components=2)
    pca2.fit(RBF_X)
    RBF_X = pca2.transform(RBF_X)

    Sig_X = sigmoid_kernel(X)
    pca3 = decomposition.PCA(n_components=2)
    pca3.fit(Sig_X)
    Sig_X = pca3.transform(Sig_X)

    linearly_separable1 = (Lap_X, y)
    linearly_separable2 = (RBF_X, y)
    linearly_separable3 = (Sig_X, y)

    datasets = [            
                linearly_separable1,
                linearly_separable2,
                linearly_separable3,
                ]

    figure = plt.figure(figsize=(30, 10))
    i = 1

    for kernel_name, ds in zip(kernel_names, datasets):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max()+0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max()+0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers)+1, i)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        ax.scatter(X_test[:, 0], X_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(kernel_name)
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            ax.scatter(X_train[:, 0], X_train[:,1], c=y_train, cmap=cm_bright)
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - 0.3, yy.min() + 0.3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=0.02, right=0.98)
    plt.show()
    figure.savefig('/Users/guichengwu/Desktop/algorithm_comparison2.png')
################draw algorithms comparison graph##############################################

