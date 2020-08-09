import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn import linear_model
import numpy
import math
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
 

#data = pd.read_csv("bank_full.csv", sep = ";", names = ['age', "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
data = pd.read_csv("bank_additional_full_fin.csv", sep=",", skiprows = 1, names = ["Age", "Workclass", "Fnlwgt", "Education", "Eduction-Number", "Marital-Status", "Occupation", "Relationship", "Race", "Sex", "Capital-gain", "Capital-loss", "HPW", "Native Country", "Income"])
#print(data.dtypes)

#Data Cleanining
#job == unknown, education == unknown, default == unknown, housing == unknown, loan == unknown,poutcome == nonexistant?
#use "unknown" to remove missing data

print("TEST - clean data")

#NaN data in this dataset is defined as 'unknown' for the job, eduction, default, housing, and loan columns.
#We must convert 'unknown' to NaN so we can call dropna function to remove all rows with NaN data from our dataset
#This can be done by using a a for loop on the desired columns and setting 'unkown' values to NaN
cols = ['job', 'education', 'default', 'housing', 'loan']
for column in cols:
    data.loc[data[column] == 'unknown', column] = numpy.nan
data.dropna(axis = 0, inplace = True)
#went from 41,188 rows to 30,547 rows of data after null values were removed
print(data.shape)

print(data['job'].unique())
Features = data['job']
enc = preprocessing.LabelEncoder()
enc.fit(Features)
Features = enc.transform(Features)
print(Features)
    

#OHE
ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(Features.reshape(-1,1))
Features = encoded.transform(Features.reshape(-1,1)).toarray()
Features[:10,:]

#OHE categorical features
def encode_string(cat_feature):
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()
    

categorical_columns = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for col in categorical_columns:
    temp = encode_string(data[col])
    Features = numpy.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])    

#Add numeric features
Features = numpy.concatenate([Features, numpy.array(data[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']])], axis = 1)
Features[:2,:]

#Split the data
labels = numpy.array(data['cons.price.idx'])
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 6000)
x_train = Features[indx[0],:]
y_train = numpy.ravel(labels[indx[0]])
x_test = Features[indx[1],:]
y_test = numpy.ravel(labels[indx[1]])

#rescale the data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
x_train


#Linear Regression Model
lin_mod = linear_model.LinearRegression(fit_intercept = False)
lin_mod.fit(x_train, y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

#Evaluate the Model
print("TEST - Evaluate the model")
def print_metrics(y_true, y_predicted, n_parameters):
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))
   
y_score = lin_mod.predict(x_test) 
print_metrics(y_test, y_score, 28)

#Residual histogram
def hist_resids(y_test, y_score):
    resids = numpy.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    
hist_resids(y_test, y_score)    

#Q-Q Normal Plot
def resid_qq(y_test, y_score): 
    resids = numpy.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))  
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    
resid_qq(y_test, y_score)   

#Residual Plot
def resid_plot(y_test, y_score):
    resids = numpy.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')

resid_plot(y_test, y_score) 

#Transformed to real values from log values
y_score_untransform = numpy.exp(y_score)
y_test_untransform = numpy.exp(y_test)
resid_plot(y_test_untransform, y_score_untransform) 

#classification
le = preprocessing.LabelEncoder()
le.fit(data.y)
data['y'] = le.transform(data.y)
labels = numpy.array(data['y'])

indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 6000)
x2_train = Features[indx[0],:]
y2_train = numpy.ravel(labels[indx[0]])
x2_test = Features[indx[1],:]
y2_test = numpy.ravel(labels[indx[1]])

scaler = preprocessing.StandardScaler().fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_train



logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(x2_train, y2_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(x2_test)
print(probabilities[:17,:])

def score_model(probs, threshold):
    return numpy.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(numpy.array(scores[:17]))
print(y2_test[:17])

print(labels), print(scores)


def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
print_metrics(y2_test, scores)    

#Area under the curve
def plot_auc(labels, probs):
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_auc(y2_test, probabilities)         


#Cross Validate
labels = labels.reshape(labels.shape[0],)
scoring = ['precision_macro', 'recall_macro', 'roc_auc']
logistic_mod = linear_model.LogisticRegression(C = 1.0, class_weight = {0:0.45, 1:0.55}) 
scores = ms.cross_validate(logistic_mod, Features, labels, scoring=scoring,
                        cv=10, return_train_score=False)

def print_format(f,x,y,z):
    print('Fold %2d    %4.3f        %4.3f      %4.3f' % (f, x, y, z))

def print_cv(scores):
    fold = [x + 1 for x in range(len(scores['test_precision_macro']))]
    print('         Precision     Recall       AUC')
    [print_format(f,x,y,z) for f,x,y,z in zip(fold, scores['test_precision_macro'], scores['test_recall_macro'],scores['test_roc_auc'])]
    print('-' * 40)
    print('Mean       %4.3f        %4.3f      %4.3f' % (numpy.mean(scores['test_precision_macro']), numpy.mean(scores['test_recall_macro']), numpy.mean(scores['test_roc_auc'])))  
    print('Std        %4.3f        %4.3f      %4.3f' % (numpy.std(scores['test_precision_macro']), numpy.std(scores['test_recall_macro']), numpy.std(scores['test_roc_auc'])))

print_cv(scores) 

inside = ms.KFold(n_splits=10, shuffle = True)
outside = ms.KFold(n_splits=10, shuffle = True)

param_grid = {"C": [0.1, 1, 10, 100, 1000]}
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.45, 0:0.55}) 

clf = ms.GridSearchCV(estimator = logistic_mod, param_grid = param_grid, 
                      cv = inside,
                      scoring = 'roc_auc',
                      return_train_score = True)


clf.fit(Features, labels)
keys = list(clf.cv_results_.keys())
for key in keys[6:16]:
    print(clf.cv_results_[key])
clf.best_estimator_.C

def plot_cv(clf, params_grid, param = 'C'):
    params = [x for x in params_grid[param]]
  
    keys = list(clf.cv_results_.keys())              
    grid = numpy.array([clf.cv_results_[key] for key in keys[6:16]])
    means = numpy.mean(grid, axis = 0)
    stds = numpy.std(grid, axis = 0)
    print('Performance metrics by parameter')
    print('Parameter   Mean performance   STD performance')
    for x,y,z in zip(params, means, stds):
        print('%8.2f        %6.5f            %6.5f' % (x,y,z))
    
    params = [math.log10(x) for x in params]
    
    plt.scatter(params * grid.shape[0], grid.flatten())
    p = plt.scatter(params, means, color = 'red', marker = '+', s = 300)
    plt.plot(params, numpy.transpose(grid))
    plt.title('Performance metric vs. log parameter value\n from cross validation')
    plt.xlabel('Log hyperparameter value')
    plt.ylabel('Performance metric')
    
plot_cv(clf, param_grid) 

