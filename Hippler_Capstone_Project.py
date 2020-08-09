# =========================Instructions========================================
# You need to produce an R or Python notebook that covers the full scope of the data science courses, 
# from exploring data to optimizing machine learning model performance. Throughout each stage of the process, 
# thoroughly explain your thought process. For example, perhaps you chose to ignore a certain variable because 
# it is too related to another feature, or because regularization indicated it was not useful.
#  
# 
# Exploratory Data Analysis: Summarize variables, visualize distributions and relationships. 
#     Generate a few interesting questions about the data and explore them with some visualizations.
# Research Methods: Calculate the sample correlation between at least one pair of variables. 
#     Come up with a hypothesis and calculate the p-value.
# Data Cleaning and Preparation: Apply any appropriate preprocessing steps, 
#     such as removing duplicates, missing values, outliers, and scaling data as appropriate 
#     (note that which model(s) is/are chosen may determine whether scaling is necessary).
# Feature Engineering: Create new features or transform existing ones to improve performance. 
#     Even if you decide not to use these features (e.g., they don’t affect performance or make it worse), 
#     keep the code and an explanation of what you tried in your notebook.
# Model Selection: Try various models (at least 3), showing your evaluation process. 
#     Clearly indicate which metrics you used and the performance of each model.
#     Be sure to address any imbalance in the data, as well as using an appropriate train/test data split.
# Performance Optimization: Use regularization, hyperparameter tuning, 
#     or other techniques to further optimize your chosen model and/or help select the best model.
#  
# At the end of your notebook, provide a brief summary (one paragraph) of your model – what it is, 
# what preprocessing, feature engineering, and optimization you did, and the final accuracy 
# (or another appropriate metric). Finally, briefly provide three ideas that could improve the model, 
# which may include collecting additional variables.
#
# Dataset:
# Census Income – Predict whether an individual’s income exceeds $50K/year based on census data (classification)
# =============================================================================


import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model as lm
from sklearn import tree
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Import the Data: Census Income
#Tried importing the data without defining the data types, and they were all imported with a dtype of "Object"
#Manually set dtype as the data is imported in order to perform operations on certain columns later
name_list = ["Age", "Workclass", "Fnlwgt", "Education", "Education-Number", "Marital-Status", "Occupation", "Relationship", "Race", "Sex", "Capital-gain", "Capital-loss", "HPW", "Native_Country", "Income"]
dtypes = {"Age":int, "Workclass":str, "Fnlwgt":int, "Education":str, "Education-Number":int, "Marital-Status":str, "Occupation":str, "Relationship":str, "Race":str, "Sex":str, "Capital-gain":int, "Capital-loss":int, "HPW":int, "Native_Country":str, "Income":str}
data = pd.read_csv("adult_data.csv", delimiter=',', names = name_list, skiprows=1, dtype=dtypes)

#Print out differnt aspects of the data to get a quick overview of what has been imported
print(data)
print(data.dtypes)
print(data.shape)


#Clean the Data
cols = ['Workclass', 'Occupation', 'Native_Country']
for column in cols:
    data.loc[data[column] == " ?", column] = numpy.nan
data.dropna(axis = 0, inplace = True)
#Use the data.shape function after this to see how many rows have been removed in cleaning
#In this case, we went from 32,561 rows to 30,162 rows
print(data.shape)

#Education level is broken up into 16 different categories in this dataset. 
    #1:preschool, 2:1st-4th, 3:5th-6th, 4:7th-8th, 5:9th, 6:10th, 7:11th, 8:12th, 9:HS-grad
    #10:Some-college, 11:Assoc-voc, 12:Assoc-acdm, 13:Bachelors, 14:Masters, 15:Prof-school, 16:Doctorate
#I want to group these back up a little, so I am grouping everything into 4 groups: Non-HS grads,
    #HS grads, College grads, and people with post-grad education
#Originally thought about just splitting the data into 2 categories (whether or not they graduated HS)
    #data['HS'] = numpy.where(data["Education-Number"] < 9, 'HS', 'College')

#New Education column that groups them as 4 categories
data['Edu'] = data["Education-Number"].astype(str)
print(data.dtypes)
data['Edu'] = data['Edu'].replace(['1','2', '3', '4', '5', '6', '7', '8'], "Non-HS")
data['Edu'] = data['Edu'].replace(['9','10'], "HS")
data['Edu'] = data['Edu'].replace(['11','12', '13'], "College")
data['Edu'] = data['Edu'].replace(['14', '15', '16'], "Post-Grad")


#Visualizations
#Want to look at a histogram of the 'Edu' column we built based on their income
data_grouped = data.groupby('Income')
plt.title('Income based on Education')
data_grouped['Edu'].hist()
plt.show()
plt.clf()

#Creating the same histogram of the 'Edu' column, but this time split into two seperate
    #plots based on whether or not they make above or below $50
data['Edu'].hist(by=data['Income'])
plt.show()
plt.clf()

#Creating a histogram to see how age and income are 
plt.title('Income based on Age')
data_grouped['Age'].plot.hist()
plt.legend()
plt.show()
plt.clf()



#Hypothesis testing: 
#H1 (Altertive): People with a education higher than a high school education are more likely
    #to have an income >$50K than a person who has at most a high school education
#H0 (Null):Education has no effect on income
#With a P-value lower than .05 we we would fail to reject the alternatiave hypothesis that 
    #education and income are related

#Chi-Squared test to get p-value and test hypothesis
crosstab = pd.crosstab(data['Income'],data['Edu'])
print(stats.chi2_contingency(crosstab))
print("Chi-Squared")
#print(stats.chisquare(crosstab))
#print(crosstab)    
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab)
print('Chi2 Stat: ', chi2_stat)
print('Degrees of Freedom: ', dof)
print('P-Value ', p_val)
float_p_val = "{:.2f}".format(p_val)
sci_p_val = "{:.2e}".format(p_val)
print('Float P-Value ', float_p_val)
print('Contingency Table:\n', ex)



#Encode the data
#In this case, we are using label encoding. We could switch to OHE and retrain the models as needed
categorical_columns = ["Workclass","Education", "Marital-Status", "Occupation", "Relationship", "Race", "Sex", "Native_Country", "Income"]
data_encoded = data.copy()
for col in categorical_columns:
   data_encoded[col] = data_encoded[col].astype('category').cat.codes
#This code will run the following for each defined column:
    #data_encoded['Workclass'] = data_encoded['Workclass'].astype('category').cat.codes


#Because Education is already labeled as a numeric in the "Education-Number" column, we can remove that column,
   #as well as any other columns that do not need to be used to train the models
data_encoded.drop(columns = ['Education', 'Edu'], axis=1, inplace=True)
print("Data Encoded \n", data_encoded)


#Create the label (y) variable from the Income column since that is waht we are trying to make predictions of,
#Create the feature (x) variable by removing the Income column and keeping the other columns
y = data_encoded['Income']
x = data_encoded.drop(columns = ['Income'], axis = 1)

#Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


#Scale the data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#Create a Logistic Regression model and determine its accuracy
logreg = lm.LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_logreg = logreg.predict(x_test)
print("Logistic Regression Accuracy: ", metrics.accuracy_score(y_test, y_pred_logreg))
#can use metrics.accuracy_score(y_test, y_pred)*100 to get accuracy as a percentage


#Create a K-Nearest Neighbors model and determine its accuracy
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_KNN = knn.predict(x_test)
print("KNN Accuracy: ",metrics.accuracy_score(y_test, y_pred_KNN))


#Create a Decision Tree model and determine its accuracy
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred_DT = decision_tree.predict(x_test)
print("Decision Tree Accuracy: ",metrics.accuracy_score(y_test, y_pred_DT))

# =============================================================================
# Accuracy at the time of being run:
#   Logistic Regression Accuracy:  0.8219692783733009
#   KNN Accuracy:  0.8218587689247431
#   Decision Tree Accuracy:  0.8060559177809703
# =============================================================================


#Create a Confusion Matrix for the Logistic Regression Model
results_LR = sklm.confusion_matrix(y_test, y_pred_logreg) 
print('Confusion Matrix - Logistic Regression:')
print(results_LR) 
print('Report: ')
print(sklm.classification_report(y_test, y_pred_DT))


#Create a Confusion Matrix for the K-Nearest Neighbors Model
results_KNN = sklm.confusion_matrix(y_test, y_pred_KNN) 
print('Confusion Matrix - K-Nearest Neighbors:')
print(results_KNN) 
print('Report: ')
print(sklm.classification_report(y_test, y_pred_KNN))

#Create a Confusion Matrix for the Decision Tree Model
results_DT = sklm.confusion_matrix(y_test, y_pred_DT) 
print('Confusion Matrix - Decision Tree:')
print(results_DT) 
print('Report: ')
print(sklm.classification_report(y_test, y_pred_DT))


#Evaluate models with AUC Graphs
#need to calculate the probabilities for each of the modls
probabilities_logreg = logreg.predict_proba(x_test)
probabilities_KNN = knn.predict_proba(x_test)
probabilities_DT = decision_tree.predict_proba(x_test)

#Define a function to graph the area under the curve for each model
def plot_auc(labels, probs, title):
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.clf()
    
#Evaluate each model with the defined function
plot_auc(y_test, probabilities_logreg, 'Receiver Operating Characteristic - Logistic Regression')
plot_auc(y_test, probabilities_KNN, 'Receiver Operating Characteristic- K-Nearest Neighbors')
plot_auc(y_test, probabilities_DT, 'Receiver Operating Characteristic - Decision Tree')

# =============================================================================
# AUC at the time of being run:
#   Logistic Regression Accuracy:  .86
#   KNN Accuracy:  .86
#   Decision Tree Accuracy:  .75
# =============================================================================

#Cross-Validation on K-Nearest Neighbors Model
cv_scores = cross_val_score(knn, x, y, cv=10)
print('Cross-validation Scores: \n', cv_scores)
print('Cross-Validation Scores Mean:{} '.format(numpy.mean(cv_scores)))


#Determining best value of number of neighbors for KNN model
# =============================================================================
# I was origanlly going to use the GridSearchCV function to determine the best value for
# n-neighbors, but decided to use a plot of the error rate vs n_neighbors values instead due 
# to some performance issues.
#
# param_grid = {'n_neighbors': numpy.arange(1, 30)}
# knn_gscv = GridSearchCV(knn, param_grid, cv=10)
# knn_gscv.fit(x_train, y_train)
# print('Best Number of Neighbors: ', knn_gscv.best_params_)
# print('Mean Score for Best: ', knn_gscv.best_score_)
# =============================================================================

error_rate = []
for x in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = x)
    knn.fit(x_train,y_train)
    pred_x = knn.predict(x_test)
    error_rate.append(numpy.mean(pred_x != y_test))
    

plt.plot(range(1,40),error_rate)
plt.title('Error Rate vs. n_neighbors')
plt.xlabel('n-neighbors')
plt.ylabel('Error Rate')
plt.show()
plt.clf()
# Based on the graph produced by running this code, the n_neighbors with the
    #lowest error rate is 10

#Create and test a new K-Nearest Neighbors Model, this time using n_neighbors 
#determined by best value from the error rate graph
#In this case n_neighbors = 10
knn_new = KNeighborsClassifier(n_neighbors=10)
knn_new.fit(x_train, y_train)
y_pred_KNN_new = knn_new.predict(x_test)
print("New KNN Accuracy: ",metrics.accuracy_score(y_test, y_pred_KNN_new))

#New Confusion MatrixCreate a Confusion Matrix for the K-Nearest_25 Neighbors Model
results_KNN_new = sklm.confusion_matrix(y_test, y_pred_KNN_new) 
print('Confusion Matrix - New K-Nearest Neighbors:')
print(results_KNN_new) 
print('Report: ')
print(sklm.classification_report(y_test, y_pred_KNN_new))

#New AUC
probabilities_KNN_new = knn_new.predict_proba(x_test)
plot_auc(y_test, probabilities_KNN_new, 'Receiver Operating Characteristic- K-Nearest Neighbors')

#==============================================================================
# Based on the AUC graphs and the accuracy the Logistic Regression and K-Nearest Neighbors
# models were the more accurate than the Decision Tree model. In the Confusion Matricies,
# the K-Nearest Neighbor had a slightly smaller accuracy, and would be the model I would use.
# Because that is the model selected, you could test changing the value of 
# The next you could do to improve the models is to try even more machine learning models
# to see if there are any models that perform better than the three tested here. Additionally,
# you could try combining different models by averaging or stacking the models. Finally, you
# can focus on the features that are being used in the model by either removing more variables
# or applying more feature engineering 
# =============================================================================
