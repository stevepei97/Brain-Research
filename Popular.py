import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statistics
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from IPython.display import Image
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

##############################################
train = pd.read_csv('diet_data.csv')
warnings.filterwarnings('ignore')

del train['Unnamed: 0']

dict_ls = {'Amc': 2, 'Med': 1, 'Veg': 0}

dict_lss = {'Over': 1, 'Lean': 0}

train['diet'].replace(dict_ls, inplace=True)

train['obs'].replace(dict_lss, inplace=True)

#############################################

X_train, X_test = train_test_split(train, test_size = 0.3, random_state = 5)

y_train = X_train['Sex']

y_test = X_test['Sex']

X_train = X_train.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)

# Instantiate
logit_model = LogisticRegression()
# Fit
logit_model = logit_model.fit(X_train, y_train)
# How accurate?
logit_model.score(X_train, y_train)

# How does it perform on the test dataset?

# Predictions on the test dataset
predicted = pd.DataFrame(logit_model.predict(X_test))
# Probabilities on the test dataset
probs = pd.DataFrame(logit_model.predict_proba(X_test))

log_accuracy = metrics.accuracy_score(y_test, predicted)
log_roc_auc = metrics.roc_auc_score(y_test, probs[1])
log_confus_matrix = metrics.confusion_matrix(y_test, predicted)
log_classification_report = metrics.classification_report(y_test, predicted)
log_precision = metrics.precision_score(y_test, predicted, pos_label=1)
log_recall = metrics.recall_score(y_test, predicted, pos_label=1)
log_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

log_cv_scores = cross_val_score(logit_model, X_test, y_test, scoring='precision', cv=10)
log_cv_mean = np.mean(log_cv_scores)



#0.7321

#############################################

# Instantiate with a max depth of 3
tree_model = tree.DecisionTreeClassifier(max_depth=3)
# Fit a decision tree
tree_model = tree_model.fit(X_train, y_train)
# Training accuracy
tree_model.score(X_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(tree_model.predict(X_test))
probs = pd.DataFrame(tree_model.predict_proba(X_test))

# Store metrics
tree_accuracy = metrics.accuracy_score(y_test, predicted)
tree_roc_auc = metrics.roc_auc_score(y_test, probs[1])
tree_confus_matrix = metrics.confusion_matrix(y_test, predicted)
tree_classification_report = metrics.classification_report(y_test, predicted)
tree_precision = metrics.precision_score(y_test, predicted, pos_label=1)
tree_recall = metrics.recall_score(y_test, predicted, pos_label=1)
tree_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
tree_cv_scores = cross_val_score(tree.DecisionTreeClassifier(max_depth=3), X_test, y_test, scoring='precision', cv=10)

# output decision plot
dot_data = tree.export_graphviz(tree_model, out_file=None,
                     feature_names=X_test.columns.tolist(),
                     class_names=['Over', 'Lean'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = graph_from_dot_data(dot_data)

##########################################

# Instantiate
rf = RandomForestClassifier()
# Fit
rf_model = rf.fit(X_train, y_train)
# training accuracy 99.74%
rf_model.score(X_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(X_test))
probs = pd.DataFrame(rf_model.predict_proba(X_test))

# Store metrics
rf_accuracy = metrics.accuracy_score(y_test, predicted)
rf_roc_auc = metrics.roc_auc_score(y_test, probs[1])
rf_confus_matrix = metrics.confusion_matrix(y_test, predicted)
rf_classification_report = metrics.classification_report(y_test, predicted)
rf_precision = metrics.precision_score(y_test, predicted, pos_label=1)
rf_recall = metrics.recall_score(y_test, predicted, pos_label=1)
rf_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# Evaluate the model using 10-fold cross-validation
rf_cv_scores = cross_val_score(RandomForestClassifier(), X_test, y_test, scoring='precision', cv=10)
rf_cv_mean = np.mean(rf_cv_scores)

#########################################

# Instantiate
svm_model = SVC(probability=True)
# Fit
svm_model = svm_model.fit(X_train, y_train)
# Accuracy
svm_model.score(X_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(svm_model.predict(X_test))
probs = pd.DataFrame(svm_model.predict_proba(X_test))

# Store metrics
svm_accuracy = metrics.accuracy_score(y_test, predicted)
svm_roc_auc = metrics.roc_auc_score(y_test, probs[1])
svm_confus_matrix = metrics.confusion_matrix(y_test, predicted)
svm_classification_report = metrics.classification_report(y_test, predicted)
svm_precision = metrics.precision_score(y_test, predicted, pos_label=1)
svm_recall = metrics.recall_score(y_test, predicted, pos_label=1)
svm_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# Evaluate the model using 10-fold cross-validation
svm_cv_scores = cross_val_score(SVC(probability=True), X_test, y_test, scoring='precision', cv=10)
svm_cv_mean = np.mean(svm_cv_scores)


########################################

# instantiate learning model (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=3)
# fit the model
knn_model.fit(X_train, y_train)
# Accuracy
knn_model.score(X_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(knn_model.predict(X_test))
probs = pd.DataFrame(knn_model.predict_proba(X_test))

# Store metrics
knn_accuracy = metrics.accuracy_score(y_test, predicted)
knn_roc_auc = metrics.roc_auc_score(y_test, probs[1])
knn_confus_matrix = metrics.confusion_matrix(y_test, predicted)
knn_classification_report = metrics.classification_report(y_test, predicted)
knn_precision = metrics.precision_score(y_test, predicted, pos_label=1)
knn_recall = metrics.recall_score(y_test, predicted, pos_label=1)
knn_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# Evaluate the model using 10-fold cross-validation
knn_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), X_test, y_test, scoring='precision', cv=10)
knn_cv_mean = np.mean(knn_cv_scores)

############################################


# Instantiate
bayes_model = GaussianNB()
# Fit the model
bayes_model.fit(X_train, y_train)
# Accuracy
bayes_model.score(X_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(bayes_model.predict(X_test))
probs = pd.DataFrame(bayes_model.predict_proba(X_test))

# Store metrics
bayes_accuracy = metrics.accuracy_score(y_test, predicted)
bayes_roc_auc = metrics.roc_auc_score(y_test, probs[1])
bayes_confus_matrix = metrics.confusion_matrix(y_test, predicted)
bayes_classification_report = metrics.classification_report(y_test, predicted)
bayes_precision = metrics.precision_score(y_test, predicted, pos_label=1)
bayes_recall = metrics.recall_score(y_test, predicted, pos_label=1)
bayes_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# Evaluate the model using 10-fold cross-validation
bayes_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), X_test, y_test, scoring='precision', cv=10)
bayes_cv_mean = np.mean(bayes_cv_scores)

##############################################3


### Results


tree_cv_mean = statistics.mean(tree_cv_scores)

# Model comparison
models = pd.DataFrame({
  'Model': ['Log','d.Tree', 'r.f.', 'SVM', 'kNN',  'Bayes'],
  'Accuracy' : [log_accuracy, tree_accuracy, rf_accuracy, svm_accuracy, knn_accuracy, bayes_accuracy],
  'Precision': [log_precision, tree_precision, rf_precision, svm_precision, knn_precision, bayes_precision],
  'recall' : [log_recall, tree_recall, rf_recall, svm_recall, knn_recall, bayes_recall],
  'F1' : [ log_f1,tree_f1, rf_f1, svm_f1, knn_f1, bayes_f1],
  'cv_precision' : [log_cv_mean, tree_cv_mean, rf_cv_mean, svm_cv_mean, knn_cv_mean, bayes_cv_mean]
})
# Print table and sort by test precision
print(models.sort_values(by='Precision', ascending=False))

new_models = pd.DataFrame({
    'Stats':['Accuracy', 'Precision', 'Recall', 'F1', 'cv_Precision'],
    'Log-Reg':[log_accuracy,log_precision,log_recall,log_f1,log_cv_mean],
    'd.Tree':[tree_accuracy,tree_precision,tree_recall,tree_f1,tree_cv_mean],
    'R.F.':[rf_accuracy,rf_precision,rf_recall,rf_f1,rf_cv_mean],
    'SVM':[svm_accuracy,svm_precision,svm_recall,svm_f1,svm_cv_mean],
    'KNN':[knn_accuracy,knn_precision,knn_recall,knn_f1,knn_cv_mean],
    'Bayes':[bayes_accuracy,bayes_precision,bayes_recall,bayes_f1,bayes_cv_mean],
})

# Print table and sort by test precision
print(models.sort_values(by='Precision', ascending=False))

#Setting the positions and width for the bars
pos = list(range(len(new_models['Log-Reg'])))
width = 0.2

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        #using df['pre_score'] data,
        new_models['Log-Reg'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#EE3224',
        # with label the first value in first_name
        label=new_models['Stats'][0])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        new_models['KNN'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the second value in first_name
        label=new_models['Stats'][4])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        new_models['R.F.'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='gold',
        # with label the third value in first_name
        label=new_models['Stats'][2])

plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        new_models['SVM'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='peachpuff',
        # with label the third value in first_name
        label=new_models['Stats'][3])

# Set the y axis label
ax.set_ylabel('Score')

# Set the chart's title
ax.set_title('Prediction Scores of Obesity')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(new_models['Stats'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
plt.ylim([0, 1])

# Adding the legend and showing the plot
plt.legend(['Log-Reg','KNN', 'R.F.', 'SVM'], loc='upper left')
plt.grid()
plt.show()