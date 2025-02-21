# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

label_encoder = LabelEncoder()

df = pd.read_csv('/content/Customer-Churn.csv')

#gereksiz column
df = df.drop(columns=['customerID'])
#convert to num
df=df[df.TotalCharges!=' ']
df.TotalCharges=pd.to_numeric(df.TotalCharges)

binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod', 'Churn']

for column in binary_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)




from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model for predictions
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': best_clf.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

from google.colab import drive
drive.mount('/content/drive')

#Implement Random Forest

from sklearn.ensemble import RandomForestClassifier

label_encoder = LabelEncoder()

df = pd.read_csv('/content/Customer-Churn.csv')

#gereksiz column
df = df.drop(columns=['customerID'])
#convert to num
df=df[df.TotalCharges!=' ']
df.TotalCharges=pd.to_numeric(df.TotalCharges)

binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod', 'Churn']

for column in binary_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


rf_clf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 4, 10]
}


rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=3, n_jobs=-1, verbose=0)
rf_grid_search.fit(X_train_resampled, y_train_resampled)


rf_best_params = rf_grid_search.best_params_
rf_best_clf = rf_grid_search.best_estimator_


rf_y_pred = rf_best_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_class_report = classification_report(y_test, rf_y_pred)


rf_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_best_clf.feature_importances_})
rf_feature_importance = rf_feature_importance.sort_values(by='Importance', ascending=False)


print("Best Hyperparameters:", rf_best_params)
print(f"Accuracy: {rf_accuracy:.4f}")
print("Confusion Matrix:\n", rf_conf_matrix)
print("Classification Report:\n", rf_class_report)
print("Feature Importance:\n", rf_feature_importance)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importance)
plt.title('Feature Importance')
plt.show()

#Implement SVM
from sklearn.svm import SVC

df = pd.read_csv("Customer-Churn.csv")
df = df.drop(['TotalCharges', 'customerID'], axis=1)
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                  'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract',
                  'PaperlessBilling', 'PaymentMethod', 'Churn']
df = pd.get_dummies(df, columns=binary_columns)
df = df.drop(['Churn_No'], axis=1)


X = df.drop(['Churn_Yes'], axis=1)
y = df['Churn_Yes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_clf = SVC(random_state=42)


svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale'],
    'degree': [3, 4]
}


svm_grid_search = GridSearchCV(svm_clf, svm_param_grid, cv=3, n_jobs=-1, verbose=0)
svm_grid_search.fit(X_train, y_train)


svm_best_params = svm_grid_search.best_params_
svm_best_clf = svm_grid_search.best_estimator_


svm_y_pred = svm_best_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_conf_matrix = confusion_matrix(y_test, svm_y_pred)
svm_class_report = classification_report(y_test, svm_y_pred)


print("Best Hyperparameters:", svm_best_params)
print(f"Accuracy: {svm_accuracy:.4f}")
print("Confusion Matrix:\n", svm_conf_matrix)
print("Classification Report:\n", svm_class_report)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

clf.fit(X_train, y_train)
dt_y_pred_prob = clf.predict_proba(X_test)[:, 1]
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_y_pred_prob)
dt_roc_auc = auc(dt_fpr, dt_tpr)

rf_clf.fit(X_train, y_train)
rf_y_pred_prob = rf_clf.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred_prob)
rf_roc_auc = auc(rf_fpr, rf_tpr)

svm_clf.fit(X_train, y_train)
svm_y_scores = svm_clf.decision_function(X_test)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_y_scores)
svm_roc_auc = auc(svm_fpr, svm_tpr)


plt.figure(figsize=(10, 8))

plt.plot(dt_fpr, dt_tpr, color='blue', lw=2, label='Decision Tree ROC curve (area = %0.2f)' % dt_roc_auc)
plt.plot(rf_fpr, rf_tpr, color='green', lw=2, label='Random Forest ROC curve (area = %0.2f)' % rf_roc_auc)
plt.plot(svm_fpr, svm_tpr, color='red', lw=2, label='SVM ROC curve (area = %0.2f)' % svm_roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multiple Models')
plt.legend(loc="lower right")
plt.show()



from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

label_encoder = LabelEncoder()

df = pd.read_csv('/content/Customer-Churn.csv')

#gereksiz column
df = df.drop(columns=['customerID'])
#convert to num
df=df[df.TotalCharges!=' ']
df.TotalCharges=pd.to_numeric(df.TotalCharges)

binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod', 'Churn']

for column in binary_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
