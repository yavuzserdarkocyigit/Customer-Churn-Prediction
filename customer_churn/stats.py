import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Load the dataset
df = pd.read_csv("Customer-Churn.csv")
df = df.drop(['TotalCharges', 'customerID'], axis=1)

# Display basic information about the dataset
# Display the first few rows of the dataset
print(df.head())

# Display summary statistics of the numerical columns
print(df.describe())

# Check for missing values

categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
# Visualize the distribution of numerical features by Churn
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

from scipy.stats import mannwhitneyu

for column in numerical_columns:
    _, p_value = mannwhitneyu(df[df['Churn'] == 'Yes'][column], df[df['Churn'] == 'No'][column])

    print(f'Mann-Whitney U test for "{column}":')
    print(f'P-value: {p_value}')

    if p_value < 0.05:
        print(f'The difference in "{column}" between Churn: Yes and Churn: No is statistically significant.')
    else:
        print(f'There is no significant difference in "{column}" between Churn: Yes and Churn: No.')

    print('\n')

from scipy.stats import chi2_contingency

for column in categorical_columns:
    contingency_table = pd.crosstab(df[column], df['Churn'])
    _, p_value, _, _ = chi2_contingency(contingency_table)

    print(f'Chi-square test for "{column}":')
    print(f'P-value: {p_value}')

    if p_value < 0.05:
        print(f'The association between "{column}" and Churn is statistically significant.')
    else:
        print(f'There is no significant association between "{column}" and Churn.')

    print('\n')




