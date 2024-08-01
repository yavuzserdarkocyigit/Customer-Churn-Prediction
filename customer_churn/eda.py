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
print(df.isnull().sum())

# Visualize the distribution of the target variable 'Churn'
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('Churn_Distribution.png')
plt.show()


plt.figure(figsize=(8, 5))
sns.kdeplot(df[df['Churn'] == 'No']['tenure'], label='Churn: No', shade=True)
sns.kdeplot(df[df['Churn'] == 'Yes']['tenure'], label='Churn: Yes', shade=True)
plt.title('Customer Tenure Distribution by Churn')
plt.xlabel('Tenure')
plt.ylabel('Density')
plt.savefig('Tenure_Distribution_by_Churn.png')
plt.show()
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
# Visualize the distribution of numerical features by Churn
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("nUMMMMMM",numerical_columns)
# Visualize the distribution of numerical features by Churn with counts
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[df['Churn'] == 'No'][column], label='Churn: No', kde=True, color='blue')
    sns.histplot(df[df['Churn'] == 'Yes'][column], label='Churn: Yes', kde=True, color='orange')

    # Annotate bars with counts
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=8, color='black', xytext=(0, 10),
                           textcoords='offset points')

    plt.title(f'{column} Distribution by Churn')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{column}_Distribution.png')
    plt.show()

# Visualize the distribution of categorical variables with counts
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column, hue='Churn', data=df)

    # Annotate bars with counts
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=8, color='black', xytext=(0, 10),
                           textcoords='offset points')

    plt.title(f'{column} Distribution by Churn')
    plt.savefig(f'{column}_Distribution_by_Churn.png')
    plt.show()

# Box plots for numerical features by Churn with counts
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y=column, data=df)

    # Annotate boxes with counts
    if 'No' in df['Churn'].unique() and 'Yes' in df['Churn'].unique():
        counts = df.groupby('Churn')[column].count().to_dict()
        for i, label in enumerate(df['Churn'].unique()):
            plt.text(i, 0.95, f'Count: {counts[label]}', ha='center', va='center',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f'{column} Box Plot by Churn')
    plt.savefig(f'{column}_Box_Plot_by_Churn.png')
    plt.show()


