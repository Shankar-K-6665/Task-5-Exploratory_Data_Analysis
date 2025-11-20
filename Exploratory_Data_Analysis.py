
### **Task 5 - Exploratory Data Analysis (Titanic Dataset)**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "/content/Titanic-Dataset.csv"
df = pd.read_csv(data_path)
df.head()

df.info()         # data types and missing counts
df.describe(include='all').T   # numeric + non-numeric summary

print("Survived counts:\n", df['Survived'].value_counts())
print("\nPclass counts:\n", df['Pclass'].value_counts())
print("\nSex counts:\n", df['Sex'].value_counts())
print("\nEmbarked counts:\n", df['Embarked'].value_counts(dropna=False))

missing = df.isnull().sum().sort_values(ascending=False)
missing = pd.DataFrame({'missing_count': missing, 'missing_pct': missing/len(df)*100})
missing

# Age histogram
df['Age'].hist(bins=30)
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Fare distribution (log scale if skewed)
sns.histplot(df['Fare'].dropna(), bins=30)
plt.title('Fare distribution')
plt.show()

sns.boxplot(x=df['Age'])
plt.title('Boxplot: Age')
plt.show()

sns.boxplot(x=df['Fare'])
plt.title('Boxplot: Fare')
plt.show()

# Survived by Sex
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Sex')
plt.show()

# Survived by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Age vs Fare colored by Survived
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare (colored by Survived)')
plt.show()

# Age distribution by Survived (violin)
sns.violinplot(x='Survived', y='Age', data=df)
plt.title('Age distribution by Survival')
plt.show()

# First convert non-numeric if needed for correlation
corr_df = df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr()
corr_df

sns.heatmap(corr_df, annot=True, cmap='Blues')
plt.title('Correlation matrix')
plt.show()

# select small subset of cols to avoid clutter
sns.pairplot(df[['Survived','Pclass','Age','Fare']].dropna(), hue='Survived', diag_kind='hist')
plt.show()

# Survival rate by sex
surv_by_sex = df.groupby('Sex')['Survived'].mean().reset_index()
surv_by_sex

# Survival rate by Pclass
surv_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
surv_by_class

# Survival by Embarked
surv_by_embarked = df.groupby('Embarked')['Survived'].mean().reset_index()
surv_by_embarked

# Option 1: fill Age with median
df['Age_median_fill'] = df['Age'].fillna(df['Age'].median())

# Option 2: more advanced — fill by median per Pclass & Sex
df['Age_fill_by_group'] = df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Simple encoding for Sex and Embarked for modeling
df_model = df.copy()
df_model['Sex_encoded'] = df_model['Sex'].map({'male':0,'female':1})
df_model['Embarked_encoded'] = df_model['Embarked'].map({'S':0,'C':1,'Q':2})
df_model[['Sex','Sex_encoded','Embarked','Embarked_encoded']].head()

