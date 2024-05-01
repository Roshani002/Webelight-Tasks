#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score


# In[3]:


dataset = pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")
dataset.head()


# Data Understanding and EDA

# In[4]:


dataset.columns


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[7]:


df= dataset.copy()


# In[8]:


df['Diabetes_01'] = df['Diabetes_012'].replace({1: 0, 2: 1})


# In[9]:


print(df['Diabetes_01'].unique())


# In[10]:


df.drop(columns=['Diabetes_012'], inplace=True)


# In[11]:


df.describe().T


# In[12]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


# In[13]:


cat_cols, num_cols, cat_but_car= grab_col_names(df)


# In[14]:


cat_cols


# In[15]:


num_cols


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
def num_summary(dataframe, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1]
    for col in num_cols:  # num_cols = grab_col_names(dataframe)["num_cols"]
        print("########## Summary Statistics of " + col + " ############")
        print(dataframe[col].describe(quantiles))
        
        if plot:
            plt.hist(data=dataframe, x=col)
            plt.xlabel(col)
            plt.title("The distribution of " + col)
            plt.grid(True)  
            plt.show(block=True)
            plt.title("The boxplot of " + col)
            plt.boxplot(x=df[col])
            plt.show(block=True)


# In[17]:


num_summary(df, plot=True)


# In[18]:


def cat_summary(dataframe, plot=False):
    import matplotlib.pyplot as plt
    for col in cat_cols:
        print("############## Frequency of Categorical Data ########################")
        print("The unique number of " + col + ": " + str(dataframe[col].nunique()))
        print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": 100* dataframe[col].value_counts() / len(dataframe)}))
        if plot:
            if dataframe[col].dtypes == "bool":
                dataframe[col] == dataframe[col].astype(int)
                plt.hist(dataframe[col])
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.title("Frequency of " + col)
                plt.show()
            else:
                values, counts = zip(*sorted(dataframe[col].value_counts().items()))
                plt.bar(values, counts)
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.title("Frequency of " + col)
                plt.show()


# In[19]:


cat_summary(df,plot=True)


# In[20]:


# Check if there are outliers
plt.figure(figsize=(40,20))
df.plot(kind='box')
plt.title("Box Plots of all features")
plt.xticks(rotation=90)
plt.show()


# In[21]:


column_name = 'BMI' 

# Plotting a box plot using matplotlib
plt.figure(figsize=(8, 6))
plt.boxplot(df[column_name])
plt.title('Box plot of {}'.format(column_name))
plt.ylabel('Value')
plt.show()


# In[22]:


column_name = 'GenHlth'

# Plotting a box plot using matplotlib
plt.figure(figsize=(8, 6))
plt.boxplot(df[column_name])
plt.title('Box plot of {}'.format(column_name))
plt.ylabel('Value')
plt.show()


# In[23]:


column_name = 'Education'

# Plotting a box plot using matplotlib
plt.figure(figsize=(8, 6))
plt.boxplot(df[column_name])
plt.title('Box plot of {}'.format(column_name))
plt.ylabel('Value')
plt.show()


# In[24]:


selected_columns = [ 'Income', 'Age']

# Create a boxplot for each selected column
plt.figure(figsize=(10, 6))
df[selected_columns].boxplot()
plt.title('Boxplot of Selected Columns')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[25]:


selected_columns = ['MentHlth','PhysHlth']

# Create a boxplot for each selected column
plt.figure(figsize=(10, 6))
df[selected_columns].boxplot()
plt.title('Boxplot of Selected Columns')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[26]:


# Removing Outlers
# Define the columns where outliers are more
outlier_columns = [ 'BMI', 'MentHlth', 'PhysHlth']  # Replace with your columns

# Calculate the first quartile (Q1) and third quartile (Q3) for each column
Q1 = df[outlier_columns].quantile(0.25)
Q3 = df[outlier_columns].quantile(0.75)

# Calculate the interquartile range (IQR) for each column
IQR = Q3 - Q1

# Define the lower and upper bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the dataset
df_no_outliers = df[~((df[outlier_columns] < lower_bound) | (df[outlier_columns] > upper_bound)).any(axis=1)]

# Display the shape of the original and modified datasets
print("Original dataset shape:", df.shape)
print("Dataset shape after removing outliers:", df_no_outliers.shape)


# In[27]:


# Calculate correlation matrix
corr_matrix = df.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set the upper triangle values to NaN
corr_matrix = corr_matrix.mask(mask)

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation')
plt.title('Correlation Matrix')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout()
plt.show()

highest_corr = corr_matrix.unstack().sort_values(ascending=False)
highest_corr_pair = highest_corr[(highest_corr < 1)].index[0]


# In[28]:


# Set up subplots
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot histograms for each feature
for i, feature in enumerate(df.columns):
    if i < len(df.columns):  # Ensure only necessary number of subplots are used
        axes[i].hist(df[feature], bins=30, color='skyblue', alpha=0.7)
        axes[i].set_title(f'Histogram: {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


# More than 57% do not have high blood pressure or high cholesterol.
# Over 90% did not report stroke or heart attack issues, excessive alcohol consumption, or difficulties visiting a doctor (due to economic reasons). Along these lines, 95.1% reported having some form of medical coverage.
# 55.7% do not smoke, and 75.7% reported exercising in the last 30 days.
# 83.2% reported no difficulty walking or climbing stairs.
# Over 90% of respondents reported eating at least one fruit per day, as well as at least one vegetable.
# 56% are women.

# 

# In[29]:


df.isnull().sum()


# In[ ]:





# In[30]:


# Split the dataset into features (X) and the target variable (y)
X = df.drop(columns=['Diabetes_01'])
y = df['Diabetes_01']


# In[31]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[33]:


# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_scaled, y_train)
y_pred_lr = logistic_regression.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression: Accuracy = {accuracy_lr:.4f}')


# In[34]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_scaled, y_train)
y_pred_dt = decision_tree.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree: Accuracy = {accuracy_dt:.4f}')


# In[35]:


# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train_scaled, y_train)
y_pred_rf = random_forest.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest: Accuracy = {accuracy_rf:.4f}')


# In[36]:


# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'K-Nearest Neighbors: Accuracy = {accuracy_knn:.4f}')


# In[37]:


naive_bayes = GaussianNB()
naive_bayes.fit(X_train_scaled, y_train)
y_pred_nb = naive_bayes.predict(X_test_scaled)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes: Accuracy = {accuracy_nb:.4f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




