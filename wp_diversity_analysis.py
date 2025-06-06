# Importing the Tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  make sure when we print data, it looks clean and doesn’t get cut off in the middle
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 350)
  
#read the two files

company = pd.read_csv("C:/Users/georg/OneDrive/Desktop/DEA/Projects/Workplace_Diversity_Analysis/company_hierarchy.csv")
print(company.head())

#  employee_id   boss_id   dept
# 0        46456  175361.0  sales
# 1       104708   29733.0     HR
# 2       120853   41991.0  sales
# 3       142630  171266.0     HR
# 4        72711  198240.0  sales

print(company.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10000 entries, 0 to 9999
# Data columns (total 3 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   employee_id  10000 non-null  int64
#  1   boss_id      9999 non-null   float64
#  2   dept         10000 non-null  object
# dtypes: float64(1), int64(1), object(1)
# memory usage: 234.5+ KB

employee = pd.read_csv("C:/Users/georg/OneDrive/Desktop/DEA/Projects/Workplace_Diversity_Analysis/employee.csv")
print(employee.head())

#    employee_id  signing_bonus    salary degree_level sex  yrs_experience
# 0       138719              0  273000.0       Master   M               2
# 1         3192              0  301000.0     Bachelor   F               1
# 2       114657              0  261000.0       Master   F               2
# 3        29039              0   86000.0  High_School   F               4
# 4       118607              0  126000.0     Bachelor   F               3

print(employee.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10000 entries, 0 to 9999
# Data columns (total 6 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   employee_id     10000 non-null  int64
#  1   signing_bonus   10000 non-null  int64
#  2   salary          10000 non-null  float64
#  3   degree_level    10000 non-null  object
#  4   sex             10000 non-null  object
#  5   yrs_experience  10000 non-null  int64
# dtypes: float64(1), int64(3), object(2)
# memory usage: 468.9+ KB

# Q: In the company there are 6 levels (described in the challenge). 
# Identify, for each employee, his/her corresponding level.

# 1. Identify Employee Levels: Classify each employee into one of six levels based on their role:
# ○	Individual Contributors (IC): Employees who do not manage anyone.
# ○	Middle Managers (MM): Direct supervisors of ICs.
# ○	Directors (D): Direct supervisors of MMs.
# ○	Vice Presidents (VP): Direct supervisors of Directors.
# ○	Executives (E): Direct supervisors of VPs.
# ○	CEO: Direct supervisor of Executives


#At first, we set everyone as IC, then we keep updating this vector
company['level'] = "IC"
  
#identify the CEO
company.loc[company.dept == "CEO", "level"] = "CEO"
  
#assign everyone else based on their boss level, climbing down the chain of command:
company_levels = ["CEO", "E", "VP", "D", "MM"]
  
for i in range(1,len(company_levels)):
  #identify IDs of the boss. This is employee ID of the level above
  boss_id = company.loc[company.level == company_levels[i-1], 'employee_id']
  company.loc[company.boss_id.isin(boss_id), "level"] = company_levels[i]

# What's happening here:

# For each level (starting from "E"), you check who reports to someone in the level above.
# If someone’s boss is, say, a VP, then that person must be a Director (D).
# This loop keeps assigning people to the correct level.
# It’s like playing detective:
# You know the CEO → find out who reports to them (E) → find out who reports to them (VP) → and so on.
  
#frequency
print(company.level.value_counts())

# level
# IC     9000
# MM      800
# D       160
# VP       35
# E         4
# CEO       1
# Name: count, dtype: int64

# Q: How many people each employee manages? 
# Consider that if John directly manages 2 people 
# and these two people manage 5 people each, 
# then we conclude that John manages 12 people.
# i.e. John manages 2 + 5 + 5 = 12 people

# Here we will use the opposite approach: 
# we start from the bottom (IC) and then count all the way up to the CEO.

#At first, we set 0. This will be true for ICs, then we keep updating this vector for all others
company['num_reports'] = 0
  
#same as before, but now we start from the bottom
company_levels = ["IC", "MM", "D", "VP", "E"]
  
i=0
while i<len(company_levels):
  #this is the count of direct reports + the prior count so we take into account direct reports of direct reports, etc.
  level_count=company.loc[company.level == company_levels[i]].groupby('boss_id')['num_reports'].agg(lambda x: x.count() + x.sum())

#  Let’s break this down:

# company.loc[company.level == company_levels[i]]:
# Get all employees at the current level (e.g., all ICs).

# .groupby('boss_id'):
# Group them by their manager's ID (because we want to update the boss's report count).

# .agg(lambda x: x.count() + x.sum()):
# For each boss, add:
# x.count() = how many people report to them directly
# x.sum() = how many people those reports already manage

#join to the main table to get the new report count for the bosses from the step above
  company=pd.merge(left=company, right=level_count, how='left', left_on="employee_id", right_on="boss_id", suffixes=('', '_updated'))
  
# If there's a new updated number (num_reports_updated), use it.
# Otherwise, keep the old one (which might be 0).
  company['num_reports'] = company.num_reports_updated.combine_first(company.num_reports)

#we can delete this now that we have updated the count
  del company['num_reports_updated']
  i+=1

# Show the top managers
print(company.sort_values(by=['num_reports'], ascending=False).head(6))

#        employee_id  boss_id         dept level  num_reports
# 2427        61554      NaN          CEO   CEO       9999.0
# 1310        11277  61554.0        sales     E       3598.0
# 2778        51535  61554.0  engineering     E       2695.0
# 1009       110464  61554.0    marketing     E       2009.0
# 9640        93708  61554.0           HR     E       1693.0
# 8995        34051  11277.0        sales    VP        607.0

# Q: Predict Employee Salaries: Build a model to predict the salary of each employee

#Let's join the two datasets
data=pd.merge(left=company, right=employee)

# And get rid of the columns we don't care about, 
# remove the CEO row (just one person — not useful for training).
# drop IDs — they're just labels, not meaningful for prediction
data=data.query("dept!=\"CEO\"").drop(["employee_id","boss_id"], axis=1)
  
# Level and degree are actually conceptually ordered. So in this case it can make sense 
# to replace them with numbers that represent the rank

# Encode Ordered Categories

# Degree levels: higher number = more education
codes = {"High_School":1, "Bachelor":2, "Master":3, "PhD":4}
data['degree_level'] = data['degree_level'].map(codes)

# Company levels: higher number = higher in the company  
codes = {"IC":1, "MM":2, "D":3,"VP":4, "E":5}
data['level'] = data['level'].map(codes)
  
# Let's make sex numerical manually
# Turn "M" into 1 and "F" into 0 and deletes the original sex column.
data['is_male'] = np.where(data['sex']=='M', 1,  0)
del data['sex']

# Many variables in this dataset are likely to be correlated. For example, 
# an employee's level in the company is likely correlated with the number of reports or years of experience. 
# While we could preprocess and clean the data before building a model, a simpler and 
# effective approach is to use a Random Forest. 
# Random Forest models are known to perform well out of the box in situations with correlated variables.

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
np.random.seed(1234)

# One-Hot Encode Categorical Variable (dept)
  
# dummy variables for the categorical one, i.e. dept. We keep all levels here. 
# It is not going to affect Random Forest badly, after all it does well with correlated variables 
# and it is just one variable with few levels. And this is going to make insights much cleaner

data_dummy = pd.get_dummies(data)

# This turns "dept" into multiple columns, one for each department, like:
# dept_HR
# dept_sales
# dept_engineering
  
#split into train and test to avoid overfitting (memorizing instead of generalizing).
train, test = train_test_split(data_dummy, test_size = 0.34)  
  
#Wewill use standard params here. They are likely close to optimal and changing them won't make much of a different in terms of insights, which is our goal here
rf = RandomForestRegressor(oob_score=True, n_estimators = 100)
rf.fit(train.drop('salary', axis=1), train['salary'])  

# Explanation:
# We create a Random Forest, which is like 100 smart "mini" models (trees) that vote on predictions.
# oob_score=True means it will self-evaluate using data it didn’t train on — a cool trick.
# We train the model using all features except salary (because that’s what we’re trying to predict).

RandomForestRegressor(
    bootstrap=True, # Each tree trains on a random sample (with replacement)
    criterion='mse', # Use Mean Squared Error to split nodes
    max_depth=None, # Trees can grow as deep as needed	Can cause overfitting if your data is noisy
    max_features=1.0, # Use all features
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=100, # Build 100 trees	More trees = better predictions (up to a point)
    n_jobs=None,
    oob_score=True, # out-of-bag scoring - Gives a free test accuracy during training
    random_state=None,
    verbose=0,
    warm_start=False
)

print("MSE on OOB is", round(metrics.mean_squared_error(train['salary'], rf.oob_prediction_)), 
      "and on test set is", round(metrics.mean_squared_error(test['salary'], rf.predict(test.drop('salary', axis=1)))))

# MSE on OOB is 5653838615 and on test set is 5785077483

print("% explained variance on OOB is", round(metrics.explained_variance_score(train['salary'], rf.oob_prediction_), 2), "and on test set is", round(metrics.explained_variance_score(test['salary'], rf.predict(test.drop('salary', axis=1))), 2))

# % explained variance on OOB is 0.28 and on test set is 0.28

