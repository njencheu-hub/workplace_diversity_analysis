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

# Understanding the exact meaning of the percentage of variance explained can be challenging, 
# especially from a product perspective. Additionally, this metric is influenced not only by 
# the quality of our model but also by the inherent variance in the starting dataset, 
# making it less informative. The good news is that our model is not overfitting.

# To make this more concrete, we can use a more practical metric: 
# the proportion of salaries for which the prediction is within 25% of the actual salary. 
# For example, if an employee's salary is 100K, we consider the model's prediction accurate 
# if it falls within 25K of the actual salary. 
# This metric can serve as a form of accuracy for continuous labels.

accuracy_25pct =  ((rf.predict(test.drop('salary', axis=1))/test['salary']-1).abs()<.25).mean()
print("We are within 25% of the actual salary in ",  accuracy_25pct.round(2)*100, "% of the cases", sep="")

# We are within 25% of the actual salary in 49.0% of the cases

# Nothing shockingly good and we would probably need more variables to predict more accurately. 
# However, the variable salary had a lot of variability to begin with.

#deciles
print(np.percentile(train['salary'], np.arange(0, 100, 10)))

# [ 60000.  77000.  97000. 125400. 154000. 182000. 209000. 237000. 272000. 314000.]

# Our model is definitely learning something, so insights will be fairly reliable, 
# and for sure directionally true.

# #Let's check variable importance
# feat_importances = pd.Series(rf.feature_importances_, index=train.drop('salary', axis=1).columns)
# feat_importances.sort_values().plot(kind='barh')
# plt.savefig('variable_importance.png')
# # plt.show()

# It looks like the model is essentially just using dept and, to a lesser extent, num_reports, 
# degree_level, and yrs_experience. All other variables are fairly irrelevant.

#Let's check partial dependence plots of the top 2 variables: dept and yrs_experience, as well as sex

from sklearn.inspection import PartialDependenceDisplay

# dept

# # Define the one-hot encoded department features
# dept_features = ['dept_HR', 'dept_engineering', 'dept_marketing', 'dept_sales']

# # Plot PDPs for all department features
# fig, ax = plt.subplots(figsize=(10, 6))
# PartialDependenceDisplay.from_estimator(
#     rf,
#     train.drop('salary', axis=1),
#     dept_features,
#     grid_resolution=2,  # Binary features: 0 and 1
#     ax=ax
# )

# plt.suptitle('Partial Dependence for Department Features', fontsize=14)
# plt.tight_layout()
# plt.savefig('partial_plot_for_dept.png')
# # plt.show()

# yrs_experience

# Compute and plot PDP for 'yrs_experience'
# fig, ax = plt.subplots(figsize=(8, 6))
# PartialDependenceDisplay.from_estimator(
#     rf,
#     train.drop('salary', axis=1),
#     ['yrs_experience'],
#     grid_resolution=50,
#     ax=ax
# )

# plt.title('Partial Dependence Plot for Years of Experience')
# plt.tight_layout()
# plt.savefig('partial_plot_for_yrs_of_experience.png')
# # plt.show()


# # Create PDP plot for 'is_male'
# fig, ax = plt.subplots(figsize=(8, 6))
# PartialDependenceDisplay.from_estimator(
#     rf,
#     train.drop('salary', axis=1),
#     ['is_male'],
#     grid_resolution=2,  # only 0 and 1 for binary features
#     ax=ax
# )

# plt.title('Partial Dependence Plot for Sex (is_male)')
# plt.tight_layout()
# plt.savefig('partial_dependence_plot_for_sex')
# # plt.show()

# The main driver of salary is undoubtedly the department. 
# There is a significant difference between HR and Engineering salaries. 
# Years of experience also plays a role, but its impact is less pronounced 
# (evident from the smaller y-range compared to the department). 
# mportantly, experience seems to matter more significantly after reaching a certain number of years. 
# This implies that salary increases substantially once an employee becomes quite senior, 
# while in the initial years, experience alone doesn't have a substantial impact.
#  Gender does not appear to be a significant factor in determining salary.

# Q: Describe the main factors impacting employee salaries. 
# Do you think the company has been treating all its employees fairly? 
# What are the next steps you would suggest to the Head of HR?

# Let’s focus on the variable sex here. 
# We already saw from the partial plot that it doesn’t matter to predict salary. 
# However, the average salary between males and females is pretty different.

#avg salary males vs females

print(data.groupby('is_male')['salary'].mean())

# is_male
# 0    171314.518394
# 1    198876.514445
# Name: salary, dtype: float64

# This is largely a function of the fact that males are more likely to be working in engineering 
# and less in HR. Indeed, if we look at sex and dept together, we now get very similar salaries by sex 
# (as a side note, we are also proving here that RF works extremely well with variables sharing information).

#avg salary males vs females by dept
print(data.groupby(['dept','is_male'])['salary'].agg({'mean', 'count'}))

#                      count           mean
# dept        is_male
# HR          0         1058   84399.810964
#             1          636   84827.044025
# engineering 0          671  246785.394933
#             1         2025  242444.444444
# marketing   0          651  192502.304147
#             1         1359  195639.440765
# sales       0         1181  194618.120237
#             1         2418  194207.196030

# One possible recommendation is to strive for a more balanced hiring across departments, 
# aiming for fewer females in HR and more in Engineering. The next step could be to analyze the candidate 
# pipeline by department and assess if the proportion of hires matches the proportion of applicants by gender.

# The observed relationship between years of experience and salary raises some concerns. 
# The company appears to reward experience predominantly at the highest levels, 
# with less emphasis on mid-level professionals. 
# This approach might negatively impact retention among mid-level employees. 
# Adjusting the salary growth to be more linear with years of experience could help 
# in retaining mid-level professionals more effectively.
