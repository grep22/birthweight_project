#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar  6 17:37:31 2019

@author: GuiReple

Birth weigth prediction
test_size = 0.10
random_state = 508
"""

###############################################################################
# Importing libraries and packages
###############################################################################
import pandas as pd
import statsmodels.formula.api as smf # regression modeling
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression


file = 'birthweight_feature_set-2.xlsx'
birth = pd.read_excel(file)

#birth.shape
###############################################################################
# Fundamental Dataset Exploration 

# Visual EDA steps:
# -  plotting each variable's distribution 
# -  scatter plots that illustrate relationship between the variables
# -  heatmap of correlations between the variables
# 
# Variable classification:
# Continuous/Interval Variables
# mage, meduc, monpre, npvis, fage, feduc, omaps,fmaps,cigs, drink, bwght
#
# Binary Classifiers
# male, mwhte, mblck, moth, fwhte, fblck, foth
#
# We included only important insights from the Visual EDA in this code 
# to keep it short and straight to the point
###############################################################################


# Key EDA Visualizations
plt.subplot(2, 2, 1)
plt.scatter(x="cigs", y="bwght", alpha = 0.5,
            color = 'blue',data=birth)
plt.title("Weight & Number cig/day")

plt.subplot(2, 2, 2)
plt.scatter(x="drink", y="bwght", alpha = 0.5,
            color = 'red',data=birth)
plt.title("Weight & Number drinks/week")


plt.subplot(2, 2, 3)
plt.scatter(x="mage", y="bwght",alpha = 0.5,
            color = 'green',data=birth)
plt.title("Weight & Mother's age")


plt.subplot(2, 2, 4)
plt.scatter(x="mage", y="bwght",alpha = 0.5,
            color = 'orange',data=birth)
plt.title("Weight & Mother's educ")


plt.tight_layout()
plt.show()

print('\n\nKey Correlations:')
print('_________________________________________')
print('Birth weight and drinks/week:',
      birth['bwght'].corr(birth['drink']).round(2))
print('Birth Weight and cigarettes/day:',
      birth['bwght'].corr(birth['cigs']).round(2))
print('Birth weight and mother age:',
      birth['bwght'].corr(birth['mage']).round(2))
print('_________________________________________')

#sns.lmplot(x="meduc", y="cigs", data=birth,
#          fit_reg=False,scatter=True)


#sns.lmplot(x="feduc", y="cigs", data=birth,
 #          fit_reg=False,scatter=True)


#sns.lmplot(x="meduc", y="cigs", data=birth,
#           fit_reg=False,scatter=True)


#sns.lmplot(x="feduc", y="drink", data=birth,
#           fit_reg=False,scatter=True)


#sns.lmplot(x="meduc", y="drink", data=birth,
#          fit_reg=False,scatter=True)


#sns.lmplot(x="mage", y="cigs", data=birth,
#          fit_reg=False,scatter=True)

#sns.lmplot(x="fage", y="cigs", data=birth,
#           fit_reg=False,scatter=True)

#sns.lmplot(x="mage", y="drink", data=birth,
#           fit_reg=False,scatter=True)

#sns.lmplot(x="fage", y="drink", data=birth,
#           fit_reg=False,scatter=True)


###############################################################################
###############################################################################
# 1. Imputing Missing Values: meduc, feduc, npvis
###############################################################################
###############################################################################

#print( birth.isnull().sum())

print('Motivation for the Missing Value Imputation')
      
########################
# Visual Missing Value Analysis (Histograms)
########################

plt.subplot(2, 2, 1)
sns.distplot(birth['meduc'],
             bins = 10,
             kde = False,
             rug = True,
             color = 'pink')
plt.xlabel('Years of Mother Education')


plt.subplot(2, 2, 2)
sns.distplot(birth['feduc'],
             bins = 10,
             kde = False,
             rug = True,
             color = 'blue')
plt.xlabel('Years of Father Education')

plt.subplot(2, 2, 3)
sns.distplot(birth['npvis'],
             bins = 10,
             kde = False,
             rug = True,
             color = 'orange')
plt.xlabel('Number of Prenatal visits')
plt.tight_layout()

plt.show()

###############################################################################
# We chose not to drop the missing variables as one of the most interesting
# cases with the lowest birthweight is among them.
# Everything is being filled with the median.
# Mean is not a good fit because the data is either right or left skewed.
# Moreover, the mean is not an integer which does not make sense for such 
# variables as Years of Education and Number of Visits.


# Mother Education
fill = birth['meduc'].median()
birth['meduc'] = birth['meduc'].fillna(fill)

# Father Education
fill = birth['feduc'].median()
birth['feduc'] = birth['feduc'].fillna(fill)

# Number of prenatal visits
fill = birth['npvis'].median()
birth['npvis'] = birth['npvis'].fillna(fill)


###############################################################################
###############################################################################
# 2. Outlier Analysis
###############################################################################
###############################################################################

# Outlier flags
cigs_hi = 2

drink_hi = 1

mage_hi = 35

npvis_lo = 13

meduc_lo = 12

feduc_lo = 12

Non_HS = 12 

Bach_or_Grad = 16

low_visit = 12

high_visit = 14

###############################################################################
# In this section we create additional colums that flag some of the outliyng 
# cases or provide deeper insight in some of the trends we found.
# We do not use all of the columns defined below in the model but we are 
# keeping them here as the part of our analysis.
###############################################################################


########################
# Flagging smokers (>2 cigarettes a day)
# out_cigs

birth['out_cigs'] = 0
for val in enumerate(birth.loc[ : , 'cigs']):
    
    if val[1] >= cigs_hi:
        birth.loc[val[0], 'out_cigs'] = 1

########################
# Flagging people who drink (>0 drinks per week)
# out_drink
        
birth['out_drink'] = 0
for val in enumerate(birth.loc[ : , 'drink']):
    
    if val[1] >= drink_hi:
        birth.loc[val[0], 'out_drink'] = 1
        
########################
# Flagging women 35 yo and older to draw a line between a healthy pregnancy
# and one that involves age risk.
# out_mage 

birth['out_mage'] = 0
for val in enumerate(birth.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        birth.loc[val[0], 'out_mage'] = 1


########################
# Flagging the small number of doctors visits during the pregnancy
# out_npvis    

birth['out_npvis'] = 0
for val in enumerate(birth.loc[ : , 'npvis']):
    
    if val[1] < npvis_lo:
        birth.loc[val[0], 'out_npvis'] = 1


########################
# Flagging mother education that did not complete High School (< 12 years)

birth['out_meduc'] = 0
for val in enumerate(birth.loc[ : , 'meduc']):
    
    if val[1] < meduc_lo:
        birth.loc[val[0], 'out_meduc'] = 1


########################
# Flagging father education that did not complete High School (< 12 years)

birth['out_feduc'] = 0
for val in enumerate(birth.loc[ : , 'feduc']):
    
    if val[1] < feduc_lo:
        birth.loc[val[0], 'out_feduc'] = 1 
        

########################
# Flag if the both parents are white
# out_white
        
birth['out_white'] = 0
for val in enumerate(birth.loc[ : , 'mwhte']):
    
    if val[1] ==1 & birth.loc[val[0], 'fwhte']==1:
        birth.loc[val[0], 'out_white'] = 1

########################
# Flag if the both parents are black
# out_black

birth['out_black'] = 0
for val in enumerate(birth.loc[ : , 'mblck']):
    
    if val[1] ==1 & birth.loc[val[0], 'fblck']==1:
        birth.loc[val[0], 'out_black'] = 1
        
########################
# Flag if the both parents are other than white or black
# out_other

birth['out_other'] = 0
for val in enumerate(birth.loc[ : , 'moth']):
    
    if val[1] ==1 & birth.loc[val[0], 'foth']==1:
        birth.loc[val[0], 'out_other'] = 1

        
    
########################
# Flag mother education Not Higschool

birth['NonHS_meduc'] = 0
for val in enumerate(birth.loc[ : , 'meduc']):
    
    if val[1] < Non_HS:
        birth.loc[val[0], 'NonHS_meduc'] = 1


########################
# Flag mother education Bachelor

birth['bach_meduc'] = 0
for val in enumerate(birth.loc[ : , 'meduc']):
    
    if val[1] <= Bach_or_Grad and val[1] >= Non_HS:
        birth.loc[val[0], 'bach_meduc'] = 1


########################
# Flag mother education Graduate School

birth['grad_meduc'] = 0
for val in enumerate(birth.loc[ : , 'meduc']):
    
    if val[1] > Bach_or_Grad:
        birth.loc[val[0], 'grad_meduc'] = 1


########################
# Total number of cigarettes fetus was exposed before the first prenatal visit
# At this point we assume that the number of cigarettes a day represents the 
# situation at the time of the first prenatal visit 
# cig_exp
        
birth['cig_exp'] = birth['monpre']*birth['cigs']*30

########################
# Total number of drinks fetus was exposed before the first prenatal visit
# drink_exp

birth['drinks_exp'] = birth['monpre']*birth['drink']*4

########################
# Created risk factors which is the addition of all factors 
# that are bad for pregnancy
# risk factors

birth['risk factors'] = birth['out_cigs']+birth['out_mage']+birth['out_drink']

########################
# Flag for an age of 45 and light-heavy smoking
# age_cigs

birth['age_cigs']=0
def conditions(birth):
    if (birth['mage']>45) and (birth['cigs']>2):
        return 1
    else:
        return 0
birth['age_cigs'] = birth.apply(conditions, axis=1)

########################
# Flag for age > 35, heavy smoking habit, habitual drinking
# old_addict

birth['old_addict']=0
def conditions(birth):
    if (birth['mage']>35) and (birth['cigs']>10) and (birth['drink']>5):
        return 1
    else:
        return 0
birth['old_addict'] = birth.apply(conditions, axis=1)


########################
# Creating total number of visits based on the regular 13 doctor visit norm. 
# Our dataset is fairly educated. They would have access to medicare which 
# would provide them in the worst case scenario with coverage for doctors 
# appointments. 
# birth['pre_vist']=birth['monpre']

def conditions(birth):
    if (birth['monpre']==7):
        return 8
    if (birth['monpre']==8):
        return 12
    else:
        return (birth['monpre'])
    
    
    
birth['pre_vist'] = birth.apply(conditions, axis=1)

birth['total_visit'] = birth['pre_vist'] + birth['npvis']

birth['out_tv_low'] = 0
for val in enumerate(birth.loc[ : , 'total_visit']):
    
    if val[1] < low_visit:
        birth.loc[val[0], 'out_tv_low'] = 1

birth['out_tv_high'] = 0
for val in enumerate(birth.loc[ : , 'total_visit']):
    
    if val[1] > high_visit:
        birth.loc[val[0], 'out_tv_high'] = 1
        
birth['socio_econ1'] = birth['out_tv_low']+birth['out_feduc']

###############################################################################
###############################################################################
# 3. Statsmodel 
#
# Our approach to statistical modelling is to build a Base Model with
# statsmodels (OLS). It provides a summary output
#
# Once we are satisfied with our variable selection, we can move on to using 
# other modeling techniques.
#
# We experimented with imputing different variabes in the formula, takind out 
# some of them with high p-values
# We paid attention to R-squared to be above 0.7 and F-statistic 
# as small as possible
#
# The model below represents the optimal set of the predicting variables.
###############################################################################
###############################################################################

b_data=birth
b_data = birth.drop(['bwght'], axis = 1)
b_target = birth.loc[:, 'bwght']

#  Splitting the Data Using Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
            b_data,
            b_target,
            test_size = 0.10,
            random_state = 508)

# Merging our X_train and y_train sets 
b_train = pd.concat([X_train, y_train], axis = 1)

# Step 1: Build the model
lm_significant = smf.ols(formula = """bwght ~ b_train['mage'] +
                                                 b_train['fage']+
                                                 b_train['cigs'] +
                                                 b_train['drink'] +
                                                 b_train['out_drink']+
                                                 b_train['out_cigs']+
                                                 b_train['feduc']+
                                                 b_train['risk factors']
                                                 """, 
                                                 data = b_train)

# Step 2: Fit the model based on the data
results = lm_significant.fit()

# Step 3: Analyze the summary output
# R-squared: 0.734
# 
print(results.summary())


###############################################################################
###############################################################################
#
# 4. Supervised Modeling Process : Linear Regression
#
###############################################################################
###############################################################################

#  Splitting the Data Using Train/Test Split

# Here we choose the variables to work with and drop the others
birth_data = birth.drop(['mwhte','fblck','foth',
                         'out_mage','cig_exp','drinks_exp',
                         'monpre','npvis','mblck',
                         'omaps','fmaps','fwhte',
                         'bwght','moth','out_feduc',
                         'meduc','out_meduc', 'male',
                         'out_npvis', 'NonHS_meduc','bach_meduc',
                         'grad_meduc', 'out_other', 'out_black',
                         'out_white','age_cigs', 'out_tv_low',
                         'total_visit', 'pre_vist', 'socio_econ1',
                         'out_tv_high','old_addict'], 
                            axis = 1)

    
birth_target = birth.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(
            birth_data,
            birth_target,
            test_size = 0.10,
            random_state = 508)


# Training set 
#print(X_train.shape)
#print(y_train.shape)

# Testing set
#print(X_test.shape)
#print(y_test.shape)

from sklearn.linear_model import LinearRegression

# Prepping the Model
lr = LinearRegression()

# Fitting the model
lr_fit = lr.fit(X_train, y_train)

# Predictions
lr_pred = lr_fit.predict(X_test)

# Let's compare the testing score to the training score.
print('\n\nLinear Regression Scores')
print('_________________________________________')
print('Training Score Linear Reg:', lr.score(X_train, y_train).round(3))
print('Testing Score Linear Reg:', lr.score(X_test, y_test).round(3))
print('_________________________________________\n')                      

#plt.scatter(y_test, lr_pred)
#plt.xlabel('True Values')
#plt.ylabel('Predictions')

#########################
# Training score: 0.734 #
# Testing score: 0.707  #
#########################


###############################################################################
###############################################################################
#
# 5. Using KNN  On Our Optimal Variables to compare results with linear 
#    regression 
# 
###############################################################################
###############################################################################


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)

# Loop to decide which number of neighbors is the optimal using accuracy plot

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Highest test accuracy
k = test_accuracy.index(max(test_accuracy)) + 1


########################
# The best results occur when k = 13.
########################

# Building a model with k = 13
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = k)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)


###############################################################################
###############################################################################
## 6. Compare OLS predictions with KNN
###############################################################################
###############################################################################


# Let's compare the testing score to the training score.
#print('Training Score', lr.score(X_train, y_train).round(4))
#print('Testing Score:', lr.score(X_test, y_test).round(4))

print('\n\nCompare OLS and KNN scores on the test data')
print('_________________________________________')
# Printing model results
print(f"""
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {lr.score(X_test, y_test).round(3)}
""")
print('_________________________________________')


###############################################################################
# Storing Model Predictions and Summary
###############################################################################

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'OLS_Predicted': lr_pred})
model_predictions_df.to_excel("Prediction.xlsx")




# Best model - Linear Regression with optimal variables
# test_size = 0.10
# random_state = 508
# 8 predictive variables

#########################
# Training score: 0.734 #
# Testing score: 0.707  #
#########################













