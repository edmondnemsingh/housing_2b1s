# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:48:19 2018

@author: Seizure
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import datetime
from sklearn.ensemble import RandomForestRegressor


################################   STEP ONE: ONE HOT ENCODING COMBINED DATA SET   ################################

train_file_path = 'C:/Users/Seizure/Documents/MachineLearning/Housing/train.csv'
test_file_path = 'C:/Users/Seizure/Documents/MachineLearning/Housing/test.csv'
combined_file_path = 'C:/Users/Seizure/Documents/MachineLearning/Housing/traintest.csv'
housing_data = pd.read_csv(combined_file_path)


#fill NA in the LotFrontage column with the average value
housing_data["LotFrontage"].fillna(housing_data.LotFrontage.mean(), inplace = True)
now = datetime.datetime.now()

coly = housing_data.columns.to_series().groupby(housing_data.dtypes).groups 
n_vars = list(housing_data.dtypes[housing_data.dtypes!="object"].index)
to_remove = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt','YearRemodAdd', 
             'GarageYrBlt', 'MoSold', 'YrSold']
for item in to_remove:
    n_vars.remove(item)
c_vars = list(housing_data.dtypes[housing_data.dtypes == "object"].index)
c_vars.extend(('OverallQual', 'OverallCond', #'YearBuilt','YearRemodAdd', 'GarageYrBlt'
                             'MoSold', 'YrSold'))

for variable in c_vars:
    dummies = pd.get_dummies(housing_data[variable], prefix=variable)
    housing_data = pd.concat([housing_data, dummies], axis = 1)
    housing_data.drop([variable], axis=1, inplace = True)

features = list(housing_data.columns.values)
icelist = ['Id', 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'SalePrice']
for icecubes in icelist:
    features.remove(icecubes)
    
combined_X = housing_data[features]

#housing_data.to_csv('OneHotEncodedHousingData.csv', index = False)
#Split up OneHotEncoded File Manually then continue yourself.    


################################   STEP THREE: MODELLING  ################################
training_data = pd.read_csv(train_file_path)
y = training_data.SalePrice
X = training_data[features]
training_data['HouseAge'] = int(now.year) - training_data['YearRemodAdd']
training_data['GarageAge'] = int(now.year) - training_data['GarageYrBlt']

#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state = 1, n_estimators = 100))

my_pipeline.fit(X, y)
training_predictions = my_pipeline.predict(X)

traintest_predictions = my_pipeline.predict(combined_X)


################################   STEP FOUR: CROSS VALIDATION   ################################
scores = cross_val_score(my_pipeline, X, y, cv=5,  scoring = "neg_mean_absolute_error")
print(scores)
answer = (sum(scores)/len(scores))
print("Using RandomForestRegressor the " + str(answer) + ' Average Negative Mean Absolute Error')
print(1+((answer)/180921.195890))

################################   STEP FIVE: PREDICT AGAINST TEST   ################################
test_data = pd.read_csv(test_file_path)
test_data["LotFrontage"].fillna(housing_data.LotFrontage.mean(), inplace = True)

test_data['HouseAge'] = int(now.year) - test_data['YearRemodAdd']
test_data['GarageAge'] = int(now.year) - test_data['GarageYrBlt']

test_X = test_data[features]

test_predictions = my_pipeline.predict(test_X)
print(test_predictions)

df = pd.concat([test_data.Id, pd.DataFrame(test_predictions, columns = ['SalePrice'])], axis = 1)
df.to_csv('HousingPredictions_RandomForestRegressor.csv', index = False)

